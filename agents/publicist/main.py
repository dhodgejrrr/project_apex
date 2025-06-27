"""Publicist Cloud Run service for Project Apex.

Generates social-media copy (tweet variations) from insights JSON using Vertex AI
Gemini. Expects Pub/Sub push payload with `analysis_path` and `insights_path`.
"""
from __future__ import annotations


import json
import logging
import os
import pathlib
import tempfile
import base64

# AI helpers
from agents.common import ai_helpers
from agents.common.request_utils import parse_request_payload, validate_required_fields
from typing import Any, Dict, List

from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud import aiplatform

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
MODEL_NAME = os.getenv("VERTEX_MODEL", "gemini-1.0-pro")
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

# --- NEW: Load the prompt templates from files on startup ---
PROMPT_TEMPLATE_PATH = pathlib.Path(__file__).parent / "prompt_template.md"
PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text()

POSTS_WITH_VISUALS_TEMPLATE_PATH = pathlib.Path(__file__).parent / "posts_with_visuals_template.md"
POSTS_WITH_VISUALS_TEMPLATE = POSTS_WITH_VISUALS_TEMPLATE_PATH.read_text()

POSTS_CRITIQUE_TEMPLATE_PATH = pathlib.Path(__file__).parent / "posts_critique_template.md"
POSTS_CRITIQUE_TEMPLATE = POSTS_CRITIQUE_TEMPLATE_PATH.read_text()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("publicist")

# ---------------------------------------------------------------------------
# Clients (lazy)
# ---------------------------------------------------------------------------
_storage_client: storage.Client | None = None


def _storage() -> storage.Client:
    global _storage_client  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

# ---------------------------------------------------------------------------
# Download & upload helpers
# ---------------------------------------------------------------------------

def _gcs_download(gcs_uri: str, dest: pathlib.Path) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    _storage().bucket(bucket_name).blob(blob_name).download_to_filename(dest)


def _gcs_upload(local_path: pathlib.Path, dest_blob: str) -> str:
    blob = _storage().bucket(ANALYZED_DATA_BUCKET).blob(dest_blob)
    blob.upload_from_filename(local_path)
    return f"gs://{ANALYZED_DATA_BUCKET}/{dest_blob}"

# ---------------------------------------------------------------------------
# Insight selection

def _select_key_insights(insights: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    """Deduplicate by type and prioritise Historical & Strategy categories."""
    selected: List[Dict[str, Any]] = []
    seen_types: set[str] = set()
    def _priority(ins):
        cat = ins.get("category", "")
        if cat == "Historical Comparison":
            return 0
        if "Strategy" in cat:
            return 1
        return 2
    valid_insights = [ins for ins in insights if isinstance(ins, dict)]
    if not valid_insights:
        return []
    sorted_in = sorted(valid_insights, key=_priority)
    for ins in sorted_in:
        if ins.get("type") in seen_types:
            continue
        selected.append(ins)
        seen_types.add(ins.get("type"))
        if len(selected) >= limit:
            break
    return selected

# Gemini helper
# ---------------------------------------------------------------------------

def _init_gemini() -> None:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)


def _gen_tweets(insights: Any, analysis: Any, max_posts: int = 7) -> List[str]:
    """Call Gemini to generate up to `max_posts` social media posts.

    The `insights` parameter can be either:
    1. A list of dictionaries (legacy behaviour) or
    2. A dictionary whose values contain lists of insight dicts (current Insight Hunter output).
    The function normalises the structure so downstream logic always works with a flat
    list of insight dictionaries.
    """
    # ---------------------------------------------------------------------
    # Normalise the insights structure so we always work with List[Dict].
    # ---------------------------------------------------------------------
    flat_insights: List[Dict[str, Any]] = []
    if isinstance(insights, list):
        # Already the expected shape
        flat_insights = [ins for ins in insights if isinstance(ins, dict)]
    elif isinstance(insights, dict):
        # Flatten all list-valued entries (e.g. manufacturer_pace_ranking, etc.)
        for val in insights.values():
            if isinstance(val, list):
                flat_insights.extend([ins for ins in val if isinstance(ins, dict)])
    else:
        LOGGER.warning("Unsupported insights type passed to _gen_tweets: %s", type(insights))

    key_ins = _select_key_insights(flat_insights)
    if not key_ins:
        return []

    if USE_AI_ENHANCED:
        # --- MODIFIED: Use the loaded template and format it with both JSON objects ---
        # WARNING: Passing the full analysis_enhanced.json can be very large and may
        # exceed model input token limits. For production, consider summarizing
        # this payload first or ensuring you use a model with a large context window.
        prompt = PROMPT_TEMPLATE.format(
            insights_json=json.dumps(insights, indent=2),
            analysis_enhanced_json=json.dumps(analysis, indent=2),
            max_posts=max_posts,
        )
        # prompt = (
        #     "You are a social media manager for a professional race team. "
        #     "Create up to " + str(max_posts) + " engaging social media posts based on the provided JSON data. "
        #     "Each post must be a standalone string, under 280 characters, and include relevant hashtags like #IMSA and appropriate emojis. "
        #     "Only reference a particular manufacturer, car, or team once. If you use them for a post, do not use them for another.  "
        #     "Your response MUST be a valid JSON array of strings, like [\"post1\", \"post2\"]. Do not return anything else.\n\n"
        #     "Insights JSON:\n" + json.dumps(insights, indent=2)
        # )
        try:
            # Slightly higher temp for creativity, more tokens for safety
            tweets = ai_helpers.generate_json_adaptive(prompt, temperature=0.8, max_output_tokens=8000)
            if isinstance(tweets, list) and all(isinstance(t, str) for t in tweets):
                return tweets[:max_posts]

            LOGGER.warning("AI response was not a list of strings: %s", tweets)
            return []  # Return empty list on malformed AI response
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("AI tweet generation failed: %s", exc)
            return []  # Return empty list on API error

    # Fallback template only if AI is not used
    fallback = [f"🏁 {ins.get('type')}: {ins.get('details')} #IMSA #ProjectApex" for ins in key_ins[:max_posts]]
    return fallback

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request():
    try:
        payload = parse_request_payload(request)
        
        # Support both old format (analysis_path + insights_path) and new format (briefing_path)
        if "briefing_path" in payload:
            # New autonomous workflow with Arbiter briefing
            briefing_uri = payload["briefing_path"]
            analysis_uri = payload.get("analysis_path")  # Optional for visual generation
            comprehensive_analysis_uri = payload.get("comprehensive_analysis_path")  # Optional comprehensive analysis
            use_autonomous = payload.get("use_autonomous", True)
            validate_required_fields(payload, ["briefing_path"])
        else:
            # Legacy workflow
            briefing_uri = None
            analysis_uri = payload["analysis_path"]
            insights_uri = payload["insights_path"] 
            comprehensive_analysis_uri = payload.get("comprehensive_analysis_path")  # Optional comprehensive analysis
            use_autonomous = False
            validate_required_fields(payload, ["analysis_path", "insights_path"])
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Bad request: %s", exc)
        return jsonify({"error": "bad_request"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        
        try:
            if briefing_uri and use_autonomous:
                # New autonomous workflow
                LOGGER.info("Using autonomous post generation with briefing data...")
                
                # Download briefing data
                local_briefing = tmp / "briefing.json"
                _gcs_download(briefing_uri, local_briefing)
                briefing_data = json.loads(local_briefing.read_text())
                
                # Download analysis data if available for visual generation
                analysis_data = {}
                if analysis_uri:
                    local_analysis = tmp / "analysis.json"
                    _gcs_download(analysis_uri, local_analysis)
                    analysis_data = json.loads(local_analysis.read_text())
                    
                    # Use comprehensive analysis for richer social content if available
                    if comprehensive_analysis_uri:
                        LOGGER.info("Using comprehensive analysis for enhanced social content")
                        local_comprehensive = tmp / "comprehensive_analysis.json"
                        try:
                            _gcs_download(comprehensive_analysis_uri, local_comprehensive)
                            comprehensive_data = json.loads(local_comprehensive.read_text())
                            # Merge comprehensive data for richer social media content
                            analysis_data.update(comprehensive_data)
                            LOGGER.info("Successfully integrated comprehensive analysis into social content")
                        except Exception as e:
                            LOGGER.warning("Could not load comprehensive analysis, using standard analysis: %s", e)
                
                # Generate posts with autonomous workflow
                result = generate_posts_with_correction(
                    briefing_data, 
                    analysis_data, 
                    analysis_uri or briefing_uri  # Use briefing_uri as fallback
                )
                
                posts = result.get("posts", [])
                
            else:
                # Legacy workflow
                LOGGER.info("Using legacy post generation workflow...")
                
                local_analysis = tmp / pathlib.Path(analysis_uri).name
                local_insights = tmp / pathlib.Path(insights_uri).name
                
                # Download input files
                _gcs_download(analysis_uri, local_analysis)
                _gcs_download(insights_uri, local_insights)

                # Load JSON content
                analysis_data = json.loads(local_analysis.read_text())
                insights_data = json.loads(local_insights.read_text())

                # Use comprehensive analysis for richer social content if available
                if comprehensive_analysis_uri:
                    LOGGER.info("Using comprehensive analysis for enhanced legacy social content")
                    local_comprehensive = tmp / "comprehensive_analysis.json"
                    try:
                        _gcs_download(comprehensive_analysis_uri, local_comprehensive)
                        comprehensive_data = json.loads(local_comprehensive.read_text())
                        # Merge comprehensive data for richer social media content
                        analysis_data.update(comprehensive_data)
                        LOGGER.info("Successfully integrated comprehensive analysis into legacy social content")
                    except Exception as e:
                        LOGGER.warning("Could not load comprehensive analysis, using standard analysis: %s", e)

                # Generate tweets using legacy method
                tweets = _gen_tweets(insights_data, analysis_data)
                posts = [{"text": tweet, "has_visual": False} for tweet in tweets]
                result = {"generation_method": "legacy"}

            # Save posts to GCS
            output_json = tmp / "social_media_posts.json"
            json.dump({
                "posts": posts,
                "metadata": {
                    "generation_method": result.get("generation_method", "legacy"),
                    "attempts": result.get("attempts", 1),
                    "final_critique": result.get("final_critique", {}),
                    "warning": result.get("warning")
                }
            }, output_json.open("w", encoding="utf-8"))

            # Extract run_id from URI
            run_id = (briefing_uri or analysis_uri).split('/')[3]
            out_uri = _gcs_upload(output_json, f"{run_id}/social/social_media_posts.json")
            LOGGER.info("Uploaded social media posts to %s", out_uri)
            
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return jsonify({"error": "internal_error"}), 500

    return jsonify({
        "social_posts_path": out_uri,
        "generation_method": result.get("generation_method", "legacy"),
        "posts_count": len(posts),
        "attempts": result.get("attempts", 1)
    }), 200

# ---------------------------------------------------------------------------
# Autonomous Social Media Generation with Self-Correction (Phase 2)
# ---------------------------------------------------------------------------

def generate_posts_with_visuals(briefing_data: Dict[str, Any], analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate social media posts with autonomous visual decision making.
    LLM decides which posts need visuals and what type of visual to generate.
    """
    # Handle previous feedback if available
    previous_feedback_text = ""
    if "previous_feedback" in briefing_data:
        previous_feedback_text = f"Previous attempt feedback:\n{json.dumps(briefing_data['previous_feedback'], indent=2)}\n\nUse this feedback to improve the current generation."
    else:
        previous_feedback_text = "This is the first generation attempt."
    
    try:
        # Use the direct template approach like other working agents
        prompt = POSTS_WITH_VISUALS_TEMPLATE.format(
            briefing_data_json=json.dumps(briefing_data, indent=2),
            previous_feedback=previous_feedback_text,
            analysis_enhanced_json=json.dumps(analysis_data, indent=2),
        )
        
        response = ai_helpers.generate_json_adaptive(
            prompt,
            temperature=0.8,
            max_output_tokens=10000
        )
        LOGGER.info(f"Generated {len(response.get('posts', []))} posts with visual decisions")
        return response
    except Exception as e:
        LOGGER.error(f"Failed to generate posts with visuals: {e}")
        return {"posts": []}


def invoke_visualizer_tool(visual_type: str, analysis_path: str, params: Dict[str, Any] = None) -> str:
    """
    Call the Visualizer toolbox to generate a specific chart.
    Returns the GCS path to the generated image.
    """
    if params is None:
        params = {}
    
    try:
        # Build parameters for visualizer tool
        tool_params = {"analysis_path": analysis_path}
        tool_params.update(params)
        
        LOGGER.info(f"Calling visualizer tool: {visual_type} with params: {tool_params}")
        
        # Call the visualizer tool
        from agents.common.tool_caller import tool_caller
        response = tool_caller.call_visualizer_tool(visual_type, **tool_params)
        
        image_path = response.get("image_gcs_path")
        if image_path:
            LOGGER.info(f"Successfully generated {visual_type} chart: {image_path}")
            return image_path
        else:
            LOGGER.warning(f"Visualizer tool returned no image path: {response}")
            return None
            
    except Exception as e:
        LOGGER.error(f"Failed to invoke visualizer tool {visual_type}: {e}")
        return None


def critique_posts(posts: List[Dict[str, Any]], briefing_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Self-critique the generated posts to assess quality and suggest improvements.
    """
    posts_for_review = []
    for post in posts:
        posts_for_review.append({
            "text": post.get("text"),
            "has_visual": post.get("image_url") is not None,
            "visual_type": post.get("visual_type"),
            "priority": post.get("priority")
        })
    
    try:
        # Use the direct template approach like other working agents
        prompt = POSTS_CRITIQUE_TEMPLATE.format(
            posts_for_review_json=json.dumps(posts_for_review, indent=2),
            briefing_data_json=json.dumps(briefing_data, indent=2)
        )
        
        critique = ai_helpers.generate_json_adaptive(
            prompt,
            temperature=0.8,
            max_output_tokens=10000
        )
        LOGGER.info(f"Critique completed: {critique.get('approved')} (score: {critique.get('overall_score')})")
        return critique
    except Exception as e:
        LOGGER.error(f"Failed to critique posts: {e}")
        return {"approved": True, "feedback": f"Critique failed: {e}"}


def generate_posts_with_correction(briefing_data: Dict[str, Any], analysis_data: Dict[str, Any], 
                                 analysis_path: str, max_attempts: int = 3) -> Dict[str, Any]:
    """
    Main autonomous generation workflow with self-correction loop.
    
    Workflow:
    1. Generate posts with visual decisions
    2. Call visualizer tools for needed visuals
    3. Self-critique the complete package
    4. If not approved, incorporate feedback and retry
    """
    LOGGER.info("Starting autonomous post generation with self-correction...")
    
    attempt = 1
    feedback_history = []
    
    while attempt <= max_attempts:
        LOGGER.info(f"Generation attempt {attempt}/{max_attempts}")
        
        # Add previous feedback to briefing data for learning
        working_briefing = briefing_data.copy()
        if feedback_history:
            working_briefing["previous_feedback"] = feedback_history
        
        # Step 1: Generate posts with visual decisions
        posts_plan = generate_posts_with_visuals(working_briefing, analysis_data)
        
        if not posts_plan.get("posts"):
            LOGGER.warning(f"No posts generated on attempt {attempt}")
            attempt += 1
            continue
        
        # Step 2: Generate visuals for posts that need them
        final_posts = []
        for post in posts_plan["posts"]:
            enhanced_post = post.copy()
            
            if post.get("needs_visual") and post.get("visual_type"):
                visual_params = post.get("visual_params", {})
                image_url = invoke_visualizer_tool(
                    post["visual_type"], 
                    analysis_path, 
                    visual_params
                )
                
                if image_url:
                    enhanced_post["image_url"] = image_url
                    enhanced_post["has_visual"] = True
                else:
                    enhanced_post["has_visual"] = False
                    LOGGER.warning(f"Failed to generate visual for post: {post.get('text')[:50]}...")
            else:
                enhanced_post["has_visual"] = False
            
            final_posts.append(enhanced_post)
        
        # Step 3: Self-critique
        critique = critique_posts(final_posts, briefing_data)
        
        if critique.get("approved", False):
            LOGGER.info(f"Posts approved on attempt {attempt}")
            return {
                "posts": final_posts,
                "attempts": attempt,
                "final_critique": critique,
                "generation_method": "autonomous_with_correction"
            }
        
        # Step 4: Learn from feedback for next attempt
        feedback_history.append({
            "attempt": attempt,
            "feedback": critique.get("feedback"),
            "issues": critique.get("specific_issues", []),
            "suggestions": critique.get("suggestions", [])
        })
        
        LOGGER.info(f"Posts not approved on attempt {attempt}. Feedback: {critique.get('feedback')}")
        attempt += 1
    
    # Max attempts reached - return best effort
    LOGGER.warning(f"Max attempts ({max_attempts}) reached. Returning final posts with warning.")
    return {
        "posts": final_posts if 'final_posts' in locals() else [],
        "attempts": max_attempts,
        "final_critique": critique if 'critique' in locals() else {},
        "warning": "Max attempts reached without approval",
        "generation_method": "autonomous_with_correction_incomplete"
    }

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "publicist"}), 200

if __name__ == "__main__":
    # Register with Tool Registry at startup
    try:
        from agents.common.tool_caller import register_agent_with_registry
        port = int(os.environ.get("PORT", 8080))
        base_url = os.getenv("PUBLICIST_URL", f"http://localhost:{port}")
        register_agent_with_registry("publicist", base_url)
    except Exception as e:
        LOGGER.warning(f"Failed to register with Tool Registry: {e}")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
