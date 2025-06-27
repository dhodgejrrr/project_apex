"""
InsightHunter Cloud Run service for Project Apex.

Listens for Pub/Sub push messages containing the Cloud Storage path to an
_enhanced.json race analysis file, derives tactical insights, stores them as a
new _insights.json file, and publishes a notification to the
`visualization-requests` topic.
"""
from __future__ import annotations


import json
import os
import logging
import os
import pathlib
import tempfile
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional
import base64

# AI helper utils - Assuming this is a local utility
from agents.common import ai_helpers
from agents.common.request_utils import parse_request_payload, validate_required_fields

from flask import Flask, request, jsonify
from google.cloud import pubsub_v1, storage

# ----------------------------------------------------------------------------
# Environment configuration & Logging
# ----------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data-project-apex-v1")
VIS_TOPIC_ID = os.getenv("VISUALIZATION_REQUESTS_TOPIC", "visualization-requests")
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("insight_hunter")

# ----------------------------------------------------------------------------
# Cloud Clients & Helpers
# ----------------------------------------------------------------------------
_storage_client: Optional[storage.Client] = None
_publisher: Optional[pubsub_v1.PublisherClient] = None


def _init_clients() -> tuple[storage.Client, pubsub_v1.PublisherClient]:
    global _storage_client, _publisher
    if _storage_client is None:
        _storage_client = storage.Client()
    if _publisher is None:
        if os.getenv("PUBSUB_EMULATOR_HOST"):
            from google.auth.credentials import AnonymousCredentials
            _publisher = pubsub_v1.PublisherClient(credentials=AnonymousCredentials())
        else:
            _publisher = pubsub_v1.PublisherClient()
    return _storage_client, _publisher

def _gcs_download(gcs_uri: str, dest_path: pathlib.Path) -> None:
    storage_client, _ = _init_clients()
    if not gcs_uri.startswith("gs://"): raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(dest_path)

def _gcs_upload(local_path: pathlib.Path, bucket: str, blob_name: str) -> str:
    storage_client, _ = _init_clients()
    blob = storage_client.bucket(bucket).blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket}/{blob_name}"

def _publish_visualization_request(analysis_uri: str, insights_uri: str) -> None:
    _, publisher = _init_clients()
    # When using the emulator, all topics are created under the 'local-dev' project.
    # In a real environment, the service will use the configured GOOGLE_CLOUD_PROJECT.
    project_id = "local-dev" if os.getenv("PUBSUB_EMULATOR_HOST") else PROJECT_ID
    if not project_id:
        raise ValueError("Project ID not found for Pub/Sub publishing.")
        
    topic_path = publisher.topic_path(project_id, VIS_TOPIC_ID)
    message = {"analysis_path": analysis_uri, "insights_path": insights_uri}
    future = publisher.publish(topic_path, json.dumps(message).encode("utf-8"))
    future.result() # Make call blocking to ensure message is sent for local test
    LOGGER.info("Published visualization request to %s: %s", topic_path, message)

# ----------------------------------------------------------------------------
# Domain-specific Helper Functions
# ----------------------------------------------------------------------------

def _time_str_to_seconds(time_str: str | None) -> float | None:
    if not isinstance(time_str, str): return None
    try:
        sign = 1
        clean_str = time_str
        if time_str.startswith("-"):
            sign = -1
            clean_str = time_str[1:]
        
        if ":" in clean_str:
            parts = clean_str.split(":")
            if len(parts) == 2: return sign * (float(parts[0]) * 60 + float(parts[1]))
            elif len(parts) == 3: return sign * (float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2]))
        return sign * float(clean_str)
    except (ValueError, TypeError):
        return None

def _format_seconds(seconds: Optional[float], precision: int = 3, unit: str = "") -> Optional[str]:
    if seconds is None: return None
    return f"{seconds:+.{precision}f}{unit}"

# ----------------------------------------------------------------------------
# Insight Ranking Algorithms
# ----------------------------------------------------------------------------

def find_pit_stop_insights(strategy_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []

    for car in strategy_data:
        car_number = car.get("car_number")
        pit_details = car.get("pit_stop_details", [])
        avg_pit_lane_time_s: List[float] = []
        for stop in pit_details:
            secs = _time_str_to_seconds(stop.get("total_pit_lane_time"))
            if secs is not None:
                avg_pit_lane_time_s.append(secs)
        if not avg_pit_lane_time_s:
            continue
        avg_time = mean(avg_pit_lane_time_s)
        # Logic 1: Pit Delta Outlier
        for idx, stop in enumerate(pit_details, start=1):
            delta_s = _time_str_to_seconds(stop.get("total_pit_lane_time"))
            if delta_s is None:
                continue
            if delta_s > 1.5 * avg_time:
                insights.append({
                    "category": "Pit Stop Intelligence",
                    "type": "Pit Delta Outlier",
                    "car_number": car_number,
                    "details": f"Stop #{idx} was {delta_s - avg_time:.1f}s slower than team average."
                })
        # Logic 2: Driver Change Cost
        stationary_changes: List[float] = []
        stationary_no_change: List[float] = []
        for stop in pit_details:
            stat_time = _time_str_to_seconds(stop.get("stationary_time"))
            if stat_time is None:
                continue
            if stop.get("driver_change"):
                stationary_changes.append(stat_time)
            else:
                stationary_no_change.append(stat_time)
        if stationary_changes and stationary_no_change:
            diff = mean(stationary_changes) - mean(stationary_no_change)
            insights.append({
                "category": "Pit Stop Intelligence",
                "type": "Driver Change Cost",
                "car_number": car_number,
                "details": f"Driver changes cost an average of {diff:.1f}s more stationary time."
            })

    # Logic 3: Stationary Time Champion (needs enhanced_strategy_analysis section)
    # This will be added from outer function because data lives elsewhere.
    return insights
    
def _rank_by_metric(
    data_list: List[Dict], group_by_key: str, metric_path: List[str], higher_is_better: bool, value_is_time: bool = True
) -> List[Dict]:
    """Generic function to rank entities (cars or manufacturers) by a given metric."""
    grouped_metrics = defaultdict(list)
    for item in data_list:
        group_name = item.get(group_by_key)
        value = item
        try:
            for key in metric_path: value = value[key]
        except (KeyError, TypeError): continue
        
        if isinstance(value, str): value = _time_str_to_seconds(value)
        
        if group_name and value is not None:
            grouped_metrics[group_name].append(value)

    if not grouped_metrics: return []

    avg_metrics = {name: mean(vals) for name, vals in grouped_metrics.items() if vals}
    if not avg_metrics: return []
    
    field_average = mean(avg_metrics.values())
    
    ranked_list = [{
        "name": name, "value": avg_val, "delta_from_avg": avg_val - field_average
    } for name, avg_val in avg_metrics.items()]

    ranked_list.sort(key=lambda x: x["value"], reverse=higher_is_better)

    return [{
        "rank": i + 1,
        group_by_key: item["name"],
        "average_value": _format_seconds(item["value"]) if value_is_time else f"{item['value']:.4f}",
        "delta_from_field_avg": _format_seconds(item["delta_from_avg"]) if value_is_time else f"{item['delta_from_avg']:+.4f}"
    } for i, item in enumerate(ranked_list)]

def _rank_cars_by_untapped_potential(fastest_by_car: List[Dict]) -> List[Dict]:
    """Ranks cars by the largest gap between their actual fastest lap and theoretical optimal lap."""
    gaps = []
    for car in fastest_by_car:
        fastest_s = _time_str_to_seconds(car.get("fastest_lap", {}).get("time"))
        optimal_s = _time_str_to_seconds(car.get("optimal_lap_time"))
        if fastest_s and optimal_s:
            gap = fastest_s - optimal_s
            if gap > 0.1:  # Only include meaningful gaps
                gaps.append({
                    "car_number": car.get("car_number"),
                    "driver_name": car.get("fastest_lap", {}).get("driver_name"),
                    "gap_seconds": gap,
                    "team": car.get("team", "N/A") # Assume team might be available
                })
    
    gaps.sort(key=lambda x: x["gap_seconds"], reverse=True)
    return [{
        "rank": i + 1,
        "car_number": item["car_number"],
        "driver_name": item["driver_name"],
        "time_left_on_track": _format_seconds(item["gap_seconds"], unit="s")
    } for i, item in enumerate(gaps)]

def _rank_drivers_by_traffic_management(traffic_data: List[Dict]) -> List[Dict]:
    """Ranks drivers by their effectiveness in traffic (lowest time lost)."""
    if not traffic_data: return []
    
    # The data is already ranked in the source file, so we just need to add context.
    all_lost_times = [d.get("avg_time_lost_total_sec") for d in traffic_data]
    valid_lost_times = [t for t in all_lost_times if t is not None]
    if not valid_lost_times: return []

    field_avg_lost_time = mean(valid_lost_times)
    
    contextual_list = []
    for item in traffic_data:
        time_lost = item.get("avg_time_lost_total_sec")
        if time_lost is not None:
            new_item = item.copy()
            new_item["performance_vs_avg"] = f"{((time_lost / field_avg_lost_time) - 1) * 100:+.1f}%"
            contextual_list.append(new_item)
            
    return contextual_list

def find_individual_outliers(data: Dict) -> List[Dict]:
    """Finds single-instance insights that aren't full rankings."""
    insights = []
    
    # Largest Teammate Pace Gap
    delta_data = data.get("driver_deltas_by_car", [])
    if delta_data:
        valid_deltas = [d for d in delta_data if _time_str_to_seconds(d.get("average_lap_time_delta_for_car")) is not None]
        if valid_deltas:
            worst_entry = max(valid_deltas, key=lambda x: _time_str_to_seconds(x.get("average_lap_time_delta_for_car")))
            gap_val = _time_str_to_seconds(worst_entry.get("average_lap_time_delta_for_car"))
            if gap_val and gap_val > 0.5: # Threshold for significance
                insights.append({
                    "category": "Driver Performance",
                    "type": "Largest Teammate Pace Gap",
                    "car_number": worst_entry.get("car_number"),
                    "details": f"Car #{worst_entry.get('car_number')} has the largest pace difference between its drivers, with an average gap of {gap_val:.3f}s in best lap times."
                })

    return insights

# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def derive_insights(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a structured dictionary of ranked insights from the analysis data."""
    
    enhanced_data = data.get("enhanced_strategy_analysis", [])
    pit_cycle_data = data.get("full_pit_cycle_analysis", [])
    
    insights = {
        "manufacturer_pace_ranking": _rank_by_metric(
            enhanced_data, "manufacturer", ["avg_green_pace_fuel_corrected"], higher_is_better=False, value_is_time=True
        ),
        "manufacturer_tire_wear_ranking": _rank_by_metric(
            enhanced_data, "manufacturer", ["tire_degradation_model", "deg_coeff_a"], higher_is_better=False, value_is_time=False
        ),
        "manufacturer_pit_cycle_ranking": _rank_by_metric(
            pit_cycle_data, "team", ["average_cycle_loss"], higher_is_better=False, value_is_time=True
        ),
        "car_untapped_potential_ranking": _rank_cars_by_untapped_potential(
            data.get("fastest_by_car_number", [])
        ),
        "driver_traffic_meister_ranking": _rank_drivers_by_traffic_management(
            data.get("traffic_management_analysis", [])
        ),
        "individual_outliers": find_individual_outliers(data)
    }
    
    return insights


def enrich_insights_with_ai(insights: Dict[str, Any]) -> Dict[str, Any]:
    """Adds LLM commentary to each list of ranked insights."""
    if not USE_AI_ENHANCED or not insights:
        return insights
        
    enriched_insights = insights.copy()
    
    # Example for one ranking list, can be expanded to others
    pace_ranking = insights.get("manufacturer_pace_ranking")
    if pace_ranking:
        prompt = (
            "You are a professional motorsport strategist AI. I will provide a JSON list of manufacturers ranked by their average race pace. "
            "Your task is to add a new key, 'llm_commentary', to each object in the list. "
            "This commentary should be a concise, professional one-sentence analysis about their performance relative to the field. "
            "You should provide an insightful commentary about the relative performance of each manufacturer, outlining how and where they compare relative to the field. "
            "Return only the updated JSON list, with no other text.\n\n"
            f"Pace Ranking:\n{json.dumps(pace_ranking, indent=2)}\n"
        )
        try:
            enriched_pace = ai_helpers.generate_json(
                prompt, max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 25000))
            )
            if isinstance(enriched_pace, list) and len(enriched_pace) == len(pace_ranking):
                enriched_insights["manufacturer_pace_ranking"] = enriched_pace
        except Exception as e:
            LOGGER.warning("AI enrichment for pace ranking failed: %s", e)
            
    return enriched_insights

# ----------------------------------------------------------------------------
# Flask application
# ----------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request():
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["analysis_path"])
        
        analysis_uri = payload["analysis_path"]
        comprehensive_analysis_uri = payload.get("comprehensive_analysis_path")  # Optional comprehensive analysis
        use_autonomous = payload.get("use_autonomous", True)  # Default to autonomous mode
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        LOGGER.exception("Invalid request: %s", exc)
        return jsonify({"error": "bad_request"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        run_id = analysis_uri.split('/')[3]
        local_analysis_path = tmp / "analysis_data.json"
        comprehensive_analysis_data = None
        
        try:
            LOGGER.info("Downloading analysis file: %s", analysis_uri)
            _gcs_download(analysis_uri, local_analysis_path)
            with local_analysis_path.open("r", encoding="utf-8") as fp:
                analysis_data = json.load(fp)

            # Also download comprehensive analysis if available for deeper insights
            if comprehensive_analysis_uri:
                LOGGER.info("Downloading comprehensive analysis file: %s", comprehensive_analysis_uri)
                local_comprehensive_path = tmp / "comprehensive_analysis.json"
                try:
                    _gcs_download(comprehensive_analysis_uri, local_comprehensive_path)
                    with local_comprehensive_path.open("r", encoding="utf-8") as fp:
                        comprehensive_analysis_data = json.load(fp)
                    LOGGER.info("Successfully loaded comprehensive analysis for deeper insights")
                except Exception as e:
                    LOGGER.warning("Could not load comprehensive analysis, using standard analysis only: %s", e)

            # Use comprehensive analysis for autonomous insight generation if available
            analysis_for_insights = comprehensive_analysis_data if comprehensive_analysis_data else analysis_data

            # Choose between autonomous and traditional insight generation
            if use_autonomous and USE_AI_ENHANCED:
                LOGGER.info("Using autonomous insight generation workflow...")
                insights = generate_insights_autonomously(analysis_for_insights, run_id)
            else:
                LOGGER.info("Using traditional rule-based insight generation...")
                insights = derive_insights(analysis_for_insights)
                
                if USE_AI_ENHANCED:
                    LOGGER.info("Enriching traditional insights with AI...")
                    insights = enrich_insights_with_ai(insights)

            insights_filename = f"{run_id}_insights.json"
            local_insights_path = tmp / insights_filename
            with local_insights_path.open("w", encoding="utf-8") as fp:
                json.dump(insights, fp, indent=2)
            LOGGER.info("Successfully generated insights file: %s", insights_filename)

            insights_gcs_path = f"{run_id}/{insights_filename}"
            insights_uri = _gcs_upload(local_insights_path, ANALYZED_DATA_BUCKET, insights_gcs_path)
            LOGGER.info("Uploaded insights to: %s", insights_uri)

            _publish_visualization_request(analysis_uri, insights_uri)
        except Exception as exc:
            LOGGER.exception("Processing failed: %s", exc)
            return jsonify({"error": "internal_error"}), 500

    return jsonify({
        "insights_path": insights_uri,
        "generation_method": "autonomous" if (use_autonomous and USE_AI_ENHANCED) else "traditional"
    }), 200

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "insight_hunter"}), 200

# ----------------------------------------------------------------------------
# Autonomous Investigation Workflow (Phase 2)
# ----------------------------------------------------------------------------

def generate_investigation_plan(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 1: Generate an investigation plan using LLM to identify anomalies and 
    specify which CoreAnalyzer tools to call for deeper analysis.
    """
    # Extract high-level summary for the prompt
    race_summary = analysis_data.get("race_summary", {})
    enhanced_strategy = analysis_data.get("enhanced_strategy_analysis", [])
    
    # Limit data to prevent token overflow
    summary_data = {
        "race_summary": race_summary,
        "car_count": len(enhanced_strategy),
        "sample_cars": enhanced_strategy[:5] if enhanced_strategy else []
    }
    
    prompt = f"""
You are an expert race strategist analyzing IMSA race data. Create a detailed investigation plan to find the most interesting and actionable insights.

High-Level Race Data:
{json.dumps(summary_data, indent=2)}

Available CoreAnalyzer Tools:
1. "driver_deltas" - Get performance gaps between drivers in the same car
   Parameters: {{"run_id": str, "car_number": optional str}}
   
2. "trend_analysis" - Get historical performance trends for a manufacturer at this track
   Parameters: {{"track": str, "session": str, "manufacturer": str, "num_years": optional int}}
   
3. "stint_analysis" - Get detailed stint and tire degradation analysis
   Parameters: {{"run_id": str, "car_number": optional str}}
   
4. "sector_analysis" - Get sector-by-sector performance breakdown
   Parameters: {{"run_id": str, "car_number": optional str}}

Instructions:
1. Identify 3-5 interesting findings from the high-level data
2. For each finding, specify which tool would provide deeper analysis
3. Explain your hypothesis about what the deeper analysis might reveal
4. Focus on actionable insights for teams and strategists

Return JSON format:
{{
  "investigation_tasks": [
    {{
      "finding": "Brief description of what caught your attention",
      "hypothesis": "What you expect to discover with deeper analysis",
      "tool": "tool_name",
      "parameters": {{"param": "value"}},
      "priority": "high|medium|low"
    }}
  ]
}}
"""
    
    try:
        plan = ai_helpers.generate_json_adaptive(prompt, temperature=0.7, max_output_tokens=8000)
        LOGGER.info(f"Generated investigation plan with {len(plan.get('investigation_tasks', []))} tasks")
        return plan
    except Exception as e:
        LOGGER.error(f"Failed to generate investigation plan: {e}")
        return {"investigation_tasks": []}


def execute_investigation_plan(plan: Dict[str, Any], run_id: str) -> List[Dict[str, Any]]:
    """
    Step 2: Execute the investigation plan by calling CoreAnalyzer tools
    and collecting detailed data for each investigation task.
    """
    results = []
    
    for task in plan.get("investigation_tasks", []):
        try:
            tool_name = task.get("tool")
            parameters = task.get("parameters", {})
            
            # Ensure run_id is included for tools that need it
            if "run_id" in parameters or tool_name in ["driver_deltas", "stint_analysis", "sector_analysis"]:
                parameters["run_id"] = run_id
            
            LOGGER.info(f"Executing tool: {tool_name} with params: {parameters}")
            
            # Call the CoreAnalyzer tool
            from agents.common.tool_caller import tool_caller
            tool_result = tool_caller.call_core_analyzer_tool(tool_name, **parameters)
            
            results.append({
                "finding": task.get("finding"),
                "hypothesis": task.get("hypothesis"),
                "tool": tool_name,
                "tool_result": tool_result,
                "status": "success",
                "priority": task.get("priority", "medium")
            })
            
            LOGGER.info(f"Successfully executed {tool_name}")
            
        except Exception as e:
            LOGGER.error(f"Tool execution failed for {task.get('tool')}: {e}")
            results.append({
                "finding": task.get("finding"),
                "hypothesis": task.get("hypothesis"),
                "tool": task.get("tool"),
                "status": "failed",
                "error": str(e),
                "priority": task.get("priority", "medium")
            })
    
    return results


def synthesize_final_insights(investigation_results: List[Dict[str, Any]], original_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Step 3: Synthesize final insights by combining initial hypotheses with 
    detailed tool results using LLM to create rich, actionable insights.
    """
    # Prepare data for synthesis prompt
    synthesis_data = []
    
    for result in investigation_results:
        if result.get("status") == "success":
            synthesis_data.append({
                "finding": result.get("finding"),
                "hypothesis": result.get("hypothesis"),
                "detailed_data": result.get("tool_result"),
                "priority": result.get("priority")
            })
    
    if not synthesis_data:
        LOGGER.warning("No successful tool results to synthesize")
        return {"autonomous_insights": [], "investigation_summary": "No successful investigations"}
    
    prompt = f"""
You are an expert race analyst creating final insights by combining initial observations with detailed analysis data.

For each investigation result, create a rich, actionable insight that:
1. Combines the initial finding with the detailed tool data
2. Provides specific, actionable recommendations
3. Explains the strategic implications
4. Includes relevant data points as evidence

Investigation Results:
{json.dumps(synthesis_data, indent=2)}

Return JSON format:
{{
  "autonomous_insights": [
    {{
      "category": "Strategic category (e.g., 'Driver Performance', 'Tire Strategy')",
      "type": "Specific insight type",
      "priority": "high|medium|low",
      "summary": "One-sentence summary of the insight",
      "detailed_analysis": "Comprehensive analysis with evidence",
      "actionable_recommendations": ["List of specific recommendations"],
      "supporting_data": {{"key_metrics": "Relevant numbers and comparisons"}},
      "confidence_level": "high|medium|low"
    }}
  ],
  "investigation_summary": {{
    "total_investigations": number,
    "successful_investigations": number,
    "key_discoveries": ["List of major discoveries"],
    "methodology": "Brief explanation of autonomous approach used"
  }}
}}
"""
    
    try:
        synthesis = ai_helpers.generate_json_adaptive(prompt, temperature=0.6, max_output_tokens=12000)
        
        # Enhance with investigation metadata
        synthesis["investigation_summary"]["methodology"] = "Autonomous LLM-driven investigation using CoreAnalyzer toolbox"
        synthesis["investigation_summary"]["total_investigations"] = len(investigation_results)
        synthesis["investigation_summary"]["successful_investigations"] = len(synthesis_data)
        
        LOGGER.info(f"Synthesized {len(synthesis.get('autonomous_insights', []))} final insights")
        return synthesis
        
    except Exception as e:
        LOGGER.error(f"Failed to synthesize insights: {e}")
        return {
            "autonomous_insights": [],
            "investigation_summary": {
                "total_investigations": len(investigation_results),
                "successful_investigations": len(synthesis_data),
                "error": str(e)
            }
        }


def generate_insights_autonomously(analysis_data: Dict[str, Any], run_id: str) -> Dict[str, Any]:
    """
    Main autonomous insight generation workflow that replaces the traditional rule-based approach.
    
    Workflow:
    1. LLM analyzes high-level data and creates investigation plan
    2. Execute plan by calling CoreAnalyzer tools for detailed data
    3. LLM synthesizes final insights combining hypotheses with detailed results
    """
    LOGGER.info("Starting autonomous insight generation workflow...")
    
    # Step 1: Generate investigation plan
    LOGGER.info("Step 1: Generating investigation plan...")
    plan = generate_investigation_plan(analysis_data)
    
    if not plan.get("investigation_tasks"):
        LOGGER.warning("No investigation tasks generated, falling back to traditional insights")
        return derive_insights(analysis_data)
    
    # Step 2: Execute investigation plan
    LOGGER.info("Step 2: Executing investigation plan...")
    investigation_results = execute_investigation_plan(plan, run_id)
    
    # Step 3: Synthesize final insights
    LOGGER.info("Step 3: Synthesizing final insights...")
    final_insights = synthesize_final_insights(investigation_results, analysis_data)
    
    # Combine autonomous insights with traditional insights for completeness
    traditional_insights = derive_insights(analysis_data)
    
    combined_insights = {
        "autonomous_insights": final_insights.get("autonomous_insights", []),
        "investigation_summary": final_insights.get("investigation_summary", {}),
        "traditional_insights": traditional_insights,
        "generation_method": "autonomous_investigation_workflow"
    }
    
    LOGGER.info("Autonomous insight generation completed successfully")
    return combined_insights

if __name__ == "__main__":
    # Register with Tool Registry at startup
    try:
        from agents.common.tool_caller import register_agent_with_registry
        port = int(os.environ.get("PORT", 8080))
        base_url = os.getenv("INSIGHT_HUNTER_URL", f"http://localhost:{port}")
        register_agent_with_registry("insight_hunter", base_url)
    except Exception as e:
        LOGGER.warning(f"Failed to register with Tool Registry: {e}")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)