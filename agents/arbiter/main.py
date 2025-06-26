import os
import json
import base64
import logging
import pathlib
import tempfile
import time
from datetime import datetime
from typing import Dict, Any

from flask import Flask, request, jsonify
from google.cloud import storage

from agents.common import ai_helpers
from agents.common.request_utils import parse_request_payload, validate_required_fields

# Configuration
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET")
ARBITER_PROMPT_TEMPLATE_PATH = pathlib.Path(__file__).parent / "prompt_template.md"
ARBITER_PROMPT_TEMPLATE = ARBITER_PROMPT_TEMPLATE_PATH.read_text()

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

_storage_client = None

def get_storage_client():
    global _storage_client
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

def _gcs_download_json(gcs_uri: str) -> Dict[str, Any]:
    """Downloads and parses a JSON file from GCS."""
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = get_storage_client().bucket(bucket_name)
    blob = bucket.blob(blob_name)
    return json.loads(blob.download_as_string())

def _gcs_upload(local_path: pathlib.Path, dest_blob: str) -> str:
    """Uploads a local file to the analyzed data bucket."""
    bucket = get_storage_client().bucket(ANALYZED_DATA_BUCKET)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)
    return f"gs://{ANALYZED_DATA_BUCKET}/{dest_blob}"

app = Flask(__name__)

@app.route("/", methods=["POST"])
def handle_request():
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["insights_path"])
        
        insights_path = payload["insights_path"]
        historical_path = payload.get("historical_path")  # May be null
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        LOGGER.error(f"Request parsing failed: {e}")
        return jsonify({"error": "invalid_request"}), 400

    try:
        tactical_insights = _gcs_download_json(insights_path)
        historical_insights = _gcs_download_json(historical_path) if historical_path else {}
        
        prompt = ARBITER_PROMPT_TEMPLATE.format(
            tactical_insights_json=json.dumps(tactical_insights, indent=2),
            historical_insights_json=json.dumps(historical_insights, indent=2)
        )
        
        briefing = ai_helpers.generate_json(prompt, temperature=0.5)
        
        run_id = pathlib.Path(insights_path).parent.name
        briefing_filename = f"{run_id}_final_briefing.json"
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".json") as tmp_file:
            json.dump(briefing, tmp_file)
            tmp_file_path = tmp_file.name

        briefing_gcs_path = _gcs_upload(pathlib.Path(tmp_file_path), f"{run_id}/briefing/{briefing_filename}")
        os.unlink(tmp_file_path)

        return jsonify({"briefing_path": briefing_gcs_path})

    except Exception as e:
        LOGGER.error(f"Arbiter agent failed: {e}", exc_info=True)
        return "Internal Server Error", 500

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "arbiter"}), 200

@app.route("/tools/capabilities", methods=["GET"])
def get_tool_capabilities():
    """Return information about available coordination tools."""
    capabilities = {
        "tools": [
            {
                "name": "coordinate",
                "endpoint": "/coordinate",
                "description": "Coordinate multiple agents and tasks",
                "required_params": ["slaves", "task_distribution"],
                "optional_params": ["timeout_seconds", "coordination_strategy"],
                "category": "coordination"
            },
            {
                "name": "aggregate_votes",
                "endpoint": "/aggregate_votes",
                "description": "Aggregate votes for consensus decision making",
                "required_params": ["decision_key", "voters", "threshold"],
                "optional_params": ["voting_strategy"],
                "category": "coordination"
            },
            {
                "name": "resolve_conflict",
                "endpoint": "/resolve_conflict",
                "description": "Resolve conflicts between agents",
                "required_params": ["conflict_data", "resolution_strategy"],
                "optional_params": ["priority_agents"],
                "category": "coordination"
            }
        ]
    }
    return jsonify(capabilities), 200

@app.route("/coordinate", methods=["POST"])
def coordinate_agents():
    """Coordinate multiple agents in a master-slave pattern."""
    try:
        payload = request.get_json()
        slaves = payload["slaves"]
        task_distribution = payload["task_distribution"]
        timeout_seconds = payload.get("timeout_seconds", 300)
        
        # Simple coordination logic
        coordination_result = {
            "coordination_id": f"coord_{int(time.time())}",
            "master_agent": "arbiter",
            "slave_agents": slaves,
            "task_assignments": task_distribution,
            "status": "coordinated",
            "timestamp": datetime.now().isoformat()
        }
        
        LOGGER.info(f"Coordinated {len(slaves)} agents with task distribution")
        
        return jsonify(coordination_result), 200
        
    except Exception as e:
        LOGGER.error(f"Coordination failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/aggregate_votes", methods=["POST"])
def aggregate_votes():
    """Aggregate votes for consensus decision making."""
    try:
        payload = request.get_json()
        decision_key = payload["decision_key"]
        voters = payload["voters"]
        threshold = payload["threshold"]
        
        # Mock vote aggregation (in real implementation, would fetch from state manager)
        votes = {}
        for voter in voters:
            # Simulate vote retrieval
            votes[voter] = {"decision": "approve", "weight": 1.0, "timestamp": datetime.now().isoformat()}
        
        # Calculate consensus
        approve_votes = sum(1 for vote in votes.values() if vote["decision"] == "approve")
        total_votes = len(votes)
        consensus_ratio = approve_votes / total_votes if total_votes > 0 else 0
        
        consensus_result = {
            "decision_key": decision_key,
            "votes": votes,
            "consensus_ratio": consensus_ratio,
            "threshold": threshold,
            "consensus_reached": consensus_ratio >= threshold,
            "final_decision": "approved" if consensus_ratio >= threshold else "rejected",
            "aggregated_at": datetime.now().isoformat()
        }
        
        LOGGER.info(f"Aggregated {total_votes} votes, consensus: {consensus_ratio:.2%}")
        
        return jsonify(consensus_result), 200
        
    except Exception as e:
        LOGGER.error(f"Vote aggregation failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/resolve_conflict", methods=["POST"])
def resolve_conflict():
    """Resolve conflicts between agents."""
    try:
        payload = request.get_json()
        conflict_data = payload["conflict_data"]
        resolution_strategy = payload["resolution_strategy"]
        priority_agents = payload.get("priority_agents", [])
        
        # Simple conflict resolution logic
        if resolution_strategy == "priority_based":
            # Give precedence to priority agents
            resolution = {
                "strategy": "priority_based",
                "winner": priority_agents[0] if priority_agents else "arbiter",
                "reasoning": "Resolved based on agent priority"
            }
        elif resolution_strategy == "timestamp_based":
            # Use first-come-first-served
            resolution = {
                "strategy": "timestamp_based", 
                "winner": "earliest_submitter",
                "reasoning": "Resolved based on submission timestamp"
            }
        else:
            # Default to arbiter decision
            resolution = {
                "strategy": "arbiter_decision",
                "winner": "arbiter_choice",
                "reasoning": "Resolved by arbiter using domain knowledge"
            }
        
        conflict_resolution = {
            "conflict_id": f"conflict_{int(time.time())}",
            "conflict_data": conflict_data,
            "resolution": resolution,
            "resolved_at": datetime.now().isoformat(),
            "resolved_by": "arbiter"
        }
        
        LOGGER.info(f"Resolved conflict using {resolution_strategy} strategy")
        
        return jsonify(conflict_resolution), 200
        
    except Exception as e:
        LOGGER.error(f"Conflict resolution failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Register with Tool Registry at startup
    try:
        from agents.common.tool_caller import register_agent_with_registry
        port = int(os.getenv("PORT", 8080))
        base_url = os.getenv("ARBITER_URL", f"http://localhost:{port}")
        register_agent_with_registry("arbiter", base_url)
    except Exception as e:
        print(f"Failed to register with Tool Registry: {e}")
    
    app.run(host="0.0.0.0", port=port)