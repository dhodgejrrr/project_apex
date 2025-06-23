from __future__ import annotations
import os
import json
import base64
import logging
from typing import Dict, Any

from adk.core import execution
from adk.core.namespace import adk
import google.auth
import requests

# --- Configuration ---
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
REGION = os.getenv("GCP_REGION", "us-central1")

# Get service URLs from environment variables
CORE_ANALYZER_URL = os.getenv("CORE_ANALYZER_URL")
INSIGHT_HUNTER_URL = os.getenv("INSIGHT_HUNTER_URL")
HISTORIAN_URL = os.getenv("HISTORIAN_URL")
VISUALIZER_URL = os.getenv("VISUALIZER_URL")
SCRIBE_URL = os.getenv("SCRIBE_URL")
PUBLICIST_URL = os.getenv("PUBLICIST_URL")

LOGGER = logging.getLogger(__name__)

# --- Helper to invoke Cloud Run services securely ---
def invoke_cloud_run(service_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invokes a Cloud Run service with authenticated request."""
    if not service_url:
        raise ValueError(f"Service URL is not set for the target agent.")
        
    auth_req = google.auth.transport.requests.Request()
    identity_token = google.auth.fetch_id_token(auth_req, service_url)

    headers = {"Authorization": f"Bearer {identity_token}"}
    response = requests.post(service_url, json=payload, headers=headers, timeout=900)
    response.raise_for_status()
    return response.json()

# --- ADK Steps ---

@adk.step()
def analyze_data(state: Dict[str, Any]) -> Dict[str, Any]:
    LOGGER.info(f"Step 1: Calling CoreAnalyzer for run_id: {state['run_id']}")
    payload = {
        "csv_path": state["csv_gcs_path"],
        "pit_json_path": state["pit_gcs_path"],
        "run_id": state["run_id"] # Pass run_id for consistent output paths
    }
    response = invoke_cloud_run(CORE_ANALYZER_URL, payload)
    state['analysis_path'] = response['analysis_path']
    LOGGER.info(f"CoreAnalyzer finished. Analysis at: {state['analysis_path']}")
    return state

@adk.step(parallel=True)
def find_tactical_insights(state: Dict[str, Any]) -> Dict[str, Any]:
    LOGGER.info("Step 2a (parallel): Calling InsightHunter.")
    response = invoke_cloud_run(INSIGHT_HUNTER_URL, {"analysis_path": state['analysis_path']})
    state['insights_path'] = response['insights_path']
    LOGGER.info(f"InsightHunter finished. Insights at: {state['insights_path']}")
    return state
    
@adk.step(parallel=True)
def find_historical_insights(state: Dict[str, Any]) -> Dict[str, Any]:
    LOGGER.info("Step 2b (parallel): Calling Historian.")
    response = invoke_cloud_run(HISTORIAN_URL, {"analysis_path": state['analysis_path']})
    # Historian might not produce output if no history exists.
    state['historical_path'] = response.get('historical_insights_path')
    LOGGER.info(f"Historian finished. Insights at: {state['historical_path']}")
    return state

@adk.step(parallel=True)
def generate_visuals(state: Dict[str, Any]) -> Dict[str, Any]:
    LOGGER.info("Step 3a (parallel): Calling Visualizer.")
    payload = {"analysis_path": state['analysis_path'], "insights_path": state['insights_path']}
    response = invoke_cloud_run(VISUALIZER_URL, payload)
    state['visuals_path'] = response['visuals_path']
    LOGGER.info(f"Visualizer finished. Visuals at: {state['visuals_path']}")
    return state

@adk.step(parallel=True)
def generate_report(state: Dict[str, Any]) -> Dict[str, Any]:
    LOGGER.info("Step 3b (parallel): Calling Scribe.")
    payload = {"analysis_path": state['analysis_path'], "insights_path": state['insights_path']}
    response = invoke_cloud_run(SCRIBE_URL, payload)
    state['report_path'] = response['report_path']
    LOGGER.info(f"Scribe finished. Report at: {state['report_path']}")
    return state

@adk.step(parallel=True)
def generate_social_posts(state: Dict[str, Any]) -> Dict[str, Any]:
    LOGGER.info("Step 3c (parallel): Calling Publicist.")
    payload = {"analysis_path": state['analysis_path'], "insights_path": state['insights_path']}
    response = invoke_cloud_run(PUBLICIST_URL, payload)
    state['social_path'] = response['social_path']
    LOGGER.info(f"Publicist finished. Posts at: {state['social_path']}")
    return state

# --- Main execution block for Cloud Run / PubSub trigger ---
app = Flask(__name__)

@app.route("/", methods=["POST"])
def main():
    """Entry point triggered by Pub/Sub message from UI Portal."""
    envelope = request.get_json()
    if not envelope or "message" not in envelope:
        return "Bad Request: Invalid Pub/Sub message format", 400

    pubsub_message = envelope["message"]
    initial_state = json.loads(base64.b64decode(pubsub_message["data"]).decode("utf-8"))
    
    LOGGER.info(f"Received orchestration request with initial state: {initial_state}")
    
    # Define the execution graph
    graph = execution.Graph()
    step1 = execution.Executable(analyze_data)
    step2a = execution.Executable(find_tactical_insights, depends_on=[step1])
    step2b = execution.Executable(find_historical_insights, depends_on=[step1])
    step3a = execution.Executable(generate_visuals, depends_on=[step2a, step2b])
    step3b = execution.Executable(generate_report, depends_on=[step2a, step2b])
    step3c = execution.Executable(generate_social_posts, depends_on=[step2a, step2b])

    graph.add_executable(step1)
    graph.add_executable(step2a)
    graph.add_executable(step2b)
    graph.add_executable(step3a)
    graph.add_executable(step3b)
    graph.add_executable(step3c)

    # Execute the graph. This is a blocking call.
    final_state = execution.execute(graph, initial_state)
    
    LOGGER.info(f"Orchestration complete for run_id {initial_state['run_id']}. Final state: {final_state}")
    
    return "OK", 200

if __name__ == '__main__':
    # This part is for local testing of the orchestrator itself if needed.
    # The primary execution path is the Flask app for Cloud Run.
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)