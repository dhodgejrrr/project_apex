"""ADK Orchestrator Agent for Project Apex

This service is deployed on Cloud Run and triggered via a Pub/Sub push
subscription. It orchestrates execution of all downstream agents using the
Google Agent Development Kit (ADK). Each downstream agent is invoked securely
via an authenticated HTTP POST request that carries the workflow `state`
dictionary between steps.
"""
from __future__ import annotations

import base64
import json
import os
import logging
import time
from typing import Dict, Any

import requests
from flask import Flask, request, jsonify
from google.auth.transport import requests as grequests
from google.oauth2 import id_token
from google.cloud import pubsub_v1

# ADK imports – assuming the library follows the public preview interface.
try:
    from adk import step, execution  # type: ignore
except ImportError:  # Local development fallback
    # Create no-op stand-ins so the file is importable in local envs without ADK.
    def step(func):  # type: ignore
        return func

    class _Executable:  # type: ignore
        def __init__(self, func, depends_on=None):
            self.func = func
            self.depends_on = depends_on or []

    class _Execution:  # type: ignore
        Executable = _Executable

        @staticmethod
        def execute(graph, initial_state):  # pylint: disable=unused-argument
            # Basic linear execution fallback for local testing.
            state = initial_state.copy()
            for node in graph:
                state = node.func(state)
            return state

    execution = _Execution()  # type: ignore

app = Flask(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def invoke_cloud_run(service_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Invokes another Cloud Run service with robust retry logic. In a real GCP environment, 
    it uses an identity token. In a local emulator environment, it makes a simple
    unauthenticated HTTP request with retries.
    """
    max_retries = 3
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Check if running in a local/emulated environment by looking for the emulator host var
            if os.getenv("PUBSUB_EMULATOR_HOST"):
                LOGGER.info(f"Invoking {service_url} via simple HTTP (local) - attempt {attempt + 1}/{max_retries}...")
                
                # Wrap payload in Pub/Sub push message format for local testing
                import base64
                message_data = base64.b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8")
                pubsub_payload = {
                    "message": {
                        "data": message_data,
                        "messageId": f"local-message-{int(time.time())}"
                    },
                    "subscription": "projects/local-dev/subscriptions/local-trigger"
                }
                
                resp = requests.post(service_url, json=pubsub_payload, timeout=900)
                resp.raise_for_status()
                LOGGER.info(f"Successfully invoked {service_url}")
                return resp.json()
            else:
                # Production path: use authenticated request
                LOGGER.info(f"Invoking {service_url} via authenticated request (production) - attempt {attempt + 1}/{max_retries}...")
                auth_req = grequests.Request()
                id_token_credentials = id_token.fetch_id_token(auth_req, service_url)

                headers = {
                    "Authorization": f"Bearer {id_token_credentials}",
                    "Content-Type": "application/json",
                    "User-Agent": "project-apex-adk-orchestrator/1.0",
                }

                resp = requests.post(service_url, json=payload, headers=headers, timeout=900)
                resp.raise_for_status()
                LOGGER.info(f"Successfully invoked {service_url}")
                return resp.json()
                
        except requests.exceptions.ConnectionError as e:
            LOGGER.warning(f"Connection error when calling {service_url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                LOGGER.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                LOGGER.error(f"Failed to connect to {service_url} after {max_retries} attempts")
                raise
        except requests.exceptions.Timeout as e:
            LOGGER.warning(f"Timeout when calling {service_url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                LOGGER.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                LOGGER.error(f"Timeout calling {service_url} after {max_retries} attempts")
                raise
        except requests.exceptions.HTTPError as e:
            # For HTTP errors (4xx, 5xx), we might want to retry 5xx but not 4xx
            if e.response.status_code >= 500 and attempt < max_retries - 1:
                LOGGER.warning(f"Server error {e.response.status_code} when calling {service_url} (attempt {attempt + 1}/{max_retries}): {e}")
                delay = base_delay * (2 ** attempt)
                LOGGER.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                LOGGER.error(f"HTTP error when calling {service_url}: {e}")
                raise
        except Exception as e:
            LOGGER.error(f"Unexpected error when calling {service_url} (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                LOGGER.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                raise
    
    # Should never reach here due to the raise statements above
    raise RuntimeError(f"Failed to invoke {service_url} after {max_retries} attempts")


# ---------------------------------------------------------------------------
# ADK step implementations
# ---------------------------------------------------------------------------

@step
def analyze_data(state: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the Core Analyzer service."""
    service_url = os.environ["CORE_ANALYZER_URL"]
    payload = {
        "run_id": state["run_id"],
        "csv_path": state["csv_gcs_path"],
        "pit_json_path": state.get("pit_gcs_path"),
        "fuel_json_path": state.get("fuel_gcs_path"),
    }
    LOGGER.info(f"Calling Core Analyzer with payload: {payload}")
    response = invoke_cloud_run(service_url, payload)
    state["analysis_path"] = response["analysis_path"]
    return state


@step
def find_tactical_insights(state: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the Insight Hunter service."""
    service_url = os.environ["INSIGHT_HUNTER_URL"]
    payload = {"analysis_path": state["analysis_path"]}
    LOGGER.info(f"Calling Insight Hunter with payload: {payload}")
    response = invoke_cloud_run(service_url, payload)
    state["insights_path"] = response["insights_path"]
    return state


@step
def find_historical_insights(state: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the Historian service."""
    service_url = os.environ["HISTORIAN_URL"]
    payload = {"analysis_path": state["analysis_path"]}
    LOGGER.info(f"Calling Historian with payload: {payload}")
    response = invoke_cloud_run(service_url, payload)
    state["historical_path"] = response.get("historical_path")
    return state


@step
def run_arbiter(state: Dict[str, Any]) -> Dict[str, Any]:
    """Calls the Arbiter service."""
    service_url = os.environ["ARBITER_URL"]
    payload = {
        "insights_path": state.get("insights_path"),
        "historical_path": state.get("historical_path"),
    }
    LOGGER.info(f"Calling Arbiter with payload: {payload}")
    response = invoke_cloud_run(service_url, payload)
    state["briefing_path"] = response.get("briefing_path")
    return state


@step
def generate_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Generate final outputs by calling visualizer, scribe, and publicist services."""
    
    # Check if running in local mode (with emulator)
    if os.getenv("PUBSUB_EMULATOR_HOST"):
        LOGGER.info("Running in local mode - calling agents directly via HTTP")
        
        # The payload for all downstream services
        payload = {
            "analysis_path": state.get("analysis_path"),
            "insights_path": state.get("insights_path"),
            "briefing_path": state.get("briefing_path")
        }
        
        # Call visualizer
        visualizer_url = os.environ.get("VISUALIZER_URL", "http://visualizer:8080/")
        LOGGER.info(f"Calling Visualizer with payload: {payload}")
        visualizer_response = invoke_cloud_run(visualizer_url, payload)
        
        # Call scribe  
        scribe_url = os.environ.get("SCRIBE_URL", "http://scribe:8080/")
        LOGGER.info(f"Calling Scribe with payload: {payload}")
        scribe_response = invoke_cloud_run(scribe_url, payload)
        
        # Call publicist
        publicist_url = os.environ.get("PUBLICIST_URL", "http://publicist:8080/")
        LOGGER.info(f"Calling Publicist with payload: {payload}")
        publicist_response = invoke_cloud_run(publicist_url, payload)
        
        # Update state with response paths
        state.update({
            "visualizations_path": visualizer_response.get("visualizations_path"),
            "report_path": scribe_response.get("report_path"),
            "social_posts_path": publicist_response.get("social_posts_path")
        })
        
        LOGGER.info("All final outputs generated successfully")
        
    else:
        # In GCP, use Pub/Sub to trigger downstream services
        LOGGER.info("Running in GCP mode - using Pub/Sub to trigger downstream services")
        project_id = os.environ["GOOGLE_CLOUD_PROJECT"]
        publisher = pubsub_v1.PublisherClient()
        topic_path = publisher.topic_path(project_id, "visualization-requests")
        
        message_payload = {
            "analysis_path": state.get("analysis_path"),
            "insights_path": state.get("insights_path"),
            "briefing_path": state.get("briefing_path")
        }
        message_data = json.dumps(message_payload).encode("utf-8")
        
        future = publisher.publish(topic_path, message_data)
        future.result(timeout=60)
        LOGGER.info(f"Published to {topic_path} to trigger final output generation.")
    
    return state


# ---------------------------------------------------------------------------
# Flask entry-point for Pub/Sub push subscription
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for container readiness."""
    return jsonify({"status": "healthy", "service": "adk-orchestrator"}), 200


@app.route("/", methods=["POST"])
def handle_pubsub_push():
    """Handle Pub/Sub messages and kick off the ADK workflow."""
    envelope = request.get_json(force=True, silent=True)
    if not envelope or "message" not in envelope:
        LOGGER.error("Invalid Pub/Sub message format")
        return ("Invalid Pub/Sub message", 400)

    try:
        message_data = base64.b64decode(envelope["message"]["data"]).decode("utf-8")
        initial_state = json.loads(message_data)
        LOGGER.info(f"Received initial state: {initial_state}")
    except Exception as e:
        LOGGER.error(f"Malformed message data: {e}")
        return ("Malformed message data", 400)

    # Build ADK execution graph.
    step1_analyze = execution.Executable(analyze_data)
    step2a_tactical = execution.Executable(find_tactical_insights, depends_on=[step1_analyze])
    step2b_historical = execution.Executable(find_historical_insights, depends_on=[step1_analyze])
    step3_arbiter = execution.Executable(run_arbiter, depends_on=[step2a_tactical, step2b_historical])
    step4_outputs = execution.Executable(generate_outputs, depends_on=[step3_arbiter])

    graph = [step1_analyze, step2a_tactical, step2b_historical, step3_arbiter, step4_outputs]
    
    try:
        execution.execute(graph, initial_state)
        return jsonify({"status": "workflow_completed", "run_id": initial_state.get("run_id")}), 200
    except Exception as e:
        LOGGER.error(f"ADK execution failed: {e}", exc_info=True)
        return jsonify({"status": "workflow_failed", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)