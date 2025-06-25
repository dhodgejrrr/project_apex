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
from typing import Dict, Any

import requests
from flask import Flask, request, jsonify
from google.auth.transport import requests as grequests
from google.oauth2 import id_token

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

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def invoke_cloud_run(service_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """Invoke another Cloud Run service with an authenticated request.

    Args:
        service_url: Fully-qualified URL of the Cloud Run service.
        payload: JSON-serialisable payload to POST.

    Returns:
        Parsed JSON response from the target service.
    """
    # Fetch an identity token for the target audience (service URL).
    req = grequests.Request()
    token = id_token.fetch_id_token(req, service_url)

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "User-Agent": "project-apex-adk-orchestrator/1.0",
    }

    resp = requests.post(service_url, json=payload, headers=headers, timeout=900)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# ADK step implementations
# ---------------------------------------------------------------------------

@step
def analyze_data(state: Dict[str, Any]) -> Dict[str, Any]:
    service_url = os.environ["CORE_ANALYZER_URL"]
    response = invoke_cloud_run(
        service_url,
        {
            "run_id": state["run_id"],
            "csv_path": state["csv_gcs_path"],
            "pit_json_path": state["pit_gcs_path"],
        },
    )
    state["analysis_path"] = response["analysis_path"]
    return state


@step
def find_tactical_insights(state: Dict[str, Any]) -> Dict[str, Any]:
    service_url = os.environ["INSIGHT_HUNTER_URL"]
    response = invoke_cloud_run(service_url, {"analysis_path": state["analysis_path"]})
    state["insights_path"] = response["insights_path"]
    return state


@step
def find_historical_insights(state: Dict[str, Any]) -> Dict[str, Any]:
    service_url = os.environ["HISTORIAN_URL"]
    response = invoke_cloud_run(service_url, {"analysis_path": state["analysis_path"]})
    state["historical_path"] = response["historical_path"]
    return state


@step
def run_arbiter(state: Dict[str, Any]) -> Dict[str, Any]:
    service_url = os.environ["ARBITER_URL"]
    response = invoke_cloud_run(
        service_url,
        {
            "insights_path": state["insights_path"],
            "historical_path": state["historical_path"],
        },
    )
    state["briefing_path"] = response["briefing_path"]
    return state


@step
def generate_outputs(state: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder final step. Outputs are produced by downstream agents."""
    return state


# ---------------------------------------------------------------------------
# Flask entry-point for Pub/Sub push subscription
# ---------------------------------------------------------------------------

@app.route("/", methods=["POST"])
def handle_pubsub_push():  # noqa: D401
    """Handle Pub/Sub messages and kick off the ADK workflow."""
    envelope = request.get_json(force=True, silent=True)
    if not envelope or "message" not in envelope:
        return ("Invalid Pub/Sub message", 400)

    pubsub_message = envelope["message"]

    try:
        message_data = base64.b64decode(pubsub_message["data"]).decode("utf-8")
        initial_state = json.loads(message_data)
    except (KeyError, ValueError, json.JSONDecodeError):
        return ("Malformed message data", 400)

    # Build ADK execution graph.
    step1_analyze = execution.Executable(analyze_data)
    step2a_tactical = execution.Executable(find_tactical_insights, depends_on=[step1_analyze])
    step2b_historical = execution.Executable(find_historical_insights, depends_on=[step1_analyze])
    step3_arbiter = execution.Executable(run_arbiter, depends_on=[step2a_tactical, step2b_historical])
    step4_outputs = execution.Executable(generate_outputs, depends_on=[step3_arbiter])

    graph = [step1_analyze, step2a_tactical, step2b_historical, step3_arbiter, step4_outputs]

    # Execute the workflow (synchronously; Cloud Run may timeout on very long runs).
    # For production, you may prefer to execute asynchronously via a task queue.
    execution.execute(graph, initial_state)

    return jsonify({"status": "workflow_started", "run_id": initial_state.get("run_id")}), 202


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
