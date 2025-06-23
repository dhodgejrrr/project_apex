"""Scribe Cloud Run service for Project Apex.

Generates a PDF engineering report from analysis and insights JSON using Jinja2
and WeasyPrint.
"""
from __future__ import annotations


import json
import logging
import os
import pathlib
import tempfile

# AI helpers
from agents.common import ai_helpers
import os
from typing import Any, Dict, List

from flask import Flask, request, jsonify
from google.cloud import storage
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
TEMPLATE_NAME = "report_template.html"
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

# --- NEW: Load the prompt template from the file on startup ---
PROMPT_TEMPLATE_PATH = pathlib.Path(__file__).parent / "prompt_template.md"
PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("scribe")

# ---------------------------------------------------------------------------
# Google Cloud client (lazy)
# ---------------------------------------------------------------------------
_storage_client: storage.Client | None = None


def _storage() -> storage.Client:
    global _storage_client  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _gcs_download(gcs_uri: str, dest: pathlib.Path) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    _storage().bucket(bucket_name).blob(blob_name).download_to_filename(dest)


def _gcs_upload(local_path: pathlib.Path, dest_blob: str) -> str:
    bucket = _storage().bucket(ANALYZED_DATA_BUCKET)
    blob = bucket.blob(dest_blob)
    blob.upload_from_filename(local_path)
    return f"gs://{ANALYZED_DATA_BUCKET}/{dest_blob}"

# ---------------------------------------------------------------------------
# AI narrative generation
# ---------------------------------------------------------------------------

def _generate_narrative(insights: Dict[str, List[Dict[str, Any]]], analysis: Dict[str, Any]) -> Dict[str, Any] | None:
    """Uses Gemini to craft executive summary paragraph and tactical recommendations."""
    if not USE_AI_ENHANCED or not insights:
        return None

    # --- MODIFIED: Use the loaded template and format it with both JSON objects ---
    # WARNING: Passing the full analysis_enhanced.json can be very large and may
    # exceed model input token limits. For production, consider summarizing
    # this payload first or ensuring you use a model with a large context window.
    prompt = PROMPT_TEMPLATE.format(
        insights_json=json.dumps(insights, indent=2),
        analysis_enhanced_json=json.dumps(analysis, indent=2),
    )

    # Pass the entire insights dictionary (already grouped by category) to the model
    # prompt = (
    #     "You are a motorsport performance engineer. Based on the race insights JSON, "
    #     "write a concise executive summary (<= 500 words) and 3 tactical recommendations. "
    #     "You should outline which cars, manufacturers, and teams are leading, mid-field, and lagging based on ALL provided data. "
    #     "Consider extreme outliers as not relevant, and focus only on those that are majorly influencing the race in various aspects of the field. "
    #     "Respond ONLY as minified JSON with keys 'executive_summary' and 'tactical_recommendations' (array of strings).\n\n"
    #     f"Insights:\n{json.dumps(insights, indent=2)}\n"
    # )
    try:
        result = ai_helpers.generate_json(
            prompt, temperature=0.5, max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 25000))
        )
        if isinstance(result, dict):
            return result
        LOGGER.warning("Unexpected narrative JSON format: %s", result)
    except Exception:
        LOGGER.exception("Narrative generation failed")

    return None

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _render_report(analysis: Dict[str, Any], insights: List[Dict[str, Any]], narrative: Dict[str, Any] | None, output_pdf: pathlib.Path) -> None:
    env = Environment(
        loader=FileSystemLoader(pathlib.Path(__file__).parent),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(TEMPLATE_NAME)

    # Simple event name extraction from analysis metadata if available
    event_id = pathlib.Path(analysis.get("metadata", {}).get("event_id", "")).stem or "Race Event"

    html_str = template.render(event_name=event_id, insights=insights, narrative=narrative or {})
    HTML(string=html_str).write_pdf(str(output_pdf))

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request():
    try:
        req_json = request.get_json(force=True, silent=True)
        if req_json is None:
            return jsonify({"error": "invalid_json"}), 400
        analysis_uri: str | None = req_json.get("analysis_path")
        insights_uri: str | None = req_json.get("insights_path")
        if not analysis_uri or not insights_uri:
            return jsonify({"error": "missing_fields"}), 400
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Bad request: %s", exc)
        return jsonify({"error": "bad_request"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        local_analysis = tmp / pathlib.Path(analysis_uri).name
        local_insights = tmp / pathlib.Path(insights_uri).name
        try:
            _gcs_download(analysis_uri, local_analysis)
            _gcs_download(insights_uri, local_insights)
            analysis_data = json.loads(local_analysis.read_text())
            insights_data = json.loads(local_insights.read_text())

            pdf_path = tmp / "race_report.pdf"
            narrative = _generate_narrative(insights_data, analysis_data)
            _render_report(analysis_data, insights_data, narrative, pdf_path)

            basename = local_analysis.stem.replace("_results_enhanced", "")
            out_uri = _gcs_upload(pdf_path, f"{basename}/reports/race_report.pdf")
            LOGGER.info("Uploaded PDF report to %s", out_uri)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return jsonify({"error": "internal_error"}), 500
    return jsonify({"report_path": out_uri}), 200
