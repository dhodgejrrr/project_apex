"""Scribe Cloud Run service for Project Apex.

Generates a PDF engineering report from analysis and insights JSON using Jinja2
and WeasyPrint.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import tempfile
from typing import Any, Dict, List

from flask import Flask, Response, request
from google.cloud import storage
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
TEMPLATE_NAME = "report_template.html"

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
# Report generation
# ---------------------------------------------------------------------------

def _render_report(analysis: Dict[str, Any], insights: List[Dict[str, Any]], output_pdf: pathlib.Path) -> None:
    env = Environment(
        loader=FileSystemLoader(pathlib.Path(__file__).parent),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(TEMPLATE_NAME)

    # Simple event name extraction from analysis metadata if available
    event_id = pathlib.Path(analysis.get("metadata", {}).get("event_id", "")).stem or "Race Event"

    html_str = template.render(event_name=event_id, insights=insights)
    HTML(string=html_str).write_pdf(str(output_pdf))

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request() -> Response:
    try:
        payload_b64 = request.get_json()["message"]["data"]
        payload = json.loads(base64.b64decode(payload_b64))
        analysis_uri: str = payload["analysis_path"]
        insights_uri: str = payload["insights_path"]
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Bad request: %s", exc)
        return Response("Bad Request", status=400)

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
            _render_report(analysis_data, insights_data, pdf_path)

            basename = local_analysis.stem.replace("_results_enhanced", "")
            out_uri = _gcs_upload(pdf_path, f"{basename}/reports/race_report.pdf")
            LOGGER.info("Uploaded PDF report to %s", out_uri)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return Response("Internal Server Error", status=500)
    return Response(status=204)
