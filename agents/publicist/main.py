"""Publicist Cloud Run service for Project Apex.

Generates social-media copy (tweet variations) from insights JSON using Vertex AI
Gemini. Expects Pub/Sub push payload with `analysis_path` and `insights_path`.
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
from google.cloud import aiplatform

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
MODEL_NAME = os.getenv("VERTEX_MODEL", "gemini-1.0-pro")

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
# Gemini helper
# ---------------------------------------------------------------------------

def _init_gemini() -> None:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)


def _gen_tweets(insights: List[Dict[str, Any]], max_posts: int = 5) -> List[str]:
    """Call Gemini to generate up to `max_posts` social media posts."""
    try:
        _init_gemini()
        model = aiplatform.TextGenerationModel.from_pretrained(MODEL_NAME)
        prompt = (
            "You are a social media manager for a professional race team. "
            "Create up to 5 engaging tweets based on these race insights. "
            "Include relevant hashtags like #IMSA and emojis where appropriate.\n\n"
            f"Insights:\n{json.dumps(insights, indent=2)}\n"
        )
        response = model.predict(prompt, temperature=0.7, max_output_tokens=256)
        # Assume response is a string with one tweet per line
        tweets = [line.strip("-• ") for line in response.text.split("\n") if line.strip()]
        return tweets[:max_posts]
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Gemini generation failed (%s). Falling back to template.", exc)
        # Fallback: simple templated messages
        fallback = []
        for ins in insights[:max_posts]:
            fallback.append(f"🏁 {ins.get('type')}: {ins.get('details')} #IMSA #ProjectApex")
        return fallback

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request() -> Response:
    try:
        payload_b64 = request.get_json()["message"]["data"]
        payload = json.loads(base64.b64decode(payload_b64))
        insights_uri: str = payload["insights_path"]
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Bad request: %s", exc)
        return Response("Bad Request", status=400)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        local_insights = tmp / pathlib.Path(insights_uri).name
        try:
            _gcs_download(insights_uri, local_insights)
            insights_data = json.loads(local_insights.read_text())

            tweets = _gen_tweets(insights_data)
            output_json = tmp / "social_media_posts.json"
            json.dump({"posts": tweets}, output_json.open("w", encoding="utf-8"))

            basename = pathlib.Path(insights_uri).stem.replace("_insights", "")
            out_uri = _gcs_upload(output_json, f"{basename}/social/social_media_posts.json")
            LOGGER.info("Uploaded social media posts to %s", out_uri)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return Response("Internal Server Error", status=500)
    return Response(status=204)
