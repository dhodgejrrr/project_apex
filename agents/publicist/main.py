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

# AI helpers
from agents.common import ai_helpers
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
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

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


def _gen_tweets(insights: List[Dict[str, Any]], max_posts: int = 5) -> List[str]:
    """Call Gemini to generate up to `max_posts` social media posts."""
    key_ins = _select_key_insights(insights)
    if USE_AI_ENHANCED and key_ins:
        prompt = (
            "You are a social media manager for a professional race team. "
            "Create between 3 and 5 engaging tweets based on the JSON insights. "
            "Each tweet should be standalone, under 280 characters, and include relevant hashtags like #IMSA and appropriate emojis. "
            "Return ONLY a JSON array of strings.\n\nInsights JSON:\n" + json.dumps(key_ins, indent=2)
        )
        try:
            tweets = ai_helpers.generate_json(prompt, temperature=0.7, max_output_tokens=256)
            if isinstance(tweets, list):
                return tweets[:max_posts]
            LOGGER.warning("Unexpected tweets JSON: %s", tweets)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("AI tweet generation failed: %s", exc)
    # fallback template
    fallback = [f"🏁 {ins.get('type')}: {ins.get('details')} #IMSA #ProjectApex" for ins in key_ins[:max_posts]]
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
