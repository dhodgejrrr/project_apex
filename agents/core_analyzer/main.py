"""CoreAnalyzer Cloud Run service for Project Apex.

Receives Pub/Sub push messages containing paths to race CSV and pit JSON files
in Google Cloud Storage, runs the IMSADataAnalyzer, writes the resulting
_enhanced.json back to Cloud Storage, and publishes a notification to the
`insight-requests` topic.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import tempfile
from typing import Any, Dict

from flask import Flask, Response, request
from google.cloud import pubsub_v1, storage

# Third-party analysis module provided separately in this repository.
from imsa_analyzer import IMSADataAnalyzer  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
INSIGHT_TOPIC_ID = os.getenv("INSIGHT_REQUESTS_TOPIC", "insight-requests")

# ---------------------------------------------------------------------------
# Logging setup (structured for Cloud Logging)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("core_analyzer")

# ---------------------------------------------------------------------------
# Google Cloud clients (lazily initialised)
# ---------------------------------------------------------------------------
_storage_client: storage.Client | None = None
_publisher: pubsub_v1.PublisherClient | None = None


def _init_clients() -> tuple[storage.Client, pubsub_v1.PublisherClient]:
    global _storage_client, _publisher  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _storage_client, _publisher

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _download_blob(gcs_uri: str, dest_path: pathlib.Path) -> None:
    """Downloads a GCS object to a local file path."""
    storage_client, _ = _init_clients()
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest_path))
    LOGGER.info("Downloaded %s to %s", gcs_uri, dest_path)


def _upload_file(local_path: pathlib.Path, dest_bucket: str, dest_blob_name: str) -> str:
    """Uploads local file to GCS and returns the gs:// URI."""
    storage_client, _ = _init_clients()
    bucket = storage_client.bucket(dest_bucket)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(str(local_path))
    gcs_uri = f"gs://{dest_bucket}/{dest_blob_name}"
    LOGGER.info("Uploaded %s to %s", local_path, gcs_uri)
    return gcs_uri


def _publish_notification(analysis_gcs_uri: str) -> None:
    """Publishes a message to the insight-requests topic with the analysis URI."""
    _, publisher = _init_clients()
    topic_path = publisher.topic_path(PROJECT_ID, INSIGHT_TOPIC_ID)
    payload = {"analysis_path": analysis_gcs_uri}
    future = publisher.publish(topic_path, json.dumps(payload).encode("utf-8"))
    future.result(timeout=30)
    LOGGER.info("Published insight request: %s", payload)


def _parse_pubsub_push(req_json: Dict[str, Any]) -> Dict[str, Any]:
    """Decodes a Pub/Sub push message and returns the inner JSON payload."""
    if "message" not in req_json or "data" not in req_json["message"]:
        raise ValueError("Invalid Pub/Sub push payload: missing 'message.data'")
    encoded_data = req_json["message"]["data"]
    decoded_bytes = base64.b64decode(encoded_data)
    return json.loads(decoded_bytes)


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request() -> Response:  # pylint: disable=too-many-return-statements
    try:
        req_json = request.get_json(force=True, silent=False)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Failed to parse request JSON: %s", exc)
        return Response("Bad Request", status=400)

    try:
        payload = _parse_pubsub_push(req_json)
        csv_uri = payload["csv_path"]
        pit_uri = payload["pit_json_path"]
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Invalid Pub/Sub payload: %s", exc)
        return Response("Bad Request", status=400)

    # Work directory in ephemeral /tmp
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        local_csv = tmp_path / pathlib.Path(csv_uri).name
        local_pit = tmp_path / pathlib.Path(pit_uri).name

        try:
            # Download inputs
            _download_blob(csv_uri, local_csv)
            _download_blob(pit_uri, local_pit)

            # Run analysis
            analyzer = IMSADataAnalyzer(str(local_csv), str(local_pit))
            results: Dict[str, Any] = analyzer.run_all_analyses()

            # Determine output name (basename_results_enhanced.json)
            basename = pathlib.Path(local_csv).stem  # e.g., 2025_mido_race
            output_filename = f"{basename}_results_enhanced.json"
            local_out = tmp_path / output_filename
            with local_out.open("w", encoding="utf-8") as fp:
                json.dump(results, fp)

            # Upload results
            analysis_gcs_uri = _upload_file(local_out, ANALYZED_DATA_BUCKET, output_filename)

            # Notify downstream agents
            _publish_notification(analysis_gcs_uri)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return Response("Internal Server Error", status=500)

    # Success – Cloud Run prefers 204 No Content for Pub/Sub push acknowledgments
    return Response(status=204)
