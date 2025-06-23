import os
import json
import base64
import logging
from typing import Dict, Any

from flask import Flask, request, jsonify
from google.cloud import storage

from agents.common import ai_helpers

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
    payload = request.get_json(force=True)
    insights_path = payload['insights_path']
    historical_path = payload.get('historical_path') # May be null

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

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))