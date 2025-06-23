from __future__ import annotations

import os
import pathlib
import secrets
import shutil
from typing import List

from flask import Flask, request, redirect, url_for
from markupsafe import Markup
from google.cloud import storage

app = Flask(__name__)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
RAW_BUCKET = os.getenv("RAW_DATA_BUCKET", "imsa-raw-data")
ANALYZED_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")

# Local development / offline mode configuration
LOCAL_MODE = os.getenv("LOCAL_MODE", "false").lower() == "true"
LOCAL_RAW_DIR = pathlib.Path(os.getenv("LOCAL_RAW_DIR", "local_raw"))
LOCAL_OUT_DIR = pathlib.Path(os.getenv("LOCAL_OUT_DIR", "out_local"))

_storage_client: storage.Client | None = None

def _storage() -> storage.Client:
    global _storage_client  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _allowed_file(filename: str) -> bool:
    return filename.lower().endswith((".csv", ".json"))


def _upload_to_gcs(bucket: str, local_path: pathlib.Path, dest_name: str | None = None) -> str:
    """Upload file to GCS or copy locally when LOCAL_MODE is enabled."""
    if LOCAL_MODE:
        LOCAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
        dest = LOCAL_RAW_DIR / (dest_name or local_path.name)
        shutil.copy(local_path, dest)
        return str(dest)

    blob_name = dest_name or local_path.name
    bucket_ref = _storage().bucket(bucket)
    blob = bucket_ref.blob(blob_name)
    blob.upload_from_filename(str(local_path))
    return f"gs://{bucket}/{blob_name}"


def _list_gcs(prefix: str) -> List[str]:
    bucket_ref = _storage().bucket(ANALYZED_BUCKET)
    blobs = bucket_ref.list_blobs(prefix=prefix)
    return [f"https://storage.googleapis.com/{ANALYZED_BUCKET}/{b.name}" for b in blobs]


def _list_results(run_id: str) -> List[str]:
    """Return visual asset paths for a run (local paths in LOCAL_MODE, URLs otherwise)."""
    if LOCAL_MODE:
        vis_dir = LOCAL_OUT_DIR / run_id / "visuals"
        if not vis_dir.exists():
            return []
        return [str(p.resolve()) for p in vis_dir.iterdir()]
    return _list_gcs(f"{run_id}/visuals/")


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        csv_file = request.files.get("csv")
        pit_file = request.files.get("pits")
        if not csv_file or not pit_file or not _allowed_file(csv_file.filename) or not _allowed_file(pit_file.filename):
            return "Invalid files. Upload CSV and pit JSON.", 400

        # Save uploads temporarily
        tmpdir = pathlib.Path("/tmp")
        csv_path = tmpdir / pathlib.Path(csv_file.filename).name
        pit_path = tmpdir / pathlib.Path(pit_file.filename).name
        csv_file.save(csv_path)
        pit_file.save(pit_path)

        # Derive run id based on file stem + random suffix to avoid collisions
        run_id = pathlib.Path(csv_file.filename).stem + "-" + secrets.token_hex(4)

        if LOCAL_MODE:
            # Copy inputs locally and run pipeline immediately
            _upload_to_gcs("", csv_path, dest_name=f"{run_id}.csv")
            _upload_to_gcs("", pit_path, dest_name=f"{run_id}_pits.json")

            from full_pipeline_local import run_pipeline  # lazy import
            output_dir = LOCAL_OUT_DIR / run_id
            run_pipeline(csv_path, pit_path, None, output_dir, live_ai=False)
        else:
            _upload_to_gcs(RAW_BUCKET, csv_path, dest_name=f"{run_id}.csv")
            _upload_to_gcs(RAW_BUCKET, pit_path, dest_name=f"{run_id}_pits.json")

        return redirect(url_for("status", run_id=run_id))

    return Markup(
        """
        <!doctype html>
        <title>Project Apex – Upload Run</title>
        <h1>Trigger New Analysis Run</h1>
        <form method=post enctype=multipart/form-data>
          <label>Race CSV: <input type=file name=csv required></label><br><br>
          <label>Pit JSON: <input type=file name=pits required></label><br><br>
          <button type=submit>Start Analysis</button>
        </form>
        """
    )


@app.route("/status/<run_id>")
def status(run_id: str):
    imgs = _list_results(run_id)
    if not imgs:
        return Markup(
            f"""<p>Run <strong>{run_id}</strong> is still processing. Refresh to check again.</p>"""
        )
    img_tags = "".join([f'<div><img src="{u}" style="max-width:100%"><br>{u}</div><hr>' for u in imgs if u.lower().endswith('.png')])
    pdf_links = [u for u in imgs if u.lower().endswith('.pdf')]
    pdf_tag = "".join([f'<p><a href="{u}">Download Report PDF</a></p>' for u in pdf_links])
    return Markup(
        f"""
        <h1>Results for {run_id}</h1>
        {pdf_tag}
        {img_tags}
        """
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=True)
