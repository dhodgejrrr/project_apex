"""Historian Cloud Run service for Project Apex.

Receives Pub/Sub push messages containing the GCS path to a current
_results_enhanced.json analysis file, fetches the prior year's analysis for the
same track/session from BigQuery, generates year-over-year comparison insights,
writes them to GCS, and completes (no downstream Pub/Sub).
"""
from __future__ import annotations


import json
import logging
import os
import pathlib
import re

# AI helper utils
from agents.common import ai_helpers
import tempfile
from typing import Any, Dict, List, Tuple

from flask import Flask, request, jsonify
from google.cloud import bigquery, storage

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
BQ_DATASET = os.getenv("BQ_DATASET", "imsa_history")
BQ_TABLE = os.getenv("BQ_TABLE", "race_analyses")
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("historian")

# ---------------------------------------------------------------------------
# Google Cloud clients (lazy)
# ---------------------------------------------------------------------------
_storage_client: storage.Client | None = None
_bq_client: bigquery.Client | None = None


def _init_clients() -> Tuple[storage.Client, bigquery.Client]:
    global _storage_client, _bq_client  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    if _bq_client is None:
        _bq_client = bigquery.Client()
    return _storage_client, _bq_client

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _gcs_download(gcs_uri: str, dest: pathlib.Path) -> None:
    storage_client, _ = _init_clients()
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    storage_client.bucket(bucket_name).blob(blob_name).download_to_filename(dest)


def _gcs_upload(local_path: pathlib.Path, bucket: str, blob_name: str) -> str:
    storage_client, _ = _init_clients()
    storage_client.bucket(bucket).blob(blob_name).upload_from_filename(local_path)
    return f"gs://{bucket}/{blob_name}"


def _parse_filename(filename: str) -> Tuple[int, str, str]:
    """Extracts (year, track, session) from filenames like 2025_mido_race*."""
    match = re.match(r"(?P<year>\d{4})_(?P<track>[a-zA-Z]+)_(?P<session>[a-zA-Z]+)", filename)
    if not match:
        raise ValueError(f"Unrecognised filename format: {filename}")
    return int(match.group("year")), match.group("track"), match.group("session")


def _time_str_to_seconds(time_str: str | None) -> float | None:
    if time_str is None:
        return None
    try:
        if ":" in time_str:
            mins, rest = time_str.split(":", 1)
            return float(mins) * 60 + float(rest)
        return float(time_str)
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_analyses(current: Dict[str, Any], historical: Dict[str, Any]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []

    # Manufacturer fastest-lap pace delta
    curr_fast = {item["manufacturer"]: _time_str_to_seconds(item.get("fastest_lap", {}).get("time"))
                 for item in current.get("fastest_by_manufacturer", [])}
    hist_fast = {item["manufacturer"]: _time_str_to_seconds(item.get("fastest_lap", {}).get("time"))
                 for item in historical.get("fastest_by_manufacturer", [])}
    for manuf, curr_time in curr_fast.items():
        hist_time = hist_fast.get(manuf)
        if curr_time is None or hist_time is None:
            continue
        delta = curr_time - hist_time
        faster_slower = "faster" if delta < 0 else "slower"
        insights.append({
            "category": "Historical Comparison",
            "type": "YoY Manufacturer Pace",
            "manufacturer": manuf,
            "details": f"{manuf} is {abs(delta):.2f}s {faster_slower} than last year."
        })

    # Tire degradation coefficient comparison
    def _coeff_map(data: Dict[str, Any]) -> Dict[str, float]:
        mapping: Dict[str, float] = {}
        for entry in data.get("enhanced_strategy_analysis", []):
            coeff = entry.get("tire_degradation_model", {}).get("deg_coeff_a")
            if coeff is not None:
                mapping[entry.get("manufacturer")] = coeff
        return mapping

    curr_coeff = _coeff_map(current)
    hist_coeff = _coeff_map(historical)
    for manuf, curr_val in curr_coeff.items():
        hist_val = hist_coeff.get(manuf)
        if hist_val is None:
            continue
        # percentage change relative to historical (positive => increase)
        if hist_val == 0:
            continue  # avoid div-by-zero
        pct_change = (curr_val - hist_val) / abs(hist_val) * 100
        trend = "improved" if pct_change < 0 else "worsened"
        insights.append({
            "category": "Historical Comparison",
            "type": "YoY Tire Degradation",
            "manufacturer": manuf,
            "details": f"{manuf} tire degradation has {trend} by {abs(pct_change):.1f}% year-over-year."
        })

    return insights


def _narrative_summary(insights: List[Dict[str, Any]]) -> str | None:
    """Generate a concise narrative summary of YoY insights via Gemini."""
    if not USE_AI_ENHANCED or not insights:
        return None
    prompt = (
        "You are a racing data analyst AI. Craft a concise paragraph (<=120 words) summarizing the following year-over-year insights.\n"
        "Focus on key trends and notable changes.\n\nInsights JSON:\n" + json.dumps(insights, indent=2) + "\n\nSummary:"
    )
    try:
        return ai_helpers.summarize(prompt, temperature=0.5, max_output_tokens=128)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Narrative generation failed: %s", exc)
        return None

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
        if not analysis_uri:
            return jsonify({"error": "missing_analysis_path"}), 400
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Invalid request: %s", exc)
        return jsonify({"message": "no_content"}), 204

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        # Derive run_id from GCS URI assuming format gs://bucket/run_id/...
        try:
            run_id = analysis_uri.split("/", 3)[3].split("/", 1)[0]
        except Exception:
            run_id = "unknown_run"
        local_analysis = tmp / pathlib.Path(analysis_uri).name
        try:
            # Download current analysis
            _gcs_download(analysis_uri, local_analysis)
            with local_analysis.open("r", encoding="utf-8") as fp:
                current_data = json.load(fp)

            # Extract event info
            year, track, session = _parse_filename(local_analysis.stem)
            prev_year = year - 1
            LOGGER.info("Comparing against historical year %s", prev_year)

            # Query BigQuery
            _, bq_client = _init_clients()
            table_full = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
            job = bq_client.query(
                f"SELECT analysis_json FROM `{table_full}`\n"
                "WHERE track = @track AND year = @year AND session_type = @session",
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("track", "STRING", track),
                        bigquery.ScalarQueryParameter("year", "INT64", prev_year),
                        bigquery.ScalarQueryParameter("session", "STRING", session),
                    ]
                ),
            )
            results = list(job.result())
            if not results:
                LOGGER.warning("No historical analysis found for %s %s %s", track, prev_year, session)
                return jsonify({"message": "no_content"}), 204
            historical_data = results[0]["analysis_json"]

            insights = compare_analyses(current_data, historical_data)
            if not insights:
                LOGGER.info("No insights generated for %s", analysis_uri)
                return jsonify({"message": "no_content"}), 204

            summary_text = _narrative_summary(insights)
            output_obj: Dict[str, Any] = {"insights": insights}
            if summary_text:
                output_obj["narrative"] = summary_text

            # Write insights file
            basename = local_analysis.stem.replace("_results_enhanced", "")
            out_filename = f"{basename}_historical_insights.json"
            local_out = tmp / out_filename
            with local_out.open("w", encoding="utf-8") as fp:
                json.dump(output_obj, fp)
            historical_gcs_uri = _gcs_upload(local_out, ANALYZED_DATA_BUCKET, f"{run_id}/{out_filename}")
            return jsonify({"historical_path": historical_gcs_uri}), 200
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return jsonify({"error": "internal_error"}), 500

    return jsonify({"message": "no_content"}), 204
