"""CoreAnalyzer Toolbox Service for Project Apex.

Provides HTTP endpoints for running specific race data analyses:
- Full analysis (original endpoint)
- Pace analysis only
- Strategy analysis only
- Comparison of multiple analyses

Input/Output via Google Cloud Storage URIs.
"""
from __future__ import annotations


import json
import logging
import os
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass, asdict
from enum import Enum

from flask import Flask, request, jsonify, Response
from google.cloud import storage

# Third-party analysis module provided separately in this repository.
from imsa_analyzer import IMSADataAnalyzer  # type: ignore
from agents.common.request_utils import parse_request_payload, validate_required_fields

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class AnalysisType(str, Enum):
    FULL = "full"
    PACE = "pace"
    STRATEGY = "strategy"

@dataclass
class AnalysisResult:
    analysis_type: AnalysisType
    gcs_uri: str
    metrics: Dict[str, Any]

class AnalysisRequest(TypedDict):
    run_id: str
    csv_path: str
    pit_json_path: str
    analysis_type: Optional[str]  # For single analysis requests

class ComparisonRequest(TypedDict):
    run_id: str
    analysis_paths: List[str]  # List of analysis URIs to compare
    comparison_metrics: List[str]  # Which metrics to include in comparison

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data-project-apex-v1")


# ---------------------------------------------------------------------------
# Logging setup (structured for Cloud Logging)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("core_analyzer")

# ---------------------------------------------------------------------------
# Google Cloud clients (lazily initialised)
# ---------------------------------------------------------------------------
_storage_client: storage.Client | None = None


def _get_storage_client() -> storage.Client:
    """Lazily initialises and returns a Google Cloud Storage client."""
    global _storage_client  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _download_blob(gcs_uri: str, dest_path: pathlib.Path) -> None:
    """Downloads a GCS object to a local file path."""
    storage_client = _get_storage_client()
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
    storage_client = _get_storage_client()
    bucket = storage_client.bucket(dest_bucket)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(str(local_path))
    gcs_uri = f"gs://{dest_bucket}/{dest_blob_name}"
    LOGGER.info("Uploaded %s to %s", local_path, gcs_uri)
    return gcs_uri









# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------

def _run_analysis(
    analyzer: IMSADataAnalyzer,
    analysis_type: AnalysisType = AnalysisType.FULL
) -> Dict[str, Any]:
    """Run the specified type of analysis."""
    if analysis_type == AnalysisType.PACE:
        return {
            "pace_analysis": analyzer.analyze_pace(),
            "metadata": {"analysis_type": "pace"}
        }
    elif analysis_type == AnalysisType.STRATEGY:
        return {
            "strategy_analysis": analyzer.analyze_strategy(),
            "metadata": {"analysis_type": "strategy"}
        }
    else:  # FULL
        return {
            **analyzer.run_all_analyses(),
            "metadata": {"analysis_type": "full"}
        }

def _compare_analyses(analysis_paths: List[str]) -> Dict[str, Any]:
    """Compare multiple analysis results."""
    comparisons = {}
    for path in analysis_paths:
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            _download_blob(path, pathlib.Path(tmp.name))
            with open(tmp.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                comparisons[path] = {
                    "summary": {
                        k: v for k, v in data.items()
                        if k in ["metadata", "race_summary"]
                    },
                    "metrics": {
                        k: v for k, v in data.items()
                        if k not in ["metadata", "race_summary"]
                    }
                }
    return {"comparisons": comparisons}

# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health")
def health_check() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route("/analyze", methods=["POST"])
def analyze() -> Response:
    """Run a full analysis."""
    return _handle_analysis_request(request, AnalysisType.FULL)

@app.route("/analyze/pace", methods=["POST"])
def analyze_pace() -> Response:
    """Run pace analysis only."""
    return _handle_analysis_request(request, AnalysisType.PACE)

@app.route("/analyze/strategy", methods=["POST"])
def analyze_strategy() -> Response:
    """Run strategy analysis only."""
    return _handle_analysis_request(request, AnalysisType.STRATEGY)

@app.route("/analyze/compare", methods=["POST"])
def compare_analyses() -> Response:
    """Compare multiple analysis results."""
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["analysis_paths"])
        
        comparison = _compare_analyses(payload["analysis_paths"])
        return jsonify(comparison), 200
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        LOGGER.exception("Comparison failed: %s", exc)
        return jsonify({"error": "comparison_failed"}), 500

def _handle_analysis_request(request, analysis_type: AnalysisType) -> Response:
    """Handle analysis requests with common logic for both HTTP and Pub/Sub formats."""
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["run_id", "csv_path", "pit_json_path"])
        
        run_id = payload["run_id"]
        csv_uri = payload["csv_path"]
        pit_uri = payload["pit_json_path"]
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        LOGGER.error(f"Request parsing failed: {e}")
        return jsonify({"error": "invalid_request"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        local_csv = tmp_path / pathlib.Path(csv_uri).name
        local_pit = tmp_path / pathlib.Path(pit_uri).name

        try:
            # Download inputs
            _download_blob(csv_uri, local_csv)
            _download_blob(pit_uri, local_pit)

            # Run specified analysis
            analyzer = IMSADataAnalyzer(str(local_csv), str(local_pit))
            results = _run_analysis(analyzer, analysis_type)

            # Save results
            output_filename = f"{run_id}_results_{analysis_type.value}.json"
            local_out = tmp_path / output_filename
            with local_out.open("w", encoding="utf-8") as fp:
                json.dump(results, fp)

            # Upload to GCS
            dest_blob_name = f"{run_id}/{output_filename}"
            gcs_uri = _upload_file(local_out, ANALYZED_DATA_BUCKET, dest_blob_name)
            
            return jsonify({
                "analysis_path": gcs_uri,
                "analysis_type": analysis_type.value
            }), 200

        except Exception as exc:
            LOGGER.exception("Analysis failed: %s", exc)
            return jsonify({"error": "analysis_failed"}), 500
