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
from google.cloud import storage, bigquery

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
RAW_DATA_BUCKET = os.getenv("RAW_DATA_BUCKET", "imsa-raw-data-project-apex-v1")
BQ_DATASET = os.getenv("BQ_DATASET", "imsa_history")
BQ_TABLE = os.getenv("BQ_TABLE", "race_analyses")

# ---------------------------------------------------------------------------
# In-Memory Analyzer Cache for Toolbox Operations
# ---------------------------------------------------------------------------
ANALYZER_CACHE: Dict[str, IMSADataAnalyzer] = {}


# ---------------------------------------------------------------------------
# Logging setup (structured for Cloud Logging)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("core_analyzer")

# ---------------------------------------------------------------------------
# Google Cloud clients (lazily initialised)
# ---------------------------------------------------------------------------
_storage_client: storage.Client | None = None
_bigquery_client: bigquery.Client | None = None


def _get_storage_client() -> storage.Client:
    """Lazily initialises and returns a Google Cloud Storage client."""
    global _storage_client  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client


def _get_bigquery_client() -> bigquery.Client:
    """Lazily initialises and returns a Google Cloud BigQuery client."""
    global _bigquery_client  # pylint: disable=global-statement
    if _bigquery_client is None:
        _bigquery_client = bigquery.Client()
    return _bigquery_client

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
# Toolbox Cache Management
# ---------------------------------------------------------------------------

def get_analyzer(run_id: str) -> IMSADataAnalyzer:
    """
    Gets an analyzer instance for the run_id, either from cache or by reconstructing
    from GCS files. This enables expensive analysis operations to be reused.
    """
    if run_id in ANALYZER_CACHE:
        LOGGER.info("Retrieved analyzer for %s from cache", run_id)
        return ANALYZER_CACHE[run_id]
    
    # Reconstruct analyzer from source files
    LOGGER.info("Reconstructing analyzer for %s from GCS files", run_id)
    
    # Assume standard file naming convention
    csv_gcs_path = f"gs://{RAW_DATA_BUCKET}/{run_id}/{run_id}.csv"
    pit_gcs_path = f"gs://{RAW_DATA_BUCKET}/{run_id}/{run_id}_pits.json"
    
    # Alternative naming patterns to try
    csv_patterns = [
        f"gs://{RAW_DATA_BUCKET}/{run_id}/{run_id}.csv",
        f"gs://{RAW_DATA_BUCKET}/{run_id}/{run_id}_race.csv"
    ]
    pit_patterns = [
        f"gs://{RAW_DATA_BUCKET}/{run_id}/{run_id}_pits.json",
        f"gs://{RAW_DATA_BUCKET}/{run_id}/{run_id}_pit.json"
    ]
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        
        # Try to find and download CSV file
        csv_found = False
        for csv_pattern in csv_patterns:
            try:
                local_csv = tmp_path / "race_data.csv"
                _download_blob(csv_pattern, local_csv)
                csv_found = True
                break
            except Exception:
                continue
        
        if not csv_found:
            raise ValueError(f"Could not find CSV file for run_id {run_id}")
        
        # Try to find and download pit file
        pit_found = False
        for pit_pattern in pit_patterns:
            try:
                local_pit = tmp_path / "pit_data.json"
                _download_blob(pit_pattern, local_pit)
                pit_found = True
                break
            except Exception:
                continue
        
        if not pit_found:
            raise ValueError(f"Could not find pit JSON file for run_id {run_id}")
        
        # Create and cache analyzer
        analyzer = IMSADataAnalyzer(str(local_csv), str(local_pit))
        ANALYZER_CACHE[run_id] = analyzer
        
        LOGGER.info("Successfully cached analyzer for %s", run_id)
        return analyzer


def _query_historical_trends(track: str, session: str, manufacturer: str, num_years: int = 5) -> List[Dict[str, Any]]:
    """
    Query BigQuery for historical performance trends at a specific track.
    Returns trend data for the specified manufacturer over the last num_years.
    """
    bq_client = _get_bigquery_client()
    
    query = f"""
    SELECT 
        EXTRACT(YEAR FROM race_date) as year,
        MIN(best_lap_time_sec) as fastest_lap_sec,
        FORMAT('%.3f', MIN(best_lap_time_sec)) as fastest_lap_formatted,
        COUNT(*) as race_count
    FROM `{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}`
    WHERE 
        LOWER(track_name) = LOWER(@track)
        AND LOWER(session_type) = LOWER(@session)
        AND LOWER(manufacturer) = LOWER(@manufacturer)
        AND race_date >= DATE_SUB(CURRENT_DATE(), INTERVAL @num_years YEAR)
    GROUP BY year
    ORDER BY year DESC
    """
    
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("track", "STRING", track),
            bigquery.ScalarQueryParameter("session", "STRING", session),
            bigquery.ScalarQueryParameter("manufacturer", "STRING", manufacturer),
            bigquery.ScalarQueryParameter("num_years", "INT64", num_years),
        ]
    )
    
    try:
        query_job = bq_client.query(query, job_config=job_config)
        results = query_job.result()
        
        trend_data = []
        for row in results:
            trend_data.append({
                "year": row.year,
                "fastest_lap_sec": row.fastest_lap_sec,
                "fastest_lap_formatted": row.fastest_lap_formatted,
                "race_count": row.race_count
            })
        
        return trend_data
    
    except Exception as e:
        LOGGER.warning("BigQuery trend analysis failed: %s", e)
        return []


def _format_seconds_to_ms_str(seconds: float) -> str:
    """Helper to format seconds into a readable time string with milliseconds."""
    if seconds is None or seconds == 0:
        return "0.000s"
    
    sign = "+" if seconds > 0 else ""
    abs_seconds = abs(seconds)
    
    return f"{sign}{abs_seconds:.3f}s"

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

            # Create analyzer and cache it for future tool calls
            analyzer = IMSADataAnalyzer(str(local_csv), str(local_pit))
            ANALYZER_CACHE[run_id] = analyzer
            LOGGER.info("Cached analyzer for run_id: %s", run_id)
            
            # Run specified analysis
            results = _run_analysis(analyzer, analysis_type)

            # Save results
            output_filename = f"{run_id}_results_{analysis_type.value}.json"
            local_out = tmp_path / output_filename
            with local_out.open("w", encoding="utf-8") as fp:
                json.dump(results, fp)

            # Upload to GCS
            dest_blob_name = f"{run_id}/{output_filename}"
            gcs_uri = _upload_file(local_out, ANALYZED_DATA_BUCKET, dest_blob_name)
            
            # For FULL analysis, also create a comprehensive analysis file using the analyzer's export method
            # This ensures the deep analysis insights are preserved and accessible to other agents
            if analysis_type == AnalysisType.FULL:
                LOGGER.info("Creating comprehensive analysis export for other agents and final output")
                comprehensive_filename = f"{run_id}_comprehensive_analysis.json"
                local_comprehensive = tmp_path / comprehensive_filename
                
                # Use the analyzer's built-in export method to ensure all analysis features are captured
                analyzer.export_to_json_file(results, str(local_comprehensive))
                
                # Upload comprehensive file to GCS for other agents to access
                comprehensive_dest_blob = f"{run_id}/{comprehensive_filename}"
                comprehensive_gcs_uri = _upload_file(local_comprehensive, ANALYZED_DATA_BUCKET, comprehensive_dest_blob)
                LOGGER.info("Uploaded comprehensive analysis to: %s", comprehensive_gcs_uri)
            
            response_data = {
                "analysis_path": gcs_uri,
                "analysis_type": analysis_type.value
            }
            
            # Include comprehensive analysis path for FULL analysis
            if analysis_type == AnalysisType.FULL:
                response_data["comprehensive_analysis_path"] = comprehensive_gcs_uri
            
            return jsonify(response_data), 200

        except Exception as exc:
            LOGGER.exception("Analysis failed: %s", exc)
            return jsonify({"error": "analysis_failed"}), 500

# ---------------------------------------------------------------------------
# Toolbox Endpoints
# ---------------------------------------------------------------------------

@app.route("/tools/capabilities", methods=["GET"])
def get_tool_capabilities() -> Response:
    """Return information about available tools in this analyzer."""
    capabilities = {
        "tools": [
            {
                "name": "driver_deltas",
                "endpoint": "/tools/driver_deltas",
                "description": "Get performance gaps between drivers in the same car",
                "required_params": ["run_id"],
                "optional_params": ["car_number"]
            },
            {
                "name": "trend_analysis", 
                "endpoint": "/tools/trend_analysis",
                "description": "Get historical performance trends for a manufacturer at a track",
                "required_params": ["track", "session", "manufacturer"],
                "optional_params": ["num_years"]
            },
            {
                "name": "stint_analysis",
                "endpoint": "/tools/stint_analysis", 
                "description": "Get detailed stint and tire degradation analysis",
                "required_params": ["run_id"],
                "optional_params": ["car_number", "stint_number"]
            },
            {
                "name": "sector_analysis",
                "endpoint": "/tools/sector_analysis",
                "description": "Get sector-by-sector performance breakdown",
                "required_params": ["run_id"],
                "optional_params": ["car_number"]
            }
        ],
        "cache_info": {
            "cached_analyses": len(ANALYZER_CACHE),
            "cache_keys": list(ANALYZER_CACHE.keys())
        }
    }
    return jsonify(capabilities), 200


@app.route("/tools/driver_deltas", methods=["POST"])
def tool_driver_deltas() -> Response:
    """Get driver performance deltas for a specific run."""
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["run_id"])
        
        run_id = payload["run_id"]
        car_number = payload.get("car_number")
        
        analyzer = get_analyzer(run_id)
        deltas = analyzer.get_driver_deltas_by_car()
        
        # Filter by car number if specified
        if car_number:
            deltas = [d for d in deltas if str(d.get("car_number")) == str(car_number)]
        
        return jsonify({
            "run_id": run_id,
            "driver_deltas": deltas,
            "filtered_by_car": car_number
        }), 200
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        LOGGER.exception("Driver deltas analysis failed: %s", exc)
        return jsonify({"error": "analysis_failed"}), 500


@app.route("/tools/trend_analysis", methods=["POST"])
def tool_trend_analysis() -> Response:
    """Get historical performance trends for a manufacturer at a track."""
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["track", "session", "manufacturer"])
        
        track = payload["track"]
        session = payload["session"]
        manufacturer = payload["manufacturer"]
        num_years = payload.get("num_years", 5)
        
        trend_data = _query_historical_trends(track, session, manufacturer, num_years)
        
        return jsonify({
            "track": track,
            "session": session,
            "manufacturer": manufacturer,
            "years_analyzed": num_years,
            "trend_data": trend_data
        }), 200
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        LOGGER.exception("Trend analysis failed: %s", exc)
        return jsonify({"error": "analysis_failed"}), 500


@app.route("/tools/stint_analysis", methods=["POST"])
def tool_stint_analysis() -> Response:
    """Get detailed stint and tire degradation analysis."""
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["run_id"])
        
        run_id = payload["run_id"]
        car_number = payload.get("car_number")
        stint_number = payload.get("stint_number")
        
        analyzer = get_analyzer(run_id)
        
        # Get full analysis to extract stint information
        full_analysis = analyzer.run_all_analyses()
        stint_data = full_analysis.get("enhanced_strategy_analysis", [])
        
        # Filter by car number if specified
        if car_number:
            stint_data = [s for s in stint_data if str(s.get("car_number")) == str(car_number)]
        
        # Extract tire degradation models
        tire_analysis = []
        for car_data in stint_data:
            if "tire_degradation_model" in car_data:
                tire_analysis.append({
                    "car_number": car_data.get("car_number"),
                    "tire_model": car_data["tire_degradation_model"],
                    "fuel_corrected_pace": car_data.get("avg_green_pace_fuel_corrected")
                })
        
        return jsonify({
            "run_id": run_id,
            "stint_analysis": stint_data,
            "tire_degradation": tire_analysis,
            "filtered_by_car": car_number,
            "filtered_by_stint": stint_number
        }), 200
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        LOGGER.exception("Stint analysis failed: %s", exc)
        return jsonify({"error": "analysis_failed"}), 500


@app.route("/tools/sector_analysis", methods=["POST"])
def tool_sector_analysis() -> Response:
    """Get sector-by-sector performance breakdown."""
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["run_id"])
        
        run_id = payload["run_id"]
        car_number = payload.get("car_number")
        
        analyzer = get_analyzer(run_id)
        
        # Get sector analysis from the full dataset
        df = analyzer.df
        
        if car_number:
            df = df[df['NUMBER'] == car_number]
        
        # Calculate sector performance statistics
        sector_stats = []
        for car_num, car_df in df.groupby('NUMBER'):
            if car_df.empty:
                continue
                
            best_s1 = car_df['S1_SEC'].min() if 'S1_SEC' in car_df.columns else None
            best_s2 = car_df['S2_SEC'].min() if 'S2_SEC' in car_df.columns else None
            best_s3 = car_df['S3_SEC'].min() if 'S3_SEC' in car_df.columns else None
            
            sector_stats.append({
                "car_number": car_num,
                "best_s1_sec": best_s1,
                "best_s2_sec": best_s2,
                "best_s3_sec": best_s3,
                "best_s1_formatted": _format_seconds_to_ms_str(best_s1) if best_s1 else None,
                "best_s2_formatted": _format_seconds_to_ms_str(best_s2) if best_s2 else None,
                "best_s3_formatted": _format_seconds_to_ms_str(best_s3) if best_s3 else None
            })
        
        return jsonify({
            "run_id": run_id,
            "sector_analysis": sector_stats,
            "filtered_by_car": car_number
        }), 200
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        LOGGER.exception("Sector analysis failed: %s", exc)
        return jsonify({"error": "analysis_failed"}), 500

if __name__ == "__main__":
    # Register with Tool Registry at startup
    try:
        from agents.common.tool_caller import register_agent_with_registry
        port = int(os.environ.get("PORT", 8080))
        base_url = os.getenv("CORE_ANALYZER_URL", f"http://localhost:{port}")
        register_agent_with_registry("core_analyzer", base_url)
    except Exception as e:
        LOGGER.warning(f"Failed to register with Tool Registry: {e}")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
