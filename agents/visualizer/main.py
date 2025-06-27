"""Visualizer Cloud Run service for Project Apex.

Downloads analysis & insights JSON, generates a set of predefined charts, and
uploads them to `gs://$ANALYZED_DATA_BUCKET/<basename>/visuals/`.
"""
from __future__ import annotations


import json
import logging
import os
import pathlib
import tempfile
import os
from typing import Any, Dict, List, Tuple
import base64

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# AI helpers
from agents.common import ai_helpers
from agents.common.request_utils import parse_request_payload, validate_required_fields
from flask import Flask, request, jsonify
from google.cloud import storage

# Matplotlib/Seaborn style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
TEAM_CAR_NUMBER = os.getenv("TEAM_CAR_NUMBER")  # optional override
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("visualizer")

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


def _time_to_seconds(time_str: str | None) -> float | None:
    if time_str is None:
        return None
    try:
        if ":" in time_str:
            m, s = time_str.split(":", 1)
            return float(m) * 60 + float(s)
        return float(time_str)
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_pit_stationary_times(analysis: Dict[str, Any], output: pathlib.Path) -> None:
    records = [
        (entry.get("car_number"), _time_to_seconds(entry.get("avg_pit_stationary_time")))
        for entry in analysis.get("enhanced_strategy_analysis", [])
        if _time_to_seconds(entry.get("avg_pit_stationary_time")) is not None
    ]
    if not records:
        LOGGER.warning("No stationary time data available.")
        return
    df = pd.DataFrame(records, columns=["car", "stationary_sec"]).sort_values("stationary_sec")
    plt.figure(figsize=(6, max(3, len(df) * 0.25)))
    sns.barplot(data=df, y="car", x="stationary_sec", palette="viridis")
    plt.xlabel("Average Stationary Time (s)")
    plt.ylabel("Car #")
    plt.title("Average Pit Stop Stationary Time by Car")
    for idx, val in enumerate(df["stationary_sec"]):
        plt.text(val + 0.05, idx, f"{val:.1f}s", va="center")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_driver_consistency(analysis: Dict[str, Any], output: pathlib.Path) -> None:
    records = [
        (entry.get("car_number"), entry.get("race_pace_consistency_stdev"))
        for entry in analysis.get("enhanced_strategy_analysis", [])
        if entry.get("race_pace_consistency_stdev") is not None
    ]
    if not records:
        LOGGER.warning("No consistency data available.")
        return
    df = pd.DataFrame(records, columns=["car", "stdev"]).sort_values("stdev")
    plt.figure(figsize=(6, max(3, len(df) * 0.25)))
    sns.barplot(data=df, y="car", x="stdev", palette="magma")
    plt.xlabel("Lap Time StDev (s)")
    plt.ylabel("Car #")
    plt.title("Race Pace Consistency (StDev of Clean Laps)")
    for idx, val in enumerate(df["stdev"]):
        plt.text(val + 0.01, idx, f"{val:.3f}s", va="center")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def plot_stint_pace_falloff(analysis: Dict[str, Any], car_number: str, output: pathlib.Path) -> None:
    target_car = next((c for c in analysis.get("race_strategy_by_car", []) if c.get("car_number") == car_number), None)
    if not target_car:
        LOGGER.warning("Car %s not found for stint pace plotting", car_number)
        return
    stints = target_car.get("stints", [])
    plt.figure(figsize=(8, 5))
    for stint in stints:
        laps = stint.get("laps", [])
        if len(laps) <= 10:
            continue
        lap_nums = [lap.get("lap_in_stint") for lap in laps]
        times = [lap.get("LAP_TIME_FUEL_CORRECTED_SEC") for lap in laps]
        if None in times:
            continue
        plt.plot(lap_nums, times, label=f"Stint {stint.get('stint_number')}")
    # Degradation polynomial curve
    model = target_car.get("tire_degradation_model", {})
    if model:
        a = model.get("deg_coeff_a", 0)
        b = model.get("deg_coeff_b", 0)
        c = model.get("deg_coeff_c", 0)
        x_vals = np.linspace(0, max(max(lap.get("lap_in_stint") for stint in stints for lap in stint.get("laps", [])), 1), 100)
        y_vals = a * x_vals**2 + b * x_vals + c
        plt.plot(x_vals, y_vals, linestyle="--", color="black", label="Deg. Model")
    plt.gca().invert_yaxis()  # faster = downwards
    plt.xlabel("Lap in Stint")
    plt.ylabel("Fuel-Corrected Lap Time (s)")
    plt.title(f"Car #{car_number} - Fuel-Corrected Pace & Tire Model by Stint")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output)
    plt.close()

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _generate_caption(plot_path: pathlib.Path, insights: Dict[str, List[Dict[str, Any]]]) -> str | None:
    """Generate a caption via Gemini for a given plot image."""
    if not USE_AI_ENHANCED:
        return None
    all_insights = [insight for insight_list in insights.values() for insight in insight_list]
    prompt = (
        "You are a data visualization expert. Write a one-sentence caption (max 25 words) for the chart saved as '" + plot_path.name + "'. "
        "Base your description on the following race insights JSON for context.\n\nInsights:\n" + json.dumps(all_insights[:10], indent=2) + "\n\nCaption:"
    )
    try:
        return ai_helpers.summarize(
            prompt, temperature=0.6, max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 25000))
        )
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Caption generation failed: %s", exc)
        return None


def generate_all_visuals(analysis: Dict[str, Any], insights: Dict[str, List[Dict[str, Any]]], dest_dir: pathlib.Path) -> List[tuple[pathlib.Path, str | None]]:
    outputs: List[Tuple[pathlib.Path, str | None]] = []

    paths = [dest_dir / "pit_stationary_times.png", dest_dir / "driver_consistency.png"]
    plot_pit_stationary_times(analysis, paths[0])
    plot_driver_consistency(analysis, paths[1])

    car_num = TEAM_CAR_NUMBER or analysis.get("race_strategy_by_car", [{}])[0].get("car_number", "")
    if car_num:
        stint_path = dest_dir / f"stint_pace_car_{car_num}.png"
        plot_stint_pace_falloff(analysis, car_num, stint_path)
        paths.append(stint_path)

    for p in paths:
        caption = _generate_caption(p, insights)
        outputs.append((p, caption))
    return outputs

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


# ---------------------------------------------------------------------------
# Tool Endpoints for Individual Plot Generation
# ---------------------------------------------------------------------------

@app.route("/tools/capabilities", methods=["GET"])
def get_tool_capabilities():
    """Return information about available plotting tools."""
    capabilities = {
        "tools": [
            {
                "name": "pit_times",
                "endpoint": "/plot/pit_times",
                "description": "Generate pit stop stationary times comparison chart",
                "required_params": ["analysis_path"],
                "optional_params": []
            },
            {
                "name": "consistency",
                "endpoint": "/plot/consistency", 
                "description": "Generate driver consistency analysis chart",
                "required_params": ["analysis_path"],
                "optional_params": []
            },
            {
                "name": "stint_falloff",
                "endpoint": "/plot/stint_falloff",
                "description": "Generate stint pace falloff chart for a specific car",
                "required_params": ["analysis_path", "car_number"],
                "optional_params": []
            }
        ]
    }
    return jsonify(capabilities), 200


@app.route("/plot/pit_times", methods=["POST"])
def plot_pit_times_tool():
    """Generate pit stop stationary times chart."""
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["analysis_path"])
        
        analysis_path = payload["analysis_path"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            local_analysis = tmp / "analysis.json"
            
            # Download analysis data
            _gcs_download(analysis_path, local_analysis)
            analysis_data = json.loads(local_analysis.read_text())
            
            # Generate plot
            plot_file = tmp / "pit_stationary_times.png"
            plot_pit_stationary_times(analysis_data, plot_file)
            
            # Upload to GCS
            run_id = analysis_path.split('/')[3]
            dest_blob = f"{run_id}/visuals/pit_stationary_times.png"
            gcs_path = _gcs_upload(plot_file, dest_blob)
            
            return jsonify({
                "image_gcs_path": gcs_path,
                "chart_type": "pit_times"
            }), 200
            
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        LOGGER.exception("Pit times plotting failed: %s", exc)
        return jsonify({"error": "plotting_failed"}), 500


@app.route("/plot/consistency", methods=["POST"])
def plot_consistency_tool():
    """Generate driver consistency chart."""
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["analysis_path"])
        
        analysis_path = payload["analysis_path"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            local_analysis = tmp / "analysis.json"
            
            # Download analysis data
            _gcs_download(analysis_path, local_analysis)
            analysis_data = json.loads(local_analysis.read_text())
            
            # Generate plot
            plot_file = tmp / "driver_consistency.png"
            plot_driver_consistency(analysis_data, plot_file)
            
            # Upload to GCS
            run_id = analysis_path.split('/')[3]
            dest_blob = f"{run_id}/visuals/driver_consistency.png"
            gcs_path = _gcs_upload(plot_file, dest_blob)
            
            return jsonify({
                "image_gcs_path": gcs_path,
                "chart_type": "consistency"
            }), 200
            
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        LOGGER.exception("Consistency plotting failed: %s", exc)
        return jsonify({"error": "plotting_failed"}), 500


@app.route("/plot/stint_falloff", methods=["POST"])
def plot_stint_falloff_tool():
    """Generate stint pace falloff chart for a specific car."""
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["analysis_path", "car_number"])
        
        analysis_path = payload["analysis_path"]
        car_number = payload["car_number"]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            local_analysis = tmp / "analysis.json"
            
            # Download analysis data
            _gcs_download(analysis_path, local_analysis)
            analysis_data = json.loads(local_analysis.read_text())
            
            # Generate plot
            plot_file = tmp / f"stint_pace_car_{car_number}.png"
            plot_stint_pace_falloff(analysis_data, str(car_number), plot_file)
            
            # Upload to GCS
            run_id = analysis_path.split('/')[3]
            dest_blob = f"{run_id}/visuals/stint_pace_car_{car_number}.png"
            gcs_path = _gcs_upload(plot_file, dest_blob)
            
            return jsonify({
                "image_gcs_path": gcs_path,
                "chart_type": "stint_falloff",
                "car_number": car_number
            }), 200
            
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as exc:
        LOGGER.exception("Stint falloff plotting failed: %s", exc)
        return jsonify({"error": "plotting_failed"}), 500


# ---------------------------------------------------------------------------
# Legacy Endpoint (for backward compatibility)
# ---------------------------------------------------------------------------

@app.route("/", methods=["POST"])
def handle_request():
    """Legacy endpoint for backward compatibility. Use individual /plot/* endpoints instead."""
    LOGGER.warning("Using deprecated monolithic visualization endpoint. Consider migrating to individual /plot/* endpoints.")
    
    try:
        payload = parse_request_payload(request)
        validate_required_fields(payload, ["analysis_path", "insights_path"])
        
        analysis_uri = payload["analysis_path"]
        insights_uri = payload["insights_path"]
        comprehensive_analysis_uri = payload.get("comprehensive_analysis_path")  # Optional comprehensive analysis
        
    except ValueError as e:
        LOGGER.error(f"Request validation failed: {e}")
        return jsonify({"error": str(e)}), 400
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

            # Use comprehensive analysis for richer visualizations if available
            if comprehensive_analysis_uri:
                LOGGER.info("Using comprehensive analysis for enhanced visualizations")
                local_comprehensive = tmp / "comprehensive_analysis.json"
                try:
                    _gcs_download(comprehensive_analysis_uri, local_comprehensive)
                    comprehensive_data = json.loads(local_comprehensive.read_text())
                    # Merge comprehensive data to provide richer visualization context
                    analysis_data.update(comprehensive_data)
                    LOGGER.info("Successfully integrated comprehensive analysis into visualizations")
                except Exception as e:
                    LOGGER.warning("Could not load comprehensive analysis, using standard analysis: %s", e)

            plot_info = generate_all_visuals(analysis_data, insights_data, tmp)

            # CORRECTED: Use the run_id from the GCS path directly.
            run_id = analysis_uri.split('/')[3]
            uploaded = []
            captions: Dict[str, str] = {}
            for p, cap in plot_info:
                dest_blob = f"{run_id}/visuals/{p.name}"
                uploaded.append(_gcs_upload(p, dest_blob))
                if cap:
                    captions[p.name] = cap
            # upload captions json if any
            if captions:
                cap_file = tmp / "captions.json"
                json.dump(captions, cap_file.open("w", encoding="utf-8"))
                _gcs_upload(cap_file, f"{run_id}/visuals/captions.json")
            LOGGER.info("Uploaded visuals: %s", uploaded)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return jsonify({"error": "internal_error"}), 500
    return jsonify({
        "visuals_prefix": f"gs://{ANALYZED_DATA_BUCKET}/{run_id}/visuals/",
        "warning": "This endpoint is deprecated. Use individual /plot/* endpoints for better control."
    }), 200

@app.route("/health")
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "healthy", "service": "visualizer"}), 200


if __name__ == "__main__":
    # Register with Tool Registry at startup
    try:
        from agents.common.tool_caller import register_agent_with_registry
        port = int(os.environ.get("PORT", 8080))
        base_url = os.getenv("VISUALIZER_URL", f"http://localhost:{port}")
        register_agent_with_registry("visualizer", base_url)
    except Exception as e:
        LOGGER.warning(f"Failed to register with Tool Registry: {e}")
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
