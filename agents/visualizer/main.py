"""Visualizer Cloud Run service for Project Apex.

Downloads analysis & insights JSON, generates a set of predefined charts, and
uploads them to `gs://$ANALYZED_DATA_BUCKET/<basename>/visuals/`.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import tempfile
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# AI helpers
from agents.common import ai_helpers
from flask import Flask, Response, request
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

            plot_info = generate_all_visuals(analysis_data, insights_data, tmp)

            # Upload all PNGs in tmp
            basename = local_analysis.stem.replace("_results_enhanced", "")
            uploaded = []
            captions: Dict[str, str] = {}
            for p, cap in plot_info:
                dest_blob = f"{basename}/visuals/{p.name}"
                uploaded.append(_gcs_upload(p, dest_blob))
                if cap:
                    captions[p.name] = cap
            # upload captions json if any
            if captions:
                cap_file = tmp / "captions.json"
                json.dump(captions, cap_file.open("w", encoding="utf-8"))
                _gcs_upload(cap_file, f"{basename}/visuals/captions.json")
            LOGGER.info("Uploaded visuals: %s", uploaded)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return Response("Internal Server Error", status=500)
    return Response(status=204)
