"""InsightHunter Cloud Run service for Project Apex.

Listens for Pub/Sub push messages containing the Cloud Storage path to an
_enhanced.json race analysis file, derives tactical insights, stores them as a
new _insights.json file, and publishes a notification to the
`visualization-requests` topic.
"""
from __future__ import annotations

import base64
import json
import logging
import os
import pathlib
import tempfile
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List

from flask import Flask, Response, request
from google.cloud import pubsub_v1, storage

# ----------------------------------------------------------------------------
# Environment configuration
# ----------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
VIS_TOPIC_ID = os.getenv("VISUALIZATION_REQUESTS_TOPIC", "visualization-requests")

# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("insight_hunter")

# ----------------------------------------------------------------------------
# Google Cloud clients (lazy)
# ----------------------------------------------------------------------------
_storage_client: storage.Client | None = None
_publisher: pubsub_v1.PublisherClient | None = None


def _init_clients() -> tuple[storage.Client, pubsub_v1.PublisherClient]:
    global _storage_client, _publisher  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _storage_client, _publisher

# ----------------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------------

def _gcs_download(gcs_uri: str, dest_path: pathlib.Path) -> None:
    storage_client, _ = _init_clients()
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    bucket.blob(blob_name).download_to_filename(dest_path)


def _gcs_upload(local_path: pathlib.Path, bucket: str, blob_name: str) -> str:
    storage_client, _ = _init_clients()
    blob = storage_client.bucket(bucket).blob(blob_name)
    blob.upload_from_filename(local_path)
    return f"gs://{bucket}/{blob_name}"


def _publish_visualization_request(analysis_uri: str, insights_uri: str) -> None:
    _, publisher = _init_clients()
    topic_path = publisher.topic_path(PROJECT_ID, VIS_TOPIC_ID)
    message = {"analysis_path": analysis_uri, "insights_path": insights_uri}
    publisher.publish(topic_path, json.dumps(message).encode("utf-8"))
    LOGGER.info("Published visualization request: %s", message)


def _parse_pubsub_push(req_json: Dict[str, Any]) -> Dict[str, Any]:
    if "message" not in req_json or "data" not in req_json["message"]:
        raise ValueError("Invalid Pub/Sub push payload")
    decoded = base64.b64decode(req_json["message"]["data"]).decode("utf-8")
    return json.loads(decoded)


# ----------------------------------------------------------------------------
# Domain-specific helper functions
# ----------------------------------------------------------------------------

def _time_str_to_seconds(time_str: str | None) -> float | None:
    """Converts time formats like '1:23.456' or '23.456' into float seconds."""
    if time_str is None:
        return None
    try:
        if ":" in time_str:
            mins, rest = time_str.split(":", 1)
            return float(mins) * 60 + float(rest)
        return float(time_str)
    except ValueError:
        return None

# ----------------------------------------------------------------------------
# Insight algorithms
# ----------------------------------------------------------------------------

def find_pit_stop_insights(strategy_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []

    for car in strategy_data:
        car_number = car.get("car_number")
        pit_details = car.get("pit_stop_details", [])
        avg_pit_lane_time_s: List[float] = []
        for stop in pit_details:
            secs = _time_str_to_seconds(stop.get("total_pit_lane_time"))
            if secs is not None:
                avg_pit_lane_time_s.append(secs)
        if not avg_pit_lane_time_s:
            continue
        avg_time = mean(avg_pit_lane_time_s)
        # Logic 1: Pit Delta Outlier
        for idx, stop in enumerate(pit_details, start=1):
            delta_s = _time_str_to_seconds(stop.get("total_pit_lane_time"))
            if delta_s is None:
                continue
            if delta_s > 1.5 * avg_time:
                insights.append({
                    "category": "Pit Stop Intelligence",
                    "type": "Pit Delta Outlier",
                    "car_number": car_number,
                    "details": f"Stop #{idx} was {delta_s - avg_time:.1f}s slower than team average."
                })
        # Logic 2: Driver Change Cost
        stationary_changes: List[float] = []
        stationary_no_change: List[float] = []
        for stop in pit_details:
            stat_time = _time_str_to_seconds(stop.get("stationary_time"))
            if stat_time is None:
                continue
            if stop.get("driver_change"):
                stationary_changes.append(stat_time)
            else:
                stationary_no_change.append(stat_time)
        if stationary_changes and stationary_no_change:
            diff = mean(stationary_changes) - mean(stationary_no_change)
            insights.append({
                "category": "Pit Stop Intelligence",
                "type": "Driver Change Cost",
                "car_number": car_number,
                "details": f"Driver changes cost an average of {diff:.1f}s more stationary time."
            })

    # Logic 3: Stationary Time Champion (needs enhanced_strategy_analysis section)
    # This will be added from outer function because data lives elsewhere.
    return insights


def _stationary_time_champion(enhanced_strategy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    min_time = float("inf")
    champion_car = None
    for entry in enhanced_strategy:
        avg_stat = _time_str_to_seconds(entry.get("avg_pit_stationary_time"))
        if avg_stat is not None and avg_stat < min_time:
            min_time = avg_stat
            champion_car = entry.get("car_number")
    if champion_car is None:
        return []
    return [{
        "category": "Pit Stop Intelligence",
        "type": "Stationary Time Champion",
        "car_number": champion_car,
        "details": f"Fastest crew on pit road with an average stationary time of {min_time:.1f}s."
    }]


def find_performance_insights(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []

    # Consistency King
    enhanced = data.get("enhanced_strategy_analysis", [])
    min_stdev = float("inf")
    king_entry = None
    for entry in enhanced:
        stdev = entry.get("race_pace_consistency_stdev")
        if stdev is not None and stdev < min_stdev:
            min_stdev = stdev
            king_entry = entry
    if king_entry is not None:
        insights.append({
            "category": "Performance",
            "type": "Consistency King",
            "car_number": king_entry.get("car_number"),
            "driver_name": king_entry.get("primary_driver"),
            "details": f"Most consistent driver with a standard deviation of {min_stdev:.3f}s on clean, fuel-corrected laps."
        })

    # Leaving Time on the Table – Optimal Lap Delta
    fastest = data.get("fastest_by_car_number", [])
    for car in fastest:
        fastest_lap_s = _time_str_to_seconds(car.get("fastest_lap", {}).get("time"))
        optimal_s = _time_str_to_seconds(car.get("optimal_lap_time"))
        if fastest_lap_s is None or optimal_s is None:
            continue
        delta = optimal_s - fastest_lap_s
        if delta > 0.3:
            insights.append({
                "category": "Performance",
                "type": "Optimal Lap Delta",
                "car_number": car.get("car_number"),
                "details": f"Car #{car.get('car_number')} has a {delta:.1f}s gap between its fastest and optimal lap, indicating inconsistent sector performance."
            })
    return insights


def find_degradation_insights(enhanced_strategy: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    coeffs: defaultdict[str, List[float]] = defaultdict(list)
    for entry in enhanced_strategy:
        model = entry.get("tire_degradation_model", {})
        coeff_a = model.get("deg_coeff_a")
        if coeff_a is not None:
            coeffs[entry.get("manufacturer")].append(coeff_a)
    if not coeffs:
        return []
    # manufacturer with highest average positive coefficient
    worst_manufacturer, worst_value = max(
        ((m, mean(vals)) for m, vals in coeffs.items()),
        key=lambda x: x[1],
    )
    return [{
        "category": "Tire Management",
        "type": "Tire-Killer Alert",
        "manufacturer": worst_manufacturer,
        "details": f"{worst_manufacturer} shows the highest tire degradation rate (avg coeff {worst_value:.4f})."
    }]


def find_driver_delta_insights(delta_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not delta_data:
        return []
    worst_entry = max(delta_data, key=lambda d: d.get("average_lap_time_delta_for_car", 0))
    gap = worst_entry.get("average_lap_time_delta_for_car", 0)
    return [{
        "category": "Driver Performance",
        "type": "Largest Teammate Pace Gap",
        "car_number": worst_entry.get("car_number"),
        "details": f"Car #{worst_entry.get('car_number')} has the largest pace delta ({gap:.2f}s) between its drivers."
    }]


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def derive_insights(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    insights = []
    strategy_data = data.get("race_strategy_by_car", [])
    enhanced_strategy = data.get("enhanced_strategy_analysis", [])
    delta_data = data.get("driver_deltas_by_car", [])

    insights.extend(find_pit_stop_insights(strategy_data))
    insights.extend(_stationary_time_champion(enhanced_strategy))
    insights.extend(find_performance_insights(data))
    insights.extend(find_degradation_insights(enhanced_strategy))
    insights.extend(find_driver_delta_insights(delta_data))
    return insights


# ----------------------------------------------------------------------------
# Flask application
# ----------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request() -> Response:
    try:
        req_json = request.get_json(force=True, silent=False)
        payload = _parse_pubsub_push(req_json)
        analysis_uri = payload["analysis_path"]
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Invalid request: %s", exc)
        return Response("Bad Request", status=400)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        local_analysis_path = tmp / pathlib.Path(analysis_uri).name
        try:
            # Download analysis file
            _gcs_download(analysis_uri, local_analysis_path)
            with local_analysis_path.open("r", encoding="utf-8") as fp:
                analysis_data = json.load(fp)

            insights = derive_insights(analysis_data)

            # Write insights to file
            basename = pathlib.Path(local_analysis_path).stem.replace("_results_enhanced", "")
            insights_filename = f"{basename}_insights.json"
            local_insights_path = tmp / insights_filename
            with local_insights_path.open("w", encoding="utf-8") as fp:
                json.dump(insights, fp)

            insights_uri = _gcs_upload(local_insights_path, ANALYZED_DATA_BUCKET, insights_filename)

            # Publish visualisation request
            _publish_visualization_request(analysis_uri, insights_uri)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return Response("Internal Server Error", status=500)

    return Response(status=204)
