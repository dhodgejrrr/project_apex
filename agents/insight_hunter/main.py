"""
InsightHunter Cloud Run service for Project Apex.

Listens for Pub/Sub push messages containing the Cloud Storage path to an
_enhanced.json race analysis file, derives tactical insights, stores them as a
new _insights.json file, and publishes a notification to the
`visualization-requests` topic.
"""
from __future__ import annotations

import base64
import json
import os
import logging
import os
import pathlib
import tempfile
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, List, Optional

# AI helper utils - Assuming this is a local utility
from agents.common import ai_helpers

from flask import Flask, Response, request
from google.cloud import pubsub_v1, storage

# ----------------------------------------------------------------------------
# Environment configuration & Logging
# ----------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
VIS_TOPIC_ID = os.getenv("VISUALIZATION_REQUESTS_TOPIC", "visualization-requests")
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("insight_hunter")

# ----------------------------------------------------------------------------
# Cloud Clients & Helpers
# ----------------------------------------------------------------------------
_storage_client: Optional[storage.Client] = None
_publisher: Optional[pubsub_v1.PublisherClient] = None


def _init_clients() -> tuple[storage.Client, pubsub_v1.PublisherClient]:
    global _storage_client, _publisher
    if _storage_client is None:
        _storage_client = storage.Client()
    if _publisher is None:
        _publisher = pubsub_v1.PublisherClient()
    return _storage_client, _publisher

def _gcs_download(gcs_uri: str, dest_path: pathlib.Path) -> None:
    storage_client, _ = _init_clients()
    if not gcs_uri.startswith("gs://"): raise ValueError(f"Invalid GCS URI: {gcs_uri}")
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
# Domain-specific Helper Functions
# ----------------------------------------------------------------------------

def _time_str_to_seconds(time_str: str | None) -> float | None:
    if not isinstance(time_str, str): return None
    try:
        sign = 1
        clean_str = time_str
        if time_str.startswith("-"):
            sign = -1
            clean_str = time_str[1:]
        
        if ":" in clean_str:
            parts = clean_str.split(":")
            if len(parts) == 2: return sign * (float(parts[0]) * 60 + float(parts[1]))
            elif len(parts) == 3: return sign * (float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2]))
        return sign * float(clean_str)
    except (ValueError, TypeError):
        return None

def _format_seconds(seconds: Optional[float], precision: int = 3, unit: str = "") -> Optional[str]:
    if seconds is None: return None
    return f"{seconds:+.{precision}f}{unit}"

# ----------------------------------------------------------------------------
# Insight Ranking Algorithms
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
    
def _rank_by_metric(
    data_list: List[Dict], group_by_key: str, metric_path: List[str], higher_is_better: bool, value_is_time: bool = True
) -> List[Dict]:
    """Generic function to rank entities (cars or manufacturers) by a given metric."""
    grouped_metrics = defaultdict(list)
    for item in data_list:
        group_name = item.get(group_by_key)
        value = item
        try:
            for key in metric_path: value = value[key]
        except (KeyError, TypeError): continue
        
        if isinstance(value, str): value = _time_str_to_seconds(value)
        
        if group_name and value is not None:
            grouped_metrics[group_name].append(value)

    if not grouped_metrics: return []

    avg_metrics = {name: mean(vals) for name, vals in grouped_metrics.items() if vals}
    if not avg_metrics: return []
    
    field_average = mean(avg_metrics.values())
    
    ranked_list = [{
        "name": name, "value": avg_val, "delta_from_avg": avg_val - field_average
    } for name, avg_val in avg_metrics.items()]

    ranked_list.sort(key=lambda x: x["value"], reverse=higher_is_better)

    return [{
        "rank": i + 1,
        group_by_key: item["name"],
        "average_value": _format_seconds(item["value"]) if value_is_time else f"{item['value']:.4f}",
        "delta_from_field_avg": _format_seconds(item["delta_from_avg"]) if value_is_time else f"{item['delta_from_avg']:+.4f}"
    } for i, item in enumerate(ranked_list)]

def _rank_cars_by_untapped_potential(fastest_by_car: List[Dict]) -> List[Dict]:
    """Ranks cars by the largest gap between their actual fastest lap and theoretical optimal lap."""
    gaps = []
    for car in fastest_by_car:
        fastest_s = _time_str_to_seconds(car.get("fastest_lap", {}).get("time"))
        optimal_s = _time_str_to_seconds(car.get("optimal_lap_time"))
        if fastest_s and optimal_s:
            gap = fastest_s - optimal_s
            if gap > 0.1:  # Only include meaningful gaps
                gaps.append({
                    "car_number": car.get("car_number"),
                    "driver_name": car.get("fastest_lap", {}).get("driver_name"),
                    "gap_seconds": gap,
                    "team": car.get("team", "N/A") # Assume team might be available
                })
    
    gaps.sort(key=lambda x: x["gap_seconds"], reverse=True)
    return [{
        "rank": i + 1,
        "car_number": item["car_number"],
        "driver_name": item["driver_name"],
        "time_left_on_track": _format_seconds(item["gap_seconds"], unit="s")
    } for i, item in enumerate(gaps)]

def _rank_drivers_by_traffic_management(traffic_data: List[Dict]) -> List[Dict]:
    """Ranks drivers by their effectiveness in traffic (lowest time lost)."""
    if not traffic_data: return []
    
    # The data is already ranked in the source file, so we just need to add context.
    all_lost_times = [_time_str_to_seconds(d.get("avg_time_lost_total_sec")) for d in traffic_data]
    valid_lost_times = [t for t in all_lost_times if t is not None]
    if not valid_lost_times: return []

    field_avg_lost_time = mean(valid_lost_times)
    
    contextual_list = []
    for item in traffic_data:
        time_lost = _time_str_to_seconds(item.get("avg_time_lost_total_sec"))
        if time_lost is not None:
            new_item = item.copy()
            new_item["performance_vs_avg"] = f"{((time_lost / field_avg_lost_time) - 1) * 100:+.1f}%"
            contextual_list.append(new_item)
            
    return contextual_list

def find_individual_outliers(data: Dict) -> List[Dict]:
    """Finds single-instance insights that aren't full rankings."""
    insights = []
    
    # Largest Teammate Pace Gap
    delta_data = data.get("driver_deltas_by_car", [])
    if delta_data:
        valid_deltas = [d for d in delta_data if _time_str_to_seconds(d.get("average_lap_time_delta_for_car")) is not None]
        if valid_deltas:
            worst_entry = max(valid_deltas, key=lambda x: _time_str_to_seconds(x.get("average_lap_time_delta_for_car")))
            gap_val = _time_str_to_seconds(worst_entry.get("average_lap_time_delta_for_car"))
            if gap_val and gap_val > 0.5: # Threshold for significance
                insights.append({
                    "category": "Driver Performance",
                    "type": "Largest Teammate Pace Gap",
                    "car_number": worst_entry.get("car_number"),
                    "details": f"Car #{worst_entry.get('car_number')} has the largest pace difference between its drivers, with an average gap of {gap_val:.3f}s in best lap times."
                })

    return insights

# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def derive_insights(data: Dict[str, Any]) -> Dict[str, Any]:
    """Generates a structured dictionary of ranked insights from the analysis data."""
    
    enhanced_data = data.get("enhanced_strategy_analysis", [])
    pit_cycle_data = data.get("full_pit_cycle_analysis", [])
    
    insights = {
        "manufacturer_pace_ranking": _rank_by_metric(
            enhanced_data, "manufacturer", ["avg_green_pace_fuel_corrected"], higher_is_better=False, value_is_time=True
        ),
        "manufacturer_tire_wear_ranking": _rank_by_metric(
            enhanced_data, "manufacturer", ["tire_degradation_model", "deg_coeff_a"], higher_is_better=False, value_is_time=False
        ),
        "manufacturer_pit_cycle_ranking": _rank_by_metric(
            pit_cycle_data, "manufacturer", ["average_cycle_loss"], higher_is_better=False, value_is_time=True
        ),
        "car_untapped_potential_ranking": _rank_cars_by_untapped_potential(
            data.get("fastest_by_car_number", [])
        ),
        "driver_traffic_meister_ranking": _rank_drivers_by_traffic_management(
            data.get("traffic_management_analysis", [])
        ),
        "individual_outliers": find_individual_outliers(data)
    }
    
    return insights


def enrich_insights_with_ai(insights: Dict[str, Any]) -> Dict[str, Any]:
    """Adds LLM commentary to each list of ranked insights."""
    if not USE_AI_ENHANCED or not insights:
        return insights
        
    enriched_insights = insights.copy()
    
    # Example for one ranking list, can be expanded to others
    pace_ranking = insights.get("manufacturer_pace_ranking")
    if pace_ranking:
        prompt = (
            "You are a professional motorsport strategist AI. I will provide a JSON list of manufacturers ranked by their average race pace. "
            "Your task is to add a new key, 'llm_commentary', to each object in the list. "
            "This commentary should be a concise, professional one-sentence analysis about their performance relative to the field. "
            "You should provide an insightful commentary about the relative performance of each manufacturer, outlining how and where they compare relative to the field. "
            "Return only the updated JSON list, with no other text.\n\n"
            f"Pace Ranking:\n{json.dumps(pace_ranking, indent=2)}\n"
        )
        try:
            enriched_pace = ai_helpers.generate_json(
                prompt, max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 25000))
            )
            if isinstance(enriched_pace, list) and len(enriched_pace) == len(pace_ranking):
                enriched_insights["manufacturer_pace_ranking"] = enriched_pace
        except Exception as e:
            LOGGER.warning("AI enrichment for pace ranking failed: %s", e)
            
    return enriched_insights

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
    except Exception as exc:
        LOGGER.exception("Invalid request: %s", exc)
        return Response("Bad Request", status=400)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        local_analysis_path = tmp / "analysis_data.json"
        try:
            LOGGER.info("Downloading analysis file: %s", analysis_uri)
            _gcs_download(analysis_uri, local_analysis_path)
            with local_analysis_path.open("r", encoding="utf-8") as fp:
                analysis_data = json.load(fp)

            LOGGER.info("Deriving ranked insights from analysis data...")
            insights = derive_insights(analysis_data)
            
            if USE_AI_ENHANCED:
                LOGGER.info("Enriching insights with AI...")
                insights = enrich_insights_with_ai(insights)

            basename = pathlib.Path(analysis_uri).stem.replace("_enhanced", "")
            insights_filename = f"{basename}_insights.json"
            local_insights_path = tmp / insights_filename
            with local_insights_path.open("w", encoding="utf-8") as fp:
                json.dump(insights, fp, indent=2)
            LOGGER.info("Successfully generated insights file: %s", insights_filename)

            insights_uri = _gcs_upload(local_insights_path, ANALYZED_DATA_BUCKET, insights_filename)
            LOGGER.info("Uploaded insights to: %s", insights_uri)

            _publish_visualization_request(analysis_uri, insights_uri)
        except Exception as exc:
            LOGGER.exception("Processing failed: %s", exc)
            return Response("Internal Server Error", status=500)

    return Response(status=204)