"""DataIngestor Cloud Function for Project Apex.

Triggered by Cloud Storage finalize events. Determines whether both the race CSV
and the corresponding _pits.json file exist for a given event. When both are
present, publishes a message to the `analysis-requests` Pub/Sub topic so that
analysis can begin.

Entry point: trigger_analysis
"""
from __future__ import annotations

import base64
import json
import logging
import os
import re
from typing import Tuple

from google.cloud import pubsub_v1
from google.cloud import storage

# Configure basic structured logging. Cloud Functions picks up stdout/stderr.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

# Pre-compile regex to extract base event name.
# Matches either `2025_mido_race.csv` or `2025_mido_race_pits.json`.
CSV_PATTERN = re.compile(r"^(?P<basename>.+?)\.csv$")
PIT_PATTERN = re.compile(r"^(?P<basename>.+?)_pits\.json$")

PROJECT_ID = os.getenv("GCP_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
TOPIC_ID = os.getenv("ANALYSIS_REQUESTS_TOPIC", "analysis-requests")

storage_client: storage.Client | None = None
publisher: pubsub_v1.PublisherClient | None = None

def _init_clients() -> Tuple[storage.Client, pubsub_v1.PublisherClient]:
    """Initialises and caches Google Cloud clients."""
    global storage_client, publisher  # pylint: disable=global-statement
    if storage_client is None:
        storage_client = storage.Client()
    if publisher is None:
        publisher = pubsub_v1.PublisherClient()
    return storage_client, publisher


def _parse_filenames(object_name: str) -> Tuple[str | None, str | None]:
    """Given the uploaded object name, returns expected sibling filename.

    Returns a tuple ``(csv_name, pit_name)`` where one of the elements may be
    ``None`` if it does not correspond to the uploaded type.
    """
    csv_name = pit_name = None
    if CSV_PATTERN.match(object_name):
        basename = CSV_PATTERN.match(object_name).group("basename")  # type: ignore[index]
        csv_name = object_name
        pit_name = f"{basename}_pits.json"
    elif PIT_PATTERN.match(object_name):
        basename = PIT_PATTERN.match(object_name).group("basename")  # type: ignore[index]
        csv_name = f"{basename}.csv"
        pit_name = object_name
    return csv_name, pit_name


def _object_exists(bucket: storage.Bucket, blob_name: str) -> bool:
    """Checks if a blob exists in the provided bucket."""
    return bucket.blob(blob_name).exists()


def _publish_message(csv_path: str, pit_json_path: str) -> None:
    """Publishes a JSON message to the analysis-requests topic."""
    _, publisher_client = _init_clients()
    topic_path = publisher_client.topic_path(PROJECT_ID, TOPIC_ID)

    payload_dict = {
        "csv_path": csv_path,
        "pit_json_path": pit_json_path,
    }
    payload_bytes = json.dumps(payload_dict).encode("utf-8")
    # Pub/Sub expects bytes. Enable message ordering not required here.
    future = publisher_client.publish(topic_path, payload_bytes)
    future.result(timeout=30)  # Block until publish succeeds for reliability.
    LOGGER.info("Published analysis request: %s", payload_dict)


# Entry point for Cloud Function
def trigger_analysis(event, context):  # pylint: disable=unused-argument
    """Background Cloud Function to be triggered by Cloud Storage finalize events.

    Args:
        event (dict):  The dictionary with data specific to this type of event.
                        The `bucket` and `name` keys are particularly important.
        context (google.cloud.functions.Context): Metadata describing the event.
    """
    bucket_name = event.get("bucket")
    object_name = event.get("name")

    if not bucket_name or not object_name:
        LOGGER.error("Missing bucket or object name in event: %s", event)
        return

    csv_name, pit_name = _parse_filenames(object_name)
    if not pit_name or not csv_name:
        # Uploaded file does not match expected patterns; ignore.
        LOGGER.info("File '%s' is not a race CSV or pits JSON. Ignoring.", object_name)
        return

    storage_client_, _ = _init_clients()
    bucket = storage_client_.bucket(bucket_name)

    # Determine which sibling needs to be present.
    sibling_name = pit_name if object_name.endswith(".csv") else csv_name
    if not _object_exists(bucket, sibling_name):
        LOGGER.info("Sibling file '%s' not found yet. Waiting for next trigger.", sibling_name)
        return

    # Both files present – publish Pub/Sub message.
    csv_gcs_uri = f"gs://{bucket_name}/{csv_name}"
    pit_gcs_uri = f"gs://{bucket_name}/{pit_name}"

    try:
        _publish_message(csv_gcs_uri, pit_gcs_uri)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Failed to publish analysis request for %s: %s", object_name, exc)
        return

    LOGGER.info("Trigger processed successfully for event '%s'", object_name)
