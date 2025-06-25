"""
Common utilities for handling both HTTP and Pub/Sub requests across all agents.
"""
import base64
import json
import logging
from typing import Any, Dict, Optional
from flask import Request

LOGGER = logging.getLogger(__name__)

def parse_request_payload(request: Request) -> Dict[str, Any]:
    """
    Parse request payload that could be either:
    1. Direct JSON payload (for direct HTTP calls)
    2. Pub/Sub push message format (for GCP Pub/Sub integration)
    
    Returns the actual payload data in both cases.
    """
    req_json = request.get_json(force=True, silent=True)
    if req_json is None:
        raise ValueError("Invalid or missing JSON payload")
    
    # Check if this is a Pub/Sub push message format
    if "message" in req_json and "data" in req_json["message"]:
        try:
            # Decode Pub/Sub message data
            decoded = base64.b64decode(req_json["message"]["data"]).decode("utf-8")
            payload = json.loads(decoded)
            LOGGER.info("Parsed Pub/Sub push message payload")
            return payload
        except Exception as e:
            LOGGER.warning(f"Failed to parse as Pub/Sub message: {e}")
            # Fall through to treat as direct JSON
    
    # Treat as direct JSON payload
    LOGGER.info("Parsed direct JSON payload")
    return req_json

def validate_required_fields(payload: Dict[str, Any], required_fields: list) -> None:
    """
    Validate that all required fields are present in the payload.
    """
    missing_fields = [field for field in required_fields if field not in payload]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
