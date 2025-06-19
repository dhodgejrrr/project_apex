"""Shared AI helper utilities for Project Apex agents.

Provides thin wrappers around Vertex AI Gemini for JSON-structured generation
with retries and centralised configuration.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

from google.cloud import aiplatform
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
MODEL_NAME = os.getenv("VERTEX_MODEL", "gemini-1.0-pro")

LOGGER = logging.getLogger("apex.ai_helpers")


# ---------------------------------------------------------------------------
# Vertex AI initialisation
# ---------------------------------------------------------------------------
_aiplatform_inited = False


def _init_vertex() -> None:
    global _aiplatform_inited  # pylint: disable=global-statement
    if not _aiplatform_inited:
        aiplatform.init(project=PROJECT_ID, location=LOCATION)
        _aiplatform_inited = True


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def generate_json(prompt: str, temperature: float = 0.7, max_output_tokens: int = 1024) -> Any:
    """Calls Gemini to generate JSON and parses the result.

    If parsing fails, raises ValueError to trigger retry.
    """
    _init_vertex()
    model = aiplatform.TextGenerationModel.from_pretrained(MODEL_NAME)
    response = model.predict(prompt, temperature=temperature, max_output_tokens=max_output_tokens)
    text = response.text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        LOGGER.warning("Gemini response not valid JSON: %s", text)
        raise ValueError("Invalid JSON from Gemini") from exc


def summarize(text: str, **kwargs: Dict[str, Any]) -> str:
    """Simple summarization wrapper returning plain text."""
    prompt = (
        "Summarize the following text in 3 concise sentences:\n\n" + text + "\n\nSummary:"
    )
    _init_vertex()
    model = aiplatform.TextGenerationModel.from_pretrained(MODEL_NAME)
    response = model.predict(prompt, **kwargs)
    return response.text.strip()
