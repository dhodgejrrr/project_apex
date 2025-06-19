"""Shared AI helper utilities for Project Apex agents.

Provides thin wrappers around Vertex AI Gemini for JSON-structured generation
with retries and centralised configuration.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

import google.auth
from vertexai.generative_models import (
    GenerativeModel, GenerationConfig, HarmCategory, HarmBlockThreshold
)
import vertexai
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOGGER = logging.getLogger("apex.ai_helpers")


# ---------------------------------------------------------------------------
# Vertex AI initialisation
# ---------------------------------------------------------------------------
_aiplatform_inited = False


def _init_vertex() -> None:
    """Initialises Vertex AI using environment variables just-in-time."""
    global _aiplatform_inited  # pylint: disable=global-statement
    if not _aiplatform_inited:
        # Use google.auth.default() to robustly find the credentials and project.
        # This is the standard and most reliable way to authenticate.
        try:
            credentials, project_id = google.auth.default()
        except google.auth.exceptions.DefaultCredentialsError as e:
            LOGGER.error(
                "Authentication failed. Please run `gcloud auth application-default login`"
            )
            raise e

        # Use found project_id if not explicitly set, and get location from env.
        project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        location = os.environ.get("VERTEX_LOCATION", "us-central1")

        if not project_id:
            raise ValueError(
                "Could not determine Project ID. Please set GOOGLE_CLOUD_PROJECT."
            )

        vertexai.init(project=project_id, location=location, credentials=credentials)
        _aiplatform_inited = True


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def generate_json(prompt: str, temperature: float = 0.7, max_output_tokens: int = 12000) -> Any:
    """Calls Gemini to generate JSON and parses the result.

    If parsing fails, raises ValueError to trigger retry.
    """
    _init_vertex()
    model_name = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")  
    LOGGER.info("Using Vertex AI model: %s", model_name)
    model = GenerativeModel(model_name)
    config = GenerationConfig(
        temperature=temperature,
        max_output_tokens=12000,
        response_mime_type="application/json", 
    )
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    response = model.generate_content(
        prompt, generation_config=config, safety_settings=safety_settings
    )
    try:
        text = response.text.strip()
        return json.loads(text)
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        # ValueError is raised by response.text if the candidate is empty/blocked.
        LOGGER.warning(
            "Gemini response not valid JSON or was blocked/truncated. Response: %s", response
        )
        raise ValueError("Invalid JSON or empty response from Gemini") from exc


def summarize(text: str, **kwargs: Dict[str, Any]) -> str:
    """Simple summarization wrapper returning plain text."""
    prompt = (
        "Summarize the following text in 3 concise sentences:\n\n" + text + "\n\nSummary:"
    )
    _init_vertex()
    model_name = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash") 
    LOGGER.info("Using Vertex AI model for summarization: %s", model_name)
    model = GenerativeModel(model_name)
    config = GenerationConfig(temperature=0.2, max_output_tokens=256)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    response = model.generate_content(
        prompt, generation_config=config, safety_settings=safety_settings
    )
    return response.text.strip()
