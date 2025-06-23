"""Shared AI helper utilities for Project Apex agents.

Provides thin wrappers around Vertex AI Gemini for JSON-structured generation
with retries and centralised configuration.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict
from collections import defaultdict

import google.auth

try:
    from vertexai.generative_models import (
        GenerativeModel,
        GenerationConfig,
        HarmCategory,
        HarmBlockThreshold,
    )
except ImportError:  # Vertex AI not available or outdated during offline/local runs
    class _Stub:  # type: ignore
        """Fallback stub for missing Vertex AI classes when running without Vertex AI SDK."""
        def __getattr__(self, name):  # noqa: D401, D401
            return self
        def __call__(self, *args, **kwargs):
            return self
    GenerativeModel = GenerationConfig = HarmCategory = HarmBlockThreshold = _Stub()  # type: ignore
import vertexai
from tenacity import retry, stop_after_attempt, wait_exponential

# ---------------------------------------------------------------------------
# Configuration & Cost Tracking
# ---------------------------------------------------------------------------

# Token pricing (USD per 1K tokens). Update as Google pricing evolves.
TOKEN_PRICES: Dict[str, Dict[str, float]] = {
    "gemini-2.5-flash": {"in": 0.0003, "out": 0.0025},
    # "gemini-pro": {"in": 0.000375, "out": 0.00075},
    # Add other models as required
}

# Accumulate usage during runtime (cumulative per model)
_usage_totals: Dict[str, Dict[str, int]] = defaultdict(lambda: {"prompt": 0, "completion": 0, "total": 0})
# Store per-call token usage details in the order that API calls were made
_usage_calls: list[Dict[str, int | str]] = []
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
# Internal helpers for usage tracking
# ---------------------------------------------------------------------------

def _record_usage(model_name: str, usage_md: Any | None) -> None:  # type: ignore[valid-type]
    """Record prompt / completion tokens for a model run.

    In addition to accumulating totals per model, we now store a per-call
    breakdown so that detailed auditing is possible. Each call appends an
    entry to the global ``_usage_calls`` list.
    """
    if not usage_md:
        return

    prompt_tokens = usage_md.prompt_token_count or 0
    completion_tokens = usage_md.candidates_token_count or 0
    total_tokens = usage_md.total_token_count or 0

    # Cumulative per-model totals
    _usage_totals[model_name]["prompt"] += prompt_tokens
    _usage_totals[model_name]["completion"] += completion_tokens
    _usage_totals[model_name]["total"] += total_tokens

    # Per-call record (order preserved)
    _usage_calls.append(
        {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
    )


def get_usage_summary() -> Dict[str, Any]:  # type: ignore[override]
    """Return detailed usage information.

    The returned dictionary now contains *both* per-model cumulative totals and
    a chronological list of per-call token counts. The structure is:

    ```json
    {
      "gemini-2.5-flash": {
        "prompt_tokens": 123,
        "completion_tokens": 456,
        "total_tokens": 579,
        "estimated_cost_usd": 0.123
      },
      "_overall": {
        "prompt_tokens": 123,
        "completion_tokens": 456,
        "total_tokens": 579,
        "estimated_cost_usd": 0.123
      },
      "_calls": [
        {"model": "gemini-2.5-flash", "prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        ...
      ]
    }
    """
    summary: Dict[str, Any] = {}
    overall_prompt = overall_completion = overall_total = overall_cost = 0.0

    # Per-model aggregates
    for model, counts in _usage_totals.items():
        price_cfg = TOKEN_PRICES.get(model, {"in": 0.0, "out": 0.0})
        cost = (
            counts["prompt"] / 1000 * price_cfg["in"] +
            counts["completion"] / 1000 * price_cfg["out"]
        )
        summary[model] = {
            "prompt_tokens": counts["prompt"],
            "completion_tokens": counts["completion"],
            "total_tokens": counts["total"],
            "estimated_cost_usd": round(cost, 6),
        }
        overall_prompt += counts["prompt"]
        overall_completion += counts["completion"]
        overall_total += counts["total"]
        overall_cost += cost

    # Grand totals across all models
    summary["_overall"] = {
        "prompt_tokens": int(overall_prompt),
        "completion_tokens": int(overall_completion),
        "total_tokens": int(overall_total),
        "estimated_cost_usd": round(overall_cost, 6),
    }

    # Chronological per-call breakdown
    summary["_calls"] = list(_usage_calls)  # shallow copy to avoid external mutation
    return summary


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
@retry(wait=wait_exponential(multiplier=1, min=2, max=10), stop=stop_after_attempt(3))
def generate_json(prompt: str, temperature: float = 0.7, max_output_tokens: int = 25000) -> Any:
    """Calls Gemini to generate JSON and parses the result.

    If parsing fails, raises ValueError to trigger retry.
    """
    _init_vertex()
    model_name = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
    LOGGER.info("Using Vertex AI model: %s", model_name)
    LOGGER.debug("Prompt for generate_json:\n%s", prompt)
    model = GenerativeModel(model_name)
    config = GenerationConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
    )
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    response = None
    try:
        response = model.generate_content(
            prompt, generation_config=config, safety_settings=safety_settings
        )
        # Track token usage metadata
        _record_usage(model_name, getattr(response, "usage_metadata", None))
        LOGGER.debug("Full AI Response: %s", response)
        text = response.text.strip()
        return json.loads(text)
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        # ValueError is raised by response.text if the candidate is empty/blocked.
        log_message = "Gemini response not valid JSON or was blocked/truncated."
        if response:
            log_message += (
                f" Prompt Feedback: {response.prompt_feedback}. "
                f"Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}."
            )
        LOGGER.warning(log_message)
        raise ValueError("Invalid JSON or empty response from Gemini") from exc


def summarize(text: str, **kwargs: Any) -> str:
    """Simple summarization wrapper returning plain text."""
    prompt = (
        "Summarize the following text in 3 concise sentences:\n\n" + text + "\n\nSummary:"
    )
    _init_vertex()
    model_name = os.environ.get("VERTEX_MODEL", "gemini-2.5-flash")
    LOGGER.info("Using Vertex AI model for summarization: %s", model_name)
    LOGGER.debug("Prompt for summarize:\n%s", prompt)
    model = GenerativeModel(model_name)

    # Allow overriding default config via kwargs
    # config_args = {"temperature": 0.2, "max_output_tokens": 256}
    # config_args.update(kwargs)
    # config = GenerationConfig(**config_args)

    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    response = None
    try:
        response = model.generate_content(
            prompt, safety_settings=safety_settings
        )
        _record_usage(model_name, getattr(response, "usage_metadata", None))
        LOGGER.debug("Full AI Response: %s", response)
        if not response.text:
            raise ValueError("Empty response from Gemini.")
        return response.text.strip()
    except (ValueError, AttributeError) as exc:
        log_message = "Summarization failed or was blocked/truncated."
        if response:
            log_message += (
                f" Prompt Feedback: {response.prompt_feedback}. "
                f"Finish Reason: {response.candidates[0].finish_reason if response.candidates else 'N/A'}."
            )
        LOGGER.warning(log_message)
        # Return empty string on failure to avoid breaking callers
        return ""
