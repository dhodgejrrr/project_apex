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


# Enhanced logging configuration
DEBUG_AI_RESPONSES = os.getenv("DEBUG_AI_RESPONSES", "false").lower() == "true"
LOG_PROMPT_CONTENT = os.getenv("LOG_PROMPT_CONTENT", "false").lower() == "true"

# ---------------------------------------------------------------------------
# Vertex AI initialisation
# ---------------------------------------------------------------------------
_aiplatform_inited = False


def _init_vertex() -> None:
    """Initialises Vertex AI using environment variables just-in-time."""
    global _aiplatform_inited  # pylint: disable=global-statement
    if _aiplatform_inited:
        return

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
def generate_json(prompt: str, temperature: float = 0.7, max_output_tokens: int = 50000) -> Any:
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
        
        # Check if response has candidates and content before proceeding
        if not response.candidates:
            raise ValueError("No candidates in response")
            
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, 'finish_reason', None)
        
        # Handle MAX_TOKENS scenario specifically
        if finish_reason == 2 or str(finish_reason) == "MAX_TOKENS":
            LOGGER.warning("Response truncated due to MAX_TOKENS limit. Consider increasing max_output_tokens.")
            # Check if we have any partial content
            if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts'):
                parts = candidate.content.parts
                if parts and hasattr(parts[0], 'text') and parts[0].text:
                    text = parts[0].text.strip()
                    LOGGER.info("Attempting to parse partial response due to token limit")
                    try:
                        return json.loads(text)
                    except json.JSONDecodeError:
                        LOGGER.warning("Partial response is not valid JSON, treating as error")
                        raise ValueError("Response truncated and not valid JSON")
            raise ValueError("Response truncated due to token limit with no usable content")
        
        # Log successful response details for comparison with errors (if debug enabled)
        if DEBUG_AI_RESPONSES:
            LOGGER.info("=== SUCCESSFUL GEMINI RESPONSE ===")
            LOGGER.info("Model: %s", model_name)
            LOGGER.info("Finish Reason: %s (%s)", finish_reason, _explain_finish_reason(finish_reason))
            LOGGER.info("Response Text Length: %d", len(response.text) if response.text else 0)
            
        LOGGER.debug("Full AI Response: %s", response)
        
        text = response.text.strip()
        parsed_json = json.loads(text)
        
        if DEBUG_AI_RESPONSES:
            LOGGER.info("JSON parsing successful")
            
        return parsed_json
    except (ValueError, json.JSONDecodeError, TypeError) as exc:
        # Enhanced logging for debugging API issues
        _log_detailed_api_error(response, exc, model_name, prompt)
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
        # Enhanced logging for debugging API issues
        _log_detailed_api_error(response, exc, model_name, prompt)
        LOGGER.warning("Summarization failed - returning empty string")
        # Return empty string on failure to avoid breaking callers
        return ""


def _log_detailed_api_error(response: Any, exc: Exception, model_name: str, prompt: str) -> None:
    """Log detailed information about API errors for debugging.
    
    This function provides comprehensive logging when Gemini API calls fail,
    including response metadata, error details, and prompt information.
    """
    # Always log basic error info
    LOGGER.error("GEMINI API ERROR: %s - %s", type(exc).__name__, str(exc))
    LOGGER.error("Model: %s", model_name)
    
    # Detailed logging only if DEBUG_AI_RESPONSES is enabled
    if not DEBUG_AI_RESPONSES:
        LOGGER.error("Enable DEBUG_AI_RESPONSES=true for detailed error analysis")
        return
        
    LOGGER.error("=== DETAILED GEMINI API ERROR ANALYSIS ===")
    
    # Log prompt details (only if LOG_PROMPT_CONTENT is enabled)
    if LOG_PROMPT_CONTENT:
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        LOGGER.error("Prompt (first 500 chars): %s", prompt_preview)
    LOGGER.error("Prompt Length: %d characters", len(prompt))
    
    if response is None:
        LOGGER.error("Response: None (API call failed completely)")
        return
    
    # Log response metadata
    LOGGER.error("Response Object Type: %s", type(response).__name__)
    
    # Check if response has prompt_feedback
    try:
        prompt_feedback = getattr(response, 'prompt_feedback', None)
        if prompt_feedback:
            LOGGER.error("Prompt Feedback: %s", prompt_feedback)
            # Log safety ratings if available
            if hasattr(prompt_feedback, 'safety_ratings'):
                LOGGER.error("Safety Ratings: %s", prompt_feedback.safety_ratings)
            if hasattr(prompt_feedback, 'block_reason'):
                LOGGER.error("Block Reason: %s", prompt_feedback.block_reason)
        else:
            LOGGER.error("Prompt Feedback: None")
    except Exception as e:
        LOGGER.error("Error accessing prompt_feedback: %s", e)
    
    # Check candidates
    try:
        candidates = getattr(response, 'candidates', [])
        LOGGER.error("Number of Candidates: %d", len(candidates))
        
        for i, candidate in enumerate(candidates):
            LOGGER.error("--- Candidate %d ---", i)
            
            # Finish reason
            finish_reason = getattr(candidate, 'finish_reason', 'UNKNOWN')
            LOGGER.error("Finish Reason: %s (%s)", finish_reason, _explain_finish_reason(finish_reason))
            
            # Safety ratings
            safety_ratings = getattr(candidate, 'safety_ratings', [])
            if safety_ratings:
                LOGGER.error("Safety Ratings:")
                for rating in safety_ratings:
                    category = getattr(rating, 'category', 'UNKNOWN')
                    probability = getattr(rating, 'probability', 'UNKNOWN')
                    LOGGER.error("  %s: %s", category, probability)
            
            # Content (if any)
            try:
                content = getattr(candidate, 'content', None)
                if content:
                    parts = getattr(content, 'parts', [])
                    for j, part in enumerate(parts):
                        text = getattr(part, 'text', '')
                        if text:
                            text_preview = text[:200] + "..." if len(text) > 200 else text
                            LOGGER.error("Content Part %d (first 200 chars): %s", j, text_preview)
                            LOGGER.error("Content Part %d Length: %d characters", j, len(text))
                            
                            # Try to parse as JSON to see if it's malformed
                            if isinstance(exc, json.JSONDecodeError):
                                LOGGER.error("JSON Parse Error Details:")
                                LOGGER.error("  Error: %s", exc.msg)
                                LOGGER.error("  Position: line %d column %d (char %d)", exc.lineno, exc.colno, exc.pos)
                                # Show context around the error
                                if exc.pos < len(text):
                                    start = max(0, exc.pos - 50)
                                    end = min(len(text), exc.pos + 50)
                                    context = text[start:end]
                                    LOGGER.error("  Context around error: %s", repr(context))
                                if LOG_PROMPT_CONTENT:
                                    LOGGER.error("  Full Content for JSON Analysis: %s", repr(text))
                        else:
                            LOGGER.error("Content Part %d: Empty", j)
                else:
                    LOGGER.error("No content in candidate")
            except Exception as e:
                LOGGER.error("Error accessing candidate content: %s", e)
                
    except Exception as e:
        LOGGER.error("Error accessing candidates: %s", e)
    
    # Usage metadata
    try:
        usage_metadata = getattr(response, 'usage_metadata', None)
        if usage_metadata:
            LOGGER.error("Usage Metadata: %s", usage_metadata)
        else:
            LOGGER.error("Usage Metadata: None")
    except Exception as e:
        LOGGER.error("Error accessing usage_metadata: %s", e)
    
    # Raw response representation (last resort)
    if LOG_PROMPT_CONTENT:
        try:
            LOGGER.error("Raw Response Representation: %s", repr(response))
        except Exception as e:
            LOGGER.error("Error getting raw response representation: %s", e)
    
    LOGGER.error("=== END DETAILED ANALYSIS ===")


def _explain_finish_reason(finish_reason: Any) -> str:
    """Provide human-readable explanation of finish reasons."""
    explanations = {
        1: "STOP - Natural completion",
        2: "MAX_TOKENS - Reached token limit", 
        3: "SAFETY - Blocked by safety filters",
        4: "RECITATION - Blocked for potential copyright issues",
        5: "OTHER - Other reason",
        "STOP": "Natural completion",
        "MAX_TOKENS": "Reached token limit",
        "SAFETY": "Blocked by safety filters", 
        "RECITATION": "Blocked for potential copyright issues",
        "OTHER": "Other reason",
        "FINISH_REASON_UNSPECIFIED": "Unspecified reason"
    }
    return explanations.get(finish_reason, f"Unknown reason: {finish_reason}")


def generate_json_adaptive(prompt: str, temperature: float = 0.7, max_output_tokens: int = 50000, 
                          adaptive_retry: bool = True) -> Any:
    """
    Calls Gemini to generate JSON with adaptive token limit adjustment.
    
    If the first attempt hits MAX_TOKENS, automatically retries with a higher limit.
    """
    try:
        return generate_json(prompt, temperature, max_output_tokens)
    except ValueError as e:
        if not adaptive_retry:
            raise
            
        error_msg = str(e).lower()
        if "token limit" in error_msg or "max_tokens" in error_msg:
            # Try with doubled token limit
            new_limit = min(max_output_tokens * 2, 100000)  # Cap at 100k tokens
            LOGGER.info("Retrying with increased token limit: %d -> %d", max_output_tokens, new_limit)
            try:
                return generate_json(prompt, temperature, new_limit)
            except ValueError as e2:
                if "token limit" in str(e2).lower():
                    # If still hitting limits, try to truncate the prompt
                    if len(prompt) > 2000:
                        truncated_prompt = prompt[:1500] + "\n\n[...content truncated due to length...]\n\n" + prompt[-500:]
                        LOGGER.info("Retrying with truncated prompt: %d -> %d chars", len(prompt), len(truncated_prompt))
                        return generate_json(truncated_prompt, temperature, new_limit)
                raise e2
        raise e
