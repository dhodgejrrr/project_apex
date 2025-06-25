You are tasked to implement a feature. Instructions are as follows:

We have been trying to deploy this code to google cloud. While it will build, deploy, and mostly run, we still need more end to end testing. Constantly building, deploying, and testing cloud based is extremely time consuming. 

We want to come up with a plan for local testing the most closely aligns with how our production environment will have all the services, agents, and UI interacting with each other. 

We just added run_local_e2e.py and docker-compose.yml

How do we use these to test locally?

Instructions for the output format:
- Output code without descriptions, unless it is important.
- Minimize prose, comments and empty lines.
- Only show the relevant code that needs to be modified. Use comments to represent the parts that are not modified.
- Make it easy to copy and paste.
- Consider other possibilities to achieve the result, do not be limited by the prompt.

.dockerignore
```dockerignore
# Git
.git
.gitignore

# Docker
.dockerignore
docker-compose.yml

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/

# Local run artifacts
out_local/
*.egg-info/

# Test files and harnesses
tests/
run_local_e2e.py
dry_run_harness.py
full_pipeline_local.py

# IDE / OS
.vscode/
.idea/
.DS_Store
```

agents/common/ai_helpers.py
```py
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
```

agents/core_analyzer/imsa_analyzer.py
```py
import pandas as pd
import json
import numpy as np
import os
import warnings

# Optional: Suppress the FutureWarning about downcasting if it's too noisy
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class IMSADataAnalyzer:
    """
    This class combines the original user-defined race analysis methods with
    new, enhanced strategic analysis capabilities, including traffic-based lap
    categorization and a predictive polynomial tire degradation model.
    """

    # <<< MODIFIED __init__ >>>
    def __init__(self, csv_filepath, pit_json_filepath=None, fuel_capacity_json_filepath=None, config=None):
        """
        Initializes the analyzer with the data and configuration.
        """
        print(f"Initializing IMSADataAnalyzer with CSV: {csv_filepath}")
        
        default_config = {'fuel_burn_rate_kg_per_lap': 2.2, 'fuel_weight_penalty_s_per_kg': 0.035, 'max_fuel_load_kg': 85.0, 'pit_lane_delta_s': 45.0, 'traffic_proximity_threshold_s': 2.0, 'traffic_compromise_threshold_s': 1.5, 'driver_potential_percentile': 0.05, 'min_laps_for_deg_model': 5, 'min_laps_for_metronome': 5, 'min_laps_for_metronome_longer': 10, 'min_laps_for_manu_showdown': 15}
        self.config = default_config
        if config: self.config.update(config)

        # <<< NEW: Load manufacturer-specific fuel capacities >>>
        self.fuel_capacities = None
        if fuel_capacity_json_filepath:
            try:
                with open(fuel_capacity_json_filepath, 'r') as f:
                    # Load and normalize keys to be lowercase for case-insensitive matching
                    loaded_caps = json.load(f)
                    self.fuel_capacities = {k.lower().strip(): v for k, v in loaded_caps.items()}
                print(f"Loaded manufacturer fuel capacities from {fuel_capacity_json_filepath}.")
            except Exception as e:
                print(f"WARNING: Failed to load fuel capacity JSON {fuel_capacity_json_filepath}: {e}. Will use default value.")
        # <<< END NEW >>>

        self.pit_data_df = None
        if pit_json_filepath:
            try:
                self.pit_data_df = self._load_pit_data(pit_json_filepath); print(f"Loaded pit JSON data from {pit_json_filepath}.")
            except Exception as e:
                print(f"WARNING: Failed to load pit JSON {pit_json_filepath}: {e}. Will use CSV-derived pit data if necessary.")
        
        try:
            self.df = pd.read_csv(csv_filepath, sep=';'); self.df.columns = self.df.columns.str.strip(); print(f"CSV loaded successfully. Shape: {self.df.shape}")
        except FileNotFoundError: raise FileNotFoundError(f"Error: The file {csv_filepath} was not found.")
        if self.df.empty: raise ValueError("CSV file is empty or not parsed correctly.")

        # <<< NEW: Filter rows based on allowed classes from environment variable >>>
        # env_classes = os.getenv('IMSA_CLASSES') or os.getenv('ALLOWED_CLASSES')
        # if env_classes:
        #     try:
        #         allowed_classes = json.loads(env_classes)
        #         if isinstance(allowed_classes, str):
        #             allowed_classes = [allowed_classes]
        #     except json.JSONDecodeError:
        #         allowed_classes = [c.strip() for c in env_classes.split(',')]
        #     allowed_classes = [c.upper().strip() for c in allowed_classes if c.strip()]
        #     if 'CLASS' in self.df.columns and allowed_classes:
        #         before_rows = len(self.df)
        #         self.df = self.df[self.df['CLASS'].str.upper().isin(allowed_classes)].copy()
        #         print(f"Applied class filter {allowed_classes}: rows {before_rows} -> {len(self.df)}.")
        # else:
        #     print("No class filter applied.")
        # <<< END NEW >>>

        self._preprocess_data()

    def _parse_time_to_seconds(self, time_str):
        if pd.isna(time_str) or not isinstance(time_str, str) or time_str.strip() == "": return np.nan
        time_str = str(time_str).strip(); parts = time_str.split(':')
        try:
            if len(parts) == 1: return float(parts[0])
            elif len(parts) == 2: return int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3: return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        except (ValueError, IndexError): return np.nan
        return np.nan

    def _format_seconds_to_ms_str(self, total_seconds):
        if pd.isna(total_seconds) or not isinstance(total_seconds, (int, float)): return None
        sign = "";
        if total_seconds < 0: sign = "-"; total_seconds = abs(total_seconds)
        minutes = int(total_seconds // 60); seconds_part = total_seconds % 60
        if minutes > 0: return f"{sign}{minutes}:{seconds_part:06.3f}"
        return f"{sign}{total_seconds:.3f}"
        
    # <<< MODIFIED _preprocess_data >>>
    def _preprocess_data(self):
        print("\n--- Starting Master Preprocessing ---"); df = self.df; time_cols = ['LAP_TIME', 'S1', 'S2', 'S3', 'PIT_TIME']
        for col in time_cols:
            if col in df.columns: df[col + '_SEC'] = df[col].apply(self._parse_time_to_seconds)
        if 'HOUR' in df.columns: df['FINISH_TIME_DT'] = pd.to_datetime(df['HOUR'], format='%H:%M:%S.%f', errors='coerce')
        else: df['FINISH_TIME_DT'] = pd.NaT
        num_cols = ['LAP_NUMBER', 'KPH']; str_cols = ['NUMBER', 'DRIVER_NUMBER', 'DRIVER_NAME', 'TEAM', 'MANUFACTURER', 'FLAG_AT_FL', 'CROSSING_FINISH_LINE_IN_PIT', 'CLASS', 'GROUP']
        for col in num_cols:
            if col in df.columns: df[col] = pd.to_numeric(df[col], errors='coerce')
        for col in str_cols:
            if col in df.columns: df[col] = df[col].fillna('').astype(str).str.strip()
        df = df.sort_values(by=['NUMBER', 'LAP_NUMBER']).copy()
        df['is_pit_stop_lap'] = df['CROSSING_FINISH_LINE_IN_PIT'] == 'B'; df['is_stint_start'] = df.groupby('NUMBER')['is_pit_stop_lap'].shift(1).fillna(True)
        df['stint_id_num'] = df.groupby('NUMBER')['is_stint_start'].cumsum(); df['stint_id'] = df['NUMBER'] + "_S" + df['stint_id_num'].astype(str)
        df['lap_in_stint'] = df.groupby('stint_id').cumcount() + 1
        
        # <<< NEW: Use manufacturer-specific fuel loads if available >>>
        if self.fuel_capacities:
            print("Applying manufacturer-specific fuel capacities.")
            # Map manufacturer to its fuel capacity, use default if not found
            df['max_fuel_load'] = df['MANUFACTURER'].str.lower().str.strip().map(self.fuel_capacities)
            df['max_fuel_load'].fillna(self.config['max_fuel_load_kg'], inplace=True)
        else:
            print("Using default fuel capacity for all cars.")
            df['max_fuel_load'] = self.config['max_fuel_load_kg']

        df['fuel_load_kg'] = df['max_fuel_load'] - ((df['lap_in_stint'] - 1) * self.config['fuel_burn_rate_kg_per_lap'])
        # <<< END NEW >>>

        df['fuel_correction_s'] = df['fuel_load_kg'] * self.config['fuel_weight_penalty_s_per_kg']; df['LAP_TIME_FUEL_CORRECTED_SEC'] = df['LAP_TIME_SEC'] + df['fuel_correction_s']
        self.df = df; print("Categorizing laps for traffic..."); self._categorize_laps_for_traffic(); print("--- Master Preprocessing Finished ---")
        
    def _categorize_laps_for_traffic(self):
        if 'FINISH_TIME_DT' not in self.df.columns or self.df['FINISH_TIME_DT'].isna().all():
            print("WARNING: 'HOUR' column missing or invalid. Skipping traffic analysis."); self.df['LAP_CATEGORY'] = 'UNKNOWN'; return
        
        clean_laps = self.df[(self.df['FLAG_AT_FL'] == 'GF') & (~self.df['is_pit_stop_lap']) & (self.df['lap_in_stint'] > 1) & (self.df['LAP_TIME_SEC'].notna())].copy()
        potentials = clean_laps.groupby('DRIVER_NAME')['LAP_TIME_SEC'].quantile(self.config['driver_potential_percentile']).to_dict()
        self.df['DRIVER_POTENTIAL_LAP_SEC'] = self.df['DRIVER_NAME'].map(potentials)
        self.df['is_compromised'] = self.df['LAP_TIME_SEC'] > (self.df['DRIVER_POTENTIAL_LAP_SEC'] + self.config['traffic_compromise_threshold_s'])
        
        df_sorted = self.df.sort_values(by='FINISH_TIME_DT').copy()
        df_sorted['time_delta_to_prev_car_s'] = (df_sorted['FINISH_TIME_DT'] - df_sorted['FINISH_TIME_DT'].shift(1)).dt.total_seconds()
        df_sorted['prev_car_class'] = df_sorted['CLASS'].shift(1)
        df_sorted['prev_car_number'] = df_sorted['NUMBER'].shift(1)
        
        conditions = [
            (df_sorted['is_compromised']) & (df_sorted['time_delta_to_prev_car_s'] <= self.config['traffic_proximity_threshold_s']) & (df_sorted['CLASS'] == df_sorted['prev_car_class']) & (df_sorted['NUMBER'] != df_sorted['prev_car_number']),
            (df_sorted['is_compromised']) & (df_sorted['time_delta_to_prev_car_s'] <= self.config['traffic_proximity_threshold_s']) & (df_sorted['CLASS'] != df_sorted['prev_car_class'])
        ]
        choices = ['TRAFFIC_IN_CLASS', 'TRAFFIC_OUT_OF_CLASS']
        df_sorted['LAP_CATEGORY'] = np.select(conditions, choices, default='NORMAL')
        
        self.df = self.df.merge(df_sorted[['NUMBER', 'LAP_NUMBER', 'LAP_CATEGORY']], on=['NUMBER', 'LAP_NUMBER'], how='left')

    def _get_row_at_min_time(self, df_group, time_col_sec, original_time_col_str=None, required_fields=None):
        if required_fields is None: required_fields = ['DRIVER_NAME', 'LAP_NUMBER', 'TEAM', 'NUMBER']
        default_data = {time_col_sec: np.nan, **{f: None for f in required_fields}}
        if original_time_col_str: default_data[original_time_col_str] = None
        valid_df = df_group.dropna(subset=[time_col_sec])
        if valid_df.empty: return pd.Series({**{c: None for c in df_group.columns}, **default_data})
        return df_group.loc[valid_df[time_col_sec].idxmin()]

    def _load_pit_data(self, pit_json_filepath):
        with open(pit_json_filepath, 'r') as fp: pit_json = json.load(fp)
        pit_entries = []
        for car_entry in pit_json.get('pit_stop_analysis', []):
            car_no = str(car_entry.get('number', '')).strip()
            for stop in car_entry.get('pit_stops', []):
                in_dt = pd.to_datetime(stop.get('in_time'), format='%H:%M:%S.%f', errors='coerce'); out_dt = pd.to_datetime(stop.get('out_time'), format='%H:%M:%S.%f', errors='coerce')
                pit_entries.append({'NUMBER': car_no, 'pit_number': stop.get('pit_number'), 'pit_time_sec': self._parse_time_to_seconds(stop.get('pit_time')), 'in_time_str': stop.get('in_time'), 'out_time_str': stop.get('out_time'), 'in_dt': in_dt, 'out_dt': out_dt, 'driver_change': stop.get('in_driver_number') != stop.get('out_driver_number') if stop.get('in_driver_number') and stop.get('out_driver_number') else np.nan})
        if not pit_entries: raise ValueError("Pit JSON file contained no pit stop information.")
        return pd.DataFrame(pit_entries)
        
    def _get_json_pit_data_for_car(self, car_number):
        if self.pit_data_df is None or self.pit_data_df.empty: return pd.DataFrame()
        return self.pit_data_df[self.pit_data_df['NUMBER'] == str(car_number)].copy()

    def _analyze_pit_stops_json(self, car_df):
        car_no = car_df['NUMBER'].iloc[0]; pit_df = self._get_json_pit_data_for_car(car_no)
        if pit_df.empty: return self._analyze_pit_stops_original(car_df)
        total_time = pit_df['pit_time_sec'].sum(); num_stops = len(pit_df)
        total_minus_travel = pit_df['pit_time_sec'].apply(lambda x: max(0, x - self.config['pit_lane_delta_s'])).sum()
        return {'total_pit_stops': num_stops, 'total_pit_time_sec': total_time, 'total_pit_time_minus_travel_sec': total_minus_travel, 'total_pit_time_formatted': self._format_seconds_to_ms_str(total_time), 'total_pit_time_minus_travel_formatted': self._format_seconds_to_ms_str(total_minus_travel), 'average_pit_time_formatted': self._format_seconds_to_ms_str(total_time / num_stops if num_stops > 0 else np.nan)}

    def _get_pit_stop_details_json(self, car_df, driver_changes):
        car_no = car_df['NUMBER'].iloc[0]; pit_df = self._get_json_pit_data_for_car(car_no)
        if pit_df.empty: return self._get_pit_stop_details_original(car_df, driver_changes)
        change_laps = {item['lap_number'] for item in driver_changes.get('change_details', [])}; details = []
        for _, row in pit_df.sort_values('pit_number').iterrows():
            lap_entry = None
            if pd.notna(row['in_dt']) and car_df['FINISH_TIME_DT'].notna().any():
                prev_laps = car_df[car_df['FINISH_TIME_DT'] <= row['in_dt']]
                if not prev_laps.empty: lap_entry = int(prev_laps['LAP_NUMBER'].iloc[-1])
                else:
                    next_laps = car_df[car_df['FINISH_TIME_DT'] > row['in_dt']]
                    if not next_laps.empty: lap_entry = int(next_laps['LAP_NUMBER'].iloc[0])
            stationary = row['pit_time_sec'] - self.config['pit_lane_delta_s'] if pd.notna(row['pit_time_sec']) else np.nan
            details.append({'stop_number': int(row['pit_number']) if pd.notna(row['pit_number']) else None, 'lap_number_entry': lap_entry, 'total_pit_lane_time': self._format_seconds_to_ms_str(row['pit_time_sec']), 'stationary_time': self._format_seconds_to_ms_str(stationary), 'driver_change': bool(row['driver_change']) if 'driver_change' in row and pd.notna(row['driver_change']) else (lap_entry is not None and (lap_entry in change_laps or (lap_entry + 1) in change_laps))})
        return details

    def get_fastest_by_car_number(self):
        results = [];
        for car_no, group in self.df.groupby('NUMBER'):
            fastest_lap_row = self._get_row_at_min_time(group, 'LAP_TIME_SEC', 'LAP_TIME');
            if pd.isna(fastest_lap_row['LAP_TIME_SEC']): continue 
            best_s1_row = self._get_row_at_min_time(group, 'S1_SEC', 'S1'); best_s2_row = self._get_row_at_min_time(group, 'S2_SEC', 'S2'); best_s3_row = self._get_row_at_min_time(group, 'S3_SEC', 'S3')
            optimal_lap_time_sec = np.nan
            if pd.notna(best_s1_row.get('S1_SEC')) and pd.notna(best_s2_row.get('S2_SEC')) and pd.notna(best_s3_row.get('S3_SEC')):
                optimal_lap_time_sec = best_s1_row['S1_SEC'] + best_s2_row['S2_SEC'] + best_s3_row['S3_SEC']
            results.append({"car_number": car_no, "fastest_lap": {"time": fastest_lap_row.get('LAP_TIME'), "driver_name": fastest_lap_row.get('DRIVER_NAME'), "lap_number": fastest_lap_row.get('LAP_NUMBER')}, "best_s1": {"time": best_s1_row.get('S1'), "driver_name": best_s1_row.get('DRIVER_NAME'), "lap_number": best_s1_row.get('LAP_NUMBER')}, "best_s2": {"time": best_s2_row.get('S2'), "driver_name": best_s2_row.get('DRIVER_NAME'), "lap_number": best_s2_row.get('LAP_NUMBER')}, "best_s3": {"time": best_s3_row.get('S3'), "driver_name": best_s3_row.get('DRIVER_NAME'), "lap_number": best_s3_row.get('LAP_NUMBER')}, "optimal_lap_time": self._format_seconds_to_ms_str(optimal_lap_time_sec)})
        return results

    def get_fastest_by_manufacturer(self):
        results = [];
        for manufacturer, group in self.df.groupby('MANUFACTURER'):
            if not manufacturer: continue
            fastest_lap_row = self._get_row_at_min_time(group, 'LAP_TIME_SEC', 'LAP_TIME');
            if pd.isna(fastest_lap_row['LAP_TIME_SEC']): continue
            best_s1_row = self._get_row_at_min_time(group, 'S1_SEC', 'S1'); best_s2_row = self._get_row_at_min_time(group, 'S2_SEC', 'S2'); best_s3_row = self._get_row_at_min_time(group, 'S3_SEC', 'S3')
            optimal_lap_time_sec = np.nan
            if pd.notna(best_s1_row.get('S1_SEC')) and pd.notna(best_s2_row.get('S2_SEC')) and pd.notna(best_s3_row.get('S3_SEC')):
                optimal_lap_time_sec = best_s1_row['S1_SEC'] + best_s2_row['S2_SEC'] + best_s3_row['S3_SEC']
            results.append({"manufacturer": manufacturer, "fastest_lap": {"time": fastest_lap_row.get('LAP_TIME'), "driver_name": fastest_lap_row.get('DRIVER_NAME'), "team": fastest_lap_row.get('TEAM'), "car_number": fastest_lap_row.get('NUMBER'), "lap_number": fastest_lap_row.get('LAP_NUMBER')}, "best_s1": {"time": best_s1_row.get('S1'), "driver_name": best_s1_row.get('DRIVER_NAME'), "team": best_s1_row.get('TEAM'), "car_number": best_s1_row.get('NUMBER'), "lap_number": best_s1_row.get('LAP_NUMBER')}, "best_s2": {"time": best_s2_row.get('S2'), "driver_name": best_s2_row.get('DRIVER_NAME'), "team": best_s2_row.get('TEAM'), "car_number": best_s2_row.get('NUMBER'), "lap_number": best_s2_row.get('LAP_NUMBER')}, "best_s3": {"time": best_s3_row.get('S3'), "driver_name": best_s3_row.get('DRIVER_NAME'), "team": best_s3_row.get('TEAM'), "car_number": best_s3_row.get('NUMBER'), "lap_number": best_s3_row.get('LAP_NUMBER')}, "optimal_lap_time": self._format_seconds_to_ms_str(optimal_lap_time_sec)})
        return results

    def get_longest_stints_by_manufacturer(self):
        results = []; stint_df = self.df
        if 'stint_id' not in stint_df.columns or stint_df['stint_id'].isna().all(): return results
        stints_data = []
        for stint_id_val, laps_this_stint_df in stint_df.groupby('stint_id'):
            if laps_this_stint_df.empty: continue
            racing_laps_for_stint_df = laps_this_stint_df.copy()
            if laps_this_stint_df.iloc[-1]['is_pit_stop_lap']:
                racing_laps_for_stint_df = laps_this_stint_df.iloc[:-1] if len(laps_this_stint_df) > 1 else pd.DataFrame(columns=laps_this_stint_df.columns)
            if racing_laps_for_stint_df.empty:
                continue

            # --- NEW: find the longest consecutive segment of valid green-flag laps ---
            stint_sorted = racing_laps_for_stint_df.sort_values('LAP_NUMBER').copy()
            stint_sorted['is_green_valid'] = (stint_sorted['FLAG_AT_FL'] == 'GF') & (stint_sorted['LAP_TIME_SEC'].notna())
            stint_sorted['segment_id'] = (stint_sorted['is_green_valid'] != stint_sorted['is_green_valid'].shift()).cumsum()

            longest_seg_len = 0
            best_segment_df = pd.DataFrame()
            for seg_id, seg_df in stint_sorted.groupby('segment_id'):
                if not seg_df.iloc[0]['is_green_valid']:
                    continue  # skip non-green segments
                seg_len = len(seg_df)
                if seg_len > longest_seg_len:
                    longest_seg_len = seg_len
                    best_segment_df = seg_df.copy()

            if longest_seg_len > 0:
                stints_data.append({
                    'stint_id': stint_id_val,
                    'manufacturer': best_segment_df['MANUFACTURER'].iloc[0],
                    'num_green_laps': longest_seg_len,
                    'laps_data_for_metrics': best_segment_df,
                })
        if not stints_data: return results
        all_valid_green_stints_df = pd.DataFrame(stints_data)
        for manufacturer_name, manu_stints_df in all_valid_green_stints_df.groupby('manufacturer'): 
            if manu_stints_df.empty or manufacturer_name == "": continue
            longest_stint_len = manu_stints_df['num_green_laps'].max()
            if longest_stint_len == 0 or pd.isna(longest_stint_len): continue
            candidate_stints_for_manu_df = manu_stints_df[manu_stints_df['num_green_laps'] == longest_stint_len]
            chosen_stint_laps_df, fastest_lap_in_chosen_stint_sec = None, float('inf')
            for _, candidate_row in candidate_stints_for_manu_df.iterrows():
                current_stint_green_laps = candidate_row['laps_data_for_metrics']
                if current_stint_green_laps.empty: continue
                min_lap_time_sec_this_stint = current_stint_green_laps['LAP_TIME_SEC'].min() 
                if pd.notna(min_lap_time_sec_this_stint) and min_lap_time_sec_this_stint < fastest_lap_in_chosen_stint_sec:
                    fastest_lap_in_chosen_stint_sec = min_lap_time_sec_this_stint; chosen_stint_laps_df = current_stint_green_laps.copy() 
            if chosen_stint_laps_df is None or chosen_stint_laps_df.empty: continue
            best_lap_row_stint = self._get_row_at_min_time(chosen_stint_laps_df, 'LAP_TIME_SEC', 'LAP_TIME')
            results.append({"manufacturer": manufacturer_name, "longest_green_stint_laps": len(chosen_stint_laps_df), "stint_details": {"car_number": best_lap_row_stint.get('NUMBER'), "driver_at_best_lap": best_lap_row_stint.get('DRIVER_NAME'), "stint_id_debug": chosen_stint_laps_df['stint_id'].iloc[0], "start_lap_number_race": int(chosen_stint_laps_df['LAP_NUMBER'].min()), "end_lap_number_race": int(chosen_stint_laps_df['LAP_NUMBER'].max()), "best_lap_time": best_lap_row_stint.get('LAP_TIME'), "best_lap_position_in_stint": int(best_lap_row_stint.get('lap_in_stint')), "best_s1_in_stint": self._get_row_at_min_time(chosen_stint_laps_df, 'S1_SEC')['S1'], "best_s2_in_stint": self._get_row_at_min_time(chosen_stint_laps_df, 'S2_SEC')['S2'], "best_s3_in_stint": self._get_row_at_min_time(chosen_stint_laps_df, 'S3_SEC')['S3'], "average_lap_time_in_stint": self._format_seconds_to_ms_str(chosen_stint_laps_df['LAP_TIME_SEC'].mean())}})
        return results

    def get_driver_deltas_by_car(self):
        results = [];
        for car_no, car_laps_df in self.df.groupby('NUMBER'):
            driver_performances = []
            for driver_num_val, driver_laps_df in car_laps_df.groupby('DRIVER_NUMBER'):
                if not driver_num_val or pd.isna(driver_num_val) or str(driver_num_val).lower() == 'nan' or driver_laps_df.empty: continue
                driver_name = driver_laps_df['DRIVER_NAME'].mode()[0] if not driver_laps_df['DRIVER_NAME'].mode().empty else f"Driver_{driver_num_val}"
                best_lap_row = self._get_row_at_min_time(driver_laps_df, 'LAP_TIME_SEC', 'LAP_TIME')
                if pd.isna(best_lap_row['LAP_TIME_SEC']): continue 
                driver_performances.append({"driver_name": driver_name, "driver_number": driver_num_val, "best_lap_time": best_lap_row.get('LAP_TIME'), "best_lap_time_sec": best_lap_row.get('LAP_TIME_SEC'), "best_s1": self._get_row_at_min_time(driver_laps_df, 'S1_SEC', 'S1').get('S1'), "best_s1_sec": self._get_row_at_min_time(driver_laps_df, 'S1_SEC', 'S1').get('S1_SEC'), "best_s2": self._get_row_at_min_time(driver_laps_df, 'S2_SEC', 'S2').get('S2'), "best_s2_sec": self._get_row_at_min_time(driver_laps_df, 'S2_SEC', 'S2').get('S2_SEC'), "best_s3": self._get_row_at_min_time(driver_laps_df, 'S3_SEC', 'S3').get('S3'), "best_s3_sec": self._get_row_at_min_time(driver_laps_df, 'S3_SEC', 'S3').get('S3_SEC')})
            if len(driver_performances) < 2: continue
            driver_performances.sort(key=lambda x: x.get('best_lap_time_sec', float('inf')))
            fastest_driver_perf = driver_performances[0]; deltas_to_fastest, lap_time_deltas_for_avg_sec = [], []
            for other_driver_perf in driver_performances[1:]:
                lap_delta = other_driver_perf['best_lap_time_sec'] - fastest_driver_perf['best_lap_time_sec']
                s1_delta = other_driver_perf.get('best_s1_sec', np.nan) - fastest_driver_perf.get('best_s1_sec', np.nan); s2_delta = other_driver_perf.get('best_s2_sec', np.nan) - fastest_driver_perf.get('best_s2_sec', np.nan); s3_delta = other_driver_perf.get('best_s3_sec', np.nan) - fastest_driver_perf.get('best_s3_sec', np.nan)
                deltas_to_fastest.append({"driver_name": other_driver_perf['driver_name'], "lap_time_delta": self._format_seconds_to_ms_str(lap_delta), "s1_delta": self._format_seconds_to_ms_str(s1_delta), "s2_delta": self._format_seconds_to_ms_str(s2_delta), "s3_delta": self._format_seconds_to_ms_str(s3_delta)})
                lap_time_deltas_for_avg_sec.append(lap_delta)
            avg_lap_time_delta_sec = np.mean(lap_time_deltas_for_avg_sec) if lap_time_deltas_for_avg_sec else np.nan
            results.append({"car_number": car_no, "drivers_performance": driver_performances, "fastest_driver_name": fastest_driver_perf['driver_name'], "deltas_to_fastest": deltas_to_fastest, "average_lap_time_delta_for_car": self._format_seconds_to_ms_str(avg_lap_time_delta_sec)})
        return results

    def get_earliest_fastest_lap_drivers(self, top_n: int = 5):
        """
        Determine which drivers achieved their car's fastest lap earliest into a stint.

        This metric helps highlight drivers who were able to deliver peak pace
        while the car was still heavy with fuel, i.e. closer to the start of the
        stint.  Drivers are ranked by the lap position within the stint
        (``lap_in_stint``) at which the overall fastest lap for their car was
        recorded.  Ties are broken using the absolute lap-time value.

        Parameters
        ----------
        top_n : int, default 5
            Number of top ranked drivers to return.

        Returns
        -------
        list[dict]
            A list of dictionaries, each containing the ranked driver and
            contextual information useful for reporting.
        """
        results = []
        # Group by car to identify each car's single fastest lap and its metadata
        for car_no, car_df in self.df.groupby('NUMBER'):
            fastest_row = self._get_row_at_min_time(car_df, 'LAP_TIME_SEC', 'LAP_TIME')
            # Require valid lap_in_stint information
            if pd.isna(fastest_row['LAP_TIME_SEC']) or pd.isna(fastest_row.get('lap_in_stint')):
                continue
            results.append({
                'driver_name': fastest_row.get('DRIVER_NAME'),
                'car_number': car_no,
                'team': fastest_row.get('TEAM'),
                'lap_in_stint': int(fastest_row.get('lap_in_stint')),
                'stint_id': fastest_row.get('stint_id'),
                'lap_number_race': int(fastest_row.get('LAP_NUMBER')),
                'fastest_lap_time': fastest_row.get('LAP_TIME')
            })
        # Sort by earliest lap in stint, then by the absolute lap time as tiebreaker
        results.sort(key=lambda x: (x['lap_in_stint'], self._parse_time_to_seconds(x['fastest_lap_time'])))
        ranked = [{
            'rank': i + 1,
            **res
        } for i, res in enumerate(results[:top_n])]
        return ranked

    def get_manufacturer_driver_pace_gap(self):
        lap_gaps, s1_gaps, s2_gaps, s3_gaps = [], [], [], []
        for manu_name, manu_df in self.df.groupby('MANUFACTURER'):
            if not manu_name or manu_df.empty or manu_df['DRIVER_NAME'].nunique() < 2:
                continue
            driver_bests = []
            for driver_name, driver_df in manu_df.groupby('DRIVER_NAME'):
                if not driver_name or driver_df.empty: continue
                best_lap_row = self._get_row_at_min_time(driver_df, 'LAP_TIME_SEC', 'LAP_TIME')
                best_s1_row = self._get_row_at_min_time(driver_df, 'S1_SEC', 'S1')
                best_s2_row = self._get_row_at_min_time(driver_df, 'S2_SEC', 'S2')
                best_s3_row = self._get_row_at_min_time(driver_df, 'S3_SEC', 'S3')
                driver_bests.append({'driver_name': driver_name, 'lap_time_sec': best_lap_row['LAP_TIME_SEC'], 'LAP_TIME': best_lap_row['LAP_TIME'], 's1_sec': best_s1_row['S1_SEC'], 'S1': best_s1_row['S1'], 's2_sec': best_s2_row['S2_SEC'], 'S2': best_s2_row['S2'], 's3_sec': best_s3_row['S3_SEC'], 'S3': best_s3_row['S3']})
            driver_bests_df = pd.DataFrame(driver_bests)
            if (lap_result := self._process_metric_gap(manu_name, driver_bests_df, 'lap_time_sec', 'LAP_TIME')): lap_gaps.append(lap_result)
            if (s1_result := self._process_metric_gap(manu_name, driver_bests_df, 's1_sec', 'S1')): s1_gaps.append(s1_result)
            if (s2_result := self._process_metric_gap(manu_name, driver_bests_df, 's2_sec', 'S2')): s2_gaps.append(s2_result)
            if (s3_result := self._process_metric_gap(manu_name, driver_bests_df, 's3_sec', 'S3')): s3_gaps.append(s3_result)
        lap_gaps.sort(key=lambda x: x['gap_seconds'], reverse=True)
        s1_gaps.sort(key=lambda x: x['gap_seconds'], reverse=True)
        s2_gaps.sort(key=lambda x: x['gap_seconds'], reverse=True)
        s3_gaps.sort(key=lambda x: x['gap_seconds'], reverse=True)
        return {'lap_time_gap_ranking': [{'rank': i + 1, **res} for i, res in enumerate(lap_gaps)], 's1_gap_ranking': [{'rank': i + 1, **res} for i, res in enumerate(s1_gaps)], 's2_gap_ranking': [{'rank': i + 1, **res} for i, res in enumerate(s2_gaps)], 's3_gap_ranking': [{'rank': i + 1, **res} for i, res in enumerate(s3_gaps)]}
        
    def get_race_strategy_by_car(self):
        results = []
        for car_no, car_df in self.df.groupby('NUMBER'):
            if car_df.empty: continue
            driver_changes = self._analyze_driver_changes_original(car_df)
            if self.pit_data_df is not None and not self._get_json_pit_data_for_car(car_no).empty:
                pit_analysis = self._analyze_pit_stops_json(car_df)
                pit_stop_details = self._get_pit_stop_details_json(car_df, driver_changes)
            else:
                pit_analysis = self._analyze_pit_stops_original(car_df)
                pit_stop_details = self._get_pit_stop_details_original(car_df, driver_changes)
            stint_analysis = self._analyze_stints_original(car_df)
            results.append({"car_number": car_no, "total_pit_time": pit_analysis.get('total_pit_time_formatted'), "average_pit_time": pit_analysis.get('average_pit_time_formatted'), "total_pit_time_minus_travel": pit_analysis.get('total_pit_time_minus_travel_formatted'), "total_pit_stops": pit_analysis.get('total_pit_stops', 0), "total_driver_changes": driver_changes.get('total_driver_changes', 0), "driver_change_details": driver_changes.get('change_details', []), "stints": stint_analysis, "pit_stop_details": pit_stop_details})
        return results

    def _process_metric_gap(self, manu_name, perfs_df, metric_sec, metric_str):
        valid_perfs = perfs_df.dropna(subset=[metric_sec]).copy()
        if len(valid_perfs) < 2: return None
        fastest_idx, slowest_idx = valid_perfs[metric_sec].idxmin(), valid_perfs[metric_sec].idxmax()
        fastest_perf, slowest_perf = valid_perfs.loc[fastest_idx], valid_perfs.loc[slowest_idx]
        if fastest_perf['driver_name'] == slowest_perf['driver_name']: return None
        gap_sec = slowest_perf[metric_sec] - fastest_perf[metric_sec]
        if gap_sec <= 0: return None
        return {"manufacturer": manu_name, "gap_seconds": gap_sec, "gap_formatted": self._format_seconds_to_ms_str(gap_sec), "fastest_driver": {"name": fastest_perf['driver_name'], "time": fastest_perf[metric_str]}, "slowest_driver": {"name": slowest_perf['driver_name'], "time": slowest_perf[metric_str]}}
        
    def _get_pit_stop_details_original(self, car_df, driver_changes):
        pit_stop_list = []; driver_change_laps = {item['lap_number'] for item in driver_changes.get('change_details', [])}
        for stop_number, (_, stop_df) in enumerate(car_df[car_df['is_pit_stop_lap']].groupby('stint_id_num'), 1):
            entry_lap = int(stop_df['LAP_NUMBER'].min()); total_time_sec = stop_df['LAP_TIME_SEC'].sum()
            pit_stop_list.append({"stop_number": stop_number, "lap_number_entry": entry_lap, "total_pit_lane_time": self._format_seconds_to_ms_str(total_time_sec), "stationary_time": self._format_seconds_to_ms_str(total_time_sec - self.config['pit_lane_delta_s']), "driver_change": entry_lap in driver_change_laps})
        return pit_stop_list

    def _analyze_stints_original(self, car_df):
        stint_results = []
        for stint_num, stint_df in car_df.groupby('stint_id_num'):
            if stint_df.empty: continue
            racing_laps = stint_df[~stint_df['is_pit_stop_lap']];
            if racing_laps.empty: continue
            flag_stats = self._categorize_laps_by_flag(racing_laps); total_stint_time_sec = racing_laps['LAP_TIME_SEC'].sum() if racing_laps['LAP_TIME_SEC'].notna().any() else 0
            green_laps = racing_laps[racing_laps['FLAG_AT_FL'] == 'GF'].dropna(subset=['LAP_TIME_SEC']); best_5_lap_avg_sec = np.nan
            if len(green_laps) >= 5: best_5_lap_avg_sec = green_laps['LAP_TIME_SEC'].nsmallest(5).mean()
            traffic_counts = racing_laps['LAP_CATEGORY'].value_counts().to_dict()
            # Collect per-lap data for visualizer (lap number within stint and fuel-corrected lap time)
            laps_data = [
                {
                    'lap_in_stint': int(row['lap_in_stint']),
                    # Use None instead of NaN to keep JSON serializable & let visualizer filter invalid laps
                    'LAP_TIME_FUEL_CORRECTED_SEC': (row['LAP_TIME_FUEL_CORRECTED_SEC'] if pd.notna(row['LAP_TIME_FUEL_CORRECTED_SEC']) else None),
                }
                for _, row in racing_laps[['lap_in_stint', 'LAP_TIME_FUEL_CORRECTED_SEC']].iterrows()
            ]

            stint_results.append({
                'stint_number': int(stint_num),
                'total_laps': len(racing_laps),
                'total_time': self._format_seconds_to_ms_str(total_stint_time_sec),
                **flag_stats,
                'lap_range': f"{int(racing_laps['LAP_NUMBER'].min())}-{int(racing_laps['LAP_NUMBER'].max())}",
                'best_5_lap_avg': self._format_seconds_to_ms_str(best_5_lap_avg_sec),
                'traffic_in_class_laps': traffic_counts.get('TRAFFIC_IN_CLASS', 0),
                'traffic_out_of_class_laps': traffic_counts.get('TRAFFIC_OUT_OF_CLASS', 0),
                'laps': laps_data,
            })
        return stint_results
    
    def _calculate_baseline_travel_time(self, car_df):
        green_laps = car_df[(car_df['FLAG_AT_FL'] == 'GF') & (car_df['LAP_TIME_SEC'].notna()) & (car_df['LAP_TIME_SEC'] > 0)]
        if green_laps.empty: return 90.0
        return green_laps['LAP_TIME_SEC'].quantile(0.1)

    def _analyze_pit_stops_original(self, car_df):
        baseline_travel_time_sec = self._calculate_baseline_travel_time(car_df); pit_analysis = {'total_pit_stops': 0, 'total_pit_time_sec': 0.0, 'total_pit_time_minus_travel_sec': 0.0}
        pit_sequences = car_df[car_df['is_pit_stop_lap']].groupby('stint_id_num'); pit_analysis['total_pit_stops'] = pit_sequences.ngroups
        for _, pit_seq_df in pit_sequences:
            pit_time_sec = pit_seq_df['LAP_TIME_SEC'].sum()
            if pd.notna(pit_time_sec) and pit_time_sec > 0:
                pit_analysis['total_pit_time_sec'] += pit_time_sec; pit_analysis['total_pit_time_minus_travel_sec'] += max(0, pit_time_sec - (baseline_travel_time_sec * len(pit_seq_df)))
        pit_analysis['total_pit_time_formatted'] = self._format_seconds_to_ms_str(pit_analysis['total_pit_time_sec']); pit_analysis['total_pit_time_minus_travel_formatted'] = self._format_seconds_to_ms_str(pit_analysis['total_pit_time_minus_travel_sec'])
        pit_analysis['average_pit_time_formatted'] = self._format_seconds_to_ms_str(pit_analysis['total_pit_time_sec'] / pit_analysis['total_pit_stops'] if pit_analysis['total_pit_stops'] > 0 else np.nan)
        return pit_analysis

    def _analyze_driver_changes_original(self, car_df):
        driver_analysis = {'total_driver_changes': 0, 'change_details': []}
        car_df_sorted = car_df.sort_values('LAP_NUMBER'); car_df_sorted['driver_changed'] = car_df_sorted['DRIVER_NUMBER'] != car_df_sorted['DRIVER_NUMBER'].shift(1)
        driver_changes = car_df_sorted[car_df_sorted['driver_changed']]
        if len(driver_changes) > 1:
            driver_analysis['total_driver_changes'] = len(driver_changes) - 1
            for _, change_row in driver_changes.iloc[1:].iterrows():
                prev_lap = car_df_sorted[car_df_sorted['LAP_NUMBER'] < change_row['LAP_NUMBER']]; prev_driver = prev_lap['DRIVER_NAME'].iloc[-1] if not prev_lap.empty else "Unknown"
                driver_analysis['change_details'].append({'lap_number': int(change_row['LAP_NUMBER']), 'from_driver': prev_driver, 'to_driver': change_row['DRIVER_NAME']})
        return driver_analysis

    def _categorize_laps_by_flag(self, laps_df):
        flag_stats = {'green_laps': 0, 'yellow_laps': 0, 'red_laps': 0, 'other_laps': 0}
        green_laps = laps_df[laps_df['FLAG_AT_FL'] == 'GF']; yellow_laps = laps_df[laps_df['FLAG_AT_FL'] == 'FCY']; red_laps = laps_df[laps_df['FLAG_AT_FL'] == 'RF']; other_laps = laps_df[~laps_df['FLAG_AT_FL'].isin(['GF', 'FCY', 'RF'])]
        flag_stats.update({'green_laps': len(green_laps), 'yellow_laps': len(yellow_laps), 'red_laps': len(red_laps), 'other_laps': len(other_laps)})
        if not green_laps.empty and green_laps['LAP_TIME_SEC'].notna().any(): flag_stats['avg_green_time_formatted'] = self._format_seconds_to_ms_str(green_laps['LAP_TIME_SEC'].mean()); flag_stats['best_green_time_formatted'] = self._format_seconds_to_ms_str(green_laps['LAP_TIME_SEC'].min())
        if not yellow_laps.empty and yellow_laps['LAP_TIME_SEC'].notna().any(): flag_stats['avg_yellow_time_formatted'] = self._format_seconds_to_ms_str(yellow_laps['LAP_TIME_SEC'].mean())
        return flag_stats

    def get_enhanced_strategy_analysis(self):
        results = [];
        for car_no, car_df in self.df.groupby('NUMBER'):
            if car_df.empty: continue
            clean_laps = car_df[(car_df['FLAG_AT_FL'] == 'GF') & (~car_df['is_pit_stop_lap']) & (car_df['LAP_CATEGORY'] == 'NORMAL')].dropna(subset=['LAP_TIME_FUEL_CORRECTED_SEC'])
            avg_green_pace_sec = clean_laps['LAP_TIME_FUEL_CORRECTED_SEC'].mean(); consistency = clean_laps['LAP_TIME_FUEL_CORRECTED_SEC'].std()
            pit_analysis = self._analyze_pit_stops(car_df) # This is a helper, not the main one
            degradation_model = self._calculate_advanced_degradation(car_df)
            results.append({"car_number": car_no, "team": car_df['TEAM'].mode()[0] if not car_df['TEAM'].mode().empty else "N/A", "manufacturer": car_df['MANUFACTURER'].mode()[0] if not car_df['MANUFACTURER'].mode().empty else "N/A", "avg_green_pace_fuel_corrected": self._format_seconds_to_ms_str(avg_green_pace_sec), "race_pace_consistency_stdev": round(consistency, 3) if pd.notna(consistency) else None, "avg_pit_stationary_time": self._format_seconds_to_ms_str(pit_analysis.get('average_stationary_time_sec')), "tire_degradation_model": degradation_model})
        return results

    def _calculate_advanced_degradation(self, car_df):
        stint_models, total_clean_laps_used = [], 0
        for _, stint_df in car_df.groupby('stint_id'):
            clean_laps = stint_df[(stint_df['FLAG_AT_FL'] == 'GF') & (~stint_df['is_pit_stop_lap']) & (stint_df['lap_in_stint'] > 1) & (stint_df['LAP_CATEGORY'] == 'NORMAL') & (stint_df['LAP_TIME_FUEL_CORRECTED_SEC'].notna())].copy()
            if len(clean_laps) >= self.config['min_laps_for_deg_model']:
                stint_models.append(np.polyfit(clean_laps['lap_in_stint'], clean_laps['LAP_TIME_FUEL_CORRECTED_SEC'], 2)); total_clean_laps_used += len(clean_laps)
        if not stint_models: return {"deg_coeff_a": None, "deg_coeff_b": None, "deg_coeff_c": None, "fastest_lap_of_stint_predicted_at": None, "model_quality": "INSUFFICIENT_DATA", "total_clean_laps_used": 0}
        avg_coeffs = np.mean(stint_models, axis=0); a, b, c = avg_coeffs[0], avg_coeffs[1], avg_coeffs[2]
        quality = "GOOD" if total_clean_laps_used > 20 else "FAIR" if total_clean_laps_used > 10 else "POOR"
        predicted_best_lap = -b / (2 * a) if a != 0 else np.nan
        return {"deg_coeff_a": round(a, 6), "deg_coeff_b": round(b, 6), "deg_coeff_c": round(c, 6), "fastest_lap_of_stint_predicted_at": round(predicted_best_lap, 1) if pd.notna(predicted_best_lap) else None, "model_quality": quality, "total_clean_laps_used": total_clean_laps_used}

    def _analyze_pit_stops(self, car_df):
        pit_stop_laps = car_df[car_df['is_pit_stop_lap']].copy()
        if pit_stop_laps.empty: return {'total_pit_stops': 0, 'average_stationary_time_sec': np.nan}
        num_stops = pit_stop_laps['stint_id_num'].nunique(); total_pit_time = pit_stop_laps['LAP_TIME_SEC'].sum()
        total_stationary = total_pit_time - (num_stops * self.config['pit_lane_delta_s'])
        return {'total_pit_stops': num_stops, 'average_stationary_time_sec': total_stationary / num_stops if num_stops > 0 else np.nan}

    def get_traffic_management_analysis(self):
        driver_results = []
        traffic_laps_df = self.df[self.df['LAP_CATEGORY'].isin(['TRAFFIC_IN_CLASS', 'TRAFFIC_OUT_OF_CLASS'])].copy()
        if traffic_laps_df.empty: return []
        traffic_laps_df['time_lost_sec'] = traffic_laps_df['LAP_TIME_SEC'] - traffic_laps_df['DRIVER_POTENTIAL_LAP_SEC']
        for driver_name, group in traffic_laps_df.groupby('DRIVER_NAME'):
            if group.empty or group['DRIVER_POTENTIAL_LAP_SEC'].isna().all(): continue
            in_class = group[group['LAP_CATEGORY'] == 'TRAFFIC_IN_CLASS']; out_of_class = group[group['LAP_CATEGORY'] == 'TRAFFIC_OUT_OF_CLASS']
            driver_results.append({'driver_name': driver_name, 'car_number': group['NUMBER'].iloc[0], 'team': group['TEAM'].iloc[0], 'avg_time_lost_total_sec': group['time_lost_sec'].mean(), 'avg_time_lost_in_class_sec': in_class['time_lost_sec'].mean() if not in_class.empty else np.nan, 'avg_time_lost_out_of_class_sec': out_of_class['time_lost_sec'].mean() if not out_of_class.empty else np.nan, 'total_traffic_laps': len(group), 'in_class_traffic_laps': len(in_class), 'out_of_class_traffic_laps': len(out_of_class)})
        driver_results.sort(key=lambda x: x.get('avg_time_lost_total_sec', float('inf')))
        return [{'rank': i + 1, **res} for i, res in enumerate(driver_results)]

    def get_full_pit_cycle_analysis(self, race_strategy_data, enhanced_strategy_data):
        car_results = []
        team_map = {car['car_number']: car.get('team', 'N/A') for car in enhanced_strategy_data}
        for car_data in race_strategy_data:
            car_number = car_data.get('car_number'); team_name = team_map.get(car_number, 'N/A'); all_losses = []
            for pit_stop in car_data.get('pit_stop_details', []):
                lap_entry = pit_stop.get('lap_number_entry')
                if lap_entry is None: continue
                stationary = max(0, self._parse_time_to_seconds(pit_stop.get('stationary_time', '0')))
                in_lap = self.df[(self.df['NUMBER'] == car_number) & (self.df['LAP_NUMBER'] == lap_entry - 1)]
                in_loss = in_lap.iloc[0]['LAP_TIME_SEC'] - in_lap.iloc[0]['DRIVER_POTENTIAL_LAP_SEC'] if not in_lap.empty and pd.notna(in_lap.iloc[0]['LAP_TIME_SEC']) and pd.notna(in_lap.iloc[0]['DRIVER_POTENTIAL_LAP_SEC']) else 0.0
                out_lap_num = next((int(s['lap_range'].split('-')[0]) for s in car_data.get('stints', []) if int(s['lap_range'].split('-')[0]) > lap_entry), -1)
                out_loss = 0.0
                if out_lap_num != -1:
                    out_lap = self.df[(self.df['NUMBER'] == car_number) & (self.df['LAP_NUMBER'] == out_lap_num)]
                    if not out_lap.empty and pd.notna(out_lap.iloc[0]['LAP_TIME_SEC']) and pd.notna(out_lap.iloc[0]['DRIVER_POTENTIAL_LAP_SEC']):
                        out_loss = out_lap.iloc[0]['LAP_TIME_SEC'] - out_lap.iloc[0]['DRIVER_POTENTIAL_LAP_SEC']
                all_losses.append(in_loss + stationary + out_loss)
            if all_losses: car_results.append({'car_number': car_number, 'team': team_name, 'average_cycle_loss_sec': np.mean(all_losses), 'average_cycle_loss': np.mean(all_losses), 'number_of_stops_analyzed': len(all_losses)})
        car_results.sort(key=lambda x: x.get('average_cycle_loss_sec', float('inf')))
        return [{'rank': i + 1, **res} for i, res in enumerate(car_results)]

    def add_degradation_cliff_analysis(self, enhanced_strategy_data):
        stint_len = 35
        for car_analysis in enhanced_strategy_data:
            model = car_analysis.get('tire_degradation_model', {})
            a, b, quality = model.get('deg_coeff_a'), model.get('deg_coeff_b'), model.get('model_quality')
            model.update({'end_of_stint_deg_rate_s_per_lap': None, 'predicted_final_5_laps_loss_s': None})
            if a is not None and b is not None and quality not in ["INSUFFICIENT_DATA", "POOR"]:
                deg_rate = (2 * a * stint_len) + b; model.update({'end_of_stint_deg_rate_s_per_lap': deg_rate, 'predicted_final_5_laps_loss_s': 5 * deg_rate})
        return enhanced_strategy_data

    def _get_social_media_highlights(self, fastest_by_car, race_strategy, enhanced_strategy):
        return {"metronome_award": self._get_metronome_award(), "metronome_award_longer": self._get_metronome_award_longer(), "perfect_lap_ranking": self._get_perfect_lap_ranking(fastest_by_car), "manufacturer_showdown": self._get_manufacturer_showdown(race_strategy, enhanced_strategy)}

    def _find_best_consistency_window(self, metric_sec, metric_str, window_size):
        # Find the most consistent window of laps for a given metric (lap time or sector time).
        green_laps = self.df[(self.df['FLAG_AT_FL'] == 'GF') & self.df[metric_sec].notna()].copy()
        if green_laps.empty:
            return {}
        
        best_std = float('inf')
        winner_data = {}
        
        for _, group in green_laps.groupby('stint_id'):
            if len(group) < window_size:
                continue
            
            # Calculate rolling standard deviation for the specified metric
            group['rolling_std'] = group[metric_sec].rolling(window=window_size).std()
            min_std_in_group = group['rolling_std'].min()

            if pd.notna(min_std_in_group) and min_std_in_group < best_std:
                best_std = min_std_in_group
                end_idx = group['rolling_std'].idxmin()
                # Ensure the start index is not negative
                start_idx = max(group.index.min(), end_idx - window_size + 1)
                window_df = group.loc[start_idx : end_idx]
                
                winner_data = {
                    'driver_name': window_df.iloc[-1]['DRIVER_NAME'],
                    'car_number': window_df.iloc[-1]['NUMBER'],
                    'team': window_df.iloc[-1]['TEAM'],
                    'consistency_stdev': best_std,
                    'start_lap': int(window_df.iloc[0]['LAP_NUMBER']),
                    'end_lap': int(window_df.iloc[-1]['LAP_NUMBER']),
                    'times': window_df[metric_str].tolist()
                }
        return winner_data

    def _get_metronome_award(self):
        window = self.config['min_laps_for_metronome']
        return {
            'lap_time': self._find_best_consistency_window('LAP_TIME_SEC', 'LAP_TIME', window),
            'sector_1': self._find_best_consistency_window('S1_SEC', 'S1', window),
            'sector_2': self._find_best_consistency_window('S2_SEC', 'S2', window),
            'sector_3': self._find_best_consistency_window('S3_SEC', 'S3', window)
        }

    def _get_metronome_award_longer(self):
        window = self.config['min_laps_for_metronome_longer']
        return {
            'lap_time': self._find_best_consistency_window('LAP_TIME_SEC', 'LAP_TIME', window),
            'sector_1': self._find_best_consistency_window('S1_SEC', 'S1', window),
            'sector_2': self._find_best_consistency_window('S2_SEC', 'S2', window),
            'sector_3': self._find_best_consistency_window('S3_SEC', 'S3', window)
        }

    def _get_perfect_lap_ranking(self, fastest_by_car):
        ranking = []
        for car in fastest_by_car:
            optimal = self._parse_time_to_seconds(car.get('optimal_lap_time')); fastest = self._parse_time_to_seconds(car.get('fastest_lap', {}).get('time'))
            if pd.notna(optimal) and pd.notna(fastest) and fastest > 0:
                pct = min(100.0, (optimal / fastest) * 100)
                ranking.append({'car_number': car['car_number'], 'driver_name': car.get('fastest_lap', {}).get('driver_name'), 'perfection_pct': pct, 'fastest_lap_time': car.get('fastest_lap', {}).get('time'), 'optimal_lap_time': car.get('optimal_lap_time')})
        ranking.sort(key=lambda x: x['perfection_pct'], reverse=True)
        return [{'rank': i + 1, **res} for i, res in enumerate(ranking)]

    def _get_manufacturer_showdown(self, race_strategy, enhanced_strategy):
        car_info = {c['car_number']: {'manufacturer': c.get('manufacturer'), 'team': c.get('team')} for c in enhanced_strategy}
        best_stints, min_laps = {}, self.config['min_laps_for_manu_showdown']
        for car in race_strategy:
            info = car_info.get(car['car_number'])
            if not info or not info.get('manufacturer'): continue
            for stint in car.get('stints', []):
                avg_time = self._parse_time_to_seconds(stint.get('avg_green_time_formatted')) 
                if pd.notna(avg_time) and stint.get('green_laps', 0) >= min_laps and (info['manufacturer'] not in best_stints or avg_time < best_stints[info['manufacturer']]['best_avg_stint_pace_sec']):
                    best_stints[info['manufacturer']] = {
                        'manufacturer': info['manufacturer'], 
                        'best_avg_stint_pace_sec': avg_time, 
                        'car_number': car['car_number'], 
                        'team': info['team'], 
                        'stint_details': {
                            'stint_number': stint.get('stint_number'),
                            'lap_count': stint.get('green_laps')
                        }
                    }
        showdown = sorted(list(best_stints.values()), key=lambda x: x['best_avg_stint_pace_sec'])
        return [{'rank': i + 1, **res} for i, res in enumerate(showdown)]

    def _remove_sec_keys_recursive(self, obj):
        if isinstance(obj, dict):
            new_dict = {}
            for k, v in obj.items():
                if isinstance(k, str) and k.endswith('_sec'): continue
                if isinstance(v, float) and np.isnan(v): new_dict[k] = None
                else: new_dict[k] = self._remove_sec_keys_recursive(v)
            return new_dict
        elif isinstance(obj, list): return [self._remove_sec_keys_recursive(item) for item in obj]
        else:
            if isinstance(obj, (np.integer, np.floating)): return obj.item() if pd.notna(obj) else None
            if pd.isna(obj): return None
            if isinstance(obj, float): return f"{obj:.3f}"
            return obj

    def _clean_output_for_json(self, analysis_results_dict):
        return self._remove_sec_keys_recursive(analysis_results_dict)

    def run_all_analyses(self):
        print("\n--- Running All Analyses (Original and New) ---")
        print("Running original analysis suite...")
        final_results = {"fastest_by_car_number": self.get_fastest_by_car_number(), "fastest_by_manufacturer": self.get_fastest_by_manufacturer(), "longest_stints_by_manufacturer": self.get_longest_stints_by_manufacturer(), "driver_deltas_by_car": self.get_driver_deltas_by_car(), "manufacturer_driver_pace_gap": self.get_manufacturer_driver_pace_gap(), "race_strategy_by_car": self.get_race_strategy_by_car(), "enhanced_strategy_analysis": self.get_enhanced_strategy_analysis()}
        print("Running new internal analysis suite...")
        final_results["traffic_management_analysis"] = self.get_traffic_management_analysis()
        final_results["earliest_fastest_lap_drivers"] = self.get_earliest_fastest_lap_drivers()
        final_results["full_pit_cycle_analysis"] = self.get_full_pit_cycle_analysis(final_results['race_strategy_by_car'], final_results['enhanced_strategy_analysis'])
        final_results["enhanced_strategy_analysis"] = self.add_degradation_cliff_analysis(final_results['enhanced_strategy_analysis'])
        print("Running new social media highlights suite...")
        final_results["social_media_highlights"] = self._get_social_media_highlights(final_results['fastest_by_car_number'], final_results['race_strategy_by_car'], final_results['enhanced_strategy_analysis'])
        print("All analyses complete. Cleaning for JSON export...")
        return self._clean_output_for_json(final_results)

    def export_to_json_file(self, data, output_filepath):
        print(f"\n--- Exporting to {output_filepath} ---");
        try:
            with open(output_filepath, 'w') as f: json.dump(data, f, indent=4); print(f"Successfully exported data to {output_filepath}")
        except TypeError as e: print(f"ERROR: TypeError during JSON export: {e}.")
        except Exception as e: print(f"ERROR: An unexpected error occurred during JSON export: {e}")

# <<< MODIFIED Main Execution Block >>>
if __name__ == '__main__':
    csv_file = '2025_impc_mido.csv'
    pit_json_file = '2025_mido_race_pits.json'
    output_json_file = '2025_mido_race_results_FINAL_with_all_features.json'
    fuel_file = '2025_mido_fuel.json' # <<< NEW FILE
    try:
        # Pass the new fuel capacity file to the analyzer
        analyzer = IMSADataAnalyzer(
            csv_file, 
            pit_json_filepath=pit_json_file, 
            fuel_capacity_json_filepath=fuel_file
        )
        all_results = analyzer.run_all_analyses()
        analyzer.export_to_json_file(all_results, output_json_file)
    except FileNotFoundError as e: print(e)
    except Exception as e: print(f"An unexpected error occurred at the top level: {e}"); import traceback; traceback.print_exc()
```

agents/core_analyzer/main.py
```py
"""CoreAnalyzer Toolbox Service for Project Apex.

Provides HTTP endpoints for running specific race data analyses:
- Full analysis (original endpoint)
- Pace analysis only
- Strategy analysis only
- Comparison of multiple analyses

Input/Output via Google Cloud Storage URIs.
"""
from __future__ import annotations


import json
import logging
import os
import pathlib
import tempfile
from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass, asdict
from enum import Enum

from flask import Flask, request, jsonify, Response
from google.cloud import storage

# Third-party analysis module provided separately in this repository.
from imsa_analyzer import IMSADataAnalyzer  # type: ignore

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
class AnalysisType(str, Enum):
    FULL = "full"
    PACE = "pace"
    STRATEGY = "strategy"

@dataclass
class AnalysisResult:
    analysis_type: AnalysisType
    gcs_uri: str
    metrics: Dict[str, Any]

class AnalysisRequest(TypedDict):
    run_id: str
    csv_path: str
    pit_json_path: str
    analysis_type: Optional[str]  # For single analysis requests

class ComparisonRequest(TypedDict):
    run_id: str
    analysis_paths: List[str]  # List of analysis URIs to compare
    comparison_metrics: List[str]  # Which metrics to include in comparison

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data-project-apex-v1")


# ---------------------------------------------------------------------------
# Logging setup (structured for Cloud Logging)
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("core_analyzer")

# ---------------------------------------------------------------------------
# Google Cloud clients (lazily initialised)
# ---------------------------------------------------------------------------
_storage_client: storage.Client | None = None


def _get_storage_client() -> storage.Client:
    """Lazily initialises and returns a Google Cloud Storage client."""
    global _storage_client  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _download_blob(gcs_uri: str, dest_path: pathlib.Path) -> None:
    """Downloads a GCS object to a local file path."""
    storage_client = _get_storage_client()
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    blob.download_to_filename(str(dest_path))
    LOGGER.info("Downloaded %s to %s", gcs_uri, dest_path)


def _upload_file(local_path: pathlib.Path, dest_bucket: str, dest_blob_name: str) -> str:
    """Uploads local file to GCS and returns the gs:// URI."""
    storage_client = _get_storage_client()
    bucket = storage_client.bucket(dest_bucket)
    blob = bucket.blob(dest_blob_name)
    blob.upload_from_filename(str(local_path))
    gcs_uri = f"gs://{dest_bucket}/{dest_blob_name}"
    LOGGER.info("Uploaded %s to %s", local_path, gcs_uri)
    return gcs_uri









# ---------------------------------------------------------------------------
# Analysis Functions
# ---------------------------------------------------------------------------

def _run_analysis(
    analyzer: IMSADataAnalyzer,
    analysis_type: AnalysisType = AnalysisType.FULL
) -> Dict[str, Any]:
    """Run the specified type of analysis."""
    if analysis_type == AnalysisType.PACE:
        return {
            "pace_analysis": analyzer.analyze_pace(),
            "metadata": {"analysis_type": "pace"}
        }
    elif analysis_type == AnalysisType.STRATEGY:
        return {
            "strategy_analysis": analyzer.analyze_strategy(),
            "metadata": {"analysis_type": "strategy"}
        }
    else:  # FULL
        return {
            **analyzer.run_all_analyses(),
            "metadata": {"analysis_type": "full"}
        }

def _compare_analyses(analysis_paths: List[str]) -> Dict[str, Any]:
    """Compare multiple analysis results."""
    comparisons = {}
    for path in analysis_paths:
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            _download_blob(path, pathlib.Path(tmp.name))
            with open(tmp.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
                comparisons[path] = {
                    "summary": {
                        k: v for k, v in data.items()
                        if k in ["metadata", "race_summary"]
                    },
                    "metrics": {
                        k: v for k, v in data.items()
                        if k not in ["metadata", "race_summary"]
                    }
                }
    return {"comparisons": comparisons}

# ---------------------------------------------------------------------------
# Flask Application
# ---------------------------------------------------------------------------
app = Flask(__name__)

@app.route("/health")
def health_check() -> Response:
    """Health check endpoint."""
    return jsonify({"status": "healthy"}), 200

@app.route("/analyze", methods=["POST"])
def analyze() -> Response:
    """Run a full analysis."""
    return _handle_analysis_request(request, AnalysisType.FULL)

@app.route("/analyze/pace", methods=["POST"])
def analyze_pace() -> Response:
    """Run pace analysis only."""
    return _handle_analysis_request(request, AnalysisType.PACE)

@app.route("/analyze/strategy", methods=["POST"])
def analyze_strategy() -> Response:
    """Run strategy analysis only."""
    return _handle_analysis_request(request, AnalysisType.STRATEGY)

@app.route("/analyze/compare", methods=["POST"])
def compare_analyses() -> Response:
    """Compare multiple analysis results."""
    req_json = request.get_json(force=True, silent=True)
    if not req_json:
        return jsonify({"error": "invalid_json"}), 400

    try:
        comparison = _compare_analyses(req_json["analysis_paths"])
        return jsonify(comparison), 200
    except Exception as exc:
        LOGGER.exception("Comparison failed: %s", exc)
        return jsonify({"error": "comparison_failed"}), 500

def _handle_analysis_request(request, analysis_type: AnalysisType) -> Response:
    """Handle analysis requests with common logic."""
    req_json = request.get_json(force=True, silent=True)
    if req_json is None:
        return jsonify({"error": "invalid_json"}), 400

    run_id = req_json.get("run_id")
    csv_uri = req_json.get("csv_path")
    pit_uri = req_json.get("pit_json_path")

    if not all([run_id, csv_uri, pit_uri]):
        return jsonify({"error": "missing_required_fields"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = pathlib.Path(tmpdir)
        local_csv = tmp_path / pathlib.Path(csv_uri).name
        local_pit = tmp_path / pathlib.Path(pit_uri).name

        try:
            # Download inputs
            _download_blob(csv_uri, local_csv)
            _download_blob(pit_uri, local_pit)

            # Run specified analysis
            analyzer = IMSADataAnalyzer(str(local_csv), str(local_pit))
            results = _run_analysis(analyzer, analysis_type)

            # Save results
            output_filename = f"{run_id}_results_{analysis_type.value}.json"
            local_out = tmp_path / output_filename
            with local_out.open("w", encoding="utf-8") as fp:
                json.dump(results, fp)

            # Upload to GCS
            dest_blob_name = f"{run_id}/{output_filename}"
            gcs_uri = _upload_file(local_out, ANALYZED_DATA_BUCKET, dest_blob_name)
            
            return jsonify({
                "analysis_path": gcs_uri,
                "analysis_type": analysis_type.value
            }), 200

        except Exception as exc:
            LOGGER.exception("Analysis failed: %s", exc)
            return jsonify({"error": "analysis_failed"}), 500
```

agents/core_analyzer/requirements.txt
```txt
Flask>=2.3.0
gunicorn>=21.2.0
pandas>=2.2.0
numpy>=1.26.0
google-cloud-storage>=2.16.0
typing-extensions>=4.0.0  # For TypedDict in Python <3.11
```

agents/historian/main.py
```py
"""Historian Cloud Run service for Project Apex.

Receives Pub/Sub push messages containing the GCS path to a current
_results_enhanced.json analysis file, fetches the prior year's analysis for the
same track/session from BigQuery, generates year-over-year comparison insights,
writes them to GCS, and completes (no downstream Pub/Sub).
"""
from __future__ import annotations


import json
import logging
import os
import pathlib
import re

# AI helper utils
from agents.common import ai_helpers
import tempfile
from typing import Any, Dict, List, Tuple

from flask import Flask, request, jsonify
from google.cloud import bigquery, storage

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
BQ_DATASET = os.getenv("BQ_DATASET", "imsa_history")
BQ_TABLE = os.getenv("BQ_TABLE", "race_analyses")
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("historian")

# ---------------------------------------------------------------------------
# Google Cloud clients (lazy)
# ---------------------------------------------------------------------------
_storage_client: storage.Client | None = None
_bq_client: bigquery.Client | None = None


def _init_clients() -> Tuple[storage.Client, bigquery.Client]:
    global _storage_client, _bq_client  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    if _bq_client is None:
        _bq_client = bigquery.Client()
    return _storage_client, _bq_client

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _parse_pubsub_push(req_json: Dict[str, Any]) -> Dict[str, Any]:
    """Decodes the data field from a Pub/Sub push message."""
    if "message" not in req_json or "data" not in req_json["message"]:
        raise ValueError("Invalid Pub/Sub push payload")
    decoded = base64.b64decode(req_json["message"]["data"]).decode("utf-8")
    return json.loads(decoded)

def _gcs_download(gcs_uri: str, dest: pathlib.Path) -> None:
    storage_client, _ = _init_clients()
    if not gcs_uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI: {gcs_uri}")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    storage_client.bucket(bucket_name).blob(blob_name).download_to_filename(dest)


def _gcs_upload(local_path: pathlib.Path, bucket: str, blob_name: str) -> str:
    storage_client, _ = _init_clients()
    storage_client.bucket(bucket).blob(blob_name).upload_from_filename(local_path)
    return f"gs://{bucket}/{blob_name}"


def _parse_filename(filename: str) -> Tuple[int, str, str]:
    """Extracts (year, track, session) from filenames like 2025_mido_race*."""
    match = re.match(r"(?P<year>\d{4})_(?P<track>[a-zA-Z]+)_(?P<session>[a-zA-Z]+)", filename)
    if not match:
        raise ValueError(f"Unrecognised filename format: {filename}")
    return int(match.group("year")), match.group("track"), match.group("session")


def _time_str_to_seconds(time_str: str | None) -> float | None:
    if time_str is None:
        return None
    try:
        if ":" in time_str:
            mins, rest = time_str.split(":", 1)
            return float(mins) * 60 + float(rest)
        return float(time_str)
    except ValueError:
        return None

# ---------------------------------------------------------------------------
# Comparison logic
# ---------------------------------------------------------------------------

def compare_analyses(current: Dict[str, Any], historical: Dict[str, Any]) -> List[Dict[str, Any]]:
    insights: List[Dict[str, Any]] = []

    # Manufacturer fastest-lap pace delta
    curr_fast = {item["manufacturer"]: _time_str_to_seconds(item.get("fastest_lap", {}).get("time"))
                 for item in current.get("fastest_by_manufacturer", [])}
    hist_fast = {item["manufacturer"]: _time_str_to_seconds(item.get("fastest_lap", {}).get("time"))
                 for item in historical.get("fastest_by_manufacturer", [])}
    for manuf, curr_time in curr_fast.items():
        hist_time = hist_fast.get(manuf)
        if curr_time is None or hist_time is None:
            continue
        delta = curr_time - hist_time
        faster_slower = "faster" if delta < 0 else "slower"
        insights.append({
            "category": "Historical Comparison",
            "type": "YoY Manufacturer Pace",
            "manufacturer": manuf,
            "details": f"{manuf} is {abs(delta):.2f}s {faster_slower} than last year."
        })

    # Tire degradation coefficient comparison
    def _coeff_map(data: Dict[str, Any]) -> Dict[str, float]:
        mapping: Dict[str, float] = {}
        for entry in data.get("enhanced_strategy_analysis", []):
            coeff = entry.get("tire_degradation_model", {}).get("deg_coeff_a")
            if coeff is not None:
                mapping[entry.get("manufacturer")] = coeff
        return mapping

    curr_coeff = _coeff_map(current)
    hist_coeff = _coeff_map(historical)
    for manuf, curr_val in curr_coeff.items():
        hist_val = hist_coeff.get(manuf)
        if hist_val is None:
            continue
        # percentage change relative to historical (positive => increase)
        if hist_val == 0:
            continue  # avoid div-by-zero
        pct_change = (curr_val - hist_val) / abs(hist_val) * 100
        trend = "improved" if pct_change < 0 else "worsened"
        insights.append({
            "category": "Historical Comparison",
            "type": "YoY Tire Degradation",
            "manufacturer": manuf,
            "details": f"{manuf} tire degradation has {trend} by {abs(pct_change):.1f}% year-over-year."
        })

    return insights


def _narrative_summary(insights: List[Dict[str, Any]]) -> str | None:
    """Generate a concise narrative summary of YoY insights via Gemini."""
    if not USE_AI_ENHANCED or not insights:
        return None
    prompt = (
        "You are a racing data analyst AI. Craft a concise paragraph (<=120 words) summarizing the following year-over-year insights.\n"
        "Focus on key trends and notable changes.\n\nInsights JSON:\n" + json.dumps(insights, indent=2) + "\n\nSummary:"
    )
    try:
        return ai_helpers.summarize(prompt, temperature=0.5, max_output_tokens=128)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.warning("Narrative generation failed: %s", exc)
        return None

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request():
    try:
        req_json = request.get_json(force=True, silent=True)
        if req_json is None:
            return jsonify({"error": "invalid_json"}), 400
            
        message_data = _parse_pubsub_push(req_json)
        analysis_uri: str | None = message_data.get("analysis_path")
        if not analysis_uri:
            return jsonify({"error": "missing_analysis_path"}), 400
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Invalid request: %s", exc)
        return jsonify({"message": "no_content"}), 204

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        # Derive run_id from GCS URI assuming format gs://bucket/run_id/...
        try:
            run_id = analysis_uri.split("/", 3)[3].split("/", 1)[0]
        except Exception:
            run_id = "unknown_run"
        local_analysis = tmp / pathlib.Path(analysis_uri).name
        try:
            # Download current analysis
            _gcs_download(analysis_uri, local_analysis)
            with local_analysis.open("r", encoding="utf-8") as fp:
                current_data = json.load(fp)

            # Extract event info
            year, track, session = _parse_filename(local_analysis.stem)
            prev_year = year - 1
            LOGGER.info("Comparing against historical year %s", prev_year)

            # Query BigQuery
            _, bq_client = _init_clients()
            table_full = f"{PROJECT_ID}.{BQ_DATASET}.{BQ_TABLE}"
            job = bq_client.query(
                f"SELECT analysis_json FROM `{table_full}`\n"
                "WHERE track = @track AND year = @year AND session_type = @session",
                job_config=bigquery.QueryJobConfig(
                    query_parameters=[
                        bigquery.ScalarQueryParameter("track", "STRING", track),
                        bigquery.ScalarQueryParameter("year", "INT64", prev_year),
                        bigquery.ScalarQueryParameter("session", "STRING", session),
                    ]
                ),
            )
            results = list(job.result())
            if not results:
                LOGGER.warning("No historical analysis found for %s %s %s", track, prev_year, session)
                return jsonify({"message": "no_content"}), 204
            historical_data = results[0]["analysis_json"]

            insights = compare_analyses(current_data, historical_data)
            if not insights:
                LOGGER.info("No insights generated for %s", analysis_uri)
                return jsonify({"message": "no_content"}), 204

            summary_text = _narrative_summary(insights)
            output_obj: Dict[str, Any] = {"insights": insights}
            if summary_text:
                output_obj["narrative"] = summary_text

            # Write insights file
            basename = local_analysis.stem.replace("_results_enhanced", "")
            out_filename = f"{basename}_historical_insights.json"
            local_out = tmp / out_filename
            with local_out.open("w", encoding="utf-8") as fp:
                json.dump(output_obj, fp)
            historical_gcs_uri = _gcs_upload(local_out, ANALYZED_DATA_BUCKET, f"{run_id}/{out_filename}")
            return jsonify({"historical_path": historical_gcs_uri}), 200
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return jsonify({"error": "internal_error"}), 500

    return jsonify({"message": "no_content"}), 204
```

agents/historian/requirements.txt
```txt
Flask>=2.3.0
gunicorn>=21.2.0
google-cloud-storage>=2.16.0
google-cloud-bigquery>=3.16.0
google-cloud-aiplatform>=1.50.0
tenacity>=8.2.3
```

agents/insight_hunter/main.py
```py
"""
InsightHunter Cloud Run service for Project Apex.

Listens for Pub/Sub push messages containing the Cloud Storage path to an
_enhanced.json race analysis file, derives tactical insights, stores them as a
new _insights.json file, and publishes a notification to the
`visualization-requests` topic.
"""
from __future__ import annotations


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

from flask import Flask, request, jsonify
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
        # The client library automatically uses PUBSUB_EMULATOR_HOST if set.
        # We provide AnonymousCredentials to bypass auth when using the emulator.
        if os.getenv("PUBSUB_EMULATOR_HOST"):
            from google.auth.credentials import AnonymousCredentials
            _publisher = pubsub_v1.PublisherClient(credentials=AnonymousCredentials())
        else:
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
def handle_request():
    try:
        req_json = request.get_json(force=True, silent=True)
        if req_json is None:
            return jsonify({"error": "invalid_json"}), 400
        
        # Parse Pub/Sub push message
        message_data = _parse_pubsub_push(req_json)
        analysis_uri: str | None = message_data.get("analysis_path")

        if not analysis_uri:
            return jsonify({"error": "missing_analysis_path"}), 400
    except Exception as exc:
        LOGGER.exception("Invalid request: %s", exc)
        return jsonify({"error": "bad_request"}), 400

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
            return jsonify({"error": "internal_error"}), 500

    return jsonify({"insights_path": insights_uri}), 200
```

agents/insight_hunter/requirements.txt
```txt
Flask>=2.3.0
gunicorn>=21.2.0
google-cloud-storage>=2.16.0
google-cloud-pubsub>=2.21.0
google-cloud-aiplatform>=1.50.0
tenacity>=8.2.3
```

agents/publicist/main.py
```py
"""Publicist Cloud Run service for Project Apex.

Generates social-media copy (tweet variations) from insights JSON using Vertex AI
Gemini. Expects Pub/Sub push payload with `analysis_path` and `insights_path`.
"""
from __future__ import annotations


import json
import logging
import os
import pathlib
import tempfile

# AI helpers
from agents.common import ai_helpers
from typing import Any, Dict, List

from flask import Flask, request, jsonify
from google.cloud import storage
from google.cloud import aiplatform

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
LOCATION = os.getenv("VERTEX_LOCATION", "us-central1")
MODEL_NAME = os.getenv("VERTEX_MODEL", "gemini-1.0-pro")
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

# --- NEW: Load the prompt template from the file on startup ---
PROMPT_TEMPLATE_PATH = pathlib.Path(__file__).parent / "prompt_template.md"
PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("publicist")

# ---------------------------------------------------------------------------
# Clients (lazy)
# ---------------------------------------------------------------------------
_storage_client: storage.Client | None = None


def _storage() -> storage.Client:
    global _storage_client  # pylint: disable=global-statement
    if _storage_client is None:
        _storage_client = storage.Client()
    return _storage_client

# ---------------------------------------------------------------------------
# Download & upload helpers
# ---------------------------------------------------------------------------

def _gcs_download(gcs_uri: str, dest: pathlib.Path) -> None:
    if not gcs_uri.startswith("gs://"):
        raise ValueError("Invalid GCS URI")
    bucket_name, blob_name = gcs_uri[5:].split("/", 1)
    _storage().bucket(bucket_name).blob(blob_name).download_to_filename(dest)


def _gcs_upload(local_path: pathlib.Path, dest_blob: str) -> str:
    blob = _storage().bucket(ANALYZED_DATA_BUCKET).blob(dest_blob)
    blob.upload_from_filename(local_path)
    return f"gs://{ANALYZED_DATA_BUCKET}/{dest_blob}"

# ---------------------------------------------------------------------------
# Insight selection

def _select_key_insights(insights: List[Dict[str, Any]], limit: int = 8) -> List[Dict[str, Any]]:
    """Deduplicate by type and prioritise Historical & Strategy categories."""
    selected: List[Dict[str, Any]] = []
    seen_types: set[str] = set()
    def _priority(ins):
        cat = ins.get("category", "")
        if cat == "Historical Comparison":
            return 0
        if "Strategy" in cat:
            return 1
        return 2
    valid_insights = [ins for ins in insights if isinstance(ins, dict)]
    if not valid_insights:
        return []
    sorted_in = sorted(valid_insights, key=_priority)
    for ins in sorted_in:
        if ins.get("type") in seen_types:
            continue
        selected.append(ins)
        seen_types.add(ins.get("type"))
        if len(selected) >= limit:
            break
    return selected

# Gemini helper
# ---------------------------------------------------------------------------

def _init_gemini() -> None:
    aiplatform.init(project=PROJECT_ID, location=LOCATION)


def _gen_tweets(insights: Any, analysis: Any, max_posts: int = 7) -> List[str]:
    """Call Gemini to generate up to `max_posts` social media posts.

    The `insights` parameter can be either:
    1. A list of dictionaries (legacy behaviour) or
    2. A dictionary whose values contain lists of insight dicts (current Insight Hunter output).
    The function normalises the structure so downstream logic always works with a flat
    list of insight dictionaries.
    """
    # ---------------------------------------------------------------------
    # Normalise the insights structure so we always work with List[Dict].
    # ---------------------------------------------------------------------
    flat_insights: List[Dict[str, Any]] = []
    if isinstance(insights, list):
        # Already the expected shape
        flat_insights = [ins for ins in insights if isinstance(ins, dict)]
    elif isinstance(insights, dict):
        # Flatten all list-valued entries (e.g. manufacturer_pace_ranking, etc.)
        for val in insights.values():
            if isinstance(val, list):
                flat_insights.extend([ins for ins in val if isinstance(ins, dict)])
    else:
        LOGGER.warning("Unsupported insights type passed to _gen_tweets: %s", type(insights))

    key_ins = _select_key_insights(flat_insights)
    if not key_ins:
        return []

    if USE_AI_ENHANCED:
        # --- MODIFIED: Use the loaded template and format it with both JSON objects ---
        # WARNING: Passing the full analysis_enhanced.json can be very large and may
        # exceed model input token limits. For production, consider summarizing
        # this payload first or ensuring you use a model with a large context window.
        prompt = PROMPT_TEMPLATE.format(
            insights_json=json.dumps(insights, indent=2),
            analysis_enhanced_json=json.dumps(analysis, indent=2),
            max_posts=max_posts,
        )
        # prompt = (
        #     "You are a social media manager for a professional race team. "
        #     "Create up to " + str(max_posts) + " engaging social media posts based on the provided JSON data. "
        #     "Each post must be a standalone string, under 280 characters, and include relevant hashtags like #IMSA and appropriate emojis. "
        #     "Only reference a particular manufacturer, car, or team once. If you use them for a post, do not use them for another.  "
        #     "Your response MUST be a valid JSON array of strings, like [\"post1\", \"post2\"]. Do not return anything else.\n\n"
        #     "Insights JSON:\n" + json.dumps(insights, indent=2)
        # )
        try:
            # Slightly higher temp for creativity, more tokens for safety
            tweets = ai_helpers.generate_json(prompt, temperature=0.8, max_output_tokens=5000)
            if isinstance(tweets, list) and all(isinstance(t, str) for t in tweets):
                return tweets[:max_posts]

            LOGGER.warning("AI response was not a list of strings: %s", tweets)
            return []  # Return empty list on malformed AI response
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning("AI tweet generation failed: %s", exc)
            return []  # Return empty list on API error

    # Fallback template only if AI is not used
    fallback = [f"🏁 {ins.get('type')}: {ins.get('details')} #IMSA #ProjectApex" for ins in key_ins[:max_posts]]
    return fallback

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request():
    try:
        req_json = request.get_json(force=True, silent=True)
        if req_json is None:
            return jsonify({"error": "invalid_json"}), 400
        analysis_uri: str | None = req_json.get("analysis_path")
        insights_uri: str | None = req_json.get("insights_path")
        if not analysis_uri or not insights_uri:
            return jsonify({"error": "missing_fields"}), 400
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Bad request: %s", exc)
        return jsonify({"error": "bad_request"}), 400

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        local_analysis = tmp / pathlib.Path(analysis_uri).name
        local_insights = tmp / pathlib.Path(insights_uri).name
        try:
            # Download input files
            _gcs_download(analysis_uri, local_analysis)
            _gcs_download(insights_uri, local_insights)

            # Load JSON content
            analysis_data = json.loads(local_analysis.read_text())
            insights_data = json.loads(local_insights.read_text())

            # Generate tweets
            tweets = _gen_tweets(insights_data, analysis_data)
            output_json = tmp / "social_media_posts.json"
            json.dump({"posts": tweets}, output_json.open("w", encoding="utf-8"))

            basename = pathlib.Path(insights_uri).stem.replace("_insights", "")
            out_uri = _gcs_upload(output_json, f"{basename}/social/social_media_posts.json")
            LOGGER.info("Uploaded social media posts to %s", out_uri)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return jsonify({"error": "internal_error"}), 500

    return jsonify({"posts_path": out_uri}), 200
```

agents/publicist/prompt_template.md
```md
You are a sharp, insightful Social Media Manager for a professional motorsports analytics firm. Your audience consists of knowledgeable race fans who appreciate deep insights, not just surface-level results.

Your goal is to generate up to **{max_posts}** engaging, short-form social media posts (for a platform like X/Twitter) using the provided JSON data.

**Data Sources:**
You have access to two JSON files:
1.  `analysis_enhanced.json`: Contains granular, detailed performance data for every car, including lap times, sector times, pit stops, and stint analysis.
2.  `insights.json`: Contains high-level, pre-calculated rankings and AI-generated commentary on manufacturer pace, tire wear, and individual performance outliers.

**Core Instructions:**

1.  **Identify Key Storylines:** Scrutinize both JSON files to find the most compelling narratives. Look for:
    *   **Top Performers:** Who had the absolute fastest lap? (`fastest_by_car_number`) Which manufacturer was dominant? (`manufacturer_pace_ranking`).
    *   **Ultimate Consistency:** Who is the "Metronome"? (`social_media_highlights.metronome_award`). This is for drivers who can post incredibly consistent times lap after lap.
    *   **Untapped Potential:** Which car/driver had a much faster theoretical "optimal lap" than their actual fastest lap? (`car_untapped_potential_ranking`). This implies they have more speed to unlock.
    *   **Teamwork & Strategy:** Which team had the most efficient pit stops? (`full_pit_cycle_analysis`). Which car had the smallest performance gap between its drivers? (`driver_deltas_by_car`).
    *   **Surprising Gaps & Outliers:** Is there a huge performance gap between teammates (`driver_deltas_by_car`) or a massive difference in tire wear between manufacturers? (`manufacturer_tire_wear_ranking`).
    *   **Freestyle:** Infer from the rest of the metrics something else interesting, compelling storylines capture attention.

2.  **Craft Compelling Posts:** For each post:
    *   **Hook:** Start with an engaging phrase (e.g., "Pace analysis is in!", "Talk about consistency!", "Digging into the data...").
    *   **Translate Data to Narrative:** Don't just state a fact. Frame it as a story. Instead of "Car #57 had the fastest lap," write "Pure dominance from the #57 Mercedes-AMG of Winward Racing, setting the pace for the entire field. Untouchable today. 🔥"
    *   **Be Specific:** Use the actual data (lap times, car numbers, driver names) to add credibility.

3.  **Ensure Variety:**
    *   Prioritize creating a diverse set of posts covering different topics (e.g., one on pace, one on consistency, one on potential).
    *   **Do not** reference the exact same car, driver, or team in more than one post. Select the most interesting story for each entity and move on to find other stories in the data.

**Constraints & Formatting Rules:**

*   **Character Limit:** Each post must be a standalone string and **strictly under 280 characters**.
*   **Hashtags:** Each post **must** include `#IMSA` and at least one other relevant hashtag (e.g., `#Motorsport`, `#RaceData`, the manufacturer's name like `#Porsche`).
*   **Emojis:** Use 1-3 relevant emojis per post to increase engagement. 🏎️💨📊⏱️
*   **Output:** Your final response **MUST** be a single, valid JSON array of strings, with each string being one social media post.

**Example Output Format:**

```json
[
  "Post 1 text content including hashtags and emojis.",
  "Post 2 text content including hashtags and emojis.",
  "Post 3 text content including hashtags and emojis."
]
```

### DATA
Insights JSON:
```json
{insights_json}
```
Analysis Enhanced JSON:
```json
{analysis_enhanced_json}
```
```

agents/publicist/requirements.txt
```txt
Flask>=2.3.0
gunicorn>=21.2.0
google-cloud-storage>=2.16.0
google-cloud-aiplatform>=1.50.0
tenacity>=8.2.3
```

agents/scribe/main.py
```py
"""Scribe Cloud Run service for Project Apex.

Generates a PDF engineering report from analysis and insights JSON using Jinja2
and WeasyPrint.
"""
from __future__ import annotations


import json
import logging
import os
import pathlib
import tempfile

# AI helpers
from agents.common import ai_helpers
import os
from typing import Any, Dict, List

from flask import Flask, request, jsonify
from google.cloud import storage
from jinja2 import Environment, FileSystemLoader, select_autoescape
from weasyprint import HTML

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GCP_PROJECT")
ANALYZED_DATA_BUCKET = os.getenv("ANALYZED_DATA_BUCKET", "imsa-analyzed-data")
TEMPLATE_NAME = "report_template.html"
USE_AI_ENHANCED = os.getenv("USE_AI_ENHANCED", "true").lower() == "true"

# --- NEW: Load the prompt template from the file on startup ---
PROMPT_TEMPLATE_PATH = pathlib.Path(__file__).parent / "prompt_template.md"
PROMPT_TEMPLATE = PROMPT_TEMPLATE_PATH.read_text()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("scribe")

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

# ---------------------------------------------------------------------------
# AI narrative generation
# ---------------------------------------------------------------------------

def _generate_narrative(insights: Dict[str, List[Dict[str, Any]]], analysis: Dict[str, Any]) -> Dict[str, Any] | None:
    """Uses Gemini to craft executive summary paragraph and tactical recommendations."""
    if not USE_AI_ENHANCED or not insights:
        return None

    # --- MODIFIED: Use the loaded template and format it with both JSON objects ---
    # WARNING: Passing the full analysis_enhanced.json can be very large and may
    # exceed model input token limits. For production, consider summarizing
    # this payload first or ensuring you use a model with a large context window.
    prompt = PROMPT_TEMPLATE.format(
        insights_json=json.dumps(insights, indent=2),
        analysis_enhanced_json=json.dumps(analysis, indent=2),
    )

    # Pass the entire insights dictionary (already grouped by category) to the model
    # prompt = (
    #     "You are a motorsport performance engineer. Based on the race insights JSON, "
    #     "write a concise executive summary (<= 500 words) and 3 tactical recommendations. "
    #     "You should outline which cars, manufacturers, and teams are leading, mid-field, and lagging based on ALL provided data. "
    #     "Consider extreme outliers as not relevant, and focus only on those that are majorly influencing the race in various aspects of the field. "
    #     "Respond ONLY as minified JSON with keys 'executive_summary' and 'tactical_recommendations' (array of strings).\n\n"
    #     f"Insights:\n{json.dumps(insights, indent=2)}\n"
    # )
    try:
        result = ai_helpers.generate_json(
            prompt, temperature=0.5, max_output_tokens=int(os.getenv("MAX_OUTPUT_TOKENS", 25000))
        )
        if isinstance(result, dict):
            return result
        LOGGER.warning("Unexpected narrative JSON format: %s", result)
    except Exception:
        LOGGER.exception("Narrative generation failed")

    return None

# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _render_report(analysis: Dict[str, Any], insights: List[Dict[str, Any]], narrative: Dict[str, Any] | None, output_pdf: pathlib.Path) -> None:
    env = Environment(
        loader=FileSystemLoader(pathlib.Path(__file__).parent),
        autoescape=select_autoescape(["html", "xml"]),
    )
    template = env.get_template(TEMPLATE_NAME)

    # Simple event name extraction from analysis metadata if available
    event_id = pathlib.Path(analysis.get("metadata", {}).get("event_id", "")).stem or "Race Event"

    html_str = template.render(event_name=event_id, insights=insights, narrative=narrative or {})
    HTML(string=html_str).write_pdf(str(output_pdf))

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request():
    try:
        req_json = request.get_json(force=True, silent=True)
        if req_json is None:
            return jsonify({"error": "invalid_json"}), 400
        analysis_uri: str | None = req_json.get("analysis_path")
        insights_uri: str | None = req_json.get("insights_path")
        if not analysis_uri or not insights_uri:
            return jsonify({"error": "missing_fields"}), 400
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

            pdf_path = tmp / "race_report.pdf"
            narrative = _generate_narrative(insights_data, analysis_data)
            _render_report(analysis_data, insights_data, narrative, pdf_path)

            basename = local_analysis.stem.replace("_results_enhanced", "")
            out_uri = _gcs_upload(pdf_path, f"{basename}/reports/race_report.pdf")
            LOGGER.info("Uploaded PDF report to %s", out_uri)
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.exception("Processing failed: %s", exc)
            return jsonify({"error": "internal_error"}), 500
    return jsonify({"report_path": out_uri}), 200
```

agents/scribe/prompt_template.md
```md
You are an expert Motorsport Performance Engineer. Your audience is the team principals and chief strategists who require a data-driven, objective, and concise analysis of the race. Your task is to analyze comprehensive race performance data and generate a high-level executive summary and specific, actionable tactical recommendations.

### CONTEXT
You will be provided with two JSON data sources:
1.  `insights.json`: Contains high-level, pre-calculated rankings and initial commentary on manufacturer pace and tire wear. Use this for top-level trends and to guide your summary.
2.  `analysis_enhanced.json`: Contains the full, granular race data, including per-car/driver performance, stint details, pit stop analysis, and tire degradation models. Use this to find the specific evidence and "the why" behind the insights and to formulate your tactical recommendations.

### TASK

1.  **Synthesize Overall Performance:** Analyze the data across multiple vectors to form a complete picture.
    *   **Pace:** Cross-reference fastest laps, optimal laps (`fastest_by_car_number`), and average green flag pace (`race_strategy_by_car.stints.avg_green_time_formatted`).
    *   **Consistency & Driver Skill:** Evaluate the standard deviation of green laps (`enhanced_strategy_analysis.race_pace_consistency_stdev`), the delta between drivers in the same car (`driver_deltas_by_car`), and traffic management performance (`traffic_management_analysis`).
    *   **Strategy & Efficiency:** Assess the true cost of pit stops using the `full_pit_cycle_analysis` and note any cars with particularly efficient or inefficient pit work.
    *   **Tire Management:** Use the tire degradation models in `enhanced_strategy_analysis.tire_degradation_model` to identify manufacturers who manage their tires well (low degradation coefficient `deg_coeff_a`) versus those who do not.

2.  **Write the Executive Summary (Max 500 words):**
    *   Start by identifying the clear performance tiers: Leaders, Mid-field, and Laggers, naming the key manufacturers and standout cars in each.
    *   Justify these placements with key data points (e.g., "Mercedes-AMG leads due to superior raw pace, backed by the best tire degradation model," or "Audi is lagging, evidenced by the slowest average green-flag pace and high pit cycle loss.").
    *   Conclude with a sentence on the biggest strategic differentiator seen in the race (e.g., tire management, pit stop execution, or driver consistency).

3.  **Develop 3 Tactical Recommendations:**
    *   These must be specific, data-driven, and reference car numbers or manufacturers.
    *   Provide **one** recommendation for a **leading team** to maintain their advantage.
    *   Provide **one** recommendation for a **mid-field team** to gain a competitive edge.
    *   Provide **one** recommendation for a **lagging team** to address a fundamental weakness.
    *   **Crucially, justify each recommendation with data** from `analysis_enhanced.json`, such as `tire_degradation_model`, `full_pit_cycle_analysis`, or `driver_deltas_by_car`.

### CONSTRAINTS

*   **Outlier Handling:** The significant pace deficit of the third driver in Car #64 (Ted Giovanis, `driver_deltas_by_car`) is an extreme outlier. Do not let this single data point heavily skew the overall analysis of Aston Martin's competitive potential. Focus on the performance of the core competitive drivers in the car.
*   **Output Format:** Respond **ONLY** with a single, minified JSON object. Do not include any text, greetings, or explanations outside of the JSON structure.

### DATA
Insights JSON:
```json
{insights_json}
```
Analysis Enhanced JSON:
```json
{analysis_enhanced_json}
```
```

agents/scribe/report_template.html
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Project Apex Race Report - {{ event_name }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        h1 { text-align: center; }
        .section { margin-top: 30px; }
        ul { list-style-type: disc; margin-left: 20px; }
        table { width: 100%; border-collapse: collapse; margin-top: 10px; }
        th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }
        th { background-color: #f0f0f0; }
    </style>
</head>
<body>

<h1>Project Apex Race Report - {{ event_name }}</h1>

<div class="section">
    <h2>Executive Summary</h2>
    {% if narrative.executive_summary %}
    <p>{{ narrative.executive_summary }}</p>
    {% else %}
    <ul>
        {% for insight in insights if insight.category == 'Historical Comparison' %}
        <li><strong>{{ insight.type }}</strong> – {{ insight.details }}</li>
        {% else %}
        <li>No historical comparison insights available.</li>
        {% endfor %}
    </ul>
    {% endif %}
</div>

<div class="section">
    <h2>Tactical Insights</h2>
    {% if narrative.tactical_recommendations %}
    <ul>
        {% for rec in narrative.tactical_recommendations %}
        <li>{{ rec }}</li>
        {% endfor %}
    </ul>
    {% else %}
    <ul>
        {% for insight in insights if insight.category != 'Historical Comparison' %}
        <li><strong>{{ insight.type }}</strong> – {{ insight.details }}</li>
        {% else %}
        <li>No tactical insights available.</li>
        {% endfor %}
    </ul>
    {% endif %}
</div>

</body>
</html>
```

agents/scribe/requirements.txt
```txt
Flask>=2.3.0
gunicorn>=21.2.0
google-cloud-storage>=2.16.0
Jinja2>=3.1.0
WeasyPrint==57.2
pydyf==0.7.0
google-cloud-aiplatform>=1.50.0
tenacity>=8.2.3
```

agents/test_data/glen_race.csv
```csv
﻿NUMBER; DRIVER_NUMBER; LAP_NUMBER; LAP_TIME; LAP_IMPROVEMENT; CROSSING_FINISH_LINE_IN_PIT; S1; S1_IMPROVEMENT; S2; S2_IMPROVEMENT; S3; S3_IMPROVEMENT; KPH; ELAPSED; HOUR;S1_LARGE;S2_LARGE;S3_LARGE;TOP_SPEED;DRIVER_NAME;PIT_TIME;CLASS;GROUP;TEAM;MANUFACTURER;FLAG_AT_FL;
10;2;1;2:18.204;0;;43.105;0;48.028;0;47.071;0;142.5;2:18.204;11:47:46.534;0:43.105;0:48.028;0:47.071;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;2;2:04.635;0;;27.804;0;49.111;0;47.720;0;158.0;4:22.839;11:49:51.169;0:27.804;0:49.111;0:47.720;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;FCY;
10;2;3;2:29.693;0;;32.261;0;59.028;0;58.404;0;131.6;6:52.532;11:52:20.862;0:32.261;0:59.028;0:58.404;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;FCY;
10;2;4;2:53.161;0;;43.585;0;1:12.005;0;57.571;0;113.8;9:45.693;11:55:14.023;0:43.585;1:12.005;0:57.571;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;5;2:04.585;0;;28.918;0;46.795;0;48.872;0;158.1;11:50.278;11:57:18.608;0:28.918;0:46.795;0:48.872;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;6;2:01.979;0;;27.971;0;46.655;0;47.353;0;161.5;13:52.257;11:59:20.587;0:27.971;0:46.655;0:47.353;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;7;2:02.747;0;;28.375;0;46.669;0;47.703;0;160.5;15:55.004;12:01:23.334;0:28.375;0:46.669;0:47.703;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;8;2:03.631;0;;28.018;0;46.511;0;49.102;0;159.3;17:58.635;12:03:26.965;0:28.018;0:46.511;0:49.102;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;9;2:02.033;0;;27.851;0;46.973;0;47.209;0;161.4;20:00.668;12:05:28.998;0:27.851;0:46.973;0:47.209;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;10;2:01.480;0;;27.898;0;46.443;1;47.139;0;162.2;22:02.148;12:07:30.478;0:27.898;0:46.443;0:47.139;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;11;2:01.869;0;;27.899;0;46.631;0;47.339;0;161.6;24:04.017;12:09:32.347;0:27.899;0:46.631;0:47.339;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;12;2:02.378;0;;27.862;0;46.930;0;47.586;0;161.0;26:06.395;12:11:34.725;0:27.862;0:46.930;0:47.586;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;13;2:02.356;0;;28.136;0;46.848;0;47.372;0;161.0;28:08.751;12:13:37.081;0:28.136;0:46.848;0:47.372;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;14;2:02.029;0;;27.912;0;46.624;0;47.493;0;161.4;30:10.780;12:15:39.110;0:27.912;0:46.624;0:47.493;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;15;2:02.613;0;;27.870;0;46.883;0;47.860;0;160.7;32:13.393;12:17:41.723;0:27.870;0:46.883;0:47.860;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;16;2:01.611;0;;27.747;1;46.537;0;47.327;0;162.0;34:15.004;12:19:43.334;0:27.747;0:46.537;0:47.327;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;17;2:02.626;0;;27.919;0;46.786;0;47.921;0;160.6;36:17.630;12:21:45.960;0:27.919;0:46.786;0:47.921;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;18;2:02.096;0;;27.845;0;46.921;0;47.330;0;161.3;38:19.726;12:23:48.056;0:27.845;0:46.921;0:47.330;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;19;2:02.029;0;;27.923;0;46.623;0;47.483;0;161.4;40:21.755;12:25:50.085;0:27.923;0:46.623;0:47.483;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;20;2:02.101;0;;27.983;0;46.617;0;47.501;0;161.3;42:23.856;12:27:52.186;0:27.983;0:46.617;0:47.501;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;21;2:01.884;0;;27.942;0;46.527;0;47.415;0;161.6;44:25.740;12:29:54.070;0:27.942;0:46.527;0:47.415;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;22;2:01.748;0;;27.941;0;46.582;0;47.225;0;161.8;46:27.488;12:31:55.818;0:27.941;0:46.582;0:47.225;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;23;2:03.058;0;;27.835;0;47.245;0;47.978;0;160.1;48:30.546;12:33:58.876;0:27.835;0:47.245;0:47.978;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;24;2:02.694;0;;28.083;0;46.598;0;48.013;0;160.5;50:33.240;12:36:01.570;0:28.083;0:46.598;0:48.013;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;25;2:01.785;0;;27.822;0;46.929;0;47.034;1;161.7;52:35.025;12:38:03.355;0:27.822;0:46.929;0:47.034;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;GF;
10;2;26;3:18.296;0;B;28.155;0;47.608;0;2:02.533;0;99.3;55:53.321;12:41:21.651;0:28.155;0:47.608;2:02.533;;Christina Lam;;TCR;;Rockwell Autosport Development;Audi;FCY;
10;1;27;2:47.348;0;;44.606;0;51.969;0;1:10.773;0;117.7;58:40.669;12:44:08.999;0:44.606;0:51.969;1:10.773;;Eric Rockwell;0:01:30.780;TCR;;Rockwell Autosport Development;Audi;FCY;
10;1;28;11:12.065;0;B;46.660;0;1:09.337;0;9:16.068;0;29.3;1:09:52.734;12:55:21.064;0:46.660;1:09.337;9:16.068;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;29;2:13.483;0;;39.930;0;46.973;0;46.580;0;147.6;1:12:06.217;12:57:34.547;0:39.930;0:46.973;0:46.580;;Eric Rockwell;0:08:27.544;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;30;1:58.431;2;;27.546;0;44.683;2;46.202;2;166.3;1:14:04.648;12:59:32.978;0:27.546;0:44.683;0:46.202;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;31;1:58.511;0;;27.546;0;44.753;0;46.212;0;166.2;1:16:03.159;13:01:31.489;0:27.546;0:44.753;0:46.212;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;32;1:59.487;0;;27.365;2;45.211;0;46.911;0;164.9;1:18:02.646;13:03:30.976;0:27.365;0:45.211;0:46.911;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;33;2:00.112;0;;27.783;0;45.843;0;46.486;0;164.0;1:20:02.758;13:05:31.088;0:27.783;0:45.843;0:46.486;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;34;1:58.557;0;;27.500;0;44.758;0;46.299;0;166.2;1:22:01.315;13:07:29.645;0:27.500;0:44.758;0:46.299;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;35;1:59.086;0;;27.532;0;44.988;0;46.566;0;165.4;1:24:00.401;13:09:28.731;0:27.532;0:44.988;0:46.566;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;36;1:59.173;0;;27.560;0;45.076;0;46.537;0;165.3;1:25:59.574;13:11:27.904;0:27.560;0:45.076;0:46.537;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;37;2:03.062;0;;27.517;0;45.736;0;49.809;0;160.1;1:28:02.636;13:13:30.966;0:27.517;0:45.736;0:49.809;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;38;1:59.444;0;;28.017;0;44.800;0;46.627;0;164.9;1:30:02.080;13:15:30.410;0:28.017;0:44.800;0:46.627;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;39;2:00.040;0;;27.600;0;44.911;0;47.529;0;164.1;1:32:02.120;13:17:30.450;0:27.600;0:44.911;0:47.529;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
10;1;40;2:00.326;0;;27.621;0;45.288;0;47.417;0;163.7;1:34:02.446;13:19:30.776;0:27.621;0:45.288;0:47.417;;Eric Rockwell;;TCR;;Rockwell Autosport Development;Audi;GF;
12;1;1;2:12.738;0;;35.328;0;48.881;0;48.529;0;148.4;2:12.738;11:47:41.068;0:35.328;0:48.881;0:48.529;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;2;2:03.791;0;;28.679;0;47.027;0;48.085;0;159.1;4:16.529;11:49:44.859;0:28.679;0:47.027;0:48.085;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;FCY;
12;1;3;2:31.114;0;;32.221;0;1:01.381;0;57.512;0;130.4;6:47.643;11:52:15.973;0:32.221;1:01.381;0:57.512;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;FCY;
12;1;4;2:52.496;0;;43.099;0;1:07.258;0;1:02.139;0;114.2;9:40.139;11:55:08.469;0:43.099;1:07.258;1:02.139;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;5;2:01.745;0;;27.585;0;46.814;0;47.346;0;161.8;11:41.884;11:57:10.214;0:27.585;0:46.814;0:47.346;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;6;2:01.145;0;;27.289;0;46.188;0;47.668;0;162.6;13:43.029;11:59:11.359;0:27.289;0:46.188;0:47.668;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;7;2:02.205;0;;28.171;0;46.808;0;47.226;0;161.2;15:45.234;12:01:13.564;0:28.171;0:46.808;0:47.226;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;8;2:01.366;0;;27.650;0;46.373;0;47.343;0;162.3;17:46.600;12:03:14.930;0:27.650;0:46.373;0:47.343;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;9;2:01.313;0;;27.299;0;46.216;0;47.798;0;162.4;19:47.913;12:05:16.243;0:27.299;0:46.216;0:47.798;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;10;2:01.136;0;;27.897;0;45.858;0;47.381;0;162.6;21:49.049;12:07:17.379;0:27.897;0:45.858;0:47.381;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;11;1:59.491;0;;27.410;0;45.535;0;46.546;1;164.9;23:48.540;12:09:16.870;0:27.410;0:45.535;0:46.546;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;12;2:03.221;0;;28.591;0;47.240;0;47.390;0;159.9;25:51.761;12:11:20.091;0:28.591;0:47.240;0:47.390;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;13;2:00.747;0;;27.558;0;46.473;0;46.716;0;163.1;27:52.508;12:13:20.838;0:27.558;0:46.473;0:46.716;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;14;2:00.724;0;;27.298;0;46.084;0;47.342;0;163.2;29:53.232;12:15:21.562;0:27.298;0:46.084;0:47.342;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;15;2:00.214;0;;27.308;0;45.895;0;47.011;0;163.9;31:53.446;12:17:21.776;0:27.308;0:45.895;0:47.011;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;16;1:59.276;0;;27.211;1;45.371;0;46.694;0;165.1;33:52.722;12:19:21.052;0:27.211;0:45.371;0:46.694;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;17;2:00.115;0;;27.529;0;45.681;0;46.905;0;164.0;35:52.837;12:21:21.167;0:27.529;0:45.681;0:46.905;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;18;2:00.640;0;;27.307;0;46.285;0;47.048;0;163.3;37:53.477;12:23:21.807;0:27.307;0:46.285;0:47.048;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;19;2:00.360;0;;27.375;0;45.724;0;47.261;0;163.7;39:53.837;12:25:22.167;0:27.375;0:45.724;0:47.261;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;20;1:59.871;0;;27.305;0;45.435;0;47.131;0;164.3;41:53.708;12:27:22.038;0:27.305;0:45.435;0:47.131;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;21;1:59.718;0;;27.373;0;45.300;1;47.045;0;164.5;43:53.426;12:29:21.756;0:27.373;0:45.300;0:47.045;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;22;1:59.758;0;;27.253;0;45.416;0;47.089;0;164.5;45:53.184;12:31:21.514;0:27.253;0:45.416;0:47.089;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;23;2:00.318;0;;27.564;0;45.977;0;46.777;0;163.7;47:53.502;12:33:21.832;0:27.564;0:45.977;0:46.777;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;24;1:59.855;0;;27.276;0;45.506;0;47.073;0;164.4;49:53.357;12:35:21.687;0:27.276;0:45.506;0:47.073;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;1;25;2:16.400;0;B;27.651;0;45.495;0;1:03.254;0;144.4;52:09.757;12:37:38.087;0:27.651;0:45.495;1:03.254;;Rafael Martinez;;GS;B;RAFA Racing;Toyota;GF;
12;2;26;3:25.072;0;;1:46.597;0;50.112;0;48.363;0;96.1;55:34.829;12:41:03.159;1:46.597;0:50.112;0:48.363;;Jim Jonsin;0:01:35.949;GS;B;RAFA Racing;Toyota;FCY;
12;2;27;3:02.092;0;;28.298;0;1:10.293;0;1:23.501;0;108.2;58:36.921;12:44:05.251;0:28.298;1:10.293;1:23.501;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;FCY;
12;2;28;3:00.963;0;;46.865;0;1:09.412;0;1:04.686;0;108.9;1:01:37.884;12:47:06.214;0:46.865;1:09.412;1:04.686;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;FCY;
12;2;29;2:58.815;0;;32.679;0;1:10.367;0;1:15.769;0;110.2;1:04:36.699;12:50:05.029;0:32.679;1:10.367;1:15.769;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;FCY;
12;2;30;3:05.935;0;;51.704;0;1:14.331;0;59.900;0;105.9;1:07:42.634;12:53:10.964;0:51.704;1:14.331;0:59.900;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;31;2:04.746;0;;29.809;0;48.332;0;46.605;0;157.9;1:09:47.380;12:55:15.710;0:29.809;0:48.332;0:46.605;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;32;2:00.568;0;;27.389;0;45.491;0;47.688;0;163.4;1:11:47.948;12:57:16.278;0:27.389;0:45.491;0:47.688;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;33;2:05.851;0;;28.789;0;47.786;0;49.276;0;156.5;1:13:53.799;12:59:22.129;0:28.789;0:47.786;0:49.276;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;34;2:00.917;0;;27.974;0;45.783;0;47.160;0;162.9;1:15:54.716;13:01:23.046;0:27.974;0:45.783;0:47.160;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;35;2:01.602;0;;27.947;0;46.851;0;46.804;0;162.0;1:17:56.318;13:03:24.648;0:27.947;0:46.851;0:46.804;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;36;1:59.895;0;;27.271;0;46.054;0;46.570;0;164.3;1:19:56.213;13:05:24.543;0:27.271;0:46.054;0:46.570;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;37;2:01.602;0;;27.301;0;46.160;0;48.141;0;162.0;1:21:57.815;13:07:26.145;0:27.301;0:46.160;0:48.141;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;38;2:00.380;0;;27.670;0;46.086;0;46.624;0;163.6;1:23:58.195;13:09:26.525;0:27.670;0:46.086;0:46.624;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;39;1:58.042;0;;27.209;0;44.855;0;45.978;0;166.9;1:25:56.237;13:11:24.567;0:27.209;0:44.855;0:45.978;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;40;1:58.269;0;;27.242;0;44.868;0;46.159;0;166.6;1:27:54.506;13:13:22.836;0:27.242;0:44.868;0:46.159;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;41;1:58.915;0;;27.592;0;45.121;0;46.202;0;165.7;1:29:53.421;13:15:21.751;0:27.592;0:45.121;0:46.202;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;42;1:58.253;0;;27.473;0;44.598;0;46.182;0;166.6;1:31:51.674;13:17:20.004;0:27.473;0:44.598;0:46.182;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;43;1:58.450;0;;27.379;0;45.192;0;45.879;0;166.3;1:33:50.124;13:19:18.454;0:27.379;0:45.192;0:45.879;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;44;1:58.064;0;;27.368;0;44.838;0;45.858;0;166.8;1:35:48.188;13:21:16.518;0:27.368;0:44.838;0:45.858;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;45;1:57.769;0;;27.304;0;44.625;0;45.840;2;167.3;1:37:45.957;13:23:14.287;0:27.304;0:44.625;0:45.840;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;46;1:57.645;0;;26.974;2;44.635;0;46.036;0;167.4;1:39:43.602;13:25:11.932;0:26.974;0:44.635;0:46.036;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;47;1:58.710;0;;27.753;0;44.882;0;46.075;0;165.9;1:41:42.312;13:27:10.642;0:27.753;0:44.882;0:46.075;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;48;1:58.076;0;;27.095;0;44.925;0;46.056;0;166.8;1:43:40.388;13:29:08.718;0:27.095;0:44.925;0:46.056;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;49;1:58.482;0;;27.473;0;45.086;0;45.923;0;166.3;1:45:38.870;13:31:07.200;0:27.473;0:45.086;0:45.923;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;FCY;
12;2;50;2:15.144;0;;31.996;0;55.065;0;48.083;0;145.8;1:47:54.014;13:33:22.344;0:31.996;0:55.065;0:48.083;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;FCY;
12;2;51;2:44.217;0;;29.094;0;1:17.334;0;57.789;0;120.0;1:50:38.231;13:36:06.561;0:29.094;1:17.334;0:57.789;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;52;2:00.724;0;;27.934;0;46.164;0;46.626;0;163.2;1:52:38.955;13:38:07.285;0:27.934;0:46.164;0:46.626;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;53;1:57.429;2;;27.039;0;44.532;2;45.858;0;167.7;1:54:36.384;13:40:04.714;0:27.039;0:44.532;0:45.858;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;GF;
12;2;54;2:03.214;0;;27.216;0;44.972;0;51.026;0;159.9;1:56:39.598;13:42:07.928;0:27.216;0:44.972;0:51.026;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;FCY;
12;2;55;3:24.791;0;;48.571;0;1:19.168;0;1:17.052;0;96.2;2:00:04.389;13:45:32.719;0:48.571;1:19.168;1:17.052;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;FCY;
12;2;56;3:26.515;0;;53.438;0;1:15.617;0;1:17.460;0;95.4;2:03:30.904;13:48:59.234;0:53.438;1:15.617;1:17.460;;Jim Jonsin;;GS;B;RAFA Racing;Toyota;FF;
13;1;1;1:59.216;0;;28.163;0;45.183;0;45.870;0;165.2;1:59.216;11:47:27.546;0:28.163;0:45.183;0:45.870;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;2;1:57.028;0;;27.020;0;44.290;0;45.718;0;168.3;3:56.244;11:49:24.574;0:27.020;0:44.290;0:45.718;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;3;2:26.067;0;;28.566;0;59.276;0;58.225;0;134.9;6:22.311;11:51:50.641;0:28.566;0:59.276;0:58.225;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;FCY;
13;1;4;3:10.938;0;;45.575;0;1:17.611;0;1:07.752;0;103.2;9:33.249;11:55:01.579;0:45.575;1:17.611;1:07.752;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;5;1:58.137;0;;27.467;0;45.038;0;45.632;0;166.7;11:31.386;11:56:59.716;0:27.467;0:45.038;0:45.632;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;6;1:56.618;0;;26.996;0;44.021;0;45.601;0;168.9;13:28.004;11:58:56.334;0:26.996;0:44.021;0:45.601;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;7;1:56.751;0;;27.143;0;44.079;0;45.529;0;168.7;15:24.755;12:00:53.085;0:27.143;0:44.079;0:45.529;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;8;1:56.567;0;;26.975;0;44.025;0;45.567;0;169.0;17:21.322;12:02:49.652;0:26.975;0:44.025;0:45.567;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;9;1:56.666;0;;26.939;0;43.977;1;45.750;0;168.8;19:17.988;12:04:46.318;0:26.939;0:43.977;0:45.750;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;10;1:56.488;2;;26.902;0;44.149;0;45.437;2;169.1;21:14.476;12:06:42.806;0:26.902;0:44.149;0:45.437;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;11;1:56.544;0;;26.906;0;44.072;0;45.566;0;169.0;23:11.020;12:08:39.350;0:26.906;0:44.072;0:45.566;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;12;1:56.720;0;;26.873;0;44.233;0;45.614;0;168.8;25:07.740;12:10:36.070;0:26.873;0:44.233;0:45.614;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;13;1:56.809;0;;26.870;0;44.332;0;45.607;0;168.6;27:04.549;12:12:32.879;0:26.870;0:44.332;0:45.607;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;14;1:57.225;0;;26.933;0;44.311;0;45.981;0;168.0;29:01.774;12:14:30.104;0:26.933;0:44.311;0:45.981;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;15;1:56.839;0;;26.969;0;44.185;0;45.685;0;168.6;30:58.613;12:16:26.943;0:26.969;0:44.185;0:45.685;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;16;1:56.872;0;;26.970;0;44.266;0;45.636;0;168.5;32:55.485;12:18:23.815;0:26.970;0:44.266;0:45.636;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;17;1:56.903;0;;26.965;0;44.231;0;45.707;0;168.5;34:52.388;12:20:20.718;0:26.965;0:44.231;0:45.707;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;18;1:56.817;0;;26.812;1;44.248;0;45.757;0;168.6;36:49.205;12:22:17.535;0:26.812;0:44.248;0:45.757;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;19;1:56.725;0;;26.889;0;44.174;0;45.662;0;168.8;38:45.930;12:24:14.260;0:26.889;0:44.174;0:45.662;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;20;1:56.978;0;;26.814;0;44.293;0;45.871;0;168.4;40:42.908;12:26:11.238;0:26.814;0:44.293;0:45.871;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;21;1:57.108;0;;26.887;0;44.370;0;45.851;0;168.2;42:40.016;12:28:08.346;0:26.887;0:44.370;0:45.851;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;22;1:56.931;0;;26.874;0;44.261;0;45.796;0;168.5;44:36.947;12:30:05.277;0:26.874;0:44.261;0:45.796;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;23;1:57.292;0;;27.093;0;44.500;0;45.699;0;167.9;46:34.239;12:32:02.569;0:27.093;0:44.500;0:45.699;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;24;1:57.041;0;;26.924;0;44.337;0;45.780;0;168.3;48:31.280;12:33:59.610;0:26.924;0:44.337;0:45.780;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;25;1:59.161;0;;27.431;0;44.892;0;46.838;0;165.3;50:30.441;12:35:58.771;0:27.431;0:44.892;0:46.838;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;1;26;2:09.956;0;B;26.890;0;44.347;0;58.719;0;151.6;52:40.397;12:38:08.727;0:26.890;0:44.347;0:58.719;;Jenson Altzman;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;27;3:23.398;0;;1:33.813;0;50.461;0;59.124;0;96.8;56:03.795;12:41:32.125;1:33.813;0:50.461;0:59.124;;Chad McCumbee;0:01:23.292;GS;;McCumbee McAleer Racing with Aerosport;Ford;FCY;
13;2;28;2:48.377;0;B;37.745;0;52.036;0;1:18.596;0;117.0;58:52.172;12:44:20.502;0:37.745;0:52.036;1:18.596;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;FCY;
13;2;29;2:58.494;0;;53.403;0;1:00.930;0;1:04.161;0;110.4;1:01:50.666;12:47:18.996;0:53.403;1:00.930;1:04.161;;Chad McCumbee;0:00:42.362;GS;;McCumbee McAleer Racing with Aerosport;Ford;FCY;
13;2;30;2:50.597;0;;33.942;0;1:01.592;0;1:15.063;0;115.5;1:04:41.263;12:50:09.593;0:33.942;1:01.592;1:15.063;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;FCY;
13;2;31;3:02.478;0;;52.527;0;1:13.497;0;56.454;0;107.9;1:07:43.741;12:53:12.071;0:52.527;1:13.497;0:56.454;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;32;1:57.903;0;;27.406;0;44.614;0;45.883;0;167.1;1:09:41.644;12:55:09.974;0:27.406;0:44.614;0:45.883;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;33;1:58.032;0;;26.825;0;45.295;0;45.912;0;166.9;1:11:39.676;12:57:08.006;0:26.825;0:45.295;0:45.912;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;34;1:56.656;0;;26.869;0;44.340;0;45.447;1;168.9;1:13:36.332;12:59:04.662;0:26.869;0:44.340;0:45.447;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;35;1:56.737;0;;26.820;0;44.308;0;45.609;0;168.7;1:15:33.069;13:01:01.399;0:26.820;0:44.308;0:45.609;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;36;1:56.870;0;;26.798;0;44.348;0;45.724;0;168.5;1:17:29.939;13:02:58.269;0:26.798;0:44.348;0:45.724;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;37;1:56.502;0;;26.854;0;43.860;2;45.788;0;169.1;1:19:26.441;13:04:54.771;0:26.854;0:43.860;0:45.788;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;38;1:56.691;0;;26.792;0;44.251;0;45.648;0;168.8;1:21:23.132;13:06:51.462;0:26.792;0:44.251;0:45.648;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;39;1:56.785;0;;26.950;0;44.168;0;45.667;0;168.7;1:23:19.917;13:08:48.247;0:26.950;0:44.168;0:45.667;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;40;1:56.851;0;;26.764;2;44.367;0;45.720;0;168.6;1:25:16.768;13:10:45.098;0:26.764;0:44.367;0:45.720;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;41;1:56.716;0;;26.998;0;44.021;0;45.697;0;168.8;1:27:13.484;13:12:41.814;0:26.998;0:44.021;0:45.697;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;42;1:57.279;0;;26.895;0;44.188;0;46.196;0;168.0;1:29:10.763;13:14:39.093;0:26.895;0:44.188;0:46.196;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;43;1:57.043;0;;26.844;0;44.450;0;45.749;0;168.3;1:31:07.806;13:16:36.136;0:26.844;0:44.450;0:45.749;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;44;1:56.962;0;;26.904;0;44.275;0;45.783;0;168.4;1:33:04.768;13:18:33.098;0:26.904;0:44.275;0:45.783;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;45;1:57.412;0;;27.203;0;44.163;0;46.046;0;167.8;1:35:02.180;13:20:30.510;0:27.203;0:44.163;0:46.046;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;46;1:57.328;0;;27.068;0;44.291;0;45.969;0;167.9;1:36:59.508;13:22:27.838;0:27.068;0:44.291;0:45.969;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;47;1:57.523;0;;26.979;0;44.693;0;45.851;0;167.6;1:38:57.031;13:24:25.361;0:26.979;0:44.693;0:45.851;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;48;1:56.843;0;;26.970;0;44.107;0;45.766;0;168.6;1:40:53.874;13:26:22.204;0:26.970;0:44.107;0:45.766;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;49;1:57.029;0;;27.003;0;44.410;0;45.616;0;168.3;1:42:50.903;13:28:19.233;0:27.003;0:44.410;0:45.616;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;50;1:57.397;0;;27.006;0;44.448;0;45.943;0;167.8;1:44:48.300;13:30:16.630;0:27.006;0:44.448;0:45.943;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;51;2:15.509;0;;26.940;0;47.932;0;1:00.637;0;145.4;1:47:03.809;13:32:32.139;0:26.940;0:47.932;1:00.637;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;FCY;
13;2;52;3:30.625;0;;54.473;0;1:29.924;0;1:06.228;0;93.5;1:50:34.434;13:36:02.764;0:54.473;1:29.924;1:06.228;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;53;1:58.389;0;;27.303;0;44.685;0;46.401;0;166.4;1:52:32.823;13:38:01.153;0:27.303;0:44.685;0:46.401;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;54;1:57.038;0;;26.868;0;44.444;0;45.726;0;168.3;1:54:29.861;13:39:58.191;0:26.868;0:44.444;0:45.726;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;GF;
13;2;55;2:02.355;0;;26.922;0;44.335;0;51.098;0;161.0;1:56:32.216;13:42:00.546;0:26.922;0:44.335;0:51.098;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;FCY;
13;2;56;3:22.694;0;;48.082;0;1:18.218;0;1:16.394;0;97.2;1:59:54.910;13:45:23.240;0:48.082;1:18.218;1:16.394;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;FCY;
13;2;57;3:26.339;0;;52.185;0;1:17.377;0;1:16.777;0;95.5;2:03:21.249;13:48:49.579;0:52.185;1:17.377;1:16.777;;Chad McCumbee;;GS;;McCumbee McAleer Racing with Aerosport;Ford;FF;
15;1;1;2:11.616;0;;34.501;0;47.510;0;49.605;0;149.7;2:11.616;11:47:39.946;0:34.501;0:47.510;0:49.605;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;2;2:03.810;0;;28.105;0;47.706;0;47.999;0;159.1;4:15.426;11:49:43.756;0:28.105;0:47.706;0:47.999;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;FCY;
15;1;3;2:30.886;0;;30.001;0;1:02.614;0;58.271;0;130.6;6:46.312;11:52:14.642;0:30.001;1:02.614;0:58.271;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;FCY;
15;1;4;2:53.292;0;;41.522;0;1:09.264;0;1:02.506;0;113.7;9:39.604;11:55:07.934;0:41.522;1:09.264;1:02.506;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;5;2:03.386;0;;27.915;0;47.940;0;47.531;0;159.6;11:42.990;11:57:11.320;0:27.915;0:47.940;0:47.531;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;6;2:06.338;0;;30.548;0;47.400;0;48.390;0;155.9;13:49.328;11:59:17.658;0:30.548;0:47.400;0:48.390;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;7;2:03.635;0;;30.078;0;46.257;0;47.300;0;159.3;15:52.963;12:01:21.293;0:30.078;0:46.257;0:47.300;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;8;2:02.560;0;;28.222;0;47.353;0;46.985;0;160.7;17:55.523;12:03:23.853;0:28.222;0:47.353;0:46.985;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;9;2:02.360;0;;27.673;0;46.383;0;48.304;0;161.0;19:57.883;12:05:26.213;0:27.673;0:46.383;0:48.304;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;10;2:01.215;0;;27.608;0;46.177;0;47.430;0;162.5;21:59.098;12:07:27.428;0:27.608;0:46.177;0:47.430;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;11;2:00.999;0;;27.654;0;45.852;1;47.493;0;162.8;24:00.097;12:09:28.427;0:27.654;0:45.852;0:47.493;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;12;2:01.730;0;;27.405;0;47.009;0;47.316;0;161.8;26:01.827;12:11:30.157;0:27.405;0:47.009;0:47.316;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;13;2:00.538;0;;27.621;0;46.134;0;46.783;1;163.4;28:02.365;12:13:30.695;0:27.621;0:46.134;0:46.783;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;14;2:01.447;0;;27.463;0;46.090;0;47.894;0;162.2;30:03.812;12:15:32.142;0:27.463;0:46.090;0:47.894;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;15;2:00.772;0;;27.404;0;46.135;0;47.233;0;163.1;32:04.584;12:17:32.914;0:27.404;0:46.135;0:47.233;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;16;2:00.718;0;;27.295;0;46.161;0;47.262;0;163.2;34:05.302;12:19:33.632;0:27.295;0:46.161;0:47.262;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;17;2:00.639;0;;27.257;1;46.141;0;47.241;0;163.3;36:05.941;12:21:34.271;0:27.257;0:46.141;0:47.241;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;18;2:00.870;0;;27.860;0;46.030;0;46.980;0;163.0;38:06.811;12:23:35.141;0:27.860;0:46.030;0:46.980;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;19;2:01.812;0;;27.565;0;46.870;0;47.377;0;161.7;40:08.623;12:25:36.953;0:27.565;0:46.870;0:47.377;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;20;2:01.261;0;;27.504;0;46.414;0;47.343;0;162.4;42:09.884;12:27:38.214;0:27.504;0:46.414;0:47.343;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;21;2:03.484;0;;27.777;0;46.901;0;48.806;0;159.5;44:13.368;12:29:41.698;0:27.777;0:46.901;0:48.806;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;22;2:02.557;0;;28.127;0;46.338;0;48.092;0;160.7;46:15.925;12:31:44.255;0:28.127;0:46.338;0:48.092;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;23;2:01.683;0;;27.814;0;46.410;0;47.459;0;161.9;48:17.608;12:33:45.938;0:27.814;0:46.410;0:47.459;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;24;2:03.596;0;;28.610;0;46.791;0;48.195;0;159.4;50:21.204;12:35:49.534;0:28.610;0:46.791;0:48.195;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;25;2:01.585;0;;27.741;0;46.170;0;47.674;0;162.0;52:22.789;12:37:51.119;0:27.741;0:46.170;0:47.674;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;26;2:02.926;0;;28.906;0;46.669;0;47.351;0;160.2;54:25.715;12:39:54.045;0:28.906;0:46.669;0:47.351;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;1;27;2:08.568;0;;29.467;0;49.684;0;49.417;0;153.2;56:34.283;12:42:02.613;0:29.467;0:49.684;0:49.417;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;FCY;
15;1;28;3:27.641;0;B;28.572;0;48.536;0;2:10.533;0;94.9;1:00:01.924;12:45:30.254;0:28.572;0:48.536;2:10.533;;Christine Sloss;;GS;B;van der Steur Racing;Aston Martin;FCY;
15;2;29;2:27.387;0;;45.977;0;51.758;0;49.652;0;133.7;1:02:29.311;12:47:57.641;0:45.977;0:51.758;0:49.652;;Ben Sloss;0:01:32.997;GS;B;van der Steur Racing;Aston Martin;FCY;
15;2;30;2:21.597;0;;28.956;0;48.317;0;1:04.324;0;139.1;1:04:50.908;12:50:19.238;0:28.956;0:48.317;1:04.324;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;FCY;
15;2;31;2:58.820;0;;53.290;0;1:13.013;0;52.517;0;110.2;1:07:49.728;12:53:18.058;0:53.290;1:13.013;0:52.517;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;32;1:58.431;2;;27.222;2;45.040;2;46.169;2;166.3;1:09:48.159;12:55:16.489;0:27.222;0:45.040;0:46.169;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;33;1:59.856;0;;27.452;0;45.682;0;46.722;0;164.4;1:11:48.015;12:57:16.345;0:27.452;0:45.682;0:46.722;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;34;2:02.706;0;;27.770;0;46.042;0;48.894;0;160.5;1:13:50.721;12:59:19.051;0:27.770;0:46.042;0:48.894;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;35;2:00.105;0;;27.337;0;46.177;0;46.591;0;164.0;1:15:50.826;13:01:19.156;0:27.337;0:46.177;0:46.591;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;36;2:01.761;0;;27.771;0;47.431;0;46.559;0;161.8;1:17:52.587;13:03:20.917;0:27.771;0:47.431;0:46.559;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;37;2:00.505;0;;28.223;0;45.515;0;46.767;0;163.5;1:19:53.092;13:05:21.422;0:28.223;0:45.515;0:46.767;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;38;2:00.352;0;;27.502;0;45.352;0;47.498;0;163.7;1:21:53.444;13:07:21.774;0:27.502;0:45.352;0:47.498;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;39;2:00.533;0;;28.257;0;45.805;0;46.471;0;163.4;1:23:53.977;13:09:22.307;0:28.257;0:45.805;0:46.471;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;40;1:59.823;0;;27.485;0;45.726;0;46.612;0;164.4;1:25:53.800;13:11:22.130;0:27.485;0:45.726;0:46.612;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;41;2:00.333;0;;27.601;0;45.840;0;46.892;0;163.7;1:27:54.133;13:13:22.463;0:27.601;0:45.840;0:46.892;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;42;2:01.020;0;;27.859;0;46.500;0;46.661;0;162.8;1:29:55.153;13:15:23.483;0:27.859;0:46.500;0:46.661;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;43;2:00.708;0;;28.390;0;45.696;0;46.622;0;163.2;1:31:55.861;13:17:24.191;0:28.390;0:45.696;0:46.622;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;44;2:00.123;0;;27.436;0;45.792;0;46.895;0;164.0;1:33:55.984;13:19:24.314;0:27.436;0:45.792;0:46.895;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;45;2:02.462;0;;28.524;0;47.239;0;46.699;0;160.9;1:35:58.446;13:21:26.776;0:28.524;0:47.239;0:46.699;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;46;2:00.077;0;;27.606;0;45.894;0;46.577;0;164.0;1:37:58.523;13:23:26.853;0:27.606;0:45.894;0:46.577;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;47;2:00.069;0;;27.648;0;45.726;0;46.695;0;164.1;1:39:58.592;13:25:26.922;0:27.648;0:45.726;0:46.695;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;48;2:00.326;0;;27.508;0;45.658;0;47.160;0;163.7;1:41:58.918;13:27:27.248;0:27.508;0:45.658;0:47.160;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;49;2:02.090;0;;27.621;0;47.473;0;46.996;0;161.3;1:44:01.008;13:29:29.338;0:27.621;0:47.473;0:46.996;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;50;2:01.059;0;;27.610;0;45.815;0;47.634;0;162.7;1:46:02.067;13:31:30.397;0:27.610;0:45.815;0:47.634;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;FCY;
15;2;51;2:09.548;0;;28.738;0;53.172;0;47.638;0;152.1;1:48:11.615;13:33:39.945;0:28.738;0:53.172;0:47.638;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;FCY;
15;2;52;2:27.558;0;;27.904;0;1:02.312;0;57.342;0;133.5;1:50:39.173;13:36:07.503;0:27.904;1:02.312;0:57.342;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;53;2:08.967;0;;27.932;0;53.586;0;47.449;0;152.7;1:52:48.140;13:38:16.470;0:27.932;0:53.586;0:47.449;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;54;2:01.665;0;;27.706;0;46.952;0;47.007;0;161.9;1:54:49.805;13:40:18.135;0:27.706;0:46.952;0:47.007;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;GF;
15;2;55;2:13.918;0;;28.023;0;48.650;0;57.245;0;147.1;1:57:03.723;13:42:32.053;0:28.023;0:48.650;0:57.245;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;FCY;
15;2;56;3:14.725;0;;40.321;0;1:16.439;0;1:17.965;0;101.2;2:00:18.448;13:45:46.778;0:40.321;1:16.439;1:17.965;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;FCY;
15;2;57;3:27.476;0;;56.543;0;1:12.963;0;1:17.970;0;94.9;2:03:45.924;13:49:14.254;0:56.543;1:12.963;1:17.970;;Ben Sloss;;GS;B;van der Steur Racing;Aston Martin;FF;
16;2;1;2:03.743;0;;30.964;0;46.180;0;46.599;0;159.2;2:03.743;11:47:32.073;0:30.964;0:46.180;0:46.599;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;2;1:58.895;0;;27.369;0;44.983;1;46.543;0;165.7;4:02.638;11:49:30.968;0:27.369;0:44.983;0:46.543;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;3;2:29.201;0;;30.086;0;59.983;0;59.132;0;132.0;6:31.839;11:52:00.169;0:30.086;0:59.983;0:59.132;;Nikita Lastochkin;;GS;;CSM;Porsche;FCY;
16;2;4;3:04.988;0;;43.881;0;1:16.695;0;1:04.412;0;106.5;9:36.827;11:55:05.157;0:43.881;1:16.695;1:04.412;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;5;1:58.943;0;;27.179;1;45.301;0;46.463;0;165.6;11:35.770;11:57:04.100;0:27.179;0:45.301;0:46.463;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;6;1:58.652;0;;27.229;0;45.069;0;46.354;0;166.0;13:34.422;11:59:02.752;0:27.229;0:45.069;0:46.354;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;7;1:58.920;0;;27.392;0;45.208;0;46.320;1;165.6;15:33.342;12:01:01.672;0:27.392;0:45.208;0:46.320;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;8;1:59.063;0;;27.295;0;45.149;0;46.619;0;165.4;17:32.405;12:03:00.735;0:27.295;0:45.149;0:46.619;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;9;2:00.873;0;;27.720;0;46.442;0;46.711;0;163.0;19:33.278;12:05:01.608;0:27.720;0:46.442;0:46.711;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;10;2:00.344;0;;27.599;0;45.624;0;47.121;0;163.7;21:33.622;12:07:01.952;0:27.599;0:45.624;0:47.121;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;11;2:02.282;0;;28.789;0;46.678;0;46.815;0;161.1;23:35.904;12:09:04.234;0:28.789;0:46.678;0:46.815;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;12;1:59.732;0;;27.359;0;45.676;0;46.697;0;164.5;25:35.636;12:11:03.966;0:27.359;0:45.676;0:46.697;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;13;1:59.140;0;;27.398;0;45.365;0;46.377;0;165.3;27:34.776;12:13:03.106;0:27.398;0:45.365;0:46.377;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;14;1:59.277;0;;27.323;0;45.473;0;46.481;0;165.1;29:34.053;12:15:02.383;0:27.323;0:45.473;0:46.481;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;15;1:59.863;0;;27.388;0;45.741;0;46.734;0;164.3;31:33.916;12:17:02.246;0:27.388;0:45.741;0:46.734;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;16;2:01.213;0;;27.429;0;46.411;0;47.373;0;162.5;33:35.129;12:19:03.459;0:27.429;0:46.411;0:47.373;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;17;1:59.769;0;;27.475;0;45.616;0;46.678;0;164.5;35:34.898;12:21:03.228;0:27.475;0:45.616;0:46.678;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;18;2:00.384;0;;27.363;0;46.116;0;46.905;0;163.6;37:35.282;12:23:03.612;0:27.363;0:46.116;0:46.905;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;19;1:59.486;0;;27.263;0;45.770;0;46.453;0;164.9;39:34.768;12:25:03.098;0:27.263;0:45.770;0:46.453;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;20;1:59.801;0;;27.339;0;45.773;0;46.689;0;164.4;41:34.569;12:27:02.899;0:27.339;0:45.773;0:46.689;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;21;1:59.599;0;;27.391;0;45.537;0;46.671;0;164.7;43:34.168;12:29:02.498;0:27.391;0:45.537;0:46.671;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;22;1:59.560;0;;27.433;0;45.569;0;46.558;0;164.8;45:33.728;12:31:02.058;0:27.433;0:45.569;0:46.558;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;23;1:59.566;0;;27.464;0;45.702;0;46.400;0;164.7;47:33.294;12:33:01.624;0:27.464;0:45.702;0:46.400;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;24;2:01.048;0;;27.402;0;46.606;0;47.040;0;162.7;49:34.342;12:35:02.672;0:27.402;0:46.606;0:47.040;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;25;2:00.930;0;;27.518;0;46.350;0;47.062;0;162.9;51:35.272;12:37:03.602;0:27.518;0:46.350;0:47.062;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;26;2:01.094;0;;27.507;0;46.873;0;46.714;0;162.7;53:36.366;12:39:04.696;0:27.507;0:46.873;0:46.714;;Nikita Lastochkin;;GS;;CSM;Porsche;GF;
16;2;27;2:07.128;0;;27.620;0;47.892;0;51.616;0;154.9;55:43.494;12:41:11.824;0:27.620;0:47.892;0:51.616;;Nikita Lastochkin;;GS;;CSM;Porsche;FCY;
16;2;28;3:04.807;0;B;29.871;0;1:02.647;0;1:32.289;0;106.6;58:48.301;12:44:16.631;0:29.871;1:02.647;1:32.289;;Nikita Lastochkin;;GS;;CSM;Porsche;FCY;
16;1;29;3:42.481;0;;2:03.779;0;49.713;0;48.989;0;88.5;1:02:30.782;12:47:59.112;2:03.779;0:49.713;0:48.989;;Zach Veach;0:01:52.913;GS;;CSM;Porsche;FCY;
16;1;30;2:21.378;0;;29.220;0;47.518;0;1:04.640;0;139.3;1:04:52.160;12:50:20.490;0:29.220;0:47.518;1:04.640;;Zach Veach;;GS;;CSM;Porsche;FCY;
16;1;31;2:58.162;0;;53.365;0;1:12.791;0;52.006;0;110.6;1:07:50.322;12:53:18.652;0:53.365;1:12.791;0:52.006;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;32;2:11.186;0;B;26.994;2;45.240;0;58.952;0;150.2;1:10:01.508;12:55:29.838;0:26.994;0:45.240;0:58.952;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;33;2:09.317;0;;38.779;0;44.687;0;45.851;0;152.3;1:12:10.825;12:57:39.155;0:38.779;0:44.687;0:45.851;;Zach Veach;0:00:30.482;GS;;CSM;Porsche;GF;
16;1;34;1:57.291;0;;27.019;0;44.399;0;45.873;0;167.9;1:14:08.116;12:59:36.446;0:27.019;0:44.399;0:45.873;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;35;1:57.124;2;;27.070;0;44.231;2;45.823;2;168.2;1:16:05.240;13:01:33.570;0:27.070;0:44.231;0:45.823;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;36;1:57.800;0;;27.078;0;44.617;0;46.105;0;167.2;1:18:03.040;13:03:31.370;0:27.078;0:44.617;0:46.105;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;37;1:58.843;0;;27.512;0;45.250;0;46.081;0;165.8;1:20:01.883;13:05:30.213;0:27.512;0:45.250;0:46.081;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;38;1:58.280;0;;27.208;0;44.718;0;46.354;0;166.5;1:22:00.163;13:07:28.493;0:27.208;0:44.718;0:46.354;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;39;1:58.598;0;;27.181;0;45.065;0;46.352;0;166.1;1:23:58.761;13:09:27.091;0:27.181;0:45.065;0:46.352;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;40;1:59.671;0;;27.042;0;45.984;0;46.645;0;164.6;1:25:58.432;13:11:26.762;0:27.042;0:45.984;0:46.645;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;41;1:58.454;0;;27.487;0;44.663;0;46.304;0;166.3;1:27:56.886;13:13:25.216;0:27.487;0:44.663;0:46.304;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;42;1:58.538;0;;27.095;0;45.101;0;46.342;0;166.2;1:29:55.424;13:15:23.754;0:27.095;0:45.101;0:46.342;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;43;1:58.625;0;;27.336;0;44.903;0;46.386;0;166.1;1:31:54.049;13:17:22.379;0:27.336;0:44.903;0:46.386;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;44;1:58.490;0;;27.274;0;44.912;0;46.304;0;166.2;1:33:52.539;13:19:20.869;0:27.274;0:44.912;0:46.304;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;45;1:58.300;0;;27.285;0;44.810;0;46.205;0;166.5;1:35:50.839;13:21:19.169;0:27.285;0:44.810;0:46.205;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;46;1:58.675;0;;27.180;0;45.018;0;46.477;0;166.0;1:37:49.514;13:23:17.844;0:27.180;0:45.018;0:46.477;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;47;1:58.669;0;;27.351;0;44.900;0;46.418;0;166.0;1:39:48.183;13:25:16.513;0:27.351;0:44.900;0:46.418;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;48;1:58.482;0;;27.156;0;44.929;0;46.397;0;166.3;1:41:46.665;13:27:14.995;0:27.156;0:44.929;0:46.397;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;49;1:58.623;0;;27.203;0;44.960;0;46.460;0;166.1;1:43:45.288;13:29:13.618;0:27.203;0:44.960;0:46.460;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;50;1:58.670;0;;27.533;0;44.955;0;46.182;0;166.0;1:45:43.958;13:31:12.288;0:27.533;0:44.955;0:46.182;;Zach Veach;;GS;;CSM;Porsche;FCY;
16;1;51;2:11.002;0;;28.981;0;53.381;0;48.640;0;150.4;1:47:54.960;13:33:23.290;0:28.981;0:53.381;0:48.640;;Zach Veach;;GS;;CSM;Porsche;FCY;
16;1;52;2:43.780;0;;29.199;0;1:17.018;0;57.563;0;120.3;1:50:38.740;13:36:07.070;0:29.199;1:17.018;0:57.563;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;53;2:01.259;0;;27.537;0;46.785;0;46.937;0;162.4;1:52:39.999;13:38:08.329;0:27.537;0:46.785;0:46.937;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;54;1:59.202;0;;27.190;0;45.477;0;46.535;0;165.3;1:54:39.201;13:40:07.531;0:27.190;0:45.477;0:46.535;;Zach Veach;;GS;;CSM;Porsche;GF;
16;1;55;2:02.271;0;;27.425;0;45.667;0;49.179;0;161.1;1:56:41.472;13:42:09.802;0:27.425;0:45.667;0:49.179;;Zach Veach;;GS;;CSM;Porsche;FCY;
16;1;56;3:24.874;0;;48.443;0;1:19.546;0;1:16.885;0;96.1;2:00:06.346;13:45:34.676;0:48.443;1:19.546;1:16.885;;Zach Veach;;GS;;CSM;Porsche;FCY;
16;1;57;3:26.061;0;;53.435;0;1:15.789;0;1:16.837;0;95.6;2:03:32.407;13:49:00.737;0:53.435;1:15.789;1:16.837;;Zach Veach;;GS;;CSM;Porsche;FF;
17;1;1;2:05.357;0;;32.018;0;46.658;0;46.681;0;157.1;2:05.357;11:47:33.687;0:32.018;0:46.658;0:46.681;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;2;1:58.928;0;;27.755;0;45.136;0;46.037;0;165.6;4:04.285;11:49:32.615;0:27.755;0:45.136;0:46.037;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;3;2:28.950;0;;29.829;0;59.787;0;59.334;0;132.2;6:33.235;11:52:01.565;0:29.829;0:59.787;0:59.334;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;FCY;
17;1;4;3:04.258;0;;43.627;0;1:16.770;0;1:03.861;0;106.9;9:37.493;11:55:05.823;0:43.627;1:16.770;1:03.861;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;5;1:59.444;0;;27.628;0;45.884;0;45.932;1;164.9;11:36.937;11:57:05.267;0:27.628;0:45.884;0:45.932;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;6;1:58.369;0;;27.508;0;44.815;0;46.046;0;166.4;13:35.306;11:59:03.636;0:27.508;0:44.815;0:46.046;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;7;1:58.275;0;;27.553;0;44.686;1;46.036;0;166.5;15:33.581;12:01:01.911;0:27.553;0:44.686;0:46.036;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;8;1:58.801;0;;27.183;0;45.331;0;46.287;0;165.8;17:32.382;12:03:00.712;0:27.183;0:45.331;0:46.287;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;9;1:58.579;0;;27.425;0;44.799;0;46.355;0;166.1;19:30.961;12:04:59.291;0:27.425;0:44.799;0:46.355;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;10;2:10.593;0;;37.160;0;46.773;0;46.660;0;150.8;21:41.554;12:07:09.884;0:37.160;0:46.773;0:46.660;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;11;2:00.441;0;;27.437;0;46.195;0;46.809;0;163.6;23:41.995;12:09:10.325;0:27.437;0:46.195;0:46.809;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;12;1:58.512;0;;27.192;0;44.898;0;46.422;0;166.2;25:40.507;12:11:08.837;0:27.192;0:44.898;0:46.422;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;13;2:00.034;0;;27.969;0;45.490;0;46.575;0;164.1;27:40.541;12:13:08.871;0:27.969;0:45.490;0:46.575;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;14;2:00.022;0;;27.150;1;46.265;0;46.607;0;164.1;29:40.563;12:15:08.893;0:27.150;0:46.265;0:46.607;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;15;2:00.952;0;;28.228;0;45.854;0;46.870;0;162.9;31:41.515;12:17:09.845;0:28.228;0:45.854;0:46.870;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;16;1:59.746;0;;27.593;0;45.755;0;46.398;0;164.5;33:41.261;12:19:09.591;0:27.593;0:45.755;0:46.398;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;17;1:58.784;0;;27.206;0;44.688;0;46.890;0;165.8;35:40.045;12:21:08.375;0:27.206;0:44.688;0:46.890;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;18;1:58.912;0;;27.324;0;45.114;0;46.474;0;165.7;37:38.957;12:23:07.287;0:27.324;0:45.114;0:46.474;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;19;1:59.124;0;;27.334;0;45.284;0;46.506;0;165.4;39:38.081;12:25:06.411;0:27.334;0:45.284;0:46.506;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;20;1:58.868;0;;27.161;0;45.258;0;46.449;0;165.7;41:36.949;12:27:05.279;0:27.161;0:45.258;0:46.449;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;21;1:59.012;0;;27.425;0;45.112;0;46.475;0;165.5;43:35.961;12:29:04.291;0:27.425;0:45.112;0:46.475;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;22;1:59.760;0;;27.501;0;45.343;0;46.916;0;164.5;45:35.721;12:31:04.051;0:27.501;0:45.343;0:46.916;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;23;2:02.390;0;;27.715;0;46.655;0;48.020;0;160.9;47:38.111;12:33:06.441;0:27.715;0:46.655;0:48.020;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;24;2:00.240;0;;27.930;0;45.370;0;46.940;0;163.8;49:38.351;12:35:06.681;0:27.930;0:45.370;0:46.940;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;25;2:01.264;0;;28.635;0;45.700;0;46.929;0;162.4;51:39.615;12:37:07.945;0:28.635;0:45.700;0:46.929;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;26;2:02.303;0;;27.786;0;46.086;0;48.431;0;161.1;53:41.918;12:39:10.248;0:27.786;0:46.086;0:48.431;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;1;27;2:13.225;0;;27.919;0;49.743;0;55.563;0;147.9;55:55.143;12:41:23.473;0:27.919;0:49.743;0:55.563;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;FCY;
17;1;28;2:53.927;0;B;35.420;0;54.982;0;1:23.525;0;113.3;58:49.070;12:44:17.400;0:35.420;0:54.982;1:23.525;;Chris Miller;;GS;;Unitronic/JDC Miller MotorSports;Porsche;FCY;
17;2;29;3:17.839;0;;1:36.487;0;49.484;0;51.868;0;99.6;1:02:06.909;12:47:35.239;1:36.487;0:49.484;0:51.868;;Mikey Taylor;0:01:25.240;GS;;Unitronic/JDC Miller MotorSports;Porsche;FCY;
17;2;30;2:42.200;0;;35.596;0;54.355;0;1:12.249;0;121.4;1:04:49.109;12:50:17.439;0:35.596;0:54.355;1:12.249;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;FCY;
17;2;31;2:58.720;0;;53.495;0;1:12.693;0;52.532;0;110.2;1:07:47.829;12:53:16.159;0:53.495;1:12.693;0:52.532;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;32;1:57.752;0;;27.055;0;44.930;0;45.767;2;167.3;1:09:45.581;12:55:13.911;0:27.055;0:44.930;0:45.767;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;33;1:57.614;0;;27.136;0;44.643;0;45.835;0;167.5;1:11:43.195;12:57:11.525;0:27.136;0:44.643;0:45.835;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;34;1:57.317;2;;27.087;0;44.166;2;46.064;0;167.9;1:13:40.512;12:59:08.842;0:27.087;0:44.166;0:46.064;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;35;1:57.345;0;;27.086;0;44.427;0;45.832;0;167.9;1:15:37.857;13:01:06.187;0:27.086;0:44.427;0:45.832;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;36;1:58.141;0;;27.197;0;44.405;0;46.539;0;166.7;1:17:35.998;13:03:04.328;0:27.197;0:44.405;0:46.539;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;37;1:57.597;0;;27.437;0;44.252;0;45.908;0;167.5;1:19:33.595;13:05:01.925;0:27.437;0:44.252;0:45.908;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;38;1:57.657;0;;27.088;0;44.538;0;46.031;0;167.4;1:21:31.252;13:06:59.582;0:27.088;0:44.538;0:46.031;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;39;1:57.786;0;;27.146;0;44.536;0;46.104;0;167.2;1:23:29.038;13:08:57.368;0:27.146;0:44.536;0:46.104;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;40;1:57.732;0;;27.107;0;44.578;0;46.047;0;167.3;1:25:26.770;13:10:55.100;0:27.107;0:44.578;0:46.047;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;41;1:57.547;0;;27.104;0;44.442;0;46.001;0;167.6;1:27:24.317;13:12:52.647;0:27.104;0:44.442;0:46.001;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;42;1:58.148;0;;27.162;0;44.718;0;46.268;0;166.7;1:29:22.465;13:14:50.795;0:27.162;0:44.718;0:46.268;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;43;1:58.145;0;;27.681;0;44.393;0;46.071;0;166.7;1:31:20.610;13:16:48.940;0:27.681;0:44.393;0:46.071;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;44;1:57.606;0;;27.131;0;44.512;0;45.963;0;167.5;1:33:18.216;13:18:46.546;0:27.131;0:44.512;0:45.963;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;45;1:57.451;0;;27.124;0;44.358;0;45.969;0;167.7;1:35:15.667;13:20:43.997;0:27.124;0:44.358;0:45.969;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;46;1:57.495;0;;27.114;0;44.503;0;45.878;0;167.7;1:37:13.162;13:22:41.492;0:27.114;0:44.503;0:45.878;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;47;1:57.901;0;;27.092;0;44.349;0;46.460;0;167.1;1:39:11.063;13:24:39.393;0:27.092;0:44.349;0:46.460;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;48;1:57.468;0;;27.110;0;44.492;0;45.866;0;167.7;1:41:08.531;13:26:36.861;0:27.110;0:44.492;0:45.866;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;49;1:57.379;0;;27.049;2;44.377;0;45.953;0;167.8;1:43:05.910;13:28:34.240;0:27.049;0:44.377;0:45.953;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;50;1:57.545;0;;27.117;0;44.514;0;45.914;0;167.6;1:45:03.455;13:30:31.785;0:27.117;0:44.514;0:45.914;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;51;2:10.786;0;;27.140;0;48.263;0;55.383;0;150.6;1:47:14.241;13:32:42.571;0:27.140;0:48.263;0:55.383;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;FCY;
17;2;52;3:22.839;0;;52.659;0;1:30.376;0;59.804;0;97.1;1:50:37.080;13:36:05.410;0:52.659;1:30.376;0:59.804;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;53;2:00.021;0;;27.257;0;46.729;0;46.035;0;164.1;1:52:37.101;13:38:05.431;0:27.257;0:46.729;0:46.035;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;54;1:57.761;0;;27.217;0;44.530;0;46.014;0;167.3;1:54:34.862;13:40:03.192;0:27.217;0:44.530;0:46.014;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;GF;
17;2;55;2:03.506;0;;27.164;0;44.554;0;51.788;0;159.5;1:56:38.368;13:42:06.698;0:27.164;0:44.554;0:51.788;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;FCY;
17;2;56;3:24.471;0;;48.239;0;1:19.137;0;1:17.095;0;96.3;2:00:02.839;13:45:31.169;0:48.239;1:19.137;1:17.095;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;FCY;
17;2;57;3:26.772;0;;52.090;0;1:16.670;0;1:18.012;0;95.3;2:03:29.611;13:48:57.941;0:52.090;1:16.670;1:18.012;;Mikey Taylor;;GS;;Unitronic/JDC Miller MotorSports;Porsche;FF;
18;1;1;2:15.203;0;;41.034;0;46.649;0;47.520;0;145.7;2:15.203;11:47:43.533;0:41.034;0:46.649;0:47.520;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;2;2:01.759;0;;28.371;0;46.481;0;46.907;0;161.8;4:16.962;11:49:45.292;0:28.371;0:46.481;0:46.907;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;1;3;2:32.187;0;;33.823;0;1:00.138;0;58.226;0;129.4;6:49.149;11:52:17.479;0:33.823;1:00.138;0:58.226;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;1;4;2:55.190;0;;42.863;0;1:13.851;0;58.476;0;112.4;9:44.339;11:55:12.669;0:42.863;1:13.851;0:58.476;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;5;2:03.422;0;;28.195;0;47.240;0;47.987;0;159.6;11:47.761;11:57:16.091;0:28.195;0:47.240;0:47.987;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;6;2:01.257;0;;28.155;0;45.325;0;47.777;0;162.5;13:49.018;11:59:17.348;0:28.155;0:45.325;0:47.777;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;7;1:59.269;0;;28.062;0;44.754;0;46.453;1;165.2;15:48.287;12:01:16.617;0:28.062;0:44.754;0:46.453;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;8;2:00.063;0;;27.914;1;45.432;0;46.717;0;164.1;17:48.350;12:03:16.680;0:27.914;0:45.432;0:46.717;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;9;2:00.490;0;;28.077;0;45.327;0;47.086;0;163.5;19:48.840;12:05:17.170;0:28.077;0:45.327;0:47.086;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;10;2:01.646;0;;28.744;0;46.012;0;46.890;0;161.9;21:50.486;12:07:18.816;0:28.744;0:46.012;0:46.890;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;11;2:00.412;0;;27.997;0;45.454;0;46.961;0;163.6;23:50.898;12:09:19.228;0:27.997;0:45.454;0:46.961;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;12;2:01.318;0;;28.136;0;45.794;0;47.388;0;162.4;25:52.216;12:11:20.546;0:28.136;0:45.794;0:47.388;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;13;2:01.324;0;;28.099;0;46.011;0;47.214;0;162.4;27:53.540;12:13:21.870;0:28.099;0:46.011;0:47.214;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;14;2:00.171;0;;27.993;0;45.304;0;46.874;0;163.9;29:53.711;12:15:22.041;0:27.993;0:45.304;0:46.874;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;15;2:00.478;0;;28.122;0;45.379;0;46.977;0;163.5;31:54.189;12:17:22.519;0:28.122;0:45.379;0:46.977;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;16;2:00.555;0;;28.072;0;45.559;0;46.924;0;163.4;33:54.744;12:19:23.074;0:28.072;0:45.559;0:46.924;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;17;2:00.196;0;;28.063;0;45.129;0;47.004;0;163.9;35:54.940;12:21:23.270;0:28.063;0:45.129;0:47.004;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;18;2:00.707;0;;28.079;0;45.965;0;46.663;0;163.2;37:55.647;12:23:23.977;0:28.079;0:45.965;0:46.663;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;19;2:00.983;0;;27.922;0;46.084;0;46.977;0;162.8;39:56.630;12:25:24.960;0:27.922;0:46.084;0:46.977;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;20;2:00.642;0;;27.978;0;45.332;0;47.332;0;163.3;41:57.272;12:27:25.602;0:27.978;0:45.332;0:47.332;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;21;1:59.903;0;;27.976;0;45.142;0;46.785;0;164.3;43:57.175;12:29:25.505;0:27.976;0:45.142;0:46.785;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;22;1:59.825;0;;28.123;0;44.955;0;46.747;0;164.4;45:57.000;12:31:25.330;0:28.123;0:44.955;0:46.747;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;23;1:59.573;0;;28.024;0;44.927;0;46.622;0;164.7;47:56.573;12:33:24.903;0:28.024;0:44.927;0:46.622;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;24;1:59.655;0;;28.112;0;44.744;1;46.799;0;164.6;49:56.228;12:35:24.558;0:28.112;0:44.744;0:46.799;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;25;2:00.473;0;;28.536;0;44.779;0;47.158;0;163.5;51:56.701;12:37:25.031;0:28.536;0:44.779;0:47.158;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;26;2:02.013;0;;28.010;0;47.053;0;46.950;0;161.4;53:58.714;12:39:27.044;0:28.010;0:47.053;0:46.950;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;1;27;2:14.431;0;;28.049;0;49.245;0;57.137;0;146.5;56:13.145;12:41:41.475;0:28.049;0:49.245;0:57.137;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;1;28;2:31.185;0;;35.192;0;55.333;0;1:00.660;0;130.3;58:44.330;12:44:12.660;0:35.192;0:55.333;1:00.660;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;1;29;3:13.377;0;B;45.822;0;1:09.519;0;1:18.036;0;101.9;1:01:57.707;12:47:26.037;0:45.822;1:09.519;1:18.036;;Lance Bergstein;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;2;30;3:28.895;0;;1:42.057;0;55.914;0;50.924;0;94.3;1:05:26.602;12:50:54.932;1:42.057;0:55.914;0:50.924;;Jon Miller;0:01:27.248;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;2;31;2:34.140;0;;34.640;0;1:07.542;0;51.958;0;127.8;1:08:00.742;12:53:29.072;0:34.640;1:07.542;0:51.958;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;32;1:59.002;0;;27.840;0;45.085;0;46.077;2;165.5;1:09:59.744;12:55:28.074;0:27.840;0:45.085;0:46.077;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;33;1:58.071;2;;27.775;0;44.117;2;46.179;0;166.8;1:11:57.815;12:57:26.145;0:27.775;0:44.117;0:46.179;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;34;1:58.476;0;;27.857;0;44.309;0;46.310;0;166.3;1:13:56.291;12:59:24.621;0:27.857;0:44.309;0:46.310;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;35;1:58.693;0;;27.703;2;44.241;0;46.749;0;166.0;1:15:54.984;13:01:23.314;0:27.703;0:44.241;0:46.749;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;36;1:59.723;0;;28.294;0;44.920;0;46.509;0;164.5;1:17:54.707;13:03:23.037;0:28.294;0:44.920;0:46.509;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;37;2:00.350;0;;28.026;0;45.423;0;46.901;0;163.7;1:19:55.057;13:05:23.387;0:28.026;0:45.423;0:46.901;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;38;2:01.823;0;;28.119;0;45.603;0;48.101;0;161.7;1:21:56.880;13:07:25.210;0:28.119;0:45.603;0:48.101;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;39;1:59.969;0;;28.128;0;45.205;0;46.636;0;164.2;1:23:56.849;13:09:25.179;0:28.128;0:45.205;0:46.636;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;40;2:01.321;0;;28.202;0;46.041;0;47.078;0;162.4;1:25:58.170;13:11:26.500;0:28.202;0:46.041;0:47.078;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;41;2:02.130;0;;28.409;0;45.623;0;48.098;0;161.3;1:28:00.300;13:13:28.630;0:28.409;0:45.623;0:48.098;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;42;2:00.451;0;;28.074;0;45.361;0;47.016;0;163.5;1:30:00.751;13:15:29.081;0:28.074;0:45.361;0:47.016;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;43;2:00.585;0;;28.277;0;45.029;0;47.279;0;163.4;1:32:01.336;13:17:29.666;0:28.277;0:45.029;0:47.279;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;44;2:02.487;0;;28.258;0;45.099;0;49.130;0;160.8;1:34:03.823;13:19:32.153;0:28.258;0:45.099;0:49.130;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;45;2:20.018;0;B;28.600;0;46.756;0;1:04.662;0;140.7;1:36:23.841;13:21:52.171;0:28.600;0:46.756;1:04.662;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;46;3:16.426;0;;1:43.999;0;45.625;0;46.802;0;100.3;1:39:40.267;13:25:08.597;1:43.999;0:45.625;0:46.802;;Jon Miller;0:01:33.233;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;47;2:13.007;0;B;28.334;0;44.971;0;59.702;0;148.1;1:41:53.274;13:27:21.604;0:28.334;0:44.971;0:59.702;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;48;2:12.256;0;;39.832;0;45.190;0;47.234;0;148.9;1:44:05.530;13:29:33.860;0:39.832;0:45.190;0:47.234;;Jon Miller;0:00:30.715;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;49;2:01.932;0;;28.566;0;45.740;0;47.626;0;161.6;1:46:07.462;13:31:35.792;0:28.566;0:45.740;0:47.626;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;2;50;2:28.622;0;B;31.938;0;52.180;0;1:04.504;0;132.5;1:48:36.084;13:34:04.414;0:31.938;0:52.180;1:04.504;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;2;51;3:29.729;0;;1:55.332;0;47.662;0;46.735;0;93.9;1:52:05.813;13:37:34.143;1:55.332;0:47.662;0:46.735;;Jon Miller;0:01:44.503;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;52;1:58.845;0;;27.950;0;44.529;0;46.366;0;165.7;1:54:04.658;13:39:32.988;0:27.950;0:44.529;0:46.366;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;GF;
18;2;53;1:59.181;0;;28.140;0;44.587;0;46.454;0;165.3;1:56:03.839;13:41:32.169;0:28.140;0:44.587;0:46.454;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;2;54;2:06.002;0;;30.674;0;47.812;0;47.516;0;156.3;1:58:09.841;13:43:38.171;0:30.674;0:47.812;0:47.516;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;2;55;2:10.203;0;;28.720;0;45.596;0;55.887;0;151.3;2:00:20.044;13:45:48.374;0:28.720;0:45.596;0:55.887;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;FCY;
18;2;56;3:27.904;0;;57.070;0;1:12.785;0;1:18.049;0;94.7;2:03:47.948;13:49:16.278;0:57.070;1:12.785;1:18.049;;Jon Miller;;TCR;;Victor Gonzalez Racing;Hyundai;FF;
19;1;1;2:08.897;0;;34.906;0;47.289;0;46.702;0;152.8;2:08.897;11:47:37.227;0:34.906;0:47.289;0:46.702;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;2;1:59.401;0;;27.361;0;45.432;0;46.608;0;165.0;4:08.298;11:49:36.628;0:27.361;0:45.432;0:46.608;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;3;2:29.922;0;;30.481;0;58.878;0;1:00.563;0;131.4;6:38.220;11:52:06.550;0:30.481;0:58.878;1:00.563;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;FCY;
19;1;4;3:00.692;0;;43.778;0;1:14.518;0;1:02.396;0;109.0;9:38.912;11:55:07.242;0:43.778;1:14.518;1:02.396;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;5;2:00.564;0;;27.752;0;45.960;0;46.852;0;163.4;11:39.476;11:57:07.806;0:27.752;0:45.960;0:46.852;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;6;1:59.975;0;;28.092;0;45.310;0;46.573;0;164.2;13:39.451;11:59:07.781;0:28.092;0:45.310;0:46.573;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;7;2:00.122;0;;27.293;0;46.774;0;46.055;1;164.0;15:39.573;12:01:07.903;0:27.293;0:46.774;0:46.055;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;8;2:00.459;0;;27.232;1;45.742;0;47.485;0;163.5;17:40.032;12:03:08.362;0:27.232;0:45.742;0:47.485;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;9;1:59.091;0;;27.262;0;45.273;0;46.556;0;165.4;19:39.123;12:05:07.453;0:27.262;0:45.273;0:46.556;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;10;2:01.862;0;;28.722;0;46.224;0;46.916;0;161.6;21:40.985;12:07:09.315;0:28.722;0:46.224;0:46.916;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;11;1:59.281;0;;27.285;0;45.387;0;46.609;0;165.1;23:40.266;12:09:08.596;0:27.285;0:45.387;0:46.609;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;12;1:59.062;0;;27.613;0;45.056;0;46.393;0;165.4;25:39.328;12:11:07.658;0:27.613;0:45.056;0:46.393;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;13;2:01.683;0;;29.391;0;45.607;0;46.685;0;161.9;27:41.011;12:13:09.341;0:29.391;0:45.607;0:46.685;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;14;1:59.880;0;;27.423;0;45.779;0;46.678;0;164.3;29:40.891;12:15:09.221;0:27.423;0:45.779;0:46.678;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;15;2:01.531;0;;28.363;0;45.629;0;47.539;0;162.1;31:42.422;12:17:10.752;0:28.363;0:45.629;0:47.539;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;16;2:00.159;0;;27.531;0;45.927;0;46.701;0;163.9;33:42.581;12:19:10.911;0:27.531;0:45.927;0:46.701;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;17;1:58.608;0;;27.278;0;44.879;0;46.451;0;166.1;35:41.189;12:21:09.519;0:27.278;0:44.879;0:46.451;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;18;1:58.907;0;;27.336;0;44.849;0;46.722;0;165.7;37:40.096;12:23:08.426;0:27.336;0:44.849;0:46.722;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;19;1:58.610;0;;27.484;0;44.771;1;46.355;0;166.1;39:38.706;12:25:07.036;0:27.484;0:44.771;0:46.355;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;20;1:58.925;0;;27.365;0;44.977;0;46.583;0;165.6;41:37.631;12:27:05.961;0:27.365;0:44.977;0:46.583;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;21;1:59.237;0;;27.420;0;45.296;0;46.521;0;165.2;43:36.868;12:29:05.198;0:27.420;0:45.296;0:46.521;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;22;2:00.679;0;;27.444;0;45.756;0;47.479;0;163.2;45:37.547;12:31:05.877;0:27.444;0:45.756;0:47.479;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;23;2:00.713;0;;27.985;0;45.270;0;47.458;0;163.2;47:38.260;12:33:06.590;0:27.985;0:45.270;0:47.458;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;24;2:00.235;0;;27.941;0;45.519;0;46.775;0;163.8;49:38.495;12:35:06.825;0:27.941;0:45.519;0:46.775;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;25;2:01.258;0;;28.685;0;45.701;0;46.872;0;162.5;51:39.753;12:37:08.083;0:28.685;0:45.701;0:46.872;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;26;2:02.648;0;;28.922;0;45.817;0;47.909;0;160.6;53:42.401;12:39:10.731;0:28.922;0:45.817;0:47.909;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;GF;
19;1;27;2:18.050;0;;27.566;0;51.865;0;58.619;0;142.7;56:00.451;12:41:28.781;0:27.566;0:51.865;0:58.619;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;FCY;
19;1;28;3:47.877;0;B;35.206;0;50.594;0;2:22.077;0;86.4;59:48.328;12:45:16.658;0:35.206;0:50.594;2:22.077;;Sean Quinlan;;GS;;Stephen Cameron Racing;Ford;FCY;
19;2;29;2:19.171;0;;40.630;0;48.313;0;50.228;0;141.5;1:02:07.499;12:47:35.829;0:40.630;0:48.313;0:50.228;;Gregory Liefooghe;0:01:28.696;GS;;Stephen Cameron Racing;Ford;FCY;
19;2;30;2:42.799;0;;36.642;0;55.028;0;1:11.129;0;121.0;1:04:50.298;12:50:18.628;0:36.642;0:55.028;1:11.129;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;FCY;
19;2;31;2:57.848;0;;53.190;0;1:12.850;0;51.808;0;110.8;1:07:48.146;12:53:16.476;0:53.190;1:12.850;0:51.808;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;32;1:58.080;0;;26.977;2;45.067;0;46.036;0;166.8;1:09:46.226;12:55:14.556;0:26.977;0:45.067;0:46.036;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;33;1:57.710;0;;27.270;0;44.467;0;45.973;0;167.3;1:11:43.936;12:57:12.266;0:27.270;0:44.467;0:45.973;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;34;1:57.256;0;;27.112;0;44.281;0;45.863;0;168.0;1:13:41.192;12:59:09.522;0:27.112;0:44.281;0:45.863;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;35;1:57.419;0;;27.234;0;44.294;0;45.891;0;167.8;1:15:38.611;13:01:06.941;0:27.234;0:44.294;0:45.891;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;36;1:57.528;0;;27.198;0;44.343;0;45.987;0;167.6;1:17:36.139;13:03:04.469;0:27.198;0:44.343;0:45.987;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;37;1:57.868;0;;27.479;0;44.454;0;45.935;0;167.1;1:19:34.007;13:05:02.337;0:27.479;0:44.454;0:45.935;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;38;1:57.986;0;;27.243;0;44.939;0;45.804;0;167.0;1:21:31.993;13:07:00.323;0:27.243;0:44.939;0:45.804;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;39;1:57.398;0;;27.260;0;44.162;0;45.976;0;167.8;1:23:29.391;13:08:57.721;0:27.260;0:44.162;0:45.976;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;40;1:57.754;0;;27.242;0;44.530;0;45.982;0;167.3;1:25:27.145;13:10:55.475;0:27.242;0:44.530;0:45.982;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;41;1:57.518;0;;27.176;0;44.409;0;45.933;0;167.6;1:27:24.663;13:12:52.993;0:27.176;0:44.409;0:45.933;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;42;1:58.034;0;;27.196;0;44.574;0;46.264;0;166.9;1:29:22.697;13:14:51.027;0:27.196;0:44.574;0:46.264;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;43;1:58.956;0;;27.627;0;44.978;0;46.351;0;165.6;1:31:21.653;13:16:49.983;0:27.627;0:44.978;0:46.351;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;44;1:57.336;0;;27.206;0;44.246;0;45.884;0;167.9;1:33:18.989;13:18:47.319;0:27.206;0:44.246;0:45.884;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;45;1:57.104;2;;27.282;0;44.121;2;45.701;2;168.2;1:35:16.093;13:20:44.423;0:27.282;0:44.121;0:45.701;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;46;1:57.661;0;;27.186;0;44.291;0;46.184;0;167.4;1:37:13.754;13:22:42.084;0:27.186;0:44.291;0:46.184;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;47;1:58.004;0;;27.405;0;44.275;0;46.324;0;166.9;1:39:11.758;13:24:40.088;0:27.405;0:44.275;0:46.324;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;48;1:57.350;0;;27.303;0;44.268;0;45.779;0;167.9;1:41:09.108;13:26:37.438;0:27.303;0:44.268;0:45.779;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;49;1:57.331;0;;27.096;0;44.350;0;45.885;0;167.9;1:43:06.439;13:28:34.769;0:27.096;0:44.350;0:45.885;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;50;1:57.541;0;;27.228;0;44.275;0;46.038;0;167.6;1:45:03.980;13:30:32.310;0:27.228;0:44.275;0:46.038;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;51;2:10.990;0;;27.158;0;48.249;0;55.583;0;150.4;1:47:14.970;13:32:43.300;0:27.158;0:48.249;0:55.583;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;FCY;
19;2;52;3:22.240;0;;52.364;0;1:30.500;0;59.376;0;97.4;1:50:37.210;13:36:05.540;0:52.364;1:30.500;0:59.376;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
19;2;53;2:00.178;0;;27.267;0;46.909;0;46.002;0;163.9;1:52:37.388;13:38:05.718;0:27.267;0:46.909;0:46.002;;Gregory Liefooghe;;GS;;Stephen Cameron Racing;Ford;GF;
2;2;1;2:07.395;0;;33.674;0;46.812;0;46.909;0;154.6;2:07.395;11:47:35.725;0:33.674;0:46.812;0:46.909;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;2;2:00.368;0;;27.522;0;45.958;0;46.888;0;163.7;4:07.763;11:49:36.093;0:27.522;0:45.958;0:46.888;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;3;2:29.322;0;;29.666;0;59.134;0;1:00.522;0;131.9;6:37.085;11:52:05.415;0:29.666;0:59.134;1:00.522;;Phil Fayer;;GS;;CSM;Porsche;FCY;
2;2;4;3:01.629;0;;43.461;0;1:15.151;0;1:03.017;0;108.5;9:38.714;11:55:07.044;0:43.461;1:15.151;1:03.017;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;5;2:00.339;0;;27.692;0;45.940;0;46.707;0;163.7;11:39.053;11:57:07.383;0:27.692;0:45.940;0:46.707;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;6;1:59.971;0;;27.776;0;45.787;0;46.408;0;164.2;13:39.024;11:59:07.354;0:27.776;0:45.787;0:46.408;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;7;1:59.507;0;;27.590;0;45.605;0;46.312;1;164.8;15:38.531;12:01:06.861;0:27.590;0:45.605;0:46.312;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;8;1:59.632;0;;27.954;0;45.279;1;46.399;0;164.7;17:38.163;12:03:06.493;0:27.954;0:45.279;0:46.399;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;9;2:00.687;0;;28.148;0;45.959;0;46.580;0;163.2;19:38.850;12:05:07.180;0:28.148;0:45.959;0:46.580;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;10;1:59.863;0;;27.596;0;45.610;0;46.657;0;164.3;21:38.713;12:07:07.043;0:27.596;0:45.610;0:46.657;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;11;2:00.973;0;;27.575;0;45.985;0;47.413;0;162.8;23:39.686;12:09:08.016;0:27.575;0:45.985;0:47.413;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;12;1:59.384;0;;27.468;0;45.429;0;46.487;0;165.0;25:39.070;12:11:07.400;0:27.468;0:45.429;0:46.487;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;13;2:00.129;0;;27.489;0;45.651;0;46.989;0;164.0;27:39.199;12:13:07.529;0:27.489;0:45.651;0:46.989;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;14;2:01.235;0;;28.264;0;46.040;0;46.931;0;162.5;29:40.434;12:15:08.764;0:28.264;0:46.040;0:46.931;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;15;2:01.805;0;;28.147;0;46.006;0;47.652;0;161.7;31:42.239;12:17:10.569;0:28.147;0:46.006;0:47.652;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;16;2:01.129;0;;27.455;0;46.232;0;47.442;0;162.6;33:43.368;12:19:11.698;0:27.455;0:46.232;0:47.442;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;17;2:01.167;0;;28.554;0;45.964;0;46.649;0;162.6;35:44.535;12:21:12.865;0:28.554;0:45.964;0:46.649;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;18;2:00.277;0;;27.314;1;46.195;0;46.768;0;163.8;37:44.812;12:23:13.142;0:27.314;0:46.195;0:46.768;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;19;2:01.418;0;;27.469;0;45.542;0;48.407;0;162.2;39:46.230;12:25:14.560;0:27.469;0:45.542;0:48.407;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;20;2:00.488;0;;27.682;0;45.771;0;47.035;0;163.5;41:46.718;12:27:15.048;0:27.682;0:45.771;0:47.035;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;2;21;3:03.105;0;B;27.756;0;45.825;0;1:49.524;0;107.6;44:49.823;12:30:18.153;0:27.756;0:45.825;1:49.524;;Phil Fayer;;GS;;CSM;Porsche;GF;
2;1;22;2:12.627;0;;40.008;0;46.177;0;46.442;0;148.5;47:02.450;12:32:30.780;0:40.008;0:46.177;0:46.442;;Robert Megennis;0:01:18.544;GS;;CSM;Porsche;GF;
2;1;23;1:57.226;0;;27.275;0;44.232;0;45.719;2;168.0;48:59.676;12:34:28.006;0:27.275;0:44.232;0:45.719;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;24;1:56.896;2;;27.123;0;44.029;2;45.744;0;168.5;50:56.572;12:36:24.902;0:27.123;0:44.029;0:45.744;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;25;1:57.088;0;;27.077;0;44.141;0;45.870;0;168.2;52:53.660;12:38:21.990;0:27.077;0:44.141;0:45.870;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;26;1:57.172;0;;27.061;0;44.152;0;45.959;0;168.1;54:50.832;12:40:19.162;0:27.061;0:44.152;0:45.959;;Robert Megennis;;GS;;CSM;Porsche;FCY;
2;1;27;2:11.742;0;;30.090;0;51.787;0;49.865;0;149.5;57:02.574;12:42:30.904;0:30.090;0:51.787;0:49.865;;Robert Megennis;;GS;;CSM;Porsche;FCY;
2;1;28;2:31.777;0;B;27.577;0;44.768;0;1:19.432;0;129.8;59:34.351;12:45:02.681;0:27.577;0:44.768;1:19.432;;Robert Megennis;;GS;;CSM;Porsche;FCY;
2;1;29;2:29.134;0;;39.392;0;45.954;0;1:03.788;0;132.1;1:02:03.485;12:47:31.815;0:39.392;0:45.954;1:03.788;;Robert Megennis;0:00:51.138;GS;;CSM;Porsche;FCY;
2;1;30;2:43.831;0;;34.807;0;55.888;0;1:13.136;0;120.2;1:04:47.316;12:50:15.646;0:34.807;0:55.888;1:13.136;;Robert Megennis;;GS;;CSM;Porsche;FCY;
2;1;31;2:59.626;0;;53.468;0;1:12.765;0;53.393;0;109.7;1:07:46.942;12:53:15.272;0:53.468;1:12.765;0:53.393;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;32;1:58.401;0;;27.092;0;45.398;0;45.911;0;166.4;1:09:45.343;12:55:13.673;0:27.092;0:45.398;0:45.911;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;33;1:57.669;0;;27.213;0;44.532;0;45.924;0;167.4;1:11:43.012;12:57:11.342;0:27.213;0:44.532;0:45.924;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;34;1:57.355;0;;27.071;0;44.199;0;46.085;0;167.9;1:13:40.367;12:59:08.697;0:27.071;0:44.199;0:46.085;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;35;1:57.365;0;;27.029;2;44.351;0;45.985;0;167.8;1:15:37.732;13:01:06.062;0:27.029;0:44.351;0:45.985;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;36;1:57.533;0;;27.164;0;44.297;0;46.072;0;167.6;1:17:35.265;13:03:03.595;0:27.164;0:44.297;0:46.072;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;37;1:57.587;0;;27.203;0;44.325;0;46.059;0;167.5;1:19:32.852;13:05:01.182;0:27.203;0:44.325;0:46.059;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;38;1:57.893;0;;27.369;0;44.426;0;46.098;0;167.1;1:21:30.745;13:06:59.075;0:27.369;0:44.426;0:46.098;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;39;1:57.797;0;;27.288;0;44.340;0;46.169;0;167.2;1:23:28.542;13:08:56.872;0:27.288;0:44.340;0:46.169;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;40;1:57.664;0;;27.220;0;44.362;0;46.082;0;167.4;1:25:26.206;13:10:54.536;0:27.220;0:44.362;0:46.082;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;41;1:57.650;0;;27.167;0;44.322;0;46.161;0;167.4;1:27:23.856;13:12:52.186;0:27.167;0:44.322;0:46.161;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;42;1:58.411;0;;27.316;0;44.499;0;46.596;0;166.4;1:29:22.267;13:14:50.597;0:27.316;0:44.499;0:46.596;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;43;1:59.920;0;;28.016;0;44.790;0;47.114;0;164.3;1:31:22.187;13:16:50.517;0:28.016;0:44.790;0:47.114;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;44;1:59.225;0;;27.611;0;44.713;0;46.901;0;165.2;1:33:21.412;13:18:49.742;0:27.611;0:44.713;0:46.901;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;45;1:58.451;0;;27.301;0;44.680;0;46.470;0;166.3;1:35:19.863;13:20:48.193;0:27.301;0:44.680;0:46.470;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;46;1:58.430;0;;27.320;0;44.761;0;46.349;0;166.3;1:37:18.293;13:22:46.623;0:27.320;0:44.761;0:46.349;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;47;1:57.845;0;;27.237;0;44.373;0;46.235;0;167.2;1:39:16.138;13:24:44.468;0:27.237;0:44.373;0:46.235;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;48;1:58.352;0;;27.316;0;44.511;0;46.525;0;166.4;1:41:14.490;13:26:42.820;0:27.316;0:44.511;0:46.525;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;49;1:58.906;0;;27.411;0;44.859;0;46.636;0;165.7;1:43:13.396;13:28:41.726;0:27.411;0:44.859;0:46.636;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;50;1:58.610;0;;27.330;0;44.786;0;46.494;0;166.1;1:45:12.006;13:30:40.336;0:27.330;0:44.786;0:46.494;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;51;2:05.110;0;;27.322;0;46.753;0;51.035;0;157.4;1:47:17.116;13:32:45.446;0:27.322;0:46.753;0:51.035;;Robert Megennis;;GS;;CSM;Porsche;FCY;
2;1;52;3:20.792;0;;51.896;0;1:30.298;0;58.598;0;98.1;1:50:37.908;13:36:06.238;0:51.896;1:30.298;0:58.598;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;53;2:01.707;0;;27.552;0;46.770;0;47.385;0;161.9;1:52:39.615;13:38:07.945;0:27.552;0:46.770;0:47.385;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;54;1:58.180;0;;27.278;0;44.556;0;46.346;0;166.7;1:54:37.795;13:40:06.125;0:27.278;0:44.556;0:46.346;;Robert Megennis;;GS;;CSM;Porsche;GF;
2;1;55;2:03.024;0;;27.559;0;45.111;0;50.354;0;160.1;1:56:40.819;13:42:09.149;0:27.559;0:45.111;0:50.354;;Robert Megennis;;GS;;CSM;Porsche;FCY;
2;1;56;3:25.011;0;;48.705;0;1:19.356;0;1:16.950;0;96.1;2:00:05.830;13:45:34.160;0:48.705;1:19.356;1:16.950;;Robert Megennis;;GS;;CSM;Porsche;FCY;
2;1;57;3:25.828;0;;53.440;0;1:15.663;0;1:16.725;0;95.7;2:03:31.658;13:48:59.988;0:53.440;1:15.663;1:16.725;;Robert Megennis;;GS;;CSM;Porsche;FF;
27;1;1;2:02.777;0;;30.349;0;45.616;0;46.812;0;160.4;2:02.777;11:47:31.107;0:30.349;0:45.616;0:46.812;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;2;1:57.687;0;;27.284;0;44.325;0;46.078;0;167.4;4:00.464;11:49:28.794;0:27.284;0:44.325;0:46.078;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;3;2:29.074;0;;30.463;0;1:00.377;0;58.234;0;132.1;6:29.538;11:51:57.868;0:30.463;1:00.377;0:58.234;;Austin Krainz;;GS;;Auto Technic Racing;BMW;FCY;
27;1;4;3:06.205;0;;44.077;0;1:17.535;0;1:04.593;0;105.8;9:35.743;11:55:04.073;0:44.077;1:17.535;1:04.593;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;5;1:58.948;0;;27.363;0;45.083;0;46.502;0;165.6;11:34.691;11:57:03.021;0:27.363;0:45.083;0:46.502;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;6;1:57.602;0;;27.206;0;44.464;0;45.932;0;167.5;13:32.293;11:59:00.623;0:27.206;0:44.464;0:45.932;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;7;1:57.907;0;;27.189;0;44.522;0;46.196;0;167.1;15:30.200;12:00:58.530;0:27.189;0:44.522;0:46.196;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;8;1:57.498;0;;27.258;0;44.246;0;45.994;0;167.6;17:27.698;12:02:56.028;0:27.258;0:44.246;0:45.994;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;9;1:57.504;0;;27.192;0;44.407;0;45.905;1;167.6;19:25.202;12:04:53.532;0:27.192;0:44.407;0:45.905;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;10;1:57.790;0;;27.298;0;44.094;1;46.398;0;167.2;21:22.992;12:06:51.322;0:27.298;0:44.094;0:46.398;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;11;1:57.890;0;;27.337;0;44.427;0;46.126;0;167.1;23:20.882;12:08:49.212;0:27.337;0:44.427;0:46.126;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;12;1:58.022;0;;27.422;0;44.502;0;46.098;0;166.9;25:18.904;12:10:47.234;0:27.422;0:44.502;0:46.098;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;13;1:57.805;0;;27.301;0;44.493;0;46.011;0;167.2;27:16.709;12:12:45.039;0:27.301;0:44.493;0:46.011;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;14;1:57.648;0;;27.215;0;44.384;0;46.049;0;167.4;29:14.357;12:14:42.687;0:27.215;0:44.384;0:46.049;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;15;1:57.865;0;;27.345;0;44.441;0;46.079;0;167.1;31:12.222;12:16:40.552;0:27.345;0:44.441;0:46.079;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;16;1:58.407;0;;27.138;1;45.068;0;46.201;0;166.4;33:10.629;12:18:38.959;0:27.138;0:45.068;0:46.201;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;17;1:57.836;0;;27.258;0;44.327;0;46.251;0;167.2;35:08.465;12:20:36.795;0:27.258;0:44.327;0:46.251;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;18;1:57.948;0;;27.344;0;44.521;0;46.083;0;167.0;37:06.413;12:22:34.743;0:27.344;0:44.521;0:46.083;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;19;1:57.846;0;;27.331;0;44.465;0;46.050;0;167.2;39:04.259;12:24:32.589;0:27.331;0:44.465;0:46.050;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;1;20;2:11.131;0;B;27.321;0;44.560;0;59.250;0;150.2;41:15.390;12:26:43.720;0:27.321;0:44.560;0:59.250;;Austin Krainz;;GS;;Auto Technic Racing;BMW;GF;
27;2;21;2:57.745;0;;1:26.348;0;45.238;0;46.159;0;110.8;44:13.135;12:29:41.465;1:26.348;0:45.238;0:46.159;;Stevan McAleer;0:01:17.164;GS;;Auto Technic Racing;BMW;GF;
27;2;22;1:56.985;0;;27.248;0;44.138;0;45.599;0;168.4;46:10.120;12:31:38.450;0:27.248;0:44.138;0:45.599;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;23;1:56.421;2;;27.071;0;43.866;0;45.484;2;169.2;48:06.541;12:33:34.871;0:27.071;0:43.866;0:45.484;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;24;1:56.911;0;;27.422;0;43.875;0;45.614;0;168.5;50:03.452;12:35:31.782;0:27.422;0:43.875;0:45.614;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;25;1:56.474;0;;27.139;0;43.773;2;45.562;0;169.1;51:59.926;12:37:28.256;0:27.139;0:43.773;0:45.562;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;26;1:56.977;0;;27.149;0;44.178;0;45.650;0;168.4;53:56.903;12:39:25.233;0:27.149;0:44.178;0:45.650;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;27;2:13.079;0;;27.706;0;47.866;0;57.507;0;148.0;56:09.982;12:41:38.312;0:27.706;0:47.866;0:57.507;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;FCY;
27;2;28;2:44.595;0;B;36.789;0;53.261;0;1:14.545;0;119.7;58:54.577;12:44:22.907;0:36.789;0:53.261;1:14.545;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;FCY;
27;2;29;3:04.651;0;;1:09.960;0;48.759;0;1:05.932;0;106.7;1:01:59.228;12:47:27.558;1:09.960;0:48.759;1:05.932;;Stevan McAleer;0:00:57.337;GS;;Auto Technic Racing;BMW;FCY;
27;2;30;2:47.035;0;;34.340;0;57.891;0;1:14.804;0;117.9;1:04:46.263;12:50:14.593;0:34.340;0:57.891;1:14.804;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;FCY;
27;2;31;2:59.818;0;;52.715;0;1:13.335;0;53.768;0;109.5;1:07:46.081;12:53:14.411;0:52.715;1:13.335;0:53.768;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;32;1:58.465;0;;27.569;0;45.444;0;45.452;0;166.3;1:09:44.546;12:55:12.876;0:27.569;0:45.444;0:45.452;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;33;1:57.248;0;;27.052;0;44.144;0;46.052;0;168.0;1:11:41.794;12:57:10.124;0:27.052;0:44.144;0:46.052;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;34;1:57.610;0;;27.107;0;44.584;0;45.919;0;167.5;1:13:39.404;12:59:07.734;0:27.107;0:44.584;0:45.919;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;35;1:57.537;0;;27.082;0;44.367;0;46.088;0;167.6;1:15:36.941;13:01:05.271;0:27.082;0:44.367;0:46.088;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;36;1:57.255;0;;27.123;0;44.200;0;45.932;0;168.0;1:17:34.196;13:03:02.526;0:27.123;0:44.200;0:45.932;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;37;1:57.235;0;;27.293;0;44.133;0;45.809;0;168.0;1:19:31.431;13:04:59.761;0:27.293;0:44.133;0:45.809;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;38;1:57.114;0;;27.104;0;44.151;0;45.859;0;168.2;1:21:28.545;13:06:56.875;0:27.104;0:44.151;0:45.859;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;39;1:57.376;0;;27.225;0;44.136;0;46.015;0;167.8;1:23:25.921;13:08:54.251;0:27.225;0:44.136;0:46.015;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;40;1:57.387;0;;27.222;0;44.341;0;45.824;0;167.8;1:25:23.308;13:10:51.638;0:27.222;0:44.341;0:45.824;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;41;1:57.259;0;;27.210;0;44.344;0;45.705;0;168.0;1:27:20.567;13:12:48.897;0:27.210;0:44.344;0:45.705;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;42;1:57.527;0;;27.185;0;44.456;0;45.886;0;167.6;1:29:18.094;13:14:46.424;0:27.185;0:44.456;0:45.886;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;43;1:57.771;0;;27.269;0;44.422;0;46.080;0;167.3;1:31:15.865;13:16:44.195;0:27.269;0:44.422;0:46.080;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;44;1:58.045;0;;27.282;0;44.580;0;46.183;0;166.9;1:33:13.910;13:18:42.240;0:27.282;0:44.580;0:46.183;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;45;1:57.655;0;;27.161;0;44.500;0;45.994;0;167.4;1:35:11.565;13:20:39.895;0:27.161;0:44.500;0:45.994;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;46;1:57.598;0;;27.229;0;44.406;0;45.963;0;167.5;1:37:09.163;13:22:37.493;0:27.229;0:44.406;0:45.963;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;47;1:57.297;0;;27.133;0;44.352;0;45.812;0;167.9;1:39:06.460;13:24:34.790;0:27.133;0:44.352;0:45.812;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;48;1:57.611;0;;27.111;0;44.479;0;46.021;0;167.5;1:41:04.071;13:26:32.401;0:27.111;0:44.479;0:46.021;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;49;1:57.655;0;;27.078;0;44.479;0;46.098;0;167.4;1:43:01.726;13:28:30.056;0:27.078;0:44.479;0:46.098;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;50;1:57.024;0;;27.183;0;44.159;0;45.682;0;168.3;1:44:58.750;13:30:27.080;0:27.183;0:44.159;0:45.682;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;51;2:11.078;0;;27.128;0;44.740;0;59.210;0;150.3;1:47:09.828;13:32:38.158;0:27.128;0:44.740;0:59.210;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;FCY;
27;2;52;3:26.296;0;;53.584;0;1:30.759;0;1:01.953;0;95.5;1:50:36.124;13:36:04.454;0:53.584;1:30.759;1:01.953;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;53;1:59.058;0;;27.541;0;45.712;0;45.805;0;165.5;1:52:35.182;13:38:03.512;0:27.541;0:45.712;0:45.805;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;54;1:58.175;0;;27.116;0;45.223;0;45.836;0;166.7;1:54:33.357;13:40:01.687;0:27.116;0:45.223;0:45.836;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;GF;
27;2;55;2:02.593;0;;27.005;2;44.275;0;51.313;0;160.7;1:56:35.950;13:42:04.280;0:27.005;0:44.275;0:51.313;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;FCY;
27;2;56;3:22.536;0;;47.651;0;1:18.928;0;1:15.957;0;97.3;1:59:58.486;13:45:26.816;0:47.651;1:18.928;1:15.957;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;FCY;
27;2;57;3:26.018;0;;52.796;0;1:16.876;0;1:16.346;0;95.6;2:03:24.504;13:48:52.834;0:52.796;1:16.876;1:16.346;;Stevan McAleer;;GS;;Auto Technic Racing;BMW;FF;
28;1;1;1:59.871;0;;28.926;0;44.857;0;46.088;0;164.3;1:59.871;11:47:28.201;0:28.926;0:44.857;0:46.088;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;2;1:57.781;0;;27.052;0;44.589;0;46.140;0;167.2;3:57.652;11:49:25.982;0:27.052;0:44.589;0:46.140;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;3;2:26.198;0;;29.086;0;59.288;0;57.824;0;134.7;6:23.850;11:51:52.180;0:29.086;0:59.288;0:57.824;;Luca Mars;;GS;;RS1;Porsche;FCY;
28;1;4;3:10.093;0;;45.597;0;1:17.305;0;1:07.191;0;103.6;9:33.943;11:55:02.273;0:45.597;1:17.305;1:07.191;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;5;1:57.914;0;;27.140;0;44.930;0;45.844;0;167.1;11:31.857;11:57:00.187;0:27.140;0:44.930;0:45.844;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;6;1:56.628;0;;27.050;0;44.058;0;45.520;0;168.9;13:28.485;11:58:56.815;0:27.050;0:44.058;0:45.520;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;7;1:56.910;0;;26.941;1;44.217;0;45.752;0;168.5;15:25.395;12:00:53.725;0:26.941;0:44.217;0:45.752;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;8;1:56.665;0;;27.103;0;44.078;0;45.484;2;168.8;17:22.060;12:02:50.390;0:27.103;0:44.078;0:45.484;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;9;1:56.771;0;;27.097;0;44.024;0;45.650;0;168.7;19:18.831;12:04:47.161;0:27.097;0:44.024;0:45.650;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;10;1:56.693;0;;26.982;0;44.057;0;45.654;0;168.8;21:15.524;12:06:43.854;0:26.982;0:44.057;0:45.654;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;11;1:56.889;0;;27.230;0;43.995;1;45.664;0;168.5;23:12.413;12:08:40.743;0:27.230;0:43.995;0:45.664;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;12;1:56.683;0;;27.067;0;44.052;0;45.564;0;168.8;25:09.096;12:10:37.426;0:27.067;0:44.052;0:45.564;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;13;1:56.776;0;;26.966;0;44.119;0;45.691;0;168.7;27:05.872;12:12:34.202;0:26.966;0:44.119;0:45.691;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;14;1:57.333;0;;27.049;0;44.528;0;45.756;0;167.9;29:03.205;12:14:31.535;0:27.049;0:44.528;0:45.756;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;15;1:56.960;0;;27.177;0;44.168;0;45.615;0;168.4;31:00.165;12:16:28.495;0:27.177;0:44.168;0:45.615;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;16;1:56.896;0;;27.016;0;44.150;0;45.730;0;168.5;32:57.061;12:18:25.391;0:27.016;0:44.150;0:45.730;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;17;1:56.973;0;;27.028;0;44.269;0;45.676;0;168.4;34:54.034;12:20:22.364;0:27.028;0:44.269;0:45.676;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;18;1:56.751;0;;26.953;0;44.184;0;45.614;0;168.7;36:50.785;12:22:19.115;0:26.953;0:44.184;0:45.614;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;19;1:56.847;0;;27.042;0;44.089;0;45.716;0;168.6;38:47.632;12:24:15.962;0:27.042;0:44.089;0:45.716;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;20;1:57.065;0;;27.086;0;44.227;0;45.752;0;168.3;40:44.697;12:26:13.027;0:27.086;0:44.227;0:45.752;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;21;1:57.542;0;;26.954;0;44.331;0;46.257;0;167.6;42:42.239;12:28:10.569;0:26.954;0:44.331;0:46.257;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;22;1:57.053;0;;26.941;0;44.237;0;45.875;0;168.3;44:39.292;12:30:07.622;0:26.941;0:44.237;0:45.875;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;23;1:57.556;0;;27.033;0;44.353;0;46.170;0;167.6;46:36.848;12:32:05.178;0:27.033;0:44.353;0:46.170;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;24;1:57.179;0;;27.024;0;44.192;0;45.963;0;168.1;48:34.027;12:34:02.357;0:27.024;0:44.192;0:45.963;;Luca Mars;;GS;;RS1;Porsche;GF;
28;1;25;2:09.737;0;B;26.987;0;44.535;0;58.215;0;151.8;50:43.764;12:36:12.094;0:26.987;0:44.535;0:58.215;;Luca Mars;;GS;;RS1;Porsche;GF;
28;2;26;3:01.664;0;;1:30.145;0;45.220;0;46.299;0;108.4;53:45.428;12:39:13.758;1:30.145;0:45.220;0:46.299;;Jan Heylen;0:01:21.095;GS;;RS1;Porsche;GF;
28;2;27;2:17.819;0;;27.374;0;50.467;0;59.978;0;142.9;56:03.247;12:41:31.577;0:27.374;0:50.467;0:59.978;;Jan Heylen;;GS;;RS1;Porsche;FCY;
28;2;28;2:47.793;0;B;37.334;0;49.535;0;1:20.924;0;117.4;58:51.040;12:44:19.370;0:37.334;0:49.535;1:20.924;;Jan Heylen;;GS;;RS1;Porsche;FCY;
28;2;29;2:57.852;0;;49.863;0;1:04.359;0;1:03.630;0;110.8;1:01:48.892;12:47:17.222;0:49.863;1:04.359;1:03.630;;Jan Heylen;0:00:39.416;GS;;RS1;Porsche;FCY;
28;2;30;2:51.217;0;;32.489;0;1:03.253;0;1:15.475;0;115.0;1:04:40.109;12:50:08.439;0:32.489;1:03.253;1:15.475;;Jan Heylen;;GS;;RS1;Porsche;FCY;
28;2;31;3:03.005;0;;51.727;0;1:13.577;0;57.701;0;107.6;1:07:43.114;12:53:11.444;0:51.727;1:13.577;0:57.701;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;32;1:57.851;0;;27.300;0;44.521;0;46.030;0;167.1;1:09:40.965;12:55:09.295;0:27.300;0:44.521;0:46.030;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;33;1:57.961;0;;27.013;0;45.010;0;45.938;0;167.0;1:11:38.926;12:57:07.256;0:27.013;0:45.010;0:45.938;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;34;1:56.455;2;;27.013;0;43.946;0;45.496;1;169.2;1:13:35.381;12:59:03.711;0:27.013;0:43.946;0:45.496;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;35;1:57.133;0;;26.894;2;43.954;0;46.285;0;168.2;1:15:32.514;13:01:00.844;0:26.894;0:43.954;0:46.285;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;36;1:56.517;0;;27.065;0;43.930;2;45.522;0;169.1;1:17:29.031;13:02:57.361;0:27.065;0:43.930;0:45.522;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;37;1:56.668;0;;26.933;0;44.055;0;45.680;0;168.8;1:19:25.699;13:04:54.029;0:26.933;0:44.055;0:45.680;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;38;1:56.924;0;;26.981;0;44.174;0;45.769;0;168.5;1:21:22.623;13:06:50.953;0:26.981;0:44.174;0:45.769;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;39;1:56.747;0;;26.947;0;44.009;0;45.791;0;168.7;1:23:19.370;13:08:47.700;0:26.947;0:44.009;0:45.791;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;40;1:56.728;0;;26.990;0;44.154;0;45.584;0;168.8;1:25:16.098;13:10:44.428;0:26.990;0:44.154;0:45.584;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;41;1:56.587;0;;26.911;0;44.007;0;45.669;0;169.0;1:27:12.685;13:12:41.015;0:26.911;0:44.007;0:45.669;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;42;1:57.573;0;;27.067;0;44.435;0;46.071;0;167.5;1:29:10.258;13:14:38.588;0:27.067;0:44.435;0:46.071;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;43;1:56.965;0;;26.939;0;44.278;0;45.748;0;168.4;1:31:07.223;13:16:35.553;0:26.939;0:44.278;0:45.748;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;44;1:56.942;0;;26.934;0;44.243;0;45.765;0;168.4;1:33:04.165;13:18:32.495;0:26.934;0:44.243;0:45.765;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;45;1:57.792;0;;27.022;0;44.430;0;46.340;0;167.2;1:35:01.957;13:20:30.287;0:27.022;0:44.430;0:46.340;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;46;1:57.340;0;;27.002;0;44.304;0;46.034;0;167.9;1:36:59.297;13:22:27.627;0:27.002;0:44.304;0:46.034;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;47;1:57.308;0;;27.076;0;44.343;0;45.889;0;167.9;1:38:56.605;13:24:24.935;0:27.076;0:44.343;0:45.889;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;48;1:56.944;0;;26.916;0;44.177;0;45.851;0;168.4;1:40:53.549;13:26:21.879;0:26.916;0:44.177;0:45.851;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;49;1:57.035;0;;27.061;0;44.278;0;45.696;0;168.3;1:42:50.584;13:28:18.914;0:27.061;0:44.278;0:45.696;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;50;1:57.534;0;;26.949;0;44.487;0;46.098;0;167.6;1:44:48.118;13:30:16.448;0:26.949;0:44.487;0:46.098;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;51;2:15.168;0;;26.972;0;47.500;0;1:00.696;0;145.7;1:47:03.286;13:32:31.616;0:26.972;0:47.500;1:00.696;;Jan Heylen;;GS;;RS1;Porsche;FCY;
28;2;52;3:30.991;0;;54.150;0;1:30.354;0;1:06.487;0;93.4;1:50:34.277;13:36:02.607;0:54.150;1:30.354;1:06.487;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;53;1:58.107;0;;27.324;0;44.627;0;46.156;0;166.8;1:52:32.384;13:38:00.714;0:27.324;0:44.627;0:46.156;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;54;1:57.059;0;;27.031;0;44.311;0;45.717;0;168.3;1:54:29.443;13:39:57.773;0:27.031;0:44.311;0:45.717;;Jan Heylen;;GS;;RS1;Porsche;GF;
28;2;55;2:02.192;0;;26.944;0;44.237;0;51.011;0;161.2;1:56:31.635;13:41:59.965;0:26.944;0:44.237;0:51.011;;Jan Heylen;;GS;;RS1;Porsche;FCY;
28;2;56;3:22.553;0;;48.096;0;1:17.937;0;1:16.520;0;97.3;1:59:54.188;13:45:22.518;0:48.096;1:17.937;1:16.520;;Jan Heylen;;GS;;RS1;Porsche;FCY;
28;2;57;3:26.614;0;;52.381;0;1:17.135;0;1:17.098;0;95.3;2:03:20.802;13:48:49.132;0:52.381;1:17.135;1:17.098;;Jan Heylen;;GS;;RS1;Porsche;FF;
31;1;1;2:12.828;0;;39.195;0;45.846;0;47.787;0;148.3;2:12.828;11:47:41.158;0:39.195;0:45.846;0:47.787;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;2;1:58.916;0;;27.629;0;45.386;0;45.901;0;165.6;4:11.744;11:49:40.074;0:27.629;0:45.386;0:45.901;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;3;2:31.039;0;;31.590;0;1:00.814;0;58.635;0;130.4;6:42.783;11:52:11.113;0:31.590;1:00.814;0:58.635;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;FCY;
31;1;4;2:59.767;0;;42.819;0;1:16.574;0;1:00.374;0;109.6;9:42.550;11:55:10.880;0:42.819;1:16.574;1:00.374;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;5;2:00.576;0;;27.974;0;45.307;0;47.295;0;163.4;11:43.126;11:57:11.456;0:27.974;0:45.307;0:47.295;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;6;1:57.400;0;;27.465;0;44.341;1;45.594;1;167.8;13:40.526;11:59:08.856;0:27.465;0:44.341;0:45.594;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;7;1:58.090;0;;27.165;1;45.135;0;45.790;0;166.8;15:38.616;12:01:06.946;0:27.165;0:45.135;0:45.790;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;8;1:58.017;0;;27.479;0;44.397;0;46.141;0;166.9;17:36.633;12:03:04.963;0:27.479;0:44.397;0:46.141;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;9;1:58.359;0;;27.328;0;44.985;0;46.046;0;166.4;19:34.992;12:05:03.322;0:27.328;0:44.985;0:46.046;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;10;1:59.831;0;;27.518;0;45.276;0;47.037;0;164.4;21:34.823;12:07:03.153;0:27.518;0:45.276;0:47.037;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;11;1:58.912;0;;27.923;0;45.030;0;45.959;0;165.7;23:33.735;12:09:02.065;0:27.923;0:45.030;0:45.959;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;12;1:58.698;0;;28.003;0;44.721;0;45.974;0;166.0;25:32.433;12:11:00.763;0:28.003;0:44.721;0:45.974;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;13;1:58.415;0;;27.600;0;44.643;0;46.172;0;166.4;27:30.848;12:12:59.178;0:27.600;0:44.643;0:46.172;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;14;1:58.647;0;;27.407;0;44.771;0;46.469;0;166.0;29:29.495;12:14:57.825;0:27.407;0:44.771;0:46.469;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;15;1:58.525;0;;27.307;0;44.773;0;46.445;0;166.2;31:28.020;12:16:56.350;0:27.307;0:44.773;0:46.445;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;16;1:58.694;0;;27.445;0;44.749;0;46.500;0;166.0;33:26.714;12:18:55.044;0:27.445;0:44.749;0:46.500;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;17;1:59.252;0;;27.479;0;45.323;0;46.450;0;165.2;35:25.966;12:20:54.296;0:27.479;0:45.323;0:46.450;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;18;1:59.729;0;;27.588;0;45.394;0;46.747;0;164.5;37:25.695;12:22:54.025;0:27.588;0:45.394;0:46.747;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;19;1:59.047;0;;27.469;0;44.912;0;46.666;0;165.5;39:24.742;12:24:53.072;0:27.469;0:44.912;0:46.666;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;20;2:00.472;0;;27.638;0;45.780;0;47.054;0;163.5;41:25.214;12:26:53.544;0:27.638;0:45.780;0:47.054;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;21;1:59.171;0;;27.622;0;45.093;0;46.456;0;165.3;43:24.385;12:28:52.715;0:27.622;0:45.093;0:46.456;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;22;1:58.872;0;;27.401;0;44.920;0;46.551;0;165.7;45:23.257;12:30:51.587;0:27.401;0:44.920;0:46.551;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;23;1:58.548;0;;27.363;0;44.828;0;46.357;0;166.2;47:21.805;12:32:50.135;0:27.363;0:44.828;0:46.357;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;24;1:58.574;0;;27.392;0;44.850;0;46.332;0;166.1;49:20.379;12:34:48.709;0:27.392;0:44.850;0:46.332;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;25;1:58.331;0;;27.389;0;44.692;0;46.250;0;166.5;51:18.710;12:36:47.040;0:27.389;0:44.692;0:46.250;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;26;1:59.380;0;;27.380;0;45.502;0;46.498;0;165.0;53:18.090;12:38:46.420;0:27.380;0:45.502;0:46.498;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;1;27;2:00.283;0;;27.362;0;44.769;0;48.152;0;163.8;55:18.373;12:40:46.703;0:27.362;0:44.769;0:48.152;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;FCY;
31;1;28;3:15.695;0;;33.329;0;1:17.343;0;1:25.023;0;100.7;58:34.068;12:44:02.398;0:33.329;1:17.343;1:25.023;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;FCY;
31;1;29;4:09.701;0;B;44.270;0;1:10.862;0;2:14.569;0;78.9;1:02:43.769;12:48:12.099;0:44.270;1:10.862;2:14.569;;Luke Rumburg;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;FCY;
31;2;30;2:32.389;0;;44.495;0;54.193;0;53.701;0;129.3;1:05:16.158;12:50:44.488;0:44.495;0:54.193;0:53.701;;Jaden Conwright;0:01:26.889;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;FCY;
31;2;31;2:39.490;0;;37.381;0;1:09.691;0;52.418;0;123.5;1:07:55.648;12:53:23.978;0:37.381;1:09.691;0:52.418;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;32;1:58.310;0;;28.274;0;44.586;0;45.450;2;166.5;1:09:53.958;12:55:22.288;0:28.274;0:44.586;0:45.450;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;33;1:56.630;2;;27.074;0;44.090;2;45.466;0;168.9;1:11:50.588;12:57:18.918;0:27.074;0:44.090;0:45.466;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;34;1:58.744;0;;27.118;0;45.333;0;46.293;0;165.9;1:13:49.332;12:59:17.662;0:27.118;0:45.333;0:46.293;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;35;1:57.870;0;;27.039;2;44.731;0;46.100;0;167.1;1:15:47.202;13:01:15.532;0:27.039;0:44.731;0:46.100;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;36;1:58.137;0;;27.236;0;44.672;0;46.229;0;166.7;1:17:45.339;13:03:13.669;0:27.236;0:44.672;0:46.229;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;37;1:57.489;0;;27.425;0;44.229;0;45.835;0;167.7;1:19:42.828;13:05:11.158;0:27.425;0:44.229;0:45.835;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;38;1:58.172;0;;27.222;0;44.717;0;46.233;0;166.7;1:21:41.000;13:07:09.330;0:27.222;0:44.717;0:46.233;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;39;1:57.892;0;;27.321;0;44.506;0;46.065;0;167.1;1:23:38.892;13:09:07.222;0:27.321;0:44.506;0:46.065;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;40;1:57.778;0;;27.321;0;44.456;0;46.001;0;167.3;1:25:36.670;13:11:05.000;0:27.321;0:44.456;0:46.001;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;41;1:58.296;0;;27.466;0;44.555;0;46.275;0;166.5;1:27:34.966;13:13:03.296;0:27.466;0:44.555;0:46.275;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;42;1:58.416;0;;27.368;0;44.832;0;46.216;0;166.3;1:29:33.382;13:15:01.712;0:27.368;0:44.832;0:46.216;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;43;1:59.426;0;;27.529;0;45.077;0;46.820;0;164.9;1:31:32.808;13:17:01.138;0:27.529;0:45.077;0:46.820;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;44;2:01.979;0;;28.680;0;46.064;0;47.235;0;161.5;1:33:34.787;13:19:03.117;0:28.680;0:46.064;0:47.235;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;45;7:35.887;0;B;30.526;0;48.331;0;6:17.030;0;43.2;1:41:10.674;13:26:39.004;0:30.526;0:48.331;6:17.030;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;46;2:10.263;0;;39.449;0;44.821;0;45.993;0;151.2;1:43:20.937;13:28:49.267;0:39.449;0:44.821;0:45.993;;Jaden Conwright;0:05:45.457;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;47;1:58.125;0;;27.423;0;44.553;0;46.149;0;166.8;1:45:19.062;13:30:47.392;0:27.423;0:44.553;0:46.149;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;48;2:02.735;0;;28.001;0;47.805;0;46.929;0;160.5;1:47:21.797;13:32:50.127;0:28.001;0:47.805;0:46.929;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;FCY;
31;2;49;3:20.611;0;;49.088;0;1:34.540;0;56.983;0;98.2;1:50:42.408;13:36:10.738;0:49.088;1:34.540;0:56.983;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;50;2:03.381;0;;27.584;0;48.305;0;47.492;0;159.7;1:52:45.789;13:38:14.119;0:27.584;0:48.305;0:47.492;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;51;2:01.308;0;;27.846;0;45.368;0;48.094;0;162.4;1:54:47.097;13:40:15.427;0:27.846;0:45.368;0:48.094;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;GF;
31;2;52;2:03.233;0;;28.193;0;45.032;0;50.008;0;159.8;1:56:50.330;13:42:18.660;0:28.193;0:45.032;0:50.008;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;FCY;
31;2;53;3:25.229;0;;49.471;0;1:17.685;0;1:18.073;0;96.0;2:00:15.559;13:45:43.889;0:49.471;1:17.685;1:18.073;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;FCY;
31;2;54;3:28.065;0;;55.948;0;1:13.437;0;1:18.680;0;94.7;2:03:43.624;13:49:11.954;0:55.948;1:13.437;1:18.680;;Jaden Conwright;;TCR;;RVA Graphics Motorsports By Speed Syndicate;Audi;FF;
33;2;1;2:15.765;0;;41.612;0;46.460;0;47.693;0;145.1;2:15.765;11:47:44.095;0:41.612;0:46.460;0:47.693;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;2;2:02.439;0;;28.085;0;46.503;0;47.851;0;160.9;4:18.204;11:49:46.534;0:28.085;0:46.503;0:47.851;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;2;3;2:32.337;0;;33.461;0;1:00.497;0;58.379;0;129.3;6:50.541;11:52:18.871;0:33.461;1:00.497;0:58.379;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;2;4;2:54.053;0;;42.596;0;1:13.655;0;57.802;0;113.2;9:44.594;11:55:12.924;0:42.596;1:13.655;0:57.802;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;5;2:03.436;0;;28.478;0;46.742;0;48.216;0;159.6;11:48.030;11:57:16.360;0:28.478;0:46.742;0:48.216;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;6;2:02.066;0;;28.989;0;46.213;0;46.864;0;161.4;13:50.096;11:59:18.426;0:28.989;0:46.213;0:46.864;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;7;1:59.992;0;;28.254;0;45.271;0;46.467;1;164.2;15:50.088;12:01:18.418;0:28.254;0:45.271;0:46.467;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;8;1:59.183;0;;27.813;0;44.784;1;46.586;0;165.3;17:49.271;12:03:17.601;0:27.813;0:44.784;0:46.586;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;9;2:00.185;0;;27.745;1;45.659;0;46.781;0;163.9;19:49.456;12:05:17.786;0:27.745;0:45.659;0:46.781;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;10;2:01.461;0;;28.373;0;46.105;0;46.983;0;162.2;21:50.917;12:07:19.247;0:28.373;0:46.105;0:46.983;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;11;2:00.235;0;;27.956;0;45.388;0;46.891;0;163.8;23:51.152;12:09:19.482;0:27.956;0:45.388;0:46.891;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;12;2:00.841;0;;27.970;0;45.481;0;47.390;0;163.0;25:51.993;12:11:20.323;0:27.970;0:45.481;0:47.390;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;13;2:01.005;0;;27.915;0;46.186;0;46.904;0;162.8;27:52.998;12:13:21.328;0:27.915;0:46.186;0:46.904;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;14;2:00.417;0;;27.979;0;45.351;0;47.087;0;163.6;29:53.415;12:15:21.745;0:27.979;0:45.351;0:47.087;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;15;2:00.407;0;;27.903;0;45.589;0;46.915;0;163.6;31:53.822;12:17:22.152;0:27.903;0:45.589;0:46.915;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;16;2:00.333;0;;27.827;0;45.192;0;47.314;0;163.7;33:54.155;12:19:22.485;0:27.827;0:45.192;0:47.314;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;17;1:59.992;0;;27.852;0;45.194;0;46.946;0;164.2;35:54.147;12:21:22.477;0:27.852;0:45.194;0:46.946;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;18;2:00.323;0;;27.961;0;45.335;0;47.027;0;163.7;37:54.470;12:23:22.800;0:27.961;0:45.335;0:47.027;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;19;2:00.589;0;;27.974;0;45.473;0;47.142;0;163.4;39:55.059;12:25:23.389;0:27.974;0:45.473;0:47.142;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;20;2:00.260;0;;27.954;0;45.357;0;46.949;0;163.8;41:55.319;12:27:23.649;0:27.954;0:45.357;0:46.949;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;21;2:00.369;0;;27.890;0;45.636;0;46.843;0;163.6;43:55.688;12:29:24.018;0:27.890;0:45.636;0:46.843;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;22;1:59.902;0;;27.795;0;45.225;0;46.882;0;164.3;45:55.590;12:31:23.920;0:27.795;0:45.225;0:46.882;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;23;1:59.891;0;;27.877;0;45.343;0;46.671;0;164.3;47:55.481;12:33:23.811;0:27.877;0:45.343;0:46.671;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;24;2:00.014;0;;27.931;0;45.288;0;46.795;0;164.1;49:55.495;12:35:23.825;0:27.931;0:45.288;0:46.795;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;25;1:59.857;0;;27.756;0;44.961;0;47.140;0;164.3;51:55.352;12:37:23.682;0:27.756;0:44.961;0:47.140;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;26;2:01.135;0;;28.207;0;45.521;0;47.407;0;162.6;53:56.487;12:39:24.817;0:28.207;0:45.521;0:47.407;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;2;27;2:15.838;0;;27.989;0;50.757;0;57.092;0;145.0;56:12.325;12:41:40.655;0:27.989;0:50.757;0:57.092;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;2;28;2:30.472;0;;35.209;0;55.655;0;59.608;0;130.9;58:42.797;12:44:11.127;0:35.209;0:55.655;0:59.608;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;2;29;3:14.251;0;B;46.477;0;1:09.162;0;1:18.612;0;101.4;1:01:57.048;12:47:25.378;0:46.477;1:09.162;1:18.612;;Taylor Hagler;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;1;30;3:27.568;0;;1:40.373;0;56.272;0;50.923;0;94.9;1:05:24.616;12:50:52.946;1:40.373;0:56.272;0:50.923;;Mark Wilkins;0:01:26.648;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;1;31;2:34.088;0;;33.825;0;1:08.371;0;51.892;0;127.8;1:07:58.704;12:53:27.034;0:33.825;1:08.371;0:51.892;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;32;1:57.994;0;;27.952;0;44.408;2;45.634;2;166.9;1:09:56.698;12:55:25.028;0:27.952;0:44.408;0:45.634;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;33;1:57.698;2;;27.325;2;44.662;0;45.711;0;167.4;1:11:54.396;12:57:22.726;0:27.325;0:44.662;0:45.711;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;34;1:58.776;0;;27.423;0;44.504;0;46.849;0;165.8;1:13:53.172;12:59:21.502;0:27.423;0:44.504;0:46.849;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;35;1:59.028;0;;27.505;0;44.627;0;46.896;0;165.5;1:15:52.200;13:01:20.530;0:27.505;0:44.627;0:46.896;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;36;1:58.831;0;;27.766;0;44.935;0;46.130;0;165.8;1:17:51.031;13:03:19.361;0:27.766;0:44.935;0:46.130;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;37;1:58.340;0;;27.555;0;44.525;0;46.260;0;166.5;1:19:49.371;13:05:17.701;0:27.555;0:44.525;0:46.260;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;38;1:58.374;0;;27.763;0;44.495;0;46.116;0;166.4;1:21:47.745;13:07:16.075;0:27.763;0:44.495;0:46.116;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;39;1:58.926;0;;27.874;0;44.739;0;46.313;0;165.6;1:23:46.671;13:09:15.001;0:27.874;0:44.739;0:46.313;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;40;1:58.946;0;;27.704;0;44.811;0;46.431;0;165.6;1:25:45.617;13:11:13.947;0:27.704;0:44.811;0:46.431;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;41;1:59.005;0;;27.832;0;44.755;0;46.418;0;165.5;1:27:44.622;13:13:12.952;0:27.832;0:44.755;0:46.418;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;42;1:59.429;0;;27.914;0;44.851;0;46.664;0;164.9;1:29:44.051;13:15:12.381;0:27.914;0:44.851;0:46.664;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;43;1:59.781;0;;27.714;0;45.065;0;47.002;0;164.5;1:31:43.832;13:17:12.162;0:27.714;0:45.065;0:47.002;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;44;2:00.234;0;;28.331;0;45.224;0;46.679;0;163.8;1:33:44.066;13:19:12.396;0:28.331;0:45.224;0:46.679;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;45;1:59.697;0;;27.967;0;44.933;0;46.797;0;164.6;1:35:43.763;13:21:12.093;0:27.967;0:44.933;0:46.797;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;46;1:59.644;0;;27.888;0;45.040;0;46.716;0;164.6;1:37:43.407;13:23:11.737;0:27.888;0:45.040;0:46.716;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;47;1:59.964;0;;27.955;0;45.204;0;46.805;0;164.2;1:39:43.371;13:25:11.701;0:27.955;0:45.204;0:46.805;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;48;2:00.787;0;;27.869;0;45.933;0;46.985;0;163.1;1:41:44.158;13:27:12.488;0:27.869;0:45.933;0:46.985;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;49;2:00.855;0;;28.061;0;45.738;0;47.056;0;163.0;1:43:45.013;13:29:13.343;0:28.061;0:45.738;0:47.056;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;50;2:25.709;0;B;28.566;0;52.422;0;1:04.721;0;135.2;1:46:10.722;13:31:39.052;0:28.566;0:52.422;1:04.721;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;1;51;2:59.662;0;;1:20.397;0;51.143;0;48.122;0;109.6;1:49:10.384;13:34:38.714;1:20.397;0:51.143;0:48.122;;Mark Wilkins;0:01:08.014;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;1;52;2:12.622;0;B;28.248;0;45.323;0;59.051;0;148.5;1:51:23.006;13:36:51.336;0:28.248;0:45.323;0:59.051;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;53;2:11.572;0;;39.746;0;46.016;0;45.810;0;149.7;1:53:34.578;13:39:02.908;0:39.746;0:46.016;0:45.810;;Mark Wilkins;0:00:30.326;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;54;1:58.709;0;;27.631;0;45.181;0;45.897;0;165.9;1:55:33.287;13:41:01.617;0:27.631;0:45.181;0:45.897;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
33;1;55;2:00.704;0;;27.744;0;46.469;0;46.491;0;163.2;1:57:33.991;13:43:02.321;0:27.744;0:46.469;0:46.491;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;1;56;2:45.465;0;;27.796;0;1:00.338;0;1:17.331;0;119.0;2:00:19.456;13:45:47.786;0:27.796;1:00.338;1:17.331;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
33;1;57;3:27.614;0;;56.744;0;1:12.562;0;1:18.308;0;94.9;2:03:47.070;13:49:15.400;0:56.744;1:12.562;1:18.308;;Mark Wilkins;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FF;
37;1;1;2:17.412;0;;42.810;0;47.986;0;46.616;0;143.4;2:17.412;11:47:45.742;0:42.810;0:47.986;0:46.616;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;2;2:03.853;0;;27.476;2;48.209;0;48.168;0;159.0;4:21.265;11:49:49.595;0:27.476;0:48.209;0:48.168;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;FCY;
37;1;3;2:31.008;0;;33.159;0;59.454;0;58.395;0;130.4;6:52.273;11:52:20.603;0:33.159;0:59.454;0:58.395;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;FCY;
37;1;4;2:53.002;0;;43.390;0;1:12.076;0;57.536;0;113.9;9:45.275;11:55:13.605;0:43.390;1:12.076;0:57.536;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;5;2:17.424;0;B;29.064;0;46.343;0;1:02.017;0;143.3;12:02.699;11:57:31.029;0:29.064;0:46.343;1:02.017;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;6;2:13.026;0;;39.968;0;46.205;0;46.853;0;148.1;14:15.725;11:59:44.055;0:39.968;0:46.205;0:46.853;;Megan Tomlinson;0:00:30.821;TCR;;Precision Racing LA;Audi;GF;
37;1;7;2:00.399;0;;27.735;0;46.183;0;46.481;2;163.6;16:16.124;12:01:44.454;0:27.735;0:46.183;0:46.481;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;8;2:00.383;0;;27.668;0;45.805;0;46.910;0;163.6;18:16.507;12:03:44.837;0:27.668;0:45.805;0:46.910;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;9;2:00.855;0;;27.634;0;46.184;0;47.037;0;163.0;20:17.362;12:05:45.692;0:27.634;0:46.184;0:47.037;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;10;2:00.924;0;;27.710;0;46.396;0;46.818;0;162.9;22:18.286;12:07:46.616;0:27.710;0:46.396;0:46.818;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;11;2:00.944;0;;27.985;0;45.898;0;47.061;0;162.9;24:19.230;12:09:47.560;0:27.985;0:45.898;0:47.061;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;12;2:00.756;0;;27.868;0;46.097;0;46.791;0;163.1;26:19.986;12:11:48.316;0:27.868;0:46.097;0:46.791;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;13;2:01.114;0;;27.810;0;46.436;0;46.868;0;162.6;28:21.100;12:13:49.430;0:27.810;0:46.436;0:46.868;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;14;2:00.447;0;;27.611;0;46.161;0;46.675;0;163.5;30:21.547;12:15:49.877;0:27.611;0:46.161;0:46.675;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;15;2:00.514;0;;27.485;0;46.181;0;46.848;0;163.5;32:22.061;12:17:50.391;0:27.485;0:46.181;0:46.848;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;16;2:00.814;0;;27.633;0;46.361;0;46.820;0;163.0;34:22.875;12:19:51.205;0:27.633;0:46.361;0:46.820;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;17;2:01.642;0;;28.240;0;46.258;0;47.144;0;161.9;36:24.517;12:21:52.847;0:28.240;0:46.258;0:47.144;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;18;2:00.610;0;;27.944;0;45.776;0;46.890;0;163.3;38:25.127;12:23:53.457;0:27.944;0:45.776;0:46.890;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;19;2:00.544;0;;27.867;0;45.748;0;46.929;0;163.4;40:25.671;12:25:54.001;0:27.867;0:45.748;0:46.929;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;20;2:00.608;0;;27.719;0;45.767;0;47.122;0;163.3;42:26.279;12:27:54.609;0:27.719;0:45.767;0:47.122;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;21;2:00.350;0;;27.551;0;45.727;1;47.072;0;163.7;44:26.629;12:29:54.959;0:27.551;0:45.727;0:47.072;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;22;2:01.386;0;;27.611;0;46.489;0;47.286;0;162.3;46:28.015;12:31:56.345;0:27.611;0:46.489;0:47.286;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;23;2:00.531;0;;27.512;0;45.825;0;47.194;0;163.4;48:28.546;12:33:56.876;0:27.512;0:45.825;0:47.194;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;24;2:02.569;0;;28.226;0;46.072;0;48.271;0;160.7;50:31.115;12:35:59.445;0:28.226;0:46.072;0:48.271;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;25;2:01.462;0;;27.765;0;46.315;0;47.382;0;162.2;52:32.577;12:38:00.907;0:27.765;0:46.315;0:47.382;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;26;2:00.716;0;;27.892;0;45.842;0;46.982;0;163.2;54:33.293;12:40:01.623;0:27.892;0:45.842;0:46.982;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;1;27;2:12.552;0;;28.367;0;53.494;0;50.691;0;148.6;56:45.845;12:42:14.175;0:28.367;0:53.494;0:50.691;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;FCY;
37;1;28;2:09.527;0;;30.409;0;49.555;0;49.563;0;152.1;58:55.372;12:44:23.702;0:30.409;0:49.555;0:49.563;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;FCY;
37;1;29;3:05.756;0;B;38.122;0;1:09.505;0;1:18.129;0;106.0;1:02:01.128;12:47:29.458;0:38.122;1:09.505;1:18.129;;Megan Tomlinson;;TCR;;Precision Racing LA;Audi;FCY;
37;2;30;3:32.624;0;;1:48.402;0;52.255;0;51.967;0;92.6;1:05:33.752;12:51:02.082;1:48.402;0:52.255;0:51.967;;Ron Tomlinson;0:01:35.579;TCR;;Precision Racing LA;Audi;FCY;
37;2;31;2:31.208;0;;30.789;0;1:05.473;0;54.946;0;130.3;1:08:04.960;12:53:33.290;0:30.789;1:05.473;0:54.946;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;32;2:04.255;0;;29.121;0;47.422;0;47.712;0;158.5;1:10:09.215;12:55:37.545;0:29.121;0:47.422;0:47.712;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;33;2:03.223;0;;28.445;0;46.956;0;47.822;0;159.9;1:12:12.438;12:57:40.768;0:28.445;0:46.956;0:47.822;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;34;2:01.298;0;;27.981;0;46.461;0;46.856;0;162.4;1:14:13.736;12:59:42.066;0:27.981;0:46.461;0:46.856;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;35;2:01.271;0;;27.787;0;46.578;0;46.906;0;162.4;1:16:15.007;13:01:43.337;0:27.787;0:46.578;0:46.906;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;36;2:01.411;0;;27.973;0;46.324;0;47.114;0;162.2;1:18:16.418;13:03:44.748;0:27.973;0:46.324;0:47.114;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;37;2:01.405;0;;28.015;0;46.141;0;47.249;0;162.3;1:20:17.823;13:05:46.153;0:28.015;0:46.141;0:47.249;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;38;2:01.201;0;;27.914;0;46.138;0;47.149;0;162.5;1:22:19.024;13:07:47.354;0:27.914;0:46.138;0:47.149;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;39;2:02.339;0;;28.517;0;46.978;0;46.844;0;161.0;1:24:21.363;13:09:49.693;0:28.517;0:46.978;0:46.844;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;40;2:00.308;0;;27.893;0;45.916;0;46.499;1;163.7;1:26:21.671;13:11:50.001;0:27.893;0:45.916;0:46.499;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;41;2:00.814;0;;27.825;0;45.840;0;47.149;0;163.0;1:28:22.485;13:13:50.815;0:27.825;0:45.840;0:47.149;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;42;2:00.188;2;;27.546;1;45.726;0;46.916;0;163.9;1:30:22.673;13:15:51.003;0:27.546;0:45.726;0:46.916;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;43;2:00.454;0;;27.784;0;45.664;2;47.006;0;163.5;1:32:23.127;13:17:51.457;0:27.784;0:45.664;0:47.006;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;44;2:01.140;0;;27.718;0;46.168;0;47.254;0;162.6;1:34:24.267;13:19:52.597;0:27.718;0:46.168;0:47.254;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;45;2:00.808;0;;27.666;0;45.982;0;47.160;0;163.1;1:36:25.075;13:21:53.405;0:27.666;0:45.982;0:47.160;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;46;2:01.909;0;;27.923;0;46.959;0;47.027;0;161.6;1:38:26.984;13:23:55.314;0:27.923;0:46.959;0:47.027;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;47;2:00.990;0;;27.661;0;45.905;0;47.424;0;162.8;1:40:27.974;13:25:56.304;0:27.661;0:45.905;0:47.424;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;48;2:00.627;0;;27.699;0;45.853;0;47.075;0;163.3;1:42:28.601;13:27:56.931;0:27.699;0:45.853;0:47.075;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;49;2:01.141;0;;27.704;0;46.257;0;47.180;0;162.6;1:44:29.742;13:29:58.072;0:27.704;0:46.257;0:47.180;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;50;2:01.310;0;;27.705;0;45.957;0;47.648;0;162.4;1:46:31.052;13:31:59.382;0:27.705;0:45.957;0:47.648;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;FCY;
37;2;51;2:04.779;0;;27.946;0;49.135;0;47.698;0;157.9;1:48:35.831;13:34:04.161;0:27.946;0:49.135;0:47.698;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;FCY;
37;2;52;2:13.668;0;;27.833;0;51.764;0;54.071;0;147.4;1:50:49.499;13:36:17.829;0:27.833;0:51.764;0:54.071;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;53;2:01.235;0;;28.196;0;46.451;0;46.588;0;162.5;1:52:50.734;13:38:19.064;0:28.196;0:46.451;0:46.588;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;54;2:00.893;0;;27.715;0;45.973;0;47.205;0;162.9;1:54:51.627;13:40:19.957;0:27.715;0:45.973;0:47.205;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;GF;
37;2;55;2:10.878;0;;27.721;0;46.327;0;56.830;0;150.5;1:57:02.505;13:42:30.835;0:27.721;0:46.327;0:56.830;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;FCY;
37;2;56;3:14.255;0;;38.797;0;1:17.787;0;1:17.671;0;101.4;2:00:16.760;13:45:45.090;0:38.797;1:17.787;1:17.671;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;FCY;
37;2;57;3:27.806;0;;56.198;0;1:13.185;0;1:18.423;0;94.8;2:03:44.566;13:49:12.896;0:56.198;1:13.185;1:18.423;;Ron Tomlinson;;TCR;;Precision Racing LA;Audi;FF;
39;1;1;2:00.143;0;;29.216;0;44.926;0;46.001;0;164.0;2:00.143;11:47:28.473;0:29.216;0:44.926;0:46.001;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;2;1:57.805;0;;27.126;0;44.461;0;46.218;0;167.2;3:57.948;11:49:26.278;0:27.126;0:44.461;0:46.218;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;3;2:26.978;0;;29.341;0;59.592;0;58.045;0;134.0;6:24.926;11:51:53.256;0:29.341;0:59.592;0:58.045;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;FCY;
39;1;4;3:09.210;0;;45.269;0;1:17.507;0;1:06.434;0;104.1;9:34.136;11:55:02.466;0:45.269;1:17.507;1:06.434;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;5;1:58.927;0;;27.340;0;45.822;0;45.765;0;165.6;11:33.063;11:57:01.393;0:27.340;0:45.822;0:45.765;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;6;1:57.042;0;;27.185;0;44.083;0;45.774;0;168.3;13:30.105;11:58:58.435;0:27.185;0:44.083;0:45.774;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;7;1:56.779;0;;27.089;0;43.984;0;45.706;0;168.7;15:26.884;12:00:55.214;0:27.089;0:43.984;0:45.706;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;8;1:56.725;0;;27.046;0;43.990;0;45.689;0;168.8;17:23.609;12:02:51.939;0:27.046;0:43.990;0:45.689;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;9;1:56.676;0;;27.070;0;43.917;0;45.689;0;168.8;19:20.285;12:04:48.615;0:27.070;0:43.917;0:45.689;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;10;1:56.753;0;;27.079;0;43.961;0;45.713;0;168.7;21:17.038;12:06:45.368;0:27.079;0:43.961;0:45.713;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;11;1:56.951;0;;27.032;0;44.327;0;45.592;0;168.4;23:13.989;12:08:42.319;0:27.032;0:44.327;0:45.592;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;12;1:56.767;0;;27.085;0;43.966;0;45.716;0;168.7;25:10.756;12:10:39.086;0:27.085;0:43.966;0:45.716;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;13;1:56.861;0;;27.038;0;44.027;0;45.796;0;168.6;27:07.617;12:12:35.947;0:27.038;0:44.027;0:45.796;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;14;1:57.077;0;;27.117;0;44.030;0;45.930;0;168.3;29:04.694;12:14:33.024;0:27.117;0:44.030;0:45.930;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;15;1:57.145;0;;27.252;0;44.145;0;45.748;0;168.2;31:01.839;12:16:30.169;0:27.252;0:44.145;0:45.748;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;16;1:56.580;0;;26.994;0;43.882;0;45.704;0;169.0;32:58.419;12:18:26.749;0:26.994;0:43.882;0:45.704;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;17;1:57.227;0;;27.030;0;43.859;0;46.338;0;168.0;34:55.646;12:20:23.976;0:27.030;0:43.859;0:46.338;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;18;1:56.173;2;;27.137;0;43.550;3;45.486;1;169.6;36:51.819;12:22:20.149;0:27.137;0:43.550;0:45.486;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;19;1:56.220;0;;27.114;0;43.569;0;45.537;0;169.5;38:48.039;12:24:16.369;0:27.114;0:43.569;0:45.537;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;20;1:56.899;0;;27.067;0;44.125;0;45.707;0;168.5;40:44.938;12:26:13.268;0:27.067;0:44.125;0:45.707;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;21;1:57.542;0;;26.962;1;44.243;0;46.337;0;167.6;42:42.480;12:28:10.810;0:26.962;0:44.243;0:46.337;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;22;1:57.121;0;;26.992;0;44.182;0;45.947;0;168.2;44:39.601;12:30:07.931;0:26.992;0:44.182;0:45.947;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;23;1:57.457;0;;27.013;0;44.263;0;46.181;0;167.7;46:37.058;12:32:05.388;0:27.013;0:44.263;0:46.181;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;24;1:57.230;0;;27.017;0;44.171;0;46.042;0;168.0;48:34.288;12:34:02.618;0:27.017;0:44.171;0:46.042;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;25;1:58.085;0;;27.034;0;44.504;0;46.547;0;166.8;50:32.373;12:36:00.703;0:27.034;0:44.504;0:46.547;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;1;26;2:59.675;0;B;27.102;0;44.575;0;1:47.998;0;109.6;53:32.048;12:39:00.378;0:27.102;0:44.575;1:47.998;;Sean McAlister;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;27;2:30.026;0;;39.262;0;51.051;0;59.713;0;131.3;56:02.074;12:41:30.404;0:39.262;0:51.051;0:59.713;;Jeff Westphal;0:01:20.033;GS;;CarBahn with Peregrine racing;BMW;FCY;
39;2;28;2:55.934;0;B;35.122;0;50.820;0;1:29.992;0;112.0;58:58.008;12:44:26.338;0:35.122;0:50.820;1:29.992;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;FCY;
39;2;29;2:49.529;0;;40.146;0;1:05.746;0;1:03.637;0;116.2;1:01:47.537;12:47:15.867;0:40.146;1:05.746;1:03.637;;Jeff Westphal;0:00:37.511;GS;;CarBahn with Peregrine racing;BMW;FCY;
39;2;30;2:51.093;0;;30.631;0;1:05.158;0;1:15.304;0;115.1;1:04:38.630;12:50:06.960;0:30.631;1:05.158;1:15.304;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;FCY;
39;2;31;3:04.213;0;;51.223;0;1:14.617;0;58.373;0;106.9;1:07:42.843;12:53:11.173;0:51.223;1:14.617;0:58.373;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;32;1:58.184;0;;27.479;0;44.865;0;45.840;0;166.7;1:09:41.027;12:55:09.357;0:27.479;0:44.865;0:45.840;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;33;1:58.110;0;;27.138;0;44.953;0;46.019;0;166.8;1:11:39.137;12:57:07.467;0:27.138;0:44.953;0:46.019;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;34;1:56.456;0;;27.121;0;43.891;1;45.444;2;169.1;1:13:35.593;12:59:03.923;0:27.121;0:43.891;0:45.444;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;35;1:57.052;0;;26.988;0;43.919;0;46.145;0;168.3;1:15:32.645;13:01:00.975;0:26.988;0:43.919;0:46.145;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;36;1:56.620;0;;27.153;0;43.932;0;45.535;0;168.9;1:17:29.265;13:02:57.595;0:27.153;0:43.932;0:45.535;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;37;1:56.702;0;;26.921;2;44.001;0;45.780;0;168.8;1:19:25.967;13:04:54.297;0:26.921;0:44.001;0:45.780;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;38;1:56.893;0;;27.065;0;44.063;0;45.765;0;168.5;1:21:22.860;13:06:51.190;0:27.065;0:44.063;0:45.765;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;39;1:56.740;0;;26.984;0;43.962;0;45.794;0;168.7;1:23:19.600;13:08:47.930;0:26.984;0:43.962;0:45.794;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;40;1:56.853;0;;26.990;0;44.162;0;45.701;0;168.6;1:25:16.453;13:10:44.783;0:26.990;0:44.162;0:45.701;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;41;1:56.508;0;;27.013;0;43.932;0;45.563;0;169.1;1:27:12.961;13:12:41.291;0:27.013;0:43.932;0:45.563;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;42;1:57.619;0;;26.995;0;44.406;0;46.218;0;167.5;1:29:10.580;13:14:38.910;0:26.995;0:44.406;0:46.218;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;43;1:56.846;0;;26.945;0;44.204;0;45.697;0;168.6;1:31:07.426;13:16:35.756;0:26.945;0:44.204;0:45.697;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;44;1:56.993;0;;27.079;0;44.105;0;45.809;0;168.4;1:33:04.419;13:18:32.749;0:27.079;0:44.105;0:45.809;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;45;1:57.835;0;;27.019;0;44.400;0;46.416;0;167.2;1:35:02.254;13:20:30.584;0:27.019;0:44.400;0:46.416;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;46;1:57.521;0;;27.259;0;44.263;0;45.999;0;167.6;1:36:59.775;13:22:28.105;0:27.259;0:44.263;0:45.999;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;47;1:57.585;0;;26.980;0;44.728;0;45.877;0;167.5;1:38:57.360;13:24:25.690;0:26.980;0:44.728;0:45.877;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;48;1:56.797;0;;26.930;0;44.105;0;45.762;0;168.7;1:40:54.157;13:26:22.487;0:26.930;0:44.105;0:45.762;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;49;1:57.150;0;;27.090;0;44.298;0;45.762;0;168.1;1:42:51.307;13:28:19.637;0:27.090;0:44.298;0:45.762;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;50;1:57.269;0;;27.042;0;44.236;0;45.991;0;168.0;1:44:48.576;13:30:16.906;0:27.042;0:44.236;0:45.991;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;51;2:15.844;0;;26.983;0;48.218;0;1:00.643;0;145.0;1:47:04.420;13:32:32.750;0:26.983;0:48.218;1:00.643;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;FCY;
39;2;52;3:30.506;0;;54.553;0;1:29.948;0;1:06.005;0;93.6;1:50:34.926;13:36:03.256;0:54.553;1:29.948;1:06.005;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;53;1:58.097;0;;27.459;0;44.305;0;46.333;0;166.8;1:52:33.023;13:38:01.353;0:27.459;0:44.305;0:46.333;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;54;1:57.201;0;;26.999;0;44.450;0;45.752;0;168.1;1:54:30.224;13:39:58.554;0:26.999;0:44.450;0:45.752;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;GF;
39;2;55;2:02.500;0;;26.967;0;44.217;0;51.316;0;160.8;1:56:32.724;13:42:01.054;0:26.967;0:44.217;0:51.316;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;FCY;
39;2;56;3:22.755;0;;48.077;0;1:18.207;0;1:16.471;0;97.2;1:59:55.479;13:45:23.809;0:48.077;1:18.207;1:16.471;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;FCY;
39;2;57;3:26.579;0;;52.543;0;1:17.343;0;1:16.693;0;95.4;2:03:22.058;13:48:50.388;0:52.543;1:17.343;1:16.693;;Jeff Westphal;;GS;;CarBahn with Peregrine racing;BMW;FF;
4;1;1;2:04.078;0;;31.188;0;46.404;0;46.486;0;158.8;2:04.078;11:47:32.408;0:31.188;0:46.404;0:46.486;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;2;1:58.809;0;;28.098;0;44.483;1;46.228;0;165.8;4:02.887;11:49:31.217;0:28.098;0:44.483;0:46.228;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;3;2:29.513;0;;30.247;0;59.873;0;59.393;0;131.8;6:32.400;11:52:00.730;0:30.247;0:59.873;0:59.393;;Bill Cain;;GS;;CarBahn;BMW;FCY;
4;1;4;3:04.853;0;;43.929;0;1:16.646;0;1:04.278;0;106.6;9:37.253;11:55:05.583;0:43.929;1:16.646;1:04.278;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;5;1:59.058;0;;27.728;0;44.980;0;46.350;0;165.5;11:36.311;11:57:04.641;0:27.728;0:44.980;0:46.350;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;6;1:58.892;0;;27.784;0;44.707;0;46.401;0;165.7;13:35.203;11:59:03.533;0:27.784;0:44.707;0:46.401;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;7;1:58.977;0;;27.954;0;45.068;0;45.955;1;165.6;15:34.180;12:01:02.510;0:27.954;0:45.068;0:45.955;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;8;1:58.454;0;;27.264;1;44.976;0;46.214;0;166.3;17:32.634;12:03:00.964;0:27.264;0:44.976;0:46.214;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;9;1:58.761;0;;27.834;0;44.917;0;46.010;0;165.9;19:31.395;12:04:59.725;0:27.834;0:44.917;0:46.010;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;10;2:15.310;0;B;29.051;0;45.002;0;1:01.257;0;145.6;21:46.705;12:07:15.035;0:29.051;0:45.002;1:01.257;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;11;2:10.756;0;;39.082;0;45.254;0;46.420;0;150.6;23:57.461;12:09:25.791;0:39.082;0:45.254;0:46.420;;Bill Cain;0:00:30.298;GS;;CarBahn;BMW;GF;
4;1;12;2:01.103;0;;27.712;0;46.688;0;46.703;0;162.7;25:58.564;12:11:26.894;0:27.712;0:46.688;0:46.703;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;13;1:58.612;0;;27.518;0;44.980;0;46.114;0;166.1;27:57.176;12:13:25.506;0:27.518;0:44.980;0:46.114;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;14;1:58.917;0;;27.394;0;45.034;0;46.489;0;165.6;29:56.093;12:15:24.423;0:27.394;0:45.034;0:46.489;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;15;1:58.435;0;;27.311;0;44.672;0;46.452;0;166.3;31:54.528;12:17:22.858;0:27.311;0:44.672;0:46.452;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;16;1:58.977;0;;27.659;0;44.980;0;46.338;0;165.6;33:53.505;12:19:21.835;0:27.659;0:44.980;0:46.338;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;17;1:59.718;0;;27.501;0;45.443;0;46.774;0;164.5;35:53.223;12:21:21.553;0:27.501;0:45.443;0:46.774;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;18;1:58.695;0;;27.350;0;45.274;0;46.071;0;166.0;37:51.918;12:23:20.248;0:27.350;0:45.274;0:46.071;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;19;1:58.619;0;;27.411;0;44.837;0;46.371;0;166.1;39:50.537;12:25:18.867;0:27.411;0:44.837;0:46.371;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;1;20;3:17.489;0;B;27.622;0;45.192;0;2:04.675;0;99.7;43:08.026;12:28:36.356;0:27.622;0:45.192;2:04.675;;Bill Cain;;GS;;CarBahn;BMW;GF;
4;2;21;2:13.117;0;;40.087;0;46.335;0;46.695;0;148.0;45:21.143;12:30:49.473;0:40.087;0:46.335;0:46.695;;Aaron Povoledo;0:01:33.824;GS;;CarBahn;BMW;GF;
4;2;22;1:57.928;0;;27.601;0;44.541;0;45.786;0;167.0;47:19.071;12:32:47.401;0:27.601;0:44.541;0:45.786;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;23;1:57.830;0;;27.279;0;44.869;0;45.682;0;167.2;49:16.901;12:34:45.231;0:27.279;0:44.869;0:45.682;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;24;1:57.211;0;;27.158;0;44.244;0;45.809;0;168.1;51:14.112;12:36:42.442;0:27.158;0:44.244;0:45.809;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;25;1:56.935;0;;27.390;0;44.026;0;45.519;0;168.5;53:11.047;12:38:39.377;0:27.390;0:44.026;0:45.519;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;26;1:58.065;0;;27.210;0;44.386;0;46.469;0;166.8;55:09.112;12:40:37.442;0:27.210;0:44.386;0:46.469;;Aaron Povoledo;;GS;;CarBahn;BMW;FCY;
4;2;27;3:52.298;0;B;39.752;0;1:18.637;0;1:53.909;0;84.8;59:01.410;12:44:29.740;0:39.752;1:18.637;1:53.909;;Aaron Povoledo;;GS;;CarBahn;BMW;FCY;
4;2;28;2:48.098;0;;39.820;0;1:04.365;0;1:03.913;0;117.2;1:01:49.508;12:47:17.838;0:39.820;1:04.365;1:03.913;;Aaron Povoledo;0:00:51.939;GS;;CarBahn;BMW;FCY;
4;2;29;2:51.173;0;;32.943;0;1:03.098;0;1:15.132;0;115.1;1:04:40.681;12:50:09.011;0:32.943;1:03.098;1:15.132;;Aaron Povoledo;;GS;;CarBahn;BMW;FCY;
4;2;30;3:02.759;0;;51.838;0;1:13.613;0;57.308;0;107.8;1:07:43.440;12:53:11.770;0:51.838;1:13.613;0:57.308;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;31;1:57.828;0;;27.334;0;44.648;0;45.846;0;167.2;1:09:41.268;12:55:09.598;0:27.334;0:44.648;0:45.846;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;32;1:58.164;0;;27.033;0;45.206;0;45.925;0;166.7;1:11:39.432;12:57:07.762;0:27.033;0:45.206;0:45.925;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;33;1:56.687;0;;26.968;0;44.249;0;45.470;2;168.8;1:13:36.119;12:59:04.449;0:26.968;0:44.249;0:45.470;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;34;1:57.413;0;;26.897;2;44.385;0;46.131;0;167.8;1:15:33.532;13:01:01.862;0:26.897;0:44.385;0:46.131;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;35;1:56.719;0;;27.033;0;44.057;0;45.629;0;168.8;1:17:30.251;13:02:58.581;0:27.033;0:44.057;0:45.629;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;36;1:56.582;2;;27.088;0;43.994;2;45.500;0;169.0;1:19:26.833;13:04:55.163;0:27.088;0:43.994;0:45.500;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;37;1:56.646;0;;26.963;0;44.007;0;45.676;0;168.9;1:21:23.479;13:06:51.809;0:26.963;0:44.007;0:45.676;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;38;1:56.957;0;;27.030;0;44.178;0;45.749;0;168.4;1:23:20.436;13:08:48.766;0:27.030;0:44.178;0:45.749;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;39;1:56.797;0;;26.968;0;44.272;0;45.557;0;168.7;1:25:17.233;13:10:45.563;0:26.968;0:44.272;0:45.557;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;40;1:56.657;0;;26.967;0;44.134;0;45.556;0;168.9;1:27:13.890;13:12:42.220;0:26.967;0:44.134;0:45.556;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;41;1:57.170;0;;27.023;0;44.067;0;46.080;0;168.1;1:29:11.060;13:14:39.390;0:27.023;0:44.067;0:46.080;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;42;1:57.166;0;;26.947;0;44.384;0;45.835;0;168.1;1:31:08.226;13:16:36.556;0:26.947;0:44.384;0:45.835;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;43;1:57.066;0;;27.013;0;44.201;0;45.852;0;168.3;1:33:05.292;13:18:33.622;0:27.013;0:44.201;0:45.852;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;44;1:57.203;0;;27.051;0;44.369;0;45.783;0;168.1;1:35:02.495;13:20:30.825;0:27.051;0:44.369;0:45.783;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;45;1:57.551;0;;27.164;0;44.416;0;45.971;0;167.6;1:37:00.046;13:22:28.376;0:27.164;0:44.416;0:45.971;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;46;1:57.623;0;;27.009;0;44.740;0;45.874;0;167.5;1:38:57.669;13:24:25.999;0:27.009;0:44.740;0:45.874;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;47;1:56.930;0;;26.946;0;44.234;0;45.750;0;168.5;1:40:54.599;13:26:22.929;0:26.946;0:44.234;0:45.750;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;48;1:57.239;0;;27.033;0;44.332;0;45.874;0;168.0;1:42:51.838;13:28:20.168;0:27.033;0:44.332;0:45.874;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;49;1:57.029;0;;27.037;0;44.319;0;45.673;0;168.3;1:44:48.867;13:30:17.197;0:27.037;0:44.319;0:45.673;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
4;2;50;2:16.203;0;;27.167;0;48.270;0;1:00.766;0;144.6;1:47:05.070;13:32:33.400;0:27.167;0:48.270;1:00.766;;Aaron Povoledo;;GS;;CarBahn;BMW;FCY;
4;2;51;3:30.238;0;;54.652;0;1:30.349;0;1:05.237;0;93.7;1:50:35.308;13:36:03.638;0:54.652;1:30.349;1:05.237;;Aaron Povoledo;;GS;;CarBahn;BMW;GF;
44;1;1;2:01.220;0;;29.429;0;45.522;0;46.269;0;162.5;2:01.220;11:47:29.550;0:29.429;0:45.522;0:46.269;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;2;1:57.703;0;;27.130;0;44.290;0;46.283;0;167.4;3:58.923;11:49:27.253;0:27.130;0:44.290;0:46.283;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;3;2:27.296;0;;29.881;0;59.471;0;57.944;0;133.7;6:26.219;11:51:54.549;0:29.881;0:59.471;0:57.944;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;FCY;
44;1;4;3:08.672;0;;45.211;0;1:17.826;0;1:05.635;0;104.4;9:34.891;11:55:03.221;0:45.211;1:17.826;1:05.635;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;5;1:58.910;0;;27.166;0;45.669;0;46.075;0;165.7;11:33.801;11:57:02.131;0:27.166;0:45.669;0:46.075;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;6;1:57.290;0;;26.985;0;44.681;0;45.624;0;167.9;13:31.091;11:58:59.421;0:26.985;0:44.681;0:45.624;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;7;1:57.356;0;;27.016;0;44.601;0;45.739;0;167.9;15:28.447;12:00:56.777;0:27.016;0:44.601;0:45.739;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;8;1:57.549;0;;27.067;0;44.665;0;45.817;0;167.6;17:25.996;12:02:54.326;0:27.067;0:44.665;0:45.817;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;9;1:57.358;0;;27.045;0;44.500;0;45.813;0;167.8;19:23.354;12:04:51.684;0:27.045;0:44.500;0:45.813;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;10;1:56.896;0;;27.055;0;44.282;0;45.559;0;168.5;21:20.250;12:06:48.580;0:27.055;0:44.282;0:45.559;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;11;1:57.140;0;;27.079;0;44.391;0;45.670;0;168.2;23:17.390;12:08:45.720;0:27.079;0:44.391;0:45.670;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;12;1:57.205;0;;26.971;0;44.405;0;45.829;0;168.1;25:14.595;12:10:42.925;0:26.971;0:44.405;0:45.829;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;13;1:56.792;0;;26.942;0;44.367;0;45.483;2;168.7;27:11.387;12:12:39.717;0:26.942;0:44.367;0:45.483;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;14;1:57.271;0;;27.003;0;44.506;0;45.762;0;168.0;29:08.658;12:14:36.988;0:27.003;0:44.506;0:45.762;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;15;1:58.312;0;;27.042;0;44.463;0;46.807;0;166.5;31:06.970;12:16:35.300;0:27.042;0:44.463;0:46.807;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;16;1:57.361;0;;27.411;0;44.208;0;45.742;0;167.8;33:04.331;12:18:32.661;0:27.411;0:44.208;0:45.742;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;17;1:56.964;0;;26.925;2;44.454;0;45.585;0;168.4;35:01.295;12:20:29.625;0:26.925;0:44.454;0:45.585;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;18;1:57.024;0;;26.963;0;44.191;1;45.870;0;168.3;36:58.319;12:22:26.649;0:26.963;0:44.191;0:45.870;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;19;1:57.396;0;;26.999;0;44.635;0;45.762;0;167.8;38:55.715;12:24:24.045;0:26.999;0:44.635;0:45.762;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;20;1:58.682;0;;27.002;0;45.936;0;45.744;0;166.0;40:54.397;12:26:22.727;0:27.002;0:45.936;0:45.744;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;21;1:57.080;0;;27.013;0;44.232;0;45.835;0;168.2;42:51.477;12:28:19.807;0:27.013;0:44.232;0:45.835;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;22;1:57.977;0;;27.101;0;44.710;0;46.166;0;167.0;44:49.454;12:30:17.784;0:27.101;0:44.710;0:46.166;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;23;1:57.695;0;;27.266;0;44.532;0;45.897;0;167.4;46:47.149;12:32:15.479;0:27.266;0:44.532;0:45.897;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;24;1:57.724;0;;27.055;0;44.790;0;45.879;0;167.3;48:44.873;12:34:13.203;0:27.055;0:44.790;0:45.879;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;1;25;2:10.935;0;B;27.191;0;44.826;0;58.918;0;150.4;50:55.808;12:36:24.138;0:27.191;0:44.826;0:58.918;;Moisey Uretsky;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;26;2:57.800;0;;1:25.844;0;45.735;0;46.221;0;110.8;53:53.608;12:39:21.938;1:25.844;0:45.735;0:46.221;;Michael Cooper;0:01:16.535;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;27;2:11.897;0;;26.977;0;46.333;0;58.587;0;149.3;56:05.505;12:41:33.835;0:26.977;0:46.333;0:58.587;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;FCY;
44;2;28;2:47.248;0;B;37.215;0;51.555;0;1:18.478;0;117.8;58:52.753;12:44:21.083;0:37.215;0:51.555;1:18.478;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;FCY;
44;2;29;3:00.794;0;;59.896;0;55.712;0;1:05.186;0;109.0;1:01:53.547;12:47:21.877;0:59.896;0:55.712;1:05.186;;Michael Cooper;0:00:50.565;GS;;Ibiza Farm Motorsports;McLaren;FCY;
44;2;30;2:49.370;0;;35.258;0;58.911;0;1:15.201;0;116.3;1:04:42.917;12:50:11.247;0:35.258;0:58.911;1:15.201;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;FCY;
44;2;31;3:01.446;0;;52.328;0;1:14.142;0;54.976;0;108.6;1:07:44.363;12:53:12.693;0:52.328;1:14.142;0:54.976;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;32;1:57.910;0;;27.191;0;44.890;0;45.829;0;167.1;1:09:42.273;12:55:10.603;0:27.191;0:44.890;0:45.829;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;33;1:57.954;0;;26.971;1;44.912;0;46.071;0;167.0;1:11:40.227;12:57:08.557;0:26.971;0:44.912;0:46.071;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;34;1:56.740;0;;27.065;0;44.111;0;45.564;0;168.7;1:13:36.967;12:59:05.297;0:27.065;0:44.111;0:45.564;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;35;1:57.289;0;;27.002;0;44.055;0;46.232;0;167.9;1:15:34.256;13:01:02.586;0:27.002;0:44.055;0:46.232;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;36;1:56.623;0;;27.017;0;43.911;0;45.695;0;168.9;1:17:30.879;13:02:59.209;0:27.017;0:43.911;0:45.695;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;37;1:56.748;0;;27.162;0;44.046;0;45.540;1;168.7;1:19:27.627;13:04:55.957;0:27.162;0:44.046;0:45.540;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;38;1:56.560;2;;27.019;0;43.868;2;45.673;0;169.0;1:21:24.187;13:06:52.517;0:27.019;0:43.868;0:45.673;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;39;1:57.037;0;;27.055;0;44.182;0;45.800;0;168.3;1:23:21.224;13:08:49.554;0:27.055;0:44.182;0:45.800;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;40;1:56.628;0;;26.979;0;44.011;0;45.638;0;168.9;1:25:17.852;13:10:46.182;0:26.979;0:44.011;0:45.638;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;41;1:56.755;0;;27.010;0;44.154;0;45.591;0;168.7;1:27:14.607;13:12:42.937;0:27.010;0:44.154;0:45.591;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;42;1:56.912;0;;27.068;0;44.066;0;45.778;0;168.5;1:29:11.519;13:14:39.849;0:27.068;0:44.066;0:45.778;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;43;1:57.315;0;;26.994;0;44.445;0;45.876;0;167.9;1:31:08.834;13:16:37.164;0:26.994;0:44.445;0:45.876;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;44;1:57.165;0;;27.024;0;44.216;0;45.925;0;168.1;1:33:05.999;13:18:34.329;0:27.024;0:44.216;0:45.925;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;45;1:57.140;0;;27.075;0;44.169;0;45.896;0;168.2;1:35:03.139;13:20:31.469;0:27.075;0:44.169;0:45.896;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;46;1:57.394;0;;27.056;0;44.492;0;45.846;0;167.8;1:37:00.533;13:22:28.863;0:27.056;0:44.492;0:45.846;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;47;1:57.654;0;;27.247;0;44.528;0;45.879;0;167.4;1:38:58.187;13:24:26.517;0:27.247;0:44.528;0:45.879;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;48;1:57.120;0;;27.052;0;44.252;0;45.816;0;168.2;1:40:55.307;13:26:23.637;0:27.052;0:44.252;0:45.816;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;49;1:57.235;0;;27.130;0;44.244;0;45.861;0;168.0;1:42:52.542;13:28:20.872;0:27.130;0:44.244;0:45.861;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;50;1:57.053;0;;27.064;0;44.084;0;45.905;0;168.3;1:44:49.595;13:30:17.925;0:27.064;0:44.084;0:45.905;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;51;2:17.399;0;;26.980;0;49.166;0;1:01.253;0;143.4;1:47:06.994;13:32:35.324;0:26.980;0:49.166;1:01.253;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;FCY;
44;2;52;3:28.593;0;;53.975;0;1:30.624;0;1:03.994;0;94.4;1:50:35.587;13:36:03.917;0:53.975;1:30.624;1:03.994;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
44;2;53;1:50.005;0;B;27.697;0;;0;;0;179.1;1:52:25.592;13:37:53.922;0:27.697;;;;Michael Cooper;;GS;;Ibiza Farm Motorsports;McLaren;GF;
46;2;1;1:59.402;0;;28.324;0;45.249;0;45.829;0;165.0;1:59.402;11:47:27.732;0:28.324;0:45.249;0:45.829;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;2;1:57.224;0;;27.009;0;44.446;0;45.769;0;168.0;3:56.626;11:49:24.956;0:27.009;0:44.446;0:45.769;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;3;2:26.215;0;;28.902;0;59.652;0;57.661;0;134.7;6:22.841;11:51:51.171;0:28.902;0:59.652;0:57.661;;Paul Holton;;GS;;Team TGM;Aston Martin;FCY;
46;2;4;3:10.622;0;;45.572;0;1:17.781;0;1:07.269;0;103.3;9:33.463;11:55:01.793;0:45.572;1:17.781;1:07.269;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;5;1:56.873;0;;27.345;0;43.988;0;45.540;0;168.5;11:30.336;11:56:58.666;0:27.345;0:43.988;0:45.540;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;6;1:56.518;0;;27.039;0;43.896;2;45.583;0;169.1;13:26.854;11:58:55.184;0:27.039;0:43.896;0:45.583;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;7;1:56.354;2;;26.991;0;43.935;0;45.428;2;169.3;15:23.208;12:00:51.538;0:26.991;0:43.935;0:45.428;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;8;1:56.803;0;;27.210;0;44.013;0;45.580;0;168.6;17:20.011;12:02:48.341;0:27.210;0:44.013;0:45.580;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;9;1:56.619;0;;27.075;0;44.010;0;45.534;0;168.9;19:16.630;12:04:44.960;0:27.075;0:44.010;0:45.534;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;10;1:56.487;0;;26.999;0;44.010;0;45.478;0;169.1;21:13.117;12:06:41.447;0:26.999;0:44.010;0:45.478;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;11;1:56.488;0;;26.932;0;44.018;0;45.538;0;169.1;23:09.605;12:08:37.935;0:26.932;0:44.018;0:45.538;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;12;1:56.649;0;;27.007;0;44.093;0;45.549;0;168.9;25:06.254;12:10:34.584;0:27.007;0:44.093;0:45.549;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;13;1:56.803;0;;27.227;0;44.097;0;45.479;0;168.6;27:03.057;12:12:31.387;0:27.227;0:44.097;0:45.479;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;14;1:56.799;0;;26.990;0;44.232;0;45.577;0;168.7;28:59.856;12:14:28.186;0:26.990;0:44.232;0:45.577;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;15;1:56.624;0;;26.907;1;44.120;0;45.597;0;168.9;30:56.480;12:16:24.810;0:26.907;0:44.120;0:45.597;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;16;1:56.757;0;;26.920;0;44.060;0;45.777;0;168.7;32:53.237;12:18:21.567;0:26.920;0:44.060;0:45.777;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;17;1:56.665;0;;26.973;0;44.015;0;45.677;0;168.8;34:49.902;12:20:18.232;0:26.973;0:44.015;0:45.677;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;18;1:56.721;0;;27.029;0;43.992;0;45.700;0;168.8;36:46.623;12:22:14.953;0:27.029;0:43.992;0:45.700;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;19;1:56.930;0;;27.048;0;44.083;0;45.799;0;168.5;38:43.553;12:24:11.883;0:27.048;0:44.083;0:45.799;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;20;1:57.194;0;;27.089;0;44.464;0;45.641;0;168.1;40:40.747;12:26:09.077;0:27.089;0:44.464;0:45.641;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;21;1:56.787;0;;26.974;0;44.172;0;45.641;0;168.7;42:37.534;12:28:05.864;0:26.974;0:44.172;0:45.641;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;22;1:57.005;0;;26.957;0;44.318;0;45.730;0;168.4;44:34.539;12:30:02.869;0:26.957;0:44.318;0:45.730;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;23;1:56.942;0;;26.991;0;44.315;0;45.636;0;168.4;46:31.481;12:31:59.811;0:26.991;0:44.315;0:45.636;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;24;1:57.387;0;;26.963;0;44.360;0;46.064;0;167.8;48:28.868;12:33:57.198;0:26.963;0:44.360;0:46.064;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;25;1:57.124;0;;27.174;0;44.222;0;45.728;0;168.2;50:25.992;12:35:54.322;0:27.174;0:44.222;0:45.728;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;26;1:56.959;0;;27.071;0;44.238;0;45.650;0;168.4;52:22.951;12:37:51.281;0:27.071;0:44.238;0:45.650;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;2;27;2:10.601;0;B;27.882;0;44.413;0;58.306;0;150.8;54:33.552;12:40:01.882;0:27.882;0:44.413;0:58.306;;Paul Holton;;GS;;Team TGM;Aston Martin;GF;
46;1;28;4:05.287;0;;1:35.947;0;1:05.682;0;1:23.658;0;80.3;58:38.839;12:44:07.169;1:35.947;1:05.682;1:23.658;;Matt Plumb;0:01:23.128;GS;;Team TGM;Aston Martin;FCY;
46;1;29;3:01.766;0;;47.104;0;1:08.995;0;1:05.667;0;108.4;1:01:40.605;12:47:08.935;0:47.104;1:08.995;1:05.667;;Matt Plumb;;GS;;Team TGM;Aston Martin;FCY;
46;1;30;2:57.151;0;;31.879;0;1:09.547;0;1:15.725;0;111.2;1:04:37.756;12:50:06.086;0:31.879;1:09.547;1:15.725;;Matt Plumb;;GS;;Team TGM;Aston Martin;FCY;
46;1;31;3:04.915;0;;51.298;0;1:15.069;0;58.548;0;106.5;1:07:42.671;12:53:11.001;0:51.298;1:15.069;0:58.548;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;32;1:57.921;0;;27.202;0;44.752;0;45.967;0;167.0;1:09:40.592;12:55:08.922;0:27.202;0:44.752;0:45.967;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;33;1:59.313;0;;26.854;0;46.200;0;46.259;0;165.1;1:11:39.905;12:57:08.235;0:26.854;0:46.200;0:46.259;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;34;1:56.776;0;;26.934;0;44.391;0;45.451;1;168.7;1:13:36.681;12:59:05.011;0:26.934;0:44.391;0:45.451;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;35;1:57.317;0;;26.742;2;44.320;0;46.255;0;167.9;1:15:33.998;13:01:02.328;0:26.742;0:44.320;0:46.255;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;36;1:56.465;0;;26.833;0;44.124;0;45.508;0;169.1;1:17:30.463;13:02:58.793;0:26.833;0:44.124;0:45.508;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;37;1:56.808;0;;27.036;0;44.252;0;45.520;0;168.6;1:19:27.271;13:04:55.601;0:27.036;0:44.252;0:45.520;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;38;1:56.706;0;;26.965;0;44.078;1;45.663;0;168.8;1:21:23.977;13:06:52.307;0:26.965;0:44.078;0:45.663;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;39;1:56.743;0;;27.038;0;44.252;0;45.453;0;168.7;1:23:20.720;13:08:49.050;0:27.038;0:44.252;0:45.453;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;40;1:56.935;0;;26.768;0;44.474;0;45.693;0;168.5;1:25:17.655;13:10:45.985;0:26.768;0:44.474;0:45.693;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;41;1:56.634;0;;26.834;0;44.280;0;45.520;0;168.9;1:27:14.289;13:12:42.619;0:26.834;0:44.280;0:45.520;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;42;1:57.024;0;;26.884;0;44.352;0;45.788;0;168.3;1:29:11.313;13:14:39.643;0:26.884;0:44.352;0:45.788;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;43;1:57.272;0;;26.816;0;44.653;0;45.803;0;168.0;1:31:08.585;13:16:36.915;0:26.816;0:44.653;0:45.803;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;44;1:56.968;0;;26.864;0;44.388;0;45.716;0;168.4;1:33:05.553;13:18:33.883;0:26.864;0:44.388;0:45.716;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;45;1:57.302;0;;26.905;0;44.610;0;45.787;0;167.9;1:35:02.855;13:20:31.185;0:26.905;0:44.610;0:45.787;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;46;1:57.424;0;;26.973;0;44.623;0;45.828;0;167.8;1:37:00.279;13:22:28.609;0:26.973;0:44.623;0:45.828;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;47;1:57.725;0;;27.057;0;44.849;0;45.819;0;167.3;1:38:58.004;13:24:26.334;0:27.057;0:44.849;0:45.819;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;48;1:57.000;0;;26.900;0;44.401;0;45.699;0;168.4;1:40:55.004;13:26:23.334;0:26.900;0:44.401;0:45.699;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;49;1:57.086;0;;26.985;0;44.338;0;45.763;0;168.2;1:42:52.090;13:28:20.420;0:26.985;0:44.338;0:45.763;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;50;1:57.164;0;;26.960;0;44.406;0;45.798;0;168.1;1:44:49.254;13:30:17.584;0:26.960;0:44.406;0:45.798;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;51;2:17.014;0;;26.886;0;49.035;0;1:01.093;0;143.8;1:47:06.268;13:32:34.598;0:26.886;0:49.035;1:01.093;;Matt Plumb;;GS;;Team TGM;Aston Martin;FCY;
46;1;52;3:29.219;0;;54.114;0;1:30.679;0;1:04.426;0;94.2;1:50:35.487;13:36:03.817;0:54.114;1:30.679;1:04.426;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;53;2:00.003;0;;27.684;0;46.509;0;45.810;0;164.1;1:52:35.490;13:38:03.820;0:27.684;0:46.509;0:45.810;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;54;1:57.301;0;;26.889;0;44.669;0;45.743;0;167.9;1:54:32.791;13:40:01.121;0:26.889;0:44.669;0:45.743;;Matt Plumb;;GS;;Team TGM;Aston Martin;GF;
46;1;55;2:02.663;0;;26.980;0;44.460;0;51.223;0;160.6;1:56:35.454;13:42:03.784;0:26.980;0:44.460;0:51.223;;Matt Plumb;;GS;;Team TGM;Aston Martin;FCY;
46;1;56;3:21.289;0;;47.215;0;1:18.633;0;1:15.441;0;97.9;1:59:56.743;13:45:25.073;0:47.215;1:18.633;1:15.441;;Matt Plumb;;GS;;Team TGM;Aston Martin;FCY;
46;1;57;3:26.780;0;;52.696;0;1:17.450;0;1:16.634;0;95.3;2:03:23.523;13:48:51.853;0:52.696;1:17.450;1:16.634;;Matt Plumb;;GS;;Team TGM;Aston Martin;FF;
5;2;1;2:16.298;0;;42.207;0;46.626;0;47.465;0;144.5;2:16.298;11:47:44.628;0:42.207;0:46.626;0:47.465;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;2;2:04.188;0;;28.090;0;46.474;0;49.624;0;158.6;4:20.486;11:49:48.816;0:28.090;0:46.474;0:49.624;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;FCY;
5;2;3;2:30.476;0;;31.510;0;1:00.509;0;58.457;0;130.9;6:50.962;11:52:19.292;0:31.510;1:00.509;0:58.457;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;FCY;
5;2;4;2:54.155;0;;42.870;0;1:13.404;0;57.881;0;113.1;9:45.117;11:55:13.447;0:42.870;1:13.404;0:57.881;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;5;2:03.300;0;;28.835;0;46.237;0;48.228;0;159.8;11:48.417;11:57:16.747;0:28.835;0:46.237;0:48.228;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;6;2:01.452;0;;28.083;0;45.626;0;47.743;0;162.2;13:49.869;11:59:18.199;0:28.083;0:45.626;0:47.743;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;7;1:59.556;0;;28.219;0;44.985;0;46.352;0;164.8;15:49.425;12:01:17.755;0:28.219;0:44.985;0:46.352;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;8;1:59.371;0;;27.559;0;45.030;0;46.782;0;165.0;17:48.796;12:03:17.126;0:27.559;0:45.030;0:46.782;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;9;2:00.223;0;;27.784;0;45.621;0;46.818;0;163.8;19:49.019;12:05:17.349;0:27.784;0:45.621;0:46.818;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;10;2:00.950;0;;28.331;0;45.627;0;46.992;0;162.9;21:49.969;12:07:18.299;0:28.331;0:45.627;0:46.992;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;11;2:00.099;0;;28.002;0;45.479;0;46.618;0;164.0;23:50.068;12:09:18.398;0:28.002;0:45.479;0:46.618;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;12;2:00.854;0;;28.652;0;45.100;0;47.102;0;163.0;25:50.922;12:11:19.252;0:28.652;0:45.100;0:47.102;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;13;2:00.447;0;;28.191;0;45.327;0;46.929;0;163.5;27:51.369;12:13:19.699;0:28.191;0:45.327;0:46.929;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;14;2:00.611;0;;28.055;0;45.238;0;47.318;0;163.3;29:51.980;12:15:20.310;0:28.055;0:45.238;0:47.318;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;15;1:59.757;0;;27.957;0;44.854;0;46.946;0;164.5;31:51.737;12:17:20.067;0:27.957;0:44.854;0:46.946;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;16;1:58.405;0;;27.447;1;44.805;0;46.153;0;166.4;33:50.142;12:19:18.472;0:27.447;0:44.805;0:46.153;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;17;1:59.076;0;;27.698;0;44.848;0;46.530;0;165.4;35:49.218;12:21:17.548;0:27.698;0:44.848;0:46.530;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;18;1:58.905;0;;27.618;0;45.055;0;46.232;0;165.7;37:48.123;12:23:16.453;0:27.618;0:45.055;0:46.232;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;19;1:58.873;0;;27.842;0;44.798;1;46.233;0;165.7;39:46.996;12:25:15.326;0:27.842;0:44.798;0:46.233;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;20;2:00.121;0;;27.667;0;45.242;0;47.212;0;164.0;41:47.117;12:27:15.447;0:27.667;0:45.242;0:47.212;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;21;2:00.733;0;;27.761;0;45.775;0;47.197;0;163.2;43:47.850;12:29:16.180;0:27.761;0:45.775;0:47.197;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;22;1:58.803;0;;27.515;0;44.964;0;46.324;0;165.8;45:46.653;12:31:14.983;0:27.515;0:44.964;0:46.324;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;23;1:59.128;0;;27.645;0;45.372;0;46.111;1;165.4;47:45.781;12:33:14.111;0:27.645;0:45.372;0:46.111;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;24;1:59.315;0;;27.548;0;45.447;0;46.320;0;165.1;49:45.096;12:35:13.426;0:27.548;0:45.447;0:46.320;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;25;1:59.655;0;;27.845;0;45.377;0;46.433;0;164.6;51:44.751;12:37:13.081;0:27.845;0:45.377;0:46.433;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;26;2:00.383;0;;27.589;0;45.055;0;47.739;0;163.6;53:45.134;12:39:13.464;0:27.589;0:45.055;0:47.739;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;2;27;2:19.014;0;;29.395;0;51.161;0;58.458;0;141.7;56:04.148;12:41:32.478;0:29.395;0:51.161;0:58.458;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;FCY;
5;2;28;2:37.329;0;;37.743;0;51.999;0;1:07.587;0;125.2;58:41.477;12:44:09.807;0:37.743;0:51.999;1:07.587;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;FCY;
5;2;29;4:08.309;0;B;46.426;0;1:09.405;0;2:12.478;0;79.3;1:02:49.786;12:48:18.116;0:46.426;1:09.405;2:12.478;;William Tally;;TCR;;KMW Motorsports with TMR Engineering;Honda;FCY;
5;1;30;2:31.991;0;;46.460;0;54.550;0;50.981;0;129.6;1:05:21.777;12:50:50.107;0:46.460;0:54.550;0:50.981;;Tim Lewis;0:01:24.551;TCR;;KMW Motorsports with TMR Engineering;Honda;FCY;
5;1;31;2:36.757;0;;35.945;0;1:08.430;0;52.382;0;125.7;1:07:58.534;12:53:26.864;0:35.945;1:08.430;0:52.382;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;32;2:02.153;0;;28.905;0;46.914;0;46.334;0;161.3;1:10:00.687;12:55:29.017;0:28.905;0:46.914;0:46.334;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;33;1:58.674;0;;27.539;0;45.419;0;45.716;0;166.0;1:11:59.361;12:57:27.691;0:27.539;0:45.419;0:45.716;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;34;1:57.350;2;;27.300;0;44.361;2;45.689;2;167.9;1:13:56.711;12:59:25.041;0:27.300;0:44.361;0:45.689;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;35;1:58.345;0;;27.433;0;44.990;0;45.922;0;166.4;1:15:55.056;13:01:23.386;0:27.433;0:44.990;0:45.922;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;36;1:59.738;0;;28.538;0;45.134;0;46.066;0;164.5;1:17:54.794;13:03:23.124;0:28.538;0:45.134;0:46.066;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;37;1:59.508;0;;28.099;0;45.324;0;46.085;0;164.8;1:19:54.302;13:05:22.632;0:28.099;0:45.324;0:46.085;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;38;1:58.237;0;;27.233;2;44.737;0;46.267;0;166.6;1:21:52.539;13:07:20.869;0:27.233;0:44.737;0:46.267;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;39;1:57.731;0;;27.339;0;44.650;0;45.742;0;167.3;1:23:50.270;13:09:18.600;0:27.339;0:44.650;0:45.742;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;40;1:57.862;0;;27.619;0;44.402;0;45.841;0;167.1;1:25:48.132;13:11:16.462;0:27.619;0:44.402;0:45.841;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;41;1:57.679;0;;27.391;0;44.483;0;45.805;0;167.4;1:27:45.811;13:13:14.141;0:27.391;0:44.483;0:45.805;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;42;1:58.491;0;;27.321;0;44.541;0;46.629;0;166.2;1:29:44.302;13:15:12.632;0:27.321;0:44.541;0:46.629;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;43;1:58.852;0;;27.588;0;45.130;0;46.134;0;165.7;1:31:43.154;13:17:11.484;0:27.588;0:45.130;0:46.134;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;44;1:58.595;0;;27.750;0;44.685;0;46.160;0;166.1;1:33:41.749;13:19:10.079;0:27.750;0:44.685;0:46.160;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;45;1:58.549;0;;27.472;0;44.892;0;46.185;0;166.2;1:35:40.298;13:21:08.628;0:27.472;0:44.892;0:46.185;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;46;1:58.644;0;;27.376;0;44.972;0;46.296;0;166.0;1:37:38.942;13:23:07.272;0:27.376;0:44.972;0:46.296;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;47;1:59.117;0;;27.988;0;44.780;0;46.349;0;165.4;1:39:38.059;13:25:06.389;0:27.988;0:44.780;0:46.349;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;48;1:58.964;0;;27.630;0;45.190;0;46.144;0;165.6;1:41:37.023;13:27:05.353;0:27.630;0:45.190;0:46.144;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;49;1:59.295;0;;28.099;0;44.941;0;46.255;0;165.1;1:43:36.318;13:29:04.648;0:28.099;0:44.941;0:46.255;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;50;1:58.915;0;;27.762;0;44.714;0;46.439;0;165.7;1:45:35.233;13:31:03.563;0:27.762;0:44.714;0:46.439;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;51;2:13.447;0;;32.091;0;51.394;0;49.962;0;147.6;1:47:48.680;13:33:17.010;0:32.091;0:51.394;0:49.962;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;FCY;
5;1;52;2:55.116;0;;30.427;0;1:29.114;0;55.575;0;112.5;1:50:43.796;13:36:12.126;0:30.427;1:29.114;0:55.575;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;53;2:00.945;0;;27.603;0;46.103;0;47.239;0;162.9;1:52:44.741;13:38:13.071;0:27.603;0:46.103;0:47.239;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;54;1:59.501;0;;27.563;0;45.513;0;46.425;0;164.8;1:54:44.242;13:40:12.572;0:27.563;0:45.513;0:46.425;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;GF;
5;1;55;2:04.785;0;;28.768;0;45.612;0;50.405;0;157.9;1:56:49.027;13:42:17.357;0:28.768;0:45.612;0:50.405;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;FCY;
5;1;56;3:23.593;0;;47.778;0;1:17.879;0;1:17.936;0;96.8;2:00:12.620;13:45:40.950;0:47.778;1:17.879;1:17.936;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;FCY;
5;1;57;3:27.918;0;;54.891;0;1:13.924;0;1:19.103;0;94.7;2:03:40.538;13:49:08.868;0:54.891;1:13.924;1:19.103;;Tim Lewis;;TCR;;KMW Motorsports with TMR Engineering;Honda;FF;
52;1;1;2:11.825;0;;38.440;0;46.074;0;47.311;0;149.4;2:11.825;11:47:40.155;0:38.440;0:46.074;0:47.311;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;2;1:59.617;0;;28.065;0;45.416;0;46.136;0;164.7;4:11.442;11:49:39.772;0:28.065;0:45.416;0:46.136;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;3;2:29.686;0;;30.751;0;1:01.116;0;57.819;0;131.6;6:41.128;11:52:09.458;0:30.751;1:01.116;0:57.819;;Sam Baker;;TCR;;Baker Racing;Audi;FCY;
52;1;4;3:01.234;0;;43.249;0;1:17.025;0;1:00.960;0;108.7;9:42.362;11:55:10.692;0:43.249;1:17.025;1:00.960;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;5;2:01.081;0;;28.496;0;46.034;0;46.551;0;162.7;11:43.443;11:57:11.773;0:28.496;0:46.034;0:46.551;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;6;1:59.636;0;;28.784;0;44.937;0;45.915;0;164.7;13:43.079;11:59:11.409;0:28.784;0:44.937;0:45.915;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;7;1:57.939;0;;27.610;0;44.595;0;45.734;1;167.0;15:41.018;12:01:09.348;0:27.610;0:44.595;0:45.734;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;8;1:58.286;0;;27.274;0;44.797;0;46.215;0;166.5;17:39.304;12:03:07.634;0:27.274;0:44.797;0:46.215;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;9;1:58.127;0;;27.265;1;44.945;0;45.917;0;166.8;19:37.431;12:05:05.761;0:27.265;0:44.945;0:45.917;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;10;1:57.834;0;;27.270;0;44.423;0;46.141;0;167.2;21:35.265;12:07:03.595;0:27.270;0:44.423;0:46.141;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;11;1:59.602;0;;28.040;0;45.333;0;46.229;0;164.7;23:34.867;12:09:03.197;0:28.040;0:45.333;0:46.229;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;12;1:59.093;0;;27.361;0;45.020;0;46.712;0;165.4;25:33.960;12:11:02.290;0:27.361;0:45.020;0:46.712;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;13;1:58.627;0;;27.386;0;45.114;0;46.127;0;166.1;27:32.587;12:13:00.917;0:27.386;0:45.114;0:46.127;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;14;1:58.477;0;;27.802;0;44.569;0;46.106;0;166.3;29:31.064;12:14:59.394;0:27.802;0:44.569;0:46.106;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;15;1:58.298;0;;27.363;0;44.814;0;46.121;0;166.5;31:29.362;12:16:57.692;0:27.363;0:44.814;0:46.121;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;16;1:58.511;0;;27.319;0;44.702;0;46.490;0;166.2;33:27.873;12:18:56.203;0:27.319;0:44.702;0:46.490;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;17;1:58.910;0;;27.354;0;44.896;0;46.660;0;165.7;35:26.783;12:20:55.113;0:27.354;0:44.896;0:46.660;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;18;1:59.578;0;;27.275;0;45.368;0;46.935;0;164.7;37:26.361;12:22:54.691;0:27.275;0:45.368;0:46.935;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;19;1:58.977;0;;27.431;0;45.085;0;46.461;0;165.6;39:25.338;12:24:53.668;0:27.431;0:45.085;0:46.461;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;20;2:00.345;0;;27.481;0;45.652;0;47.212;0;163.7;41:25.683;12:26:54.013;0:27.481;0:45.652;0:47.212;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;21;2:01.177;0;;27.325;0;46.482;0;47.370;0;162.6;43:26.860;12:28:55.190;0:27.325;0:46.482;0:47.370;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;22;1:58.559;0;;27.743;0;44.738;0;46.078;0;166.1;45:25.419;12:30:53.749;0:27.743;0:44.738;0:46.078;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;23;1:57.977;0;;27.412;0;44.382;1;46.183;0;167.0;47:23.396;12:32:51.726;0:27.412;0:44.382;0:46.183;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;24;1:58.006;0;;27.453;0;44.419;0;46.134;0;166.9;49:21.402;12:34:49.732;0:27.453;0:44.419;0:46.134;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;25;1:58.776;0;;27.633;0;44.659;0;46.484;0;165.8;51:20.178;12:36:48.508;0:27.633;0:44.659;0:46.484;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;26;1:58.634;0;;27.514;0;44.499;0;46.621;0;166.0;53:18.812;12:38:47.142;0:27.514;0:44.499;0:46.621;;Sam Baker;;TCR;;Baker Racing;Audi;GF;
52;1;27;2:00.079;0;;27.352;0;44.658;0;48.069;0;164.0;55:18.891;12:40:47.221;0:27.352;0:44.658;0:48.069;;Sam Baker;;TCR;;Baker Racing;Audi;FCY;
52;1;28;3:15.617;0;;33.474;0;1:17.398;0;1:24.745;0;100.7;58:34.508;12:44:02.838;0:33.474;1:17.398;1:24.745;;Sam Baker;;TCR;;Baker Racing;Audi;FCY;
52;1;29;4:08.823;0;B;44.465;0;1:10.868;0;2:13.490;0;79.2;1:02:43.331;12:48:11.661;0:44.465;1:10.868;2:13.490;;Sam Baker;;TCR;;Baker Racing;Audi;FCY;
52;2;30;2:31.844;0;;44.341;0;53.285;0;54.218;0;129.7;1:05:15.175;12:50:43.505;0:44.341;0:53.285;0:54.218;;James Vance;0:01:25.454;TCR;;Baker Racing;Audi;FCY;
52;2;31;2:40.279;0;;37.078;0;1:10.236;0;52.965;0;122.9;1:07:55.454;12:53:23.784;0:37.078;1:10.236;0:52.965;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;32;1:57.622;0;;27.739;0;44.341;0;45.542;2;167.5;1:09:53.076;12:55:21.406;0:27.739;0:44.341;0:45.542;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;33;1:56.553;2;;27.046;2;43.946;2;45.561;0;169.0;1:11:49.629;12:57:17.959;0:27.046;0:43.946;0:45.561;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;34;1:58.997;0;;27.276;0;45.526;0;46.195;0;165.5;1:13:48.626;12:59:16.956;0:27.276;0:45.526;0:46.195;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;35;1:57.191;0;;27.166;0;44.255;0;45.770;0;168.1;1:15:45.817;13:01:14.147;0:27.166;0:44.255;0:45.770;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;36;1:57.786;0;;27.291;0;44.457;0;46.038;0;167.2;1:17:43.603;13:03:11.933;0:27.291;0:44.457;0:46.038;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;37;1:58.067;0;;27.399;0;44.407;0;46.261;0;166.8;1:19:41.670;13:05:10.000;0:27.399;0:44.407;0:46.261;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;38;1:59.934;0;;27.730;0;44.937;0;47.267;0;164.2;1:21:41.604;13:07:09.934;0:27.730;0:44.937;0:47.267;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;39;1:58.931;0;;27.553;0;44.912;0;46.466;0;165.6;1:23:40.535;13:09:08.865;0:27.553;0:44.912;0:46.466;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;40;1:59.382;0;;27.539;0;44.983;0;46.860;0;165.0;1:25:39.917;13:11:08.247;0:27.539;0:44.983;0:46.860;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;41;1:59.457;0;;27.715;0;44.975;0;46.767;0;164.9;1:27:39.374;13:13:07.704;0:27.715;0:44.975;0:46.767;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;42;5:19.151;0;B;28.421;0;46.272;0;4:04.458;0;61.7;1:32:58.525;13:18:26.855;0:28.421;0:46.272;4:04.458;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;43;2:12.774;0;;39.665;0;46.101;0;47.008;0;148.4;1:35:11.299;13:20:39.629;0:39.665;0:46.101;0:47.008;;James Vance;0:03:34.772;TCR;;Baker Racing;Audi;GF;
52;2;44;2:00.416;0;;28.086;0;45.115;0;47.215;0;163.6;1:37:11.715;13:22:40.045;0:28.086;0:45.115;0:47.215;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;45;2:00.134;0;;27.512;0;44.769;0;47.853;0;164.0;1:39:11.849;13:24:40.179;0:27.512;0:44.769;0:47.853;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;46;2:00.040;0;;28.380;0;45.117;0;46.543;0;164.1;1:41:11.889;13:26:40.219;0:28.380;0:45.117;0:46.543;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;47;1:59.230;0;;27.486;0;45.201;0;46.543;0;165.2;1:43:11.119;13:28:39.449;0:27.486;0:45.201;0:46.543;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;48;1:59.151;0;;27.619;0;45.055;0;46.477;0;165.3;1:45:10.270;13:30:38.600;0:27.619;0:45.055;0:46.477;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;49;2:06.258;0;;27.580;0;46.024;0;52.654;0;156.0;1:47:16.528;13:32:44.858;0:27.580;0:46.024;0:52.654;;James Vance;;TCR;;Baker Racing;Audi;FCY;
52;2;50;2:20.080;0;;45.375;0;47.315;0;47.390;0;140.6;1:49:36.608;13:35:04.938;0:45.375;0:47.315;0:47.390;;James Vance;;TCR;;Baker Racing;Audi;FCY;
52;2;51;1:59.622;0;;27.737;0;45.379;0;46.506;0;164.7;1:51:36.230;13:37:04.560;0:27.737;0:45.379;0:46.506;;James Vance;;TCR;;Baker Racing;Audi;GF;
52;2;52;2:00.017;0;;27.595;0;45.820;0;46.602;0;164.1;1:53:36.247;13:39:04.577;0:27.595;0:45.820;0:46.602;;James Vance;;TCR;;Baker Racing;Audi;GF;
54;1;1;2:00.516;0;;29.326;0;45.166;0;46.024;0;163.5;2:00.516;11:47:28.846;0:29.326;0:45.166;0:46.024;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;2;1:57.581;0;;27.264;0;44.192;0;46.125;0;167.5;3:58.097;11:49:26.427;0:27.264;0:44.192;0:46.125;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;3;2:27.738;0;;30.025;0;59.315;0;58.398;0;133.3;6:25.835;11:51:54.165;0:30.025;0:59.315;0:58.398;;Caio Chaves;;GS;;Panam Motorsport;Toyota;FCY;
54;1;4;3:08.679;0;;45.275;0;1:17.785;0;1:05.619;0;104.4;9:34.514;11:55:02.844;0:45.275;1:17.785;1:05.619;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;5;1:58.139;0;;27.069;0;45.327;0;45.743;0;166.7;11:32.653;11:57:00.983;0:27.069;0:45.327;0:45.743;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;6;1:57.152;0;;26.867;2;44.524;0;45.761;0;168.1;13:29.805;11:58:58.135;0:26.867;0:44.524;0:45.761;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;7;1:56.663;0;;26.943;0;44.127;0;45.593;0;168.8;15:26.468;12:00:54.798;0:26.943;0:44.127;0:45.593;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;8;1:56.862;0;;26.928;0;44.265;0;45.669;0;168.6;17:23.330;12:02:51.660;0:26.928;0:44.265;0:45.669;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;9;1:56.653;0;;26.907;0;44.062;0;45.684;0;168.9;19:19.983;12:04:48.313;0:26.907;0:44.062;0:45.684;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;10;1:56.718;0;;27.039;0;44.036;0;45.643;0;168.8;21:16.701;12:06:45.031;0:27.039;0:44.036;0:45.643;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;11;1:56.974;0;;27.046;0;44.237;0;45.691;0;168.4;23:13.675;12:08:42.005;0:27.046;0:44.237;0:45.691;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;12;1:56.629;0;;26.935;0;44.033;0;45.661;0;168.9;25:10.304;12:10:38.634;0:26.935;0:44.033;0:45.661;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;13;1:56.745;0;;27.020;0;44.096;0;45.629;0;168.7;27:07.049;12:12:35.379;0:27.020;0:44.096;0:45.629;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;14;1:57.100;0;;27.347;0;44.090;0;45.663;0;168.2;29:04.149;12:14:32.479;0:27.347;0:44.090;0:45.663;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;15;1:57.219;0;;27.321;0;44.173;0;45.725;0;168.0;31:01.368;12:16:29.698;0:27.321;0:44.173;0:45.725;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;16;1:56.611;0;;26.875;0;44.031;0;45.705;0;168.9;32:57.979;12:18:26.309;0:26.875;0:44.031;0:45.705;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;17;2:09.722;0;B;26.957;0;43.943;0;58.822;0;151.9;35:07.701;12:20:36.031;0:26.957;0:43.943;0:58.822;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;18;2:39.504;0;;1:09.249;0;44.307;0;45.948;0;123.5;37:47.205;12:23:15.535;1:09.249;0:44.307;0:45.948;;Caio Chaves;0:01:01.234;GS;;Panam Motorsport;Toyota;GF;
54;1;19;1:57.540;0;;27.000;0;44.105;0;46.435;0;167.6;39:44.745;12:25:13.075;0:27.000;0:44.105;0:46.435;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;20;1:57.173;0;;27.218;0;44.131;0;45.824;0;168.1;41:41.918;12:27:10.248;0:27.218;0:44.131;0:45.824;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;21;1:56.817;0;;26.984;0;44.054;0;45.779;0;168.6;43:38.735;12:29:07.065;0:26.984;0:44.054;0:45.779;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;22;1:57.790;0;;27.136;0;44.392;0;46.262;0;167.2;45:36.525;12:31:04.855;0:27.136;0:44.392;0:46.262;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;23;1:58.411;0;;27.727;0;44.938;0;45.746;0;166.4;47:34.936;12:33:03.266;0:27.727;0:44.938;0:45.746;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;24;1:57.643;0;;27.193;0;44.665;0;45.785;0;167.4;49:32.579;12:35:00.909;0:27.193;0:44.665;0:45.785;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;25;1:56.928;0;;27.061;0;43.888;0;45.979;0;168.5;51:29.507;12:36:57.837;0:27.061;0:43.888;0:45.979;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;26;1:56.851;0;;27.177;0;43.925;0;45.749;0;168.6;53:26.358;12:38:54.688;0:27.177;0:43.925;0:45.749;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;27;2:01.174;0;;26.953;0;44.694;0;49.527;0;162.6;55:27.532;12:40:55.862;0:26.953;0:44.694;0:49.527;;Caio Chaves;;GS;;Panam Motorsport;Toyota;FCY;
54;1;28;3:08.785;0;;28.990;0;1:15.556;0;1:24.239;0;104.3;58:36.317;12:44:04.647;0:28.990;1:15.556;1:24.239;;Caio Chaves;;GS;;Panam Motorsport;Toyota;FCY;
54;1;29;3:00.268;0;;46.228;0;1:09.984;0;1:04.056;0;109.3;1:01:36.585;12:47:04.915;0:46.228;1:09.984;1:04.056;;Caio Chaves;;GS;;Panam Motorsport;Toyota;FCY;
54;1;30;2:59.388;0;;33.342;0;1:09.915;0;1:16.131;0;109.8;1:04:35.973;12:50:04.303;0:33.342;1:09.915;1:16.131;;Caio Chaves;;GS;;Panam Motorsport;Toyota;FCY;
54;1;31;3:06.207;0;;51.930;0;1:13.924;0;1:00.353;0;105.8;1:07:42.180;12:53:10.510;0:51.930;1:13.924;1:00.353;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;32;1:58.112;0;;27.468;0;44.850;0;45.794;0;166.8;1:09:40.292;12:55:08.622;0:27.468;0:44.850;0:45.794;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;33;1:58.408;0;;26.948;0;45.557;0;45.903;0;166.4;1:11:38.700;12:57:07.030;0:26.948;0:45.557;0:45.903;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;34;1:56.355;0;;26.944;0;43.892;0;45.519;1;169.3;1:13:35.055;12:59:03.385;0:26.944;0:43.892;0:45.519;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;1;35;2:09.770;0;B;26.874;0;43.876;1;59.020;0;151.8;1:15:44.825;13:01:13.155;0:26.874;0:43.876;0:59.020;;Caio Chaves;;GS;;Panam Motorsport;Toyota;GF;
54;2;36;3:12.157;0;;1:40.290;0;45.517;0;46.350;0;102.5;1:18:56.982;13:04:25.312;1:40.290;0:45.517;0:46.350;;Werner Neugebauer;0:01:31.198;GS;;Panam Motorsport;Toyota;GF;
54;2;37;1:56.839;0;;27.181;0;44.075;0;45.583;0;168.6;1:20:53.821;13:06:22.151;0:27.181;0:44.075;0:45.583;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;38;1:56.350;0;;26.908;1;43.744;0;45.698;0;169.3;1:22:50.171;13:08:18.501;0:26.908;0:43.744;0:45.698;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;39;1:56.496;0;;27.045;0;43.940;0;45.511;0;169.1;1:24:46.667;13:10:14.997;0:27.045;0:43.940;0:45.511;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;40;1:56.254;2;;27.009;0;43.802;0;45.443;2;169.4;1:26:42.921;13:12:11.251;0:27.009;0:43.802;0:45.443;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;41;1:56.569;0;;26.979;0;43.894;0;45.696;0;169.0;1:28:39.490;13:14:07.820;0:26.979;0:43.894;0:45.696;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;42;1:56.514;0;;27.111;0;43.895;0;45.508;0;169.1;1:30:36.004;13:16:04.334;0:27.111;0:43.895;0:45.508;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;43;1:56.392;0;;27.067;0;43.843;0;45.482;0;169.2;1:32:32.396;13:18:00.726;0:27.067;0:43.843;0:45.482;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;44;1:56.784;0;;27.110;0;44.019;0;45.655;0;168.7;1:34:29.180;13:19:57.510;0:27.110;0:44.019;0:45.655;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;45;1:57.164;0;;27.065;0;44.454;0;45.645;0;168.1;1:36:26.344;13:21:54.674;0:27.065;0:44.454;0:45.645;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;46;1:56.948;0;;27.030;0;44.379;0;45.539;0;168.4;1:38:23.292;13:23:51.622;0:27.030;0:44.379;0:45.539;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;47;1:56.662;0;;27.018;0;43.893;0;45.751;0;168.8;1:40:19.954;13:25:48.284;0:27.018;0:43.893;0:45.751;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;48;1:56.577;0;;26.995;0;43.891;0;45.691;0;169.0;1:42:16.531;13:27:44.861;0:26.995;0:43.891;0:45.691;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;49;1:56.345;0;;27.015;0;43.679;2;45.651;0;169.3;1:44:12.876;13:29:41.206;0:27.015;0:43.679;0:45.651;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;50;1:57.220;0;;27.049;0;44.259;0;45.912;0;168.0;1:46:10.096;13:31:38.426;0:27.049;0:44.259;0:45.912;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;FCY;
54;2;51;2:12.096;0;;29.774;0;52.102;0;50.220;0;149.1;1:48:22.192;13:33:50.522;0:29.774;0:52.102;0:50.220;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;FCY;
54;2;52;2:17.439;0;;27.445;0;53.379;0;56.615;0;143.3;1:50:39.631;13:36:07.961;0:27.445;0:53.379;0:56.615;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;53;2:00.503;0;;27.564;0;46.176;0;46.763;0;163.5;1:52:40.134;13:38:08.464;0:27.564;0:46.176;0:46.763;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;54;1:57.808;0;;27.159;0;44.819;0;45.830;0;167.2;1:54:37.942;13:40:06.272;0:27.159;0:44.819;0:45.830;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;GF;
54;2;55;2:02.078;0;;27.332;0;44.236;0;50.510;0;161.4;1:56:40.020;13:42:08.350;0:27.332;0:44.236;0:50.510;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;FCY;
54;2;56;3:24.893;0;;48.775;0;1:19.034;0;1:17.084;0;96.1;2:00:04.913;13:45:33.243;0:48.775;1:19.034;1:17.084;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;FCY;
54;2;57;3:26.562;0;;53.441;0;1:15.626;0;1:17.495;0;95.4;2:03:31.475;13:48:59.805;0:53.441;1:15.626;1:17.495;;Werner Neugebauer;;GS;;Panam Motorsport;Toyota;FF;
55;2;1;2:14.367;0;;40.260;0;45.731;0;48.376;0;146.6;2:14.367;11:47:42.697;0:40.260;0:45.731;0:48.376;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;2;2:02.058;0;;28.333;0;46.434;0;47.291;0;161.4;4:16.425;11:49:44.755;0:28.333;0:46.434;0:47.291;;Eduardo Gou;;TCR;;Gou Racing;Cupra;FCY;
55;2;3;2:30.746;0;;31.309;0;1:01.388;0;58.049;0;130.7;6:47.171;11:52:15.501;0:31.309;1:01.388;0:58.049;;Eduardo Gou;;TCR;;Gou Racing;Cupra;FCY;
55;2;4;2:56.666;0;;42.484;0;1:15.014;0;59.168;0;111.5;9:43.837;11:55:12.167;0:42.484;1:15.014;0:59.168;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;5;2:01.359;0;;28.087;0;46.713;0;46.559;0;162.3;11:45.196;11:57:13.526;0:28.087;0:46.713;0:46.559;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;6;2:02.137;0;;28.809;0;46.991;0;46.337;0;161.3;13:47.333;11:59:15.663;0:28.809;0:46.991;0:46.337;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;7;1:59.880;0;;27.641;0;45.814;0;46.425;0;164.3;15:47.213;12:01:15.543;0:27.641;0:45.814;0:46.425;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;8;1:59.751;0;;27.830;0;45.491;0;46.430;0;164.5;17:46.964;12:03:15.294;0:27.830;0:45.491;0:46.430;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;9;2:01.052;0;;28.133;0;45.554;0;47.365;0;162.7;19:48.016;12:05:16.346;0:28.133;0:45.554;0:47.365;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;10;2:01.158;0;;28.230;0;45.687;0;47.241;0;162.6;21:49.174;12:07:17.504;0:28.230;0:45.687;0:47.241;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;11;1:59.492;0;;27.676;0;45.422;0;46.394;0;164.9;23:48.666;12:09:16.996;0:27.676;0:45.422;0:46.394;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;12;1:59.451;0;;28.062;0;44.898;0;46.491;0;164.9;25:48.117;12:11:16.447;0:28.062;0:44.898;0:46.491;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;13;1:59.338;0;;27.574;0;45.441;0;46.323;0;165.1;27:47.455;12:13:15.785;0:27.574;0:45.441;0:46.323;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;14;1:58.566;0;;27.414;1;45.084;0;46.068;1;166.1;29:46.021;12:15:14.351;0:27.414;0:45.084;0:46.068;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;15;1:58.952;0;;27.631;0;44.907;0;46.414;0;165.6;31:44.973;12:17:13.303;0:27.631;0:44.907;0:46.414;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;16;1:58.526;0;;27.420;0;44.575;1;46.531;0;166.2;33:43.499;12:19:11.829;0:27.420;0:44.575;0:46.531;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;17;1:59.011;0;;28.066;0;44.787;0;46.158;0;165.5;35:42.510;12:21:10.840;0:28.066;0:44.787;0:46.158;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;18;1:58.895;0;;27.443;0;44.884;0;46.568;0;165.7;37:41.405;12:23:09.735;0:27.443;0:44.884;0:46.568;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;19;1:58.488;0;;27.481;0;44.799;0;46.208;0;166.2;39:39.893;12:25:08.223;0:27.481;0:44.799;0:46.208;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;20;1:59.029;0;;27.583;0;45.178;0;46.268;0;165.5;41:38.922;12:27:07.252;0:27.583;0:45.178;0:46.268;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;21;1:59.397;0;;27.698;0;45.190;0;46.509;0;165.0;43:38.319;12:29:06.649;0:27.698;0:45.190;0:46.509;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;22;1:59.363;0;;27.528;0;45.111;0;46.724;0;165.0;45:37.682;12:31:06.012;0:27.528;0:45.111;0:46.724;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;23;2:00.712;0;;28.246;0;45.308;0;47.158;0;163.2;47:38.394;12:33:06.724;0:28.246;0:45.308;0:47.158;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;24;2:00.623;0;;28.195;0;45.590;0;46.838;0;163.3;49:39.017;12:35:07.347;0:28.195;0:45.590;0:46.838;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;25;2:00.948;0;;28.437;0;45.883;0;46.628;0;162.9;51:39.965;12:37:08.295;0:28.437;0:45.883;0:46.628;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;26;2:02.673;0;;28.238;0;45.510;0;48.925;0;160.6;53:42.638;12:39:10.968;0:28.238;0:45.510;0:48.925;;Eduardo Gou;;TCR;;Gou Racing;Cupra;GF;
55;2;27;2:18.251;0;;28.062;0;50.767;0;59.422;0;142.5;56:00.889;12:41:29.219;0:28.062;0:50.767;0:59.422;;Eduardo Gou;;TCR;;Gou Racing;Cupra;FCY;
55;2;28;2:39.431;0;;35.432;0;50.752;0;1:13.247;0;123.6;58:40.320;12:44:08.650;0:35.432;0:50.752;1:13.247;;Eduardo Gou;;TCR;;Gou Racing;Cupra;FCY;
55;2;29;4:06.689;0;B;46.633;0;1:08.922;0;2:11.134;0;79.9;1:02:47.009;12:48:15.339;0:46.633;1:08.922;2:11.134;;Eduardo Gou;;TCR;;Gou Racing;Cupra;FCY;
55;1;30;2:30.827;0;;43.510;0;53.896;0;53.421;0;130.6;1:05:17.836;12:50:46.166;0:43.510;0:53.896;0:53.421;;Eddie Gou;0:01:24.473;TCR;;Gou Racing;Cupra;FCY;
55;1;31;2:39.144;0;;37.383;0;1:09.593;0;52.168;0;123.8;1:07:56.980;12:53:25.310;0:37.383;1:09.593;0:52.168;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;32;1:58.341;0;;27.714;0;45.097;0;45.530;2;166.5;1:09:55.321;12:55:23.651;0:27.714;0:45.097;0:45.530;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;33;1:57.500;0;;27.092;0;44.729;0;45.679;0;167.6;1:11:52.821;12:57:21.151;0:27.092;0:44.729;0:45.679;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;34;1:58.680;0;;27.278;0;44.609;0;46.793;0;166.0;1:13:51.501;12:59:19.831;0:27.278;0:44.609;0:46.793;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;35;1:57.909;0;;27.006;2;45.019;0;45.884;0;167.1;1:15:49.410;13:01:17.740;0:27.006;0:45.019;0:45.884;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;36;1:57.807;0;;27.211;0;44.386;0;46.210;0;167.2;1:17:47.217;13:03:15.547;0:27.211;0:44.386;0:46.210;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;37;1:57.483;2;;27.248;0;44.273;2;45.962;0;167.7;1:19:44.700;13:05:13.030;0:27.248;0:44.273;0:45.962;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;38;1:58.235;0;;27.161;0;45.237;0;45.837;0;166.6;1:21:42.935;13:07:11.265;0:27.161;0:45.237;0:45.837;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;39;1:58.448;0;;27.351;0;45.141;0;45.956;0;166.3;1:23:41.383;13:09:09.713;0:27.351;0:45.141;0:45.956;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;40;1:58.681;0;;27.452;0;44.761;0;46.468;0;166.0;1:25:40.064;13:11:08.394;0:27.452;0:44.761;0:46.468;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;41;1:59.605;0;;27.767;0;45.018;0;46.820;0;164.7;1:27:39.669;13:13:07.999;0:27.767;0:45.018;0:46.820;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;42;1:59.612;0;;28.160;0;45.412;0;46.040;0;164.7;1:29:39.281;13:15:07.611;0:28.160;0:45.412;0:46.040;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;43;1:58.308;0;;27.477;0;44.701;0;46.130;0;166.5;1:31:37.589;13:17:05.919;0:27.477;0:44.701;0:46.130;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;44;1:59.145;0;;27.760;0;45.066;0;46.319;0;165.3;1:33:36.734;13:19:05.064;0:27.760;0:45.066;0:46.319;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;45;1:58.466;0;;27.492;0;44.814;0;46.160;0;166.3;1:35:35.200;13:21:03.530;0:27.492;0:44.814;0:46.160;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;46;1:58.913;0;;27.600;0;45.052;0;46.261;0;165.7;1:37:34.113;13:23:02.443;0:27.600;0:45.052;0:46.261;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;47;1:58.749;0;;27.481;0;44.924;0;46.344;0;165.9;1:39:32.862;13:25:01.192;0:27.481;0:44.924;0:46.344;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;48;1:58.641;0;;27.507;0;44.980;0;46.154;0;166.0;1:41:31.503;13:26:59.833;0:27.507;0:44.980;0:46.154;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;49;1:58.609;0;;27.516;0;44.860;0;46.233;0;166.1;1:43:30.112;13:28:58.442;0:27.516;0:44.860;0:46.233;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;50;1:58.631;0;;27.444;0;44.944;0;46.243;0;166.0;1:45:28.743;13:30:57.073;0:27.444;0:44.944;0:46.243;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;51;2:03.770;0;;27.693;0;47.724;0;48.353;0;159.2;1:47:32.513;13:33:00.843;0:27.693;0:47.724;0:48.353;;Eddie Gou;;TCR;;Gou Racing;Cupra;FCY;
55;1;52;3:10.118;0;;39.270;0;1:34.271;0;56.577;0;103.6;1:50:42.631;13:36:10.961;0:39.270;1:34.271;0:56.577;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;53;1:59.905;0;;27.509;0;46.366;0;46.030;0;164.3;1:52:42.536;13:38:10.866;0:27.509;0:46.366;0:46.030;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;54;1:58.745;0;;27.491;0;44.984;0;46.270;0;165.9;1:54:41.281;13:40:09.611;0:27.491;0:44.984;0:46.270;;Eddie Gou;;TCR;;Gou Racing;Cupra;GF;
55;1;55;2:01.502;0;;27.487;0;45.120;0;48.895;0;162.1;1:56:42.783;13:42:11.113;0:27.487;0:45.120;0:48.895;;Eddie Gou;;TCR;;Gou Racing;Cupra;FCY;
55;1;56;3:24.978;0;;48.602;0;1:19.718;0;1:16.658;0;96.1;2:00:07.761;13:45:36.091;0:48.602;1:19.718;1:16.658;;Eddie Gou;;TCR;;Gou Racing;Cupra;FCY;
55;1;57;3:26.976;0;;53.902;0;1:15.351;0;1:17.723;0;95.2;2:03:34.737;13:49:03.067;0:53.902;1:15.351;1:17.723;;Eddie Gou;;TCR;;Gou Racing;Cupra;FF;
56;1;1;7:40.684;0;B;45.070;0;53.294;0;6:02.320;0;42.8;7:40.684;11:53:09.014;0:45.070;0:53.294;6:02.320;;Dean Baker;0:00:18.484;TCR;;Baker Racing;Audi;FCY;
56;1;2;2:17.005;0;;41.677;0;48.179;0;47.149;0;143.8;9:57.689;11:55:26.019;0:41.677;0:48.179;0:47.149;;Dean Baker;0:05:17.320;TCR;;Baker Racing;Audi;GF;
56;1;3;2:00.339;0;;27.906;0;45.732;0;46.701;0;163.7;11:58.028;11:57:26.358;0:27.906;0:45.732;0:46.701;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;4;2:00.101;0;;27.437;0;46.035;0;46.629;0;164.0;13:58.129;11:59:26.459;0:27.437;0:46.035;0:46.629;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;5;1:58.984;2;;27.465;0;45.293;0;46.226;2;165.6;15:57.113;12:01:25.443;0:27.465;0:45.293;0:46.226;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;6;2:00.053;0;;27.404;2;45.381;0;47.268;0;164.1;17:57.166;12:03:25.496;0:27.404;0:45.381;0:47.268;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;7;2:00.403;0;;27.413;0;45.634;0;47.356;0;163.6;19:57.569;12:05:25.899;0:27.413;0:45.634;0:47.356;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;8;1:59.513;0;;27.567;0;45.377;0;46.569;0;164.8;21:57.082;12:07:25.412;0:27.567;0:45.377;0:46.569;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;9;1:59.630;0;;27.468;0;45.526;0;46.636;0;164.7;23:56.712;12:09:25.042;0:27.468;0:45.526;0:46.636;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;10;1:59.788;0;;27.678;0;45.577;0;46.533;0;164.4;25:56.500;12:11:24.830;0:27.678;0:45.577;0:46.533;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;11;1:59.660;0;;27.540;0;45.449;0;46.671;0;164.6;27:56.160;12:13:24.490;0:27.540;0:45.449;0:46.671;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;12;2:01.044;0;;28.186;0;45.787;0;47.071;0;162.7;29:57.204;12:15:25.534;0:28.186;0:45.787;0:47.071;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;13;1:59.487;0;;27.514;0;45.215;2;46.758;0;164.9;31:56.691;12:17:25.021;0:27.514;0:45.215;0:46.758;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;14;1:59.421;0;;27.486;0;45.334;0;46.601;0;164.9;33:56.112;12:19:24.442;0:27.486;0:45.334;0:46.601;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;15;1:59.771;0;;27.644;0;45.365;0;46.762;0;164.5;35:55.883;12:21:24.213;0:27.644;0:45.365;0:46.762;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;16;2:00.399;0;;27.470;0;46.337;0;46.592;0;163.6;37:56.282;12:23:24.612;0:27.470;0:46.337;0:46.592;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;17;2:00.029;0;;27.406;0;45.482;0;47.141;0;164.1;39:56.311;12:25:24.641;0:27.406;0:45.482;0:47.141;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;18;2:00.052;0;;27.757;0;45.382;0;46.913;0;164.1;41:56.363;12:27:24.693;0:27.757;0:45.382;0:46.913;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;19;1:59.852;0;;27.610;0;45.217;0;47.025;0;164.4;43:56.215;12:29:24.545;0:27.610;0:45.217;0:47.025;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;20;1:59.908;0;;27.705;0;45.239;0;46.964;0;164.3;45:56.123;12:31:24.453;0:27.705;0:45.239;0:46.964;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;21;2:04.028;0;;28.629;0;47.919;0;47.480;0;158.8;48:00.151;12:33:28.481;0:28.629;0:47.919;0:47.480;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;1;22;3:13.594;0;B;28.469;0;46.243;0;1:58.882;0;101.8;51:13.745;12:36:42.075;0:28.469;0:46.243;1:58.882;;Dean Baker;;TCR;;Baker Racing;Audi;GF;
56;2;23;8:07.897;0;B;44.559;0;54.322;0;6:29.016;0;40.4;59:21.642;12:44:49.972;0:44.559;0:54.322;6:29.016;;Bruno Junqueira;0:01:26.365;TCR;;Baker Racing;Audi;FCY;
56;2;24;2:39.153;0;;46.264;0;50.564;0;1:02.325;0;123.8;1:02:00.795;12:47:29.125;0:46.264;0:50.564;1:02.325;;Bruno Junqueira;0:05:51.707;TCR;;Baker Racing;Audi;FCY;
56;2;25;2:42.431;0;;34.915;0;57.289;0;1:10.227;0;121.3;1:04:43.226;12:50:11.556;0:34.915;0:57.289;1:10.227;;Bruno Junqueira;;TCR;;Baker Racing;Audi;FCY;
56;2;26;2:22.114;0;;42.126;0;50.304;0;49.684;0;138.6;1:07:05.340;12:52:33.670;0:42.126;0:50.304;0:49.684;;Bruno Junqueira;;TCR;;Baker Racing;Audi;FCY;
56;2;27;2:20.126;0;B;28.853;0;47.854;0;1:03.419;0;140.6;1:09:25.466;12:54:53.796;0:28.853;0:47.854;1:03.419;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;28;2:21.410;0;;40.049;0;53.274;0;48.087;0;139.3;1:11:46.876;12:57:15.206;0:40.049;0:53.274;0:48.087;;Bruno Junqueira;0:00:29.431;TCR;;Baker Racing;Audi;GF;
56;2;29;2:08.343;0;;27.752;0;45.919;0;54.672;0;153.5;1:13:55.219;12:59:23.549;0:27.752;0:45.919;0:54.672;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;30;2:06.903;0;;28.441;0;48.205;0;50.257;0;155.2;1:16:02.122;13:01:30.452;0:28.441;0:48.205;0:50.257;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;31;2:00.575;0;;27.575;0;45.379;1;47.621;0;163.4;1:18:02.697;13:03:31.027;0:27.575;0:45.379;0:47.621;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;32;2:02.499;0;;28.919;0;46.170;0;47.410;0;160.8;1:20:05.196;13:05:33.526;0:28.919;0:46.170;0:47.410;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;33;2:00.019;0;;27.788;0;45.718;0;46.513;1;164.1;1:22:05.215;13:07:33.545;0:27.788;0:45.718;0:46.513;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;34;2:00.832;0;;27.742;0;45.994;0;47.096;0;163.0;1:24:06.047;13:09:34.377;0:27.742;0:45.994;0:47.096;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;35;2:50.481;0;B;29.075;0;48.728;0;1:32.678;0;115.5;1:26:56.528;13:12:24.858;0:29.075;0:48.728;1:32.678;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;36;2:21.386;0;;40.515;0;50.465;0;50.406;0;139.3;1:29:17.914;13:14:46.244;0:40.515;0:50.465;0:50.406;;Bruno Junqueira;0:00:58.042;TCR;;Baker Racing;Audi;GF;
56;2;37;10:35.464;0;B;28.931;0;46.490;0;9:20.043;0;31.0;1:39:53.378;13:25:21.708;0:28.931;0:46.490;9:20.043;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;38;2:17.644;0;;40.667;0;46.679;0;50.298;0;143.1;1:42:11.022;13:27:39.352;0:40.667;0:46.679;0:50.298;;Bruno Junqueira;0:08:45.186;TCR;;Baker Racing;Audi;GF;
56;2;39;2:00.178;0;;27.625;0;45.622;0;46.931;0;163.9;1:44:11.200;13:29:39.530;0:27.625;0:45.622;0:46.931;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;40;2:01.375;0;;27.477;1;46.284;0;47.614;0;162.3;1:46:12.575;13:31:40.905;0:27.477;0:46.284;0:47.614;;Bruno Junqueira;;TCR;;Baker Racing;Audi;FCY;
56;2;41;2:15.654;0;;30.163;0;54.740;0;50.751;0;145.2;1:48:28.229;13:33:56.559;0:30.163;0:54.740;0:50.751;;Bruno Junqueira;;TCR;;Baker Racing;Audi;FCY;
56;2;42;2:20.388;0;;32.183;0;54.234;0;53.971;0;140.3;1:50:48.617;13:36:16.947;0:32.183;0:54.234;0:53.971;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
56;2;43;2:25.368;0;B;31.006;0;49.412;0;1:04.950;0;135.5;1:53:13.985;13:38:42.315;0:31.006;0:49.412;1:04.950;;Bruno Junqueira;;TCR;;Baker Racing;Audi;GF;
57;1;1;2:02.171;0;;30.119;0;45.702;0;46.350;0;161.2;2:02.171;11:47:30.501;0:30.119;0:45.702;0:46.350;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;2;1:57.786;0;;27.160;0;44.667;0;45.959;0;167.2;3:59.957;11:49:28.287;0:27.160;0:44.667;0:45.959;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;3;2:28.804;0;;30.407;0;1:00.590;0;57.807;0;132.4;6:28.761;11:51:57.091;0:30.407;1:00.590;0:57.807;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;FCY;
57;1;4;3:06.699;0;;44.385;0;1:17.499;0;1:04.815;0;105.5;9:35.460;11:55:03.790;0:44.385;1:17.499;1:04.815;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;5;1:58.832;0;;27.563;0;45.036;0;46.233;0;165.8;11:34.292;11:57:02.622;0:27.563;0:45.036;0:46.233;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;6;1:57.675;0;;27.003;1;44.741;0;45.931;0;167.4;13:31.967;11:59:00.297;0:27.003;0:44.741;0:45.931;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;7;1:57.691;0;;27.110;0;44.554;0;46.027;0;167.4;15:29.658;12:00:57.988;0:27.110;0:44.554;0:46.027;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;8;1:57.540;0;;27.070;0;44.617;0;45.853;0;167.6;17:27.198;12:02:55.528;0:27.070;0:44.617;0:45.853;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;9;1:57.348;0;;27.111;0;44.416;1;45.821;1;167.9;19:24.546;12:04:52.876;0:27.111;0:44.416;0:45.821;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;10;1:58.021;0;;27.418;0;44.430;0;46.173;0;166.9;21:22.567;12:06:50.897;0:27.418;0:44.430;0:46.173;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;11;1:57.844;0;;27.192;0;44.698;0;45.954;0;167.2;23:20.411;12:08:48.741;0:27.192;0:44.698;0:45.954;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;12;1:58.024;0;;27.141;0;44.786;0;46.097;0;166.9;25:18.435;12:10:46.765;0:27.141;0:44.786;0:46.097;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;13;1:57.571;0;;27.234;0;44.494;0;45.843;0;167.5;27:16.006;12:12:44.336;0:27.234;0:44.494;0:45.843;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;14;1:58.149;0;;27.175;0;44.821;0;46.153;0;166.7;29:14.155;12:14:42.485;0:27.175;0:44.821;0:46.153;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;15;1:57.763;0;;27.131;0;44.622;0;46.010;0;167.3;31:11.918;12:16:40.248;0:27.131;0:44.622;0:46.010;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;16;1:59.467;0;;27.006;0;45.949;0;46.512;0;164.9;33:11.385;12:18:39.715;0:27.006;0:45.949;0:46.512;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;17;1:58.790;0;;27.054;0;44.918;0;46.818;0;165.8;35:10.175;12:20:38.505;0:27.054;0:44.918;0:46.818;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;18;1:58.426;0;;27.103;0;44.913;0;46.410;0;166.3;37:08.601;12:22:36.931;0:27.103;0:44.913;0:46.410;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;19;1:58.838;0;;27.166;0;45.037;0;46.635;0;165.8;39:07.439;12:24:35.769;0:27.166;0:45.037;0:46.635;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;1;20;2:11.060;0;B;27.139;0;44.736;0;59.185;0;150.3;41:18.499;12:26:46.829;0:27.139;0:44.736;0:59.185;;Bryce Ward;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;21;2:53.857;0;;1:22.578;0;45.039;0;46.240;0;113.3;44:12.356;12:29:40.686;1:22.578;0:45.039;0:46.240;;Daniel Morad;0:01:13.534;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;22;1:56.829;0;;27.003;0;44.332;0;45.494;0;168.6;46:09.185;12:31:37.515;0:27.003;0:44.332;0:45.494;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;23;1:56.516;0;;27.036;0;43.843;0;45.637;0;169.1;48:05.701;12:33:34.031;0:27.036;0:43.843;0:45.637;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;24;1:56.535;0;;27.019;0;43.718;2;45.798;0;169.0;50:02.236;12:35:30.566;0:27.019;0:43.718;0:45.798;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;25;1:56.044;2;;27.032;0;43.720;0;45.292;3;169.7;51:58.280;12:37:26.610;0:27.032;0:43.720;0:45.292;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;26;1:57.279;0;;26.931;0;44.222;0;46.126;0;168.0;53:55.559;12:39:23.889;0:26.931;0:44.222;0:46.126;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;27;2:12.847;0;;26.902;2;47.413;0;58.532;0;148.3;56:08.406;12:41:36.736;0:26.902;0:47.413;0:58.532;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;FCY;
57;2;28;2:45.409;0;B;36.397;0;53.377;0;1:15.635;0;119.1;58:53.815;12:44:22.145;0:36.397;0:53.377;1:15.635;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;FCY;
57;2;29;3:02.692;0;;1:04.235;0;51.912;0;1:06.545;0;107.8;1:01:56.507;12:47:24.837;1:04.235;0:51.912;1:06.545;;Daniel Morad;0:00:51.799;GS;;Winward Racing;Mercedes-AMG;FCY;
57;2;30;2:47.791;0;;34.965;0;58.109;0;1:14.717;0;117.4;1:04:44.298;12:50:12.628;0:34.965;0:58.109;1:14.717;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;FCY;
57;2;31;3:00.730;0;;52.948;0;1:13.561;0;54.221;0;109.0;1:07:45.028;12:53:13.358;0:52.948;1:13.561;0:54.221;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;32;1:57.850;0;;27.010;0;45.275;0;45.565;0;167.1;1:09:42.878;12:55:11.208;0:27.010;0:45.275;0:45.565;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;33;1:57.961;0;;26.966;0;44.803;0;46.192;0;167.0;1:11:40.839;12:57:09.169;0:26.966;0:44.803;0:46.192;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;34;1:56.907;0;;27.046;0;44.161;0;45.700;0;168.5;1:13:37.746;12:59:06.076;0:27.046;0:44.161;0:45.700;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;35;1:56.745;0;;27.052;0;43.988;0;45.705;0;168.7;1:15:34.491;13:01:02.821;0:27.052;0:43.988;0:45.705;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;36;1:56.915;0;;26.989;0;44.170;0;45.756;0;168.5;1:17:31.406;13:02:59.736;0:26.989;0:44.170;0:45.756;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;37;1:56.689;0;;26.925;0;44.091;0;45.673;0;168.8;1:19:28.095;13:04:56.425;0:26.925;0:44.091;0:45.673;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;38;1:56.814;0;;26.921;0;44.123;0;45.770;0;168.6;1:21:24.909;13:06:53.239;0:26.921;0:44.123;0:45.770;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;39;1:56.706;0;;27.028;0;43.821;0;45.857;0;168.8;1:23:21.615;13:08:49.945;0:27.028;0:43.821;0:45.857;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;40;1:56.898;0;;26.959;0;44.242;0;45.697;0;168.5;1:25:18.513;13:10:46.843;0:26.959;0:44.242;0:45.697;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;41;1:56.861;0;;27.118;0;43.926;0;45.817;0;168.6;1:27:15.374;13:12:43.704;0:27.118;0:43.926;0:45.817;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;42;1:56.782;0;;26.914;0;44.145;0;45.723;0;168.7;1:29:12.156;13:14:40.486;0:26.914;0:44.145;0:45.723;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;43;1:57.079;0;;26.959;0;44.202;0;45.918;0;168.2;1:31:09.235;13:16:37.565;0:26.959;0:44.202;0:45.918;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
57;2;44;2:29.961;0;B;27.073;0;49.084;0;1:13.804;0;131.4;1:33:39.196;13:19:07.526;0:27.073;0:49.084;1:13.804;;Daniel Morad;;GS;;Winward Racing;Mercedes-AMG;GF;
59;1;1;2:03.329;0;;30.830;0;46.010;0;46.489;0;159.7;2:03.329;11:47:31.659;0:30.830;0:46.010;0:46.489;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;2;1:58.896;0;;27.241;0;45.191;0;46.464;0;165.7;4:02.225;11:49:30.555;0:27.241;0:45.191;0:46.464;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;3;2:28.871;0;;29.958;0;1:00.107;0;58.806;0;132.3;6:31.096;11:51:59.426;0:29.958;1:00.107;0:58.806;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;FCY;
59;1;4;3:05.313;0;;43.757;0;1:17.066;0;1:04.490;0;106.3;9:36.409;11:55:04.739;0:43.757;1:17.066;1:04.490;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;5;1:58.901;0;;27.219;0;45.333;0;46.349;0;165.7;11:35.310;11:57:03.640;0:27.219;0:45.333;0:46.349;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;6;1:58.293;0;;27.080;0;45.054;0;46.159;0;166.5;13:33.603;11:59:01.933;0:27.080;0:45.054;0:46.159;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;7;1:59.190;0;;27.046;0;45.685;0;46.459;0;165.3;15:32.793;12:01:01.123;0:27.046;0:45.685;0:46.459;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;8;1:57.876;0;;27.108;0;44.841;0;45.927;1;167.1;17:30.669;12:02:58.999;0:27.108;0:44.841;0:45.927;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;9;1:58.162;0;;27.052;0;44.809;0;46.301;0;166.7;19:28.831;12:04:57.161;0:27.052;0:44.809;0:46.301;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;10;1:58.169;0;;27.112;0;45.015;0;46.042;0;166.7;21:27.000;12:06:55.330;0:27.112;0:45.015;0:46.042;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;11;1:58.148;0;;27.090;0;44.996;0;46.062;0;166.7;23:25.148;12:08:53.478;0:27.090;0:44.996;0:46.062;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;12;1:57.767;0;;27.064;0;44.647;1;46.056;0;167.3;25:22.915;12:10:51.245;0:27.064;0:44.647;0:46.056;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;13;1:58.544;0;;27.123;0;45.320;0;46.101;0;166.2;27:21.459;12:12:49.789;0:27.123;0:45.320;0:46.101;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;14;1:57.711;0;;27.073;0;44.680;0;45.958;0;167.3;29:19.170;12:14:47.500;0:27.073;0:44.680;0:45.958;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;15;1:57.876;0;;27.152;0;44.704;0;46.020;0;167.1;31:17.046;12:16:45.376;0:27.152;0:44.704;0:46.020;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;16;1:58.689;0;;27.133;0;45.284;0;46.272;0;166.0;33:15.735;12:18:44.065;0:27.133;0:45.284;0:46.272;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;17;1:59.795;0;;27.343;0;45.204;0;47.248;0;164.4;35:15.530;12:20:43.860;0:27.343;0:45.204;0:47.248;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;18;1:58.745;0;;27.413;0;45.300;0;46.032;0;165.9;37:14.275;12:22:42.605;0:27.413;0:45.300;0:46.032;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;19;1:58.121;0;;27.148;0;44.963;0;46.010;0;166.8;39:12.396;12:24:40.726;0:27.148;0:44.963;0:46.010;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;20;1:59.313;0;;27.189;0;45.701;0;46.423;0;165.1;41:11.709;12:26:40.039;0:27.189;0:45.701;0:46.423;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;21;1:58.098;0;;27.097;0;44.988;0;46.013;0;166.8;43:09.807;12:28:38.137;0:27.097;0:44.988;0:46.013;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;22;1:58.181;0;;27.191;0;44.914;0;46.076;0;166.7;45:07.988;12:30:36.318;0:27.191;0:44.914;0:46.076;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;23;1:57.873;0;;27.117;0;44.796;0;45.960;0;167.1;47:05.861;12:32:34.191;0:27.117;0:44.796;0:45.960;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;24;1:58.090;0;;27.079;0;44.950;0;46.061;0;166.8;49:03.951;12:34:32.281;0:27.079;0:44.950;0:46.061;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;25;1:57.997;0;;27.062;0;44.852;0;46.083;0;166.9;51:01.948;12:36:30.278;0:27.062;0:44.852;0:46.083;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;26;1:58.134;0;;27.139;0;44.819;0;46.176;0;166.7;53:00.082;12:38:28.412;0:27.139;0:44.819;0:46.176;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;1;27;2:04.826;0;;26.987;1;45.373;0;52.466;0;157.8;55:04.908;12:40:33.238;0:26.987;0:45.373;0:52.466;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;FCY;
59;1;28;4:25.846;0;B;39.285;0;1:21.076;0;2:25.485;0;74.1;59:30.754;12:44:59.084;0:39.285;1:21.076;2:25.485;;Robert Michaelian;;GS;;KOHR MOTORSPORTS;Ford;FCY;
59;2;29;2:30.461;0;;40.095;0;48.050;0;1:02.316;0;130.9;1:02:01.215;12:47:29.545;0:40.095;0:48.050;1:02.316;;Billy Johnson;0:01:23.713;GS;;KOHR MOTORSPORTS;Ford;FCY;
59;2;30;2:45.398;0;;35.552;0;56.584;0;1:13.262;0;119.1;1:04:46.613;12:50:14.943;0:35.552;0:56.584;1:13.262;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;FCY;
59;2;31;2:59.677;0;;52.979;0;1:13.281;0;53.417;0;109.6;1:07:46.290;12:53:14.620;0:52.979;1:13.281;0:53.417;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;32;1:57.591;0;;27.165;0;45.075;0;45.351;2;167.5;1:09:43.881;12:55:12.211;0:27.165;0:45.075;0:45.351;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;33;1:57.312;0;;26.677;3;44.468;0;46.167;0;167.9;1:11:41.193;12:57:09.523;0:26.677;0:44.468;0:46.167;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;34;1:57.416;0;;26.897;0;44.804;0;45.715;0;167.8;1:13:38.609;12:59:06.939;0:26.897;0:44.804;0:45.715;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;35;1:57.114;0;;26.894;0;44.720;0;45.500;0;168.2;1:15:35.723;13:01:04.053;0:26.894;0:44.720;0:45.500;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;36;1:56.771;0;;27.100;0;44.040;2;45.631;0;168.7;1:17:32.494;13:03:00.824;0:27.100;0:44.040;0:45.631;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;37;1:56.786;0;;27.096;0;44.068;0;45.622;0;168.7;1:19:29.280;13:04:57.610;0:27.096;0:44.068;0:45.622;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;38;1:56.756;2;;27.027;0;44.052;0;45.677;0;168.7;1:21:26.036;13:06:54.366;0:27.027;0:44.052;0:45.677;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;39;1:56.775;0;;26.981;0;44.165;0;45.629;0;168.7;1:23:22.811;13:08:51.141;0:26.981;0:44.165;0:45.629;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;40;1:56.888;0;;27.033;0;44.119;0;45.736;0;168.5;1:25:19.699;13:10:48.029;0:27.033;0:44.119;0:45.736;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;41;1:56.946;0;;27.091;0;44.157;0;45.698;0;168.4;1:27:16.645;13:12:44.975;0:27.091;0:44.157;0:45.698;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;42;1:57.097;0;;27.026;0;44.363;0;45.708;0;168.2;1:29:13.742;13:14:42.072;0:27.026;0:44.363;0:45.708;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;43;1:57.150;0;;27.231;0;44.138;0;45.781;0;168.1;1:31:10.892;13:16:39.222;0:27.231;0:44.138;0:45.781;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;44;1:57.597;0;;27.129;0;44.833;0;45.635;0;167.5;1:33:08.489;13:18:36.819;0:27.129;0:44.833;0:45.635;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;45;1:57.170;0;;27.154;0;44.224;0;45.792;0;168.1;1:35:05.659;13:20:33.989;0:27.154;0:44.224;0:45.792;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;46;1:56.971;0;;27.102;0;44.126;0;45.743;0;168.4;1:37:02.630;13:22:30.960;0:27.102;0:44.126;0:45.743;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;47;1:57.075;0;;27.153;0;44.190;0;45.732;0;168.3;1:38:59.705;13:24:28.035;0:27.153;0:44.190;0:45.732;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;48;1:57.022;0;;26.940;0;44.138;0;45.944;0;168.3;1:40:56.727;13:26:25.057;0:26.940;0:44.138;0:45.944;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;49;1:57.045;0;;27.054;0;44.150;0;45.841;0;168.3;1:42:53.772;13:28:22.102;0:27.054;0:44.150;0:45.841;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;50;1:57.308;0;;27.351;0;44.199;0;45.758;0;167.9;1:44:51.080;13:30:19.410;0:27.351;0:44.199;0:45.758;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;51;2:16.601;0;;27.222;0;48.386;0;1:00.993;0;144.2;1:47:07.681;13:32:36.011;0:27.222;0:48.386;1:00.993;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;FCY;
59;2;52;3:27.961;0;;53.769;0;1:30.802;0;1:03.390;0;94.7;1:50:35.642;13:36:03.972;0:53.769;1:30.802;1:03.390;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;53;1:58.045;0;;27.587;0;44.806;0;45.652;0;166.9;1:52:33.687;13:38:02.017;0:27.587;0:44.806;0:45.652;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;54;1:56.954;0;;26.950;0;44.276;0;45.728;0;168.4;1:54:30.641;13:39:58.971;0:26.950;0:44.276;0:45.728;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;GF;
59;2;55;2:02.433;0;;26.928;0;44.240;0;51.265;0;160.9;1:56:33.074;13:42:01.404;0:26.928;0:44.240;0:51.265;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;FCY;
59;2;56;3:22.943;0;;48.117;0;1:18.209;0;1:16.617;0;97.1;1:59:56.017;13:45:24.347;0:48.117;1:18.209;1:16.617;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;FCY;
59;2;57;3:26.779;0;;52.843;0;1:17.286;0;1:16.650;0;95.3;2:03:22.796;13:48:51.126;0:52.843;1:17.286;1:16.650;;Billy Johnson;;GS;;KOHR MOTORSPORTS;Ford;FF;
64;1;1;2:13.958;0;;35.141;0;49.255;0;49.562;0;147.0;2:13.958;11:47:42.288;0:35.141;0:49.255;0:49.562;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;2;2:06.831;0;;27.700;0;48.379;0;50.752;0;155.3;4:20.789;11:49:49.119;0:27.700;0:48.379;0:50.752;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;FCY;
64;1;3;2:30.913;0;;33.224;0;59.193;0;58.496;0;130.5;6:51.702;11:52:20.032;0:33.224;0:59.193;0:58.496;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;FCY;
64;1;4;2:49.112;0;;42.975;0;1:04.737;0;1:01.400;0;116.5;9:40.814;11:55:09.144;0:42.975;1:04.737;1:01.400;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;5;2:06.231;0;;29.535;0;48.970;0;47.726;0;156.1;11:47.045;11:57:15.375;0:29.535;0:48.970;0:47.726;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;6;2:03.648;0;;27.719;0;48.359;0;47.570;0;159.3;13:50.693;11:59:19.023;0:27.719;0:48.359;0:47.570;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;7;2:03.652;0;;28.941;0;47.412;0;47.299;0;159.3;15:54.345;12:01:22.675;0:28.941;0:47.412;0:47.299;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;8;2:02.098;0;;27.707;0;46.995;0;47.396;0;161.3;17:56.443;12:03:24.773;0:27.707;0:46.995;0:47.396;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;9;2:02.137;0;;27.459;0;47.506;0;47.172;0;161.3;19:58.580;12:05:26.910;0:27.459;0:47.506;0:47.172;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;10;2:01.375;0;;27.726;0;46.521;0;47.128;0;162.3;21:59.955;12:07:28.285;0:27.726;0:46.521;0:47.128;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;11;2:00.986;0;;27.469;0;46.125;0;47.392;0;162.8;24:00.941;12:09:29.271;0:27.469;0:46.125;0:47.392;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;12;2:01.569;0;;27.462;0;46.740;0;47.367;0;162.0;26:02.510;12:11:30.840;0:27.462;0:46.740;0:47.367;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;13;2:00.957;0;;27.487;0;46.323;0;47.147;0;162.9;28:03.467;12:13:31.797;0:27.487;0:46.323;0:47.147;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;14;2:01.492;0;;27.588;0;46.516;0;47.388;0;162.1;30:04.959;12:15:33.289;0:27.588;0:46.516;0:47.388;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;15;2:00.890;0;;27.265;0;46.469;0;47.156;0;162.9;32:05.849;12:17:34.179;0:27.265;0:46.469;0:47.156;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;16;2:00.877;0;;27.309;0;46.597;0;46.971;0;163.0;34:06.726;12:19:35.056;0:27.309;0:46.597;0:46.971;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;17;2:00.423;0;;27.322;0;46.249;0;46.852;1;163.6;36:07.149;12:21:35.479;0:27.322;0:46.249;0:46.852;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;18;2:01.113;0;;27.237;1;45.931;1;47.945;0;162.6;38:08.262;12:23:36.592;0:27.237;0:45.931;0:47.945;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;19;2:01.215;0;;27.860;0;46.224;0;47.131;0;162.5;40:09.477;12:25:37.807;0:27.860;0:46.224;0:47.131;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;1;20;2:17.160;0;B;27.402;0;46.263;0;1:03.495;0;143.6;42:26.637;12:27:54.967;0:27.402;0:46.263;1:03.495;;Ted Giovanis;;GS;B;Team TGM;Aston Martin;GF;
64;2;21;3:52.951;0;B;2:03.809;0;47.016;0;1:02.126;0;84.6;46:19.588;12:31:47.918;2:03.809;0:47.016;1:02.126;;Hugh Plumb;0:01:51.473;GS;B;Team TGM;Aston Martin;GF;
64;2;22;2:11.087;0;;40.147;0;45.033;0;45.907;0;150.3;48:30.675;12:33:59.005;0:40.147;0:45.033;0:45.907;;Hugh Plumb;0:00:33.260;GS;B;Team TGM;Aston Martin;GF;
64;2;23;1:57.373;0;;27.283;0;44.500;0;45.590;0;167.8;50:28.048;12:35:56.378;0:27.283;0:44.500;0:45.590;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;24;1:57.281;0;;27.218;0;44.232;0;45.831;0;168.0;52:25.329;12:37:53.659;0:27.218;0:44.232;0:45.831;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;25;1:57.692;0;;26.976;0;45.151;0;45.565;0;167.4;54:23.021;12:39:51.351;0:26.976;0:45.151;0:45.565;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;26;2:03.739;0;;27.445;0;49.860;0;46.434;0;159.2;56:26.760;12:41:55.090;0:27.445;0:49.860;0:46.434;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;FCY;
64;2;27;2:19.952;0;;28.058;0;53.224;0;58.670;0;140.8;58:46.712;12:44:15.042;0:28.058;0:53.224;0:58.670;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;FCY;
64;2;28;2:57.956;0;;45.588;0;1:08.813;0;1:03.555;0;110.7;1:01:44.668;12:47:12.998;0:45.588;1:08.813;1:03.555;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;FCY;
64;2;29;3:00.752;0;B;29.286;0;1:08.643;0;1:22.823;0;109.0;1:04:45.420;12:50:13.750;0:29.286;1:08.643;1:22.823;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;FCY;
64;2;30;3:05.241;0;;1:07.388;0;1:06.174;0;51.679;0;106.3;1:07:50.661;12:53:18.991;1:07.388;1:06.174;0:51.679;;Hugh Plumb;0:00:51.641;GS;B;Team TGM;Aston Martin;GF;
64;2;31;1:58.673;0;;27.027;0;45.022;0;46.624;0;166.0;1:09:49.334;12:55:17.664;0:27.027;0:45.022;0:46.624;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;32;1:58.224;0;;27.004;0;44.243;0;46.977;0;166.6;1:11:47.558;12:57:15.888;0:27.004;0:44.243;0:46.977;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;33;1:56.698;0;;27.014;0;44.089;0;45.595;0;168.8;1:13:44.256;12:59:12.586;0:27.014;0:44.089;0:45.595;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;34;1:56.638;2;;27.091;0;44.087;0;45.460;0;168.9;1:15:40.894;13:01:09.224;0:27.091;0:44.087;0:45.460;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;35;1:56.827;0;;27.109;0;44.065;2;45.653;0;168.6;1:17:37.721;13:03:06.051;0:27.109;0:44.065;0:45.653;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;36;1:56.729;0;;27.133;0;44.160;0;45.436;2;168.8;1:19:34.450;13:05:02.780;0:27.133;0:44.160;0:45.436;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;37;1:58.047;0;;26.973;2;45.254;0;45.820;0;166.9;1:21:32.497;13:07:00.827;0:26.973;0:45.254;0:45.820;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;38;1:57.455;0;;27.085;0;44.471;0;45.899;0;167.7;1:23:29.952;13:08:58.282;0:27.085;0:44.471;0:45.899;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;39;1:57.420;0;;27.018;0;44.510;0;45.892;0;167.8;1:25:27.372;13:10:55.702;0:27.018;0:44.510;0:45.892;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;40;1:57.599;0;;27.138;0;44.610;0;45.851;0;167.5;1:27:24.971;13:12:53.301;0:27.138;0:44.610;0:45.851;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;41;1:58.056;0;;27.245;0;44.593;0;46.218;0;166.9;1:29:23.027;13:14:51.357;0:27.245;0:44.593;0:46.218;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;42;1:59.166;0;;27.419;0;45.247;0;46.500;0;165.3;1:31:22.193;13:16:50.523;0:27.419;0:45.247;0:46.500;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;43;1:58.017;0;;27.383;0;44.448;0;46.186;0;166.9;1:33:20.210;13:18:48.540;0:27.383;0:44.448;0:46.186;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;44;1:57.329;0;;27.163;0;44.320;0;45.846;0;167.9;1:35:17.539;13:20:45.869;0:27.163;0:44.320;0:45.846;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;45;1:57.522;0;;27.256;0;44.299;0;45.967;0;167.6;1:37:15.061;13:22:43.391;0:27.256;0:44.299;0:45.967;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;46;1:57.427;0;;27.253;0;44.328;0;45.846;0;167.7;1:39:12.488;13:24:40.818;0:27.253;0:44.328;0:45.846;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;47;1:57.522;0;;27.181;0;44.404;0;45.937;0;167.6;1:41:10.010;13:26:38.340;0:27.181;0:44.404;0:45.937;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;48;1:57.409;0;;27.161;0;44.352;0;45.896;0;167.8;1:43:07.419;13:28:35.749;0:27.161;0:44.352;0:45.896;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;49;1:57.381;0;;27.204;0;44.337;0;45.840;0;167.8;1:45:04.800;13:30:33.130;0:27.204;0:44.337;0:45.840;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;50;2:10.927;0;;27.159;0;48.387;0;55.381;0;150.5;1:47:15.727;13:32:44.057;0:27.159;0:48.387;0:55.381;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;FCY;
64;2;51;3:22.060;0;;52.456;0;1:30.446;0;59.158;0;97.5;1:50:37.787;13:36:06.117;0:52.456;1:30.446;0:59.158;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;52;2:00.364;0;;27.884;0;46.343;0;46.137;0;163.7;1:52:38.151;13:38:06.481;0:27.884;0:46.343;0:46.137;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;53;1:57.539;0;;27.342;0;44.244;0;45.953;0;167.6;1:54:35.690;13:40:04.020;0:27.342;0:44.244;0:45.953;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;GF;
64;2;54;2:03.149;0;;27.233;0;44.297;0;51.619;0;160.0;1:56:38.839;13:42:07.169;0:27.233;0:44.297;0:51.619;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;FCY;
64;2;55;3:24.703;0;;48.571;0;1:18.923;0;1:17.209;0;96.2;2:00:03.542;13:45:31.872;0:48.571;1:18.923;1:17.209;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;FCY;
64;2;56;3:26.611;0;;53.232;0;1:15.892;0;1:17.487;0;95.3;2:03:30.153;13:48:58.483;0:53.232;1:15.892;1:17.487;;Hugh Plumb;;GS;B;Team TGM;Aston Martin;FF;
67;1;1;2:03.119;0;;30.671;0;45.688;0;46.760;0;160.0;2:03.119;11:47:31.449;0:30.671;0:45.688;0:46.760;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;2;1:57.926;0;;27.238;0;44.622;0;46.066;0;167.0;4:01.045;11:49:29.375;0:27.238;0:44.622;0:46.066;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;3;2:29.130;0;;30.389;0;1:00.333;0;58.408;0;132.1;6:30.175;11:51:58.505;0:30.389;1:00.333;0:58.408;;Gordon Scully;;GS;;CSM;Porsche;FCY;
67;1;4;3:05.989;0;;43.891;0;1:17.405;0;1:04.693;0;105.9;9:36.164;11:55:04.494;0:43.891;1:17.405;1:04.693;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;5;1:58.869;0;;27.158;1;45.083;0;46.628;0;165.7;11:35.033;11:57:03.363;0:27.158;0:45.083;0:46.628;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;6;1:58.046;0;;27.209;0;44.697;0;46.140;0;166.9;13:33.079;11:59:01.409;0:27.209;0:44.697;0:46.140;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;7;1:59.361;0;;27.328;0;45.491;0;46.542;0;165.0;15:32.440;12:01:00.770;0:27.328;0:45.491;0:46.542;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;8;1:57.705;0;;27.264;0;44.493;0;45.948;1;167.4;17:30.145;12:02:58.475;0:27.264;0:44.493;0:45.948;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;9;1:58.247;0;;27.275;0;44.648;0;46.324;0;166.6;19:28.392;12:04:56.722;0:27.275;0:44.648;0:46.324;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;10;1:58.008;0;;27.244;0;44.631;0;46.133;0;166.9;21:26.400;12:06:54.730;0:27.244;0:44.631;0:46.133;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;11;1:57.897;0;;27.342;0;44.315;1;46.240;0;167.1;23:24.297;12:08:52.627;0:27.342;0:44.315;0:46.240;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;12;1:58.026;0;;27.334;0;44.560;0;46.132;0;166.9;25:22.323;12:10:50.653;0:27.334;0:44.560;0:46.132;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;13;1:57.973;0;;27.213;0;44.585;0;46.175;0;167.0;27:20.296;12:12:48.626;0:27.213;0:44.585;0:46.175;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;14;1:58.308;0;;27.393;0;44.664;0;46.251;0;166.5;29:18.604;12:14:46.934;0:27.393;0:44.664;0:46.251;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;15;1:58.093;0;;27.265;0;44.544;0;46.284;0;166.8;31:16.697;12:16:45.027;0:27.265;0:44.544;0:46.284;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;16;1:58.901;0;;27.392;0;45.132;0;46.377;0;165.7;33:15.598;12:18:43.928;0:27.392;0:45.132;0:46.377;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;17;1:59.819;0;;27.326;0;45.190;0;47.303;0;164.4;35:15.417;12:20:43.747;0:27.326;0:45.190;0:47.303;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;18;2:00.717;0;;27.464;0;46.268;0;46.985;0;163.2;37:16.134;12:22:44.464;0:27.464;0:46.268;0:46.985;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;19;1:59.317;0;;27.492;0;45.178;0;46.647;0;165.1;39:15.451;12:24:43.781;0:27.492;0:45.178;0:46.647;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;20;1:58.839;0;;27.435;0;44.977;0;46.427;0;165.8;41:14.290;12:26:42.620;0:27.435;0:44.977;0:46.427;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;21;1:58.491;0;;27.201;0;45.119;0;46.171;0;166.2;43:12.781;12:28:41.111;0:27.201;0:45.119;0:46.171;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;22;1:58.592;0;;27.391;0;44.904;0;46.297;0;166.1;45:11.373;12:30:39.703;0:27.391;0:44.904;0:46.297;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;23;1:58.252;0;;27.269;0;44.591;0;46.392;0;166.6;47:09.625;12:32:37.955;0:27.269;0:44.591;0:46.392;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;24;1:58.925;0;;27.385;0;45.218;0;46.322;0;165.6;49:08.550;12:34:36.880;0:27.385;0:45.218;0:46.322;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;25;1:58.411;0;;27.271;0;44.698;0;46.442;0;166.4;51:06.961;12:36:35.291;0:27.271;0:44.698;0:46.442;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;26;1:58.357;0;;27.276;0;44.761;0;46.320;0;166.4;53:05.318;12:38:33.648;0:27.276;0:44.761;0:46.320;;Gordon Scully;;GS;;CSM;Porsche;GF;
67;1;27;2:01.733;0;;27.458;0;45.335;0;48.940;0;161.8;55:07.051;12:40:35.381;0:27.458;0:45.335;0:48.940;;Gordon Scully;;GS;;CSM;Porsche;FCY;
67;1;28;4:29.407;0;B;40.419;0;1:19.018;0;2:29.970;0;73.1;59:36.458;12:45:04.788;0:40.419;1:19.018;2:29.970;;Gordon Scully;;GS;;CSM;Porsche;FCY;
67;2;29;2:27.934;0;;41.225;0;48.061;0;58.648;0;133.2;1:02:04.392;12:47:32.722;0:41.225;0:48.061;0:58.648;;Morgan Burkhard;0:01:28.273;GS;;CSM;Porsche;FCY;
67;2;30;2:43.474;0;;35.207;0;55.571;0;1:12.696;0;120.5;1:04:47.866;12:50:16.196;0:35.207;0:55.571;1:12.696;;Morgan Burkhard;;GS;;CSM;Porsche;FCY;
67;2;31;2:59.319;0;;53.400;0;1:13.039;0;52.880;0;109.9;1:07:47.185;12:53:15.515;0:53.400;1:13.039;0:52.880;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;32;1:58.769;0;;27.148;0;45.734;0;45.887;0;165.9;1:09:45.954;12:55:14.284;0:27.148;0:45.734;0:45.887;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;33;1:57.649;0;;27.110;2;44.496;0;46.043;0;167.4;1:11:43.603;12:57:11.933;0:27.110;0:44.496;0:46.043;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;34;1:57.317;0;;27.267;0;44.122;0;45.928;0;167.9;1:13:40.920;12:59:09.250;0:27.267;0:44.122;0:45.928;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;35;1:57.296;0;;27.162;0;44.356;0;45.778;2;167.9;1:15:38.216;13:01:06.546;0:27.162;0:44.356;0:45.778;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;36;1:57.741;0;;27.186;0;44.451;0;46.104;0;167.3;1:17:35.957;13:03:04.287;0:27.186;0:44.451;0:46.104;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;37;1:57.283;2;;27.322;0;44.046;2;45.915;0;168.0;1:19:33.240;13:05:01.570;0:27.322;0:44.046;0:45.915;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;38;1:57.788;0;;27.178;0;44.549;0;46.061;0;167.2;1:21:31.028;13:06:59.358;0:27.178;0:44.549;0:46.061;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;39;1:57.831;0;;27.200;0;44.412;0;46.219;0;167.2;1:23:28.859;13:08:57.189;0:27.200;0:44.412;0:46.219;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;40;1:57.661;0;;27.120;0;44.489;0;46.052;0;167.4;1:25:26.520;13:10:54.850;0:27.120;0:44.489;0:46.052;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;41;1:57.588;0;;27.194;0;44.363;0;46.031;0;167.5;1:27:24.108;13:12:52.438;0:27.194;0:44.363;0:46.031;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;42;1:58.101;0;;27.225;0;44.661;0;46.215;0;166.8;1:29:22.209;13:14:50.539;0:27.225;0:44.661;0:46.215;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;43;1:57.730;0;;27.209;0;44.407;0;46.114;0;167.3;1:31:19.939;13:16:48.269;0:27.209;0:44.407;0:46.114;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;44;1:57.701;0;;27.188;0;44.492;0;46.021;0;167.4;1:33:17.640;13:18:45.970;0:27.188;0:44.492;0:46.021;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;45;1:57.487;0;;27.207;0;44.199;0;46.081;0;167.7;1:35:15.127;13:20:43.457;0:27.207;0:44.199;0:46.081;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;46;1:57.288;0;;27.146;0;44.164;0;45.978;0;167.9;1:37:12.415;13:22:40.745;0:27.146;0:44.164;0:45.978;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;47;1:58.122;0;;27.201;0;44.651;0;46.270;0;166.8;1:39:10.537;13:24:38.867;0:27.201;0:44.651;0:46.270;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;48;1:57.439;0;;27.207;0;44.301;0;45.931;0;167.7;1:41:07.976;13:26:36.306;0:27.207;0:44.301;0:45.931;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;49;1:57.410;0;;27.251;0;44.299;0;45.860;0;167.8;1:43:05.386;13:28:33.716;0:27.251;0:44.299;0:45.860;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;50;1:57.601;0;;27.198;0;44.304;0;46.099;0;167.5;1:45:02.987;13:30:31.317;0:27.198;0:44.304;0:46.099;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;51;2:09.623;0;;27.269;0;46.251;0;56.103;0;152.0;1:47:12.610;13:32:40.940;0:27.269;0:46.251;0:56.103;;Morgan Burkhard;;GS;;CSM;Porsche;FCY;
67;2;52;3:24.249;0;;53.287;0;1:30.455;0;1:00.507;0;96.4;1:50:36.859;13:36:05.189;0:53.287;1:30.455;1:00.507;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;53;2:00.038;0;;27.287;0;46.640;0;46.111;0;164.1;1:52:36.897;13:38:05.227;0:27.287;0:46.640;0:46.111;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;54;1:57.737;0;;27.149;0;44.525;0;46.063;0;167.3;1:54:34.634;13:40:02.964;0:27.149;0:44.525;0:46.063;;Morgan Burkhard;;GS;;CSM;Porsche;GF;
67;2;55;2:03.067;0;;27.165;0;44.409;0;51.493;0;160.1;1:56:37.701;13:42:06.031;0:27.165;0:44.409;0:51.493;;Morgan Burkhard;;GS;;CSM;Porsche;FCY;
67;2;56;3:24.039;0;;47.906;0;1:19.219;0;1:16.914;0;96.5;2:00:01.740;13:45:30.070;0:47.906;1:19.219;1:16.914;;Morgan Burkhard;;GS;;CSM;Porsche;FCY;
67;2;57;3:26.665;0;;52.393;0;1:16.561;0;1:17.711;0;95.3;2:03:28.405;13:48:56.735;0:52.393;1:16.561;1:17.711;;Morgan Burkhard;;GS;;CSM;Porsche;FF;
7;1;1;2:10.866;0;;37.746;0;45.930;0;47.190;0;150.5;2:10.866;11:47:39.196;0:37.746;0:45.930;0:47.190;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;2;1:57.666;0;;27.378;0;44.151;0;46.137;0;167.4;4:08.532;11:49:36.862;0:27.378;0:44.151;0:46.137;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;3;2:31.558;0;;31.499;0;1:01.257;0;58.802;0;130.0;6:40.090;11:52:08.420;0:31.499;1:01.257;0:58.802;;Celso Neto;;TCR;;Precision Racing LA;Audi;FCY;
7;1;4;3:01.513;0;;43.501;0;1:17.355;0;1:00.657;0;108.5;9:41.603;11:55:09.933;0:43.501;1:17.355;1:00.657;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;5;2:01.725;0;;27.740;0;46.290;0;47.695;0;161.8;11:43.328;11:57:11.658;0:27.740;0:46.290;0:47.695;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;6;1:58.306;0;;28.380;0;43.969;2;45.957;1;166.5;13:41.634;11:59:09.964;0:28.380;0:43.969;0:45.957;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;7;1:58.258;0;;27.418;0;44.727;0;46.113;0;166.6;15:39.892;12:01:08.222;0:27.418;0:44.727;0:46.113;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;8;1:58.384;0;;27.391;0;44.872;0;46.121;0;166.4;17:38.276;12:03:06.606;0:27.391;0:44.872;0:46.121;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;9;1:58.109;0;;27.626;0;44.412;0;46.071;0;166.8;19:36.385;12:05:04.715;0:27.626;0:44.412;0:46.071;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;10;1:58.597;0;;27.371;1;44.548;0;46.678;0;166.1;21:34.982;12:07:03.312;0:27.371;0:44.548;0:46.678;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;11;1:59.485;0;;27.955;0;45.387;0;46.143;0;164.9;23:34.467;12:09:02.797;0:27.955;0:45.387;0:46.143;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;12;1:58.909;0;;27.416;0;45.132;0;46.361;0;165.7;25:33.376;12:11:01.706;0:27.416;0:45.132;0:46.361;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;13;1:58.197;0;;27.406;0;44.602;0;46.189;0;166.7;27:31.573;12:12:59.903;0:27.406;0:44.602;0:46.189;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;14;1:58.504;0;;27.494;0;44.710;0;46.300;0;166.2;29:30.077;12:14:58.407;0:27.494;0:44.710;0:46.300;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;15;1:58.574;0;;27.534;0;44.679;0;46.361;0;166.1;31:28.651;12:16:56.981;0:27.534;0:44.679;0:46.361;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;16;1:58.591;0;;27.451;0;44.743;0;46.397;0;166.1;33:27.242;12:18:55.572;0:27.451;0:44.743;0:46.397;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;17;1:59.119;0;;27.425;0;45.097;0;46.597;0;165.4;35:26.361;12:20:54.691;0:27.425;0:45.097;0:46.597;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;18;1:59.826;0;;27.434;0;45.327;0;47.065;0;164.4;37:26.187;12:22:54.517;0:27.434;0:45.327;0:47.065;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;19;1:58.856;0;;27.447;0;44.775;0;46.634;0;165.7;39:25.043;12:24:53.373;0:27.447;0:44.775;0:46.634;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;20;2:00.027;0;;27.567;0;45.673;0;46.787;0;164.1;41:25.070;12:26:53.400;0:27.567;0:45.673;0:46.787;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;21;1:58.652;0;;27.577;0;44.792;0;46.283;0;166.0;43:23.722;12:28:52.052;0:27.577;0:44.792;0:46.283;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;22;1:58.216;0;;27.498;0;44.380;0;46.338;0;166.6;45:21.938;12:30:50.268;0:27.498;0:44.380;0:46.338;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;23;1:58.415;0;;27.456;0;44.695;0;46.264;0;166.4;47:20.353;12:32:48.683;0:27.456;0:44.695;0:46.264;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;24;1:58.254;0;;27.555;0;44.415;0;46.284;0;166.6;49:18.607;12:34:46.937;0:27.555;0:44.415;0:46.284;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;25;1:58.778;0;;27.546;0;44.416;0;46.816;0;165.8;51:17.385;12:36:45.715;0:27.546;0:44.416;0:46.816;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;26;1:59.329;0;;27.876;0;45.158;0;46.295;0;165.1;53:16.714;12:38:45.044;0:27.876;0:45.158;0:46.295;;Celso Neto;;TCR;;Precision Racing LA;Audi;GF;
7;1;27;1:58.813;0;;27.465;0;44.508;0;46.840;0;165.8;55:15.527;12:40:43.857;0:27.465;0:44.508;0:46.840;;Celso Neto;;TCR;;Precision Racing LA;Audi;FCY;
7;1;28;3:17.820;0;;34.793;0;1:18.023;0;1:25.004;0;99.6;58:33.347;12:44:01.677;0:34.793;1:18.023;1:25.004;;Celso Neto;;TCR;;Precision Racing LA;Audi;FCY;
7;1;29;3:12.942;0;B;44.318;0;1:10.216;0;1:18.408;0;102.1;1:01:46.289;12:47:14.619;0:44.318;1:10.216;1:18.408;;Celso Neto;;TCR;;Precision Racing LA;Audi;FCY;
7;2;30;3:20.524;0;;1:28.786;0;56.721;0;55.017;0;98.2;1:05:06.813;12:50:35.143;1:28.786;0:56.721;0:55.017;;Ryan Eversley;0:01:13.234;TCR;;Precision Racing LA;Audi;FCY;
7;2;31;2:46.502;0;;41.464;0;1:11.553;0;53.485;0;118.3;1:07:53.315;12:53:21.645;0:41.464;1:11.553;0:53.485;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;32;1:58.198;0;;27.546;0;44.686;0;45.966;0;166.7;1:09:51.513;12:55:19.843;0:27.546;0:44.686;0:45.966;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;33;1:57.348;0;;27.241;0;44.154;0;45.953;0;167.9;1:11:48.861;12:57:17.191;0:27.241;0:44.154;0:45.953;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;34;1:58.254;0;;27.403;0;44.638;0;46.213;0;166.6;1:13:47.115;12:59:15.445;0:27.403;0:44.638;0:46.213;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;35;1:57.074;2;;27.236;2;44.100;0;45.738;2;168.3;1:15:44.189;13:01:12.519;0:27.236;0:44.100;0:45.738;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;36;1:57.242;0;;27.289;0;44.042;1;45.911;0;168.0;1:17:41.431;13:03:09.761;0:27.289;0:44.042;0:45.911;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;37;1:57.587;0;;27.353;0;44.162;0;46.072;0;167.5;1:19:39.018;13:05:07.348;0:27.353;0:44.162;0:46.072;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;38;2:10.457;0;B;27.455;0;44.386;0;58.616;0;151.0;1:21:49.475;13:07:17.805;0:27.455;0:44.386;0:58.616;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;39;2:30.537;0;;59.422;0;44.884;0;46.231;0;130.9;1:24:20.012;13:09:48.342;0:59.422;0:44.884;0:46.231;;Ryan Eversley;0:00:48.678;TCR;;Precision Racing LA;Audi;GF;
7;2;40;1:58.202;0;;27.431;0;44.561;0;46.210;0;166.7;1:26:18.214;13:11:46.544;0:27.431;0:44.561;0:46.210;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;41;1:58.169;0;;27.448;0;44.469;0;46.252;0;166.7;1:28:16.383;13:13:44.713;0:27.448;0:44.469;0:46.252;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;42;1:58.735;0;;27.365;0;44.894;0;46.476;0;165.9;1:30:15.118;13:15:43.448;0:27.365;0:44.894;0:46.476;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;43;1:58.372;0;;27.501;0;44.543;0;46.328;0;166.4;1:32:13.490;13:17:41.820;0:27.501;0:44.543;0:46.328;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;44;1:59.326;0;;27.513;0;44.691;0;47.122;0;165.1;1:34:12.816;13:19:41.146;0:27.513;0:44.691;0:47.122;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;45;1:59.446;0;;27.544;0;44.614;0;47.288;0;164.9;1:36:12.262;13:21:40.592;0:27.544;0:44.614;0:47.288;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;46;1:58.947;0;;27.644;0;45.067;0;46.236;0;165.6;1:38:11.209;13:23:39.539;0:27.644;0:45.067;0:46.236;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;47;1:58.423;0;;27.663;0;44.505;0;46.255;0;166.3;1:40:09.632;13:25:37.962;0:27.663;0:44.505;0:46.255;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;48;1:58.473;0;;27.520;0;44.676;0;46.277;0;166.3;1:42:08.105;13:27:36.435;0:27.520;0:44.676;0:46.277;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;49;1:58.440;0;;27.568;0;44.568;0;46.304;0;166.3;1:44:06.545;13:29:34.875;0:27.568;0:44.568;0:46.304;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;50;1:58.864;0;;27.554;0;44.421;0;46.889;0;165.7;1:46:05.409;13:31:33.739;0:27.554;0:44.421;0:46.889;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;FCY;
7;2;51;2:07.310;0;;28.595;0;50.819;0;47.896;0;154.7;1:48:12.719;13:33:41.049;0:28.595;0:50.819;0:47.896;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;FCY;
7;2;52;2:32.425;0;;28.166;0;1:11.737;0;52.522;0;129.2;1:50:45.144;13:36:13.474;0:28.166;1:11.737;0:52.522;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;53;2:00.447;0;;27.623;0;45.803;0;47.021;0;163.5;1:52:45.591;13:38:13.921;0:27.623;0:45.803;0:47.021;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;54;2:00.638;0;;27.848;0;45.279;0;47.511;0;163.3;1:54:46.229;13:40:14.559;0:27.848;0:45.279;0:47.511;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;GF;
7;2;55;2:02.993;0;;27.604;0;45.440;0;49.949;0;160.2;1:56:49.222;13:42:17.552;0:27.604;0:45.440;0:49.949;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;FCY;
7;2;56;3:24.893;0;;48.424;0;1:17.823;0;1:18.646;0;96.1;2:00:14.115;13:45:42.445;0:48.424;1:17.823;1:18.646;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;FCY;
7;2;57;3:28.087;0;;55.329;0;1:13.712;0;1:19.046;0;94.7;2:03:42.202;13:49:10.532;0:55.329;1:13.712;1:19.046;;Ryan Eversley;;TCR;;Precision Racing LA;Audi;FF;
71;1;1;2:06.564;0;;32.628;0;47.119;0;46.817;0;155.6;2:06.564;11:47:34.894;0:32.628;0:47.119;0:46.817;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;2;1:59.514;0;;27.374;0;45.748;0;46.392;0;164.8;4:06.078;11:49:34.408;0:27.374;0:45.748;0:46.392;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;3;2:28.938;0;;29.605;0;59.375;0;59.958;0;132.3;6:35.016;11:52:03.346;0:29.605;0:59.375;0:59.958;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;FCY;
71;1;4;3:03.351;0;;43.505;0;1:16.501;0;1:03.345;0;107.4;9:38.367;11:55:06.697;0:43.505;1:16.501;1:03.345;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;5;2:00.315;0;;27.417;0;46.179;0;46.719;0;163.7;11:38.682;11:57:07.012;0:27.417;0:46.179;0:46.719;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;6;1:58.605;0;;27.132;0;45.197;0;46.276;0;166.1;13:37.287;11:59:05.617;0:27.132;0:45.197;0:46.276;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;7;1:59.312;0;;27.725;0;45.234;0;46.353;0;165.1;15:36.599;12:01:04.929;0:27.725;0:45.234;0:46.353;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;8;1:58.666;0;;27.637;0;44.756;1;46.273;0;166.0;17:35.265;12:03:03.595;0:27.637;0:44.756;0:46.273;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;9;1:58.568;0;;27.376;0;44.812;0;46.380;0;166.1;19:33.833;12:05:02.163;0:27.376;0:44.812;0:46.380;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;10;1:59.957;0;;27.722;0;45.537;0;46.698;0;164.2;21:33.790;12:07:02.120;0:27.722;0:45.537;0:46.698;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;11;1:58.195;0;;27.382;0;44.806;0;46.007;1;166.7;23:31.985;12:09:00.315;0:27.382;0:44.806;0:46.007;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;12;1:59.055;0;;27.337;0;44.877;0;46.841;0;165.5;25:31.040;12:10:59.370;0:27.337;0:44.877;0:46.841;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;13;1:58.806;0;;27.312;0;45.196;0;46.298;0;165.8;27:29.846;12:12:58.176;0:27.312;0:45.196;0:46.298;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;14;1:59.091;0;;27.295;0;45.413;0;46.383;0;165.4;29:28.937;12:14:57.267;0:27.295;0:45.413;0:46.383;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;15;1:58.402;0;;27.079;1;45.108;0;46.215;0;166.4;31:27.339;12:16:55.669;0:27.079;0:45.108;0:46.215;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;16;1:58.981;0;;27.134;0;45.379;0;46.468;0;165.6;33:26.320;12:18:54.650;0:27.134;0:45.379;0:46.468;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;17;1:59.329;0;;27.544;0;45.254;0;46.531;0;165.1;35:25.649;12:20:53.979;0:27.544;0:45.254;0:46.531;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;18;1:59.279;0;;27.411;0;45.447;0;46.421;0;165.1;37:24.928;12:22:53.258;0:27.411;0:45.447;0:46.421;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;19;1:59.616;0;;27.639;0;45.137;0;46.840;0;164.7;39:24.544;12:24:52.874;0:27.639;0:45.137;0:46.840;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;20;2:00.238;0;;27.541;0;45.666;0;47.031;0;163.8;41:24.782;12:26:53.112;0:27.541;0:45.666;0:47.031;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;1;21;3:00.400;0;B;27.444;0;46.828;0;1:46.128;0;109.2;44:25.182;12:29:53.512;0:27.444;0:46.828;1:46.128;;Frank DePew;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;22;2:10.236;0;;39.821;0;44.654;0;45.761;0;151.3;46:35.418;12:32:03.748;0:39.821;0:44.654;0:45.761;;Robin Liddell;0:01:14.899;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;23;1:56.040;0;;26.838;0;43.756;0;45.446;0;169.8;48:31.458;12:33:59.788;0:26.838;0:43.756;0:45.446;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;24;1:57.724;0;;27.454;0;44.779;0;45.491;0;167.3;50:29.182;12:35:57.512;0:27.454;0:44.779;0:45.491;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;25;1:55.836;3;;26.777;0;43.626;0;45.433;2;170.1;52:25.018;12:37:53.348;0:26.777;0:43.626;0:45.433;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;26;1:55.909;0;;26.774;0;43.586;2;45.549;0;169.9;54:20.927;12:39:49.257;0:26.774;0:43.586;0:45.549;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;27;2:02.175;0;;27.148;0;47.176;0;47.851;0;161.2;56:23.102;12:41:51.432;0:27.148;0:47.176;0:47.851;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;FCY;
71;2;28;2:51.333;0;B;29.605;0;53.189;0;1:28.539;0;115.0;59:14.435;12:44:42.765;0:29.605;0:53.189;1:28.539;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;FCY;
71;2;29;2:40.475;0;;43.224;0;51.588;0;1:05.663;0;122.8;1:01:54.910;12:47:23.240;0:43.224;0:51.588;1:05.663;;Robin Liddell;0:00:48.350;GS;;Rebel Rock Racing;Aston Martin;FCY;
71;2;30;2:48.488;0;;35.610;0;58.108;0;1:14.770;0;116.9;1:04:43.398;12:50:11.728;0:35.610;0:58.108;1:14.770;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;FCY;
71;2;31;3:01.168;0;;53.042;0;1:13.602;0;54.524;0;108.7;1:07:44.566;12:53:12.896;0:53.042;1:13.602;0:54.524;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;32;1:58.003;0;;27.340;0;44.856;0;45.807;0;166.9;1:09:42.569;12:55:10.899;0:27.340;0:44.856;0:45.807;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;33;1:58.357;0;;26.802;0;45.360;0;46.195;0;166.4;1:11:40.926;12:57:09.256;0:26.802;0:45.360;0:46.195;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;34;1:57.357;0;;27.019;0;44.633;0;45.705;0;167.8;1:13:38.283;12:59:06.613;0:27.019;0:44.633;0:45.705;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;35;1:56.483;0;;27.048;0;43.961;0;45.474;0;169.1;1:15:34.766;13:01:03.096;0:27.048;0:43.961;0:45.474;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;36;1:56.766;0;;26.835;0;44.213;0;45.718;0;168.7;1:17:31.532;13:02:59.862;0:26.835;0:44.213;0:45.718;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;37;1:56.900;0;;26.938;0;44.201;0;45.761;0;168.5;1:19:28.432;13:04:56.762;0:26.938;0:44.201;0:45.761;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;38;1:56.790;0;;26.826;0;44.127;0;45.837;0;168.7;1:21:25.222;13:06:53.552;0:26.826;0:44.127;0:45.837;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;39;1:56.656;0;;26.860;0;44.105;0;45.691;0;168.9;1:23:21.878;13:08:50.208;0:26.860;0:44.105;0:45.691;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;40;1:57.094;0;;26.744;2;44.410;0;45.940;0;168.2;1:25:18.972;13:10:47.302;0:26.744;0:44.410;0:45.940;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;41;1:56.816;0;;26.861;0;44.202;0;45.753;0;168.6;1:27:15.788;13:12:44.118;0:26.861;0:44.202;0:45.753;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;42;1:56.659;0;;26.920;0;44.078;0;45.661;0;168.9;1:29:12.447;13:14:40.777;0:26.920;0:44.078;0:45.661;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;43;1:57.022;0;;26.771;0;44.442;0;45.809;0;168.3;1:31:09.469;13:16:37.799;0:26.771;0:44.442;0:45.809;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;44;2:07.566;0;;26.817;0;54.726;0;46.023;0;154.4;1:33:17.035;13:18:45.365;0:26.817;0:54.726;0:46.023;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;45;1:57.255;0;;27.082;0;44.206;0;45.967;0;168.0;1:35:14.290;13:20:42.620;0:27.082;0:44.206;0:45.967;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;46;1:56.814;0;;26.907;0;44.127;0;45.780;0;168.6;1:37:11.104;13:22:39.434;0:26.907;0:44.127;0:45.780;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;47;1:56.777;0;;26.962;0;44.096;0;45.719;0;168.7;1:39:07.881;13:24:36.211;0:26.962;0:44.096;0:45.719;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;48;1:57.139;0;;26.882;0;44.370;0;45.887;0;168.2;1:41:05.020;13:26:33.350;0:26.882;0:44.370;0:45.887;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;49;1:57.068;0;;26.967;0;44.054;0;46.047;0;168.3;1:43:02.088;13:28:30.418;0:26.967;0:44.054;0:46.047;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;50;1:57.062;0;;27.077;0;44.219;0;45.766;0;168.3;1:44:59.150;13:30:27.480;0:27.077;0:44.219;0:45.766;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;51;2:11.636;0;;26.996;0;46.601;0;58.039;0;149.6;1:47:10.786;13:32:39.116;0:26.996;0:46.601;0:58.039;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;FCY;
71;2;52;3:25.488;0;;53.340;0;1:30.714;0;1:01.434;0;95.9;1:50:36.274;13:36:04.604;0:53.340;1:30.714;1:01.434;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;53;1:59.473;0;;27.498;0;46.375;0;45.600;0;164.9;1:52:35.747;13:38:04.077;0:27.498;0:46.375;0:45.600;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;54;1:57.772;0;;27.030;0;44.883;0;45.859;0;167.3;1:54:33.519;13:40:01.849;0:27.030;0:44.883;0:45.859;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;GF;
71;2;55;2:02.961;0;;26.967;0;44.395;0;51.599;0;160.2;1:56:36.480;13:42:04.810;0:26.967;0:44.395;0:51.599;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;FCY;
71;2;56;3:23.065;0;;47.757;0;1:18.659;0;1:16.649;0;97.0;1:59:59.545;13:45:27.875;0:47.757;1:18.659;1:16.649;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;FCY;
71;2;57;3:26.477;0;;52.735;0;1:16.778;0;1:16.964;0;95.4;2:03:26.022;13:48:54.352;0:52.735;1:16.778;1:16.964;;Robin Liddell;;GS;;Rebel Rock Racing;Aston Martin;FF;
72;1;1;2:16.781;0;;42.645;0;46.762;0;47.374;0;144.0;2:16.781;11:47:45.111;0:42.645;0:46.762;0:47.374;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;2;4:10.281;0;;27.837;0;2:44.739;0;57.705;0;78.7;6:27.062;11:51:55.392;0:27.837;2:44.739;0:57.705;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;FCY;
72;1;3;2:24.908;0;;44.869;0;52.037;0;48.002;0;135.9;8:51.970;11:54:20.300;0:44.869;0:52.037;0:48.002;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;FCY;
72;1;4;2:00.906;0;;28.134;0;45.833;0;46.939;0;162.9;10:52.876;11:56:21.206;0:28.134;0:45.833;0:46.939;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;5;2:00.714;0;;27.896;0;45.808;0;47.010;0;163.2;12:53.590;11:58:21.920;0:27.896;0:45.808;0:47.010;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;6;2:00.985;0;;27.938;0;45.985;0;47.062;0;162.8;14:54.575;12:00:22.905;0:27.938;0:45.985;0:47.062;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;7;2:01.384;0;;27.943;0;46.308;0;47.133;0;162.3;16:55.959;12:02:24.289;0:27.943;0:46.308;0:47.133;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;8;2:01.443;0;;27.718;0;46.095;0;47.630;0;162.2;18:57.402;12:04:25.732;0:27.718;0:46.095;0:47.630;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;9;2:01.141;0;;27.960;0;46.080;0;47.101;0;162.6;20:58.543;12:06:26.873;0:27.960;0:46.080;0:47.101;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;10;2:01.078;0;;28.257;0;45.816;0;47.005;0;162.7;22:59.621;12:08:27.951;0:28.257;0:45.816;0:47.005;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;11;2:00.639;0;;27.823;0;45.606;0;47.210;0;163.3;25:00.260;12:10:28.590;0:27.823;0:45.606;0:47.210;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;12;2:00.684;0;;27.726;0;46.029;0;46.929;0;163.2;27:00.944;12:12:29.274;0:27.726;0:46.029;0:46.929;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;13;2:02.077;0;;27.766;0;46.549;0;47.762;0;161.4;29:03.021;12:14:31.351;0:27.766;0:46.549;0:47.762;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;14;2:03.683;0;;29.515;0;46.697;0;47.471;0;159.3;31:06.704;12:16:35.034;0:29.515;0:46.697;0:47.471;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;15;2:01.186;0;;28.563;0;45.745;0;46.878;0;162.5;33:07.890;12:18:36.220;0:28.563;0:45.745;0:46.878;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;16;2:01.656;0;;27.907;0;45.999;0;47.750;0;161.9;35:09.546;12:20:37.876;0:27.907;0:45.999;0:47.750;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;17;2:01.186;0;;27.610;0;46.493;0;47.083;0;162.5;37:10.732;12:22:39.062;0:27.610;0:46.493;0:47.083;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;18;2:00.415;0;;27.644;0;45.880;0;46.891;0;163.6;39:11.147;12:24:39.477;0:27.644;0:45.880;0:46.891;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;19;2:02.469;0;;27.774;0;46.033;0;48.662;0;160.8;41:13.616;12:26:41.946;0:27.774;0:46.033;0:48.662;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;20;2:01.198;0;;27.635;0;46.561;0;47.002;0;162.5;43:14.814;12:28:43.144;0:27.635;0:46.561;0:47.002;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;21;2:00.223;0;;27.644;0;45.690;0;46.889;0;163.8;45:15.037;12:30:43.367;0:27.644;0:45.690;0:46.889;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;22;2:00.463;0;;27.613;0;45.535;1;47.315;0;163.5;47:15.500;12:32:43.830;0:27.613;0:45.535;0:47.315;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;23;2:00.584;0;;27.853;0;45.963;0;46.768;1;163.4;49:16.084;12:34:44.414;0:27.853;0:45.963;0:46.768;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;24;2:00.952;0;;27.800;0;46.116;0;47.036;0;162.9;51:17.036;12:36:45.366;0:27.800;0:46.116;0:47.036;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;25;2:02.717;0;;28.087;0;47.745;0;46.885;0;160.5;53:19.753;12:38:48.083;0:28.087;0:47.745;0:46.885;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;1;26;3:25.643;0;;27.588;1;2:07.135;0;50.920;0;95.8;56:45.396;12:42:13.726;0:27.588;2:07.135;0:50.920;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;FCY;
72;1;27;2:09.450;0;;30.452;0;49.427;0;49.571;0;152.2;58:54.846;12:44:23.176;0:30.452;0:49.427;0:49.571;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;FCY;
72;1;28;3:41.863;0;B;38.210;0;1:09.050;0;1:54.603;0;88.8;1:02:36.709;12:48:05.039;0:38.210;1:09.050;1:54.603;;Riley Pegram;;TCR;;Pegram Racing;Hyundai;FCY;
72;2;29;2:31.234;0;;43.734;0;53.007;0;54.493;0;130.3;1:05:07.943;12:50:36.273;0:43.734;0:53.007;0:54.493;;Larry Pegram;0:01:07.547;TCR;;Pegram Racing;Hyundai;FCY;
72;2;30;2:47.668;0;;41.626;0;1:11.066;0;54.976;0;117.5;1:07:55.611;12:53:23.941;0:41.626;1:11.066;0:54.976;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;31;2:07.113;0;;32.547;0;47.703;0;46.863;0;155.0;1:10:02.724;12:55:31.054;0:32.547;0:47.703;0:46.863;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;32;1:59.342;0;;27.810;0;44.967;0;46.565;0;165.1;1:12:02.066;12:57:30.396;0:27.810;0:44.967;0:46.565;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;33;1:58.458;0;;27.631;0;44.688;0;46.139;2;166.3;1:14:00.524;12:59:28.854;0:27.631;0:44.688;0:46.139;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;34;1:58.602;0;;27.483;0;44.487;0;46.632;0;166.1;1:15:59.126;13:01:27.456;0:27.483;0:44.487;0:46.632;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;35;1:58.254;2;;27.531;0;44.316;2;46.407;0;166.6;1:17:57.380;13:03:25.710;0:27.531;0:44.316;0:46.407;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;36;1:59.541;0;;28.175;0;44.806;0;46.560;0;164.8;1:19:56.921;13:05:25.251;0:28.175;0:44.806;0:46.560;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;37;2:00.603;0;;27.389;2;45.544;0;47.670;0;163.3;1:21:57.524;13:07:25.854;0:27.389;0:45.544;0:47.670;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;38;2:06.685;0;;27.613;0;52.502;0;46.570;0;155.5;1:24:04.209;13:09:32.539;0:27.613;0:52.502;0:46.570;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;39;1:58.721;0;;27.597;0;44.701;0;46.423;0;165.9;1:26:02.930;13:11:31.260;0:27.597;0:44.701;0:46.423;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;40;1:59.595;0;;27.686;0;44.465;0;47.444;0;164.7;1:28:02.525;13:13:30.855;0:27.686;0:44.465;0:47.444;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;41;1:59.881;0;;28.704;0;44.594;0;46.583;0;164.3;1:30:02.406;13:15:30.736;0:28.704;0:44.594;0:46.583;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;42;2:00.248;0;;27.492;0;45.028;0;47.728;0;163.8;1:32:02.654;13:17:30.984;0:27.492;0:45.028;0:47.728;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;43;1:59.966;0;;27.765;0;45.082;0;47.119;0;164.2;1:34:02.620;13:19:30.950;0:27.765;0:45.082;0:47.119;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;44;1:59.211;0;;27.879;0;44.840;0;46.492;0;165.2;1:36:01.831;13:21:30.161;0:27.879;0:44.840;0:46.492;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;45;1:58.728;0;;27.632;0;44.598;0;46.498;0;165.9;1:38:00.559;13:23:28.889;0:27.632;0:44.598;0:46.498;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;46;1:59.143;0;;27.751;0;44.699;0;46.693;0;165.3;1:39:59.702;13:25:28.032;0:27.751;0:44.699;0:46.693;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;47;1:59.521;0;;27.571;0;44.944;0;47.006;0;164.8;1:41:59.223;13:27:27.553;0:27.571;0:44.944;0:47.006;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
72;2;48;2:00.095;0;;27.807;0;45.549;0;46.739;0;164.0;1:43:59.318;13:29:27.648;0:27.807;0:45.549;0:46.739;;Larry Pegram;;TCR;;Pegram Racing;Hyundai;GF;
76;2;1;2:14.318;0;;39.465;0;45.937;0;48.916;0;146.7;2:14.318;11:47:42.648;0:39.465;0:45.937;0:48.916;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;2;2:02.027;0;;28.158;0;46.293;0;47.576;0;161.4;4:16.345;11:49:44.675;0:28.158;0:46.293;0:47.576;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
76;2;3;2:30.387;0;;30.643;0;1:01.617;0;58.127;0;131.0;6:46.732;11:52:15.062;0:30.643;1:01.617;0:58.127;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
76;2;4;2:56.918;0;;42.219;0;1:14.814;0;59.885;0;111.3;9:43.650;11:55:11.980;0:42.219;1:14.814;0:59.885;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;5;2:00.867;0;;27.938;0;46.545;0;46.384;0;163.0;11:44.517;11:57:12.847;0:27.938;0:46.545;0:46.384;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;6;2:02.172;0;;29.400;0;46.453;0;46.319;1;161.2;13:46.689;11:59:15.019;0:29.400;0:46.453;0:46.319;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;7;1:59.243;0;;28.018;0;44.464;1;46.761;0;165.2;15:45.932;12:01:14.262;0:28.018;0:44.464;0:46.761;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;8;2:00.891;0;;28.942;0;45.052;0;46.897;0;162.9;17:46.823;12:03:15.153;0:28.942;0:45.052;0:46.897;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;9;2:01.269;0;;28.105;0;45.487;0;47.677;0;162.4;19:48.092;12:05:16.422;0:28.105;0:45.487;0:47.677;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;10;2:01.344;0;;28.731;0;45.393;0;47.220;0;162.3;21:49.436;12:07:17.766;0:28.731;0:45.393;0:47.220;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;11;2:00.469;0;;28.054;0;45.566;0;46.849;0;163.5;23:49.905;12:09:18.235;0:28.054;0:45.566;0:46.849;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;12;2:00.751;0;;28.254;0;45.443;0;47.054;0;163.1;25:50.656;12:11:18.986;0:28.254;0:45.443;0:47.054;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;13;2:00.401;0;;28.258;0;44.892;0;47.251;0;163.6;27:51.057;12:13:19.387;0:28.258;0:44.892;0:47.251;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;14;2:00.519;0;;28.218;0;45.047;0;47.254;0;163.4;29:51.576;12:15:19.906;0:28.218;0:45.047;0:47.254;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;15;2:00.709;0;;28.168;0;44.765;0;47.776;0;163.2;31:52.285;12:17:20.615;0:28.168;0:44.765;0:47.776;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;16;2:00.853;0;;28.161;0;45.953;0;46.739;0;163.0;33:53.138;12:19:21.468;0:28.161;0:45.953;0:46.739;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;17;2:00.461;0;;27.796;1;45.793;0;46.872;0;163.5;35:53.599;12:21:21.929;0:27.796;0:45.793;0:46.872;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;18;2:00.614;0;;28.387;0;45.075;0;47.152;0;163.3;37:54.213;12:23:22.543;0:28.387;0:45.075;0:47.152;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;19;2:00.496;0;;28.091;0;45.249;0;47.156;0;163.5;39:54.709;12:25:23.039;0:28.091;0:45.249;0:47.156;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;20;1:59.248;0;;28.021;0;44.606;0;46.621;0;165.2;41:53.957;12:27:22.287;0:28.021;0:44.606;0:46.621;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;21;2:00.196;0;;28.098;0;45.071;0;47.027;0;163.9;43:54.153;12:29:22.483;0:28.098;0:45.071;0:47.027;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;22;1:59.772;0;;28.124;0;44.801;0;46.847;0;164.5;45:53.925;12:31:22.255;0:28.124;0:44.801;0:46.847;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;23;2:00.094;0;;27.872;0;45.290;0;46.932;0;164.0;47:54.019;12:33:22.349;0:27.872;0:45.290;0:46.932;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;24;1:59.719;0;;27.970;0;45.049;0;46.700;0;164.5;49:53.738;12:35:22.068;0:27.970;0:45.049;0:46.700;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;25;2:01.035;0;;27.820;0;45.420;0;47.795;0;162.7;51:54.773;12:37:23.103;0:27.820;0:45.420;0:47.795;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;26;2:01.212;0;;27.883;0;46.128;0;47.201;0;162.5;53:55.985;12:39:24.315;0:27.883;0:46.128;0:47.201;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;2;27;2:13.026;0;;28.197;0;47.137;0;57.692;0;148.1;56:09.011;12:41:37.341;0:28.197;0:47.137;0:57.692;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
76;2;28;2:33.206;0;;36.385;0;53.994;0;1:02.827;0;128.6;58:42.217;12:44:10.547;0:36.385;0:53.994;1:02.827;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
76;2;29;3:14.250;0;B;46.407;0;1:09.401;0;1:18.442;0;101.4;1:01:56.467;12:47:24.797;0:46.407;1:09.401;1:18.442;;Preston Brown;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
76;1;30;3:29.419;0;;1:42.434;0;55.952;0;51.033;0;94.1;1:05:25.886;12:50:54.216;1:42.434;0:55.952;0:51.033;;Denis Dupont;0:01:27.788;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
76;1;31;2:33.287;0;;34.535;0;1:07.447;0;51.305;0;128.5;1:07:59.173;12:53:27.503;0:34.535;1:07.447;0:51.305;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;32;1:59.128;0;;28.464;0;45.204;0;45.460;2;165.4;1:09:58.301;12:55:26.631;0:28.464;0:45.204;0:45.460;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;33;1:56.871;2;;27.368;2;43.712;2;45.791;0;168.5;1:11:55.172;12:57:23.502;0:27.368;0:43.712;0:45.791;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;34;1:58.180;0;;27.582;0;43.975;0;46.623;0;166.7;1:13:53.352;12:59:21.682;0:27.582;0:43.975;0:46.623;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;35;1:58.877;0;;27.574;0;44.648;0;46.655;0;165.7;1:15:52.229;13:01:20.559;0:27.574;0:44.648;0:46.655;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;36;1:58.239;0;;27.600;0;44.563;0;46.076;0;166.6;1:17:50.468;13:03:18.798;0:27.600;0:44.563;0:46.076;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;37;1:58.200;0;;27.688;0;44.301;0;46.211;0;166.7;1:19:48.668;13:05:16.998;0:27.688;0:44.301;0:46.211;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;38;1:58.336;0;;27.542;0;44.715;0;46.079;0;166.5;1:21:47.004;13:07:15.334;0:27.542;0:44.715;0:46.079;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;39;1:58.660;0;;27.681;0;44.705;0;46.274;0;166.0;1:23:45.664;13:09:13.994;0:27.681;0:44.705;0:46.274;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;40;1:58.879;0;;27.628;0;44.780;0;46.471;0;165.7;1:25:44.543;13:11:12.873;0:27.628;0:44.780;0:46.471;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;41;1:58.755;0;;27.761;0;44.520;0;46.474;0;165.9;1:27:43.298;13:13:11.628;0:27.761;0:44.520;0:46.474;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;42;1:59.242;0;;27.666;0;44.643;0;46.933;0;165.2;1:29:42.540;13:15:10.870;0:27.666;0:44.643;0:46.933;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;43;1:58.979;0;;27.811;0;44.670;0;46.498;0;165.6;1:31:41.519;13:17:09.849;0:27.811;0:44.670;0:46.498;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;44;1:58.937;0;;27.688;0;44.810;0;46.439;0;165.6;1:33:40.456;13:19:08.786;0:27.688;0:44.810;0:46.439;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;45;1:59.183;0;;27.795;0;44.981;0;46.407;0;165.3;1:35:39.639;13:21:07.969;0:27.795;0:44.981;0:46.407;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;46;1:58.991;0;;27.826;0;44.686;0;46.479;0;165.5;1:37:38.630;13:23:06.960;0:27.826;0:44.686;0:46.479;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;47;1:59.099;0;;28.009;0;44.453;0;46.637;0;165.4;1:39:37.729;13:25:06.059;0:28.009;0:44.453;0:46.637;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;48;1:59.132;0;;27.859;0;44.742;0;46.531;0;165.3;1:41:36.861;13:27:05.191;0:27.859;0:44.742;0:46.531;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;49;1:59.268;0;;28.114;0;44.647;0;46.507;0;165.2;1:43:36.129;13:29:04.459;0:28.114;0:44.647;0:46.507;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;50;2:00.236;0;;27.880;0;46.004;0;46.352;0;163.8;1:45:36.365;13:31:04.695;0:27.880;0:46.004;0:46.352;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;51;2:13.835;0;;32.585;0;52.600;0;48.650;0;147.2;1:47:50.200;13:33:18.530;0:32.585;0:52.600;0:48.650;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
76;1;52;2:54.098;0;;30.942;0;1:28.503;0;54.653;0;113.1;1:50:44.298;13:36:12.628;0:30.942;1:28.503;0:54.653;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;53;2:00.935;0;;27.981;0;46.037;0;46.917;0;162.9;1:52:45.233;13:38:13.563;0:27.981;0:46.037;0:46.917;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;54;2:00.141;0;;27.710;0;45.588;0;46.843;0;164.0;1:54:45.374;13:40:13.704;0:27.710;0:45.588;0:46.843;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
76;1;55;2:02.650;0;;27.677;0;44.925;0;50.048;0;160.6;1:56:48.024;13:42:16.354;0:27.677;0:44.925;0:50.048;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
76;1;56;3:23.832;0;;47.905;0;1:17.865;0;1:18.062;0;96.6;2:00:11.856;13:45:40.186;0:47.905;1:17.865;1:18.062;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
76;1;57;3:27.495;0;;54.409;0;1:14.157;0;1:18.929;0;94.9;2:03:39.351;13:49:07.681;0:54.409;1:14.157;1:18.929;;Denis Dupont;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FF;
89;2;1;2:15.925;0;;42.054;0;46.407;0;47.464;0;144.9;2:15.925;11:47:44.255;0:42.054;0:46.407;0:47.464;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;2;2:01.592;0;;28.085;0;46.557;0;46.950;0;162.0;4:17.517;11:49:45.847;0:28.085;0:46.557;0:46.950;;Tyler Chambers;;TCR;;HART;Honda;FCY;
89;2;3;2:32.167;0;;33.700;0;1:00.149;0;58.318;0;129.5;6:49.684;11:52:18.014;0:33.700;1:00.149;0:58.318;;Tyler Chambers;;TCR;;HART;Honda;FCY;
89;2;4;2:55.151;0;;42.947;0;1:13.737;0;58.467;0;112.5;9:44.835;11:55:13.165;0:42.947;1:13.737;0:58.467;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;5;2:03.827;0;;28.657;0;46.502;0;48.668;0;159.1;11:48.662;11:57:16.992;0:28.657;0:46.502;0:48.668;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;6;2:00.792;0;;27.969;0;45.949;0;46.874;0;163.1;13:49.454;11:59:17.784;0:27.969;0:45.949;0:46.874;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;7;1:59.363;0;;27.902;0;45.093;0;46.368;0;165.0;15:48.817;12:01:17.147;0:27.902;0:45.093;0:46.368;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;8;1:58.454;0;;27.436;0;44.737;0;46.281;0;166.3;17:47.271;12:03:15.601;0:27.436;0:44.737;0:46.281;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;9;2:01.144;0;;28.162;0;45.552;0;47.430;0;162.6;19:48.415;12:05:16.745;0:28.162;0:45.552;0:47.430;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;10;2:01.217;0;;28.501;0;45.522;0;47.194;0;162.5;21:49.632;12:07:17.962;0:28.501;0:45.522;0:47.194;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;11;1:59.424;0;;27.950;0;45.010;0;46.464;0;164.9;23:49.056;12:09:17.386;0:27.950;0:45.010;0:46.464;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;12;2:00.336;0;;28.503;0;45.544;0;46.289;0;163.7;25:49.392;12:11:17.722;0:28.503;0:45.544;0:46.289;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;13;1:58.503;0;;27.537;0;44.590;0;46.376;0;166.2;27:47.895;12:13:16.225;0:27.537;0:44.590;0:46.376;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;14;1:58.651;0;;27.591;0;45.085;0;45.975;0;166.0;29:46.546;12:15:14.876;0:27.591;0:45.085;0:45.975;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;15;1:58.693;0;;27.673;0;44.695;0;46.325;0;166.0;31:45.239;12:17:13.569;0:27.673;0:44.695;0:46.325;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;16;1:58.512;0;;27.411;0;44.818;0;46.283;0;166.2;33:43.751;12:19:12.081;0:27.411;0:44.818;0:46.283;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;17;2:00.893;0;;28.395;0;45.951;0;46.547;0;162.9;35:44.644;12:21:12.974;0:28.395;0:45.951;0:46.547;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;18;1:58.672;0;;27.820;0;44.776;0;46.076;0;166.0;37:43.316;12:23:11.646;0:27.820;0:44.776;0:46.076;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;19;1:57.682;0;;27.404;0;44.600;0;45.678;1;167.4;39:40.998;12:25:09.328;0:27.404;0:44.600;0:45.678;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;20;1:58.207;0;;27.455;0;44.575;1;46.177;0;166.6;41:39.205;12:27:07.535;0:27.455;0:44.575;0:46.177;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;21;1:58.222;0;;27.577;0;44.679;0;45.966;0;166.6;43:37.427;12:29:05.757;0:27.577;0:44.679;0:45.966;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;22;1:58.772;0;;27.465;0;44.927;0;46.380;0;165.9;45:36.199;12:31:04.529;0:27.465;0:44.927;0:46.380;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;23;2:00.464;0;;28.079;0;45.867;0;46.518;0;163.5;47:36.663;12:33:04.993;0:28.079;0:45.867;0:46.518;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;24;1:58.251;0;;27.465;0;44.717;0;46.069;0;166.6;49:34.914;12:35:03.244;0:27.465;0:44.717;0:46.069;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;25;1:58.516;0;;27.551;0;44.868;0;46.097;0;166.2;51:33.430;12:37:01.760;0:27.551;0:44.868;0:46.097;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;26;1:59.763;0;;27.385;1;45.980;0;46.398;0;164.5;53:33.193;12:39:01.523;0:27.385;0:45.980;0:46.398;;Tyler Chambers;;TCR;;HART;Honda;GF;
89;2;27;2:04.608;0;;27.788;0;46.389;0;50.431;0;158.1;55:37.801;12:41:06.131;0:27.788;0:46.389;0:50.431;;Tyler Chambers;;TCR;;HART;Honda;FCY;
89;2;28;2:59.801;0;;28.843;0;1:07.566;0;1:23.392;0;109.6;58:37.602;12:44:05.932;0:28.843;1:07.566;1:23.392;;Tyler Chambers;;TCR;;HART;Honda;FCY;
89;2;29;3:13.004;0;B;46.871;0;1:09.249;0;1:16.884;0;102.1;1:01:50.606;12:47:18.936;0:46.871;1:09.249;1:16.884;;Tyler Chambers;;TCR;;HART;Honda;FCY;
89;1;30;3:27.951;0;;1:40.429;0;55.294;0;52.228;0;94.7;1:05:18.557;12:50:46.887;1:40.429;0:55.294;0:52.228;;CHAD GILSINGER;0:01:27.581;TCR;;HART;Honda;FCY;
89;1;31;2:38.930;0;;37.435;0;1:09.303;0;52.192;0;123.9;1:07:57.487;12:53:25.817;0:37.435;1:09.303;0:52.192;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;32;1:58.849;0;;27.848;0;45.230;0;45.771;0;165.7;1:09:56.336;12:55:24.666;0:27.848;0:45.230;0:45.771;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;33;1:57.496;2;;27.194;2;44.717;0;45.585;2;167.7;1:11:53.832;12:57:22.162;0:27.194;0:44.717;0:45.585;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;34;1:59.052;0;;27.380;0;44.539;0;47.133;0;165.5;1:13:52.884;12:59:21.214;0:27.380;0:44.539;0:47.133;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;35;2:02.490;0;;27.347;0;44.623;0;50.520;0;160.8;1:15:55.374;13:01:23.704;0:27.347;0:44.623;0:50.520;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;36;1:59.766;0;;28.352;0;45.552;0;45.862;0;164.5;1:17:55.140;13:03:23.470;0:28.352;0:45.552;0:45.862;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;37;2:00.113;0;;27.923;0;45.441;0;46.749;0;164.0;1:19:55.253;13:05:23.583;0:27.923;0:45.441;0:46.749;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;38;1:58.432;0;;28.017;0;44.570;0;45.845;0;166.3;1:21:53.685;13:07:22.015;0:28.017;0:44.570;0:45.845;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;39;1:57.659;0;;27.617;0;44.385;2;45.657;0;167.4;1:23:51.344;13:09:19.674;0:27.617;0:44.385;0:45.657;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;40;1:57.561;0;;27.348;0;44.570;0;45.643;0;167.6;1:25:48.905;13:11:17.235;0:27.348;0:44.570;0:45.643;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;41;1:58.205;0;;27.316;0;44.627;0;46.262;0;166.6;1:27:47.110;13:13:15.440;0:27.316;0:44.627;0:46.262;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;42;1:57.900;0;;27.366;0;44.543;0;45.991;0;167.1;1:29:45.010;13:15:13.340;0:27.366;0:44.543;0:45.991;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;43;1:58.899;0;;27.263;0;45.114;0;46.522;0;165.7;1:31:43.909;13:17:12.239;0:27.263;0:45.114;0:46.522;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;44;1:58.646;0;;27.945;0;44.815;0;45.886;0;166.0;1:33:42.555;13:19:10.885;0:27.945;0:44.815;0:45.886;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;45;1:58.041;0;;27.329;0;44.629;0;46.083;0;166.9;1:35:40.596;13:21:08.926;0:27.329;0:44.629;0:46.083;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;46;1:58.537;0;;27.375;0;44.937;0;46.225;0;166.2;1:37:39.133;13:23:07.463;0:27.375;0:44.937;0:46.225;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;47;1:59.148;0;;28.066;0;44.965;0;46.117;0;165.3;1:39:38.281;13:25:06.611;0:28.066;0:44.965;0:46.117;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;48;1:59.046;0;;27.546;0;45.442;0;46.058;0;165.5;1:41:37.327;13:27:05.657;0:27.546;0:45.442;0:46.058;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;49;1:59.393;0;;28.034;0;44.998;0;46.361;0;165.0;1:43:36.720;13:29:05.050;0:28.034;0:44.998;0:46.361;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;50;1:59.048;0;;27.502;0;45.394;0;46.152;0;165.5;1:45:35.768;13:31:04.098;0:27.502;0:45.394;0:46.152;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;51;2:13.779;0;;32.158;0;51.799;0;49.822;0;147.2;1:47:49.547;13:33:17.877;0:32.158;0:51.799;0:49.822;;CHAD GILSINGER;;TCR;;HART;Honda;FCY;
89;1;52;2:54.510;0;;30.810;0;1:28.618;0;55.082;0;112.9;1:50:44.057;13:36:12.387;0:30.810;1:28.618;0:55.082;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;53;2:00.106;0;;27.671;0;46.146;0;46.289;0;164.0;1:52:44.163;13:38:12.493;0:27.671;0:46.146;0:46.289;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;54;2:00.370;0;;28.454;0;45.463;0;46.453;0;163.6;1:54:44.533;13:40:12.863;0:28.454;0:45.463;0:46.453;;CHAD GILSINGER;;TCR;;HART;Honda;GF;
89;1;55;2:02.741;0;;28.075;0;44.850;0;49.816;0;160.5;1:56:47.274;13:42:15.604;0:28.075;0:44.850;0:49.816;;CHAD GILSINGER;;TCR;;HART;Honda;FCY;
89;1;56;3:22.834;0;;46.909;0;1:18.985;0;1:16.940;0;97.1;2:00:10.108;13:45:38.438;0:46.909;1:18.985;1:16.940;;CHAD GILSINGER;;TCR;;HART;Honda;FCY;
89;1;57;3:27.068;0;;54.219;0;1:14.607;0;1:18.242;0;95.1;2:03:37.176;13:49:05.506;0:54.219;1:14.607;1:18.242;;CHAD GILSINGER;;TCR;;HART;Honda;FF;
9;1;1;2:14.944;0;;40.776;0;46.619;0;47.549;0;146.0;2:14.944;11:47:43.274;0:40.776;0:46.619;0:47.549;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;2;2:01.831;0;;28.147;0;46.710;0;46.974;0;161.7;4:16.775;11:49:45.105;0:28.147;0:46.710;0:46.974;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;1;3;2:31.665;0;;33.439;0;1:00.439;0;57.787;0;129.9;6:48.440;11:52:16.770;0:33.439;1:00.439;0:57.787;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;1;4;2:55.607;0;;42.868;0;1:13.898;0;58.841;0;112.2;9:44.047;11:55:12.377;0:42.868;1:13.898;0:58.841;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;5;2:03.757;0;;28.230;0;47.280;0;48.247;0;159.2;11:47.804;11:57:16.134;0:28.230;0:47.280;0:48.247;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;6;2:03.494;0;;28.450;0;47.453;0;47.591;0;159.5;13:51.298;11:59:19.628;0:28.450;0:47.453;0:47.591;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;7;2:01.957;0;;28.593;0;46.443;0;46.921;0;161.5;15:53.255;12:01:21.585;0:28.593;0:46.443;0:46.921;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;8;2:01.410;0;;28.203;0;46.142;0;47.065;0;162.2;17:54.665;12:03:22.995;0:28.203;0:46.142;0:47.065;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;9;2:00.562;0;;28.066;0;45.605;0;46.891;0;163.4;19:55.227;12:05:23.557;0:28.066;0:45.605;0:46.891;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;10;2:00.257;0;;28.055;0;45.497;0;46.705;0;163.8;21:55.484;12:07:23.814;0:28.055;0:45.497;0:46.705;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;11;2:00.795;0;;28.073;0;45.894;0;46.828;0;163.1;23:56.279;12:09:24.609;0:28.073;0:45.894;0:46.828;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;12;2:01.829;0;;27.954;0;47.169;0;46.706;0;161.7;25:58.108;12:11:26.438;0:27.954;0:47.169;0:46.706;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;13;2:00.295;0;;27.897;0;45.635;0;46.763;0;163.8;27:58.403;12:13:26.733;0:27.897;0:45.635;0:46.763;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;14;2:00.097;0;;27.952;0;45.569;0;46.576;1;164.0;29:58.500;12:15:26.830;0:27.952;0:45.569;0:46.576;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;15;2:00.483;0;;27.854;0;45.867;0;46.762;0;163.5;31:58.983;12:17:27.313;0:27.854;0:45.867;0:46.762;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;16;2:00.331;0;;27.700;1;45.657;0;46.974;0;163.7;33:59.314;12:19:27.644;0:27.700;0:45.657;0:46.974;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;17;2:00.783;0;;27.919;0;46.153;0;46.711;0;163.1;36:00.097;12:21:28.427;0:27.919;0:46.153;0:46.711;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;18;2:00.462;0;;27.959;0;45.639;0;46.864;0;163.5;38:00.559;12:23:28.889;0:27.959;0:45.639;0:46.864;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;19;2:00.634;0;;27.855;0;45.817;0;46.962;0;163.3;40:01.193;12:25:29.523;0:27.855;0:45.817;0:46.962;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;20;2:00.661;0;;27.890;0;45.871;0;46.900;0;163.3;42:01.854;12:27:30.184;0:27.890;0:45.871;0:46.900;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;21;2:00.301;0;;27.842;0;45.731;0;46.728;0;163.7;44:02.155;12:29:30.485;0:27.842;0:45.731;0:46.728;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;22;2:03.001;0;;28.584;0;47.112;0;47.305;0;160.1;46:05.156;12:31:33.486;0:28.584;0:47.112;0:47.305;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;23;2:01.026;0;;28.007;0;45.609;0;47.410;0;162.8;48:06.182;12:33:34.512;0:28.007;0:45.609;0:47.410;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;24;2:00.856;0;;28.679;0;45.398;1;46.779;0;163.0;50:07.038;12:35:35.368;0:28.679;0:45.398;0:46.779;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;25;2:00.971;0;;28.225;0;45.783;0;46.963;0;162.8;52:08.009;12:37:36.339;0:28.225;0:45.783;0:46.963;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;26;2:00.259;0;;27.981;0;45.564;0;46.714;0;163.8;54:08.268;12:39:36.598;0:27.981;0:45.564;0:46.714;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;1;27;2:05.602;0;;28.245;0;48.212;0;49.145;0;156.8;56:13.870;12:41:42.200;0:28.245;0:48.212;0:49.145;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;1;28;2:31.726;0;;35.138;0;55.394;0;1:01.194;0;129.8;58:45.596;12:44:13.926;0:35.138;0:55.394;1:01.194;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;1;29;3:13.175;0;B;45.374;0;1:09.145;0;1:18.656;0;102.0;1:01:58.771;12:47:27.101;0:45.374;1:09.145;1:18.656;;Madison Aust;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;2;30;3:35.729;0;;1:51.759;0;52.704;0;51.266;0;91.3;1:05:34.500;12:51:02.830;1:51.759;0:52.704;0:51.266;;Suellio Almeida;0:01:38.502;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;2;31;2:29.927;0;;30.668;0;1:05.551;0;53.708;0;131.4;1:08:04.427;12:53:32.757;0:30.668;1:05.551;0:53.708;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;32;1:59.947;0;;28.094;0;45.304;0;46.549;0;164.2;1:10:04.374;12:55:32.704;0:28.094;0:45.304;0:46.549;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;33;1:58.734;0;;27.712;0;44.916;0;46.106;0;165.9;1:12:03.108;12:57:31.438;0:27.712;0:44.916;0:46.106;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;34;1:58.541;0;;27.673;0;44.880;0;45.988;0;166.2;1:14:01.649;12:59:29.979;0:27.673;0:44.880;0:45.988;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;35;1:58.340;0;;27.675;0;44.660;0;46.005;0;166.5;1:15:59.989;13:01:28.319;0:27.675;0:44.660;0:46.005;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;36;1:58.132;2;;27.562;2;44.600;2;45.970;2;166.7;1:17:58.121;13:03:26.451;0:27.562;0:44.600;0:45.970;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;37;1:59.121;0;;27.755;0;44.953;0;46.413;0;165.4;1:19:57.242;13:05:25.572;0:27.755;0:44.953;0:46.413;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;38;2:00.640;0;;27.648;0;45.150;0;47.842;0;163.3;1:21:57.882;13:07:26.212;0:27.648;0:45.150;0:47.842;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;39;1:59.473;0;;27.962;0;45.330;0;46.181;0;164.9;1:23:57.355;13:09:25.685;0:27.962;0:45.330;0:46.181;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;40;2:01.387;0;;27.891;0;46.188;0;47.308;0;162.3;1:25:58.742;13:11:27.072;0:27.891;0:46.188;0:47.308;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;41;2:00.406;0;;27.960;0;45.856;0;46.590;0;163.6;1:27:59.148;13:13:27.478;0:27.960;0:45.856;0:46.590;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;42;1:58.840;0;;27.739;0;44.687;0;46.414;0;165.8;1:29:57.988;13:15:26.318;0:27.739;0:44.687;0:46.414;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;43;1:59.193;0;;27.701;0;44.826;0;46.666;0;165.3;1:31:57.181;13:17:25.511;0:27.701;0:44.826;0:46.666;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;44;1:59.350;0;;27.605;0;45.030;0;46.715;0;165.0;1:33:56.531;13:19:24.861;0:27.605;0:45.030;0:46.715;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;45;2:01.145;0;;28.172;0;46.209;0;46.764;0;162.6;1:35:57.676;13:21:26.006;0:28.172;0:46.209;0:46.764;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;46;1:59.439;0;;27.945;0;45.060;0;46.434;0;164.9;1:37:57.115;13:23:25.445;0:27.945;0:45.060;0:46.434;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;47;1:59.300;0;;27.793;0;44.953;0;46.554;0;165.1;1:39:56.415;13:25:24.745;0:27.793;0:44.953;0:46.554;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;48;1:59.187;0;;27.860;0;44.937;0;46.390;0;165.3;1:41:55.602;13:27:23.932;0:27.860;0:44.937;0:46.390;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;49;1:59.017;0;;27.757;0;44.851;0;46.409;0;165.5;1:43:54.619;13:29:22.949;0:27.757;0:44.851;0:46.409;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;50;2:00.217;0;;27.746;0;44.890;0;47.581;0;163.9;1:45:54.836;13:31:23.166;0:27.746;0:44.890;0:47.581;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;2;51;2:08.289;0;;28.538;0;52.367;0;47.384;0;153.5;1:48:03.125;13:33:31.455;0:28.538;0:52.367;0:47.384;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;2;52;2:41.706;0;;28.037;0;1:20.016;0;53.653;0;121.8;1:50:44.831;13:36:13.161;0:28.037;1:20.016;0:53.653;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;53;2:01.331;0;;27.856;0;46.604;0;46.871;0;162.4;1:52:46.162;13:38:14.492;0:27.856;0:46.604;0:46.871;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;54;2:00.859;0;;27.825;0;45.600;0;47.434;0;163.0;1:54:47.021;13:40:15.351;0:27.825;0:45.600;0:47.434;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;GF;
9;2;55;2:02.617;0;;27.970;0;44.956;0;49.691;0;160.6;1:56:49.638;13:42:17.968;0:27.970;0:44.956;0:49.691;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;2;56;3:25.237;0;;49.046;0;1:17.553;0;1:18.638;0;96.0;2:00:14.875;13:45:43.205;0:49.046;1:17.553;1:18.638;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FCY;
9;2;57;3:27.912;0;;55.636;0;1:13.561;0;1:18.715;0;94.7;2:03:42.787;13:49:11.117;0:55.636;1:13.561;1:18.715;;Suellio Almeida;;TCR;;Bryan Herta Autosport w Curb Agajanian;Hyundai;FF;
93;2;1;2:09.404;0;;37.426;0;45.725;0;46.253;0;152.2;2:09.404;11:47:37.734;0:37.426;0:45.725;0:46.253;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;2;1:57.030;0;;27.277;0;44.381;0;45.372;1;168.3;4:06.434;11:49:34.764;0:27.277;0:44.381;0:45.372;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;3;2:30.206;0;;30.331;0;59.209;0;1:00.666;0;131.1;6:36.640;11:52:04.970;0:30.331;0:59.209;1:00.666;;LP Montour;;TCR;;MMG;Honda;FCY;
93;2;4;3:04.627;0;;43.101;0;1:20.379;0;1:01.147;0;106.7;9:41.267;11:55:09.597;0:43.101;1:20.379;1:01.147;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;5;1:58.445;0;;27.850;0;44.861;0;45.734;0;166.3;11:39.712;11:57:08.042;0:27.850;0:44.861;0:45.734;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;6;1:57.770;0;;27.393;0;44.779;0;45.598;0;167.3;13:37.482;11:59:05.812;0:27.393;0:44.779;0:45.598;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;7;1:58.482;0;;27.969;0;44.279;0;46.234;0;166.3;15:35.964;12:01:04.294;0:27.969;0:44.279;0:46.234;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;8;1:56.868;0;;27.389;0;44.035;1;45.444;0;168.6;17:32.832;12:03:01.162;0:27.389;0:44.035;0:45.444;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;9;1:58.766;0;;28.374;0;44.513;0;45.879;0;165.9;19:31.598;12:04:59.928;0:28.374;0:44.513;0:45.879;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;10;2:00.246;0;;29.757;0;44.517;0;45.972;0;163.8;21:31.844;12:07:00.174;0:29.757;0:44.517;0:45.972;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;11;1:57.881;0;;27.501;0;44.710;0;45.670;0;167.1;23:29.725;12:08:58.055;0:27.501;0:44.710;0:45.670;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;12;1:57.676;0;;27.329;0;44.530;0;45.817;0;167.4;25:27.401;12:10:55.731;0:27.329;0:44.530;0:45.817;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;13;1:57.461;0;;27.319;0;44.337;0;45.805;0;167.7;27:24.862;12:12:53.192;0:27.319;0:44.337;0:45.805;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;14;1:57.768;0;;27.384;0;44.632;0;45.752;0;167.3;29:22.630;12:14:50.960;0:27.384;0:44.632;0:45.752;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;15;1:57.992;0;;27.283;0;44.688;0;46.021;0;166.9;31:20.622;12:16:48.952;0:27.283;0:44.688;0:46.021;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;16;1:57.904;0;;27.333;0;44.861;0;45.710;0;167.1;33:18.526;12:18:46.856;0:27.333;0:44.861;0:45.710;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;17;1:57.677;0;;27.385;0;44.521;0;45.771;0;167.4;35:16.203;12:20:44.533;0:27.385;0:44.521;0:45.771;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;18;1:58.342;0;;27.128;2;45.237;0;45.977;0;166.5;37:14.545;12:22:42.875;0:27.128;0:45.237;0:45.977;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;19;1:58.191;0;;27.384;0;44.837;0;45.970;0;166.7;39:12.736;12:24:41.066;0:27.384;0:44.837;0:45.970;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;20;1:59.313;0;;27.497;0;45.275;0;46.541;0;165.1;41:12.049;12:26:40.379;0:27.497;0:45.275;0:46.541;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;21;1:58.396;0;;27.523;0;44.840;0;46.033;0;166.4;43:10.445;12:28:38.775;0:27.523;0:44.840;0:46.033;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;22;1:57.983;0;;27.440;0;44.616;0;45.927;0;167.0;45:08.428;12:30:36.758;0:27.440;0:44.616;0:45.927;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;23;1:57.934;0;;27.273;0;44.544;0;46.117;0;167.0;47:06.362;12:32:34.692;0:27.273;0:44.544;0:46.117;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;24;1:57.924;0;;27.363;0;44.622;0;45.939;0;167.0;49:04.286;12:34:32.616;0:27.363;0:44.622;0:45.939;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;25;1:58.228;0;;27.420;0;44.604;0;46.204;0;166.6;51:02.514;12:36:30.844;0:27.420;0:44.604;0:46.204;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;26;1:57.920;0;;27.273;0;44.691;0;45.956;0;167.0;53:00.434;12:38:28.764;0:27.273;0:44.691;0:45.956;;LP Montour;;TCR;;MMG;Honda;GF;
93;2;27;2:05.141;0;;27.288;0;45.094;0;52.759;0;157.4;55:05.575;12:40:33.905;0:27.288;0:45.094;0:52.759;;LP Montour;;TCR;;MMG;Honda;FCY;
93;2;28;3:27.322;0;;41.287;0;1:19.038;0;1:26.997;0;95.0;58:32.897;12:44:01.227;0:41.287;1:19.038;1:26.997;;LP Montour;;TCR;;MMG;Honda;FCY;
93;2;29;3:12.925;0;B;42.144;0;1:11.742;0;1:19.039;0;102.1;1:01:45.822;12:47:14.152;0:42.144;1:11.742;1:19.039;;LP Montour;;TCR;;MMG;Honda;FCY;
93;1;30;3:28.986;0;;1:40.496;0;54.178;0;54.312;0;94.3;1:05:14.808;12:50:43.138;1:40.496;0:54.178;0:54.312;;Karl Wittmer;0:01:24.860;TCR;;MMG;Honda;FCY;
93;1;31;2:40.823;0;;36.890;0;1:10.161;0;53.772;0;122.5;1:07:55.631;12:53:23.961;0:36.890;1:10.161;0:53.772;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;32;1:57.776;0;;28.091;0;44.351;0;45.334;2;167.3;1:09:53.407;12:55:21.737;0:28.091;0:44.351;0:45.334;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;33;1:56.699;2;;27.272;0;43.800;2;45.627;0;168.8;1:11:50.106;12:57:18.436;0:27.272;0:43.800;0:45.627;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;34;1:58.784;0;;27.309;0;45.315;0;46.160;0;165.8;1:13:48.890;12:59:17.220;0:27.309;0:45.315;0:46.160;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;35;1:57.956;0;;27.215;1;44.585;0;46.156;0;167.0;1:15:46.846;13:01:15.176;0:27.215;0:44.585;0:46.156;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;36;1:56.911;0;;27.327;0;43.837;0;45.747;0;168.5;1:17:43.757;13:03:12.087;0:27.327;0:43.837;0:45.747;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;37;1:58.095;0;;27.516;0;44.334;0;46.245;0;166.8;1:19:41.852;13:05:10.182;0:27.516;0:44.334;0:46.245;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;38;1:58.395;0;;27.531;0;45.072;0;45.792;0;166.4;1:21:40.247;13:07:08.577;0:27.531;0:45.072;0:45.792;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;39;1:57.348;0;;27.428;0;44.402;0;45.518;0;167.9;1:23:37.595;13:09:05.925;0:27.428;0:44.402;0:45.518;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;40;1:57.576;0;;27.247;0;44.469;0;45.860;0;167.5;1:25:35.171;13:11:03.501;0:27.247;0:44.469;0:45.860;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;41;1:57.483;0;;27.363;0;44.375;0;45.745;0;167.7;1:27:32.654;13:13:00.984;0:27.363;0:44.375;0:45.745;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;42;1:57.699;0;;27.404;0;44.584;0;45.711;0;167.4;1:29:30.353;13:14:58.683;0:27.404;0:44.584;0:45.711;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;43;1:57.842;0;;27.370;0;44.669;0;45.803;0;167.2;1:31:28.195;13:16:56.525;0:27.370;0:44.669;0:45.803;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;44;1:58.984;0;;27.433;0;45.069;0;46.482;0;165.6;1:33:27.179;13:18:55.509;0:27.433;0:45.069;0:46.482;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;45;1:58.396;0;;27.515;0;45.003;0;45.878;0;166.4;1:35:25.575;13:20:53.905;0:27.515;0:45.003;0:45.878;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;46;1:58.341;0;;27.411;0;44.928;0;46.002;0;166.5;1:37:23.916;13:22:52.246;0:27.411;0:44.928;0:46.002;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;47;1:58.821;0;;27.536;0;45.205;0;46.080;0;165.8;1:39:22.737;13:24:51.067;0:27.536;0:45.205;0:46.080;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;48;1:58.803;0;;27.434;0;45.123;0;46.246;0;165.8;1:41:21.540;13:26:49.870;0:27.434;0:45.123;0:46.246;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;49;1:58.308;0;;27.465;0;45.251;0;45.592;0;166.5;1:43:19.848;13:28:48.178;0:27.465;0:45.251;0:45.592;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;50;1:58.462;0;;27.419;0;45.066;0;45.977;0;166.3;1:45:18.310;13:30:46.640;0:27.419;0:45.066;0:45.977;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;51;2:02.444;0;;27.827;0;47.534;0;47.083;0;160.9;1:47:20.754;13:32:49.084;0:27.827;0:47.534;0:47.083;;Karl Wittmer;;TCR;;MMG;Honda;FCY;
93;1;52;3:21.373;0;;49.171;0;1:34.725;0;57.477;0;97.8;1:50:42.127;13:36:10.457;0:49.171;1:34.725;0:57.477;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;53;1:59.124;0;;27.600;0;45.926;0;45.598;0;165.4;1:52:41.251;13:38:09.581;0:27.600;0:45.926;0:45.598;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;54;1:58.399;0;;27.376;0;44.912;0;46.111;0;166.4;1:54:39.650;13:40:07.980;0:27.376;0:44.912;0:46.111;;Karl Wittmer;;TCR;;MMG;Honda;GF;
93;1;55;2:02.645;0;;27.439;0;45.780;0;49.426;0;160.6;1:56:42.295;13:42:10.625;0:27.439;0:45.780;0:49.426;;Karl Wittmer;;TCR;;MMG;Honda;FCY;
93;1;56;3:24.781;0;;48.372;0;1:19.648;0;1:16.761;0;96.2;2:00:07.076;13:45:35.406;0:48.372;1:19.648;1:16.761;;Karl Wittmer;;TCR;;MMG;Honda;FCY;
93;1;57;3:27.125;0;;53.620;0;1:15.580;0;1:17.925;0;95.1;2:03:34.201;13:49:02.531;0:53.620;1:15.580;1:17.925;;Karl Wittmer;;TCR;;MMG;Honda;FF;
95;2;1;2:01.463;0;;29.685;0;45.567;0;46.211;0;162.2;2:01.463;11:47:29.793;0:29.685;0:45.567;0:46.211;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;2;1:57.723;0;;27.351;0;44.637;0;45.735;0;167.3;3:59.186;11:49:27.516;0:27.351;0:44.637;0:45.735;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;3;2:29.029;0;;30.324;0;1:01.058;0;57.647;0;132.2;6:28.215;11:51:56.545;0:30.324;1:01.058;0:57.647;;Dillon Machavern;;GS;;Turner Motorsport;BMW;FCY;
95;2;4;3:06.986;0;;44.154;0;1:17.465;0;1:05.367;0;105.3;9:35.201;11:55:03.531;0:44.154;1:17.465;1:05.367;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;5;1:58.269;0;;27.276;0;44.983;0;46.010;0;166.6;11:33.470;11:57:01.800;0:27.276;0:44.983;0:46.010;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;6;1:57.027;0;;27.098;0;44.370;0;45.559;0;168.3;13:30.497;11:58:58.827;0:27.098;0:44.370;0:45.559;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;7;1:56.919;0;;27.206;0;44.137;1;45.576;0;168.5;15:27.416;12:00:55.746;0:27.206;0:44.137;0:45.576;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;8;1:56.858;2;;27.114;0;44.138;0;45.606;0;168.6;17:24.274;12:02:52.604;0:27.114;0:44.138;0:45.606;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;9;1:57.166;0;;27.193;0;44.410;0;45.563;0;168.1;19:21.440;12:04:49.770;0:27.193;0:44.410;0:45.563;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;10;1:57.081;0;;27.168;0;44.139;0;45.774;0;168.2;21:18.521;12:06:46.851;0:27.168;0:44.139;0:45.774;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;11;1:56.879;0;;27.134;0;44.241;0;45.504;2;168.5;23:15.400;12:08:43.730;0:27.134;0:44.241;0:45.504;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;12;1:57.146;0;;27.215;0;44.239;0;45.692;0;168.2;25:12.546;12:10:40.876;0:27.215;0:44.239;0:45.692;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;13;1:57.075;0;;27.128;0;44.206;0;45.741;0;168.3;27:09.621;12:12:37.951;0:27.128;0:44.206;0:45.741;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;14;1:57.404;0;;27.050;1;44.584;0;45.770;0;167.8;29:07.025;12:14:35.355;0:27.050;0:44.584;0:45.770;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;15;1:57.805;0;;27.135;0;44.704;0;45.966;0;167.2;31:04.830;12:16:33.160;0:27.135;0:44.704;0:45.966;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;16;1:57.929;0;;27.120;0;44.487;0;46.322;0;167.0;33:02.759;12:18:31.089;0:27.120;0:44.487;0:46.322;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;17;1:57.645;0;;27.302;0;44.520;0;45.823;0;167.4;35:00.404;12:20:28.734;0:27.302;0:44.520;0:45.823;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;18;1:57.439;0;;27.106;0;44.279;0;46.054;0;167.7;36:57.843;12:22:26.173;0:27.106;0:44.279;0:46.054;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;19;1:57.599;0;;27.231;0;44.486;0;45.882;0;167.5;38:55.442;12:24:23.772;0:27.231;0:44.486;0:45.882;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;20;1:57.733;0;;27.166;0;44.704;0;45.863;0;167.3;40:53.175;12:26:21.505;0:27.166;0:44.704;0:45.863;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;21;1:57.592;0;;27.197;0;44.416;0;45.979;0;167.5;42:50.767;12:28:19.097;0:27.197;0:44.416;0:45.979;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;2;22;2:56.359;0;B;27.384;0;44.540;0;1:44.435;0;111.7;45:47.126;12:31:15.456;0:27.384;0:44.540;1:44.435;;Dillon Machavern;;GS;;Turner Motorsport;BMW;GF;
95;1;23;2:13.942;0;;41.418;0;46.185;0;46.339;0;147.1;48:01.068;12:33:29.398;0:41.418;0:46.185;0:46.339;;Francis Selldorff;0:01:17.289;GS;;Turner Motorsport;BMW;GF;
95;1;24;1:57.620;0;;27.341;0;44.719;0;45.560;1;167.5;49:58.688;12:35:27.018;0:27.341;0:44.719;0:45.560;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;25;1:56.908;0;;27.186;0;43.877;2;45.845;0;168.5;51:55.596;12:37:23.926;0:27.186;0:43.877;0:45.845;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;26;1:58.309;0;;27.432;0;44.528;0;46.349;0;166.5;53:53.905;12:39:22.235;0:27.432;0:44.528;0:46.349;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;27;2:13.410;0;;27.077;0;47.704;0;58.629;0;147.7;56:07.315;12:41:35.645;0:27.077;0:47.704;0:58.629;;Francis Selldorff;;GS;;Turner Motorsport;BMW;FCY;
95;1;28;3:01.105;0;B;36.493;0;53.863;0;1:30.749;0;108.8;59:08.420;12:44:36.750;0:36.493;0:53.863;1:30.749;;Francis Selldorff;;GS;;Turner Motorsport;BMW;FCY;
95;1;29;2:43.418;0;;41.835;0;57.545;0;1:04.038;0;120.5;1:01:51.838;12:47:20.168;0:41.835;0:57.545;1:04.038;;Francis Selldorff;0:00:46.096;GS;;Turner Motorsport;BMW;FCY;
95;1;30;2:50.050;0;;35.559;0;59.619;0;1:14.872;0;115.8;1:04:41.888;12:50:10.218;0:35.559;0:59.619;1:14.872;;Francis Selldorff;;GS;;Turner Motorsport;BMW;FCY;
95;1;31;3:02.273;0;;52.560;0;1:13.388;0;56.325;0;108.1;1:07:44.161;12:53:12.491;0:52.560;1:13.388;0:56.325;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;32;1:57.938;0;;27.431;0;44.693;0;45.814;0;167.0;1:09:42.099;12:55:10.429;0:27.431;0:44.693;0:45.814;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;33;1:58.503;0;;27.053;0;44.941;0;46.509;0;166.2;1:11:40.602;12:57:08.932;0:27.053;0:44.941;0:46.509;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;34;1:57.523;0;;27.161;0;44.582;0;45.780;0;167.6;1:13:38.125;12:59:06.455;0:27.161;0:44.582;0:45.780;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;35;1:58.002;0;;27.210;0;45.169;0;45.623;0;166.9;1:15:36.127;13:01:04.457;0:27.210;0:45.169;0:45.623;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;36;1:57.063;0;;27.109;0;44.124;0;45.830;0;168.3;1:17:33.190;13:03:01.520;0:27.109;0:44.124;0:45.830;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;37;1:57.046;0;;27.168;0;44.130;0;45.748;0;168.3;1:19:30.236;13:04:58.566;0:27.168;0:44.130;0:45.748;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;38;1:56.982;0;;27.036;2;44.152;0;45.794;0;168.4;1:21:27.218;13:06:55.548;0:27.036;0:44.152;0:45.794;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;39;1:57.330;0;;27.224;0;44.203;0;45.903;0;167.9;1:23:24.548;13:08:52.878;0:27.224;0:44.203;0:45.903;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;40;1:57.091;0;;27.098;0;44.251;0;45.742;0;168.2;1:25:21.639;13:10:49.969;0:27.098;0:44.251;0:45.742;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;41;1:57.214;0;;27.136;0;44.286;0;45.792;0;168.1;1:27:18.853;13:12:47.183;0:27.136;0:44.286;0:45.792;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;42;1:57.697;0;;27.170;0;44.309;0;46.218;0;167.4;1:29:16.550;13:14:44.880;0:27.170;0:44.309;0:46.218;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;43;1:57.242;0;;27.148;0;44.209;0;45.885;0;168.0;1:31:13.792;13:16:42.122;0:27.148;0:44.209;0:45.885;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;44;1:57.769;0;;27.259;0;44.432;0;46.078;0;167.3;1:33:11.561;13:18:39.891;0:27.259;0:44.432;0:46.078;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;45;1:57.848;0;;27.142;0;44.752;0;45.954;0;167.2;1:35:09.409;13:20:37.739;0:27.142;0:44.752;0:45.954;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;46;1:57.582;0;;27.292;0;44.392;0;45.898;0;167.5;1:37:06.991;13:22:35.321;0:27.292;0:44.392;0:45.898;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;47;1:57.419;0;;27.183;0;44.334;0;45.902;0;167.8;1:39:04.410;13:24:32.740;0:27.183;0:44.334;0:45.902;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;48;1:57.754;0;;27.283;0;44.518;0;45.953;0;167.3;1:41:02.164;13:26:30.494;0:27.283;0:44.518;0:45.953;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;49;1:57.456;0;;27.233;0;44.309;0;45.914;0;167.7;1:42:59.620;13:28:27.950;0:27.233;0:44.309;0:45.914;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;50;1:57.542;0;;27.132;0;44.462;0;45.948;0;167.6;1:44:57.162;13:30:25.492;0:27.132;0:44.462;0:45.948;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;51;2:11.132;0;;27.214;0;45.342;0;58.576;0;150.2;1:47:08.294;13:32:36.624;0:27.214;0:45.342;0:58.576;;Francis Selldorff;;GS;;Turner Motorsport;BMW;FCY;
95;1;52;3:27.602;0;;53.771;0;1:31.056;0;1:02.775;0;94.9;1:50:35.896;13:36:04.226;0:53.771;1:31.056;1:02.775;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;53;2:01.803;0;;27.695;0;48.240;0;45.868;0;161.7;1:52:37.699;13:38:06.029;0:27.695;0:48.240;0:45.868;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;54;1:57.819;0;;27.263;0;44.375;0;46.181;0;167.2;1:54:35.518;13:40:03.848;0:27.263;0:44.375;0:46.181;;Francis Selldorff;;GS;;Turner Motorsport;BMW;GF;
95;1;55;2:27.600;0;;27.877;0;59.092;0;1:00.631;0;133.5;1:57:03.118;13:42:31.448;0:27.877;0:59.092;1:00.631;;Francis Selldorff;;GS;;Turner Motorsport;BMW;FCY;
95;1;56;3:14.254;0;;39.953;0;1:16.404;0;1:17.897;0;101.4;2:00:17.372;13:45:45.702;0:39.953;1:16.404;1:17.897;;Francis Selldorff;;GS;;Turner Motorsport;BMW;FCY;
95;1;57;3:27.720;0;;56.423;0;1:13.035;0;1:18.262;0;94.8;2:03:45.092;13:49:13.422;0:56.423;1:13.035;1:18.262;;Francis Selldorff;;GS;;Turner Motorsport;BMW;FF;
96;1;1;2:06.267;0;;32.344;0;46.849;0;47.074;0;156.0;2:06.267;11:47:34.597;0:32.344;0:46.849;0:47.074;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;2;1:59.297;0;;27.528;0;45.118;0;46.651;0;165.1;4:05.564;11:49:33.894;0:27.528;0:45.118;0:46.651;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;3;2:28.380;0;;29.291;0;59.532;0;59.557;0;132.8;6:33.944;11:52:02.274;0:29.291;0:59.532;0:59.557;;Matt Dalton;;GS;;Turner Motorsport;BMW;FCY;
96;1;4;3:04.036;0;;43.646;0;1:16.721;0;1:03.669;0;107.0;9:37.980;11:55:06.310;0:43.646;1:16.721;1:03.669;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;5;2:00.079;0;;27.606;0;45.963;0;46.510;0;164.0;11:38.059;11:57:06.389;0:27.606;0:45.963;0:46.510;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;6;1:58.877;0;;27.600;0;44.975;0;46.302;0;165.7;13:36.936;11:59:05.266;0:27.600;0:44.975;0:46.302;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;7;1:59.672;0;;27.595;0;44.957;0;47.120;0;164.6;15:36.608;12:01:04.938;0:27.595;0:44.957;0:47.120;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;8;1:59.659;0;;28.146;0;44.738;0;46.775;0;164.6;17:36.267;12:03:04.597;0:28.146;0:44.738;0:46.775;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;9;1:58.420;0;;27.498;0;44.784;0;46.138;1;166.3;19:34.687;12:05:03.017;0:27.498;0:44.784;0:46.138;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;10;1:59.729;0;;27.577;0;45.225;0;46.927;0;164.5;21:34.416;12:07:02.746;0:27.577;0:45.225;0:46.927;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;11;2:01.669;0;;29.837;0;45.029;0;46.803;0;161.9;23:36.085;12:09:04.415;0:29.837;0:45.029;0:46.803;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;12;1:59.848;0;;27.467;0;45.553;0;46.828;0;164.4;25:35.933;12:11:04.263;0:27.467;0:45.553;0:46.828;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;13;1:59.247;0;;27.269;1;45.363;0;46.615;0;165.2;27:35.180;12:13:03.510;0:27.269;0:45.363;0:46.615;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;14;1:59.009;0;;27.677;0;45.011;0;46.321;0;165.5;29:34.189;12:15:02.519;0:27.677;0:45.011;0:46.321;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;15;2:00.804;0;;27.478;0;46.227;0;47.099;0;163.1;31:34.993;12:17:03.323;0:27.478;0:46.227;0:47.099;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;16;1:59.406;0;;27.399;0;45.366;0;46.641;0;165.0;33:34.399;12:19:02.729;0:27.399;0:45.366;0:46.641;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;17;1:58.682;0;;27.400;0;44.859;0;46.423;0;166.0;35:33.081;12:21:01.411;0:27.400;0:44.859;0:46.423;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;18;1:58.344;0;;27.369;0;44.656;1;46.319;0;166.5;37:31.425;12:22:59.755;0:27.369;0:44.656;0:46.319;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;19;1:59.347;0;;27.313;0;45.476;0;46.558;0;165.1;39:30.772;12:24:59.102;0:27.313;0:45.476;0:46.558;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;1;20;2:59.963;0;B;27.493;0;45.005;0;1:47.465;0;109.5;42:30.735;12:27:59.065;0:27.493;0:45.005;1:47.465;;Matt Dalton;;GS;;Turner Motorsport;BMW;GF;
96;2;21;2:13.095;0;;40.934;0;45.811;0;46.350;0;148.0;44:43.830;12:30:12.160;0:40.934;0:45.811;0:46.350;;Patrick Gallagher;0:01:15.917;GS;;Turner Motorsport;BMW;GF;
96;2;22;1:57.183;0;;27.459;0;44.170;0;45.554;0;168.1;46:41.013;12:32:09.343;0:27.459;0:44.170;0:45.554;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;23;1:56.330;2;;27.129;0;43.866;2;45.335;2;169.3;48:37.343;12:34:05.673;0:27.129;0:43.866;0:45.335;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;24;1:56.822;0;;27.217;0;43.902;0;45.703;0;168.6;50:34.165;12:36:02.495;0:27.217;0:43.902;0:45.703;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;25;1:57.150;0;;27.174;0;44.016;0;45.960;0;168.1;52:31.315;12:37:59.645;0:27.174;0:44.016;0:45.960;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;26;1:57.019;0;;27.289;0;43.943;0;45.787;0;168.3;54:28.334;12:39:56.664;0:27.289;0:43.943;0:45.787;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;27;2:07.710;0;;29.044;0;49.319;0;49.347;0;154.2;56:36.044;12:42:04.374;0:29.044;0:49.319;0:49.347;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;FCY;
96;2;28;2:42.193;0;B;29.037;0;47.459;0;1:25.697;0;121.5;59:18.237;12:44:46.567;0:29.037;0:47.459;1:25.697;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;FCY;
96;2;29;2:38.916;0;;41.360;0;51.607;0;1:05.949;0;124.0;1:01:57.153;12:47:25.483;0:41.360;0:51.607;1:05.949;;Patrick Gallagher;0:00:48.375;GS;;Turner Motorsport;BMW;FCY;
96;2;30;2:47.742;0;;35.284;0;57.826;0;1:14.632;0;117.4;1:04:44.895;12:50:13.225;0:35.284;0:57.826;1:14.632;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;FCY;
96;2;31;3:00.757;0;;52.941;0;1:13.674;0;54.142;0;109.0;1:07:45.652;12:53:13.982;0:52.941;1:13.674;0:54.142;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;32;1:57.836;0;;27.205;0;44.997;0;45.634;0;167.2;1:09:43.488;12:55:11.818;0:27.205;0:44.997;0:45.634;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;33;1:58.106;0;;27.017;2;44.759;0;46.330;0;166.8;1:11:41.594;12:57:09.924;0:27.017;0:44.759;0:46.330;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;34;1:57.452;0;;27.134;0;44.503;0;45.815;0;167.7;1:13:39.046;12:59:07.376;0:27.134;0:44.503;0:45.815;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;35;1:57.617;0;;27.074;0;44.601;0;45.942;0;167.5;1:15:36.663;13:01:04.993;0:27.074;0:44.601;0:45.942;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;36;1:57.081;0;;27.089;0;44.118;0;45.874;0;168.2;1:17:33.744;13:03:02.074;0:27.089;0:44.118;0:45.874;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;37;1:57.021;0;;27.075;0;44.232;0;45.714;0;168.3;1:19:30.765;13:04:59.095;0:27.075;0:44.232;0:45.714;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;38;1:57.154;0;;27.106;0;44.206;0;45.842;0;168.1;1:21:27.919;13:06:56.249;0:27.106;0:44.206;0:45.842;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;39;1:57.429;0;;27.212;0;44.274;0;45.943;0;167.7;1:23:25.348;13:08:53.678;0:27.212;0:44.274;0:45.943;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;40;1:57.329;0;;27.137;0;44.441;0;45.751;0;167.9;1:25:22.677;13:10:51.007;0:27.137;0:44.441;0:45.751;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;41;1:57.384;0;;27.211;0;44.355;0;45.818;0;167.8;1:27:20.061;13:12:48.391;0:27.211;0:44.355;0:45.818;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;42;1:57.673;0;;27.246;0;44.332;0;46.095;0;167.4;1:29:17.734;13:14:46.064;0:27.246;0:44.332;0:46.095;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;43;1:57.472;0;;27.202;0;44.339;0;45.931;0;167.7;1:31:15.206;13:16:43.536;0:27.202;0:44.339;0:45.931;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;44;1:57.930;0;;27.202;0;44.714;0;46.014;0;167.0;1:33:13.136;13:18:41.466;0:27.202;0:44.714;0:46.014;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;45;1:57.973;0;;27.234;0;44.609;0;46.130;0;167.0;1:35:11.109;13:20:39.439;0:27.234;0:44.609;0:46.130;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;46;1:57.315;0;;27.215;0;44.238;0;45.862;0;167.9;1:37:08.424;13:22:36.754;0:27.215;0:44.238;0:45.862;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;47;1:57.637;0;;27.215;0;44.479;0;45.943;0;167.5;1:39:06.061;13:24:34.391;0:27.215;0:44.479;0:45.943;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;48;1:57.573;0;;27.218;0;44.336;0;46.019;0;167.5;1:41:03.634;13:26:31.964;0:27.218;0:44.336;0:46.019;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;49;1:58.426;0;;27.314;0;44.524;0;46.588;0;166.3;1:43:02.060;13:28:30.390;0:27.314;0:44.524;0:46.588;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;50;1:57.969;0;;27.494;0;44.472;0;46.003;0;167.0;1:45:00.029;13:30:28.359;0:27.494;0:44.472;0:46.003;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;51;2:11.774;0;;27.214;0;46.824;0;57.736;0;149.5;1:47:11.803;13:32:40.133;0:27.214;0:46.824;0:57.736;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;FCY;
96;2;52;3:24.880;0;;53.126;0;1:30.684;0;1:01.070;0;96.1;1:50:36.683;13:36:05.013;0:53.126;1:30.684;1:01.070;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;53;1:59.745;0;;27.289;0;46.561;0;45.895;0;164.5;1:52:36.428;13:38:04.758;0:27.289;0:46.561;0:45.895;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;54;1:57.616;0;;27.137;0;44.505;0;45.974;0;167.5;1:54:34.044;13:40:02.374;0:27.137;0:44.505;0:45.974;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;GF;
96;2;55;2:02.907;0;;27.118;0;44.361;0;51.428;0;160.3;1:56:36.951;13:42:05.281;0:27.118;0:44.361;0:51.428;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;FCY;
96;2;56;3:23.779;0;;47.948;0;1:19.233;0;1:16.598;0;96.7;2:00:00.730;13:45:29.060;0:47.948;1:19.233;1:16.598;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;FCY;
96;2;57;3:26.637;0;;52.371;0;1:16.761;0;1:17.505;0;95.3;2:03:27.367;13:48:55.697;0:52.371;1:16.761;1:17.505;;Patrick Gallagher;;GS;;Turner Motorsport;BMW;FF;
98;2;1;2:12.874;0;;39.037;0;45.724;0;48.113;0;148.2;2:12.874;11:47:41.204;0:39.037;0:45.724;0:48.113;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;2;2:00.060;0;;28.164;0;45.716;0;46.180;1;164.1;4:12.934;11:49:41.264;0:28.164;0:45.716;0:46.180;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
98;2;3;2:32.353;0;;31.796;0;1:02.215;0;58.342;0;129.3;6:45.287;11:52:13.617;0:31.796;1:02.215;0:58.342;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
98;2;4;2:57.723;0;;41.832;0;1:16.322;0;59.569;0;110.8;9:43.010;11:55:11.340;0:41.832;1:16.322;0:59.569;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;5;2:00.702;0;;28.134;0;46.156;0;46.412;0;163.2;11:43.712;11:57:12.042;0:28.134;0:46.156;0:46.412;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;6;2:00.436;0;;29.039;0;45.087;0;46.310;0;163.6;13:44.148;11:59:12.478;0:29.039;0:45.087;0:46.310;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;7;1:59.162;0;;27.605;0;45.303;0;46.254;0;165.3;15:43.310;12:01:11.640;0:27.605;0:45.303;0:46.254;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;8;1:58.385;0;;27.587;1;44.353;0;46.445;0;166.4;17:41.695;12:03:10.025;0:27.587;0:44.353;0:46.445;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;9;1:58.909;0;;27.629;0;44.621;0;46.659;0;165.7;19:40.604;12:05:08.934;0:27.629;0:44.621;0:46.659;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;10;1:59.073;0;;27.641;0;45.067;0;46.365;0;165.4;21:39.677;12:07:08.007;0:27.641;0:45.067;0:46.365;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;11;1:58.979;0;;27.782;0;44.620;0;46.577;0;165.6;23:38.656;12:09:06.986;0:27.782;0:44.620;0:46.577;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;12;1:58.839;0;;27.719;0;44.675;0;46.445;0;165.8;25:37.495;12:11:05.825;0:27.719;0:44.675;0:46.445;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;13;1:58.733;0;;27.732;0;44.678;0;46.323;0;165.9;27:36.228;12:13:04.558;0:27.732;0:44.678;0:46.323;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;14;1:58.686;0;;27.731;0;44.545;0;46.410;0;166.0;29:34.914;12:15:03.244;0:27.731;0:44.545;0:46.410;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;15;1:59.474;0;;27.701;0;45.124;0;46.649;0;164.9;31:34.388;12:17:02.718;0:27.701;0:45.124;0:46.649;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;16;1:58.971;0;;27.743;0;44.882;0;46.346;0;165.6;33:33.359;12:19:01.689;0:27.743;0:44.882;0:46.346;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;17;1:58.551;0;;27.727;0;44.498;0;46.326;0;166.2;35:31.910;12:21:00.240;0:27.727;0:44.498;0:46.326;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;18;1:58.683;0;;27.692;0;44.662;0;46.329;0;166.0;37:30.593;12:22:58.923;0:27.692;0:44.662;0:46.329;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;19;1:59.546;0;;27.857;0;45.415;0;46.274;0;164.8;39:30.139;12:24:58.469;0:27.857;0:45.415;0:46.274;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;20;1:58.384;0;;27.725;0;44.290;1;46.369;0;166.4;41:28.523;12:26:56.853;0:27.725;0:44.290;0:46.369;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;21;1:59.347;0;;27.676;0;44.574;0;47.097;0;165.1;43:27.870;12:28:56.200;0:27.676;0:44.574;0:47.097;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;22;1:58.639;0;;27.771;0;44.477;0;46.391;0;166.0;45:26.509;12:30:54.839;0:27.771;0:44.477;0:46.391;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;23;1:58.591;0;;27.631;0;44.450;0;46.510;0;166.1;47:25.100;12:32:53.430;0:27.631;0:44.450;0:46.510;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;24;1:58.538;0;;27.701;0;44.477;0;46.360;0;166.2;49:23.638;12:34:51.968;0:27.701;0:44.477;0:46.360;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;25;1:58.658;0;;27.802;0;44.483;0;46.373;0;166.0;51:22.296;12:36:50.626;0:27.802;0:44.483;0:46.373;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;26;1:58.352;0;;27.668;0;44.449;0;46.235;0;166.4;53:20.648;12:38:48.978;0:27.668;0:44.449;0:46.235;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;2;27;2:03.930;0;;27.757;0;45.527;0;50.646;0;158.9;55:24.578;12:40:52.908;0:27.757;0:45.527;0:50.646;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
98;2;28;3:10.408;0;;30.295;0;1:15.662;0;1:24.451;0;103.5;58:34.986;12:44:03.316;0:30.295;1:15.662;1:24.451;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
98;2;29;3:13.603;0;B;45.109;0;1:10.241;0;1:18.253;0;101.7;1:01:48.589;12:47:16.919;0:45.109;1:10.241;1:18.253;;Parker Chase;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
98;1;30;3:28.692;0;;1:40.643;0;53.794;0;54.255;0;94.4;1:05:17.281;12:50:45.611;1:40.643;0:53.794;0:54.255;;Harry Gottsacker;0:01:26.085;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
98;1;31;2:38.809;0;;37.098;0;1:09.698;0;52.013;0;124.0;1:07:56.090;12:53:24.420;0:37.098;1:09.698;0:52.013;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;32;1:58.793;0;;28.408;0;44.580;0;45.805;2;165.8;1:09:54.883;12:55:23.213;0:28.408;0:44.580;0:45.805;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;33;1:58.727;0;;27.445;2;45.393;0;45.889;0;165.9;1:11:53.610;12:57:21.940;0:27.445;0:45.393;0:45.889;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;34;1:58.953;0;;27.487;0;44.366;0;47.100;0;165.6;1:13:52.563;12:59:20.893;0:27.487;0:44.366;0:47.100;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;35;1:59.316;0;;27.510;0;44.493;0;47.313;0;165.1;1:15:51.879;13:01:20.209;0:27.510;0:44.493;0:47.313;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;36;1:58.282;0;;27.665;0;44.473;0;46.144;0;166.5;1:17:50.161;13:03:18.491;0:27.665;0:44.473;0:46.144;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;37;1:58.208;2;;27.687;0;44.223;2;46.298;0;166.6;1:19:48.369;13:05:16.699;0:27.687;0:44.223;0:46.298;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;38;1:58.387;0;;27.682;0;44.459;0;46.246;0;166.4;1:21:46.756;13:07:15.086;0:27.682;0:44.459;0:46.246;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;39;1:58.663;0;;27.719;0;44.564;0;46.380;0;166.0;1:23:45.419;13:09:13.749;0:27.719;0:44.564;0:46.380;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;40;1:58.843;0;;27.735;0;44.703;0;46.405;0;165.8;1:25:44.262;13:11:12.592;0:27.735;0:44.703;0:46.405;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;41;1:58.697;0;;27.736;0;44.578;0;46.383;0;166.0;1:27:42.959;13:13:11.289;0:27.736;0:44.578;0:46.383;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;42;1:59.407;0;;27.708;0;44.663;0;47.036;0;165.0;1:29:42.366;13:15:10.696;0:27.708;0:44.663;0:47.036;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;43;1:58.796;0;;27.777;0;44.570;0;46.449;0;165.8;1:31:41.162;13:17:09.492;0:27.777;0:44.570;0:46.449;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;44;1:59.034;0;;27.846;0;44.694;0;46.494;0;165.5;1:33:40.196;13:19:08.526;0:27.846;0:44.694;0:46.494;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;45;1:59.199;0;;27.817;0;44.795;0;46.587;0;165.3;1:35:39.395;13:21:07.725;0:27.817;0:44.795;0:46.587;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;46;1:58.947;0;;27.792;0;44.637;0;46.518;0;165.6;1:37:38.342;13:23:06.672;0:27.792;0:44.637;0:46.518;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;47;1:59.091;0;;27.856;0;44.588;0;46.647;0;165.4;1:39:37.433;13:25:05.763;0:27.856;0:44.588;0:46.647;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;48;1:59.203;0;;27.815;0;44.778;0;46.610;0;165.3;1:41:36.636;13:27:04.966;0:27.815;0:44.778;0:46.610;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;49;1:59.159;0;;27.969;0;44.638;0;46.552;0;165.3;1:43:35.795;13:29:04.125;0:27.969;0:44.638;0:46.552;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;50;1:59.292;0;;27.845;0;44.691;0;46.756;0;165.1;1:45:35.087;13:31:03.417;0:27.845;0:44.691;0:46.756;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;51;2:11.142;0;;31.596;0;50.399;0;49.147;0;150.2;1:47:46.229;13:33:14.559;0:31.596;0:50.399;0:49.147;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
98;1;52;2:56.971;0;;31.202;0;1:30.123;0;55.646;0;111.3;1:50:43.200;13:36:11.530;0:31.202;1:30.123;0:55.646;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;53;2:00.832;0;;27.857;0;46.256;0;46.719;0;163.0;1:52:44.032;13:38:12.362;0:27.857;0:46.256;0:46.719;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;54;2:00.043;0;;28.171;0;45.314;0;46.558;0;164.1;1:54:44.075;13:40:12.405;0:28.171;0:45.314;0:46.558;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;GF;
98;1;55;2:02.643;0;;28.083;0;44.754;0;49.806;0;160.6;1:56:46.718;13:42:15.048;0:28.083;0:44.754;0:49.806;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
98;1;56;3:22.851;0;;46.963;0;1:18.983;0;1:16.905;0;97.1;2:00:09.569;13:45:37.899;0:46.963;1:18.983;1:16.905;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FCY;
98;1;57;3:26.974;0;;54.213;0;1:14.761;0;1:18.000;0;95.2;2:03:36.543;13:49:04.873;0:54.213;1:14.761;1:18.000;;Harry Gottsacker;;TCR;;Bryan Herta Autosport with Curb Agajanian;Hyundai;FF;
99;2;1;2:12.533;0;;39.001;0;45.780;0;47.752;0;148.6;2:12.533;11:47:40.863;0:39.001;0:45.780;0:47.752;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;2;2:00.278;0;;27.894;0;45.836;0;46.548;0;163.8;4:12.811;11:49:41.141;0:27.894;0:45.836;0:46.548;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;FCY;
99;2;3;2:31.327;0;;31.069;0;1:01.503;0;58.755;0;130.2;6:44.138;11:52:12.468;0:31.069;1:01.503;0:58.755;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;FCY;
99;2;4;2:58.665;0;;42.216;0;1:16.683;0;59.766;0;110.3;9:42.803;11:55:11.133;0:42.216;1:16.683;0:59.766;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;5;2:01.148;0;;28.299;0;46.479;0;46.370;0;162.6;11:43.951;11:57:12.281;0:28.299;0:46.479;0:46.370;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;6;1:59.789;0;;28.655;0;44.955;0;46.179;0;164.4;13:43.740;11:59:12.070;0:28.655;0:44.955;0:46.179;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;7;1:59.072;0;;27.805;0;45.194;0;46.073;1;165.4;15:42.812;12:01:11.142;0:27.805;0:45.194;0:46.073;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;8;1:58.436;0;;27.646;1;44.316;0;46.474;0;166.3;17:41.248;12:03:09.578;0:27.646;0:44.316;0:46.474;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;9;1:58.596;0;;27.817;0;44.422;0;46.357;0;166.1;19:39.844;12:05:08.174;0:27.817;0:44.422;0:46.357;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;10;2:00.424;0;;28.244;0;45.726;0;46.454;0;163.6;21:40.268;12:07:08.598;0:28.244;0:45.726;0:46.454;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;11;1:59.565;0;;27.787;0;44.726;0;47.052;0;164.8;23:39.833;12:09:08.163;0:27.787;0:44.726;0:47.052;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;12;1:59.948;0;;27.990;0;45.515;0;46.443;0;164.2;25:39.781;12:11:08.111;0:27.990;0:45.515;0:46.443;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;13;1:59.379;0;;27.846;0;45.017;0;46.516;0;165.0;27:39.160;12:13:07.490;0:27.846;0:45.017;0:46.516;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;14;1:59.026;0;;28.126;0;44.508;0;46.392;0;165.5;29:38.186;12:15:06.516;0:28.126;0:44.508;0:46.392;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;15;1:59.259;0;;27.827;0;44.998;0;46.434;0;165.2;31:37.445;12:17:05.775;0:27.827;0:44.998;0:46.434;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;16;1:58.670;0;;27.784;0;44.472;0;46.414;0;166.0;33:36.115;12:19:04.445;0:27.784;0:44.472;0:46.414;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;17;1:59.198;0;;28.231;0;44.484;0;46.483;0;165.3;35:35.313;12:21:03.643;0:28.231;0:44.484;0:46.483;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;18;1:58.968;0;;27.836;0;44.593;0;46.539;0;165.6;37:34.281;12:23:02.611;0:27.836;0:44.593;0:46.539;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;19;1:58.712;0;;27.845;0;44.459;0;46.408;0;165.9;39:32.993;12:25:01.323;0:27.845;0:44.459;0:46.408;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;20;1:59.078;0;;27.925;0;44.404;0;46.749;0;165.4;41:32.071;12:27:00.401;0:27.925;0:44.404;0:46.749;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;21;1:58.737;0;;27.814;0;44.479;0;46.444;0;165.9;43:30.808;12:28:59.138;0:27.814;0:44.479;0:46.444;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;22;1:58.322;0;;27.723;0;44.237;1;46.362;0;166.5;45:29.130;12:30:57.460;0:27.723;0:44.237;0:46.362;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;23;1:58.656;0;;27.862;0;44.480;0;46.314;0;166.0;47:27.786;12:32:56.116;0:27.862;0:44.480;0:46.314;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;24;1:59.003;0;;28.121;0;44.470;0;46.412;0;165.5;49:26.789;12:34:55.119;0:28.121;0:44.470;0:46.412;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;25;1:58.679;0;;27.847;0;44.444;0;46.388;0;166.0;51:25.468;12:36:53.798;0:27.847;0:44.444;0:46.388;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;26;1:58.919;0;;27.851;0;44.533;0;46.535;0;165.6;53:24.387;12:38:52.717;0:27.851;0:44.533;0:46.535;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;2;27;2:01.899;0;;27.726;0;44.906;0;49.267;0;161.6;55:26.286;12:40:54.616;0:27.726;0:44.906;0:49.267;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;FCY;
99;2;28;3:09.287;0;;29.910;0;1:15.074;0;1:24.303;0;104.1;58:35.573;12:44:03.903;0:29.910;1:15.074;1:24.303;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;FCY;
99;2;29;3:13.652;0;B;46.413;0;1:09.025;0;1:18.214;0;101.7;1:01:49.225;12:47:17.555;0:46.413;1:09.025;1:18.214;;Eric Powell;;TCR;;Victor Gonzalez Racing Team;Hyundai;FCY;
99;1;30;3:25.039;0;;1:36.621;0;53.521;0;54.897;0;96.1;1:05:14.264;12:50:42.594;1:36.621;0:53.521;0:54.897;;Tyler Gonzalez;0:01:21.201;TCR;;Victor Gonzalez Racing Team;Hyundai;FCY;
99;1;31;2:40.089;0;;36.850;0;1:10.363;0;52.876;0;123.0;1:07:54.353;12:53:22.683;0:36.850;1:10.363;0:52.876;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;32;1:57.455;0;;27.608;0;44.116;0;45.731;2;167.7;1:09:51.808;12:55:20.138;0:27.608;0:44.116;0:45.731;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;33;1:57.413;2;;27.555;2;43.836;2;46.022;0;167.8;1:11:49.221;12:57:17.551;0:27.555;0:43.836;0:46.022;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;34;1:58.412;0;;27.636;0;44.500;0;46.276;0;166.4;1:13:47.633;12:59:15.963;0:27.636;0:44.500;0:46.276;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;35;1:58.912;0;;28.241;0;44.550;0;46.121;0;165.7;1:15:46.545;13:01:14.875;0:28.241;0:44.550;0:46.121;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;36;1:58.524;0;;27.624;0;44.602;0;46.298;0;166.2;1:17:45.069;13:03:13.399;0:27.624;0:44.602;0:46.298;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;37;1:58.643;0;;27.616;0;44.723;0;46.304;0;166.0;1:19:43.712;13:05:12.042;0:27.616;0:44.723;0:46.304;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;38;1:58.785;0;;27.610;0;44.893;0;46.282;0;165.8;1:21:42.497;13:07:10.827;0:27.610;0:44.893;0:46.282;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;39;1:59.808;0;;27.668;0;45.675;0;46.465;0;164.4;1:23:42.305;13:09:10.635;0:27.668;0:45.675;0:46.465;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;40;1:58.438;0;;27.651;0;44.173;0;46.614;0;166.3;1:25:40.743;13:11:09.073;0:27.651;0:44.173;0:46.614;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;41;1:59.423;0;;27.616;0;44.902;0;46.905;0;164.9;1:27:40.166;13:13:08.496;0:27.616;0:44.902;0:46.905;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;42;1:59.821;0;;27.800;0;45.507;0;46.514;0;164.4;1:29:39.987;13:15:08.317;0:27.800;0:45.507;0:46.514;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;43;1:58.466;0;;27.728;0;44.358;0;46.380;0;166.3;1:31:38.453;13:17:06.783;0:27.728;0:44.358;0:46.380;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;44;1:58.962;0;;27.706;0;44.721;0;46.535;0;165.6;1:33:37.415;13:19:05.745;0:27.706;0:44.721;0:46.535;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;45;1:58.713;0;;27.785;0;44.505;0;46.423;0;165.9;1:35:36.128;13:21:04.458;0:27.785;0:44.505;0:46.423;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;46;1:58.739;0;;27.735;0;44.487;0;46.517;0;165.9;1:37:34.867;13:23:03.197;0:27.735;0:44.487;0:46.517;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;47;1:58.727;0;;27.778;0;44.399;0;46.550;0;165.9;1:39:33.594;13:25:01.924;0:27.778;0:44.399;0:46.550;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;48;1:58.673;0;;27.858;0;44.440;0;46.375;0;166.0;1:41:32.267;13:27:00.597;0:27.858;0:44.440;0:46.375;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;49;1:58.699;0;;27.781;0;44.380;0;46.538;0;166.0;1:43:30.966;13:28:59.296;0:27.781;0:44.380;0:46.538;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;50;1:58.880;0;;27.687;0;44.639;0;46.554;0;165.7;1:45:29.846;13:30:58.176;0:27.687;0:44.639;0:46.554;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;51;2:03.925;0;;28.970;0;47.174;0;47.781;0;159.0;1:47:33.771;13:33:02.101;0:28.970;0:47.174;0:47.781;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;FCY;
99;1;52;3:09.172;0;;39.033;0;1:33.910;0;56.229;0;104.1;1:50:42.943;13:36:11.273;0:39.033;1:33.910;0:56.229;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;53;2:00.275;0;;27.901;0;46.116;0;46.258;0;163.8;1:52:43.218;13:38:11.548;0:27.901;0:46.116;0:46.258;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;54;1:58.718;0;;27.808;0;44.413;0;46.497;0;165.9;1:54:41.936;13:40:10.266;0:27.808;0:44.413;0:46.497;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;GF;
99;1;55;2:01.949;0;;27.754;0;44.700;0;49.495;0;161.5;1:56:43.885;13:42:12.215;0:27.754;0:44.700;0:49.495;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;FCY;
99;1;56;3:24.681;0;;48.248;0;1:19.643;0;1:16.790;0;96.2;2:00:08.566;13:45:36.896;0:48.248;1:19.643;1:16.790;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;FCY;
99;1;57;3:26.553;0;;53.826;0;1:15.410;0;1:17.317;0;95.4;2:03:35.119;13:49:03.449;0:53.826;1:15.410;1:17.317;;Tyler Gonzalez;;TCR;;Victor Gonzalez Racing Team;Hyundai;FF;
```

agents/visualizer/main.py
```py
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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# AI helpers
from agents.common import ai_helpers
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

def _parse_pubsub_push(req_json: Dict[str, Any]) -> Dict[str, Any]:
    """Decodes the data field from a Pub/Sub push message."""
    if "message" not in req_json or "data" not in req_json["message"]:
        raise ValueError("Invalid Pub/Sub push payload")
    decoded = base64.b64decode(req_json["message"]["data"]).decode("utf-8")
    return json.loads(decoded)

# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------
app = Flask(__name__)


@app.route("/", methods=["POST"])
def handle_request():
    try:
        req_json = request.get_json(force=True, silent=True)
        if req_json is None:
            return jsonify({"error": "invalid_json"}), 400
        
        message_data = _parse_pubsub_push(req_json)
        analysis_uri: str | None = message_data.get("analysis_path")
        insights_uri: str | None = message_data.get("insights_path")
        if not analysis_uri or not insights_uri:
            return jsonify({"error": "missing_fields"}), 400
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
            return jsonify({"error": "internal_error"}), 500
    return jsonify({"visuals_prefix": f"gs://{ANALYZED_DATA_BUCKET}/{basename}/visuals/"}), 200
```

agents/visualizer/requirements.txt
```txt
Flask>=2.3.0
gunicorn>=21.2.0
google-cloud-storage>=2.16.0
matplotlib>=3.8.0
seaborn>=0.13.0
pandas>=2.2.0
numpy>=1.26.0
```

cloudbuild.yaml
```yaml
# Master Cloud Build configuration for all Project Apex agents.

steps:
# Define the build step for each agent. They can all run in parallel.
# Each step calls the same universal Dockerfile but passes a different
# --build-arg to customize it for the specific agent.

- name: 'gcr.io/cloud-builders/docker'
  id: 'build-ui-portal'
  args:
  - 'build'
  - '--tag'
  - '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/ui-portal:latest'
  - '--build-arg'
  - 'AGENT_NAME=ui_portal'
  - '.' # Build context is the project root

- name: 'gcr.io/cloud-builders/docker'
  id: 'build-core-analyzer'
  args: ['build', '-t', '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/core-analyzer:latest', '--build-arg', 'AGENT_NAME=core_analyzer', '.']

- name: 'gcr.io/cloud-builders/docker'
  id: 'build-adk-orchestrator'
  args: ['build', '-t', '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/adk-orchestrator:latest', '--build-arg', 'AGENT_NAME=adk_orchestrator', '.']

- name: 'gcr.io/cloud-builders/docker'
  id: 'build-insight-hunter'
  args: ['build', '-t', '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/insight-hunter:latest', '--build-arg', 'AGENT_NAME=insight_hunter', '.']

- name: 'gcr.io/cloud-builders/docker'
  id: 'build-historian'
  args: ['build', '-t', '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/historian:latest', '--build-arg', 'AGENT_NAME=historian', '.']

- name: 'gcr.io/cloud-builders/docker'
  id: 'build-visualizer'
  args: ['build', '-t', '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/visualizer:latest', '--build-arg', 'AGENT_NAME=visualizer', '.']

- name: 'gcr.io/cloud-builders/docker'
  id: 'build-scribe'
  args: ['build', '-t', '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/scribe:latest', '--build-arg', 'AGENT_NAME=scribe', '.']

- name: 'gcr.io/cloud-builders/docker'
  id: 'build-publicist'
  args: ['build', '-t', '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/publicist:latest', '--build-arg', 'AGENT_NAME=publicist', '.']
  
- name: 'gcr.io/cloud-builders/docker'
  id: 'build-arbiter'
  args: ['build', '-t', '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/arbiter:latest', '--build-arg', 'AGENT_NAME=arbiter', '.']

# --- Push all images to the registry in parallel ---
images:
- '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/adk-orchestrator:latest'
- '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/core-analyzer:latest'
- '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/insight-hunter:latest'
# ... add all other agent image names here ...
- '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/historian:latest'
- '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/visualizer:latest'
- '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/scribe:latest'
- '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/publicist:latest'
- '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/arbiter:latest'
- '${_GCP_REGION}-docker.pkg.dev/${PROJECT_ID}/${_AR_REPO}/ui-portal:latest'
```

docker-compose.yml
```yml
version: '3.8'

services:
  pubsub-emulator:
    image: gcr.io/google.com/cloudsdktool/cloud-sdk:emulators
    command: gcloud beta emulators pubsub start --project=local-dev --host-port=0.0.0.0:8085
    ports:
      - "8085:8085"

  gcs-emulator:
    image: fsouza/fake-gcs-server:latest
    command: -scheme http -port 4443 -public-host localhost
    ports:
      - "4443:4443"

  core-analyzer:
    build:
      context: .
      args:
        AGENT_NAME: core_analyzer
    ports:
      - "8080:8080"
    environment:
      - PORT=8080
      - STORAGE_EMULATOR_HOST=http://gcs-emulator:4443
      - GOOGLE_APPLICATION_CREDENTIALS=
      - GOOGLE_CLOUD_PROJECT=local-dev
      - ANALYZED_DATA_BUCKET=imsa-analyzed-data-project-apex-v1
    depends_on:
      - gcs-emulator

  insight-hunter:
    build:
      context: .
      args:
        AGENT_NAME: insight_hunter
    ports:
      - "8081:8080"
    environment:
      - PORT=8080
      - STORAGE_EMULATOR_HOST=http://gcs-emulator:4443
      - PUBSUB_EMULATOR_HOST=pubsub-emulator:8085
      - GOOGLE_APPLICATION_CREDENTIALS=
      - GOOGLE_CLOUD_PROJECT=local-dev
      - USE_AI_ENHANCED=false
    depends_on:
      - gcs-emulator
      - pubsub-emulator

  historian:
    build: { context: ., args: { AGENT_NAME: historian } }
    ports: [ "8082:8080" ]
    environment:
      - PORT=8080
      - STORAGE_EMULATOR_HOST=http://gcs-emulator:4443
      - PUBSUB_EMULATOR_HOST=pubsub-emulator:8085
      - GOOGLE_APPLICATION_CREDENTIALS=
      - GOOGLE_CLOUD_PROJECT=local-dev
      - USE_AI_ENHANCED=false
    depends_on: [ gcs-emulator, pubsub-emulator ]
  
  visualizer:
    build: { context: ., args: { AGENT_NAME: visualizer } }
    ports: [ "8083:8080" ]
    environment:
      - PORT=8080
      - STORAGE_EMULATOR_HOST=http://gcs-emulator:4443
      - PUBSUB_EMULATOR_HOST=pubsub-emulator:8085
      - GOOGLE_APPLICATION_CREDENTIALS=
      - GOOGLE_CLOUD_PROJECT=local-dev
      - USE_AI_ENHANCED=false
    depends_on: [ gcs-emulator, pubsub-emulator ]

  scribe:
    build: { context: ., args: { AGENT_NAME: scribe } }
    ports: [ "8084:8080" ]
    environment:
      - PORT=8080
      - STORAGE_EMULATOR_HOST=http://gcs-emulator:4443
      - GOOGLE_APPLICATION_CREDENTIALS=
      - GOOGLE_CLOUD_PROJECT=local-dev
      - USE_AI_ENHANCED=false
    depends_on: [ gcs-emulator ]

  publicist:
    build: { context: ., args: { AGENT_NAME: publicist } }
    ports: [ "8086:8080" ]
    environment:
      - PORT=8080
      - STORAGE_EMULATOR_HOST=http://gcs-emulator:4443
      - GOOGLE_APPLICATION_CREDENTIALS=
      - GOOGLE_CLOUD_PROJECT=local-dev
      - USE_AI_ENHANCED=false
    depends_on: [ gcs-emulator ]
```

Dockerfile
```Dockerfile
# This is a universal, multi-stage Dockerfile for all Project Apex agents.
# It is designed to be built from the project root.

# --- Builder Stage ---
# Use a full python image that has all necessary build tools.
FROM python:3.11-slim as builder
WORKDIR /app
COPY . .

# First, install git, which is needed by the ADK requirements.
RUN apt-get update && apt-get install -y git

# Install all Python packages for all agents into a shared layer.
# This is efficient because Cloud Build will cache this layer.
# We first combine all requirements into one file.
RUN cat agents/*/requirements.txt > all_requirements.txt && \
    pip install --no-cache-dir --prefix="/install" -r all_requirements.txt && \
    pip install --no-cache-dir --prefix="/install" -e .

# --- Final Stage ---
# Start from a clean, lightweight image.
FROM python:3.11-slim
ARG AGENT_NAME

ENV PYTHONUNBUFFERED=true
WORKDIR /app/agents/${AGENT_NAME}

# Copy the pre-built dependencies from the 'builder' stage.
COPY --from=builder /install /usr/local

# Copy the entire 'agents' source code.
COPY --from=builder /app/agents /app/agents

ENV PORT 8080
CMD gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 main:app
```

dry_run_harness.py
```py
"""End-to-end dry-run harness for Project Apex.

This script exercises the AI-enhanced logic of Historian, Visualizer, Scribe and
Publicist without making external Vertex-AI calls or hitting Google Cloud
services.  It monkey-patches `agents.common.ai_helpers` with deterministic
stubbed functions so the pipeline can be executed completely offline.

Run:
    python dry_run_harness.py
"""
from __future__ import annotations

import json
import logging
import sys
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest import mock

class _MockModel:
    def __init__(self, *_, **__):
        pass

    def predict(self, _prompt: str, **_kwargs):
        # Return object with .text attribute to mimic Vertex response
        return SimpleNamespace(text="{}")

    def generate_content(self, *_, **__):
        return self.predict("")

def activate():
    """Activates all mock patches for an offline run."""
    # ---------------------------------------------------------------------------
    # Google API stubs (prevents import errors when google-cloud libs are absent)
    # ---------------------------------------------------------------------------
    import types, sys
    _dummy_google = types.ModuleType("google")
    _dummy_google_cloud = types.ModuleType("google.cloud")

    _dummy_aiplatform = types.ModuleType("google.cloud.aiplatform")
    _dummy_aiplatform.init = lambda *_, **__: None

    _dummy_vertexai = types.ModuleType("vertexai")
    _dummy_vertexai_genmodels = types.ModuleType("vertexai.generative_models")
    setattr(_dummy_vertexai_genmodels, "GenerativeModel", _MockModel)
    setattr(_dummy_vertexai_genmodels, "GenerationConfig", lambda **_: None)

    sys.modules.setdefault("google", _dummy_google)
    sys.modules.setdefault("google.cloud", _dummy_google_cloud)
    sys.modules.setdefault("google.cloud.aiplatform", _dummy_aiplatform)
    sys.modules.setdefault("vertexai", _dummy_vertexai)
    sys.modules.setdefault("vertexai.generative_models", _dummy_vertexai_genmodels)

    # ---------------------------------------------------------------------------
    # Additional dependency stubs
    # ---------------------------------------------------------------------------
    from types import ModuleType

    def _stub_module(fullname: str, attrs: Dict[str, Any] | None = None):
        mod = ModuleType(fullname)
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules.setdefault(fullname, mod)
        return mod

    # Flask minimal stub
    class _FlaskStub:
        def __init__(self, *_, **__):
            pass
        def route(self, _path: str, **__):
            def decorator(func):
                return func
            return decorator

    _stub_module("flask", {
        "Flask": _FlaskStub,
        "Response": object,
        "request": SimpleNamespace(get_json=lambda *_, **__: {}),
    })

    # Google Cloud stubs
    _stub_module("google.cloud.storage", {"Client": lambda *_, **__: None})
    _stub_module("google.cloud.bigquery", {"Client": lambda *_, **__: None, "ScalarQueryParameter": object, "QueryJobConfig": object})

    # Matplotlib / seaborn / pandas / numpy stubs
    # seaborn stub with set_theme
    seaborn_stub = _stub_module("seaborn")
    setattr(seaborn_stub, "set_theme", lambda *_, **__: None)
    setattr(seaborn_stub, "barplot", lambda *_, **__: None)

    # matplotlib
    mat_stub = _stub_module("matplotlib")
    plt_stub = _stub_module("matplotlib.pyplot")
    plt_stub.rcParams = {}
    import pathlib as _pl

    def _mock_savefig(path, *_, **__):
        try:
            p = _pl.Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            # create tiny placeholder file if not exists
            if not p.exists():
                p.write_bytes(b"PNG")
        except Exception:  # pragma: no cover
            pass

    for fname in [
        "figure",
        "bar",
        "plot",
        "close",
        "text",
        "xlabel",
        "ylabel",
        "title",
        "legend",
        "tight_layout",
    ]:
        setattr(plt_stub, fname, lambda *_, **__: None)
    setattr(plt_stub, "savefig", _mock_savefig)
    setattr(plt_stub, "gca", lambda: SimpleNamespace(invert_yaxis=lambda: None))

    # numpy & pandas – only stub if not installed
    try:
        import numpy  # type: ignore  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover
        _stub_module("numpy", {"linspace": lambda *_, **__: []})

    try:
        import pandas  # type: ignore  # noqa: F401
    except ModuleNotFoundError:  # pragma: no cover
        _stub_module("pandas", {"DataFrame": lambda *_, **__: object, "errors": SimpleNamespace(PerformanceWarning=Warning)})

    # Jinja2 & WeasyPrint stubs
    _stub_module("jinja2", {"Environment": lambda *_, **__: SimpleNamespace(get_template=lambda _name: SimpleNamespace(render=lambda **__: "<html></html>")),
                              "FileSystemLoader": lambda *_, **__: None,
                              "select_autoescape": lambda *_: None})
    class _HTMLStub:
        def __init__(self, *_, **__):
            pass
        def write_pdf(self, *args: Any, **__: Any):
            """Create a tiny placeholder PDF so downstream code sees a real file."""

# ---------------------------------------------------------------------------
# Mock AI helper functions
# ---------------------------------------------------------------------------

def _mock_summarize(prompt: str, **_: Any) -> str:  # noqa: D401
    """Return a deterministic summary irrespective of the prompt."""
    return "[MOCK SUMMARY] Key performance improved; tyre degradation in control; strategic consistency evident."


def _mock_generate_json(prompt: str, **_: Any):  # noqa: D401
    """Return canned JSON structures based on keywords in the prompt."""
    low = prompt.lower()
    if "executive summary" in low:
        return {
            "executive_summary": (
                "Despite stiff competition, the team demonstrated superior pace in key phases, "
                "leveraging lower tyre degradation to secure consistent lap times."
            ),
            "tactical_recommendations": [
                "Focus on maintaining tyre temperature in opening stints.",
                "Explore undercut opportunities during early cautions.",
                "Increase fuel-corrected pace in the final third to pressure rivals.",
            ],
        }
    if "create between 3 and 5 engaging tweets" in low or "json array of strings" in low:
        return [
            "🏁 Stellar stint pace keeps us in the hunt! 💪 #IMSA #ProjectApex",
            "Strategy pays off – minimal tyre deg and lightning pit work 🚀 #RaceDay",
            "Consistency is king; watch us climb the charts! 📈 #Motorsport",
        ]
    import re
    if "append two new keys" in low:  # insight enrichment
        match = re.search(r"Insights:\s*(\[.*?\])\s*$", prompt, re.S)
        if not match:
            return []
        try:
            insights = json.loads(match.group(1))
        except Exception:
            return []
        for ins in insights:
            if isinstance(ins, dict):
                ins["llm_commentary"] = "[MOCK COMMENTARY]"
                ins["recommended_action"] = "[MOCK ACTION]"
        return insights
    # default fallback
    return []


# ---------------------------------------------------------------------------
# Sample minimal input data used across agents
# ---------------------------------------------------------------------------

CURRENT_ANALYSIS: Dict[str, Any] = {
    "fastest_by_manufacturer": [
        {"manufacturer": "Acura", "fastest_lap": {"time": "72.345"}},
        {"manufacturer": "Porsche", "fastest_lap": {"time": "72.890"}},
    ],
    "enhanced_strategy_analysis": [
        {
            "manufacturer": "Acura",
            "deg_coeff_a": -0.002,
            "car_number": "10",
            "avg_pit_stationary_time": "31.2",
            "race_pace_consistency_stdev": 0.135,
        }
    ],
    "race_strategy_by_car": [
        {
            "car_number": "10",
            "stints": [
                {
                    "stint_number": 1,
                    "laps": [
                        {"lap_in_stint": i, "LAP_TIME_FUEL_CORRECTED_SEC": 73 + i * 0.03}
                        for i in range(1, 25)
                    ],
                }
            ],
            "tire_degradation_model": {"deg_coeff_a": 0.002, "deg_coeff_b": 0.1, "deg_coeff_c": 73.1},
        }
    ],
    "metadata": {"event_id": "2025_mido_race"},
}

HISTORICAL_ANALYSIS: Dict[str, Any] = {
    "fastest_by_manufacturer": [
        {"manufacturer": "Acura", "fastest_lap": {"time": "72.900"}},
        {"manufacturer": "Porsche", "fastest_lap": {"time": "73.100"}},
    ],
    "enhanced_strategy_analysis": [
        {"manufacturer": "Acura", "tire_degradation_model": {"deg_coeff_a": -0.0015}},
    ],
}

INSIGHTS_PLACEHOLDER: List[Dict[str, Any]] = [
    {"category": "Historical Comparison", "type": "YoY Manufacturer Pace", "details": "Acura is 0.55s faster than last year."},
    {"category": "Strategy", "type": "Pit Stop Efficiency", "details": "Average stationary time improved by 1.3s."},
]


# ---------------------------------------------------------------------------
# Harness execution
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger("dry_run")


def main() -> None:  # noqa: D401
    """Entry point – patches AI helpers then runs each agent's logic."""
    import importlib

    # Patch ai_helpers first
    import agents.common.ai_helpers as helpers  # pylint: disable=import-error

    with mock.patch.object(helpers, "summarize", _mock_summarize), mock.patch.object(
        helpers, "generate_json", _mock_generate_json
    ):
        # Historian
        historian = importlib.import_module("agents.historian.main")
        insights = historian.compare_analyses(CURRENT_ANALYSIS, HISTORICAL_ANALYSIS)
        narrative = historian._narrative_summary(insights)  # type: ignore[attr-defined]
        LOGGER.info("Historian produced %d insights and narrative: %s", len(insights), narrative)

        # Visualizer caption (for one dummy plot path)
        from pathlib import Path

        visualizer = importlib.import_module("agents.visualizer.main")
        caption = visualizer._generate_caption(Path("pit_stationary_times.png"), insights)  # type: ignore[attr-defined]
        LOGGER.info("Visualizer caption: %s", caption)

        # Scribe narrative generation
        scribe = importlib.import_module("agents.scribe.main")
        scribe_narr = scribe._generate_narrative(insights)  # type: ignore[attr-defined]
        LOGGER.info("Scribe narrative: %s", json.dumps(scribe_narr, indent=2))

        # Publicist tweets
        publicist = importlib.import_module("agents.publicist.main")
        tweets = publicist._gen_tweets(insights)
        LOGGER.info("Publicist tweets: %s", tweets)

    print("\n=== DRY RUN COMPLETE ===")
    print(f"Historian insights: {len(insights)}")
    print(f"Historian narrative: {narrative}")
    print(f"Visualizer caption: {caption}")
    print(f"Scribe summary keys: {list((scribe_narr or {}).keys())}")
    print(f"Tweets generated: {len(tweets)}")


if __name__ == "__main__":
    sys.exit(main() or 0)
```

full_pipeline_local.py
```py
"""Offline end-to-end pipeline runner for Project Apex.

Uses the sample data in `agents/test_data` (or paths provided via CLI) to run
through the entire analysis → insight → visualisation → report → social posts
workflow without needing any Google Cloud services or Vertex AI access.

It relies on the stubs & AI mocks that are defined inside `dry_run_harness.py`.

Usage:
    python full_pipeline_local.py \
        --csv agents/test_data/race.csv \
        --pits agents/test_data/fuel.json \
        --out_dir out_local
"""
from __future__ import annotations

import dotenv
dotenv.load_dotenv()

import argparse
import json
import logging
import pathlib
import shutil
import sys
import tempfile
import contextlib
from types import SimpleNamespace
from typing import List, Dict, Any, Sequence
from unittest import mock

import dry_run_harness


LOGGER = logging.getLogger("full_pipeline")



def run_pipeline(csv_path: pathlib.Path, pit_json: pathlib.Path, fuel_json: pathlib.Path | None, out_dir: pathlib.Path, live_ai: bool = False) -> None:  # noqa: D401
    """Run the entire local pipeline and write outputs into `out_dir`."""

    if not live_ai:
        dry_run_harness.activate()
    else:
        LOGGER.warning("--- RUNNING WITH LIVE VERTEX AI – THIS WILL INCUR COSTS ---")

    # Must import agent modules AFTER harness is potentially activated
    from agents.core_analyzer.imsa_analyzer import IMSADataAnalyzer  # type: ignore
    import agents.insight_hunter.main as insight_hunter
    import agents.visualizer.main as visualizer
    import agents.scribe.main as scribe
    import agents.publicist.main as publicist
    from agents.common import ai_helpers
    from dry_run_harness import _mock_generate_json, _mock_summarize

    out_dir.mkdir(parents=True, exist_ok=True)

    # Use mock context manager for AI functions if not in live mode
    mock_context = contextlib.nullcontext()
    if not live_ai:
        mock_context = mock.patch.multiple(
            ai_helpers,
            summarize=_mock_summarize,
            generate_json=_mock_generate_json,
        )

    with mock_context:
        # 1. Core analysis
        LOGGER.info("Running IMSADataAnalyzer …")
        analyzer = IMSADataAnalyzer(
            str(csv_path),
            str(pit_json),
            str(fuel_json) if fuel_json else None,
        )
        analysis_data: Dict[str, Any] = analyzer.run_all_analyses()
        (out_dir / "analysis_enhanced.json").write_text(json.dumps(analysis_data, indent=2))

        # 2. Insight Hunter
        LOGGER.info("Deriving insights …")
        insights: List[Dict[str, Any]] = insight_hunter.derive_insights(analysis_data)
        insights = insight_hunter.enrich_insights_with_ai(insights)
        (out_dir / "insights.json").write_text(json.dumps(insights, indent=2))

        # 3. Visualizer (plots + captions)
        LOGGER.info("Generating visuals …")
        visuals_info = visualizer.generate_all_visuals(analysis_data, insights, out_dir)
        captions: Dict[str, str] = {}
        for p, cap in visuals_info:
            if cap:
                captions[p.name] = cap
        if captions:
            (out_dir / "captions.json").write_text(json.dumps(captions, indent=2))

        # 4. Scribe report
        LOGGER.info("Rendering PDF report …")
        pdf_path = out_dir / "race_report.pdf"
        narrative = scribe._generate_narrative(insights, analysis_data)  # type: ignore[attr-defined]
        scribe._render_report(analysis_data, insights, narrative, pdf_path)  # type: ignore[attr-defined]

        # 5. Publicist tweets
        LOGGER.info("Composing tweets …")
        tweets = publicist._gen_tweets(insights, analysis_data)  # type: ignore[attr-defined]
        (out_dir / "tweets.json").write_text(json.dumps(tweets, indent=2))

        # 6. Token usage summary (only meaningful in live mode)
        usage_summary = ai_helpers.get_usage_summary()
        if usage_summary:
            (out_dir / "token_usage.json").write_text(json.dumps(usage_summary, indent=2))
            LOGGER.info("Token usage summary written to token_usage.json: %s", usage_summary)

    LOGGER.info("\n=== LOCAL PIPELINE COMPLETE ===")
    LOGGER.info("Outputs in %s", out_dir.resolve())
    LOGGER.info("Report: %s", pdf_path.name)
    LOGGER.info("Visuals: %s", [p.name for p, _ in visuals_info])
    LOGGER.info("Tweets: %s", len(tweets))


# ----------------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run Project Apex pipeline locally.")
    parser.add_argument(
        "--csv",
        type=pathlib.Path,
        default=pathlib.Path("agents/test_data/glen_race.csv"),
        help="Race CSV file path.",
    )
    parser.add_argument(
        "--pit_json",
        type=pathlib.Path,
        default=pathlib.Path("agents/test_data/glen_pit.json"),
        help="Pit-stop JSON file path.",
    )
    parser.add_argument(
        "--fuel_caps",
        "--pits",  # backward-compat alias
        type=pathlib.Path,
        default=pathlib.Path("agents/test_data/mido_fuel.json"),
        help="Fuel capacity JSON file path (optional).",
        nargs="?",
    )
    parser.add_argument(
        "--out_dir",
        type=pathlib.Path,
        default=pathlib.Path("out_local"),
        help="Output directory.",
    )
    parser.add_argument(
        "--live-ai",
        action="store_true",
        help="Use live Vertex AI APIs instead of mock data.",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    """Pipeline entrypoint."""
    args = parse_args(argv)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,  # override any prior logging setup so --log works reliably
    )

    if not args.csv.exists() or not args.pit_json.exists():
        sys.exit("CSV or pit JSON file not found.")

    fuel_json_path = (
        args.fuel_caps if args.fuel_caps and args.fuel_caps.exists() else None
    )

    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)

    run_pipeline(
        args.csv, args.pit_json, fuel_json_path, args.out_dir, live_ai=args.live_ai
    )


if __name__ == "__main__":
    main()
```

run_local_e2e.py
```py
"""
Local End-to-End Pipeline Runner for Project Apex.

This script uses Docker Compose to spin up all agent services and emulators,
then orchestrates a full test run by making HTTP requests and publishing
to the Pub/Sub emulator to simulate the production workflow.

Prerequisites:
- Docker and docker-compose installed.
- Python packages from `tests/requirements.txt` (e.g., `pip install google-cloud-pubsub requests`).

Usage:
    python run_local_e2e.py
"""
import os
import json
import time
import pathlib
import subprocess
import requests
from google.cloud import pubsub_v1
from google.cloud import storage
from google.auth.credentials import AnonymousCredentials

# --- Configuration ---
GCP_PROJECT = "local-dev"
PUBSUB_EMULATOR_HOST = "localhost:8085"
STORAGE_EMULATOR_HOST = "http://localhost:4443"
BUCKET_NAME = "imsa-analyzed-data-project-apex-v1"
RUN_ID = f"local-run-{int(time.time())}"

# Agent service URLs
CORE_ANALYZER_URL = "http://localhost:8080"
SCRIBE_URL = "http://localhost:8084"
PUBLICIST_URL = "http://localhost:8086"

# Pub/Sub Topics and Subscriptions
TOPICS = {
    "analysis-completed": [
        {"name": "to-insight-hunter", "endpoint": "http://insight-hunter:8080/"},
        {"name": "to-historian", "endpoint": "http://historian:8080/"},
    ],
    "visualization-requests": [
        {"name": "to-visualizer", "endpoint": "http://visualizer:8080/"}
    ],
}

def setup_emulators():
    """Create GCS bucket and Pub/Sub topics/subscriptions."""
    print("--- Setting up GCS and Pub/Sub emulators ---")
    # GCS
    storage_client = storage.Client(project=GCP_PROJECT, credentials=AnonymousCredentials())
    if not storage_client.lookup_bucket(BUCKET_NAME):
        storage_client.create_bucket(BUCKET_NAME)
    print(f"Bucket '{BUCKET_NAME}' is ready.")

    # Pub/Sub
    publisher = pubsub_v1.PublisherClient(credentials=AnonymousCredentials())
    subscriber = pubsub_v1.SubscriberClient(credentials=AnonymousCredentials())
    for topic_id, subs in TOPICS.items():
        topic_path = publisher.topic_path(GCP_PROJECT, topic_id)
        try:
            publisher.create_topic(request={"name": topic_path})
            print(f"Topic '{topic_id}' created.")
        except Exception:
            print(f"Topic '{topic_id}' already exists.")

        for sub_info in subs:
            sub_path = subscriber.subscription_path(GCP_PROJECT, sub_info["name"])
            try:
                subscriber.create_subscription(
                    request={
                        "name": sub_path,
                        "topic": topic_path,
                        "push_config": {"push_endpoint": sub_info["endpoint"]},
                        "ack_deadline_seconds": 60,
                    }
                )
                print(f"  Subscription '{sub_info['name']}' created.")
            except Exception:
                print(f"  Subscription '{sub_info['name']}' already exists.")

def main():
    print("--- Starting Local E2E Test ---")
    os.environ["PUBSUB_EMULATOR_HOST"] = PUBSUB_EMULATOR_HOST
    os.environ["STORAGE_EMULATOR_HOST"] = STORAGE_EMULATOR_HOST.replace("http://", "")

    try:
        # 1. Start services
        print("\n--- Building and starting Docker services ---")
        subprocess.run(["docker-compose", "up", "--build", "-d"], check=True)
        time.sleep(10) # Wait for services to be ready

        # 2. Setup
        setup_emulators()

        # 3. Upload test data
        print("\n--- Uploading test data to GCS emulator ---")
        storage_client = storage.Client(project=GCP_PROJECT, credentials=AnonymousCredentials())
        bucket = storage_client.bucket(BUCKET_NAME)
        
        csv_blob = bucket.blob(f"{RUN_ID}/glen_race.csv")
        csv_blob.upload_from_filename("agents/test_data/glen_race.csv")
        pit_blob = bucket.blob(f"{RUN_ID}/glen_pit.json")
        pit_blob.upload_from_filename("agents/test_data/glen_pit.json") # Assuming this file exists
        print("Test data uploaded.")

        # 4. Trigger Core Analyzer
        print("\n--- Triggering Core Analyzer service ---")
        resp = requests.post(f"{CORE_ANALYZER_URL}/analyze", json={
            "run_id": RUN_ID,
            "csv_path": f"gs://{BUCKET_NAME}/{csv_blob.name}",
            "pit_json_path": f"gs://{BUCKET_NAME}/{pit_blob.name}"
        })
        resp.raise_for_status()
        analysis_path = resp.json()["analysis_path"]
        print(f"Analysis complete: {analysis_path}")
        
        # 5. Simulate GCS trigger to Pub/Sub
        print("\n--- Simulating GCS trigger to Pub/Sub ---")
        publisher = pubsub_v1.PublisherClient(credentials=AnonymousCredentials())
        topic_path = publisher.topic_path(GCP_PROJECT, "analysis-completed")
        message_data = json.dumps({"analysis_path": analysis_path}).encode("utf-8")
        future = publisher.publish(topic_path, message_data)
        future.result()
        print("Published message to 'analysis-completed' topic.")

        # Normally we'd wait for all artifacts, but for a simple test, we'll just wait
        print("\n--- Waiting for pipeline to process... ---")
        time.sleep(30) # Allow time for async processing
        
        print("\n--- E2E Test finished. Check GCS bucket for results. ---")
        print(f"Results should be in gs://{BUCKET_NAME}/{RUN_ID}/")
        
    finally:
        print("\n--- Tearing down Docker services ---")
        subprocess.run(["docker-compose", "down"])

if __name__ == "__main__":
    main()
```

setup.py
```py
from setuptools import setup, find_packages

setup(
    name="apex-agents",
    version="0.1.0",
    packages=find_packages(),
)
```

tests/e2e_runner.py
```py
"""Offline end-to-end runner for Project Apex.

This module wires up the dry-run harness stubs and executes the full
analysis → insight → visualisation → report → tweets pipeline on a given
fixture set.  It mirrors production behaviour while remaining fully local and
self-contained.

Usage (CLI):

    python -m tests.e2e_runner --fixtures agents/test_data --out out_local

Or import and call `run()` from test code.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Ensure repository root is on PYTHONPATH so that local imports resolve even
# when executed via `python -m` inside the tests package.
# ---------------------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Register all dependency stubs / AI mocks *before* importing pipeline code.
import dry_run_harness  # noqa: F401  pylint: disable=unused-import

from full_pipeline_local import run_pipeline  # noqa: E402  pylint: disable=wrong-import-position


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------

def run(fixture_dir: pathlib.Path, out_dir: pathlib.Path | None = None) -> pathlib.Path:  # noqa: D401
    """Execute the entire pipeline on `fixture_dir`.

    `fixture_dir` must contain at minimum:
        • race.csv  – Mandatory race telemetry
        • pit.json  – Pit-stop events JSON (may be empty)
        • fuel.json – Manufacturer fuel capacity JSON (optional)

    Returns the path to the output directory containing all generated files.
    """
    fixture_dir = fixture_dir.expanduser().resolve()
    if not fixture_dir.is_dir():
        raise FileNotFoundError(f"Fixture directory not found: {fixture_dir}")

    csv_path = fixture_dir / "race.csv"
    pit_json = fixture_dir / "pit.json"
    fuel_json = fixture_dir / "fuel.json"

    if not csv_path.exists() or not pit_json.exists():
        raise FileNotFoundError("Fixture set must include race.csv and pit.json")

    out_dir = (out_dir or (fixture_dir.parent / "out_local")).expanduser().resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)

    run_pipeline(csv_path, pit_json, fuel_json if fuel_json.exists() else None, out_dir)
    return out_dir


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project Apex offline pipeline on a fixture set.")
    parser.add_argument("--fixtures", type=pathlib.Path, default=pathlib.Path("agents/test_data"), help="Directory containing fixture files (race.csv, pit.json, fuel.json).")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("out_local"), help="Output directory.")
    return parser.parse_args(argv)


def _main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    out_path = run(args.fixtures, args.out)
    print(f"Pipeline finished. Outputs in {out_path}")


if __name__ == "__main__":  # pragma: no cover
    _main()
```

tests/test_end_to_end.py
```py
"""Pytest for full offline Project Apex pipeline."""
from __future__ import annotations

import pathlib

from tests.e2e_runner import run


def test_pipeline(tmp_path: pathlib.Path):
    out_dir = run(pathlib.Path("agents/test_data"), tmp_path)

    assert (out_dir / "analysis_enhanced.json").exists(), "Analysis JSON missing"
    assert (out_dir / "race_report.pdf").exists(), "PDF report missing"
    # visuals
    pngs = list(out_dir.glob("*.png"))
    assert pngs, "Plot PNGs not generated"
    # tweets
    tweets_path = out_dir / "tweets.json"
    assert tweets_path.exists() and tweets_path.stat().st_size > 2, "Tweets JSON empty"
```