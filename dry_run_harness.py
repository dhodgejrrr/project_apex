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

# ---------------------------------------------------------------------------
# Google API stubs (prevents import errors when google-cloud libs are absent)
# ---------------------------------------------------------------------------
import types, sys
_dummy_google = types.ModuleType("google")
_dummy_google_cloud = types.ModuleType("google.cloud")

class _MockModel:
    @staticmethod
    def from_pretrained(_name: str):
        return _MockModel()

    def predict(self, _prompt: str, **_kwargs):
        # Return object with .text attribute to mimic Vertex response
        return SimpleNamespace(text="{}")

_dummy_aiplatform = types.ModuleType("google.cloud.aiplatform")
_dummy_aiplatform.init = lambda *_, **__: None
_dummy_aiplatform.TextGenerationModel = _MockModel

sys.modules.setdefault("google", _dummy_google)
sys.modules.setdefault("google.cloud", _dummy_google_cloud)
sys.modules.setdefault("google.cloud.aiplatform", _dummy_aiplatform)

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
for fname in [
    "figure",
    "bar",
    "plot",
    "close",
    "savefig",
    "text",
    "xlabel",
    "ylabel",
    "title",
    "legend",
    "tight_layout",
]:
    setattr(plt_stub, fname, lambda *_, **__: None)
setattr(plt_stub, "gca", lambda: SimpleNamespace(invert_yaxis=lambda: None))

# numpy, pandas
_stub_module("numpy", {"linspace": lambda *_, **__: []})
_stub_module("pandas", {"DataFrame": lambda *_, **__: object})

# Jinja2 & WeasyPrint stubs
_stub_module("jinja2", {"Environment": lambda *_, **__: SimpleNamespace(get_template=lambda _name: SimpleNamespace(render=lambda **__: "<html></html>")),
                          "FileSystemLoader": object,
                          "select_autoescape": lambda *_: None})
_stub_module("weasyprint", {"HTML": SimpleNamespace(write_pdf=lambda *_: None)})

# ---------------------------------------------------------------------------
# Mock AI helper functions
# ---------------------------------------------------------------------------

def _mock_summarize(prompt: str, **_: Any) -> str:  # noqa: D401
    """Return a deterministic summary irrespective of the prompt."""
    return "[MOCK SUMMARY] Key performance improved; tyre degradation in control; strategic consistency evident."


def _mock_generate_json(prompt: str, **_: Any):  # noqa: D401
    """Return canned JSON structures based on keywords in the prompt."""
    if "executive summary" in prompt.lower():
        return {
            "executive_summary": (
                "Despite stiff competition, the team demonstrated superior pace in critical phases, "
                "leveraging lower tyre degradation to secure consistent lap times."
            ),
            "tactical_recommendations": [
                "Focus on maintaining tyre temperature in opening stints.",
                "Explore undercut opportunities during early cautions.",
                "Increase fuel-corrected pace in final third to pressure rivals.",
            ],
        }
    # otherwise assume tweet generation
    return [
        "🏁 Stellar stint pace keeps us in the hunt! 💪 #IMSA #ProjectApex",
        "Strategy pays off – minimal tyre deg and lightning pit work 🚀 #RaceDay",
        "Consistency is king; watch us climb the charts! 📈 #Motorsport",
    ]


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
