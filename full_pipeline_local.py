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
        default=pathlib.Path("agents/test_data/race.csv"),
        help="Race CSV file path.",
    )
    parser.add_argument(
        "--pit_json",
        type=pathlib.Path,
        default=pathlib.Path("agents/test_data/pit.json"),
        help="Pit-stop JSON file path.",
    )
    parser.add_argument(
        "--fuel_caps",
        "--pits",  # backward-compat alias
        type=pathlib.Path,
        default=pathlib.Path("agents/test_data/fuel_capacity.json"),
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
