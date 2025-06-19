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

import argparse
import json
import logging
import pathlib
import shutil
import sys
import tempfile
from types import SimpleNamespace
from typing import List, Dict, Any, Sequence
from unittest import mock

# Import dry_run_harness for AI & GCP stubs (does NOT override pandas/numpy)
import dry_run_harness  # noqa: F401  pylint: disable=unused-import
from dry_run_harness import _mock_generate_json, _mock_summarize

# Project modules
from agents.core_analyzer.imsa_analyzer import IMSADataAnalyzer  # type: ignore
import agents.insight_hunter.main as insight_hunter
import agents.visualizer.main as visualizer
import agents.scribe.main as scribe
import agents.publicist.main as publicist
from agents.common import ai_helpers


LOGGER = logging.getLogger("full_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_pipeline(csv_path: pathlib.Path, pit_path: pathlib.Path, out_dir: pathlib.Path) -> None:  # noqa: D401
    """Run the entire local pipeline and write outputs into `out_dir`."""

    out_dir.mkdir(parents=True, exist_ok=True)

    with mock.patch.object(ai_helpers, "summarize", _mock_summarize), mock.patch.object(
        ai_helpers, "generate_json", _mock_generate_json
    ):
        # 1. Core analysis
        LOGGER.info("Running IMSADataAnalyzer …")
        analyzer = IMSADataAnalyzer(str(csv_path), str(pit_path))
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
        narrative = scribe._generate_narrative(insights)  # type: ignore[attr-defined]
        scribe._render_report(analysis_data, insights, narrative, pdf_path)  # type: ignore[attr-defined]

        # 5. Publicist tweets
        LOGGER.info("Composing tweets …")
        tweets = publicist._gen_tweets(insights)  # type: ignore[attr-defined]
        (out_dir / "tweets.json").write_text(json.dumps(tweets, indent=2))

    LOGGER.info("\n=== LOCAL PIPELINE COMPLETE ===")
    LOGGER.info("Outputs in %s", out_dir.resolve())
    LOGGER.info("Report: %s", pdf_path.name)
    LOGGER.info("Visuals: %s", [p.name for p, _ in visuals_info])
    LOGGER.info("Tweets: %s", len(tweets))


# ----------------------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project Apex pipeline locally.")
    parser.add_argument("--csv", type=pathlib.Path, default=pathlib.Path("agents/test_data/race.csv"), help="Race CSV file path.")
    parser.add_argument("--pits", type=pathlib.Path, default=pathlib.Path("agents/test_data/fuel.json"), help="Pit/fuel JSON file path.")
    parser.add_argument("--out_dir", type=pathlib.Path, default=pathlib.Path("out_local"), help="Output directory.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not args.csv.exists() or not args.pits.exists():
        sys.exit("CSV or Pits file not found.")
    if args.out_dir.exists():
        shutil.rmtree(args.out_dir)
    run_pipeline(args.csv, args.pits, args.out_dir)


if __name__ == "__main__":
    main()
