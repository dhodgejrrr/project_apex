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
