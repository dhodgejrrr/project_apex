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
