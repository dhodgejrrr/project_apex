"""Project Apex ADK Orchestrator

This script is a lightweight entry point for analysts to trigger the full
Project Apex pipeline. It uploads a race CSV and matching pit JSON file to the
raw data Cloud Storage bucket which in turn triggers the DataIngestor Cloud
Function. That kicks off the entire asynchronous analysis chain.

Usage:
    python adk_orchestrator.py <csv_file> <pit_json_file> --bucket imsa-raw-data

The bucket must be the raw-data bucket monitored by DataIngestor. The two local
files must have the naming convention required by DataIngestor (for example
`2025_impc_mido.csv` and `2025_impc_mido_pits.json`).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Sequence

from google.cloud import storage


def upload_file(client: storage.Client, bucket_name: str, local_path: pathlib.Path) -> None:
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(local_path.name)
    print(f"Uploading {local_path} to gs://{bucket_name}/{local_path.name} ...", end=" ")
    blob.upload_from_filename(str(local_path))
    print("Done.")


def start_analysis_workflow(csv_file: pathlib.Path, pit_file: pathlib.Path, bucket_name: str) -> None:
    client = storage.Client()

    if not csv_file.exists() or not pit_file.exists():
        sys.exit("Error: One or both input files do not exist.")

    upload_file(client, bucket_name, csv_file)
    upload_file(client, bucket_name, pit_file)

    print("\nWorkflow triggered. Monitor Cloud Functions / Cloud Run logs in Google Cloud Console.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trigger Project Apex Analysis Workflow.")
    parser.add_argument("csv", type=pathlib.Path, help="Path to the local CSV data file.")
    parser.add_argument("pits", type=pathlib.Path, help="Path to the local pit JSON data file.")
    parser.add_argument("--bucket", required=True, help="Name of the GCS raw data bucket.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    start_analysis_workflow(args.csv, args.pits, args.bucket)


if __name__ == "__main__":
    main()
