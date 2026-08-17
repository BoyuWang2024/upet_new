"""Publish raw ensemble predictions for a configured campaign."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from ..bootstrap.campaign import load_campaign
from ..bootstrap.errors import HardFailure
from ..bootstrap.native_prediction import predict_campaign

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--run", action="append", dest="runs")
    parser.add_argument("--dataset", action="append", dest="datasets")
    arguments = parser.parse_args(argv)
    try:
        results = predict_campaign(load_campaign(arguments.campaign), arguments.runs, arguments.datasets)
    except HardFailure as error:
        print(f"campaign prediction failed: {error}", file=sys.stderr)
        return 2
    for result in results:
        print(f"{result.run_label}/{result.dataset_label}: {result.manifest_path}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
