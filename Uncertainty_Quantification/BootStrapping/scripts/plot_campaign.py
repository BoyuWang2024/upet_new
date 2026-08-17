"""Render the deterministic Carnet-style campaign plots."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
from ..bootstrap.campaign import load_campaign
from ..bootstrap.errors import HardFailure
from ..bootstrap.plot_store import publish_campaign_plots

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = publish_campaign_plots(load_campaign(arguments.campaign))
    except HardFailure as error:
        print(f"campaign plotting failed: {error}", file=sys.stderr)
        return 2
    print(result.plot_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
