"""Compute canonical UQ for an explicitly located published run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..bootstrap.config import load_config
from ..bootstrap.errors import HardFailure
from ..bootstrap.uncertainty import compute_store_uncertainty


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        units = {
            "energy": config.data.units.energy,
            "forces": config.data.units.forces,
            "stress": config.data.units.stress,
        }
        for split in config.prediction.splits:
            for mode in config.uncertainty.parameter_modes:
                compute_store_uncertainty(
                    arguments.run_root / "predictions",
                    arguments.run_root / "uncertainty",
                    split=split,
                    mode=mode,
                    member_count=config.bootstrap.ensemble_size,
                    units=units,
                )
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
