"""Validate canonical UQ publications and finalize a bootstrap run."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..bootstrap.config import load_config
from ..bootstrap.errors import HardFailure
from ..bootstrap.finalization import finalize_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        finalize_run(
            arguments.run_root,
            splits=config.prediction.splits,
            modes=config.uncertainty.parameter_modes,
            member_count=config.bootstrap.ensemble_size,
        )
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
