"""Run native PET bootstrap member prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..bootstrap.config import load_config
from ..bootstrap.errors import HardFailure
from ..bootstrap.native_prediction import predict_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--split", action="append", choices=("val", "test"))
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args(argv)
    try:
        predict_run(
            load_config(arguments.config),
            arguments.run_root,
            splits=arguments.split,
            structure_limit=arguments.limit,
        )
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
