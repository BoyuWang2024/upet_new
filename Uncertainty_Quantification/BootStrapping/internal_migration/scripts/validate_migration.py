"""Validate every checkpoint and prediction slice after normalization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...bootstrap.config import load_config
from ...bootstrap.errors import HardFailure
from ..migration.validation import validate_migrated_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        result = validate_migrated_run(
            arguments.source, arguments.destination, load_config(arguments.config)
        )
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "member_count": result.member_count,
                "validated_checkpoints": result.validated_checkpoints,
                "validated_chunks": result.validated_chunks,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
