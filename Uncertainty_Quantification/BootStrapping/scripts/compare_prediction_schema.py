"""Compare two canonical prediction stores without binding dataset size."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..bootstrap.errors import HardFailure
from ..bootstrap.schema import prediction_schema_signature


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--split", required=True, choices=("val", "test"))
    parser.add_argument(
        "--mode", action="append", required=True, choices=("raw", "ema")
    )
    parser.add_argument("--left-members", required=True, type=int)
    parser.add_argument("--right-members", required=True, type=int)
    arguments = parser.parse_args(argv)
    try:
        left = prediction_schema_signature(
            arguments.left,
            split=arguments.split,
            modes=arguments.mode,
            member_count=arguments.left_members,
        )
        right = prediction_schema_signature(
            arguments.right,
            split=arguments.split,
            modes=arguments.mode,
            member_count=arguments.right_members,
        )
        if left != right:
            raise HardFailure("prediction schema signatures differ")
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    print(json.dumps(left, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
