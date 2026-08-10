from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from ...fge.config import load_config
from ...fge.validation import validate_result
from ._cli import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate a canonical migrated FGE result"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--destination", required=True)
    return parser


def _command(argv: Sequence[str] | None) -> None:
    args = build_parser().parse_args(argv)
    validate_result(load_config(args.config), Path(args.destination))


def main(argv: Sequence[str] | None = None) -> int:
    return run(_command, argv)


if __name__ == "__main__":
    raise SystemExit(main())
