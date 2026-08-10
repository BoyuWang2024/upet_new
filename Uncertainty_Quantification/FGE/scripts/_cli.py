"""Shared strict CLI parsing and error handling."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from ..fge.config import load_config
from ..fge.errors import HardFailure


def build_parser(
    description: str, *, preflight: bool = False
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, type=Path)
    if preflight:
        parser.add_argument(
            "--stage", required=True, choices=("train", "predict", "evaluate")
        )
    return parser


def run_stage(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    stage: Callable[[object], object] | None = None,
    *,
    preflight: Callable[[object, str], object] | None = None,
) -> int:
    try:
        args = parser.parse_args(argv)
        config = load_config(args.config)
        if preflight is not None:
            preflight(config, args.stage)
        elif stage is not None:
            stage(config)
        else:
            raise HardFailure("CLI stage is not configured")
    except HardFailure as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
