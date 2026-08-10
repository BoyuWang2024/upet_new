"""Run formal native FGE training."""

from __future__ import annotations

from collections.abc import Sequence

from ..fge.training import train_fge
from ._cli import build_parser as _build_parser
from ._cli import run_stage


def build_parser():
    return _build_parser("Run FGE training")


def main(argv: Sequence[str] | None = None) -> int:
    return run_stage(build_parser(), argv, train_fge)
