"""Run formal FGE member prediction."""

from __future__ import annotations

from collections.abc import Sequence

from ..fge.prediction import predict_members
from ._cli import build_parser as _build_parser
from ._cli import run_stage


def build_parser():
    return _build_parser("Run FGE prediction")


def main(argv: Sequence[str] | None = None) -> int:
    return run_stage(build_parser(), argv, predict_members)
