"""Run formal FGE evaluation."""

from __future__ import annotations

from collections.abc import Sequence

from ..fge.evaluation import evaluate_fge
from ._cli import build_parser as _build_parser
from ._cli import run_stage


def build_parser():
    return _build_parser("Run FGE evaluation")


def main(argv: Sequence[str] | None = None) -> int:
    return run_stage(build_parser(), argv, evaluate_fge)
