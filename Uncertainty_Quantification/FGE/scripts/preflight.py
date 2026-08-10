"""Run one compute-free FGE preflight stage."""

from __future__ import annotations

from collections.abc import Sequence

from ..fge.preflight import run_preflight
from ._cli import build_parser as _build_parser
from ._cli import run_stage


def build_parser():
    return _build_parser("Run FGE preflight", preflight=True)


def main(argv: Sequence[str] | None = None) -> int:
    return run_stage(build_parser(), argv, preflight=run_preflight)
