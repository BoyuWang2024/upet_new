"""Validate and publish one formal FGE result."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..fge.config import FGEConfig
from ..fge.validation import validate_result
from ._cli import build_parser as _build_parser
from ._cli import run_stage


def build_parser():
    return _build_parser("Validate FGE result")


def _validate(config: FGEConfig) -> object:
    return validate_result(config, Path(config.paths.output_root) / config.project.name)


def main(argv: Sequence[str] | None = None) -> int:
    return run_stage(build_parser(), argv, _validate)
