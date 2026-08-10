"""Validate and publish one formal FGE result."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Uncertainty_Quantification.FGE.fge.config import FGEConfig
    from Uncertainty_Quantification.FGE.fge.validation import validate_result
    from Uncertainty_Quantification.FGE.scripts._cli import (
        build_parser as _build_parser,
    )
    from Uncertainty_Quantification.FGE.scripts._cli import run_stage
else:
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


if __name__ == "__main__":
    raise SystemExit(main())
