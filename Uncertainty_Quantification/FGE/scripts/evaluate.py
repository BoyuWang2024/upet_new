"""Run formal FGE evaluation."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Uncertainty_Quantification.FGE.fge.evaluation import evaluate_fge
    from Uncertainty_Quantification.FGE.scripts._cli import (
        build_parser as _build_parser,
    )
    from Uncertainty_Quantification.FGE.scripts._cli import run_stage
else:
    from ..fge.evaluation import evaluate_fge
    from ._cli import build_parser as _build_parser
    from ._cli import run_stage


def build_parser():
    return _build_parser("Run FGE evaluation")


def main(argv: Sequence[str] | None = None) -> int:
    return run_stage(build_parser(), argv, evaluate_fge)


if __name__ == "__main__":
    raise SystemExit(main())
