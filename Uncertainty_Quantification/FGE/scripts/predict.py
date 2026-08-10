"""Run formal FGE member prediction."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Uncertainty_Quantification.FGE.fge.prediction import predict_members
    from Uncertainty_Quantification.FGE.scripts._cli import (
        build_parser as _build_parser,
    )
    from Uncertainty_Quantification.FGE.scripts._cli import run_stage
else:
    from ..fge.prediction import predict_members
    from ._cli import build_parser as _build_parser
    from ._cli import run_stage


def build_parser():
    return _build_parser("Run FGE prediction")


def main(argv: Sequence[str] | None = None) -> int:
    return run_stage(build_parser(), argv, predict_members)


if __name__ == "__main__":
    raise SystemExit(main())
