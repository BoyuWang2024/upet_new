"""Run one resumable inference-only FGE dataset prediction."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.inference_config import (
        load_inference_config,
    )
    from Uncertainty_Quantification.FGE.fge.inference_only import (
        predict_inference_dataset,
    )
else:
    from ..fge.errors import HardFailure
    from ..fge.inference_config import load_inference_config
    from ..fge.inference_only import predict_inference_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict one FGE dataset")
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        predict_inference_dataset(load_inference_config(arguments.config))
    except HardFailure as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
