"""Thin command-line orchestration for the UPET LLPR workflow."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from .artifacts import verify_run as _verify_run
from .calibration import resolve_calibration_stage as _resolve_calibration_stage
from .config import load_llpr_config
from .curvature import resolve_curvature_stage as _resolve_curvature_stage
from .inference import run_evaluate as _run_evaluate
from .plotting import load_plot_config
from .plotting import run_plot as _run_plot


LOGGER = logging.getLogger(__name__)


def run_build(config_path: Path) -> Path:
    return _resolve_curvature_stage(load_llpr_config(config_path))


def run_calibrate(config_path: Path) -> Path:
    return _resolve_calibration_stage(load_llpr_config(config_path))


def run_evaluate(config_path: Path) -> Path:
    return _run_evaluate(load_llpr_config(config_path))


def verify_run(run_root: Path) -> dict[str, object]:
    return _verify_run(run_root, level="full")


def run_plot(config_path: Path) -> Path:
    return _run_plot(load_plot_config(config_path))


def build_parser() -> argparse.ArgumentParser:
    """Build the six-command LLPR command-line interface."""
    parser = argparse.ArgumentParser(
        prog="python -m Uncertainty_Quantification.LLPR.llpr",
        description="Build, calibrate, evaluate, verify, and plot UPET LLPR.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    descriptions = {
        "build": "Build or resume curvature from the configured build split.",
        "calibrate": "Calibrate fixed or fitted ridge parameters on validation.",
        "evaluate": "Evaluate the configured test split.",
        "run": "Run build, calibrate, and evaluate in sequence.",
        "verify": "Fully verify all manifests below an output directory.",
        "plot": "Plot a completed evaluation without inference or recalibration.",
    }
    for command, description in descriptions.items():
        subparser = subparsers.add_parser(command, help=description)
        subparser.add_argument(
            "--config",
            type=Path,
            required=True,
            help=(
                "YAML configuration path; for verify, this is the run output directory."
            ),
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch one CLI invocation and let failures produce a nonzero exit."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    config_path = args.config
    if args.command == "run":
        for stage in (run_build, run_calibrate, run_evaluate):
            stage_output = stage(config_path)
            LOGGER.info("completed %s: %s", stage.__name__, stage_output)
        return 0
    runners = {
        "build": run_build,
        "calibrate": run_calibrate,
        "evaluate": run_evaluate,
        "verify": verify_run,
        "plot": run_plot,
    }
    output = runners[args.command](config_path)
    LOGGER.info("completed %s: %s", args.command, output)
    return 0
