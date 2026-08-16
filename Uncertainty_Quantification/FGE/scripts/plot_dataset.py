"""Render independent uncertainty-residual panels for one FGE dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.plot_analysis import (
        analyze_plot_input,
        load_completed_fge_plot_input,
        load_inference_plot_input,
        load_plot_config,
    )
    from Uncertainty_Quantification.FGE.fge.plot_rendering import render_plot_suite
else:
    from ..fge.errors import HardFailure
    from ..fge.plot_analysis import (
        analyze_plot_input,
        load_completed_fge_plot_input,
        load_inference_plot_input,
        load_plot_config,
    )
    from ..fge.plot_rendering import render_plot_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot one FGE dataset")
    parser.add_argument("--config", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = build_parser().parse_args(argv)
        config = load_plot_config(arguments.config)
        if config.input_kind == "completed_fge":
            plot_input = load_completed_fge_plot_input(config.input_root)
        else:
            plot_input = load_inference_plot_input(config.input_root)
        result = analyze_plot_input(plot_input, config.settings)
        render_plot_suite(result, config.output_root)
    except HardFailure as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
