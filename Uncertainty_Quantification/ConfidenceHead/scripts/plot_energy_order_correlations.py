from __future__ import annotations

import sys
from argparse import ArgumentParser
from collections.abc import Sequence
from pathlib import Path


CONFIDENCE_HEAD_ROOT = Path(__file__).resolve().parents[1]
if str(CONFIDENCE_HEAD_ROOT) not in sys.path:
    sys.path.insert(0, str(CONFIDENCE_HEAD_ROOT))


def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--run-dir", required=True, action="append", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if len(args.run_dir) != 8:
        parser.error("--run-dir must be supplied exactly eight times")

    from confidence_head.plot_analysis import (
        energy_series_by_order,
        load_plot_series,
        plot_energy_correlations,
        validate_comparable_energy,
    )

    series = energy_series_by_order(
        tuple(load_plot_series(path.resolve()) for path in args.run_dir)
    )
    rows = validate_comparable_energy(series)
    artifacts = plot_energy_correlations(rows, args.output_dir.resolve())
    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
