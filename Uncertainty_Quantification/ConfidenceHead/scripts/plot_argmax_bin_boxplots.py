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
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    from confidence_head.plot_analysis import load_plot_series, plot_single_boxplot

    run_dir = args.run_dir.resolve()
    artifacts = plot_single_boxplot(
        load_plot_series(run_dir),
        run_dir / "plots" / "argmax_bin_boxplots",
    )
    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
