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
    parser.add_argument("--runs-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-dir", action="append", default=[], type=Path)
    args = parser.parse_args(argv)

    from confidence_head.plot_analysis import (
        discover_completed_runs,
        plot_completed_runs,
    )

    runs = discover_completed_runs(
        args.runs_root.resolve(),
        explicit_run_dirs=tuple(path.resolve() for path in args.run_dir),
    )
    artifacts = plot_completed_runs(
        runs,
        comparisons_dir=args.output_root.resolve() / "comparisons",
    )
    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
