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
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--stage",
        choices=("predict", "postprocess", "plot", "all"),
        default="all",
    )
    args = parser.parse_args(argv)

    from confidence_head.e0_config import load_e0_config
    from confidence_head.workflows.e0_commands import dispatch_e0_stage

    outputs = dispatch_e0_stage(
        load_e0_config(args.config.resolve()),
        args.stage,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
