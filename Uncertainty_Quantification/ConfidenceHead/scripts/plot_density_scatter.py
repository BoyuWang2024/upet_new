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
    parser.add_argument("--dataset", action="append", default=[])
    args = parser.parse_args(argv)

    from confidence_head.density_config import load_density_config
    from confidence_head.workflows.density_commands import plot_density_from_config

    manifests = plot_density_from_config(
        load_density_config(args.config.resolve()),
        names=tuple(args.dataset),
    )
    for manifest in manifests:
        print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
