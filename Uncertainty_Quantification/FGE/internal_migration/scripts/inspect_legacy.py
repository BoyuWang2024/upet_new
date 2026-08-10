from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from ..migration.converter import _default_expectations
from ..migration.legacy_reader import read_legacy_run
from ._cli import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the authenticated legacy FGE run"
    )
    parser.add_argument("--source", required=True)
    return parser


def inspect(source: Path) -> dict[str, object]:
    run = read_legacy_run(source, _default_expectations())
    return {
        "status": "PASS",
        "member_ids": [member.member_id for member in run.members],
        "statistics": run.prediction["statistics"],
        "critical_file_count": len(run.source_snapshot.files),
    }


def _command(argv: Sequence[str] | None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(inspect(Path(args.source)), sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    return run(_command, argv)


if __name__ == "__main__":
    raise SystemExit(main())
