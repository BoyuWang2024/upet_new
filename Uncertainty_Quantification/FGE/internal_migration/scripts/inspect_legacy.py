from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    from Uncertainty_Quantification.FGE.fge.config import load_config
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.internal_migration.migration import (
        legacy_reader,
    )
    from Uncertainty_Quantification.FGE.internal_migration.migration.converter import (
        _default_expectations,
    )
    from Uncertainty_Quantification.FGE.internal_migration.scripts._cli import run
else:
    from ...fge.config import load_config
    from ...fge.errors import HardFailure
    from ..migration import legacy_reader
    from ..migration.converter import _default_expectations
    from ._cli import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect the authenticated legacy FGE run"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--config", required=True)
    return parser


def inspect(source: Path, config_path: Path) -> dict[str, object]:
    config = load_config(config_path)
    expected = _default_expectations()
    if config.identity.test_data_sha256 != expected.test_data_sha256:
        raise HardFailure("legacy test identity differs from configuration")
    run = legacy_reader.read_legacy_run(source, expected)
    return {
        "status": "PASS",
        "member_ids": [member.member_id for member in run.members],
        "statistics": run.prediction["statistics"],
        "critical_file_count": len(run.source_snapshot.files),
    }


def _command(argv: Sequence[str] | None) -> None:
    args = build_parser().parse_args(argv)
    print(json.dumps(inspect(Path(args.source), Path(args.config)), sort_keys=True))


def main(argv: Sequence[str] | None = None) -> int:
    return run(_command, argv)


if __name__ == "__main__":
    raise SystemExit(main())
