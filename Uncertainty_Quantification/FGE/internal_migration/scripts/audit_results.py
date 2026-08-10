from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

from ...fge.artifacts import sha256_file
from ...fge.errors import HardFailure
from ._cli import run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Recheck a legacy migration audit")
    parser.add_argument("--audit", required=True)
    return parser


def audit(path: Path) -> None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardFailure("external audit cannot be loaded") from exc
    if (
        not isinstance(document, Mapping)
        or document.get("publication_authorized") is not True
    ):
        raise HardFailure("external audit is not an authorized migration")
    root_value = document.get("source_root")
    before = document.get("source_hashes_before")
    after = document.get("source_hashes_after")
    if (
        not isinstance(root_value, str)
        or not isinstance(before, Mapping)
        or before != after
    ):
        raise HardFailure("external audit source identity is invalid")
    root = Path(root_value).resolve()
    for relative, digest in before.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise HardFailure("external audit source hash inventory is invalid")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise HardFailure("external audit source path escapes root") from exc
        if (
            not candidate.is_file()
            or candidate.is_symlink()
            or sha256_file(candidate) != digest
        ):
            raise HardFailure("external audit source has changed")


def _command(argv: Sequence[str] | None) -> None:
    args = build_parser().parse_args(argv)
    audit(Path(args.audit))


def main(argv: Sequence[str] | None = None) -> int:
    return run(_command, argv)


if __name__ == "__main__":
    raise SystemExit(main())
