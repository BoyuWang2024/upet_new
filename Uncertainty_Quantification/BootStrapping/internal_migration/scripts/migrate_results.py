"""Normalize an existing BootStrapping run without model computation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ...bootstrap.config import load_config
from ...bootstrap.errors import HardFailure
from ..migration.converter import convert_legacy_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--audit-root", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        config = load_config(arguments.config)
        publication = convert_legacy_run(
            arguments.source,
            arguments.destination,
            config,
            arguments.audit_root,
        )
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    print(publication.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
