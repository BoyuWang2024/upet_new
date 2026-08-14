"""Build the formal BootStrapping source release archive."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..bootstrap.errors import HardFailure
from ..bootstrap.release import build_source_release


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        archive = build_source_release(arguments.source_root, arguments.output)
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    print(archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
