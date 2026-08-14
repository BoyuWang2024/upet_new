"""Read-only audit of an existing BootStrapping result tree."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ...bootstrap.config import load_config
from ...bootstrap.errors import HardFailure
from ..migration.legacy_reader import inspect_legacy_run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    arguments = parser.parse_args(argv)
    try:
        audit = inspect_legacy_run(arguments.source, load_config(arguments.config))
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "member_count": len(audit.members),
                "splits": list(audit.splits),
                "chunks_per_member": {
                    split: len(audit.members[0].chunks[split]) for split in audit.splits
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
