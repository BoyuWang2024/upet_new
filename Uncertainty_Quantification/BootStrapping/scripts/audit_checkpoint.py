"""Audit one canonical or supported v1 PET bootstrap checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..bootstrap.checkpoint import audit_checkpoint
from ..bootstrap.errors import HardFailure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-parameter-count", type=int)
    arguments = parser.parse_args(argv)
    try:
        audit = audit_checkpoint(
            arguments.checkpoint,
            expected_parameter_count=arguments.expected_parameter_count,
        )
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "sha256": audit.sha256,
                "epoch": audit.epoch,
                "validation_loss": audit.validation_loss,
                "parameter_count": audit.parameter_count,
                "inference_ready": audit.inference_ready,
                "resume_ready": audit.resume_ready,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
