"""Shared command-line behavior for BootStrapping stages."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence

from ..bootstrap.config import BootstrapConfig, load_config
from ..bootstrap.errors import HardFailure


def run_stage(
    parser: argparse.ArgumentParser,
    argv: Sequence[str] | None,
    stage: Callable[[BootstrapConfig], None],
) -> int:
    """Load ``--config``, execute a stage, and standardize domain failures."""

    try:
        arguments = parser.parse_args(argv)
        config = load_config(arguments.config)
        stage(config)
    except HardFailure as error:
        print(error, file=sys.stderr)
        return 2
    return 0
