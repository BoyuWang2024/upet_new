from __future__ import annotations

import sys
from collections.abc import Callable, Sequence

from ...fge.errors import HardFailure


def run(
    callback: Callable[[Sequence[str] | None], None], argv: Sequence[str] | None
) -> int:
    try:
        callback(argv)
    except HardFailure as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0
