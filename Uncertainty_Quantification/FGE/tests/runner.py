"""Platform-independent test selection for the FGE tox environment."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest


def discover_test_directories(root: Path) -> tuple[Path, ...]:
    """Return formal tests and migration tests when the latter are available."""
    formal_tests = root / "tests"
    migration_tests = root / "internal_migration" / "tests"
    if migration_tests.is_dir():
        return formal_tests, migration_tests
    return (formal_tests,)


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the FGE test suites selected without platform-specific shell syntax."""
    root = Path(__file__).parents[1]
    selected = [str(path) for path in discover_test_directories(root)]
    return pytest.main([*selected, *(arguments or ())])


if __name__ == "__main__":
    raise SystemExit(main())
