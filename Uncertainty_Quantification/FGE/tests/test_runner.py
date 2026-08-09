from __future__ import annotations

from pathlib import Path

from Uncertainty_Quantification.FGE.tests.runner import discover_test_directories


def test_directories_runs_formal_tests_when_migration_tests_are_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "FGE"

    assert discover_test_directories(root) == (root / "tests",)


def test_directories_includes_migration_tests_when_present(tmp_path: Path) -> None:
    root = tmp_path / "FGE"
    migration_tests = root / "internal_migration" / "tests"
    migration_tests.mkdir(parents=True)

    assert discover_test_directories(root) == (root / "tests", migration_tests)
