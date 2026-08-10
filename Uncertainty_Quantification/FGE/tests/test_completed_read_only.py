"""Completed formal results remain byte-for-byte read-only."""

from __future__ import annotations

from pathlib import Path

from Uncertainty_Quantification.FGE.tests.test_validation import make_canonical_result


def test_revalidation_of_a_completed_result_does_not_mutate_any_file(
    tmp_path: Path,
) -> None:
    """A valid completion marker switches validation into pure read-only mode."""
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    validate_result(config, root)
    before = {
        path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }

    report = validate_result(config, root)
    after = {
        path.relative_to(root): (path.read_bytes(), path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }

    assert report.status == "PASS"
    assert after == before
