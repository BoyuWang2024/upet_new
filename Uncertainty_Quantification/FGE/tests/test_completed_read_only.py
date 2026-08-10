"""Completed formal results remain byte-for-byte read-only."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("field", ("unknown", "extra", "mismatch"))
def test_completed_result_rejects_tampered_writer_identities(
    tmp_path: Path, field: str
) -> None:
    """Completion identities must retain the exact training writer identities."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import validate_result

    config, root = make_canonical_result(tmp_path)
    validate_result(config, root)
    path = root / "result_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if field == "unknown":
        manifest["artifact_writer_code_identity"] = {"status": "unknown"}
    elif field == "extra":
        manifest["validator_code_identity"]["extra"] = "x"
    else:
        manifest["validator_code_identity"] = {
            "commit": "0" * 40,
            "dirty_sha256": "0" * 64,
        }
    path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HardFailure):
        validate_result(config, root)
