import json
from pathlib import Path

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    publish_run_manifest,
    verify_run,
)


def test_publish_run_manifest_is_neutral_and_verifiable(tmp_path: Path) -> None:
    root = tmp_path / "experiment"

    manifest_path = publish_run_manifest(
        root,
        curvature_manifest={"identity": "curvature-id"},
        calibration_manifest={"identity": "calibration-id"},
        evaluation_manifest={"identity": "evaluation-id"},
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "run"
    assert manifest["status"] == "complete"
    assert manifest["files"] == {}
    assert manifest["curvature_identity"] == "curvature-id"
    assert manifest["calibration_identity"] == "calibration-id"
    assert manifest["evaluation_identity"] == "evaluation-id"
    assert "origin" not in manifest
    assert verify_run(root, level="full") == {
        "status": "complete",
        "level": "full",
        "manifest_count": 1,
        "verified_file_count": 0,
    }
