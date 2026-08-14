from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_build_and_validate_manifest_audits_files(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.manifests import (
        build_run_manifest,
        validate_manifest,
    )

    root = tmp_path / "run"
    artifact = root / "members" / "member_000" / "best.ckpt"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"checkpoint")
    manifest_path = build_run_manifest(
        root,
        schema="upet.bootstrap.run/v1",
        artifacts=[artifact],
        metadata={"run_id": "tiny"},
    )

    manifest = validate_manifest(manifest_path, expected_schema="upet.bootstrap.run/v1")

    assert manifest["metadata"] == {"run_id": "tiny"}
    assert manifest["artifacts"][0]["path"] == "members/member_000/best.ckpt"


def test_validate_manifest_rejects_hash_drift_and_escape(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.bootstrap.manifests import (
        build_run_manifest,
        validate_manifest,
    )

    root = tmp_path / "run"
    artifact = root / "result.bin"
    root.mkdir()
    artifact.write_bytes(b"first")
    manifest_path = build_run_manifest(
        root,
        schema="upet.bootstrap.run/v1",
        artifacts=[artifact],
    )
    artifact.write_bytes(b"other")
    with pytest.raises(HardFailure, match="SHA-256"):
        validate_manifest(manifest_path, expected_schema="upet.bootstrap.run/v1")

    document = json.loads(manifest_path.read_text(encoding="utf-8"))
    document["artifacts"][0]["path"] = "../outside.bin"
    manifest_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(HardFailure, match="escapes"):
        validate_manifest(manifest_path, expected_schema="upet.bootstrap.run/v1")
