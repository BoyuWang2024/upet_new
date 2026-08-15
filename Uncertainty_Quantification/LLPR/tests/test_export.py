from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    atomic_json_dump,
    atomic_npz_save,
    load_verified_manifest,
    sha256_file,
)
from Uncertainty_Quantification.LLPR.llpr.export import export_evaluation_summary


def _evaluation(root: Path) -> Path:
    root.mkdir(parents=True)
    details = root / "details.npz"
    summary = root / "summary.json"
    preview = root / "preview.json"
    atomic_npz_save(details, {"energy_residual": np.array([0.1, -0.2])})
    atomic_json_dump(summary, {"structure_count": 2, "energy_mae": 0.15})
    atomic_json_dump(preview, {"energy": [{"structure_index": 0}]})
    atomic_json_dump(
        root / "manifest.json",
        {
            "status": "complete",
            "identity": "evaluation-id",
            "files": {
                "details.npz": sha256_file(details),
                "summary.json": sha256_file(summary),
                "preview.json": sha256_file(preview),
            },
        },
    )
    return root


def test_export_contains_only_verified_summary_files(tmp_path: Path) -> None:
    source = _evaluation(tmp_path / "evaluation")
    destination = tmp_path / "published-summary"

    result = export_evaluation_summary(source, destination)

    assert result == destination
    assert {path.name for path in destination.iterdir()} == {
        "summary.json",
        "preview.json",
        "manifest.json",
    }
    manifest = load_verified_manifest(destination / "manifest.json")
    assert manifest["status"] == "complete"
    assert manifest["stage"] == "evaluation-summary"
    assert manifest["evaluation_identity"] == "evaluation-id"
    assert manifest["details_sha256"] == sha256_file(source / "details.npz")
    assert isinstance(manifest["files"], dict)
    assert set(manifest["files"]) == {"summary.json", "preview.json"}
    assert str(source.resolve()) not in json.dumps(manifest)
    assert not (destination / "details.npz").exists()


def test_export_rejects_unverified_source_and_existing_destination(
    tmp_path: Path,
) -> None:
    source = _evaluation(tmp_path / "evaluation")
    destination = tmp_path / "published-summary"
    destination.mkdir()
    (destination / "unrelated.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="destination already exists"):
        export_evaluation_summary(source, destination)
    assert (destination / "unrelated.txt").read_text(encoding="utf-8") == "keep"

    destination.rename(tmp_path / "existing")
    (source / "summary.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        export_evaluation_summary(source, destination)
    assert not destination.exists()
