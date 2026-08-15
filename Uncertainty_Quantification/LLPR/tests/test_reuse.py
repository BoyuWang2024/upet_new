from __future__ import annotations

import errno
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    atomic_json_dump,
    atomic_npz_save,
    load_verified_manifest,
    sha256_file,
)
from Uncertainty_Quantification.LLPR.llpr.reuse import (
    materialize_reused_stage,
    validate_curvature_layout,
)


CURVATURE_ID = "a" * 16
CHECKPOINT_SHA = "b" * 64


def _write_curvature_source(root: Path) -> Path:
    source = root / "source"
    source.mkdir(parents=True)
    matrix_path = source / "curvature.npz"
    diagnostics_path = source / "diagnostics.json"
    atomic_npz_save(
        matrix_path,
        {"energy": np.eye(2, dtype=np.float64), "force": 2 * np.eye(3)},
    )
    atomic_json_dump(
        diagnostics_path,
        {
            "energy_dimension": 2,
            "force_dimension": 3,
            "total_dimension": 5,
            "structure_count": 4,
            "atom_count": 8,
            "force_component_count": 24,
        },
    )
    atomic_json_dump(
        source / "manifest.json",
        {
            "stage": "curvature",
            "status": "complete",
            "identity": CURVATURE_ID,
            "schema_version": SCHEMA_VERSION,
            "formula_version": FORMULA_VERSION,
            "layout_hash": "layout",
            "payload": {"checkpoint_sha256": CHECKPOINT_SHA},
            "files": {
                matrix_path.name: sha256_file(matrix_path),
                diagnostics_path.name: sha256_file(diagnostics_path),
            },
        },
    )
    return source


def _materialize(source: Path, root: Path) -> Path:
    return materialize_reused_stage(
        source,
        root / "target/curvature",
        stage="curvature",
        identity=CURVATURE_ID,
        expected_payload={"checkpoint_sha256": CHECKPOINT_SHA},
    )


def test_reuse_materializes_only_manifest_and_declared_files(tmp_path: Path) -> None:
    source = _write_curvature_source(tmp_path)
    (source / "scratch.bin").write_bytes(b"not formal")

    target = _materialize(source, tmp_path)

    assert {path.name for path in target.iterdir()} == {
        "curvature.npz",
        "diagnostics.json",
        "manifest.json",
    }
    manifest = load_verified_manifest(target / "manifest.json", verify_npz=True)
    assert manifest["identity"] == CURVATURE_ID


def test_reuse_falls_back_to_copy_on_cross_device_link(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_curvature_source(tmp_path)

    def cross_device_link(source_path: Path, destination_path: Path) -> None:
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", cross_device_link)

    target = _materialize(source, tmp_path)

    assert (target / "curvature.npz").read_bytes() == (
        source / "curvature.npz"
    ).read_bytes()
    assert (target / "curvature.npz").stat().st_ino != (
        source / "curvature.npz"
    ).stat().st_ino


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("stage", "calibration", "identity mismatch"),
        ("identity", "c" * 16, "identity mismatch"),
        ("schema_version", "wrong-schema", "identity mismatch"),
        ("formula_version", "wrong-formula", "identity mismatch"),
    ],
)
def test_reuse_rejects_incompatible_manifest_identity_fields(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    source = _write_curvature_source(tmp_path)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    atomic_json_dump(manifest_path, manifest)

    with pytest.raises(ValueError, match=message):
        _materialize(source, tmp_path)

    assert not (tmp_path / "target/curvature" / CURVATURE_ID).exists()


def test_reuse_rejects_checkpoint_payload_mismatch(tmp_path: Path) -> None:
    source = _write_curvature_source(tmp_path)

    with pytest.raises(ValueError, match="payload.*checkpoint_sha256"):
        materialize_reused_stage(
            source,
            tmp_path / "target/curvature",
            stage="curvature",
            identity=CURVATURE_ID,
            expected_payload={"checkpoint_sha256": "d" * 64},
        )


def test_reuse_rejects_tampered_declared_file(tmp_path: Path) -> None:
    source = _write_curvature_source(tmp_path)
    (source / "diagnostics.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA mismatch"):
        _materialize(source, tmp_path)


def test_reuse_rejects_symlink_source_directory(tmp_path: Path) -> None:
    source = _write_curvature_source(tmp_path)
    symlink = tmp_path / "source-link"
    symlink.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="ordinary directory"):
        _materialize(symlink, tmp_path)


def test_reuse_returns_identical_existing_target(tmp_path: Path) -> None:
    source = _write_curvature_source(tmp_path)
    first = _materialize(source, tmp_path)
    inode = (first / "curvature.npz").stat().st_ino

    second = _materialize(source, tmp_path)

    assert second == first
    assert (second / "curvature.npz").stat().st_ino == inode


def test_reuse_rejects_conflicting_existing_target(tmp_path: Path) -> None:
    source = _write_curvature_source(tmp_path)
    target = _materialize(source, tmp_path)
    replacement = target / ".diagnostics.json.replacement"
    replacement.write_text("{}\n", encoding="utf-8")
    replacement.replace(target / "diagnostics.json")

    with pytest.raises(ValueError, match="existing reuse target"):
        _materialize(source, tmp_path)

    assert (target / "diagnostics.json").read_text(encoding="utf-8") == "{}\n"
    assert (source / "diagnostics.json").read_text(encoding="utf-8") != "{}\n"


def test_curvature_layout_matches_manifest_diagnostics_and_matrices(
    tmp_path: Path,
) -> None:
    source = _write_curvature_source(tmp_path)
    manifest = load_verified_manifest(source / "manifest.json", verify_npz=True)
    layout = SimpleNamespace(
        energy=SimpleNamespace(dimension=2),
        force=SimpleNamespace(dimension=3),
        layout_hash="layout",
    )

    validate_curvature_layout(source, manifest, layout)


@pytest.mark.parametrize(
    ("energy_dimension", "force_dimension", "layout_hash", "message"),
    [
        (4, 3, "layout", "energy matrix dimension"),
        (2, 4, "layout", "force matrix dimension"),
        (2, 3, "different", "layout hash"),
    ],
)
def test_curvature_layout_rejects_incompatible_model(
    tmp_path: Path,
    energy_dimension: int,
    force_dimension: int,
    layout_hash: str,
    message: str,
) -> None:
    source = _write_curvature_source(tmp_path)
    manifest = load_verified_manifest(source / "manifest.json", verify_npz=True)
    layout = SimpleNamespace(
        energy=SimpleNamespace(dimension=energy_dimension),
        force=SimpleNamespace(dimension=force_dimension),
        layout_hash=layout_hash,
    )

    with pytest.raises(ValueError, match=message):
        validate_curvature_layout(source, manifest, layout)
