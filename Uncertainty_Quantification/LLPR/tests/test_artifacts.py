import json
from pathlib import Path

import numpy as np
import pytest

from Uncertainty_Quantification.LLPR.llpr.artifacts import (
    FORMULA_VERSION,
    SCHEMA_VERSION,
    RunPaths,
    atomic_json_dump,
    atomic_npz_save,
    canonical_json,
    load_complete_manifest,
    load_verified_manifest,
    sha256_file,
    stable_id,
    stage_identity,
)


def test_stage_identity_is_order_independent() -> None:
    left = stable_id({"stage": "curvature", "payload": {"b": 2, "a": 1}})
    right = stable_id({"payload": {"a": 1, "b": 2}, "stage": "curvature"})

    assert left == right
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_stage_identity_binds_schema_formula_and_stage() -> None:
    identity = stage_identity("curvature", {"dataset": "abc"})

    assert identity["schema_version"] == SCHEMA_VERSION
    assert identity["formula_version"] == FORMULA_VERSION
    assert identity["stage"] == "curvature"
    assert identity["identity"] == stable_id(
        {
            "schema_version": SCHEMA_VERSION,
            "formula_version": FORMULA_VERSION,
            "stage": "curvature",
            "payload": {"dataset": "abc"},
        }
    )


def test_atomic_npz_does_not_publish_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "matrix.npz"

    def raising_savez(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected write failure")

    monkeypatch.setattr(np, "savez_compressed", raising_savez)
    with pytest.raises(RuntimeError, match="injected write failure"):
        atomic_npz_save(destination, {"matrix": np.eye(2)})

    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_npz_round_trips_without_suffix_surprise(tmp_path: Path) -> None:
    destination = tmp_path / "matrix.npz"

    atomic_npz_save(destination, {"matrix": np.eye(2)})

    with np.load(destination) as loaded:
        np.testing.assert_array_equal(loaded["matrix"], np.eye(2))
    assert sha256_file(destination)
    assert not (tmp_path / "matrix.npz.npz").exists()


def test_incomplete_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"status": "in_progress"}', encoding="utf-8")

    with pytest.raises(ValueError, match="status.*complete"):
        load_complete_manifest(path)


def test_manifest_identity_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    atomic_json_dump(
        path,
        {
            "status": "complete",
            "identity": "actual",
            "schema_version": SCHEMA_VERSION,
        },
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        load_complete_manifest(path, {"identity": "expected"})


def test_run_paths_constructor_has_no_filesystem_side_effect(tmp_path: Path) -> None:
    root = tmp_path / "experiment"

    paths = RunPaths(root)

    assert paths.root == root
    assert paths.curvature == root / "curvature"
    assert paths.calibration == root / "calibration"
    assert paths.evaluation == root / "evaluation"
    assert paths.plots == root / "plots"
    assert paths.legacy_raw == root / "legacy_raw"
    assert not root.exists()


def test_atomic_json_is_stable_and_terminated(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"

    atomic_json_dump(path, {"z": 1, "a": 2})

    assert (
        path.read_text(encoding="utf-8")
        == json.dumps({"a": 2, "z": 1}, indent=2, sort_keys=True) + "\n"
    )


def test_verified_manifest_rejects_escaping_declared_path(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"data")
    stage = tmp_path / "stage"
    stage.mkdir()
    atomic_json_dump(
        stage / "manifest.json",
        {
            "status": "complete",
            "identity": "x",
            "files": {"../outside.bin": sha256_file(outside)},
        },
    )

    with pytest.raises(ValueError, match="escapes"):
        load_verified_manifest(stage / "manifest.json")


def test_verified_manifest_rejects_tampered_declared_file(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    artifact = stage / "artifact.bin"
    artifact.write_bytes(b"original")
    atomic_json_dump(
        stage / "manifest.json",
        {
            "status": "complete",
            "identity": "x",
            "files": {"artifact.bin": sha256_file(artifact)},
        },
    )
    artifact.write_bytes(b"changed")

    with pytest.raises(ValueError, match="SHA mismatch"):
        load_verified_manifest(stage / "manifest.json")
