from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from Uncertainty_Quantification.FGE.fge.artifacts import (
    ExperimentLayout,
    assert_safe_result_path,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_yaml,
    normalize_artifact_path,
    sha256_file,
    sibling_staging,
)
from Uncertainty_Quantification.FGE.fge.errors import HardFailure


def test_atomic_json_fsyncs_a_sibling_temporary_file_then_replaces_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result.json"
    observed: dict[str, object] = {}
    original_fsync = os.fsync
    original_replace = os.replace

    def record_fsync(fd: int) -> None:
        observed["fsync"] = True
        original_fsync(fd)

    def record_replace(source: os.PathLike[str], destination: os.PathLike[str]) -> None:
        observed["source"] = Path(source)
        observed["destination"] = Path(destination)
        original_replace(source, destination)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "replace", record_replace)

    atomic_write_json(target, {"status": "PASS"})

    assert observed["fsync"] is True
    source = observed["source"]
    assert isinstance(source, Path)
    assert source.parent == target.parent
    assert observed["destination"] == target
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert not source.exists()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_atomic_json_rejects_non_finite_numbers(tmp_path: Path, value: float) -> None:
    target = tmp_path / "invalid.json"

    with pytest.raises(HardFailure, match="non-finite JSON"):
        atomic_write_json(target, {"value": value})

    assert not target.exists()


def test_atomic_yaml_writes_a_parseable_document(tmp_path: Path) -> None:
    target = tmp_path / "result.yaml"

    atomic_write_yaml(target, {"status": "PASS", "count": 2})

    assert yaml.safe_load(target.read_text(encoding="utf-8")) == {
        "status": "PASS",
        "count": 2,
    }


def test_atomic_torch_save_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "tensor.pt"
    payload = {"tensor": torch.tensor([1.0, 2.0])}

    atomic_torch_save(target, payload)

    loaded = torch.load(target, map_location="cpu", weights_only=True)
    assert torch.equal(loaded["tensor"], payload["tensor"])
    assert len(sha256_file(target)) == 64


def test_atomic_write_preserves_existing_destination_when_replace_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source: os.PathLike[str], _destination: os.PathLike[str]) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(target, {"new": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.iterdir()) == [target]


def test_normalize_artifact_path_returns_relative_posix_path(tmp_path: Path) -> None:
    root = tmp_path / "run"
    target = root / "prediction" / "test_raw.pt"
    target.parent.mkdir(parents=True)
    target.touch()

    assert normalize_artifact_path(root, target) == "prediction/test_raw.pt"


def test_normalize_artifact_path_rejects_root_escape(tmp_path: Path) -> None:
    root = tmp_path / "run"
    root.mkdir()

    with pytest.raises(HardFailure, match="inside result root"):
        normalize_artifact_path(root, tmp_path / "escape.pt")


def test_experiment_layout_uses_fixed_paths(tmp_path: Path) -> None:
    root = tmp_path / "run"
    layout = ExperimentLayout(root)

    assert layout.preflight_dir == root / "preflight"
    assert layout.training_dir == root / "training"
    assert layout.prediction_dir == root / "prediction"
    assert layout.evaluation_dir == root / "evaluation"
    assert layout.validation == root / "validation.json"
    assert layout.result_manifest == root / "result_manifest.json"


def test_sibling_staging_publishes_a_complete_directory_atomically(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "published"

    with sibling_staging(destination) as staging:
        assert staging.parent == destination.parent
        (staging / "result.txt").write_text("complete", encoding="utf-8")

    assert (destination / "result.txt").read_text(encoding="utf-8") == "complete"


def test_sibling_staging_refuses_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "published"
    destination.mkdir()

    with pytest.raises(HardFailure, match="already exists"):
        with sibling_staging(destination):
            pass


@pytest.mark.parametrize(
    ("directory", "destination"),
    [
        ("training", "training/members/member_001.pt"),
        ("prediction", "prediction/test_raw.pt"),
        ("evaluation", "evaluation/legacy_equal_weight"),
    ],
)
def test_formal_result_paths_reject_internal_symlink_ancestors(
    tmp_path: Path, directory: str, destination: str
) -> None:
    """A staged formal write cannot escape through an internal result symlink."""
    root = tmp_path / "upet_fge_full"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    os.symlink(outside, root / directory, target_is_directory=True)

    with pytest.raises(HardFailure, match="ancestor is a symlink"):
        assert_safe_result_path(root, root / destination)


def test_formal_result_path_rejects_a_broken_destination_symlink(
    tmp_path: Path,
) -> None:
    """A broken staged evaluation destination must never be replaced or followed."""
    root = tmp_path / "upet_fge_full"
    destination = root / "evaluation" / "legacy_equal_weight"
    root.mkdir()
    destination.parent.mkdir()
    os.symlink(tmp_path / "outside", destination, target_is_directory=True)

    with pytest.raises(HardFailure):
        assert_safe_result_path(root, destination)
