from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
import yaml

from Uncertainty_Quantification.FGE.fge import artifacts
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


def test_atomic_json_fsyncs_an_unnamed_inode_then_links_exact_fd(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result.json"
    observed: dict[str, object] = {}
    original_fsync = os.fsync
    original_link = artifacts._link_open_file

    def record_fsync(fd: int) -> None:
        observed["fsync"] = True
        original_fsync(fd)

    def record_link(fd: int, parent_fd: int, name: str) -> None:
        observed["inode"] = (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
        original_link(fd, parent_fd, name)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(artifacts, "_link_open_file", record_link)
    atomic_write_json(target, {"status": "PASS"})

    assert observed["fsync"] is True
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert observed["inode"] == (target.stat().st_dev, target.stat().st_ino)
    assert list(tmp_path.iterdir()) == [target]


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


def test_atomic_write_refuses_to_replace_existing_immutable_destination(
    tmp_path: Path,
) -> None:
    target = tmp_path / "result.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    with pytest.raises(HardFailure, match="immutable artifact destination"):
        atomic_write_json(target, {"new": True})

    assert target.read_text(encoding="utf-8") == '{"old": true}\n'


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
        assert staging.is_dir()
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


def test_atomic_publish_fails_if_configured_parent_identity_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "formal" / "training"
    parent.mkdir(parents=True)
    owned = parent.with_name(".owned_training")
    outside = tmp_path / "outside"
    outside.mkdir()
    sentinel = outside / "manifest.json"
    sentinel.write_bytes(b"outside")
    target = parent / "manifest.json"
    original_link = artifacts._link_open_file

    def swap_then_link(fd: int, parent_fd: int, name: str) -> None:
        os.rename(parent, owned)
        parent.symlink_to(outside, target_is_directory=True)
        original_link(fd, parent_fd, name)

    monkeypatch.setattr(artifacts, "_link_open_file", swap_then_link)
    with pytest.raises(HardFailure, match="descriptor-bound|parent identity"):
        atomic_write_json(target, {"status": "unpublished"})

    assert sentinel.read_bytes() == b"outside"
    assert json.loads((owned / target.name).read_text(encoding="utf-8")) == {
        "status": "unpublished"
    }


def test_failed_sibling_staging_retains_bound_directory_after_ancestor_swap(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "formal"
    parent.mkdir()
    owned = tmp_path / ".owned_formal"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "published"
    staging_name = ""

    with pytest.raises(RuntimeError, match="injected failure"):
        with sibling_staging(destination) as staging:
            resolved = staging.resolve()
            staging_name = resolved.name
            (staging / "partial.txt").write_text("partial", encoding="utf-8")
            os.rename(parent, owned)
            parent.symlink_to(outside, target_is_directory=True)
            raise RuntimeError("injected failure")

    assert (owned / staging_name / "partial.txt").is_file()
    assert not any(outside.iterdir())


def test_atomic_link_cannot_be_redirected_by_parent_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "formal" / "training"
    parent.mkdir(parents=True)
    owned = parent.with_name(".owned_training")
    outside = tmp_path / "outside_bound"
    outside.mkdir()
    target = parent / "manifest.json"
    sentinel = outside / target.name
    sentinel.write_bytes(b"outside")
    original_link = artifacts._link_open_file

    def swap_then_link(fd: int, parent_fd: int, name: str) -> None:
        os.rename(parent, owned)
        parent.symlink_to(outside, target_is_directory=True)
        original_link(fd, parent_fd, name)

    monkeypatch.setattr(artifacts, "_link_open_file", swap_then_link)
    with pytest.raises(HardFailure, match="descriptor-bound|parent identity"):
        atomic_write_json(target, {"status": "PASS"})

    assert sentinel.read_bytes() == b"outside"
    assert json.loads((owned / target.name).read_text(encoding="utf-8")) == {
        "status": "PASS"
    }


def test_artifact_writes_fail_closed_without_secure_primitives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(artifacts, "_SECURE_ARTIFACT_PRIMITIVES", False)
    target = tmp_path / "formal" / "manifest.json"

    with pytest.raises(HardFailure, match="race-safe artifact operations"):
        atomic_write_json(target, {"status": "blocked"})

    assert not target.parent.exists()


def test_sibling_publication_rejects_parent_identity_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "formal_publish"
    parent.mkdir()
    owned = tmp_path / ".owned_formal_publish"
    outside = tmp_path / "outside_publish"
    outside.mkdir()
    destination = parent / "published"
    original_rename = artifacts._rename_directory_noreplace

    def swap_then_rename(parent_fd: int, source: str, target: str) -> None:
        os.rename(parent, owned)
        parent.symlink_to(outside, target_is_directory=True)
        original_rename(parent_fd, source, target)

    monkeypatch.setattr(artifacts, "_rename_directory_noreplace", swap_then_rename)
    with pytest.raises(HardFailure, match="descriptor-bound|parent identity"):
        with sibling_staging(destination) as staging:
            (staging / "result.txt").write_text("complete", encoding="utf-8")

    assert not (outside / destination.name).exists()
    assert (owned / destination.name / "result.txt").read_text(
        encoding="utf-8"
    ) == "complete"


def test_atomic_publish_never_exposes_a_swappable_source_basename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "result.json"
    original_link = artifacts._link_open_file

    def attacker_creates_destination(fd: int, parent_fd: int, name: str) -> None:
        attacker = os.open(
            name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600, dir_fd=parent_fd
        )
        os.write(attacker, b"attacker")
        os.close(attacker)
        original_link(fd, parent_fd, name)

    monkeypatch.setattr(artifacts, "_link_open_file", attacker_creates_destination)
    with pytest.raises(HardFailure, match="already exists|descriptor-bound"):
        atomic_write_json(target, {"status": "PASS"})

    assert target.read_bytes() == b"attacker"


def test_sibling_staging_rejects_source_basename_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "published"
    original_rename = artifacts._rename_directory_noreplace
    observed: dict[str, str] = {}

    def substitute(parent_fd: int, source: str, target: str) -> None:
        owned = f".owned-{source}"
        os.rename(source, owned, src_dir_fd=parent_fd, dst_dir_fd=parent_fd)
        os.mkdir(source, 0o700, dir_fd=parent_fd)
        attacker_fd = os.open(source, artifacts._DIRECTORY_FLAGS, dir_fd=parent_fd)
        try:
            marker = os.open(
                "attacker.txt",
                os.O_WRONLY | os.O_CREAT | os.O_EXCL,
                0o600,
                dir_fd=attacker_fd,
            )
            os.write(marker, b"attacker")
            os.close(marker)
        finally:
            os.close(attacker_fd)
        observed["owned"] = owned
        original_rename(parent_fd, source, target)

    monkeypatch.setattr(artifacts, "_rename_directory_noreplace", substitute)
    with pytest.raises(HardFailure, match="identity differs"):
        with sibling_staging(destination) as staging:
            (staging / "result.txt").write_text("complete", encoding="utf-8")

    assert (destination / "attacker.txt").read_bytes() == b"attacker"
    assert not (destination / "result_manifest.json").exists()
    assert (tmp_path / observed["owned"] / "result.txt").read_text(
        encoding="utf-8"
    ) == "complete"
