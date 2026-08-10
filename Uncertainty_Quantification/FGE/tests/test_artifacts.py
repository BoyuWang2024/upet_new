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

    def record_replace(source, destination, **kwargs) -> None:
        observed["source"] = Path(source)
        observed["destination"] = Path(destination)
        observed["replace_kwargs"] = kwargs
        original_replace(source, destination, **kwargs)

    monkeypatch.setattr(os, "fsync", record_fsync)
    monkeypatch.setattr(os, "replace", record_replace)

    atomic_write_json(target, {"status": "PASS"})

    assert observed["fsync"] is True
    source = observed["source"]
    assert isinstance(source, Path)
    assert source.name.startswith(f".{target.name}.")
    assert observed["destination"] == Path(target.name)
    replace_kwargs = observed["replace_kwargs"]
    assert isinstance(replace_kwargs, dict)
    assert replace_kwargs["src_dir_fd"] == replace_kwargs["dst_dir_fd"]
    assert json.loads(target.read_text(encoding="utf-8")) == {"status": "PASS"}
    assert not (target.parent / source).exists()


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

    def fail_replace(_source, _destination, **_kwargs) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(target, {"new": True})

    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    retained = [path for path in tmp_path.iterdir() if path != target]
    assert len(retained) == 1
    assert retained[0].name.startswith(f".{target.name}.")


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


def test_failed_atomic_write_never_deletes_through_swapped_ancestor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "formal" / "training"
    parent.mkdir(parents=True)
    owned = parent.with_name(".owned_training")
    outside = tmp_path / "outside"
    outside.mkdir()
    target = parent / "manifest.json"
    observed: dict[str, Path] = {}

    def swap_and_fail(source, _destination, **_kwargs) -> None:
        temporary = Path(source)
        os.rename(parent, owned)
        parent.symlink_to(outside, target_is_directory=True)
        external = outside / temporary.name
        external.write_bytes(b"outside")
        observed["temporary"] = temporary
        observed["external"] = external
        raise OSError("replace failed")

    monkeypatch.setattr(os, "replace", swap_and_fail)
    with pytest.raises(OSError, match="replace failed"):
        atomic_write_json(target, {"status": "unpublished"})

    temporary = observed["temporary"]
    external = observed["external"]
    assert external.read_bytes() == b"outside"
    assert (owned / temporary.name).is_file()
    assert not target.exists()


def test_failed_sibling_staging_never_deletes_through_swapped_ancestor(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "formal"
    parent.mkdir()
    owned = tmp_path / ".owned_formal"
    outside = tmp_path / "outside"
    outside.mkdir()
    destination = parent / "published"
    observed: dict[str, Path] = {}

    with pytest.raises(RuntimeError, match="injected failure"):
        with sibling_staging(destination) as staging:
            (staging / "partial.txt").write_text("partial", encoding="utf-8")
            os.rename(parent, owned)
            parent.symlink_to(outside, target_is_directory=True)
            external = outside / staging.name
            external.mkdir()
            sentinel = external / "sentinel.txt"
            sentinel.write_text("outside", encoding="utf-8")
            observed["staging"] = staging
            observed["sentinel"] = sentinel
            raise RuntimeError("injected failure")

    staging = observed["staging"]
    sentinel = observed["sentinel"]
    assert sentinel.read_text(encoding="utf-8") == "outside"
    assert (owned / staging.name / "partial.txt").is_file()


def test_atomic_replace_remains_bound_when_parent_is_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "formal" / "training"
    parent.mkdir(parents=True)
    owned = parent.with_name(".owned_training")
    outside = tmp_path / "outside_bound"
    outside.mkdir()
    target = parent / "manifest.json"
    real_replace = os.replace
    observed: dict[str, Path] = {}

    def swap_then_replace(source, destination, **kwargs):
        os.rename(parent, owned)
        parent.symlink_to(outside, target_is_directory=True)
        external_source = outside / Path(source).name
        external_source.write_bytes(b"attacker")
        external_target = outside / Path(destination).name
        external_target.write_bytes(b"outside")
        observed["source"] = external_source
        observed["target"] = external_target
        return real_replace(source, destination, **kwargs)

    monkeypatch.setattr(os, "replace", swap_then_replace)
    atomic_write_json(target, {"status": "PASS"})

    assert observed["source"].read_bytes() == b"attacker"
    assert observed["target"].read_bytes() == b"outside"
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


def test_sibling_publication_remains_bound_when_parent_is_swapped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "formal_publish"
    parent.mkdir()
    owned = tmp_path / ".owned_formal_publish"
    outside = tmp_path / "outside_publish"
    outside.mkdir()
    destination = parent / "published"
    real_replace = os.replace
    observed: dict[str, Path] = {}

    def swap_then_replace(source, target, **kwargs):
        bound = kwargs.get("dst_dir_fd")
        if bound is not None:
            bound_parent = (Path("/proc/self/fd") / str(bound)).resolve()
            if bound_parent != parent:
                return real_replace(source, target, **kwargs)
        elif Path(target) != destination:
            return real_replace(source, target, **kwargs)
        os.rename(parent, owned)
        parent.symlink_to(outside, target_is_directory=True)
        attacker = outside / Path(source).name
        attacker.mkdir()
        (attacker / "attacker.txt").write_text("outside", encoding="utf-8")
        observed["attacker"] = attacker
        return real_replace(source, target, **kwargs)

    monkeypatch.setattr(os, "replace", swap_then_replace)
    with sibling_staging(destination) as staging:
        (staging / "result.txt").write_text("complete", encoding="utf-8")

    assert (observed["attacker"] / "attacker.txt").is_file()
    assert not (outside / destination.name).exists()
    assert (owned / destination.name / "result.txt").read_text(
        encoding="utf-8"
    ) == "complete"
