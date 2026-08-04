from __future__ import annotations

import json
import pickle
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head import (
    cache as cache_module,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.cache import (
    SCHEMA_VERSION,
    CachedSplitDataset,
    RawStructure,
    _validate_complete_cache,
    build_raw_cache,
    collate_cached_structures,
    prepare_raw_cache,
)


ATOM_FIELDS = (
    "atomic_numbers",
    "force_prediction",
    "force_reference",
    "force_features",
    "energy_features",
)
ENERGY_FIELDS = ("energy_prediction", "energy_reference")


def _raw_structure(structure_id: int, num_atoms: int) -> RawStructure:
    atom_values = torch.arange(num_atoms, dtype=torch.float32)
    return RawStructure(
        structure_id=structure_id,
        atomic_numbers=torch.arange(1, num_atoms + 1, dtype=torch.int64),
        force_prediction=torch.stack(
            (atom_values, atom_values + 1.0, atom_values + 2.0), dim=1
        ),
        force_reference=torch.stack(
            (atom_values + 3.0, atom_values + 4.0, atom_values + 5.0), dim=1
        ),
        energy_prediction=torch.tensor(structure_id / 10, dtype=torch.float32),
        energy_reference=float(structure_id / 10 + 0.5),
        force_features=torch.stack((atom_values, atom_values + 0.25), dim=1),
        energy_features=torch.stack((atom_values + 10.0, atom_values + 20.0), dim=1),
    )


@pytest.fixture
def raw_structures() -> list[RawStructure]:
    return [_raw_structure(101, 2), _raw_structure(102, 3), _raw_structure(103, 1)]


@pytest.fixture
def identity_payload() -> dict[str, Any]:
    return {
        "checkpoint": {"sha256": "a" * 64},
        "readouts": {
            "force_prediction": "non_conservative_forces",
            "force_features": "mtt::aux::non_conservative_forces_last_layer_features",
            "energy_prediction": "energy",
            "energy_features": "mtt::aux::energy_last_layer_features",
        },
        "splits": {
            "train": {
                "sha256": "b" * 64,
                "structure_count": 3,
                "atom_count": 6,
                "force_component_count": 18,
            }
        },
    }


def _build(
    root: Path,
    structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    atoms = sum(len(structure.atomic_numbers) for structure in structures)
    payload = {
        **identity_payload,
        "splits": {
            "train": {
                **identity_payload["splits"]["train"],
                "structure_count": len(structures),
                "atom_count": atoms,
                "force_component_count": 3 * atoms,
            }
        },
    }
    path = build_raw_cache(root, {"train": structures}, payload)
    return path, json.loads(path.read_text(encoding="utf-8"))


def test_raw_structure_is_frozen(raw_structures: list[RawStructure]) -> None:
    with pytest.raises(FrozenInstanceError):
        raw_structures[0].structure_id = 999  # type: ignore[misc]


def test_build_writes_v2_continuous_array_manifest(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)

    assert SCHEMA_VERSION == "upet_confidence_raw_cache_v2"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["status"] == "complete"
    assert manifest_path == tmp_path / manifest["cache_id"] / "manifest.json"
    split = manifest["splits"]["train"]
    assert split["structure_count"] == 3
    assert split["atom_count"] == 6
    assert split["force_feature_dim"] == 2
    assert split["energy_feature_dim"] == 2
    assert set(split["arrays"]) == {
        "structure_offsets",
        "structure_ids",
        *ATOM_FIELDS,
        *ENERGY_FIELDS,
    }
    assert "shards" not in split
    for field, descriptor in split["arrays"].items():
        path = manifest_path.parent / descriptor["path"]
        assert path.is_file(), field
        assert not Path(descriptor["path"]).is_absolute()
        assert descriptor["bytes"] == path.stat().st_size
        assert len(descriptor["sha256"]) == 64


def test_writer_streams_into_preallocated_arrays_without_tensor_concatenation(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_cat(*args: Any, **kwargs: Any) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("cache writer must not materialize a split with torch.cat")

    monkeypatch.setattr(torch, "cat", forbidden_cat)
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)

    assert manifest_path.is_file()
    assert manifest["splits"]["train"]["atom_count"] == 6


def test_dataset_random_slices_and_collate_are_exact(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    dataset = CachedSplitDataset(manifest_path, "train", manifest["identity"])

    assert len(dataset) == 3
    assert dataset.num_atoms(0) == 2
    assert dataset.num_atoms(1) == 3
    assert dataset[-1]["structure_id"] == 103
    first, second = dataset[0], dataset[1]
    for field in ATOM_FIELDS:
        torch.testing.assert_close(first[field], getattr(raw_structures[0], field))
    for field in ENERGY_FIELDS:
        torch.testing.assert_close(
            first[field],
            torch.as_tensor(getattr(raw_structures[0], field), dtype=torch.float32),
        )

    batch = collate_cached_structures([first, second])
    torch.testing.assert_close(batch["atom_counts"], torch.tensor([2, 3]))
    torch.testing.assert_close(batch["num_atoms"], torch.tensor([2, 3]))
    torch.testing.assert_close(batch["atom_offsets"], torch.tensor([0, 2, 5]))
    assert batch["force_features"].is_contiguous()
    assert batch["energy_features"].is_contiguous()


def test_dataset_fast_path_never_hashes_arrays(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)

    def forbidden_hash(path: Path) -> str:
        raise AssertionError(f"unexpected training-time hash: {path}")

    monkeypatch.setattr(cache_module, "sha256_file", forbidden_hash)
    dataset = CachedSplitDataset(manifest_path, "train", manifest["identity"])
    assert dataset[2]["structure_id"] == 103
    assert dataset[0]["structure_id"] == 101


def test_dataset_pickle_drops_open_memmaps_and_reopens_lazily(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    dataset = CachedSplitDataset(manifest_path, "train", manifest["identity"])
    dataset[0]
    assert dataset._arrays

    restored = pickle.loads(pickle.dumps(dataset))

    assert restored._arrays == {}
    assert restored[1]["structure_id"] == 102
    assert restored._arrays


def test_full_validation_detects_mutated_array_byte(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    descriptor = manifest["splits"]["train"]["arrays"]["force_features"]
    path = manifest_path.parent / descriptor["path"]
    with path.open("r+b") as handle:
        handle.seek(-1, 2)
        final = handle.read(1)
        handle.seek(-1, 2)
        handle.write(bytes([final[0] ^ 1]))

    CachedSplitDataset(manifest_path, "train", manifest["identity"])
    with pytest.raises(ValueError, match="force_features.*sha256"):
        _validate_complete_cache(
            manifest_path,
            expected_identity=manifest["identity"],
            load_shards=True,
        )


def test_fast_validation_rejects_wrong_array_size(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    descriptor = manifest["splits"]["train"]["arrays"]["energy_reference"]
    path = manifest_path.parent / descriptor["path"]
    path.write_bytes(path.read_bytes()[:-1])

    with pytest.raises(ValueError, match="energy_reference.*size"):
        CachedSplitDataset(manifest_path, "train", manifest["identity"])


def test_nonmonotonic_offsets_are_rejected_even_with_updated_metadata(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    descriptor = manifest["splits"]["train"]["arrays"]["structure_offsets"]
    path = manifest_path.parent / descriptor["path"]
    offsets = np.load(path, mmap_mode="r+")
    offsets[1] = offsets[0]
    offsets.flush()

    with pytest.raises(ValueError, match="offsets.*increasing"):
        CachedSplitDataset(manifest_path, "train", manifest["identity"])


def test_force_and_energy_features_use_independent_memmaps(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    dataset = CachedSplitDataset(manifest_path, "train", manifest["identity"])
    item = dataset[1]

    assert item["force_features"].untyped_storage().data_ptr() != (
        item["energy_features"].untyped_storage().data_ptr()
    )


def test_invalid_structure_leaves_only_incomplete_staging(
    tmp_path: Path,
    identity_payload: dict[str, Any],
) -> None:
    invalid = replace(
        _raw_structure(7, 2),
        force_features=torch.tensor([[float("nan"), 0.0], [1.0, 2.0]]),
    )

    with pytest.raises(ValueError, match="train.*7.*force_features"):
        build_raw_cache(tmp_path, {"train": [invalid]}, identity_payload)

    manifests = list(tmp_path.glob(".staging-*/manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["status"] == "incomplete"


def test_duplicate_structure_id_is_rejected(
    tmp_path: Path,
    identity_payload: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="duplicate structure ID 8"):
        build_raw_cache(
            tmp_path,
            {"train": [_raw_structure(8, 1), _raw_structure(8, 2)]},
            identity_payload,
        )


def test_prepare_raw_cache_creates_independent_incomplete_staging(
    tmp_path: Path,
) -> None:
    first = prepare_raw_cache(tmp_path)
    second = prepare_raw_cache(tmp_path)
    assert first.root != second.root
    assert json.loads(first.manifest_path.read_text())["status"] == "incomplete"
    assert json.loads(second.manifest_path.read_text())["status"] == "incomplete"


@pytest.mark.parametrize("split", ["/absolute", "../escape", "nested/name", "", "."])
def test_unsafe_split_name_is_rejected_before_write(
    tmp_path: Path,
    identity_payload: dict[str, Any],
    split: str,
) -> None:
    with pytest.raises(ValueError, match="unsafe cache split"):
        build_raw_cache(tmp_path, {split: [_raw_structure(1, 1)]}, identity_payload)


def test_schema_v1_manifest_is_explicitly_rejected(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "upet_confidence_raw_cache_v1",
                "status": "complete",
            }
        )
    )

    with pytest.raises(ValueError, match="schema"):
        CachedSplitDataset(manifest, "train", "cache-anything")
