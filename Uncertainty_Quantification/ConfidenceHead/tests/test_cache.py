from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head import (
    cache as cache_module,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.cache import (
    CachedSplitDataset,
    RawStructure,
    build_raw_cache,
    collate_cached_structures,
    prepare_raw_cache,
)


SCHEMA_VERSION = "upet_confidence_raw_cache_v1"
ATOM_FIELDS = (
    "atomic_numbers",
    "force_prediction",
    "force_reference",
    "force_features",
    "energy_features",
)
ENERGY_FIELDS = ("energy_prediction", "energy_reference")
FORBIDDEN_DERIVED_FIELDS = {
    "labels",
    "bin_indices",
    "logits",
    "expected_uncertainty",
}


def _raw_structure(structure_id: int, num_atoms: int) -> RawStructure:
    atom_values = torch.arange(num_atoms, dtype=torch.float32)
    force_features = torch.stack((atom_values, atom_values + 0.25), dim=1)
    energy_features = torch.stack((atom_values + 10.0, atom_values + 20.0), dim=1)
    assert force_features.untyped_storage().data_ptr() != (
        energy_features.untyped_storage().data_ptr()
    )
    return RawStructure(
        structure_id=structure_id,
        atomic_numbers=torch.arange(1, num_atoms + 1, dtype=torch.int64),
        force_prediction=torch.stack(
            (atom_values, atom_values + 1.0, atom_values + 2.0),
            dim=1,
        ),
        force_reference=torch.stack(
            (atom_values + 3.0, atom_values + 4.0, atom_values + 5.0),
            dim=1,
        ),
        energy_prediction=torch.tensor(structure_id / 10, dtype=torch.float32),
        energy_reference=float(structure_id / 10 + 0.5),
        force_features=force_features,
        energy_features=energy_features,
    )


@pytest.fixture
def raw_structures() -> list[RawStructure]:
    return [
        _raw_structure(101, 2),
        _raw_structure(102, 3),
        _raw_structure(103, 1),
    ]


@pytest.fixture
def identity_payload() -> dict[str, Any]:
    return {
        "checkpoint": {"sha256": "a" * 64},
        "readouts": {
            "force_prediction": "non_conservative_forces",
            "force_features": "mtt::aux::last_layer_features",
            "energy_prediction": "energy",
            "energy_features": "mtt::aux::energy_last_layer_features",
        },
        "splits": {"train": {"sha256": "b" * 64}},
    }


def _build(
    output_root: Path,
    structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> tuple[Path, dict[str, Any]]:
    manifest_path = build_raw_cache(
        output_root=output_root,
        split_structures={"train": iter(structures)},
        identity_payload=identity_payload,
        shard_max_atoms=4,
    )
    return manifest_path, json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_shards(
    manifest_path: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    return [
        torch.load(
            manifest_path.parent / shard["path"],
            map_location="cpu",
            weights_only=True,
            mmap=True,
        )
        for shard in manifest["splits"]["train"]["shards"]
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contains_forbidden_field(value: Any) -> bool:
    if isinstance(value, dict):
        return bool(FORBIDDEN_DERIVED_FIELDS.intersection(value)) or any(
            _contains_forbidden_field(item) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_forbidden_field(item) for item in value)
    return False


def test_raw_structure_is_frozen(raw_structures: list[RawStructure]) -> None:
    with pytest.raises(FrozenInstanceError):
        raw_structures[0].structure_id = 999  # type: ignore[misc]


def test_build_writes_complete_manifest_and_exact_shard_schema(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(
        tmp_path / "cache",
        raw_structures,
        identity_payload,
    )

    assert manifest_path == tmp_path / "cache" / manifest["cache_id"] / "manifest.json"
    assert manifest["status"] == "complete"
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["identity"] == manifest["cache_id"]
    assert manifest["identity_payload"] == identity_payload

    split = manifest["splits"]["train"]
    assert split["structures"] == 3
    assert split["atoms"] == 6
    assert split["force_components"] == 18
    assert len(split["shards"]) == 2
    assert split["structure_to_shard"] == [0, 1, 1]
    assert split["structure_to_index"] == [0, 0, 1]

    shards = _load_shards(manifest_path, manifest)
    expected_structure_ids = ([101], [102, 103])
    expected_num_atoms = ([2], [3, 1])
    expected_atoms = (2, 4)
    for index, (entry, shard) in enumerate(zip(split["shards"], shards, strict=True)):
        shard_path = manifest_path.parent / entry["path"]
        assert not Path(entry["path"]).is_absolute()
        assert entry == {
            "path": entry["path"],
            "sha256": _sha256(shard_path),
            "structures": len(expected_structure_ids[index]),
            "atoms": expected_atoms[index],
            "force_components": 3 * expected_atoms[index],
            "force_feature_dim": 2,
            "energy_feature_dim": 2,
        }
        assert shard["schema_version"] == SCHEMA_VERSION
        assert shard["split"] == "train"
        torch.testing.assert_close(
            shard["structure_ids"],
            torch.tensor(expected_structure_ids[index], dtype=torch.int64),
        )
        torch.testing.assert_close(
            shard["num_atoms"],
            torch.tensor(expected_num_atoms[index], dtype=torch.int64),
        )
        expected_dtypes = {
            "structure_ids": torch.int64,
            "num_atoms": torch.int64,
            "atom_offsets": torch.int64,
            "atomic_numbers": torch.int64,
            "force_prediction": torch.float32,
            "force_reference": torch.float32,
            "energy_prediction": torch.float32,
            "energy_reference": torch.float32,
            "force_features": torch.float32,
            "energy_features": torch.float32,
        }
        assert set(shard) == {"schema_version", "split", *expected_dtypes}
        for field, dtype in expected_dtypes.items():
            assert shard[field].dtype == dtype
        assert shard["atomic_numbers"].shape == (expected_atoms[index],)
        assert shard["force_prediction"].shape == (expected_atoms[index], 3)
        assert shard["force_reference"].shape == (expected_atoms[index], 3)
        assert shard["energy_prediction"].shape == (len(expected_structure_ids[index]),)
        assert shard["energy_reference"].shape == (len(expected_structure_ids[index]),)
        assert shard["force_features"].shape == (expected_atoms[index], 2)
        assert shard["energy_features"].shape == (expected_atoms[index], 2)
        assert shard["force_features"].untyped_storage().data_ptr() != (
            shard["energy_features"].untyped_storage().data_ptr()
        )

    assert sum(entry["structures"] for entry in split["shards"]) == split["structures"]
    assert sum(entry["atoms"] for entry in split["shards"]) == split["atoms"]
    assert (
        sum(entry["force_components"] for entry in split["shards"])
        == split["force_components"]
    )
    assert not _contains_forbidden_field(manifest)
    assert all(not _contains_forbidden_field(shard) for shard in shards)


def test_shards_flush_only_at_whole_structure_boundaries(
    tmp_path: Path,
    identity_payload: dict[str, Any],
) -> None:
    structures = [
        _raw_structure(1, 2),
        _raw_structure(2, 5),
        _raw_structure(3, 1),
    ]
    manifest_path, manifest = _build(tmp_path, structures, identity_payload)
    shards = _load_shards(manifest_path, manifest)

    assert [shard["num_atoms"].tolist() for shard in shards] == [[2], [5], [1]]
    for shard in shards:
        offsets = shard["atom_offsets"]
        assert offsets[0].item() == 0
        assert offsets[-1].item() == shard["atomic_numbers"].shape[0]
        assert torch.all(offsets[1:] > offsets[:-1])


def test_duplicate_structure_id_leaves_only_an_incomplete_manifest(
    tmp_path: Path,
    identity_payload: dict[str, Any],
) -> None:
    structures = [
        _raw_structure(41, 2),
        _raw_structure(42, 3),
        _raw_structure(41, 1),
    ]

    with pytest.raises(ValueError, match=r"train.*41"):
        build_raw_cache(
            tmp_path,
            {"train": structures},
            identity_payload,
            shard_max_atoms=4,
        )

    manifests = list(tmp_path.glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
    assert manifest["status"] == "incomplete"
    with pytest.raises(ValueError, match="complete"):
        CachedSplitDataset(
            manifests[0],
            split="train",
            expected_identity=manifest["identity"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("force_prediction", float("nan")),
        ("force_reference", float("inf")),
        ("force_features", float("-inf")),
        ("energy_features", float("nan")),
        ("energy_prediction", float("inf")),
        ("energy_reference", float("nan")),
    ],
)
def test_nonfinite_values_report_split_destination_shard_structure_and_field(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
    field: str,
    value: float,
) -> None:
    invalid = getattr(raw_structures[1], field)
    if isinstance(invalid, torch.Tensor):
        invalid = invalid.clone()
        invalid.reshape(-1)[0] = value
    else:
        invalid = value
    raw_structures[1] = replace(raw_structures[1], **{field: invalid})

    with pytest.raises(
        ValueError,
        match=rf"train.*shard 1.*102.*{field}",
    ):
        build_raw_cache(
            tmp_path / field,
            {"train": raw_structures},
            identity_payload,
            shard_max_atoms=4,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "atomic_numbers",
            torch.tensor([1.0, 2.0]),
            r"atomic_numbers.*int64",
        ),
        (
            "force_prediction",
            torch.zeros((2, 2)),
            r"force_prediction.*\(2, 3\)",
        ),
        (
            "force_features",
            torch.zeros((3, 2)),
            r"force_features.*2",
        ),
        (
            "energy_prediction",
            torch.zeros(2),
            r"energy_prediction.*scalar",
        ),
    ],
)
def test_invalid_dtype_shape_or_atom_count_is_rejected_clearly(
    tmp_path: Path,
    identity_payload: dict[str, Any],
    field: str,
    value: Any,
    message: str,
) -> None:
    invalid = replace(_raw_structure(73, 2), **{field: value})

    with pytest.raises(ValueError, match=rf"train.*73.*{message}"):
        build_raw_cache(
            tmp_path / field,
            {"train": [invalid]},
            identity_payload,
            shard_max_atoms=4,
        )


def test_dataset_requires_complete_manifest_and_matching_identity(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)

    with pytest.raises(ValueError, match="identity"):
        CachedSplitDataset(
            manifest_path,
            split="train",
            expected_identity="cache-wrong",
        )

    manifest["status"] = "incomplete"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="complete"):
        CachedSplitDataset(
            manifest_path,
            split="train",
            expected_identity=manifest["identity"],
        )


def test_dataset_mmap_loads_only_needed_shards_and_uses_small_lru(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    original_load = torch.load
    calls: list[tuple[Path, dict[str, Any]]] = []

    def counting_load(path: Path, **kwargs: Any) -> Any:
        calls.append((Path(path), kwargs))
        return original_load(path, **kwargs)

    monkeypatch.setattr(cache_module.torch, "load", counting_load)
    dataset = CachedSplitDataset(
        manifest_path,
        split="train",
        expected_identity=manifest["identity"],
        max_cached_shards=1,
    )
    assert len(dataset) == 3
    assert calls == []

    assert dataset[0]["structure_id"] == 101
    assert dataset[0]["structure_id"] == 101
    assert dataset[1]["structure_id"] == 102
    assert dataset[2]["structure_id"] == 103
    assert dataset[0]["structure_id"] == 101

    assert [path.name for path, _ in calls] == [
        Path(manifest["splits"]["train"]["shards"][0]["path"]).name,
        Path(manifest["splits"]["train"]["shards"][1]["path"]).name,
        Path(manifest["splits"]["train"]["shards"][0]["path"]).name,
    ]
    assert all(
        kwargs
        == {
            "map_location": "cpu",
            "weights_only": True,
            "mmap": True,
        }
        for _, kwargs in calls
    )


def test_dataset_returns_structure_slices_and_collate_rebuilds_offsets(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    dataset = CachedSplitDataset(
        manifest_path,
        split="train",
        expected_identity=manifest["identity"],
    )

    first = dataset[0]
    second = dataset[1]
    assert first["structure_id"] == 101
    assert first["num_atoms"] == 2
    for field in ATOM_FIELDS:
        torch.testing.assert_close(first[field], getattr(raw_structures[0], field))
    for field in ENERGY_FIELDS:
        expected = torch.as_tensor(
            getattr(raw_structures[0], field),
            dtype=torch.float32,
        )
        torch.testing.assert_close(first[field], expected)

    batch = collate_cached_structures([first, second])
    torch.testing.assert_close(
        batch["structure_ids"],
        torch.tensor([101, 102], dtype=torch.int64),
    )
    torch.testing.assert_close(
        batch["num_atoms"],
        torch.tensor([2, 3], dtype=torch.int64),
    )
    torch.testing.assert_close(
        batch["atom_offsets"],
        torch.tensor([0, 2, 5], dtype=torch.int64),
    )
    for field in ATOM_FIELDS:
        torch.testing.assert_close(
            batch[field],
            torch.cat(
                (getattr(raw_structures[0], field), getattr(raw_structures[1], field))
            ),
        )
    for field in ENERGY_FIELDS:
        torch.testing.assert_close(
            batch[field],
            torch.tensor(
                [
                    getattr(raw_structures[0], field),
                    getattr(raw_structures[1], field),
                ],
                dtype=torch.float32,
            ),
        )


def test_dataset_rejects_missing_or_corrupted_shard(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    first_shard = (
        manifest_path.parent / manifest["splits"]["train"]["shards"][0]["path"]
    )
    first_shard.unlink()
    dataset = CachedSplitDataset(
        manifest_path,
        split="train",
        expected_identity=manifest["identity"],
    )
    with pytest.raises(ValueError, match=r"shard.*missing"):
        dataset[0]

    manifest_path, manifest = _build(
        tmp_path / "corrupt",
        raw_structures,
        identity_payload,
    )
    first_shard = (
        manifest_path.parent / manifest["splits"]["train"]["shards"][0]["path"]
    )
    first_shard.write_bytes(first_shard.read_bytes() + b"tampered")
    dataset = CachedSplitDataset(
        manifest_path,
        split="train",
        expected_identity=manifest["identity"],
    )
    with pytest.raises(ValueError, match=r"shard.*sha256"):
        dataset[0]


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ({"structure_to_shard": [0, 1]}, "structure_to_shard"),
        ({"structure_to_index": [0, 0, 0, 0]}, "structure_to_index"),
        ({"structure_to_shard": [0, 1, 0]}, "structure_to_shard"),
    ],
)
def test_dataset_rejects_missing_duplicate_or_inconsistent_structure_mapping(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
    mapping: dict[str, list[int]],
    message: str,
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    manifest["splits"]["train"].update(mapping)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        CachedSplitDataset(
            manifest_path,
            split="train",
            expected_identity=manifest["identity"],
        )


def test_cache_identity_is_deterministic_and_output_path_independent(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    first_path, first = _build(
        tmp_path / "absolute" / "one",
        raw_structures,
        identity_payload,
    )
    second_path, second = _build(
        tmp_path / "different" / "two",
        raw_structures,
        identity_payload,
    )

    assert first["cache_id"] == second["cache_id"]
    assert first["identity"] == second["identity"]
    assert first_path.parent.name == second_path.parent.name == first["cache_id"]
    encoded_manifest = json.dumps(first)
    assert str(tmp_path) not in encoded_manifest
    assert FORBIDDEN_DERIVED_FIELDS.isdisjoint(identity_payload)


def test_manifest_records_normative_split_feature_metadata(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    _, manifest = _build(tmp_path, raw_structures, identity_payload)
    split = manifest["splits"]["train"]
    assert split["structure_count"] == 3
    assert split["atom_count"] == 6
    assert split["force_component_count"] == 18
    assert split["force_feature_dim"] == 2
    assert split["energy_feature_dim"] == 2
    assert split["feature_dtype"] == "float32"


def test_prepare_creates_independent_incomplete_staging(tmp_path: Path) -> None:
    first = prepare_raw_cache(tmp_path)
    second = prepare_raw_cache(tmp_path)
    assert first.root != second.root
    for staging in (first, second):
        manifest = json.loads(staging.manifest_path.read_text(encoding="utf-8"))
        assert manifest["status"] == "incomplete"
        assert manifest["splits"] == {}


def test_valid_complete_cache_is_reused_without_changing_bytes(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    paths = [manifest_path]
    paths.extend(
        manifest_path.parent / item["path"]
        for item in manifest["splits"]["train"]["shards"]
    )
    before = {path: path.read_bytes() for path in paths}

    rebuilt = build_raw_cache(
        tmp_path,
        {"train": (_ for _ in ())},
        identity_payload,
        shard_max_atoms=4,
    )

    assert rebuilt == manifest_path
    assert {path: path.read_bytes() for path in paths} == before


def test_structural_corruption_is_rejected_even_with_updated_sha(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
) -> None:
    manifest_path, manifest = _build(tmp_path, raw_structures, identity_payload)
    entry = manifest["splits"]["train"]["shards"][0]
    shard_path = manifest_path.parent / entry["path"]
    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
    shard["atom_offsets"] = shard["atom_offsets"].clone()
    shard["atom_offsets"][-1] += 1
    torch.save(shard, shard_path)
    entry["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    before_manifest = manifest_path.read_bytes()
    before_shard = shard_path.read_bytes()

    with pytest.raises(ValueError, match="atom_offsets"):
        CachedSplitDataset(
            manifest_path,
            split="train",
            expected_identity=manifest["identity"],
        )[0]
    with pytest.raises(ValueError, match="atom_offsets"):
        build_raw_cache(
            tmp_path,
            {"train": raw_structures},
            identity_payload,
            shard_max_atoms=4,
        )

    assert manifest_path.read_bytes() == before_manifest
    assert shard_path.read_bytes() == before_shard


def test_publication_strictly_reloads_every_shard(
    tmp_path: Path,
    raw_structures: list[RawStructure],
    identity_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = torch.load
    calls: list[dict[str, Any]] = []

    def recording_load(path: Path, **kwargs: Any) -> Any:
        calls.append(kwargs)
        return original(path, **kwargs)

    monkeypatch.setattr(cache_module.torch, "load", recording_load)
    _, manifest = _build(tmp_path, raw_structures, identity_payload)
    shard_count = len(manifest["splits"]["train"]["shards"])
    assert len(calls) == 2 * shard_count
    assert all(
        call == {"map_location": "cpu", "weights_only": True, "mmap": True}
        for call in calls
    )
