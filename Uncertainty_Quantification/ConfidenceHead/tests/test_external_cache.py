from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from confidence_head.cache import RawStructure, build_raw_cache
from confidence_head.external_cache import (
    build_external_dataset_cache,
    resolve_declared_cache_split,
)
from confidence_head.external_config import (
    CacheSplitSource,
    ExternalPredictionConfig,
    ExtXYZSource,
)


DATA_SHA = "1" * 64
CHECKPOINT_SHA = "2" * 64
READOUTS = {
    "energy_prediction": "energy",
    "force_prediction": "non_conservative_forces",
    "energy_features": "energy_features",
    "force_features": "force_features",
}


def _raw(structure_id: int = 7) -> RawStructure:
    return RawStructure(
        structure_id=structure_id,
        atomic_numbers=torch.tensor([1, 8], dtype=torch.int64),
        force_prediction=torch.zeros((2, 3)),
        force_reference=torch.ones((2, 3)),
        energy_prediction=torch.tensor(2.0),
        energy_reference=torch.tensor(4.0),
        force_features=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        energy_features=torch.arange(6, dtype=torch.float32).reshape(2, 3),
    )


def _identity_payload(split: str = "train") -> dict[str, object]:
    return {
        "schema_version": "upet_confidence_raw_cache_v2",
        "checkpoint": {"sha256": CHECKPOINT_SHA},
        "splits": {
            split: {
                "sha256": DATA_SHA,
                "structure_count": 1,
                "atom_count": 2,
                "force_component_count": 6,
            }
        },
        "outputs": READOUTS,
        "features": {"force_dim": 4, "energy_dim": 3, "dtype": "float32"},
        "cache": {"batch_size": 2},
        "execution": {
            "device": "cpu",
            "model_dtype": "float32",
            "system_dtype": "float32",
            "autocast": False,
            "autocast_dtype": None,
        },
        "versions": {
            "python": "3.11",
            "torch": "test",
            "metatomic": "test",
            "metatrain": "test",
            "upet": "test",
            "upet_git": "test",
        },
    }


def _cache_run(tmp_path: Path) -> tuple[Path, CacheSplitSource]:
    outputs = tmp_path / "outputs"
    manifest_path = build_raw_cache(
        outputs / "cache",
        {"train": [_raw()]},
        _identity_payload(),
    )
    cache_id = manifest_path.parent.name
    run_dir = outputs / "runs" / "energy-order1"
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_text(
        json.dumps({"status": "complete", "cache_id": cache_id}),
        encoding="utf-8",
    )
    source = CacheSplitSource(
        source="cache_split",
        split="train",
        expected_sha256=DATA_SHA,
    )
    return run_dir, source


def test_resolve_cache_split_binds_run_declared_cache_and_dataset_sha(
    tmp_path: Path,
) -> None:
    run_dir, source = _cache_run(tmp_path)

    resolved = resolve_declared_cache_split(run_dir, source)

    assert resolved.split == "train"
    assert resolved.dataset_sha256 == DATA_SHA
    assert resolved.cache_id == resolved.manifest_path.parent.name
    assert resolved.compatibility.force_dim == 4
    assert resolved.compatibility.energy_dim == 3
    assert resolved.compatibility.checkpoint_sha256 == CHECKPOINT_SHA


def test_resolve_cache_split_rejects_sha_even_when_shapes_match(
    tmp_path: Path,
) -> None:
    run_dir, source = _cache_run(tmp_path)
    wrong = source.model_copy(update={"expected_sha256": "0" * 64})

    with pytest.raises(ValueError, match="dataset SHA"):
        resolve_declared_cache_split(run_dir, wrong)


def test_resolve_cache_split_rejects_cache_path_escape(tmp_path: Path) -> None:
    run_dir, source = _cache_run(tmp_path)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest["cache_id"] = "../outside"
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cache_id|safe"):
        resolve_declared_cache_split(run_dir, source)


def test_build_external_cache_publishes_one_dataset_split_and_reuses_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "mad.xyz"
    dataset.write_text("unused", encoding="utf-8")
    config = ExternalPredictionConfig.model_validate(
        {
            "profile": "smoke",
            "checkpoint": {
                "path": tmp_path / "model.ckpt",
                "expected_sha256": CHECKPOINT_SHA,
            },
            "datasets": {
                "mad": ExtXYZSource(
                    source="extxyz",
                    path=dataset,
                    expected_sha256=DATA_SHA,
                )
            },
            "runs_root": tmp_path / "outputs/runs",
            "cache_root": tmp_path / "outputs/prediction_cache",
            "plots_root": tmp_path / "plots",
            "batch_size": 2,
            "device": "cpu",
        }
    )
    import confidence_head.external_cache as module

    payload = _identity_payload("dataset")
    extractions = 0

    def extract(*_args: object, **_kwargs: object):
        nonlocal extractions
        extractions += 1
        return iter([_raw()]), payload

    monkeypatch.setattr(
        module,
        "_expected_external_payload",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        module,
        "_extract_external_stream",
        extract,
    )

    first = build_external_dataset_cache(config, "mad")
    second = build_external_dataset_cache(config, "mad")

    assert first == second
    assert extractions == 1
    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["splits"]) == {"dataset"}
    assert first.split == "dataset"
