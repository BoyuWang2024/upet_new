"""Verified cache reuse and single-dataset cache construction."""

from __future__ import annotations

import itertools
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from .cache import (
    CachedSplitDataset,
    RawStructure,
    _validate_complete_cache,
    build_raw_cache,
    prepare_raw_cache,
)
from .checkpoint import load_upet_checkpoint
from .config import ReadoutConfig
from .data import dataset_identity
from .external_config import (
    CacheSplitSource,
    ExternalPredictionConfig,
    ExtXYZSource,
)
from .workflows.build_cache import (
    _execution_policy,
    _identity_payload,
    _raw_structures,
    _verify_outputs,
)


@dataclass(frozen=True)
class FeatureCompatibility:
    """The model/readout contract required by a trained confidence head."""

    checkpoint_sha256: str
    readouts: Mapping[str, str]
    force_dim: int
    energy_dim: int
    dtype: str


@dataclass(frozen=True)
class DatasetCache:
    """One verified split of an immutable raw cache."""

    manifest_path: Path
    cache_id: str
    split: str
    dataset_sha256: str
    compatibility: FeatureCompatibility

    def dataset(self) -> CachedSplitDataset:
        return CachedSplitDataset(self.manifest_path, self.split, self.cache_id)


def _mapping(path: Path, *, context: str) -> dict[str, Any]:
    try:
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {context} {path}: {error}") from error
    if not isinstance(loaded, dict):
        raise ValueError(f"{context} must contain a mapping: {path}")
    return loaded


def _safe_component(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value in {".", ".."}
        or Path(value).name != value
    ):
        raise ValueError(f"{field} must be a safe path component")
    return value


def _compatibility(manifest: Mapping[str, Any], split: str) -> FeatureCompatibility:
    payload = manifest.get("identity_payload")
    splits = manifest.get("splits")
    if not isinstance(payload, Mapping) or not isinstance(splits, Mapping):
        raise ValueError("cache compatibility metadata is missing")
    checkpoint = payload.get("checkpoint")
    readouts = payload.get("outputs", payload.get("readouts"))
    split_manifest = splits.get(split)
    if (
        not isinstance(checkpoint, Mapping)
        or not isinstance(readouts, Mapping)
        or not isinstance(split_manifest, Mapping)
    ):
        raise ValueError("cache compatibility metadata is malformed")
    checkpoint_sha = checkpoint.get("sha256")
    force_dim = split_manifest.get("force_feature_dim")
    energy_dim = split_manifest.get("energy_feature_dim")
    dtype = split_manifest.get("feature_dtype")
    if not isinstance(checkpoint_sha, str) or len(checkpoint_sha) != 64:
        raise ValueError("cache checkpoint SHA is invalid")
    if (
        not isinstance(force_dim, int)
        or force_dim <= 0
        or not isinstance(energy_dim, int)
        or energy_dim <= 0
        or dtype != "float32"
    ):
        raise ValueError("cache feature compatibility is invalid")
    normalized = {str(key): str(value) for key, value in readouts.items()}
    required = {
        "energy_prediction",
        "force_prediction",
        "energy_features",
        "force_features",
    }
    if set(normalized) != required:
        raise ValueError("cache readout compatibility is invalid")
    return FeatureCompatibility(
        checkpoint_sha256=checkpoint_sha,
        readouts=normalized,
        force_dim=force_dim,
        energy_dim=energy_dim,
        dtype=dtype,
    )


def _dataset_cache(manifest_path: Path, split: str, expected_sha: str) -> DatasetCache:
    manifest = _validate_complete_cache(
        manifest_path,
        expected_identity=None,
        load_shards=False,
    )
    payload = manifest["identity_payload"]
    split_payload = payload.get("splits", {}).get(split)
    if not isinstance(split_payload, Mapping):
        raise ValueError(f"cache manifest is missing split {split!r}")
    actual_sha = split_payload.get("sha256")
    if actual_sha != expected_sha:
        raise ValueError(f"dataset SHA mismatch: {actual_sha} != {expected_sha}")
    cache_id = manifest.get("cache_id")
    if not isinstance(cache_id, str):
        raise ValueError("cache manifest cache_id is invalid")
    return DatasetCache(
        manifest_path=Path(manifest_path).resolve(),
        cache_id=cache_id,
        split=split,
        dataset_sha256=expected_sha,
        compatibility=_compatibility(manifest, split),
    )


def resolve_declared_cache_split(
    run_dir: Path,
    source: CacheSplitSource,
) -> DatasetCache:
    """Resolve only the raw cache identity declared by a completed run."""

    run = Path(run_dir).resolve()
    if run.parent.name != "runs":
        raise ValueError("run directory must be directly below a runs directory")
    run_manifest = _mapping(run / "manifest.json", context="run manifest")
    if run_manifest.get("status") != "complete":
        raise ValueError("run manifest must be complete")
    cache_id = _safe_component(run_manifest.get("cache_id"), field="cache_id")
    cache_root = run.parent.parent / "cache"
    manifest_path = (cache_root / cache_id / "manifest.json").resolve()
    if not manifest_path.is_relative_to(cache_root.resolve()):
        raise ValueError("cache_id escapes cache root")
    return _dataset_cache(manifest_path, source.split, source.expected_sha256)


def _extraction_config(config: ExternalPredictionConfig) -> Any:
    return SimpleNamespace(
        readouts=ReadoutConfig(),
        cache=SimpleNamespace(batch_size=config.batch_size),
        run=SimpleNamespace(device=config.device, amp=False),
    )


def _extract_external_stream(
    config: ExternalPredictionConfig,
    dataset_name: str,
) -> tuple[Iterator[RawStructure], dict[str, Any]]:
    source = config.datasets.get(dataset_name)
    if not isinstance(source, ExtXYZSource):
        raise ValueError(f"dataset {dataset_name!r} is not an extxyz source")
    identity = dataset_identity(source.path, expected_sha256=source.expected_sha256)
    adapter = _extraction_config(config)
    execution = _execution_policy(adapter)
    device = torch.device(execution["device"])
    checkpoint = load_upet_checkpoint(
        config.checkpoint.path,
        expected_sha256=config.checkpoint.expected_sha256,
        device=device,
        dtype=torch.float32,
    )
    outputs = _verify_outputs(checkpoint.model, adapter)
    stream = _raw_structures(
        path=source.path,
        model=checkpoint.model,
        config=adapter,
        device=device,
        execution=execution,
        expected_sha256=identity.sha256,
    )
    try:
        first = next(stream)
    except StopIteration as error:
        raise ValueError("external dataset must contain at least one structure") from error
    feature_dims = (
        int(first.force_features.shape[1]),
        int(first.energy_features.shape[1]),
    )
    payload = _identity_payload(
        adapter,
        checkpoint.sha256,
        {"dataset": identity},
        outputs,
        execution,
        feature_dims,
    )
    return itertools.chain([first], stream), payload


def build_external_dataset_cache(
    config: ExternalPredictionConfig,
    dataset_name: str,
) -> DatasetCache:
    """Build or reuse one identity-addressed external extxyz cache."""

    source = config.datasets.get(dataset_name)
    if not isinstance(source, ExtXYZSource):
        raise ValueError(f"dataset {dataset_name!r} is not an extxyz source")
    stream, payload = _extract_external_stream(config, dataset_name)
    manifest_path = build_raw_cache(
        output_root=config.cache_root,
        split_structures={"dataset": stream},
        identity_payload=payload,
        staging=prepare_raw_cache(config.cache_root),
    )
    return _dataset_cache(manifest_path, "dataset", source.expected_sha256)
