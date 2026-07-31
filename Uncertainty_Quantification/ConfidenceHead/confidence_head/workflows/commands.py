"""Configuration-driven wrappers around the confidence-head workflow stages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..cache import SCHEMA_VERSION
from ..config import ConfidenceConfig
from ..identity import cache_id
from ..run_naming import build_run_name, resolve_run_dir
from .build_cache import build_cache
from .evaluate import evaluate_run
from .train import train_run
from .verify import verify_run


def _expected_cache_identity(config: ConfidenceConfig) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {"sha256": config.checkpoint.expected_sha256},
        "splits": {
            split: {"sha256": getattr(config.data, split).expected_sha256}
            for split in ("train", "validation", "test")
        },
        "outputs": config.readouts.model_dump(),
        "cache": {
            "batch_size": config.cache.batch_size,
            "shard_max_atoms": config.cache.shard_max_atoms,
        },
        "execution": {
            "device": str(torch.device(config.run.device)),
            "model_dtype": "float32",
            "system_dtype": "float32",
            "autocast": config.run.amp,
            "autocast_dtype": "float16" if config.run.amp else None,
        },
    }


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_complete_identity_payload(
    payload: Mapping[str, object],
    expected: Mapping[str, object],
) -> bool:
    if set(payload) != {
        "schema_version",
        "checkpoint",
        "splits",
        "outputs",
        "features",
        "cache",
        "execution",
        "versions",
    }:
        return False
    if payload.get("schema_version") != expected["schema_version"]:
        return False

    checkpoint = payload.get("checkpoint")
    if checkpoint != expected["checkpoint"]:
        return False

    splits = payload.get("splits")
    expected_splits = expected["splits"]
    if not isinstance(splits, Mapping) or not isinstance(expected_splits, Mapping):
        return False
    if set(splits) != {"train", "validation", "test"}:
        return False
    for name, expected_split in expected_splits.items():
        if not isinstance(expected_split, Mapping):
            return False
        split = splits.get(name)
        if (
            not isinstance(split, Mapping)
            or set(split)
            != {
                "sha256",
                "structure_count",
                "atom_count",
                "force_component_count",
            }
            or split.get("sha256") != expected_split.get("sha256")
            or not all(
                _positive_int(split.get(field))
                for field in (
                    "structure_count",
                    "atom_count",
                    "force_component_count",
                )
            )
        ):
            return False

    if payload.get("outputs") != expected["outputs"]:
        return False
    features = payload.get("features")
    if (
        not isinstance(features, Mapping)
        or set(features) != {"force_dim", "energy_dim", "dtype"}
        or not _positive_int(features.get("force_dim"))
        or not _positive_int(features.get("energy_dim"))
        or features.get("dtype") != "float32"
    ):
        return False
    if payload.get("cache") != expected["cache"]:
        return False
    if payload.get("execution") != expected["execution"]:
        return False

    versions = payload.get("versions")
    return isinstance(versions, Mapping) and set(versions) == {
        "python",
        "torch",
        "metatomic",
        "metatrain",
        "upet",
        "upet_git",
    } and all(isinstance(value, str) and value for value in versions.values())


def _matching_cache_id(
    path: Path,
    expected_identity: Mapping[str, object],
    cache_root: Path,
) -> str | None:
    resolved = path.resolve()
    if not resolved.is_relative_to(cache_root):
        return None
    try:
        manifest = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, Mapping):
        return None
    payload = manifest.get("identity_payload")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(payload, Mapping)
        or not _is_complete_identity_payload(payload, expected_identity)
    ):
        return None
    derived = cache_id(
        {
            "schema_version": SCHEMA_VERSION,
            "identity_payload": dict(payload),
        }
    )
    if manifest.get("identity") != derived or manifest.get("cache_id") != derived:
        return None
    return derived


def resolve_unique_cache_manifest(config: ConfidenceConfig) -> Path:
    """Find the single complete cache whose input identity matches config."""
    expected_identity = _expected_cache_identity(config)
    cache_root = (config.run.output_root / "cache").resolve()
    matches = [
        path.resolve()
        for path in cache_root.glob("*/manifest.json")
        if _matching_cache_id(path, expected_identity, cache_root) is not None
    ]
    if not matches:
        raise ValueError("matching cache manifest was not found")
    if len(matches) != 1:
        raise ValueError("multiple matching cache manifests")
    return matches[0]


def _selected_cache_id(config: ConfidenceConfig, cache_manifest_path: Path) -> str:
    cache_root = (config.run.output_root / "cache").resolve()
    cache_identity = _matching_cache_id(
        cache_manifest_path,
        _expected_cache_identity(config),
        cache_root,
    )
    if cache_identity is None:
        raise ValueError("matching cache manifest was not found")
    return cache_identity


def _validate_resume_path(run_dir: Path, resume_from: Path | None) -> Path | None:
    if resume_from is None:
        return None
    resolved = Path(resume_from).resolve()
    if not resolved.is_relative_to(run_dir):
        raise ValueError("resume checkpoint must be within derived run directory")
    return resolved


def build_cache_from_config(config: ConfidenceConfig) -> Path:
    """Build and return the cache selected by config."""
    return build_cache(config)


def train_from_config(config: ConfidenceConfig) -> Path:
    """Train the named run using the unique cache selected by config."""
    cache_manifest_path = resolve_unique_cache_manifest(config)
    run_dir = resolve_run_dir(config)
    resume_from = _validate_resume_path(run_dir, config.trainer.resume_from)
    return train_run(
        config,
        cache_manifest_path=cache_manifest_path,
        run_name=build_run_name(config),
        resume_from=resume_from,
    )


def evaluate_from_config(config: ConfidenceConfig) -> Path:
    """Evaluate the best checkpoint of the derived run."""
    cache_manifest_path = resolve_unique_cache_manifest(config)
    run_dir = resolve_run_dir(config)
    return evaluate_run(
        run_dir,
        cache_manifest_path=cache_manifest_path,
        checkpoint_path=None,
    )


def verify_from_config(config: ConfidenceConfig) -> dict[str, Any]:
    """Fully verify the derived run after confirming its cache identity."""
    cache_manifest_path = resolve_unique_cache_manifest(config)
    cache_identity = _selected_cache_id(config, cache_manifest_path)
    run_dir = resolve_run_dir(config)
    try:
        run_manifest = json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid run manifest: {error}") from error
    if not isinstance(run_manifest, Mapping):
        raise ValueError("run manifest must contain a mapping")
    if run_manifest.get("cache_id") != cache_identity:
        raise ValueError("run cache_id does not match selected cache")
    return verify_run(run_dir, full=True)
