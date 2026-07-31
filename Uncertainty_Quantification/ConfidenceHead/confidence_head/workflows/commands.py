"""Configuration-driven wrappers around the confidence-head workflow stages."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from ..cache import SCHEMA_VERSION
from ..config import ConfidenceConfig
from ..run_naming import build_run_name, resolve_run_dir
from .build_cache import build_cache
from .evaluate import evaluate_run
from .train import train_run
from .verify import verify_run


def _expected_cache_identity(config: ConfidenceConfig) -> dict[str, object]:
    return {
        "checkpoint": {"sha256": config.checkpoint.expected_sha256},
        "splits": {
            split: {"sha256": getattr(config.data, split).expected_sha256}
            for split in ("train", "validation", "test")
        },
        "outputs": config.readouts.model_dump(),
        "execution": {
            "device": str(torch.device(config.run.device)),
            "model_dtype": "float32",
            "system_dtype": "float32",
            "autocast": config.run.amp,
            "autocast_dtype": "float16" if config.run.amp else None,
        },
    }


def _is_matching_cache_manifest(
    path: Path, expected_identity: Mapping[str, object]
) -> bool:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, Mapping):
        return False
    payload = manifest.get("identity_payload")
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(payload, Mapping)
    ):
        return False
    return all(payload.get(field) == value for field, value in expected_identity.items())


def resolve_unique_cache_manifest(config: ConfidenceConfig) -> Path:
    """Find the single complete cache whose input identity matches config."""
    expected_identity = _expected_cache_identity(config)
    cache_root = config.run.output_root / "cache"
    matches = [
        path.resolve()
        for path in cache_root.glob("*/manifest.json")
        if _is_matching_cache_manifest(path, expected_identity)
    ]
    if not matches:
        raise ValueError("matching cache manifest was not found")
    if len(matches) != 1:
        raise ValueError("multiple matching cache manifests")
    return matches[0]


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
    resolve_unique_cache_manifest(config)
    return verify_run(resolve_run_dir(config), full=True)
