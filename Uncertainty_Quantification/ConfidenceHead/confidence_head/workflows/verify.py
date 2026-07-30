"""Verify a confidence-head run and its declared artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from ..artifacts import sha256_file
from ..identity import binning_id, config_id, model_loss_id, run_id
from ..trainer import CHECKPOINT_SCHEMA_VERSION
from .evaluate import EVALUATION_SCHEMA_VERSION
from .train import RUN_SCHEMA_VERSION


_IMAGE_SUFFIXES = {".png", ".pdf", ".svg"}


def _mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid manifest {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"manifest {path} must contain a mapping")
    return value


def _yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as error:
        raise ValueError(f"invalid YAML {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"YAML {path} must contain a mapping")
    return value


def _confined(root: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise ValueError("declared artifact path must be a string")
    path = (root.resolve() / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"declared artifact escapes run directory: {relative}")
    return path


def _verify_artifacts(root: Path, artifacts: Any, *, full: bool) -> dict[str, Path]:
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("manifest artifacts must be a non-empty mapping")
    paths: dict[str, Path] = {}
    for name, raw_entry in artifacts.items():
        if not isinstance(name, str) or not isinstance(raw_entry, Mapping):
            raise ValueError("artifact declarations must be mappings")
        path = _confined(root, raw_entry.get("path"))
        if not path.is_file():
            raise ValueError(f"declared artifact is missing: {path}")
        expected_hash = raw_entry.get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValueError(f"declared artifact has invalid sha256: {name}")
        if full and sha256_file(path) != expected_hash:
            raise ValueError(f"declared artifact sha256 mismatch: {name}")
        paths[name] = path
    return paths


def _verify_identity(root: Path, manifest: Mapping[str, Any]) -> None:
    resolved = _yaml_mapping(root / "resolved_config.yaml")
    bins = _mapping(root / "binning.json")
    expected_config_id = config_id(resolved)
    expected_binning_id = binning_id(bins)
    expected_model_loss_id = model_loss_id(
        {
            "model": resolved["model"],
            "loss": resolved["loss"],
            "force_num_bins": resolved["binning"]["force_num_bins"],
            "energy_num_bins": resolved["binning"]["energy_num_bins"],
        }
    )
    expected = {
        "config_id": expected_config_id,
        "cache_id": manifest.get("cache_id"),
        "binning_id": expected_binning_id,
        "model_loss_id": expected_model_loss_id,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"run {field} identity mismatch")
    if manifest.get("run_id") != run_id(expected):
        raise ValueError("run identity hash mismatch")


def _verify_checkpoint(path: Path, manifest: Mapping[str, Any]) -> None:
    snapshot = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(snapshot, Mapping):
        raise ValueError(f"checkpoint {path} must contain a mapping")
    if snapshot.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"checkpoint {path} schema mismatch")
    for field in ("config_id", "cache_id", "binning_id", "model_loss_id"):
        if snapshot.get(field) != manifest.get(field):
            raise ValueError(f"checkpoint {path} {field} mismatch")


def _verify_predictions(
    prediction_path: Path,
    evaluation_manifest: Mapping[str, Any],
) -> None:
    predictions = torch.load(
        prediction_path, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(predictions, Mapping):
        raise ValueError("test predictions must contain a mapping")
    required = {
        "force_logits",
        "force_labels",
        "force_observed_errors",
        "force_expected_errors",
        "energy_logits",
        "energy_labels",
        "energy_observed_errors",
        "energy_expected_errors",
        "structure_ids",
        "atom_offsets",
        "force_representatives",
        "energy_representatives",
    }
    if set(predictions) != required:
        raise ValueError("test prediction fields mismatch")
    if not all(isinstance(value, torch.Tensor) for value in predictions.values()):
        raise ValueError("prediction fields must be tensors")

    force_logits = predictions["force_logits"]
    force_labels = predictions["force_labels"]
    force_observed = predictions["force_observed_errors"]
    force_expected = predictions["force_expected_errors"]
    energy_logits = predictions["energy_logits"]
    energy_labels = predictions["energy_labels"]
    energy_observed = predictions["energy_observed_errors"]
    energy_expected = predictions["energy_expected_errors"]
    ids = predictions["structure_ids"]
    offsets = predictions["atom_offsets"]
    force_representatives = predictions["force_representatives"]
    energy_representatives = predictions["energy_representatives"]

    floating = (
        force_logits,
        force_observed,
        force_expected,
        energy_logits,
        energy_observed,
        energy_expected,
        force_representatives,
        energy_representatives,
    )
    if not all(value.is_floating_point() for value in floating):
        raise ValueError("prediction dtype mismatch: values must be floating point")
    if not all(
        value.dtype == torch.int64
        for value in (force_labels, energy_labels, ids, offsets)
    ):
        raise ValueError("prediction dtype mismatch: indices must be int64")
    if (
        force_logits.ndim != 3
        or force_labels.ndim != 2
        or force_labels.shape[1] != 3
        or force_logits.shape[:2] != force_labels.shape
        or force_observed.shape != force_labels.shape
        or force_expected.shape != force_labels.shape
    ):
        raise ValueError("force prediction shapes are inconsistent")
    if (
        energy_logits.ndim != 2
        or energy_labels.ndim != 1
        or energy_logits.shape[0] != len(energy_labels)
        or len(energy_labels) != len(ids)
        or energy_observed.shape != energy_labels.shape
        or energy_expected.shape != energy_labels.shape
    ):
        raise ValueError("energy prediction shapes are inconsistent")
    if (
        ids.ndim != 1
        or offsets.ndim != 1
        or offsets.shape != (len(ids) + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(force_labels)
        or not bool(torch.all(offsets[1:] > offsets[:-1]))
    ):
        raise ValueError("test atom offset shape or values are inconsistent")
    counts = evaluation_manifest.get("test_counts")
    if not isinstance(counts, Mapping) or (
        counts.get("structures") != len(ids)
        or counts.get("atoms") != len(force_labels)
        or counts.get("force_components") != force_labels.numel()
    ):
        raise ValueError("test prediction counts disagree with manifest")
    if (
        force_representatives.ndim != 1
        or len(force_representatives) != force_logits.shape[-1]
        or energy_representatives.ndim != 1
        or len(energy_representatives) != energy_logits.shape[-1]
    ):
        raise ValueError("prediction representative shape disagrees with logits")
    if not bool(
        torch.all(force_representatives[1:] > force_representatives[:-1])
    ) or not bool(torch.all(energy_representatives[1:] > energy_representatives[:-1])):
        raise ValueError("prediction representatives must be strictly increasing")
    if any(
        not bool(torch.all(value >= 0))
        for value in (
            force_observed,
            force_expected,
            energy_observed,
            energy_expected,
            force_representatives,
            energy_representatives,
        )
    ):
        raise ValueError("prediction errors and representatives must be nonnegative")
    if not bool(
        torch.all((force_labels >= 0) & (force_labels < force_logits.shape[-1]))
    ) or not bool(
        torch.all((energy_labels >= 0) & (energy_labels < energy_logits.shape[-1]))
    ):
        raise ValueError("prediction label range is invalid")
    if len(torch.unique(ids)) != len(ids):
        raise ValueError("prediction structure IDs must be unique")
    if any(
        value.is_floating_point() and not bool(torch.isfinite(value).all())
        for value in predictions.values()
    ):
        raise ValueError("test predictions contain non-finite values")
    calculated_force = torch.softmax(force_logits, dim=-1) @ force_representatives
    calculated_energy = torch.softmax(energy_logits, dim=-1) @ energy_representatives
    if not torch.allclose(
        calculated_force, force_expected, rtol=1e-5, atol=1e-6
    ) or not torch.allclose(calculated_energy, energy_expected, rtol=1e-5, atol=1e-6):
        raise ValueError("prediction expected error is inconsistent with logits")


def _verify_cache(manifest: Mapping[str, Any], counts: Mapping[str, Any]) -> None:
    cache = _mapping(Path(str(manifest.get("cache_manifest"))).resolve())
    if cache.get("status") != "complete":
        raise ValueError("cache manifest must be complete")
    if cache.get("cache_id") != manifest.get("cache_id"):
        raise ValueError("cache identity mismatch")
    payload = cache.get("identity_payload")
    splits = payload.get("splits") if isinstance(payload, Mapping) else None
    test = splits.get("test") if isinstance(splits, Mapping) else None
    if not isinstance(test, Mapping):
        raise ValueError("cache test identity/counts are missing")
    expected = {
        "structures": test.get("structure_count"),
        "atoms": test.get("atom_count"),
        "force_components": test.get("force_component_count"),
    }
    if any(counts.get(field) != value for field, value in expected.items()):
        raise ValueError("evaluation test counts disagree with complete cache")


def verify_run(run_dir: Path, *, full: bool = True) -> dict[str, Any]:
    """Verify completeness, identity linkage, hashes, shapes, and no figures."""
    root = Path(run_dir).resolve()
    manifest = _mapping(root / "manifest.json")
    if (
        manifest.get("schema_version") != RUN_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or manifest.get("identity") != manifest.get("run_id")
    ):
        raise ValueError("run manifest schema/status/identity mismatch")
    paths = _verify_artifacts(root, manifest.get("artifacts"), full=full)
    _verify_identity(root, manifest)
    if full:
        _verify_checkpoint(paths["checkpoints/best.pt"], manifest)
        _verify_checkpoint(paths["checkpoints/last.pt"], manifest)
    images = [
        path for path in root.rglob("*") if path.suffix.lower() in _IMAGE_SUFFIXES
    ]
    if images:
        raise ValueError(f"run contains forbidden figure artifacts: {images}")

    evaluation = manifest.get("evaluation")
    if isinstance(evaluation, Mapping) and evaluation.get("status") == "complete":
        evaluation_path = _confined(root, evaluation.get("manifest"))
        evaluation_manifest = _mapping(evaluation_path)
        if (
            evaluation_manifest.get("schema_version") != EVALUATION_SCHEMA_VERSION
            or evaluation_manifest.get("status") != "complete"
            or evaluation_manifest.get("run_id") != manifest.get("run_id")
            or evaluation_manifest.get("cache_id") != manifest.get("cache_id")
        ):
            raise ValueError("evaluation manifest identity/status mismatch")
        checkpoint = evaluation_manifest.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise ValueError("evaluation checkpoint declaration is missing")
        checkpoint_path = (
            evaluation_path.parent / str(checkpoint.get("path"))
        ).resolve()
        if not checkpoint_path.is_relative_to(root):
            raise ValueError("evaluation checkpoint escapes run directory")
        if not checkpoint_path.is_file():
            raise ValueError("evaluation checkpoint is missing")
        if full and sha256_file(checkpoint_path) != checkpoint.get("sha256"):
            raise ValueError("evaluation checkpoint sha256 mismatch")
        _verify_artifacts(
            evaluation_path.parent,
            evaluation_manifest.get("artifacts"),
            full=full,
        )
        counts = evaluation_manifest.get("test_counts")
        if not isinstance(counts, Mapping):
            raise ValueError("evaluation test counts are missing")
        _verify_cache(manifest, counts)
        _verify_predictions(
            paths["evaluation/test_predictions.pt"], evaluation_manifest
        )
    return {
        "status": "complete",
        "run_id": manifest["run_id"],
        "full": full,
        "artifact_count": len(paths),
    }
