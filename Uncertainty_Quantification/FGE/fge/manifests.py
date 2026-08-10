"""Strict, source-independent manifest builders for formal FGE artifacts."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path

from .artifacts import normalize_artifact_path, sha256_file
from .errors import HardFailure


_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT_RE = re.compile(r"[0-9a-f]{40}\Z")


def _json_safe(value: object, label: str = "manifest") -> None:
    if value is None or isinstance(value, (bool, str, int)):
        if isinstance(value, str) and Path(value).is_absolute():
            raise HardFailure(f"{label} contains an absolute path")
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HardFailure(f"{label} contains non-finite JSON")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise HardFailure(f"{label} contains a non-string JSON key")
            _json_safe(item, f"{label}.{key}")
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _json_safe(item, f"{label}[{index}]")
        return
    raise HardFailure(f"{label} contains unsupported JSON data")


def _forbid_provenance(value: object) -> None:
    if isinstance(value, Mapping):
        if {"source", "source_path", "migration", "legacy", "old_member_sha"} & set(
            value
        ):
            raise HardFailure(
                "manifest contains forbidden source or migration provenance"
            )
        for item in value.values():
            _forbid_provenance(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _forbid_provenance(item)


def _sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise HardFailure(f"{label} must be a lowercase SHA256")
    return value


def _sha_identity(value: object, label: str) -> str:
    if not isinstance(value, Mapping) or set(value) != {"sha256"}:
        raise HardFailure(f"{label} has an invalid schema")
    return _sha256(value["sha256"], f"{label}.sha256")


def _config_resolved_identity(value: object) -> dict[str, str]:
    expected_top_level = {
        "schema_version",
        "project",
        "paths",
        "identity",
        "data",
        "training",
        "fge",
        "ema",
        "prediction",
        "evaluation",
        "scientific",
    }
    if not isinstance(value, Mapping) or set(value) != expected_top_level:
        raise HardFailure("config_resolved has an invalid schema")
    paths = value["paths"]
    roles = ("base_checkpoint", "train_data", "val_data", "test_data")
    if not isinstance(paths, Mapping) or set(paths) != {*roles, "output_root"}:
        raise HardFailure("config_resolved.paths has an invalid schema")
    result: dict[str, str] = {}
    for role in roles:
        identity = paths[role]
        if not isinstance(identity, Mapping) or set(identity) != {"role", "sha256"}:
            raise HardFailure("config_resolved path identity has an invalid schema")
        if identity["role"] != role:
            raise HardFailure("config_resolved path role is invalid")
        result[role] = _sha256(
            identity["sha256"], f"config_resolved.paths.{role}.sha256"
        )
    output_root = paths["output_root"]
    if not isinstance(output_root, Mapping) or dict(output_root) != {
        "role": "output_root"
    }:
        raise HardFailure("config_resolved output_root identity is invalid")
    return result


def _data_identities(value: object) -> dict[str, str]:
    roles = ("train", "val", "test")
    if not isinstance(value, Mapping) or set(value) != set(roles):
        raise HardFailure("data_identities has an invalid schema")
    return {
        role: _sha_identity(value[role], f"data_identities.{role}") for role in roles
    }


def _scientific_flags(value: object) -> None:
    names = (
        "path_feasibility_only",
        "split_leakage",
        "scientific_evaluation",
        "inference_only",
    )
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise HardFailure("scientific_flags has an invalid schema")
    if any(not isinstance(value[name], bool) for name in names):
        raise HardFailure("scientific_flags must contain booleans")


def _dependency_snapshot(value: object) -> None:
    names = ("torch", "metatrain")
    if not isinstance(value, Mapping) or set(value) != set(names):
        raise HardFailure("dependency_snapshot has an invalid schema")
    if any(not isinstance(value[name], str) or not value[name] for name in names):
        raise HardFailure("dependency_snapshot must contain non-empty versions")


def _code_identity(
    value: object, label: str, *, allow_unavailable: bool
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise HardFailure(f"{label} must be a mapping")
    if allow_unavailable and dict(value) == {"status": "unavailable"}:
        return {"status": "unavailable"}
    if set(value) != {"commit", "dirty_sha256"}:
        raise HardFailure(f"{label} has an invalid schema")
    commit = value["commit"]
    if not isinstance(commit, str) or _COMMIT_RE.fullmatch(commit) is None:
        raise HardFailure(f"{label}.commit must be a lowercase git commit")
    return {"commit": commit, "dirty_sha256": _sha256(value["dirty_sha256"], label)}


def _members(value: object) -> list[dict[str, object]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise HardFailure("members must be a non-empty sequence")
    result: list[dict[str, object]] = []
    for index, member in enumerate(value, start=1):
        if not isinstance(member, Mapping) or set(member) != {
            "member_id",
            "sha256",
            "cycle",
            "endpoint_global_step",
        }:
            raise HardFailure("member manifest entries have an invalid schema")
        expected_id = f"member_{index:03d}"
        if member["member_id"] != expected_id or member["cycle"] != index:
            raise HardFailure("member order and cycle must be contiguous")
        step = member["endpoint_global_step"]
        if isinstance(step, bool) or not isinstance(step, int) or step < 1:
            raise HardFailure("member endpoint_global_step must be positive")
        result.append(
            {
                "member_id": expected_id,
                "sha256": _sha256(member["sha256"], "member sha256"),
                "cycle": index,
                "endpoint_global_step": step,
            }
        )
    if len({item["sha256"] for item in result}) != len(result):
        raise HardFailure("member SHA256 values must be unique")
    return result


def build_training_manifest(
    *,
    project_name: str,
    config_resolved: Mapping[str, object],
    config_identity: Mapping[str, object],
    checkpoint_identity: Mapping[str, object],
    data_identities: Mapping[str, object],
    model_contract: Mapping[str, object],
    frozen_fingerprint_identity: Mapping[str, object],
    dependency_snapshot: Mapping[str, object],
    scientific_flags: Mapping[str, object],
    artifact_writer_code_identity: Mapping[str, object],
    validator_code_identity: Mapping[str, object],
    training_code_identity: Mapping[str, object],
    member_count: int,
    members: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Build the native/migrated-invariant training provenance key tree."""
    if not isinstance(project_name, str) or not project_name:
        raise HardFailure("project_name must be a non-empty string")
    for label, value in (
        ("config_resolved", config_resolved),
        ("config_identity", config_identity),
        ("checkpoint_identity", checkpoint_identity),
        ("data_identities", data_identities),
        ("model_contract", model_contract),
        ("frozen_fingerprint_identity", frozen_fingerprint_identity),
        ("dependency_snapshot", dependency_snapshot),
        ("scientific_flags", scientific_flags),
    ):
        _json_safe(value, label)
        _forbid_provenance(value)
    path_hashes = _config_resolved_identity(config_resolved)
    _sha_identity(config_identity, "config_identity")
    checkpoint_sha256 = _sha_identity(checkpoint_identity, "checkpoint_identity")
    data_sha256 = _data_identities(data_identities)
    if path_hashes["base_checkpoint"] != checkpoint_sha256:
        raise HardFailure("base checkpoint path SHA differs from checkpoint identity")
    for split in ("train", "val", "test"):
        if path_hashes[f"{split}_data"] != data_sha256[split]:
            raise HardFailure("data path SHA differs from data identity")
    if model_contract != {
        "readout_tensor_count": 12,
        "readout_parameter_count": 13338,
    }:
        raise HardFailure("model_contract must be the formal 12-tensor/13,338 contract")
    _sha_identity(frozen_fingerprint_identity, "frozen_fingerprint_identity")
    _dependency_snapshot(dependency_snapshot)
    _scientific_flags(scientific_flags)
    if (
        isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count < 2
    ):
        raise HardFailure("member_count must be an integer of at least two")
    canonical_members = _members(members)
    if member_count != len(canonical_members):
        raise HardFailure("member_count must match canonical members")
    return {
        "schema_version": "upet.fge.training.v1",
        "project_name": project_name,
        "config_resolved": deepcopy(dict(config_resolved)),
        "config_identity": deepcopy(dict(config_identity)),
        "checkpoint_identity": deepcopy(dict(checkpoint_identity)),
        "data_identities": deepcopy(dict(data_identities)),
        "model_contract": deepcopy(dict(model_contract)),
        "frozen_fingerprint_identity": deepcopy(dict(frozen_fingerprint_identity)),
        "dependency_snapshot": deepcopy(dict(dependency_snapshot)),
        "scientific_flags": deepcopy(dict(scientific_flags)),
        "training_code_identity": _code_identity(
            training_code_identity, "training_code_identity", allow_unavailable=True
        ),
        "artifact_writer_code_identity": _code_identity(
            artifact_writer_code_identity,
            "artifact_writer_code_identity",
            allow_unavailable=True,
        ),
        "validator_code_identity": _code_identity(
            validator_code_identity, "validator_code_identity", allow_unavailable=True
        ),
        "member_count": member_count,
        "members": canonical_members,
    }


def _artifact(root: Path, path: Path, role: str) -> dict[str, object]:
    relative = normalize_artifact_path(root, path)
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise HardFailure("formal artifact must be a non-symlink regular file")
    return {
        "role": role,
        "path": relative,
        "bytes": source.stat().st_size,
        "sha256": sha256_file(source),
    }


def build_prediction_manifest(
    *,
    root: Path,
    prediction_path: Path,
    member_ids: Sequence[str],
    shape: Mapping[str, object],
    config_identity: Mapping[str, object],
    test_data_identity: Mapping[str, object],
    target_names: Mapping[str, object],
    units: Mapping[str, object],
    artifact_writer_code_identity: Mapping[str, object],
    validator_code_identity: Mapping[str, object],
) -> dict[str, object]:
    """Describe one validated canonical prediction tensor artifact."""
    if (
        not isinstance(member_ids, tuple)
        or len(member_ids) < 2
        or any(not isinstance(item, str) or not item for item in member_ids)
        or len(set(member_ids)) != len(member_ids)
    ):
        raise HardFailure("member_ids must be an ordered unique tuple")
    if set(shape) != {"K", "S", "A"} or any(
        isinstance(shape[key], bool) or not isinstance(shape[key], int)
        for key in ("K", "S", "A")
    ):
        raise HardFailure("prediction shape has an invalid schema")
    K = shape["K"]
    S = shape["S"]
    A = shape["A"]
    if not isinstance(K, int) or not isinstance(S, int) or not isinstance(A, int):
        raise HardFailure("prediction shape has an invalid schema")
    if K != len(member_ids) or S < 1 or A < 1:
        raise HardFailure("prediction shape must match its canonical members and data")
    _json_safe(config_identity, "config_identity")
    _json_safe(test_data_identity, "test_data_identity")
    _json_safe(target_names, "target_names")
    _json_safe(units, "units")
    _forbid_provenance(config_identity)
    _forbid_provenance(test_data_identity)
    _forbid_provenance(target_names)
    _forbid_provenance(units)
    config = {"sha256": _sha_identity(config_identity, "config_identity")}
    test_data = {"sha256": _sha_identity(test_data_identity, "test_data_identity")}
    expected_observables = {"energy", "forces", "stress"}
    if (
        set(target_names) != expected_observables
        or set(units) != expected_observables
        or any(not isinstance(value, str) or not value for value in target_names.values())
        or any(not isinstance(value, str) or not value for value in units.values())
    ):
        raise HardFailure("prediction target names or units have an invalid schema")
    artifact = _artifact(Path(root), Path(prediction_path), "prediction")
    return {
        "schema_version": "upet.fge.prediction.v1",
        "config_identity": config,
        "test_data_identity": test_data,
        "member_ids": list(member_ids),
        "shape": dict(shape),
        "target_names": dict(target_names),
        "units": dict(units),
        "artifact": artifact,
        "artifact_writer_code_identity": _code_identity(
            artifact_writer_code_identity,
            "artifact_writer_code_identity",
            allow_unavailable=True,
        ),
        "validator_code_identity": _code_identity(
            validator_code_identity, "validator_code_identity", allow_unavailable=True
        ),
    }


def _default_formal_artifacts(root: Path) -> dict[str, Path]:
    roles = {
        "config_resolved.yaml": "config_resolved",
        "preflight/train.json": "preflight_train",
        "preflight/predict.json": "preflight_predict",
        "preflight/evaluate.json": "preflight_evaluate",
        "training/manifest.json": "training_manifest",
        "prediction/manifest.json": "prediction_manifest",
        "prediction/test_raw.pt": "prediction",
        "evaluation/legacy_equal_weight/ensemble.pt": "ensemble",
        "evaluation/legacy_equal_weight/uncertainty.pt": "uncertainty",
        "evaluation/legacy_equal_weight/metrics.json": "metrics",
        "evaluation/legacy_equal_weight/report.md": "report",
        "validation.json": "validation",
    }
    artifacts = {
        role: root / relative
        for relative, role in roles.items()
        if (root / relative).is_file()
    }
    members_directory = root / "training" / "members"
    if members_directory.is_dir():
        for member in sorted(members_directory.glob("member_*.pt")):
            if re.fullmatch(r"member_\d{3}\.pt", member.name):
                artifacts[f"training_{member.stem}"] = member
    return artifacts


def build_result_manifest(
    *,
    root: Path,
    project_name: str,
    artifact_writer_code_identity: Mapping[str, object],
    validator_code_identity: Mapping[str, object],
    formal_artifacts: Mapping[str, Path] | None = None,
) -> dict[str, object]:
    """Inventory all already-written formal artifacts before final publication."""
    result_root = Path(root).resolve()
    if not isinstance(project_name, str) or not project_name:
        raise HardFailure("project_name must be a non-empty string")
    artifacts = (
        _default_formal_artifacts(result_root)
        if formal_artifacts is None
        else formal_artifacts
    )
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise HardFailure("formal_artifacts must be a non-empty role mapping")
    inventory = [
        _artifact(result_root, Path(path), role) for role, path in artifacts.items()
    ]
    if any(not isinstance(item["role"], str) or not item["role"] for item in inventory):
        raise HardFailure("formal artifact roles must be non-empty strings")
    if len({item["role"] for item in inventory}) != len(inventory) or len(
        {item["path"] for item in inventory}
    ) != len(inventory):
        raise HardFailure("formal artifact roles and paths must be unique")
    inventory.sort(key=lambda item: str(item["path"]))
    return {
        "schema_version": "upet.fge.result.v1",
        "project_name": project_name,
        "status": "PASS",
        "artifact_writer_code_identity": _code_identity(
            artifact_writer_code_identity,
            "artifact_writer_code_identity",
            allow_unavailable=False,
        ),
        "validator_code_identity": _code_identity(
            validator_code_identity, "validator_code_identity", allow_unavailable=False
        ),
        "artifacts": inventory,
    }
