"""Compute-free preflight checks for the formal FGE stages."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import yaml

from .artifacts import (
    ExperimentLayout,
    atomic_write_json,
    atomic_write_yaml,
    sha256_file,
)
from .config import FGEConfig
from .errors import HardFailure
from .prediction import validate_prediction_payload


_STAGES = frozenset({"train", "predict", "evaluate"})
_BASES = frozenset({"runtime_inputs", "canonical_artifacts"})
_TARGETS = {
    "energy": "energy",
    "forces": "non_conservative_forces",
    "stress": "non_conservative_stress",
}
_UNITS = {
    "energy": "eV",
    "forces": "eV/angstrom",
    "stress": "eV/angstrom^3",
}
_FORBIDDEN_PROVENANCE = frozenset(
    {"source", "source_path", "migration", "legacy", "old_member_sha"}
)


def _fail_unless(condition: bool, message: str) -> None:
    if not condition:
        raise HardFailure(message)


def _paths(config: FGEConfig) -> tuple[Path, Path, Path, Path]:
    paths = config.paths
    try:
        return (
            Path(paths.base_checkpoint),
            Path(paths.train_data),
            Path(paths.val_data),
            Path(paths.test_data),
        )
    except AttributeError as exc:
        raise HardFailure("preflight configuration paths are incomplete") from exc


def _identity(config: FGEConfig) -> dict[str, Any]:
    identity = config.identity
    data = config.data
    training = config.training
    fge = config.fge
    base, train, val, test = _paths(config)
    expected_hashes = {
        "base_checkpoint": identity.base_checkpoint_sha256,
        "train_data": identity.train_data_sha256,
        "val_data": identity.val_data_sha256,
        "test_data": identity.test_data_sha256,
    }
    for role, path in zip(expected_hashes, (base, train, val, test), strict=True):
        _fail_unless(
            path.is_file() and not path.is_symlink(), f"{role} is not a regular file"
        )
        _fail_unless(
            sha256_file(path) == expected_hashes[role],
            f"{role} SHA256 does not match the configuration",
        )
    _fail_unless(
        {
            "energy": data.energy_target,
            "forces": data.forces_target,
            "stress": data.stress_target,
        }
        == _TARGETS,
        "preflight target contract is invalid",
    )
    _fail_unless(
        {
            "energy": data.energy_unit,
            "forces": data.forces_unit,
            "stress": data.stress_unit,
        }
        == _UNITS,
        "preflight unit contract is invalid",
    )
    model_contract = {
        "readout_tensor_count": training.expected_readout_tensor_count,
        "readout_parameter_count": training.expected_readout_parameter_count,
    }
    _fail_unless(
        model_contract
        == {"readout_tensor_count": 12, "readout_parameter_count": 13338},
        "preflight readout contract is invalid",
    )
    _fail_unless(
        isinstance(fge.member_count, int)
        and not isinstance(fge.member_count, bool)
        and fge.member_count >= 2,
        "preflight member count is invalid",
    )
    _fail_unless(training.dtype == "float32", "preflight dtype must be float32")
    _fail_unless(training.device in {"cpu", "cuda"}, "preflight device is invalid")
    project = config.project
    _fail_unless(project is not None, "preflight configuration has no project")
    if project.name == "upet_fge_n20_cpu":
        _fail_unless(
            (fge.member_count, training.device) == (2, "cpu"),
            "n20 preflight runtime scale is invalid",
        )
    elif project.name == "upet_fge_full":
        _fail_unless(
            (fge.member_count, training.device) == (8, "cuda"),
            "full preflight runtime scale is invalid",
        )
    else:
        raise HardFailure("preflight project is invalid")
    _restart_state(base)
    return {
        "inputs": expected_hashes,
        "targets": dict(_TARGETS),
        "units": dict(_UNITS),
        "model_contract": model_contract,
        "member_count": fge.member_count,
        "runtime": {"device": training.device, "dtype": training.dtype},
    }


def _restart_state(path: Path) -> None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise HardFailure(f"unable to parse restart checkpoint: {path}") from exc
    _fail_unless(
        isinstance(checkpoint, Mapping)
        and isinstance(checkpoint.get("model_state_dict"), Mapping),
        "checkpoint has no restart model_state_dict",
    )


def _scientific_flags(config: FGEConfig, stage: str) -> dict[str, bool]:
    scientific = config.scientific
    selected = scientific.training if stage == "train" else scientific.evaluation
    names = (
        "path_feasibility_only",
        "split_leakage",
        "scientific_evaluation",
        "inference_only",
    )
    flags = {name: getattr(selected, name) for name in names}
    _fail_unless(
        all(type(value) is bool for value in flags.values()),
        "scientific flags must be booleans",
    )
    project_name = config.project.name
    expected = (
        {
            "path_feasibility_only": True,
            "split_leakage": True,
            "scientific_evaluation": False,
            "inference_only": False,
        }
        if stage == "train"
        else (
            {
                "path_feasibility_only": True,
                "split_leakage": True,
                "scientific_evaluation": False,
                "inference_only": True,
            }
            if project_name == "upet_fge_n20_cpu"
            else {
                "path_feasibility_only": False,
                "split_leakage": False,
                "scientific_evaluation": True,
                "inference_only": True,
            }
        )
    )
    _fail_unless(flags == expected, "scientific flags do not match the formal stage")
    return flags


def _canonical_config(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(dict(value), sort_keys=True))


def _canonical_config_identity(value: Mapping[str, object]) -> dict[str, str]:
    encoded = json.dumps(
        _canonical_config(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest()}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise HardFailure(f"{label} is invalid")
    return value


def _validate_training_prior(
    config: FGEConfig, layout: ExperimentLayout, training: Mapping[str, object]
) -> list[str]:
    _fail_unless(
        training.get("schema_version") == "upet.fge.training.v1",
        "training schema is invalid",
    )
    _fail_unless(
        training.get("project_name") == config.project.name,
        "training project identity is invalid",
    )
    sanitized = config.sanitized()
    _fail_unless(isinstance(sanitized, Mapping), "sanitized configuration is invalid")
    _fail_unless(
        training.get("config_resolved") == _canonical_config(sanitized)
        and training.get("config_identity") == _canonical_config_identity(sanitized),
        "training config identity is invalid",
    )
    expected_flags = {
        name: getattr(config.scientific.training, name)
        for name in (
            "path_feasibility_only",
            "split_leakage",
            "scientific_evaluation",
            "inference_only",
        )
    }
    flags = training.get("scientific_flags")
    _fail_unless(
        isinstance(flags, Mapping)
        and set(flags) == set(expected_flags)
        and all(type(value) is bool for value in flags.values())
        and dict(flags) == expected_flags,
        "training scientific flags are invalid",
    )
    model_contract = training.get("model_contract")
    _fail_unless(
        model_contract
        == {"readout_tensor_count": 12, "readout_parameter_count": 13338},
        "training model contract is invalid",
    )
    expected_hashes = {
        "base_checkpoint": config.identity.base_checkpoint_sha256,
        "train_data": config.identity.train_data_sha256,
        "val_data": config.identity.val_data_sha256,
        "test_data": config.identity.test_data_sha256,
    }
    checkpoint = _mapping(training.get("checkpoint_identity"), "checkpoint identity")
    _fail_unless(
        checkpoint.get("sha256") == expected_hashes["base_checkpoint"],
        "training checkpoint identity is invalid",
    )
    identities = _mapping(training.get("data_identities"), "data identities")
    for split in ("train", "val", "test"):
        identity = _mapping(identities.get(split), "data identity")
        _fail_unless(
            identity.get("sha256") == expected_hashes[f"{split}_data"],
            "training data identity is invalid",
        )
    members = training.get("members")
    member_count = training.get("member_count")
    if (
        not isinstance(members, list)
        or isinstance(member_count, bool)
        or not isinstance(member_count, int)
        or member_count != config.fge.member_count
        or len(members) != member_count
    ):
        raise HardFailure("training member identity is invalid")
    member_ids: list[str] = []
    for index, member in enumerate(members, start=1):
        entry = _mapping(member, "training member")
        member_id = f"member_{index:03d}"
        _fail_unless(
            entry.get("member_id") == member_id and entry.get("cycle") == index,
            "training member order is invalid",
        )
        path = layout.root / "training" / "members" / f"{member_id}.pt"
        _fail_unless(
            path.is_file()
            and not path.is_symlink()
            and entry.get("sha256") == sha256_file(path),
            "training member hash is invalid",
        )
        member_ids.append(member_id)
    return member_ids


def _canonical_documents(
    config: FGEConfig, layout: ExperimentLayout, stage: str
) -> None:
    config_path = layout.root / "config_resolved.yaml"
    _fail_unless(
        config_path.is_file() and not config_path.is_symlink(),
        f"canonical artifact is missing: {config_path}",
    )
    try:
        resolved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HardFailure("canonical resolved config is invalid") from exc
    _assert_source_independent(resolved)
    _fail_unless(
        isinstance(resolved, Mapping)
        and resolved == _canonical_config(config.sanitized()),
        "canonical resolved config identity is invalid",
    )
    path = layout.training_manifest
    _fail_unless(
        path.is_file() and not path.is_symlink(),
        f"canonical artifact is missing: {path}",
    )
    try:
        training = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardFailure(f"canonical artifact is not valid JSON: {path}") from exc
    _assert_source_independent(training)
    member_ids = _validate_training_prior(
        config, layout, _mapping(training, "training manifest")
    )
    if stage != "evaluate":
        return
    path = layout.prediction_manifest
    _fail_unless(
        path.is_file() and not path.is_symlink(),
        f"canonical artifact is missing: {path}",
    )
    try:
        prediction = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardFailure(f"canonical artifact is not valid JSON: {path}") from exc
    _assert_source_independent(prediction)
    prediction_mapping = _mapping(prediction, "prediction manifest")
    _fail_unless(
        prediction_mapping.get("schema_version") == "upet.fge.prediction.v1"
        and prediction_mapping.get("member_ids") == member_ids,
        "prediction prior identity is invalid",
    )
    artifact = _mapping(prediction_mapping.get("artifact"), "prediction artifact")
    artifact_path = layout.root / "prediction" / "test_raw.pt"
    _fail_unless(
        artifact.get("path") == "prediction/test_raw.pt"
        and artifact_path.is_file()
        and not artifact_path.is_symlink()
        and artifact.get("sha256") == sha256_file(artifact_path)
        and artifact.get("bytes") == artifact_path.stat().st_size,
        "prediction artifact identity is invalid",
    )
    try:
        payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure("prediction artifact cannot be loaded") from exc
    if not isinstance(payload, Mapping):
        raise HardFailure("prediction artifact has an invalid schema")
    try:
        shape = validate_prediction_payload(payload)
    except (TypeError, ValueError) as exc:
        raise HardFailure("prediction payload is invalid") from exc
    expected_targets = {
        "energy": config.data.energy_target,
        "forces": config.data.forces_target,
        "stress": config.data.stress_target,
    }
    expected_units = {
        "energy": config.data.energy_unit,
        "forces": config.data.forces_unit,
        "stress": config.data.stress_unit,
    }
    _fail_unless(
        prediction_mapping.get("shape") == {"K": shape.K, "S": shape.S, "A": shape.A}
        and payload.get("member_ids") == tuple(member_ids)
        and payload.get("statistics") == {"K": shape.K, "S": shape.S, "A": shape.A}
        and payload.get("target_names") == expected_targets
        and payload.get("units") == expected_units,
        "prediction payload metadata is invalid",
    )


def _assert_source_independent(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise HardFailure("canonical artifact contains a non-string key")
            if key.lower() in _FORBIDDEN_PROVENANCE:
                raise HardFailure(
                    "canonical artifact contains source or migration provenance"
                )
            _assert_source_independent(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_source_independent(nested)
    elif isinstance(value, str) and Path(value).is_absolute():
        raise HardFailure("canonical artifact contains an absolute path")


def _layout(config: FGEConfig) -> ExperimentLayout:
    paths = config.paths
    project = config.project
    output_root = Path(paths.output_root)
    _fail_unless(
        not output_root.exists() or output_root.is_dir(),
        "output_root is not a directory",
    )
    output_root.mkdir(parents=True, exist_ok=True)
    _fail_unless(
        shutil.disk_usage(output_root).free > 0, "output filesystem has no free space"
    )
    return ExperimentLayout(output_root / project.name)


def _write_initial_config(config: FGEConfig, layout: ExperimentLayout) -> None:
    if layout.root.exists() and layout.result_manifest.exists():
        raise HardFailure("completed formal result is immutable")
    if layout.root.exists() and layout.root.is_symlink():
        raise HardFailure("formal output root must not be a symlink")
    if layout.root.exists() and layout.root.is_file():
        raise HardFailure("formal output root must be a directory")
    if layout.root.exists() and (layout.root / "config_resolved.yaml").exists():
        return
    payload = config.sanitized()
    _fail_unless(
        isinstance(payload, Mapping), "sanitized configuration must be a mapping"
    )
    atomic_write_yaml(layout.root / "config_resolved.yaml", dict(payload))


def run_preflight(
    config: FGEConfig, stage: str, basis: str = "runtime_inputs"
) -> dict[str, Any]:
    """Validate one formal stage without model forward, backward, or optimization."""
    if stage not in _STAGES:
        raise HardFailure(f"unsupported FGE stage: {stage}")
    if basis not in _BASES:
        raise HardFailure(f"unsupported preflight basis: {basis}")
    layout = _layout(config)
    if layout.result_manifest.exists():
        raise HardFailure("completed formal result is immutable")
    identity = _identity(config)
    scientific_flags = _scientific_flags(config, stage)
    if basis == "canonical_artifacts":
        _canonical_documents(config, layout, stage)
    if stage == "train" and basis == "runtime_inputs":
        _write_initial_config(config, layout)
    report: dict[str, Any] = {
        "stage": stage,
        "status": "PASS",
        "basis": basis,
        "identity": identity,
        "scientific_flags": scientific_flags,
        "config_identity": _canonical_config_identity(config.sanitized()),
    }
    atomic_write_json(layout.preflight_dir / f"{stage}.json", report)
    return report
