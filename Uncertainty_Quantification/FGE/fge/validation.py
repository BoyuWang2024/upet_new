"""Independent, disk-backed validation for immutable FGE results."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .artifacts import atomic_write_json, sha256_file
from .config import FGEConfig
from .errors import HardFailure
from .evaluation import evaluate_prediction
from .manifests import _code_identity, build_result_manifest
from .members import ReadoutAudit, load_member
from .prediction import validate_prediction_payload


_ATOL = 2e-7
_RTOL = 2e-6
_FORBIDDEN_KEYS = frozenset(
    {"source", "source_path", "migration", "legacy", "old_member_sha"}
)
_PREDICTION_FIELD_SHAPES: dict[str, list[int | str]] = {
    "energy_prediction": ["K", "S"],
    "forces_prediction": ["K", "A", 3],
    "stress_prediction": ["K", "S", 3, 3],
    "energy_reference": ["S"],
    "forces_reference": ["A", 3],
    "stress_reference": ["S", 3, 3],
    "n_atoms": ["S"],
    "structure_offsets": ["S+1"],
    "atomic_numbers": ["A"],
    "structure_mapping": ["A"],
}


@dataclass(frozen=True)
class ValidationReport:
    """The durable outcome of a canonical FGE result validation."""

    status: str
    mode: str
    artifact_count: int

    def as_json(self) -> dict[str, object]:
        return {
            "schema_version": "upet.fge.validation.v1",
            "status": self.status,
            "mode": self.mode,
            "artifact_count": self.artifact_count,
        }


def _fail_unless(condition: bool, message: str) -> None:
    if not condition:
        raise HardFailure(message)


def _string_key_mapping(value: object, message: str) -> dict[str, Any]:
    """Return a concrete string-keyed mapping after strict schema validation."""
    if not isinstance(value, Mapping):
        raise HardFailure(message)
    result: dict[str, Any] = {}
    for key, nested in value.items():
        if not isinstance(key, str):
            raise HardFailure(message)
        result[key] = nested
    return result


def _load_json(path: Path) -> dict[str, Any]:
    _regular(path, "JSON artifact")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HardFailure(f"unable to load JSON artifact: {path}") from exc
    _fail_unless(isinstance(value, dict), f"JSON artifact must be a mapping: {path}")
    _assert_clean(value)
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    _regular(path, "YAML artifact")
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise HardFailure(f"unable to load YAML artifact: {path}") from exc
    _fail_unless(isinstance(value, dict), f"YAML artifact must be a mapping: {path}")
    _assert_clean(value)
    return value


def _load_torch(path: Path) -> Mapping[str, Any]:
    _regular(path, "tensor artifact")
    try:
        value = torch.load(path, map_location="cpu", weights_only=True)
    except (OSError, RuntimeError, ValueError, TypeError) as exc:
        raise HardFailure(f"unable to load tensor artifact: {path}") from exc
    _fail_unless(
        isinstance(value, Mapping), f"tensor artifact must be a mapping: {path}"
    )
    _assert_clean(value)
    _assert_finite(value)
    return value


def _regular(path: Path, label: str) -> None:
    _fail_unless(
        path.is_file() and not path.is_symlink(),
        f"{label} is missing or not regular: {path}",
    )


def _assert_clean(value: object) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            _fail_unless(
                isinstance(key, str), "formal artifact contains non-string key"
            )
            _fail_unless(
                key.lower() not in _FORBIDDEN_KEYS,
                "formal artifact contains forbidden provenance",
            )
            _assert_clean(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _assert_clean(nested)
    elif isinstance(value, str):
        _fail_unless(
            not Path(value).is_absolute(), "formal artifact contains an absolute path"
        )
    elif isinstance(value, float):
        _fail_unless(math.isfinite(value), "formal artifact contains non-finite JSON")


def _assert_finite(value: object) -> None:
    if isinstance(value, torch.Tensor):
        _fail_unless(
            bool(torch.isfinite(value).all().item()),
            "tensor artifact contains non-finite values",
        )
    elif isinstance(value, Mapping):
        for nested in value.values():
            _assert_finite(nested)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for nested in value:
            _assert_finite(nested)


def _canonical_config(value: Mapping[str, object]) -> dict[str, object]:
    return json.loads(json.dumps(dict(value), sort_keys=True))


def _config_identity(value: Mapping[str, object]) -> dict[str, str]:
    encoded = json.dumps(
        _canonical_config(value), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest()}


def _identity(
    config: FGEConfig, training: Mapping[str, Any], prediction: Mapping[str, Any]
) -> None:
    _fail_unless(
        training.get("schema_version") == "upet.fge.training.v1",
        "training schema is invalid",
    )
    _fail_unless(
        prediction.get("schema_version") == "upet.fge.prediction.v1",
        "prediction schema is invalid",
    )
    project = config.project.name
    _fail_unless(
        project == training.get("project_name"),
        "project identity does not match training manifest",
    )
    sanitized = config.sanitized()
    _fail_unless(isinstance(sanitized, Mapping), "config identity is invalid")
    expected_config_identity = _config_identity(sanitized)
    _fail_unless(
        training.get("config_identity") == expected_config_identity,
        "training config identity does not match configuration",
    )
    _fail_unless(
        prediction.get("config_identity") == expected_config_identity,
        "prediction config identity does not match configuration",
    )
    _fail_unless(
        prediction.get("test_data_identity")
        == {"sha256": config.identity.test_data_sha256},
        "prediction test data identity does not match configuration",
    )
    try:
        expected_training_flags = {
            name: getattr(config.scientific.training, name)
            for name in (
                "path_feasibility_only",
                "split_leakage",
                "scientific_evaluation",
                "inference_only",
            )
        }
    except AttributeError as exc:
        raise HardFailure("validation scientific flags are invalid") from exc
    scientific_flags = training.get("scientific_flags")
    _fail_unless(
        isinstance(scientific_flags, Mapping)
        and set(scientific_flags) == set(expected_training_flags)
        and all(type(value) is bool for value in scientific_flags.values())
        and dict(scientific_flags) == expected_training_flags,
        "training scientific flags do not match configuration",
    )
    members = training.get("members")
    if not isinstance(members, list) or len(members) < 2:
        raise HardFailure("training members are invalid")
    member_ids = [
        member.get("member_id") if isinstance(member, Mapping) else None
        for member in members
    ]
    expected_ids = [f"member_{index:03d}" for index in range(1, len(members) + 1)]
    _fail_unless(member_ids == expected_ids, "training member order is invalid")
    _fail_unless(
        training.get("member_count") == len(members), "training member count is invalid"
    )
    _fail_unless(
        config.fge.member_count == len(members),
        "config member count does not match training manifest",
    )
    _fail_unless(
        prediction.get("member_ids") == expected_ids,
        "prediction member order is invalid",
    )
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
        prediction.get("target_names") == expected_targets,
        "prediction targets do not match configuration",
    )
    _fail_unless(
        prediction.get("units") == expected_units,
        "prediction units do not match configuration",
    )
    _fail_unless(
        training.get("model_contract")
        == {"readout_tensor_count": 12, "readout_parameter_count": 13338},
        "training readout contract is invalid",
    )
    config_resolved = _string_key_mapping(
        training.get("config_resolved"), "training config identity is invalid"
    )
    paths = _string_key_mapping(
        config_resolved.get("paths"), "training config path identity is invalid"
    )
    checkpoint_identity = _string_key_mapping(
        training.get("checkpoint_identity"), "checkpoint identity binding is invalid"
    )
    checkpoint_path_identity = _string_key_mapping(
        paths.get("base_checkpoint"), "checkpoint identity binding is invalid"
    )
    _fail_unless(
        checkpoint_identity.get("sha256") == checkpoint_path_identity.get("sha256"),
        "checkpoint identity binding is invalid",
    )
    data = _string_key_mapping(
        training.get("data_identities"), "training data identities are invalid"
    )
    for split in ("train", "val", "test"):
        data_identity = _string_key_mapping(
            data.get(split), "data identity binding is invalid"
        )
        path_identity = _string_key_mapping(
            paths.get(f"{split}_data"), "data identity binding is invalid"
        )
        _fail_unless(
            data_identity.get("sha256") == path_identity.get("sha256"),
            "data identity binding is invalid",
        )


def _member_index(member_id: object) -> int:
    if not isinstance(member_id, str) or not member_id.startswith("member_"):
        raise HardFailure("training member ID is invalid")
    suffix = member_id.removeprefix("member_")
    if len(suffix) != 3 or not suffix.isdecimal():
        raise HardFailure("training member ID is invalid")
    return int(suffix)


def _member_audit(path: Path) -> ReadoutAudit:
    raw = _load_torch(path)
    entries = raw.get("tensors")
    if not isinstance(entries, Sequence) or len(entries) != 12:
        raise HardFailure("member tensor count is invalid")
    names: list[str] = []
    shapes: list[tuple[int, ...]] = []
    dtypes: list[str] = []
    scalar_count = 0
    for entry in entries:
        mapping = _string_key_mapping(entry, "member tensor is invalid")
        name, dtype, shape, value = (
            mapping.get("name"),
            mapping.get("dtype"),
            mapping.get("shape"),
            mapping.get("value"),
        )
        if (
            not isinstance(name, str)
            or not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(value, torch.Tensor)
        ):
            raise HardFailure("member tensor is invalid")
        names.append(name)
        shapes.append(tuple(shape))
        dtypes.append(dtype)
        scalar_count += value.numel()
    return ReadoutAudit(
        names=tuple(names),
        tensor_count=len(names),
        scalar_count=scalar_count,
        shapes=tuple(shapes),
        dtypes=tuple(dtypes),
        all_finite=True,
    )


def _validate_members(root: Path, training: Mapping[str, Any]) -> None:
    members = training.get("members")
    checkpoint = _string_key_mapping(
        training.get("checkpoint_identity"), "checkpoint identity binding is invalid"
    )
    base_sha256 = checkpoint.get("sha256")
    if not isinstance(members, list) or not isinstance(base_sha256, str):
        raise HardFailure("training members are invalid")
    audit: ReadoutAudit | None = None
    for expected_index, entry in enumerate(members, start=1):
        member = _string_key_mapping(entry, "training member is invalid")
        index = _member_index(member.get("member_id"))
        _fail_unless(index == expected_index, "training member order is invalid")
        path = root / "training" / "members" / f"member_{index:03d}.pt"
        _regular(path, "member artifact")
        _fail_unless(
            member.get("sha256") == sha256_file(path), "member SHA256 does not match"
        )
        if audit is None:
            audit = _member_audit(path)
        payload = load_member(path, base_sha256, audit)
        _fail_unless(payload.member_id == index, "member payload ID is invalid")
        _fail_unless(payload.cycle == expected_index, "member payload cycle is invalid")
        _fail_unless(
            payload.global_step == member.get("endpoint_global_step"),
            "member payload endpoint step is invalid",
        )
    _fail_unless(
        audit is not None
        and audit.tensor_count == 12
        and audit.scalar_count == 13338
        and audit.all_finite,
        "member readout contract is invalid",
    )


def _exact_or_close(actual: object, expected: object, name: str) -> None:
    if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
        raise HardFailure(f"{name} is not a tensor")
    _fail_unless(
        actual.dtype == expected.dtype and actual.shape == expected.shape,
        f"{name} dtype or shape differs",
    )
    _fail_unless(
        torch.allclose(actual, expected, rtol=_RTOL, atol=_ATOL, equal_nan=False),
        f"{name} numerical value differs",
    )


def _validate_prediction(
    root: Path, prediction_manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    artifact = prediction_manifest.get("artifact")
    if not isinstance(artifact, Mapping):
        raise HardFailure("prediction artifact identity is invalid")
    _fail_unless(
        artifact.get("path") == "prediction/test_raw.pt",
        "prediction artifact path is invalid",
    )
    artifact_path = artifact.get("path")
    if not isinstance(artifact_path, str):
        raise HardFailure("prediction artifact path is invalid")
    path = root / artifact_path
    _regular(path, "prediction artifact")
    _fail_unless(
        artifact.get("sha256") == sha256_file(path), "prediction SHA256 does not match"
    )
    _fail_unless(
        artifact.get("bytes") == path.stat().st_size,
        "prediction byte count does not match",
    )
    payload = _load_torch(path)
    shape = validate_prediction_payload(payload)
    _fail_unless(
        prediction_manifest.get("shape") == {"K": shape.K, "S": shape.S, "A": shape.A},
        "prediction shape binding is invalid",
    )
    _fail_unless(
        prediction_manifest.get("target_names") == payload.get("target_names"),
        "prediction targets do not match payload",
    )
    _fail_unless(
        prediction_manifest.get("units") == payload.get("units"),
        "prediction units do not match payload",
    )
    return payload


def _compare_value(actual: object, expected: object, name: str) -> None:
    if isinstance(expected, torch.Tensor):
        _exact_or_close(actual, expected, name)
        return
    if isinstance(expected, Mapping):
        actual_mapping = _string_key_mapping(actual, f"{name} is invalid")
        _fail_unless(set(actual_mapping) == set(expected), f"{name} keys differ")
        for key, nested in expected.items():
            _compare_value(actual_mapping[key], nested, f"{name}.{key}")
        return
    if isinstance(expected, Sequence) and not isinstance(expected, (str, bytes)):
        if not isinstance(actual, Sequence) or isinstance(actual, (str, bytes)):
            raise HardFailure(f"{name} is invalid")
        _fail_unless(len(actual) == len(expected), f"{name} length differs")
        for index, nested in enumerate(expected):
            _compare_value(actual[index], nested, f"{name}[{index}]")
        return
    if isinstance(expected, float):
        _fail_unless(
            isinstance(actual, (float, int))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), expected, rel_tol=_RTOL, abs_tol=_ATOL),
            f"{name} differs",
        )
        return
    _fail_unless(
        actual == expected and type(actual) is type(expected), f"{name} differs"
    )


def _validate_evaluation(
    root: Path, config: FGEConfig, payload: Mapping[str, Any]
) -> None:
    directory = root / "evaluation" / "legacy_equal_weight"
    ensemble = _load_torch(directory / "ensemble.pt")
    uncertainty = _load_torch(directory / "uncertainty.pt")
    metrics = _load_json(directory / "metrics.json")
    report_path = directory / "report.md"
    _regular(report_path, "evaluation report")
    evaluation = getattr(config, "evaluation", None)
    if evaluation is None:
        raise HardFailure("validation configuration has no evaluation settings")
    recomputed = evaluate_prediction(
        payload, evaluation.risk_coverages, evaluation.constant_tolerance
    )
    _compare_value(ensemble, recomputed.ensemble, "ensemble")
    _compare_value(uncertainty, recomputed.uncertainty, "uncertainty")
    _compare_value(metrics, recomputed.metrics, "metrics")
    expected_report_inputs = json.dumps(recomputed.report_inputs, sort_keys=True)
    _fail_unless(
        expected_report_inputs in report_path.read_text(encoding="utf-8"),
        "evaluation report inputs differ",
    )


def _preflight_identity(
    config: FGEConfig, training: Mapping[str, Any]
) -> dict[str, object]:
    data = _string_key_mapping(
        training.get("data_identities"), "training data identities are invalid"
    )
    checkpoint = _string_key_mapping(
        training.get("checkpoint_identity"), "checkpoint identity is invalid"
    )
    contract = training.get("model_contract")
    try:
        targets = {
            "energy": config.data.energy_target,
            "forces": config.data.forces_target,
            "stress": config.data.stress_target,
        }
        units = {
            "energy": config.data.energy_unit,
            "forces": config.data.forces_unit,
            "stress": config.data.stress_unit,
        }
        runtime = {
            "device": config.training.device,
            "dtype": config.training.dtype,
        }
    except AttributeError as exc:
        raise HardFailure("validation configuration is incomplete") from exc
    return {
        "inputs": {
            "base_checkpoint": checkpoint.get("sha256"),
            "train_data": _string_key_mapping(data.get("train"), "data identity").get(
                "sha256"
            ),
            "val_data": _string_key_mapping(data.get("val"), "data identity").get(
                "sha256"
            ),
            "test_data": _string_key_mapping(data.get("test"), "data identity").get(
                "sha256"
            ),
        },
        "targets": targets,
        "units": units,
        "model_contract": contract,
        "member_count": training.get("member_count"),
        "runtime": runtime,
    }


def _preflight_flags(config: FGEConfig, stage: str) -> dict[str, bool]:
    try:
        selected = (
            config.scientific.training
            if stage == "train"
            else config.scientific.evaluation
        )
        return {
            name: getattr(selected, name)
            for name in (
                "path_feasibility_only",
                "split_leakage",
                "scientific_evaluation",
                "inference_only",
            )
        }
    except AttributeError as exc:
        raise HardFailure("validation scientific flags are invalid") from exc


def _validate_preflight(
    root: Path,
    config: FGEConfig,
    training: Mapping[str, Any],
    config_resolved: Mapping[str, object],
) -> None:
    expected_identity = _preflight_identity(config, training)
    expected_config = _config_identity(config_resolved)
    bases: set[str] = set()
    for stage in ("train", "predict", "evaluate"):
        report = _load_json(root / "preflight" / f"{stage}.json")
        _fail_unless(
            set(report)
            == {
                "stage",
                "status",
                "basis",
                "identity",
                "scientific_flags",
                "config_identity",
            }
            and report.get("stage") == stage
            and report.get("status") == "PASS"
            and report.get("identity") == expected_identity
            and report.get("scientific_flags") == _preflight_flags(config, stage)
            and report.get("config_identity") == expected_config,
            "preflight report is invalid",
        )
        basis = report.get("basis")
        if basis not in {"runtime_inputs", "canonical_artifacts"}:
            raise HardFailure("preflight basis is invalid")
        bases.add(basis)
    _fail_unless(len(bases) == 1, "preflight bases are inconsistent")


def _expected_manifest_roles(member_count: int) -> dict[str, str]:
    expected = {
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
    for index in range(1, member_count + 1):
        expected[f"training/members/member_{index:03d}.pt"] = (
            f"training_member_{index:03d}"
        )
    return expected


def _validate_formal_tree(
    root: Path,
    member_count: int,
    *,
    validation_exists: bool,
    completion_exists: bool,
) -> None:
    expected = set(_expected_manifest_roles(member_count))
    if not validation_exists:
        expected.remove("validation.json")
    if completion_exists:
        expected.add("result_manifest.json")
    allowed_directories = {
        "preflight",
        "training",
        "training/members",
        "prediction",
        "evaluation",
        "evaluation/legacy_equal_weight",
    }
    actual: set[str] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            _fail_unless(
                not path.is_symlink() and relative in allowed_directories,
                "formal tree contains unallowed directory residue",
            )
            continue
        _regular(path, "formal tree artifact")
        actual.add(relative)
    _fail_unless(actual == expected, "formal tree contains unallowed residue")


def _manifest_artifact_path(root: Path, path_text: object) -> Path:
    if not isinstance(path_text, str):
        raise HardFailure("result manifest path is invalid")
    candidate = Path(path_text)
    if (
        candidate.is_absolute()
        or ".." in candidate.parts
        or path_text == "result_manifest.json"
    ):
        raise HardFailure("result manifest path is invalid")
    try:
        (root / candidate).resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise HardFailure("result manifest path escapes result root") from exc
    return root / candidate


def _validate_completed_manifest(root: Path, expected_project_name: str) -> int:
    manifest = _load_json(root / "result_manifest.json")
    _fail_unless(
        manifest.get("schema_version") == "upet.fge.result.v1"
        and manifest.get("status") == "PASS",
        "result manifest is invalid",
    )
    _fail_unless(
        manifest.get("project_name") == expected_project_name,
        "result manifest project is invalid",
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise HardFailure("result manifest inventory is invalid")
    training = _load_json(root / "training" / "manifest.json")
    for key in ("artifact_writer_code_identity", "validator_code_identity"):
        expected_identity = _code_identity(
            training.get(key), key, allow_unavailable=True
        )
        actual_identity = _code_identity(manifest.get(key), key, allow_unavailable=True)
        _fail_unless(
            actual_identity == expected_identity,
            f"result manifest {key} differs from training manifest",
        )
    member_count = training.get("member_count")
    if isinstance(member_count, bool) or not isinstance(member_count, int):
        raise HardFailure("training member count is invalid")
    expected = _expected_manifest_roles(member_count)
    actual: dict[str, str] = {}
    for artifact in artifacts:
        artifact_mapping = _string_key_mapping(
            artifact, "result manifest artifact is invalid"
        )
        if set(artifact_mapping) != {"role", "path", "bytes", "sha256"}:
            raise HardFailure("result manifest artifact schema is invalid")
        role = artifact_mapping.get("role")
        path = _manifest_artifact_path(root, artifact_mapping.get("path"))
        relative = path.relative_to(root).as_posix()
        if not isinstance(role, str) or expected.get(relative) != role:
            raise HardFailure("result manifest canonical role/inventory is invalid")
        if relative in actual:
            raise HardFailure("result manifest contains duplicate artifact")
        actual[relative] = role
        _regular(path, "result manifest artifact")
        _fail_unless(
            artifact_mapping.get("sha256") == sha256_file(path),
            "result manifest SHA256 differs",
        )
        _fail_unless(
            artifact_mapping.get("bytes") == path.stat().st_size,
            "result manifest byte count differs",
        )
    _fail_unless(actual == expected, "result manifest inventory is incomplete")
    return len(artifacts)


def validate_completed_result(root: str | Path) -> ValidationReport:
    """Reopen and hash-check a completed formal tree without writing any file."""

    result_root = Path(root)
    _fail_unless(
        result_root.is_dir() and not result_root.is_symlink(), "result root is invalid"
    )
    config_resolved = _load_yaml(result_root / "config_resolved.yaml")
    project = config_resolved.get("project")
    if not isinstance(project, Mapping) or not isinstance(project.get("name"), str):
        raise HardFailure("resolved config project identity is invalid")
    training = _load_json(result_root / "training" / "manifest.json")
    member_count = training.get("member_count")
    if isinstance(member_count, bool) or not isinstance(member_count, int):
        raise HardFailure("training member count is invalid")
    _validate_members(result_root, training)
    _validate_formal_tree(
        result_root,
        member_count,
        validation_exists=True,
        completion_exists=True,
    )
    count = _validate_completed_manifest(result_root, project["name"])
    return ValidationReport("PASS", "read_only", count)


def validate_result(
    config: FGEConfig, root: str | Path, *, publish_completion: bool = True
) -> ValidationReport:
    """Reopen every canonical artifact, recompute evaluation, then publish once."""
    result_root = Path(root)
    _fail_unless(
        result_root.is_dir() and not result_root.is_symlink(), "result root is invalid"
    )
    completed = result_root / "result_manifest.json"
    config_resolved = _load_yaml(result_root / "config_resolved.yaml")
    sanitized = config.sanitized()
    _fail_unless(isinstance(sanitized, Mapping), "validation configuration is invalid")
    _fail_unless(
        config_resolved == _canonical_config(sanitized),
        "resolved config differs from validation configuration",
    )
    training = _load_json(result_root / "training" / "manifest.json")
    _validate_preflight(result_root, config, training, config_resolved)
    _fail_unless(
        training.get("config_resolved") == config_resolved,
        "resolved config differs from training manifest",
    )
    prediction_manifest = _load_json(result_root / "prediction" / "manifest.json")
    _identity(config, training, prediction_manifest)
    _validate_members(result_root, training)
    payload = _validate_prediction(result_root, prediction_manifest)
    _validate_evaluation(result_root, config, payload)
    member_count = training.get("member_count")
    if isinstance(member_count, bool) or not isinstance(member_count, int):
        raise HardFailure("training member count is invalid")
    _validate_formal_tree(
        result_root,
        member_count,
        validation_exists=completed.exists()
        or (result_root / "validation.json").exists(),
        completion_exists=completed.exists(),
    )
    if completed.exists():
        return ValidationReport(
            "PASS",
            "read_only",
            _validate_completed_manifest(result_root, config.project.name),
        )
    report = ValidationReport(
        "PASS", "published" if publish_completion else "validated", 0
    )
    if not publish_completion:
        return report
    atomic_write_json(result_root / "validation.json", report.as_json())
    _validate_formal_tree(
        result_root, member_count, validation_exists=True, completion_exists=False
    )
    manifest = build_result_manifest(
        root=result_root,
        project_name=config.project.name,
        artifact_writer_code_identity=training["artifact_writer_code_identity"],
        validator_code_identity=training["validator_code_identity"],
    )
    atomic_write_json(completed, manifest)
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, list):
        raise HardFailure("published result manifest inventory is invalid")
    return ValidationReport("PASS", "published", len(manifest_artifacts))


def _shape(shape: torch.Size, K: int, S: int, A: int) -> list[int | str]:
    symbols: list[int | str] = []
    for dimension in shape:
        if dimension == K:
            symbols.append("K")
        elif dimension == S:
            symbols.append("S")
        elif dimension == A:
            symbols.append("A")
        else:
            symbols.append(int(dimension))
    return symbols


def _tensor_signature(value: object, K: int, S: int, A: int) -> object:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype).removeprefix("torch."),
            "shape": list(value.shape),
        }
    if isinstance(value, Mapping):
        if set(value) == {"name", "dtype", "shape", "value"}:
            name, dtype, shape, tensor = (
                value["name"],
                value["dtype"],
                value["shape"],
                value["value"],
            )
            if (
                not isinstance(name, str)
                or not isinstance(dtype, str)
                or not isinstance(shape, list)
                or not isinstance(tensor, torch.Tensor)
            ):
                raise HardFailure("member tensor signature is invalid")
            return {
                "name": name,
                "dtype": dtype,
                "shape": list(shape),
                "value": {
                    "dtype": str(tensor.dtype).removeprefix("torch."),
                    "shape": list(tensor.shape),
                },
            }
        signature = {
            str(key): _tensor_signature(nested, K, S, A)
            for key, nested in sorted(value.items())
        }
        for key in ("schema_version", "formula_version", "metric_schema_version"):
            literal = value.get(key)
            if literal is not None:
                if isinstance(literal, bool) or not isinstance(literal, (str, int)):
                    raise HardFailure("formal version field is invalid")
                signature[key] = literal
        return signature
    if isinstance(value, tuple) and all(isinstance(item, str) for item in value):
        return ["string"]
    if isinstance(value, (list, tuple)):
        return [_tensor_signature(item, K, S, A) for item in value]
    return type(value).__name__


def _prediction_signature(value: Mapping[str, Any], K: int, S: int, A: int) -> object:
    signature = _string_key_mapping(
        _tensor_signature(value, K, S, A), "prediction signature is invalid"
    )
    for field, symbolic_shape in _PREDICTION_FIELD_SHAPES.items():
        field_signature = signature.get(field)
        field_mapping = _string_key_mapping(
            field_signature, f"prediction signature field is invalid: {field}"
        )
        if "shape" not in field_mapping:
            raise HardFailure(f"prediction signature field is invalid: {field}")
        field_mapping["shape"] = symbolic_shape
        signature[field] = field_mapping
    return signature


def _key_tree(value: object, key: str | None = None) -> object:
    if key in {"schema_version", "formula_version", "metric_schema_version"}:
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            return value
        raise HardFailure("formal version field is invalid")
    if isinstance(value, Mapping):
        return {
            str(nested_key): _key_tree(nested, str(nested_key))
            for nested_key, nested in sorted(value.items())
        }
    if isinstance(value, list):
        return [_key_tree(value[0])] if value else []
    return type(value).__name__


def _result_manifest_signature(document: Mapping[str, Any]) -> dict[str, object]:
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list):
        raise HardFailure("result manifest signature is invalid")
    references: list[dict[str, str]] = []
    for artifact in artifacts:
        mapping = _string_key_mapping(artifact, "result manifest signature is invalid")
        role, path = mapping.get("role"), mapping.get("path")
        if not isinstance(role, str) or not isinstance(path, str):
            raise HardFailure("result manifest signature is invalid")
        if role.startswith("training_member_"):
            role = "training_member_NNN"
        if path.startswith("training/members/member_"):
            path = "training/members/member_NNN.pt"
        references.append({"role": role, "path": path})
    return {
        "schema_version": document.get("schema_version"),
        "project_name": type(document.get("project_name")).__name__,
        "status": type(document.get("status")).__name__,
        "artifact_writer_code_identity": _key_tree(
            document.get("artifact_writer_code_identity")
        ),
        "validator_code_identity": _key_tree(document.get("validator_code_identity")),
        "artifacts": [
            {"role": role, "path": path}
            for path, role in sorted(
                {(item["path"], item["role"]) for item in references}
            )
        ],
    }


def schema_signature(root: str | Path) -> dict[str, object]:
    """Return an identity-free structural signature with symbolic K/S/A dimensions."""
    result_root = Path(root)
    prediction = _load_torch(result_root / "prediction" / "test_raw.pt")
    shape = validate_prediction_payload(prediction)
    documents: dict[str, object] = {}
    tensors: dict[str, object] = {}
    member_signature: object | None = None
    for path in sorted(result_root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(result_root).as_posix()
        if path.name == "result_manifest.json":
            documents[relative] = _result_manifest_signature(_load_json(path))
        elif path.suffix == ".json":
            documents[relative] = _key_tree(_load_json(path))
        elif path.suffix in {".yaml", ".yml"}:
            documents[relative] = _key_tree(_load_yaml(path))
        elif path.suffix == ".pt":
            payload = _load_torch(path)
            if relative == "prediction/test_raw.pt":
                tensors[relative] = _prediction_signature(
                    payload, shape.K, shape.S, shape.A
                )
            elif relative.startswith("training/members/member_"):
                current_signature = _tensor_signature(
                    payload, shape.K, shape.S, shape.A
                )
                if member_signature is None:
                    member_signature = current_signature
                else:
                    _fail_unless(
                        current_signature == member_signature,
                        "training member tensor schemas differ",
                    )
                tensors["training/members/member_NNN.pt"] = member_signature
            else:
                tensors[relative] = _tensor_signature(
                    payload, shape.K, shape.S, shape.A
                )
    return {"documents": documents, "tensors": tensors}
