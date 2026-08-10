"""Independent, disk-backed validation for immutable FGE results."""

from __future__ import annotations

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
from .manifests import build_result_manifest
from .prediction import validate_prediction_payload


_ATOL = 2e-7
_RTOL = 2e-6
_FORBIDDEN = ("source", "migration", "bootstrap", "wandb")
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
            lowered = key.lower()
            _fail_unless(
                not any(marker in lowered for marker in _FORBIDDEN),
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
        lowered = value.lower()
        _fail_unless(
            not any(marker in lowered for marker in _FORBIDDEN),
            "formal artifact contains forbidden provenance",
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


def _validate_members(root: Path, training: Mapping[str, Any]) -> None:
    members = training.get("members")
    if not isinstance(members, list):
        raise HardFailure("training members are invalid")
    for member in members:
        if not isinstance(member, Mapping):
            raise HardFailure("training member is invalid")
        member_id = member.get("member_id")
        _fail_unless(isinstance(member_id, str), "training member ID is invalid")
        path = root / "training" / "members" / f"{member_id}.pt"
        _fail_unless(
            member.get("sha256") == sha256_file(path), "member SHA256 does not match"
        )
        payload = _load_torch(path)
        _fail_unless(
            payload.get("member_id") == member_id, "member payload ID is invalid"
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
    return payload


def _validate_evaluation(
    root: Path, config: FGEConfig, payload: Mapping[str, Any]
) -> None:
    directory = root / "evaluation" / "legacy_equal_weight"
    ensemble = _load_torch(directory / "ensemble.pt")
    uncertainty = _load_torch(directory / "uncertainty.pt")
    metrics = _load_json(directory / "metrics.json")
    _regular(directory / "report.md", "evaluation report")
    evaluation = getattr(config, "evaluation", None)
    if evaluation is None:
        raise HardFailure("validation configuration has no evaluation settings")
    recomputed = evaluate_prediction(
        payload, evaluation.risk_coverages, evaluation.constant_tolerance
    )
    _fail_unless(
        set(ensemble) == {"energy", "forces", "stress"}, "ensemble keys are invalid"
    )
    for name, expected in recomputed.ensemble.items():
        _exact_or_close(ensemble[name], expected, f"ensemble.{name}")
    _fail_unless(
        uncertainty.get("formula_version") == evaluation.formula_version,
        "uncertainty formula is invalid",
    )
    for name, expected in recomputed.uncertainty["energy_total"].items():
        _exact_or_close(
            uncertainty.get("energy_total", {}).get(name),
            expected,
            f"uncertainty.energy_total.{name}",
        )
    _fail_unless(
        metrics.get("schema_version") == evaluation.metric_schema_version,
        "metrics schema is invalid",
    )
    stored_mae = metrics.get("mae")
    if not isinstance(stored_mae, Mapping):
        raise HardFailure("metrics MAE is invalid")
    for name, expected in recomputed.metrics["mae"].items():
        value = stored_mae.get(name)
        _fail_unless(
            isinstance(value, (int, float))
            and math.isclose(
                float(value), float(expected), rel_tol=_RTOL, abs_tol=_ATOL
            ),
            f"metrics MAE differs: {name}",
        )


def _validate_preflight(root: Path) -> None:
    for stage in ("train", "predict", "evaluate"):
        report = _load_json(root / "preflight" / f"{stage}.json")
        _fail_unless(
            report.get("stage") == stage and report.get("status") == "PASS",
            "preflight report is invalid",
        )


def _validate_completed_manifest(root: Path, config: FGEConfig) -> int:
    manifest = _load_json(root / "result_manifest.json")
    _fail_unless(
        manifest.get("schema_version") == "upet.fge.result.v1"
        and manifest.get("status") == "PASS",
        "result manifest is invalid",
    )
    _fail_unless(
        manifest.get("project_name") == config.project.name,
        "result manifest project is invalid",
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise HardFailure("result manifest inventory is invalid")
    for artifact in artifacts:
        artifact_mapping = _string_key_mapping(
            artifact, "result manifest artifact is invalid"
        )
        path_text = artifact_mapping.get("path")
        if (
            not isinstance(path_text, str)
            or path_text == "result_manifest.json"
            or Path(path_text).is_absolute()
        ):
            raise HardFailure("result manifest path is invalid")
        path = root / path_text
        _regular(path, "result manifest artifact")
        _fail_unless(
            artifact_mapping.get("sha256") == sha256_file(path),
            "result manifest SHA256 differs",
        )
        _fail_unless(
            artifact_mapping.get("bytes") == path.stat().st_size,
            "result manifest byte count differs",
        )
    return len(artifacts)


def validate_result(
    config: FGEConfig, root: str | Path, *, publish_completion: bool = True
) -> ValidationReport:
    """Reopen every canonical artifact, recompute evaluation, then publish once."""
    result_root = Path(root)
    _fail_unless(
        result_root.is_dir() and not result_root.is_symlink(), "result root is invalid"
    )
    completed = result_root / "result_manifest.json"
    _validate_preflight(result_root)
    config_resolved = _load_yaml(result_root / "config_resolved.yaml")
    training = _load_json(result_root / "training" / "manifest.json")
    _fail_unless(
        training.get("config_resolved") == config_resolved,
        "resolved config differs from training manifest",
    )
    prediction_manifest = _load_json(result_root / "prediction" / "manifest.json")
    _identity(config, training, prediction_manifest)
    _validate_members(result_root, training)
    payload = _validate_prediction(result_root, prediction_manifest)
    _validate_evaluation(result_root, config, payload)
    if completed.exists():
        return ValidationReport(
            "PASS", "read_only", _validate_completed_manifest(result_root, config)
        )
    report = ValidationReport(
        "PASS", "published" if publish_completion else "validated", 0
    )
    if not publish_completion:
        return report
    atomic_write_json(result_root / "validation.json", report.as_json())
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
            "shape": _shape(value.shape, K, S, A),
        }
    if isinstance(value, Mapping):
        return {
            str(key): _tensor_signature(nested, K, S, A)
            for key, nested in sorted(value.items())
        }
    if isinstance(value, tuple):
        if not value:
            return []
        return ["string" if isinstance(value[0], str) else type(value[0]).__name__]
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


def _key_tree(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _key_tree(nested) for key, nested in sorted(value.items())}
    if isinstance(value, list):
        return [_key_tree(value[0])] if value else []
    return type(value).__name__


def schema_signature(root: str | Path) -> dict[str, object]:
    """Return an identity-free structural signature with symbolic K/S/A dimensions."""
    result_root = Path(root)
    prediction = _load_torch(result_root / "prediction" / "test_raw.pt")
    shape = validate_prediction_payload(prediction)
    documents: dict[str, object] = {}
    tensors: dict[str, object] = {}
    member_signature: object | None = None
    for path in sorted(result_root.rglob("*")):
        if not path.is_file() or path.name == "result_manifest.json":
            continue
        relative = path.relative_to(result_root).as_posix()
        if path.suffix == ".json":
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
