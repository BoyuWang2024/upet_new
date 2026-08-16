"""Read-only validation for completed inference-only FGE result trees."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from .artifacts import sha256_file
from .errors import HardFailure
from .inference_evaluation import (
    _metrics,
    _uncertainty_identity,
    _validate_uq_payload,
    evaluate_prediction_chunk,
)
from .inference_only import validate_prediction_chunk
from .uncertainty import FORMULA_VERSION


@dataclass(frozen=True)
class InferenceValidationReport:
    status: str
    mode: str
    artifact_count: int


def _json(path: Path, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise HardFailure(f"{label} is not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise HardFailure(f"{label} cannot be loaded") from exc
    if not isinstance(value, Mapping):
        raise HardFailure(f"{label} has an invalid schema")
    return cast(Mapping[str, object], value)


def _torch(path: Path, label: str) -> Mapping[str, object]:
    if path.is_symlink() or not path.is_file():
        raise HardFailure(f"{label} is not a regular file")
    try:
        value = torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure(f"{label} cannot be loaded") from exc
    if not isinstance(value, Mapping):
        raise HardFailure(f"{label} has an invalid schema")
    return cast(Mapping[str, object], value)


def _identity_binding(
    run: Mapping[str, object],
    prediction: Mapping[str, object],
    uncertainty: Mapping[str, object],
) -> None:
    for key in ("run_identity", "ensemble_identity", "dataset_identity", "member_ids"):
        value = run.get(key)
        if prediction.get(key) != value or uncertainty.get(key) != value:
            raise HardFailure(f"inference manifests differ for {key}")
    if prediction.get("chunk_count") != uncertainty.get("chunk_count") or run.get(
        "chunk_count"
    ) != prediction.get("chunk_count"):
        raise HardFailure("inference manifest chunk counts differ")


def _prediction_identity_binding(
    run: Mapping[str, object], payload: Mapping[str, object]
) -> None:
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise HardFailure("prediction chunk identity is invalid")
    run_identity = run.get("run_identity")
    if not isinstance(run_identity, Mapping) or identity.get(
        "run_sha256"
    ) != run_identity.get("sha256"):
        raise HardFailure("prediction chunk run identity differs")
    if identity.get("ensemble") != run.get("ensemble_identity"):
        raise HardFailure("prediction chunk ensemble identity differs")
    dataset = identity.get("dataset")
    run_dataset = run.get("dataset_identity")
    if not isinstance(dataset, Mapping) or not isinstance(run_dataset, Mapping):
        raise HardFailure("prediction chunk dataset identity is invalid")
    for key in ("label", "sha256", "split"):
        if dataset.get(key) != run_dataset.get(key):
            raise HardFailure("prediction chunk dataset identity differs")
    if identity.get("reference_availability") != run_dataset.get(
        "reference_availability"
    ):
        raise HardFailure("prediction chunk reference availability differs")
    if list(cast(tuple[str, ...], identity.get("member_ids"))) != run.get("member_ids"):
        raise HardFailure("prediction chunk member identity differs")


def _entry_path(root: Path, entry: Mapping[str, object], label: str) -> Path:
    relative = entry.get("path")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise HardFailure(f"{label} path is invalid")
    path = root / relative
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise HardFailure(f"{label} path escapes result root") from exc
    return path


def _validate_chunks(
    root: Path,
    run: Mapping[str, object],
    prediction: Mapping[str, object],
    uncertainty: Mapping[str, object],
) -> dict[str, list[torch.Tensor]]:
    prediction_entries = prediction.get("chunks")
    uncertainty_entries = uncertainty.get("chunks")
    if not isinstance(prediction_entries, list) or not isinstance(
        uncertainty_entries, list
    ):
        raise HardFailure("inference chunk manifests are invalid")
    if len(prediction_entries) != len(uncertainty_entries):
        raise HardFailure("prediction and uncertainty chunk counts differ")
    domain_arrays: dict[str, list[torch.Tensor]] = {}
    next_structure = 0
    atom_count = 0
    for index, (prediction_entry, uncertainty_entry) in enumerate(
        zip(prediction_entries, uncertainty_entries, strict=True)
    ):
        if not isinstance(prediction_entry, Mapping) or not isinstance(
            uncertainty_entry, Mapping
        ):
            raise HardFailure("inference chunk entry is invalid")
        chunk_id = f"chunk_{index:06d}"
        if (
            prediction_entry.get("chunk_id") != chunk_id
            or uncertainty_entry.get("chunk_id") != chunk_id
            or prediction_entry.get("start_structure") != next_structure
            or uncertainty_entry.get("start_structure") != next_structure
            or prediction_entry.get("stop_structure")
            != uncertainty_entry.get("stop_structure")
        ):
            raise HardFailure("inference chunk order or range is noncontiguous")
        stop = prediction_entry.get("stop_structure")
        if type(stop) is not int or stop <= next_structure:
            raise HardFailure("inference chunk range is invalid")
        next_structure = stop

        prediction_path = _entry_path(root, prediction_entry, "prediction chunk")
        if (
            prediction_entry.get("sha256") != sha256_file(prediction_path)
            or prediction_entry.get("bytes") != prediction_path.stat().st_size
        ):
            raise HardFailure("prediction chunk hash or bytes differ")
        prediction_payload = _torch(prediction_path, "prediction chunk")
        identity = prediction_payload.get("identity")
        if not isinstance(identity, Mapping):
            raise HardFailure("prediction chunk identity is invalid")
        shape = validate_prediction_chunk(prediction_payload, identity)
        atom_count += shape.A
        _prediction_identity_binding(run, prediction_payload)
        if prediction_entry.get("S") != shape.S or prediction_entry.get("A") != shape.A:
            raise HardFailure("prediction chunk shape differs from manifest")

        uncertainty_path = _entry_path(root, uncertainty_entry, "uncertainty chunk")
        if (
            uncertainty_entry.get("sha256") != sha256_file(uncertainty_path)
            or uncertainty_entry.get("bytes") != uncertainty_path.stat().st_size
        ):
            raise HardFailure("uncertainty chunk hash or bytes differ")
        uncertainty_payload = _torch(uncertainty_path, "uncertainty chunk")
        arrays = evaluate_prediction_chunk(prediction_payload)
        expected_identity = _uncertainty_identity(
            prediction_entry,
            prediction_payload,
        )
        _validate_uq_payload(uncertainty_payload, expected_identity, arrays)
        for domain, prefix in (
            ("energy", "energy_per_atom"),
            ("force", "force_component"),
            ("stress", "stress_component"),
        ):
            domain_arrays.setdefault(f"{domain}_uncertainty", []).append(
                cast(torch.Tensor, uncertainty_payload[f"{prefix}_std"])
            )
            residual = f"{prefix}_absolute_residual"
            if residual in uncertainty_payload:
                domain_arrays.setdefault(f"{domain}_residual", []).append(
                    cast(torch.Tensor, uncertainty_payload[residual])
                )
    dataset = run.get("dataset_identity")
    if (
        not isinstance(dataset, Mapping)
        or next_structure != dataset.get("structure_count")
        or atom_count != dataset.get("atom_count")
    ):
        raise HardFailure("inference chunks do not cover the dataset")
    return domain_arrays


def _validate_metrics(
    root: Path,
    uncertainty: Mapping[str, object],
    arrays: Mapping[str, list[torch.Tensor]],
) -> None:
    metrics_path = root / "metrics.json"
    metrics = _json(metrics_path, "inference metrics")
    if metrics != _metrics(arrays):
        raise HardFailure("inference metrics differ from global recomputation")
    identity = uncertainty.get("metrics")
    if (
        not isinstance(identity, Mapping)
        or identity.get("path") != "metrics.json"
        or identity.get("role") != "metrics"
        or identity.get("sha256") != sha256_file(metrics_path)
        or identity.get("bytes") != metrics_path.stat().st_size
    ):
        raise HardFailure("inference metrics artifact identity differs")


def _validate_inventory(root: Path, run: Mapping[str, object]) -> int:
    artifacts = run.get("artifacts")
    if not isinstance(artifacts, list):
        raise HardFailure("inference result artifact inventory is invalid")
    declared: set[str] = set()
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or set(artifact) != {
            "role",
            "path",
            "bytes",
            "sha256",
        }:
            raise HardFailure("inference result artifact entry is invalid")
        path = _entry_path(root, artifact, "inference artifact")
        relative = path.relative_to(root).as_posix()
        if relative in declared:
            raise HardFailure("inference result artifact inventory has duplicates")
        declared.add(relative)
        if (
            path.is_symlink()
            or not path.is_file()
            or artifact.get("sha256") != sha256_file(path)
            or artifact.get("bytes") != path.stat().st_size
        ):
            raise HardFailure(f"inference artifact hash differs: {relative}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name != "run_manifest.json"
    }
    if actual != declared:
        raise HardFailure("inference result inventory has residue or missing files")
    allowed_directories = {
        "prediction",
        "prediction/chunks",
        "uncertainty",
        "uncertainty/chunks",
    }
    for path in root.rglob("*"):
        if path.is_symlink():
            raise HardFailure("inference result contains a symlink")
        if (
            path.is_dir()
            and path.relative_to(root).as_posix() not in allowed_directories
        ):
            raise HardFailure("inference result contains directory residue")
    return len(artifacts)


def validate_inference_result(
    root: str | Path,
    *,
    read_only: bool = True,
) -> InferenceValidationReport:
    """Recompute and hash-check a completed inference result without writing."""
    if read_only is not True:
        raise HardFailure("inference validation supports read-only mode only")
    result_root = Path(root)
    if result_root.is_symlink() or not result_root.is_dir():
        raise HardFailure("inference result root is invalid")
    run = _json(result_root / "run_manifest.json", "inference run manifest")
    if (
        run.get("schema_version") != "upet.fge.inference-result.v1"
        or run.get("status") != "PASS"
    ):
        raise HardFailure("inference run manifest is not completed")
    if run.get("formula_version") != FORMULA_VERSION:
        raise HardFailure("inference run formula version differs")
    prediction = _json(
        result_root / "prediction" / "manifest.json",
        "prediction manifest",
    )
    uncertainty = _json(
        result_root / "uncertainty" / "manifest.json",
        "uncertainty manifest",
    )
    if uncertainty.get("formula_version") != FORMULA_VERSION:
        raise HardFailure("uncertainty formula version differs")
    _identity_binding(run, prediction, uncertainty)
    arrays = _validate_chunks(result_root, run, prediction, uncertainty)
    _validate_metrics(result_root, uncertainty, arrays)
    count = _validate_inventory(result_root, run)
    return InferenceValidationReport("PASS", "read_only", count)


__all__ = ["InferenceValidationReport", "validate_inference_result"]
