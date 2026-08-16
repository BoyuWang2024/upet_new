"""Population-STD uncertainty evaluation for chunked inference-only results."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import cast

import torch

from .artifacts import (
    assert_safe_result_path,
    atomic_torch_save,
    atomic_write_json,
    sha256_file,
)
from .errors import HardFailure
from .inference_authority import open_ensemble_authority
from .inference_config import InferenceConfig
from .inference_data import iter_dataset_chunks, scan_dataset
from .inference_only import (
    _chunk_identity,
    _load_existing_manifest,
    _manifest_chunk,
    _manifest_header,
    _validate_manifest_entry,
    _validate_manifest_header,
    validate_prediction_chunk,
)
from .uncertainty import (
    FORMULA_VERSION,
    population_std,
    tensor_to_voigt_symmetric,
)


_UQ_BASE_KEYS = {
    "schema_version",
    "formula_version",
    "identity",
    "energy_per_atom_std",
    "force_component_std",
    "stress_component_std",
    "content_sha256",
}
_RESIDUAL_KEYS = {
    "energy": "energy_per_atom_absolute_residual",
    "forces": "force_component_absolute_residual",
    "stress": "stress_component_absolute_residual",
}


def _prediction_tensor(
    payload: Mapping[str, object], name: str, rank: int
) -> torch.Tensor:
    value = payload.get(name)
    if (
        not isinstance(value, torch.Tensor)
        or value.device.type != "cpu"
        or value.dtype != torch.float32
        or value.ndim != rank
        or not bool(torch.isfinite(value).all())
    ):
        raise HardFailure(f"prediction chunk {name} is invalid for UQ")
    return value


def evaluate_prediction_chunk(
    payload: Mapping[str, object],
) -> dict[str, torch.Tensor]:
    """Compute component-grain population STD and available absolute residuals."""
    energy = _prediction_tensor(payload, "energy_prediction", 2)
    forces = _prediction_tensor(payload, "forces_prediction", 3)
    stress = _prediction_tensor(payload, "stress_prediction", 4)
    n_atoms = payload.get("n_atoms")
    if (
        not isinstance(n_atoms, torch.Tensor)
        or n_atoms.dtype != torch.int64
        or n_atoms.ndim != 1
        or bool((n_atoms <= 0).any())
        or energy.shape[1] != n_atoms.shape[0]
    ):
        raise HardFailure("prediction chunk n_atoms is invalid for UQ")
    energy_per_atom = energy / n_atoms.to(dtype=energy.dtype).unsqueeze(0)
    force_components = forces.reshape(forces.shape[0], -1)
    stress_components = tensor_to_voigt_symmetric(stress).reshape(stress.shape[0], -1)
    result = {
        "energy_per_atom_std": population_std(energy_per_atom),
        "force_component_std": population_std(force_components),
        "stress_component_std": population_std(stress_components),
    }

    energy_reference = payload.get("energy_reference")
    if energy_reference is not None:
        if not isinstance(energy_reference, torch.Tensor):
            raise HardFailure("energy reference is invalid for UQ")
        reference = energy_reference / n_atoms.to(dtype=energy_reference.dtype)
        result[_RESIDUAL_KEYS["energy"]] = torch.abs(
            energy_per_atom.mean(dim=0) - reference
        )
    forces_reference = payload.get("forces_reference")
    if forces_reference is not None:
        if not isinstance(forces_reference, torch.Tensor):
            raise HardFailure("forces reference is invalid for UQ")
        result[_RESIDUAL_KEYS["forces"]] = torch.abs(
            forces.mean(dim=0) - forces_reference
        ).reshape(-1)
    stress_reference = payload.get("stress_reference")
    if stress_reference is not None:
        if not isinstance(stress_reference, torch.Tensor):
            raise HardFailure("stress reference is invalid for UQ")
        reference_voigt = tensor_to_voigt_symmetric(stress_reference).reshape(-1)
        result[_RESIDUAL_KEYS["stress"]] = torch.abs(
            stress_components.mean(dim=0) - reference_voigt
        )
    for name, tensor in result.items():
        if tensor.ndim != 1 or not bool(torch.isfinite(tensor).all()):
            raise HardFailure(f"inference UQ {name} must be a finite vector")
    return result


def _tensor_content_sha256(payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256()
    metadata = {
        key: value
        for key, value in payload.items()
        if not isinstance(value, torch.Tensor) and key != "content_sha256"
    }
    digest.update(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    for name in sorted(
        key for key, value in payload.items() if isinstance(value, torch.Tensor)
    ):
        tensor = cast(torch.Tensor, payload[name]).detach().cpu().contiguous()
        digest.update(name.encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _uncertainty_identity(
    prediction_entry: Mapping[str, object],
    prediction_payload: Mapping[str, object],
) -> dict[str, object]:
    identity = prediction_payload.get("identity")
    if not isinstance(identity, Mapping):
        raise HardFailure("prediction identity is invalid for UQ")
    return {
        "prediction_path": prediction_entry["path"],
        "prediction_sha256": prediction_entry["sha256"],
        "prediction_content_sha256": prediction_payload["content_sha256"],
        "chunk_id": identity["chunk_id"],
        "chunk_index": identity["chunk_index"],
        "start_structure": identity["start_structure"],
        "stop_structure": identity["stop_structure"],
        "reference_availability": dict(
            cast(Mapping[str, object], identity["reference_availability"])
        ),
    }


def _build_uq_payload(
    prediction_entry: Mapping[str, object],
    prediction_payload: Mapping[str, object],
) -> dict[str, object]:
    arrays = evaluate_prediction_chunk(prediction_payload)
    payload: dict[str, object] = {
        "schema_version": "upet.fge.inference-uncertainty-chunk.v1",
        "formula_version": FORMULA_VERSION,
        "identity": _uncertainty_identity(prediction_entry, prediction_payload),
        **arrays,
    }
    payload["content_sha256"] = _tensor_content_sha256(payload)
    _validate_uq_payload(payload, payload["identity"], arrays)
    return payload


def _validate_uq_payload(
    payload: Mapping[str, object],
    expected_identity: object,
    expected_arrays: Mapping[str, torch.Tensor],
) -> None:
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise HardFailure("uncertainty chunk identity is invalid")
    availability = identity.get("reference_availability")
    if not isinstance(availability, Mapping):
        raise HardFailure("uncertainty reference availability is invalid")
    expected_keys = set(_UQ_BASE_KEYS)
    for observable, key in _RESIDUAL_KEYS.items():
        availability_key = "forces" if observable == "forces" else observable
        if availability.get(availability_key) is True:
            expected_keys.add(key)
    if set(payload) != expected_keys:
        raise HardFailure("uncertainty chunk has missing or unknown keys")
    if payload.get("formula_version") != FORMULA_VERSION:
        raise HardFailure("uncertainty chunk formula version differs")
    if identity != expected_identity:
        raise HardFailure("uncertainty chunk identity differs")
    for name, expected in expected_arrays.items():
        actual = payload.get(name)
        if not isinstance(actual, torch.Tensor) or not torch.equal(actual, expected):
            raise HardFailure(f"uncertainty chunk {name} differs from recomputation")
    if payload.get("content_sha256") != _tensor_content_sha256(payload):
        raise HardFailure("uncertainty chunk content SHA-256 differs")


def _open_prediction_payload(path: Path) -> Mapping[str, object]:
    try:
        value = torch.load(path, weights_only=True, map_location="cpu")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise HardFailure("prediction chunk cannot be loaded for UQ") from exc
    if not isinstance(value, Mapping):
        raise HardFailure("prediction chunk is not a mapping for UQ")
    return cast(Mapping[str, object], value)


def _verified_prediction_chunks(
    config: InferenceConfig,
) -> Iterator[tuple[Mapping[str, object], Mapping[str, object]]]:
    authority = open_ensemble_authority(config)
    scan = scan_dataset(config)
    manifest_path = config.output.root / "prediction" / "manifest.json"
    manifest = _load_existing_manifest(manifest_path)
    if manifest is None:
        raise HardFailure("prediction manifest is required before UQ")
    header = _manifest_header(config, authority, scan)
    entries = _validate_manifest_header(manifest, header)
    count = 0
    for chunk in iter_dataset_chunks(config, scan):
        if chunk.chunk_index >= len(entries):
            raise HardFailure("prediction manifest is missing a chunk for UQ")
        entry = entries[chunk.chunk_index]
        relative = entry.get("path")
        if not isinstance(relative, str):
            raise HardFailure("prediction chunk path is invalid for UQ")
        path = config.output.root / relative
        payload = _open_prediction_payload(path)
        identity = _chunk_identity(config, authority, chunk)
        shape = validate_prediction_chunk(payload, identity)
        expected_entry = _manifest_chunk(config.output.root, path, chunk, shape)
        _validate_manifest_entry(config.output.root, entry, expected_entry)
        count += 1
        yield entry, payload
    if count != len(entries):
        raise HardFailure("prediction manifest has extra chunks for UQ")


def _average_tied_ranks(values: torch.Tensor) -> torch.Tensor:
    order = sorted(
        range(values.numel()), key=lambda index: (float(values[index]), index)
    )
    ranks = torch.empty_like(values, dtype=torch.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(
            values[order[start]]
        ):
            end += 1
        rank = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def _pearson(left: torch.Tensor, right: torch.Tensor) -> float | None:
    left = left.to(dtype=torch.float64)
    right = right.to(dtype=torch.float64)
    left = left - left.mean()
    right = right - right.mean()
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) == 0.0:
        return None
    return float(torch.dot(left, right) / denominator)


def _summary(values: torch.Tensor) -> dict[str, float] | None:
    values = values.to(dtype=torch.float64)
    if values.numel() == 0:
        return None
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "median": float(values.median()),
    }


def _domain_metrics(
    uncertainty: torch.Tensor,
    residual: torch.Tensor,
) -> dict[str, object]:
    uncertainty = uncertainty.reshape(-1).to(dtype=torch.float64)
    residual = residual.reshape(-1).to(dtype=torch.float64)
    if uncertainty.shape != residual.shape or uncertainty.numel() == 0:
        raise HardFailure("global inference metric arrays are invalid")
    finite = torch.isfinite(uncertainty) & torch.isfinite(residual)
    positive = finite & (uncertainty > 0) & (residual > 0)
    finite_u = uncertainty[finite]
    finite_r = residual[finite]
    if finite_u.numel() == 0:
        raise HardFailure("global inference metrics have no finite pairs")
    spearman = _pearson(_average_tied_ranks(finite_u), _average_tied_ranks(finite_r))
    pearson_log10 = (
        _pearson(torch.log10(uncertainty[positive]), torch.log10(residual[positive]))
        if int(positive.sum()) >= 2
        else None
    )
    ratio_mask = finite & (uncertainty > 0)
    return {
        "count_total": uncertainty.numel(),
        "count_finite": int(finite.sum()),
        "count_positive_log": int(positive.sum()),
        "excluded_nan": int((torch.isnan(uncertainty) | torch.isnan(residual)).sum()),
        "excluded_inf": int((torch.isinf(uncertainty) | torch.isinf(residual)).sum()),
        "excluded_nonpositive": int((finite & ~positive).sum()),
        "spearman": spearman,
        "pearson_log10": pearson_log10,
        "uncertainty": _summary(finite_u),
        "absolute_residual": _summary(finite_r),
        "residual_to_uncertainty": _summary(
            residual[ratio_mask] / uncertainty[ratio_mask]
        ),
    }


def _metrics(domain_arrays: Mapping[str, list[torch.Tensor]]) -> dict[str, object]:
    domains: dict[str, object] = {}
    for domain in ("energy", "force", "stress"):
        uncertainty = domain_arrays.get(f"{domain}_uncertainty", [])
        residual = domain_arrays.get(f"{domain}_residual", [])
        if uncertainty and residual:
            domains[domain] = _domain_metrics(
                torch.cat(uncertainty),
                torch.cat(residual),
            )
    return {
        "schema_version": "upet.fge.inference-metrics.v1",
        "formula_version": FORMULA_VERSION,
        "domains": domains,
    }


def _publish_or_match_json(path: Path, value: Mapping[str, object], label: str) -> None:
    if not path.exists():
        atomic_write_json(path, value)
        return
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise HardFailure(f"existing {label} cannot be loaded") from exc
    if stored != value:
        raise HardFailure(f"existing {label} differs from recomputation")


def _artifact(root: Path, path: Path, role: str) -> dict[str, object]:
    return {
        "role": role,
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _run_manifest(
    root: Path,
    prediction_manifest: Mapping[str, object],
    uncertainty_manifest: Mapping[str, object],
) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for path in sorted(
        item
        for item in root.rglob("*")
        if item.is_file() and item.name != "run_manifest.json"
    ):
        relative = path.relative_to(root).as_posix()
        if relative == "prediction/manifest.json":
            role = "prediction_manifest"
        elif relative == "uncertainty/manifest.json":
            role = "uncertainty_manifest"
        elif relative == "metrics.json":
            role = "metrics"
        elif relative.startswith("prediction/chunks/chunk_"):
            role = f"prediction_{path.stem}"
        elif relative.startswith("uncertainty/chunks/chunk_"):
            role = f"uncertainty_{path.stem}"
        else:
            raise HardFailure(f"inference result has unrecognized artifact: {relative}")
        artifacts.append(_artifact(root, path, role))
    return {
        "schema_version": "upet.fge.inference-result.v1",
        "status": "PASS",
        "formula_version": FORMULA_VERSION,
        "run_identity": prediction_manifest["run_identity"],
        "ensemble_identity": prediction_manifest["ensemble_identity"],
        "dataset_identity": prediction_manifest["dataset_identity"],
        "member_ids": prediction_manifest["member_ids"],
        "chunk_count": uncertainty_manifest["chunk_count"],
        "artifacts": artifacts,
    }


def evaluate_inference_dataset(config: InferenceConfig) -> Path:
    """Publish resumable UQ chunks, global metrics, and completion manifest."""
    root = config.output.root
    prediction_manifest_path = root / "prediction" / "manifest.json"
    prediction_manifest = _load_existing_manifest(prediction_manifest_path)
    if prediction_manifest is None:
        raise HardFailure("prediction manifest is required before UQ")
    run_manifest_path = root / "run_manifest.json"
    if run_manifest_path.exists():
        from .inference_validation import validate_inference_result

        validate_inference_result(root)
        return root / "uncertainty" / "manifest.json"

    entries: list[dict[str, object]] = []
    domain_arrays: dict[str, list[torch.Tensor]] = {}
    for prediction_entry, prediction_payload in _verified_prediction_chunks(config):
        identity = _uncertainty_identity(prediction_entry, prediction_payload)
        chunk_id = cast(str, identity["chunk_id"])
        path = root / "uncertainty" / "chunks" / f"{chunk_id}.pt"
        assert_safe_result_path(root, path)
        expected_arrays = evaluate_prediction_chunk(prediction_payload)
        expected = _build_uq_payload(prediction_entry, prediction_payload)
        if not path.exists():
            atomic_torch_save(path, expected)
        try:
            stored = torch.load(path, weights_only=True, map_location="cpu")
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            raise HardFailure("uncertainty chunk cannot be reopened") from exc
        if not isinstance(stored, Mapping):
            raise HardFailure("uncertainty chunk is not a mapping")
        _validate_uq_payload(stored, identity, expected_arrays)
        prediction_identity = cast(Mapping[str, object], prediction_payload["identity"])
        entry = {
            "chunk_id": chunk_id,
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "start_structure": prediction_identity["start_structure"],
            "stop_structure": prediction_identity["stop_structure"],
        }
        entries.append(entry)
        for domain, prefix in (
            ("energy", "energy_per_atom"),
            ("force", "force_component"),
            ("stress", "stress_component"),
        ):
            domain_arrays.setdefault(f"{domain}_uncertainty", []).append(
                cast(torch.Tensor, stored[f"{prefix}_std"])
            )
            residual_name = f"{prefix}_absolute_residual"
            if residual_name in stored:
                domain_arrays.setdefault(f"{domain}_residual", []).append(
                    cast(torch.Tensor, stored[residual_name])
                )

    metrics = _metrics(domain_arrays)
    metrics_path = root / "metrics.json"
    assert_safe_result_path(root, metrics_path)
    _publish_or_match_json(metrics_path, metrics, "inference metrics")
    uncertainty_manifest = {
        "schema_version": "upet.fge.inference-uncertainty.v1",
        "formula_version": FORMULA_VERSION,
        "run_identity": prediction_manifest["run_identity"],
        "ensemble_identity": prediction_manifest["ensemble_identity"],
        "dataset_identity": prediction_manifest["dataset_identity"],
        "member_ids": prediction_manifest["member_ids"],
        "chunk_count": len(entries),
        "chunks": entries,
        "metrics": _artifact(root, metrics_path, "metrics"),
    }
    uncertainty_manifest_path = root / "uncertainty" / "manifest.json"
    assert_safe_result_path(root, uncertainty_manifest_path)
    _publish_or_match_json(
        uncertainty_manifest_path,
        uncertainty_manifest,
        "uncertainty manifest",
    )
    run_manifest = _run_manifest(root, prediction_manifest, uncertainty_manifest)
    assert_safe_result_path(root, run_manifest_path)
    atomic_write_json(run_manifest_path, run_manifest)
    return uncertainty_manifest_path


__all__ = ["evaluate_inference_dataset", "evaluate_prediction_chunk"]
