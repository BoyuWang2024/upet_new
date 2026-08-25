"""Audited publication of ConfidenceHead E0-derived energy variants."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from ase.data import chemical_symbols
from torch import Tensor

from .artifacts import (
    atomic_torch_save,
    atomic_write_json,
    load_verified_torch,
    sha256_file,
)
from .binning import labels_from_thresholds
from .e0_calibration import (
    E0Dataset,
    SvdSolution,
    apply_direct_e0,
    apply_model_aware_e0,
    energy_metrics,
    energy_observed_errors,
    fit_direct_mad_e0,
    fit_model_aware,
)
from .external_prediction import verify_external_prediction
from .metrics import classification_metrics


E0_CAMPAIGN_SCHEMA_VERSION = "upet_confidence_e0_postprocessing_v1"
E0_VARIANTS = (
    "uncorrected",
    "direct_mad_e0_test_informed",
    "model_aware_reestimated_val_calibrated",
)


@dataclass(frozen=True)
class E0CampaignInputs:
    """All verified numeric and raw prediction inputs for one campaign."""

    checkpoint_sha256: str
    validation_sha256: str
    test_sha256: str
    atomic_types: tuple[int, ...]
    model_e0: Tensor
    validation: E0Dataset
    test: E0Dataset
    validation_sources: Mapping[int, Path]
    test_sources: Mapping[int, Path]


@dataclass(frozen=True)
class _RawOrder:
    order: int
    source: Path
    manifest_sha256: str
    predictions_sha256: str
    predictions: Mapping[str, Any]


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"e0-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _mapping(path: Path, *, context: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid {context} {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{context} must contain a mapping")
    return value


def _confined(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("E0 artifact path must be a relative string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("E0 artifact path must be relative")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("E0 artifact escapes campaign root")
    return resolved


def _external(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or Path(relative).is_absolute():
        raise ValueError("raw input path must be relative")
    return (root / relative).resolve()


def _relative(root: Path, path: Path) -> str:
    return Path(os.path.relpath(Path(path).resolve(), Path(root).resolve())).as_posix()


def _tensor(
    predictions: Mapping[str, Any],
    name: str,
    *,
    dtype: torch.dtype | None = None,
) -> Tensor:
    if name not in predictions:
        raise ValueError(f"raw energy prediction is missing {name}")
    value = torch.as_tensor(predictions[name]).detach().cpu()
    if dtype is not None:
        value = value.to(dtype)
    if value.numel() == 0 or (
        value.is_floating_point() and not bool(torch.isfinite(value).all())
    ):
        raise ValueError(f"raw energy prediction {name} must be finite and non-empty")
    return value


def _load_raw_order(source: Path, order: int) -> _RawOrder:
    root = Path(source).resolve()
    manifest_path = root / "manifest.json"
    manifest = verify_external_prediction(manifest_path, full=True)
    payload = manifest["identity_payload"]
    if payload.get("target") != "energy" or payload.get("order") != order:
        raise ValueError(f"raw prediction target/order mismatch for order {order}")
    descriptor = manifest["artifacts"]["predictions.pt"]
    predictions_path = root / str(descriptor["path"])
    loaded = load_verified_torch(
        predictions_path,
        expected_sha256=str(descriptor["sha256"]),
        weights_only=True,
    )
    if not isinstance(loaded, Mapping):
        raise ValueError("raw predictions must contain a mapping")
    return _RawOrder(
        order=order,
        source=root,
        manifest_sha256=sha256_file(manifest_path),
        predictions_sha256=sha256_file(predictions_path),
        predictions=cast(Mapping[str, Any], loaded),
    )


def _validate_dataset(raw: _RawOrder, dataset: E0Dataset, label: str) -> None:
    structure_ids = _tensor(raw.predictions, "structure_ids").reshape(-1)
    offsets = _tensor(raw.predictions, "atom_offsets").reshape(-1)
    reference = _tensor(
        raw.predictions,
        "energy_reference",
        dtype=torch.float32,
    ).reshape(-1)
    if not torch.equal(structure_ids.to(torch.int64), dataset.structure_ids):
        raise ValueError(f"{label} structure IDs mismatch for order {raw.order}")
    if not torch.equal(offsets.to(torch.int64), dataset.atom_offsets):
        raise ValueError(f"{label} atom offsets mismatch for order {raw.order}")
    expected_reference = dataset.target_energy_r2scan.to(torch.float32)
    if not torch.equal(reference, expected_reference):
        raise ValueError(f"{label} energy reference mismatch for order {raw.order}")


def _load_raw_set(
    sources: Mapping[int, Path],
    dataset: E0Dataset,
    label: str,
) -> dict[int, _RawOrder]:
    if set(sources) != set(range(1, 9)):
        raise ValueError(f"{label} raw predictions must contain orders 1 through 8")
    loaded = {
        order: _load_raw_order(Path(sources[order]), order) for order in range(1, 9)
    }
    baseline = loaded[1]
    _validate_dataset(baseline, dataset, label)
    raw_energy = _tensor(
        baseline.predictions,
        "energy_prediction",
        dtype=torch.float32,
    ).reshape(-1)
    representatives = _tensor(
        baseline.predictions,
        "energy_representatives",
        dtype=torch.float32,
    ).reshape(-1)
    for order in range(2, 9):
        current = loaded[order]
        _validate_dataset(current, dataset, label)
        if not torch.equal(
            _tensor(
                current.predictions,
                "energy_prediction",
                dtype=torch.float32,
            ).reshape(-1),
            raw_energy,
        ):
            raise ValueError(f"{label} raw energy mismatch for order {order}")
        if not torch.equal(
            _tensor(
                current.predictions,
                "energy_representatives",
                dtype=torch.float32,
            ).reshape(-1),
            representatives,
        ):
            raise ValueError(f"{label} representatives mismatch for order {order}")
    return loaded


def _thresholds(raw: _RawOrder) -> Tensor:
    representatives = _tensor(
        raw.predictions,
        "energy_representatives",
        dtype=torch.float64,
    ).reshape(-1)
    if representatives.numel() < 3:
        raise ValueError("energy representatives must contain at least three bins")
    widths = representatives[1:] - representatives[:-1]
    if not bool(torch.all(widths > 0)):
        raise ValueError("energy representatives must be strictly increasing")
    return (representatives[:-1] + representatives[1:]) / 2.0


def _raw_bindings(root: Path, values: Mapping[int, _RawOrder]) -> dict[str, Any]:
    bindings: dict[str, Any] = {}
    for order, raw in values.items():
        manifest = raw.source / "manifest.json"
        predictions = raw.source / "predictions.pt"
        bindings[str(order)] = {
            "manifest": {
                "path": _relative(root, manifest),
                "sha256": raw.manifest_sha256,
            },
            "predictions": {
                "path": _relative(root, predictions),
                "sha256": raw.predictions_sha256,
            },
        }
    return bindings


def _solution_payload(solution: SvdSolution) -> dict[str, Any]:
    return {
        "solution": solution.solution.tolist(),
        "rank": solution.rank,
        "singular_values": solution.singular_values.tolist(),
        "condition_number": solution.condition_number,
        "residual_rmse_ev_per_structure": solution.residual_rmse,
        "residual_max_abs_ev_per_structure": solution.residual_max_abs,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("E0 comparison CSV requires at least one row")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _artifact_descriptors(root: Path) -> dict[str, dict[str, str]]:
    return {
        path.relative_to(root).as_posix(): {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _all_artifact_descriptors(root: Path) -> dict[str, dict[str, str]]:
    return {
        path.relative_to(root).as_posix(): {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != root / "manifest.json"
    }


def _variant_metrics(
    orders: Mapping[int, _RawOrder],
    corrected: Tensor,
    target: Tensor,
    atom_counts: Tensor,
    observed: Tensor,
    labels: Tensor,
) -> dict[str, Any]:
    by_order: dict[str, Any] = {}
    for order, raw in orders.items():
        logits = _tensor(raw.predictions, "energy_logits", dtype=torch.float64)
        representatives = _tensor(
            raw.predictions,
            "energy_representatives",
            dtype=torch.float64,
        ).reshape(-1)
        by_order[str(order)] = classification_metrics(
            logits,
            labels,
            observed,
            representatives,
        )
    return {
        "energy": energy_metrics(corrected, target, atom_counts),
        "uq_by_order": by_order,
    }


def _variant_data(
    dataset: E0Dataset,
    raw: Tensor,
    corrected: Tensor,
    thresholds: Tensor,
) -> dict[str, Tensor]:
    observed = energy_observed_errors(
        corrected,
        dataset.target_energy_r2scan,
        dataset.atom_counts,
    )
    return {
        "structure_ids": dataset.structure_ids,
        "atom_counts": dataset.atom_counts,
        "composition": dataset.composition,
        "model_energy_raw": raw,
        "target_energy_r2scan": dataset.target_energy_r2scan,
        "model_energy_corrected": corrected,
        "energy_observed_errors": observed,
        "energy_labels": labels_from_thresholds(observed, thresholds),
    }


def _publish_variant(
    root: Path,
    name: str,
    data: Mapping[str, Tensor],
    orders: Mapping[int, _RawOrder],
    raw_bindings: Mapping[str, Any],
    calibration: str | None,
) -> None:
    variant = root / "variants" / name
    variant.mkdir(parents=True)
    data_path = variant / "energy_data.pt"
    atomic_torch_save(data_path, data)
    metrics_path = variant / "metrics.json"
    atomic_write_json(
        metrics_path,
        _variant_metrics(
            orders,
            data["model_energy_corrected"],
            data["target_energy_r2scan"],
            data["atom_counts"],
            data["energy_observed_errors"],
            data["energy_labels"],
        ),
    )
    payload = {
        "variant": name,
        "raw_inputs": raw_bindings,
        "calibration": calibration,
    }
    manifest = {
        "schema_version": E0_CAMPAIGN_SCHEMA_VERSION,
        "status": "complete",
        "variant": name,
        "identity": _identity(payload),
        "identity_payload": payload,
        "artifacts": _artifact_descriptors(variant),
    }
    atomic_write_json(variant / "manifest.json", manifest)


def _validate_input_shapes(inputs: E0CampaignInputs) -> None:
    atomic_types = tuple(int(value) for value in inputs.atomic_types)
    if atomic_types != tuple(sorted(set(atomic_types))):
        raise ValueError("campaign atomic_types must be unique and sorted")
    model_e0 = torch.as_tensor(inputs.model_e0, dtype=torch.float64).reshape(-1)
    if model_e0.shape != (len(atomic_types),) or not bool(
        torch.isfinite(model_e0).all()
    ):
        raise ValueError("campaign model E0 shape or values are invalid")
    for label, dataset in (
        ("validation", inputs.validation),
        ("test", inputs.test),
    ):
        structures = len(dataset.structure_ids)
        if dataset.composition.shape != (structures, len(atomic_types)):
            raise ValueError(f"{label} composition shape mismatch")
        if dataset.atom_offsets.shape != (structures + 1,):
            raise ValueError(f"{label} atom offsets shape mismatch")
        if int(dataset.atom_offsets[-1]) != len(dataset.atomic_numbers):
            raise ValueError(f"{label} atom count mismatch")
        if not bool(torch.isfinite(dataset.composition).all()):
            raise ValueError(f"{label} composition must be finite")
        observed_types = set(int(value) for value in dataset.atomic_numbers.tolist())
        if not observed_types.issubset(atomic_types):
            raise ValueError(f"{label} contains unsupported atomic types")
    validation_coverage = set(inputs.validation.atomic_numbers.to(torch.int64).tolist())
    test_coverage = set(inputs.test.atomic_numbers.to(torch.int64).tolist())
    missing = sorted(test_coverage - validation_coverage)
    if missing:
        raise ValueError(f"test contains validation-uncovered elements: {missing}")


def _calibration_files(
    root: Path,
    inputs: E0CampaignInputs,
    direct: SvdSolution,
    aware: SvdSolution,
) -> None:
    calibration = root / "calibration"
    calibration.mkdir(parents=True)
    direct_path = calibration / "direct_mad_e0_test_informed.json"
    aware_path = calibration / "model_aware_reestimated_val_calibrated.json"
    atomic_write_json(
        direct_path,
        {
            "method": "direct_mad_e0_test_informed",
            "data_use": "test-informed/oracle",
            **_solution_payload(direct),
        },
    )
    atomic_write_json(
        aware_path,
        {
            "method": "model_aware_reestimated_val_calibrated",
            "data_use": "validation-only calibration",
            "fit": "unweighted total-energy OLS using SVD minimum-norm solution",
            **_solution_payload(aware),
        },
    )
    model_e0 = torch.as_tensor(inputs.model_e0, dtype=torch.float64).reshape(-1)
    val_counts = inputs.validation.composition.sum(dim=0).to(torch.int64)
    test_counts = inputs.test.composition.sum(dim=0).to(torch.int64)
    rows = []
    for index, atomic_number in enumerate(inputs.atomic_types):
        rows.append(
            {
                "atomic_number": atomic_number,
                "element": chemical_symbols[atomic_number],
                "checkpoint_e0_ev": float(model_e0[index]),
                "test_informed_mad_e0_ev": float(direct.solution[index]),
                "model_aware_delta_e0_ev": float(aware.solution[index]),
                "model_aware_reestimated_e0_ev": float(
                    model_e0[index] + aware.solution[index]
                ),
                "validation_atom_count": int(val_counts[index]),
                "test_atom_count": int(test_counts[index]),
            }
        )
    comparison_path = calibration / "element_e0_comparison.csv"
    _write_csv(comparison_path, rows)
    artifacts = {
        path.name: {"path": path.name, "sha256": sha256_file(path)}
        for path in (direct_path, aware_path, comparison_path)
    }
    atomic_write_json(
        calibration / "manifest.json",
        {
            "schema_version": E0_CAMPAIGN_SCHEMA_VERSION,
            "status": "complete",
            "kind": "calibration",
            "artifacts": artifacts,
        },
    )


def verify_e0_campaign(
    manifest_path: Path,
    *,
    full: bool = True,
) -> dict[str, Any]:
    """Verify campaign identity, internal artifacts, and referenced raw inputs."""

    path = Path(manifest_path).resolve()
    manifest = _mapping(path, context="E0 campaign manifest")
    payload = manifest.get("identity_payload")
    if (
        manifest.get("schema_version") != E0_CAMPAIGN_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(payload, Mapping)
        or manifest.get("identity") != _identity(payload)
    ):
        raise ValueError("E0 campaign schema/status/identity mismatch")
    variants = manifest.get("variants")
    if not isinstance(variants, Mapping) or set(variants) != set(E0_VARIANTS):
        raise ValueError("E0 campaign variants are incomplete")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("E0 campaign artifacts are missing")
    for name, descriptor in artifacts.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise ValueError("E0 campaign artifact descriptor is invalid")
        artifact = _confined(path.parent, descriptor.get("path"))
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError(f"E0 campaign artifact is missing or empty: {name}")
        expected = descriptor.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError(f"E0 campaign artifact SHA is invalid: {name}")
        if full and sha256_file(artifact) != expected:
            raise ValueError(f"E0 campaign artifact SHA mismatch: {name}")
    raw_inputs = payload.get("raw_inputs")
    if not isinstance(raw_inputs, Mapping):
        raise ValueError("E0 campaign raw input bindings are missing")
    if full:
        for split in ("validation", "test"):
            orders = raw_inputs.get(split)
            if not isinstance(orders, Mapping) or set(orders) != {
                str(value) for value in range(1, 9)
            }:
                raise ValueError(f"E0 campaign {split} raw inputs are incomplete")
            for order, binding in orders.items():
                if not isinstance(binding, Mapping):
                    raise ValueError(f"raw input binding is invalid: {split}/{order}")
                for name in ("manifest", "predictions"):
                    descriptor = binding.get(name)
                    if not isinstance(descriptor, Mapping):
                        raise ValueError(f"raw input descriptor is missing: {name}")
                    artifact = _external(path.parent, descriptor.get("path"))
                    expected = descriptor.get("sha256")
                    if not artifact.is_file() or sha256_file(artifact) != expected:
                        raise ValueError(
                            f"raw input SHA mismatch: {split}/{order}/{name}"
                        )
                verify_external_prediction(
                    _external(path.parent, binding["manifest"]["path"]),
                    full=True,
                )
    return manifest


def publish_e0_campaign(
    inputs: E0CampaignInputs,
    output_root: Path,
) -> Path:
    """Derive and atomically publish all three energy-only E0 variants."""

    _validate_input_shapes(inputs)
    validation_orders = _load_raw_set(
        inputs.validation_sources,
        inputs.validation,
        "validation",
    )
    test_orders = _load_raw_set(inputs.test_sources, inputs.test, "test")
    raw_validation = _tensor(
        validation_orders[1].predictions,
        "energy_prediction",
        dtype=torch.float64,
    ).reshape(-1)
    raw_test = _tensor(
        test_orders[1].predictions,
        "energy_prediction",
        dtype=torch.float64,
    ).reshape(-1)
    direct = fit_direct_mad_e0(
        inputs.test.composition,
        inputs.test.target_energy_r2scan,
        inputs.test.atomization_energy,
    )
    aware = fit_model_aware(
        inputs.validation.composition,
        inputs.validation.target_energy_r2scan,
        raw_validation,
    )
    model_e0 = torch.as_tensor(inputs.model_e0, dtype=torch.float64).reshape(-1)
    corrected = {
        "uncorrected": raw_test,
        "direct_mad_e0_test_informed": apply_direct_e0(
            raw_test,
            inputs.test.composition,
            model_e0,
            direct.solution,
        ),
        "model_aware_reestimated_val_calibrated": apply_model_aware_e0(
            raw_test,
            inputs.test.composition,
            aware.solution,
        ),
    }
    thresholds = _thresholds(test_orders[1])
    target = Path(output_root).resolve()
    validation_bindings = _raw_bindings(target, validation_orders)
    test_bindings = _raw_bindings(target, test_orders)
    identity_payload = {
        "algorithm": E0_CAMPAIGN_SCHEMA_VERSION,
        "checkpoint_sha256": inputs.checkpoint_sha256,
        "validation_sha256": inputs.validation_sha256,
        "test_sha256": inputs.test_sha256,
        "atomic_types": list(inputs.atomic_types),
        "raw_inputs": {
            "validation": validation_bindings,
            "test": test_bindings,
        },
    }
    identity = _identity(identity_payload)
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        existing = verify_e0_campaign(manifest_path, full=True)
        if existing.get("identity") != identity:
            raise ValueError("E0 campaign identity conflicts with existing output")
        return target
    if target.exists():
        raise ValueError("E0 campaign target exists without a complete identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".staging-{target.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    started_at = datetime.now(UTC).isoformat()
    try:
        _calibration_files(staging, inputs, direct, aware)
        calibration_by_variant = {
            "uncorrected": None,
            "direct_mad_e0_test_informed": (
                "calibration/direct_mad_e0_test_informed.json"
            ),
            "model_aware_reestimated_val_calibrated": (
                "calibration/model_aware_reestimated_val_calibrated.json"
            ),
        }
        variants: dict[str, Any] = {}
        for name in E0_VARIANTS:
            data = _variant_data(
                inputs.test,
                raw_test,
                corrected[name],
                thresholds,
            )
            _publish_variant(
                staging,
                name,
                data,
                test_orders,
                test_bindings,
                calibration_by_variant[name],
            )
            variants[name] = {
                "manifest": f"variants/{name}/manifest.json",
                "energy_data": f"variants/{name}/energy_data.pt",
                "metrics": f"variants/{name}/metrics.json",
            }
        manifest = {
            "schema_version": E0_CAMPAIGN_SCHEMA_VERSION,
            "status": "complete",
            "identity": identity,
            "identity_payload": identity_payload,
            "variants": variants,
            "artifacts": _all_artifact_descriptors(staging),
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        verify_e0_campaign(staging / "manifest.json", full=True)
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target
