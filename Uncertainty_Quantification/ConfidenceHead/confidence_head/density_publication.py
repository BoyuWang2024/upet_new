"""Atomic publication and verification for ConfidenceHead density plots."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .artifacts import atomic_write_json, sha256_file
from .density_plotting import (
    DensitySettings,
    analyze_density_panel,
    shared_log_limits,
)
from .density_rendering import (
    analysis_payload,
    render_density_panel,
    render_energy_comparison,
)
from .external_plotting import DatasetSeries
from .plot_analysis import PlotSeries


DENSITY_SCHEMA_VERSION = "upet_confidence_density_plots_v1"


@dataclass(frozen=True)
class DensityPublication:
    dataset: DatasetSeries
    root: Path
    manifest: Path


def _identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return f"density-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _confined(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("density artifact path must be a relative string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("density artifact path must be relative")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("density artifact escapes publication root")
    return path


def verify_density_publication(
    manifest_path: Path,
    *,
    full: bool = True,
) -> dict[str, Any]:
    """Verify schema, identity, artifact confinement, and optional SHA256."""

    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid density manifest {path}: {error}") from error
    payload = manifest.get("identity_payload") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != DENSITY_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(payload, Mapping)
        or manifest.get("identity") != _identity(payload)
    ):
        raise ValueError("density manifest schema/status/identity mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("density manifest artifacts are missing")
    for name, descriptor in artifacts.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise ValueError("density artifact descriptor is invalid")
        artifact = _confined(path.parent, descriptor.get("path"))
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError(f"density artifact is missing or empty: {name}")
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"density artifact SHA is invalid: {name}")
        if full and sha256_file(artifact) != digest:
            raise ValueError(f"density artifact SHA mismatch: {name}")
    return manifest


def _ordered_energy(dataset: DatasetSeries) -> list[PlotSeries]:
    if set(dataset.energy_by_order) != set(range(1, 9)):
        raise ValueError("density publication requires energy orders 1 through 8")
    values = [dataset.energy_by_order[order] for order in range(1, 9)]
    if any(
        value.target != "energy" or value.order != order
        for order, value in enumerate(values, 1)
    ):
        raise ValueError("density publication energy metadata is inconsistent")
    if (
        dataset.force.target != "force"
        or dataset.force.force_target_mode != "atom_mean"
    ):
        raise ValueError("density publication requires one atom_mean force series")
    run_names = [value.run_dir.name for value in (*values, dataset.force)]
    if len(set(run_names)) != 9:
        raise ValueError("density publication run names must be unique")
    return values


def _artifact_descriptors(root: Path) -> dict[str, dict[str, str]]:
    return {
        path.relative_to(root).as_posix(): {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def _write_energy_summary(
    path: Path,
    ordered: Sequence[PlotSeries],
    settings: DensitySettings,
    limits: tuple[float, float],
) -> tuple[Path, Path]:
    payloads = [
        analysis_payload(
            analyze_density_panel(series, settings, log_limits=limits), settings
        )
        for series in ordered
    ]
    csv_path = path / "energy_density_summary.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "order",
        "original_count",
        "valid_count",
        "spearman_log10",
        "pearson_log10",
        "histogram_count",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({name: item[name] for name in fields} for item in payloads)
        handle.flush()
        os.fsync(handle.fileno())
    json_path = path / "energy_density_summary.json"
    atomic_write_json(
        json_path,
        {"shared_log_limits": list(limits), "orders": payloads},
    )
    return csv_path, json_path


def _identity_payload(
    dataset: DatasetSeries,
    settings: DensitySettings,
) -> dict[str, Any]:
    return {
        "kind": "dataset",
        "dataset": dataset.name,
        "inputs": dict(sorted(dataset.input_manifests.items())),
        "targets": {
            "energy_orders": list(range(1, 9)),
            "force_target_mode": "atom_mean",
        },
        "settings": asdict(settings),
        "algorithm": DENSITY_SCHEMA_VERSION,
    }


def publish_density_dataset(
    dataset: DatasetSeries,
    output_root: Path,
    settings: DensitySettings,
) -> DensityPublication:
    """Atomically publish one dataset's nine density analyses and comparison."""

    ordered = _ordered_energy(dataset)
    energy_limits = shared_log_limits(ordered, margin=settings.log_margin)
    analyze_density_panel(dataset.force, settings)
    payload = _identity_payload(dataset, settings)
    identity = _identity(payload)
    parent = Path(output_root).resolve()
    target = parent / dataset.name
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        existing = verify_density_publication(manifest_path, full=True)
        if existing.get("identity") != identity:
            raise ValueError("density plot identity conflicts with existing output")
        return DensityPublication(dataset, target, manifest_path)
    if target.exists():
        raise ValueError("density plot target exists without a complete identity")

    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".staging-{dataset.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    started_at = datetime.now(UTC).isoformat()
    try:
        for series in ordered:
            render_density_panel(
                series,
                settings,
                staging / "runs" / series.run_dir.name,
                log_limits=energy_limits,
            )
        render_density_panel(
            dataset.force,
            settings,
            staging / "runs" / dataset.force.run_dir.name,
        )
        comparison = staging / "comparisons"
        render_energy_comparison(dataset.energy_by_order, settings, comparison)
        _write_energy_summary(comparison, ordered, settings, energy_limits)
        manifest = {
            "schema_version": DENSITY_SCHEMA_VERSION,
            "status": "complete",
            "kind": "dataset",
            "dataset": dataset.name,
            "identity": identity,
            "identity_payload": payload,
            "artifacts": _artifact_descriptors(staging),
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        verify_density_publication(staging / "manifest.json", full=True)
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return DensityPublication(dataset, target, target / "manifest.json")
