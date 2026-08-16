"""Three-dataset ConfidenceHead plots built from verified prediction artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import matplotlib
import torch


matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from .artifacts import atomic_write_json, load_verified_torch, sha256_file
from .external_prediction import verify_external_prediction
from .metrics import _correlation, _ranks
from .plot_analysis import (
    PlotSeries,
    _active_target,
    _draw_boxplot,
    _mapping,
    _save_figure,
    _tensor,
    energy_series_by_order,
    load_plot_series,
    plot_energy_correlations,
    plot_single_boxplot,
    validate_comparable_energy,
)
from .workflows.verify import verify_run


PLOT_SCHEMA_VERSION = "upet_confidence_external_plots_v1"
_NUM_BINS = 50
_ATOM_MEAN = "atom_mean"
_ATOM_MEAN_ERROR = "abs_cartesian_component_mean_v1"


@dataclass(frozen=True)
class DatasetSeries:
    """Nine verified series belonging to one named dataset."""

    name: str
    force: PlotSeries
    energy_by_order: Mapping[int, PlotSeries]
    input_manifests: Mapping[str, str]


@dataclass(frozen=True)
class PlotPublication:
    """One atomically published dataset plot suite."""

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
    return f"plots-{hashlib.sha256(encoded).hexdigest()[:16]}"


def _confined(root: Path, relative: object) -> Path:
    if not isinstance(relative, str):
        raise ValueError("plot artifact path must be a relative string")
    candidate = Path(relative)
    if candidate.is_absolute():
        raise ValueError("plot artifact path must be relative")
    path = (root / candidate).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("plot artifact escapes publication root")
    return path


def verify_plot_publication(
    manifest_path: Path,
    *,
    full: bool = True,
) -> dict[str, Any]:
    """Verify a dataset or cross-dataset plot manifest."""

    path = Path(manifest_path).resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid plot manifest {path}: {error}") from error
    payload = manifest.get("identity_payload") if isinstance(manifest, dict) else None
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != PLOT_SCHEMA_VERSION
        or manifest.get("status") != "complete"
        or not isinstance(payload, Mapping)
        or manifest.get("identity") != _identity(payload)
    ):
        raise ValueError("plot manifest schema/status/identity mismatch")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("plot manifest artifacts are missing")
    for name, descriptor in artifacts.items():
        if not isinstance(name, str) or not isinstance(descriptor, Mapping):
            raise ValueError("plot artifact descriptor is invalid")
        artifact = _confined(path.parent, descriptor.get("path"))
        if not artifact.is_file() or artifact.stat().st_size <= 0:
            raise ValueError(f"plot artifact is missing or empty: {name}")
        digest = descriptor.get("sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"plot artifact SHA is invalid: {name}")
        if full and sha256_file(artifact) != digest:
            raise ValueError(f"plot artifact SHA mismatch: {name}")
    return manifest


def _external_series(source: Path) -> PlotSeries:
    run = source.parent.parent
    verify_run(run, full=True)
    external_manifest = verify_external_prediction(source / "manifest.json", full=True)
    config = _mapping(run / "resolved_config.yaml", yaml_file=True)
    bins = _mapping(run / "binning.json")
    metrics = _mapping(source / "metrics.json")
    descriptor = external_manifest["artifacts"]["predictions.pt"]
    loaded = load_verified_torch(
        source / str(descriptor["path"]),
        expected_sha256=str(descriptor["sha256"]),
        weights_only=True,
    )
    if not isinstance(loaded, Mapping):
        raise ValueError("external predictions must contain a mapping")
    predictions = cast(Mapping[str, Any], loaded)
    target = _active_target(config)
    branch_config = cast(Mapping[str, Any], config["model"])[target]
    branch_bins = bins.get(target)
    branch_metrics = metrics.get(target)
    if (
        not isinstance(branch_config, Mapping)
        or not isinstance(branch_bins, Mapping)
        or not isinstance(branch_metrics, Mapping)
    ):
        raise ValueError("external plot branch metadata is incomplete")
    if branch_bins.get("algorithm") != "fixed_linear_v1":
        raise ValueError("external plots require fixed_linear_v1 bins")
    if int(branch_config.get("num_bins", 0)) != _NUM_BINS:
        raise ValueError("external plots require exactly 50 model bins")
    if int(branch_bins.get("num_bins", 0)) != _NUM_BINS:
        raise ValueError("external plots require exactly 50 declared bins")
    logits = _tensor(predictions, f"{target}_logits", dtype=torch.float64)
    observed = _tensor(
        predictions, f"{target}_observed_errors", dtype=torch.float64
    ).reshape(-1)
    expected = _tensor(
        predictions, f"{target}_expected_errors", dtype=torch.float64
    ).reshape(-1)
    representatives = _tensor(
        predictions, f"{target}_representatives", dtype=torch.float64
    ).reshape(-1)
    structure_ids = _tensor(predictions, "structure_ids").reshape(-1)
    if logits.shape != (len(observed), _NUM_BINS):
        raise ValueError(f"external {target} logits must have shape [N, 50]")
    if expected.shape != observed.shape or representatives.shape != (_NUM_BINS,):
        raise ValueError(f"external {target} error/representative shapes mismatch")
    if not all(
        bool(torch.isfinite(value).all()) for value in (logits, observed, expected)
    ):
        raise ValueError(f"external {target} plot values must be finite")
    if bool(torch.any(observed < 0) or torch.any(expected < 0)):
        raise ValueError(f"external {target} plot errors must be non-negative")
    if int(branch_metrics.get("sample_count", -1)) != len(observed):
        raise ValueError(f"external {target} metric sample_count mismatch")
    declared = torch.as_tensor(
        branch_bins.get("representatives"), dtype=torch.float64
    ).reshape(-1)
    if declared.shape != representatives.shape or not torch.allclose(
        declared, representatives, rtol=1e-6, atol=1e-8
    ):
        raise ValueError(f"external {target} representatives mismatch")
    order: int | None = None
    force_mode: str | None = None
    if target == "energy":
        order = int(branch_config.get("cumulant_order", 0))
        if order not in range(1, 9) or len(structure_ids) != len(observed):
            raise ValueError("external energy order or structure count is invalid")
    else:
        force_mode = str(predictions.get("force_target_mode"))
        if (
            branch_config.get("target_mode"),
            branch_bins.get("target_mode"),
            force_mode,
            branch_bins.get("error_definition"),
            predictions.get("force_error_definition"),
        ) != (
            _ATOM_MEAN,
            _ATOM_MEAN,
            _ATOM_MEAN,
            _ATOM_MEAN_ERROR,
            _ATOM_MEAN_ERROR,
        ):
            raise ValueError("external force plots require atom_mean semantics")
    return PlotSeries(
        run_dir=run,
        structure_ids=structure_ids,
        target=target,
        order=order,
        logits=logits,
        observed=observed,
        expected=expected,
        representatives=representatives,
        stored_metrics=cast(Mapping[str, int | float], branch_metrics),
        force_target_mode=force_mode,
    )


def load_dataset_series(
    sources: Sequence[Path],
    dataset_name: str,
) -> DatasetSeries:
    """Load nine evaluation/external sources into one comparable dataset."""

    if len(sources) != 9:
        raise ValueError("dataset plotting requires exactly nine sources")
    series: list[PlotSeries] = []
    inputs: dict[str, str] = {}
    for raw in sources:
        source = Path(raw).resolve()
        if source.name == "evaluation":
            manifest = source / "manifest.json"
            value = load_plot_series(source.parent)
        elif source.parent.name == "predictions":
            manifest = source / "manifest.json"
            value = _external_series(source)
        else:
            raise ValueError(f"unsupported plot source directory: {source}")
        digest = sha256_file(manifest)
        inputs[value.run_dir.name] = digest
        series.append(value)
    force = [value for value in series if value.target == "force"]
    if len(force) != 1:
        raise ValueError("dataset plotting requires exactly one force series")
    energies = energy_series_by_order(
        [value for value in series if value.target == "energy"]
    )
    validate_comparable_energy(energies)
    return DatasetSeries(dataset_name, force[0], energies, inputs)


def _set_shared_nonnegative_scale(
    axes: Sequence[Any], series: Sequence[PlotSeries]
) -> None:
    maximum = max(float(value.observed.max()) for value in series)
    upper = maximum * 1.08 if maximum > 0 else 1e-4
    for axis in axes:
        axis.set_ylim(0.0, upper)


def _combined_energy(
    series_by_order: Mapping[int, PlotSeries], root: Path
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(4, 2, figsize=(24, 26))
    flat = list(axes.reshape(-1))
    try:
        values = [series_by_order[order] for order in range(1, 9)]
        for order, axis, series in zip(range(1, 9), flat, values, strict=True):
            _draw_boxplot(
                axis,
                series,
                title=f"Energy confidence order {order}",
                tick_fontsize=4,
            )
        _set_shared_nonnegative_scale(flat, values)
        figure.suptitle("UPET energy argmax-bin error distributions", fontsize=16)
        figure.tight_layout(rect=(0, 0, 1, 0.985))
        return (
            _save_figure(
                figure, root / "combined_energy_argmax_bin_boxplots.png", dpi=300
            ),
            _save_figure(figure, root / "combined_energy_argmax_bin_boxplots.pdf"),
        )
    finally:
        plt.close(figure)


def _combined_force(series: PlotSeries, root: Path) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(20, 7))
    try:
        _draw_boxplot(
            axis,
            series,
            title="Force confidence (per-atom Cartesian-component mean)",
            tick_fontsize=6,
        )
        _set_shared_nonnegative_scale([axis], [series])
        figure.tight_layout()
        return (
            _save_figure(
                figure, root / "combined_force_argmax_bin_boxplots.png", dpi=300
            ),
            _save_figure(figure, root / "combined_force_argmax_bin_boxplots.pdf"),
        )
    finally:
        plt.close(figure)


def _artifact_descriptors(root: Path) -> dict[str, dict[str, str]]:
    return {
        path.relative_to(root).as_posix(): {
            "path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "manifest.json"
    }


def plot_dataset_suite(dataset: DatasetSeries, output_root: Path) -> PlotPublication:
    """Atomically publish nine single plots plus within-dataset comparisons."""

    validate_comparable_energy(dataset.energy_by_order)
    if dataset.force.target != "force" or dataset.force.force_target_mode != _ATOM_MEAN:
        raise ValueError("dataset force series must use atom_mean semantics")
    payload = {
        "kind": "dataset",
        "dataset": dataset.name,
        "inputs": dict(dataset.input_manifests),
    }
    identity = _identity(payload)
    parent = Path(output_root).resolve()
    target = parent / dataset.name
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        existing = verify_plot_publication(manifest_path, full=True)
        if existing.get("identity") != identity:
            raise ValueError("dataset plot identity conflicts with existing output")
        return PlotPublication(dataset, target, manifest_path)
    if target.exists():
        raise ValueError("dataset plot target exists without a complete identity")
    parent.mkdir(parents=True, exist_ok=True)
    staging = parent / f".staging-{dataset.name}-{uuid.uuid4().hex}"
    staging.mkdir()
    started_at = datetime.now(UTC).isoformat()
    try:
        for order in range(1, 9):
            series = dataset.energy_by_order[order]
            plot_single_boxplot(series, staging / "runs" / series.run_dir.name)
        plot_single_boxplot(
            dataset.force,
            staging / "runs" / dataset.force.run_dir.name,
        )
        comparison = staging / "comparisons"
        _combined_energy(dataset.energy_by_order, comparison / "argmax_bin_boxplots")
        _combined_force(dataset.force, comparison / "argmax_bin_boxplots")
        correlations = validate_comparable_energy(dataset.energy_by_order)
        plot_energy_correlations(correlations, comparison / "energy_correlations")
        manifest = {
            "schema_version": PLOT_SCHEMA_VERSION,
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
        verify_plot_publication(staging / "manifest.json", full=True)
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return PlotPublication(dataset, target, target / "manifest.json")


def _force_correlation(series: PlotSeries) -> tuple[int, float, float]:
    if series.target != "force" or len(series.observed) < 2:
        raise ValueError("force correlation requires at least two force samples")
    if series.expected.unique().numel() < 2 or series.observed.unique().numel() < 2:
        raise ValueError("force correlation is undefined for constant arrays")
    pearson = _correlation(series.expected, series.observed)
    spearman = _correlation(_ranks(series.expected), _ranks(series.observed))
    if not math.isfinite(pearson) or not math.isfinite(spearman):
        raise ValueError("force correlations must be finite")
    return len(series.observed), pearson, spearman


def _write_rows(
    path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())


def _cross_energy_plot(
    rows: Sequence[Mapping[str, Any]], root: Path
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    try:
        for dataset in sorted({str(row["dataset"]) for row in rows}):
            selected = sorted(
                (row for row in rows if row["dataset"] == dataset),
                key=lambda row: int(row["order"]),
            )
            orders = [int(row["order"]) for row in selected]
            axes[0].plot(
                orders, [row["pearson"] for row in selected], marker="o", label=dataset
            )
            axes[1].plot(
                orders, [row["spearman"] for row in selected], marker="s", label=dataset
            )
        for axis, title in zip(axes, ("Pearson", "Spearman"), strict=True):
            axis.set_ylim(-1.0, 1.0)
            axis.set_xticks(range(1, 9))
            axis.set_xlabel("Energy cumulant order")
            axis.set_ylabel("Correlation of expected and observed error")
            axis.set_title(title)
            axis.grid(alpha=0.3)
            axis.legend()
        figure.tight_layout()
        return (
            _save_figure(
                figure, root / "cross_dataset_energy_correlations.png", dpi=300
            ),
            _save_figure(figure, root / "cross_dataset_energy_correlations.pdf"),
        )
    finally:
        plt.close(figure)


def _cross_force_plot(
    rows: Sequence[Mapping[str, Any]], root: Path
) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(9, 6))
    try:
        datasets = [str(row["dataset"]) for row in rows]
        positions = torch.arange(len(rows), dtype=torch.float64).numpy()
        width = 0.35
        axis.bar(
            positions - width / 2,
            [row["pearson"] for row in rows],
            width,
            label="Pearson",
        )
        axis.bar(
            positions + width / 2,
            [row["spearman"] for row in rows],
            width,
            label="Spearman",
        )
        axis.set_xticks(positions, datasets)
        axis.set_ylim(-1.0, 1.0)
        axis.set_ylabel("Correlation of expected and observed error")
        axis.set_title("UPET force confidence across datasets")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        return (
            _save_figure(
                figure, root / "cross_dataset_force_correlations.png", dpi=300
            ),
            _save_figure(figure, root / "cross_dataset_force_correlations.pdf"),
        )
    finally:
        plt.close(figure)


def plot_cross_dataset_correlations(
    publications: Sequence[PlotPublication],
    output_dir: Path,
) -> Path:
    """Publish energy-order and force correlations for exactly three datasets."""

    if len(publications) != 3 or len({item.dataset.name for item in publications}) != 3:
        raise ValueError("cross-dataset plotting requires exactly three datasets")
    payload = {
        "kind": "cross_dataset",
        "inputs": {
            item.dataset.name: sha256_file(item.manifest) for item in publications
        },
    }
    identity = _identity(payload)
    target = Path(output_dir).resolve()
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        existing = verify_plot_publication(manifest_path, full=True)
        if existing.get("identity") != identity:
            raise ValueError(
                "cross-dataset plot identity conflicts with existing output"
            )
        return manifest_path
    if target.exists():
        raise ValueError("cross-dataset plot target exists without a complete identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".staging-comparisons-{uuid.uuid4().hex}"
    staging.mkdir()
    started_at = datetime.now(UTC).isoformat()
    try:
        energy_rows: list[dict[str, Any]] = []
        force_rows: list[dict[str, Any]] = []
        for publication in publications:
            dataset = publication.dataset
            for row in validate_comparable_energy(dataset.energy_by_order):
                energy_rows.append({"dataset": dataset.name, **asdict(row)})
            count, pearson, spearman = _force_correlation(dataset.force)
            force_rows.append(
                {
                    "dataset": dataset.name,
                    "target": "force",
                    "order": "",
                    "sample_count": count,
                    "pearson": pearson,
                    "spearman": spearman,
                }
            )
        energy_path = staging / "cross_dataset_energy_correlations.csv"
        force_path = staging / "cross_dataset_force_correlations.csv"
        _write_rows(
            energy_path,
            ("dataset", "order", "sample_count", "pearson", "spearman"),
            energy_rows,
        )
        _write_rows(
            force_path,
            ("dataset", "target", "order", "sample_count", "pearson", "spearman"),
            force_rows,
        )
        _cross_energy_plot(energy_rows, staging)
        _cross_force_plot(force_rows, staging)
        manifest = {
            "schema_version": PLOT_SCHEMA_VERSION,
            "status": "complete",
            "kind": "cross_dataset",
            "identity": identity,
            "identity_payload": payload,
            "artifacts": _artifact_descriptors(staging),
            "started_at": started_at,
            "completed_at": datetime.now(UTC).isoformat(),
        }
        atomic_write_json(staging / "manifest.json", manifest)
        verify_plot_publication(staging / "manifest.json", full=True)
        os.replace(staging, target)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return target / "manifest.json"
