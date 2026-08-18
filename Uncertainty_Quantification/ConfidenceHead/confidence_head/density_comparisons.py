"""Cross-dataset summaries for density publications."""

from __future__ import annotations

import csv
import os
import shutil
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from .artifacts import atomic_write_json, sha256_file
from .density_plotting import DensitySettings, analyze_density_panel
from .density_publication import (
    DENSITY_SCHEMA_VERSION,
    DensityPublication,
    _artifact_descriptors,
    _identity,
    verify_density_publication,
)
from .density_rendering import _save_figure


def _write_rows(
    path: Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _energy_figure(rows: Sequence[Mapping[str, Any]], root: Path, dpi: int) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)
    try:
        for dataset in sorted({str(row["dataset"]) for row in rows}):
            selected = sorted(
                (row for row in rows if row["dataset"] == dataset),
                key=lambda row: int(row["order"]),
            )
            orders = [int(row["order"]) for row in selected]
            axes[0].plot(
                orders,
                [row["pearson_log10"] for row in selected],
                marker="o",
                label=dataset,
            )
            axes[1].plot(
                orders,
                [row["spearman_log10"] for row in selected],
                marker="s",
                label=dataset,
            )
        for axis, title in zip(axes, ("Pearson r (log10)", "Spearman rho"), strict=True):
            axis.set_ylim(-1.0, 1.0)
            axis.set_xticks(range(1, 9))
            axis.set_xlabel("Energy cumulant order")
            axis.set_ylabel("Expected vs observed error correlation")
            axis.set_title(title)
            axis.grid(alpha=0.3)
            axis.legend()
        figure.tight_layout()
        _save_figure(figure, root / "cross_dataset_energy_correlations.png", dpi=dpi)
        _save_figure(figure, root / "cross_dataset_energy_correlations.pdf")
    finally:
        plt.close(figure)


def _force_figure(rows: Sequence[Mapping[str, Any]], root: Path, dpi: int) -> None:
    figure, axis = plt.subplots(figsize=(9, 6))
    try:
        positions = np.arange(len(rows), dtype=np.float64)
        width = 0.35
        axis.bar(
            positions - width / 2,
            [row["pearson_log10"] for row in rows],
            width,
            label="Pearson r (log10)",
        )
        axis.bar(
            positions + width / 2,
            [row["spearman_log10"] for row in rows],
            width,
            label="Spearman rho",
        )
        axis.set_xticks(positions, [str(row["dataset"]) for row in rows])
        axis.set_ylim(-1.0, 1.0)
        axis.set_ylabel("Expected vs observed error correlation")
        axis.set_title("UPET force confidence across datasets")
        axis.grid(axis="y", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        _save_figure(figure, root / "cross_dataset_force_correlations.png", dpi=dpi)
        _save_figure(figure, root / "cross_dataset_force_correlations.pdf")
    finally:
        plt.close(figure)


def publish_density_cross_dataset(
    publications: Sequence[DensityPublication],
    settings: DensitySettings,
    output_dir: Path,
) -> Path:
    """Publish log-space correlation summaries for exactly three datasets."""

    names = [publication.dataset.name for publication in publications]
    if len(publications) != 3 or len(set(names)) != 3:
        raise ValueError("density comparison requires exactly three datasets")
    for publication in publications:
        verify_density_publication(publication.manifest, full=True)
    payload = {
        "kind": "cross_dataset",
        "inputs": {
            publication.dataset.name: sha256_file(publication.manifest)
            for publication in publications
        },
        "settings": asdict(settings),
        "algorithm": DENSITY_SCHEMA_VERSION,
    }
    identity = _identity(payload)
    target = Path(output_dir).resolve()
    manifest_path = target / "manifest.json"
    if manifest_path.is_file():
        existing = verify_density_publication(manifest_path, full=True)
        if existing.get("identity") != identity:
            raise ValueError("density comparison identity conflicts with existing output")
        return manifest_path
    if target.exists():
        raise ValueError("density comparison target exists without a complete identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / f".staging-comparisons-{uuid.uuid4().hex}"
    staging.mkdir()
    started_at = datetime.now(UTC).isoformat()
    try:
        energy_rows: list[dict[str, Any]] = []
        force_rows: list[dict[str, Any]] = []
        for publication in publications:
            dataset = publication.dataset
            for order in range(1, 9):
                panel = analyze_density_panel(dataset.energy_by_order[order], settings)
                energy_rows.append(
                    {
                        "dataset": dataset.name,
                        "order": order,
                        "sample_count": panel.filtered.valid_count,
                        "pearson_log10": panel.pearson_log10,
                        "spearman_log10": panel.spearman_log10,
                    }
                )
            panel = analyze_density_panel(dataset.force, settings)
            force_rows.append(
                {
                    "dataset": dataset.name,
                    "sample_count": panel.filtered.valid_count,
                    "pearson_log10": panel.pearson_log10,
                    "spearman_log10": panel.spearman_log10,
                }
            )
        _write_rows(
            staging / "cross_dataset_energy_correlations.csv",
            ("dataset", "order", "sample_count", "pearson_log10", "spearman_log10"),
            energy_rows,
        )
        _write_rows(
            staging / "cross_dataset_force_correlations.csv",
            ("dataset", "sample_count", "pearson_log10", "spearman_log10"),
            force_rows,
        )
        _energy_figure(energy_rows, staging, settings.dpi)
        _force_figure(force_rows, staging, settings.dpi)
        manifest = {
            "schema_version": DENSITY_SCHEMA_VERSION,
            "status": "complete",
            "kind": "cross_dataset",
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
    return target / "manifest.json"
