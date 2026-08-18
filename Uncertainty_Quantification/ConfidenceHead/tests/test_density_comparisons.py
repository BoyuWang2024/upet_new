from __future__ import annotations

import csv
from pathlib import Path

import torch
from confidence_head.density_comparisons import publish_density_cross_dataset
from confidence_head.density_plotting import DensitySettings
from confidence_head.density_publication import (
    publish_density_dataset,
    verify_density_publication,
)
from confidence_head.external_plotting import DatasetSeries
from confidence_head.plot_analysis import PlotSeries


def _series(root: Path, *, target: str, order: int | None, scale: float) -> PlotSeries:
    observed = torch.logspace(-4, -1, 64, dtype=torch.float64) * scale
    expected = observed.pow(1.0 + 0.002 * (order or 1)) * 1.05
    logits = torch.zeros((len(observed), 50), dtype=torch.float64)
    logits[:, 0] = 1.0
    return PlotSeries(
        run_dir=root / (f"energy-{order}" if target == "energy" else "force"),
        structure_ids=torch.arange(len(observed)),
        target=target,  # type: ignore[arg-type]
        order=order,
        logits=logits,
        observed=observed,
        expected=expected,
        representatives=torch.linspace(0.005, 0.495, 50, dtype=torch.float64),
        stored_metrics={"sample_count": len(observed), "pearson": 0.9, "spearman": 0.9},
        force_target_mode="atom_mean" if target == "force" else None,
    )


def _dataset(root: Path, name: str, scale: float) -> DatasetSeries:
    return DatasetSeries(
        name=name,
        force=_series(root, target="force", order=None, scale=scale),
        energy_by_order={
            order: _series(root, target="energy", order=order, scale=scale)
            for order in range(1, 9)
        },
        input_manifests={f"run-{index}": f"{index:064x}" for index in range(9)},
    )


def test_cross_dataset_density_correlations_cover_three_datasets(
    tmp_path: Path,
) -> None:
    settings = DensitySettings(grid_size=32, dpi=72)
    publications = [
        publish_density_dataset(
            _dataset(tmp_path / "runs" / name, name, scale),
            tmp_path / "density_scatter",
            settings,
        )
        for name, scale in (
            ("matpes_train", 1.0),
            ("matpes_test", 2.0),
            ("mad_test", 3.0),
        )
    ]

    manifest_path = publish_density_cross_dataset(
        publications, settings, tmp_path / "density_scatter" / "comparisons"
    )

    verify_density_publication(manifest_path, full=True)
    with (manifest_path.parent / "cross_dataset_energy_correlations.csv").open(
        newline=""
    ) as handle:
        energy_rows = list(csv.DictReader(handle))
    with (manifest_path.parent / "cross_dataset_force_correlations.csv").open(
        newline=""
    ) as handle:
        force_rows = list(csv.DictReader(handle))
    assert len(energy_rows) == 24
    assert len(force_rows) == 3
    assert {row["dataset"] for row in force_rows} == {
        "matpes_train",
        "matpes_test",
        "mad_test",
    }
