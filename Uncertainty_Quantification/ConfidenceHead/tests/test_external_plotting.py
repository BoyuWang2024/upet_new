from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
from confidence_head.external_plotting import (
    DatasetSeries,
    plot_cross_dataset_correlations,
    plot_dataset_suite,
    verify_plot_publication,
)
from confidence_head.metrics import _correlation, _ranks
from confidence_head.plot_analysis import PlotSeries


def _series(
    root: Path,
    *,
    target: str,
    order: int | None,
    scale: float = 1.0,
) -> PlotSeries:
    observed = scale * torch.tensor([0.02, 0.08, 0.15, 0.24], dtype=torch.float64)
    expected = scale * torch.tensor([0.01, 0.10, 0.13, 0.30], dtype=torch.float64)
    logits = torch.full((4, 50), -4.0, dtype=torch.float64)
    logits[torch.arange(4), torch.tensor([0, 4, 8, 12])] = 4.0
    metrics = {
        "sample_count": 4,
        "pearson": _correlation(expected, observed),
        "spearman": _correlation(_ranks(expected), _ranks(observed)),
    }
    run = root / (f"energy-{order}" if target == "energy" else "force")
    return PlotSeries(
        run_dir=run,
        structure_ids=torch.tensor([10, 11, 12, 13]),
        target=target,
        order=order,
        logits=logits,
        observed=observed,
        expected=expected,
        representatives=torch.linspace(0.005, 0.495, 50, dtype=torch.float64),
        stored_metrics=metrics,
        force_target_mode="atom_mean" if target == "force" else None,
    )


def _dataset(root: Path, name: str, scale: float = 1.0) -> DatasetSeries:
    return DatasetSeries(
        name=name,
        force=_series(root, target="force", order=None, scale=scale),
        energy_by_order={
            order: _series(root, target="energy", order=order, scale=scale)
            for order in range(1, 9)
        },
        input_manifests={f"run-{index}": f"{index:064x}" for index in range(9)},
    )


def test_dataset_suite_writes_nine_png_pdf_csv_and_comparisons(
    tmp_path: Path,
) -> None:
    publication = plot_dataset_suite(
        _dataset(tmp_path / "runs", "mad_test"),
        tmp_path / "plots",
    )

    manifest = verify_plot_publication(publication.manifest, full=True)
    assert manifest["status"] == "complete"
    assert manifest["dataset"] == "mad_test"
    files = [path for path in publication.root.rglob("*") if path.is_file()]
    assert sum(path.suffix == ".png" for path in files) == 12
    assert sum(path.suffix == ".pdf" for path in files) == 12
    assert sum(path.suffix == ".csv" for path in files) == 10


def test_cross_dataset_correlations_cover_all_energy_orders_and_force(
    tmp_path: Path,
) -> None:
    publications = [
        plot_dataset_suite(
            _dataset(tmp_path / "runs", name, index + 1.0), tmp_path / "plots"
        )
        for index, name in enumerate(("matpes_test", "mad_test", "matpes_train"))
    ]

    manifest_path = plot_cross_dataset_correlations(
        publications,
        tmp_path / "plots" / "comparisons",
    )
    verify_plot_publication(manifest_path, full=True)
    energy_csv = manifest_path.parent / "cross_dataset_energy_correlations.csv"
    force_csv = manifest_path.parent / "cross_dataset_force_correlations.csv"
    with energy_csv.open(encoding="utf-8", newline="") as handle:
        energy_rows = list(csv.DictReader(handle))
    with force_csv.open(encoding="utf-8", newline="") as handle:
        force_rows = list(csv.DictReader(handle))

    assert {(row["dataset"], int(row["order"])) for row in energy_rows} == {
        (name, order)
        for name in ("matpes_test", "mad_test", "matpes_train")
        for order in range(1, 9)
    }
    assert {row["dataset"] for row in force_rows} == {
        "matpes_test",
        "mad_test",
        "matpes_train",
    }
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["kind"] == (
        "cross_dataset"
    )


def test_dataset_suite_reuses_identical_and_rejects_conflicting_inputs(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "runs", "mad_test")
    first = plot_dataset_suite(dataset, tmp_path / "plots")
    second = plot_dataset_suite(dataset, tmp_path / "plots")
    assert first.root == second.root

    changed = DatasetSeries(
        name=dataset.name,
        force=dataset.force,
        energy_by_order=dataset.energy_by_order,
        input_manifests={**dataset.input_manifests, "run-0": "f" * 64},
    )
    try:
        plot_dataset_suite(changed, tmp_path / "plots")
    except ValueError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("conflicting plot identity was accepted")
