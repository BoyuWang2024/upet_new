from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head import plot_analysis


matplotlib.use("Agg")


def _series(target: plot_analysis.Target = "energy") -> plot_analysis.PlotSeries:
    logits = torch.full((4, 50), -10.0, dtype=torch.float64)
    logits[torch.arange(4), torch.tensor([0, 1, 1, 49])] = 10.0
    return plot_analysis.PlotSeries(
        run_dir=Path("/synthetic/run"),
        structure_ids=torch.tensor([11, 12, 13, 14]),
        target=target,
        order=1 if target == "energy" else None,
        logits=logits,
        observed=torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64),
        expected=torch.tensor([2.0, 1.0, 4.0, 3.0], dtype=torch.float64),
        representatives=torch.linspace(0.003, 0.297, 50, dtype=torch.float64),
        stored_metrics={"sample_count": 4, "pearson": 0.6, "spearman": 0.6},
        force_target_mode="atom_mean" if target == "force" else None,
    )


def test_write_bin_csv_keeps_all_bins_and_empty_values(tmp_path: Path) -> None:
    path = tmp_path / "statistics.csv"

    result = plot_analysis.write_bin_csv(
        path,
        plot_analysis.bin_rows(_series()),
    )

    assert result == path
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 50
    assert rows[0]["sample_count"] == "1"
    assert rows[2]["sample_count"] == "0"
    assert rows[2]["mean_observed_error"] == ""


def test_plot_single_boxplot_writes_valid_png_pdf_and_csv(tmp_path: Path) -> None:
    output_dir = tmp_path / "plots"

    artifacts = plot_analysis.plot_single_boxplot(_series(), output_dir)

    assert {path.name for path in artifacts} == {
        "test_energy_argmax_bin_boxplot.png",
        "test_energy_argmax_bin_boxplot.pdf",
        "test_energy_argmax_bin_statistics.csv",
    }
    assert all(path.stat().st_size > 0 for path in artifacts)
    assert mpimg.imread(output_dir / "test_energy_argmax_bin_boxplot.png").ndim == 3
    assert (
        (output_dir / "test_energy_argmax_bin_boxplot.pdf")
        .read_bytes()
        .startswith(b"%PDF")
    )


def test_plot_combined_energy_boxplots_uses_orders_one_to_eight(
    tmp_path: Path,
) -> None:
    series = _series()
    series_by_order = {order: replace(series, order=order) for order in range(1, 9)}

    path = plot_analysis.plot_combined_energy_boxplots(
        series_by_order,
        tmp_path / "comparisons",
    )

    assert path.name == "combined_energy_argmax_bin_boxplots.pdf"
    assert path.read_bytes().startswith(b"%PDF")


def test_plot_combined_force_boxplot_is_separate_from_energy(tmp_path: Path) -> None:
    path = plot_analysis.plot_combined_force_boxplot(
        _series("force"),
        tmp_path / "comparisons",
    )

    assert path.name == "combined_force_argmax_bin_boxplots.pdf"
    assert path.read_bytes().startswith(b"%PDF")


def test_plot_energy_correlations_writes_ordered_csv_png_and_pdf(
    tmp_path: Path,
) -> None:
    rows = tuple(
        plot_analysis.CorrelationRow(
            order=order,
            sample_count=4,
            pearson=0.1 * order,
            spearman=0.08 * order,
        )
        for order in range(1, 9)
    )

    artifacts = plot_analysis.plot_energy_correlations(rows, tmp_path / "correlations")

    assert {path.name for path in artifacts} == {
        "linear_order_correlations_no_ci.csv",
        "linear_order_correlations_no_ci.png",
        "linear_order_correlations_no_ci.pdf",
    }
    csv_path = tmp_path / "correlations" / "linear_order_correlations_no_ci.csv"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [int(row["order"]) for row in csv_rows] == list(range(1, 9))
    assert (
        mpimg.imread(
            tmp_path / "correlations" / "linear_order_correlations_no_ci.png"
        ).ndim
        == 3
    )
