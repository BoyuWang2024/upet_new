from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from confidence_head.density_plotting import (
    DensitySettings,
    build_density_figure,
    build_energy_comparison_figure,
    render_density_panel,
    render_energy_comparison,
)
from confidence_head.plot_analysis import PlotSeries


def _series(root: Path, *, target: str, order: int | None) -> PlotSeries:
    observed = torch.logspace(-4, -1, 128, dtype=torch.float64)
    expected = observed.pow(1.08) * (1.0 + 0.02 * (order or 1))
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


def test_density_figure_is_square_and_log_scaled(tmp_path: Path) -> None:
    figure = build_density_figure(
        _series(tmp_path, target="energy", order=3), DensitySettings(grid_size=32)
    )
    try:
        axis = figure.axes[0]
        assert axis.get_xscale() == "log"
        assert axis.get_yscale() == "log"
        assert axis.get_aspect() == 1.0
    finally:
        plt.close(figure)


def test_render_density_panel_writes_png_pdf_csv_and_json(tmp_path: Path) -> None:
    artifacts = render_density_panel(
        _series(tmp_path, target="force", order=None),
        DensitySettings(grid_size=32, dpi=72),
        tmp_path / "plots",
    )

    assert {path.suffix for path in artifacts} == {".png", ".pdf", ".csv", ".json"}
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts)
    analysis = json.loads((tmp_path / "plots" / "density_analysis.json").read_text())
    assert analysis["target"] == "force"
    assert analysis["valid_count"] == 128
    assert set(analysis["excluded"]) == {"nan", "inf", "negative", "zero"}
    assert len(analysis["actual_contour_masses"]) == 5
    with (tmp_path / "plots" / "density_statistics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert int(rows[0]["valid_count"]) == 128


def test_energy_comparison_uses_shared_limits_and_writes_png_pdf(
    tmp_path: Path,
) -> None:
    series = {
        order: _series(tmp_path, target="energy", order=order) for order in range(1, 9)
    }
    settings = DensitySettings(grid_size=32, dpi=72)
    figure = build_energy_comparison_figure(series, settings)
    try:
        assert len(figure.axes) == 8
        x_limits = {tuple(axis.get_xlim()) for axis in figure.axes}
        y_limits = {tuple(axis.get_ylim()) for axis in figure.axes}
        assert len(x_limits) == 1
        assert len(y_limits) == 1
        assert x_limits == y_limits
    finally:
        plt.close(figure)

    artifacts = render_energy_comparison(series, settings, tmp_path / "comparison")
    assert {path.suffix for path in artifacts} == {".png", ".pdf"}
    assert all(path.stat().st_size > 0 for path in artifacts)
