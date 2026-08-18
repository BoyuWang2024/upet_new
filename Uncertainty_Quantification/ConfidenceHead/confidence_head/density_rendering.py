"""FGE-style rendering for ConfidenceHead density analyses."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np


matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .density_plotting import (
    DensityPanel,
    DensitySettings,
    analyze_density_panel,
    shared_log_limits,
)
from .plot_analysis import PlotSeries


_ORANGE = "#f28e2b"
_DARK_ORANGE = "#c96a13"


def _style_axis(axis: Any) -> None:
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.65, alpha=0.7)
    axis.grid(True, which="minor", color="#EEEEEE", linewidth=0.4, alpha=0.45)
    for spine in axis.spines.values():
        spine.set_color("#555555")


def _target_label(series: PlotSeries) -> str:
    if series.target == "energy":
        return "absolute energy error per atom (eV/atom)"
    return "mean absolute force-component error per atom (eV/A)"


def _target_title(series: PlotSeries) -> str:
    if series.target == "energy":
        return f"UPET / Energy / order {series.order}"
    return "UPET / Force / atom mean"


def _draw_density_axis(axis: Any, panel: DensityPanel) -> None:
    low, high = panel.log_limits
    bounds = np.asarray([10.0**low, 10.0**high], dtype=np.float64)
    axis.fill_between(
        bounds,
        bounds[0],
        bounds,
        color="#B8B8B8",
        alpha=0.22,
        label="observed error <= expected error",
        zorder=0,
    )
    indices = panel.scatter_indices.detach().cpu().numpy()
    axis.scatter(
        panel.filtered.expected.detach().cpu().numpy()[indices],
        panel.filtered.observed.detach().cpu().numpy()[indices],
        s=2.0,
        alpha=0.035,
        color=_ORANGE,
        edgecolors="none",
        rasterized=True,
        label="samples",
        zorder=2,
    )
    density = panel.density
    if density.contour_levels:
        axis.contour(
            np.power(10.0, density.x_centers),
            np.power(10.0, density.y_centers),
            density.grid.T,
            levels=density.contour_levels,
            colors=_DARK_ORANGE,
            linewidths=0.9,
            alpha=0.9,
            zorder=3,
        )
    axis.plot(
        bounds,
        bounds,
        color="#222222",
        linewidth=1.15,
        linestyle="--",
        label="y = x",
        zorder=4,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(bounds)
    axis.set_ylim(bounds)
    axis.set_aspect("equal", adjustable="box")
    label = _target_label(panel.series)
    axis.set_xlabel(f"Expected {label}")
    axis.set_ylabel(f"Observed {label}")
    axis.set_title(_target_title(panel.series))
    axis.text(
        0.04,
        0.96,
        "\n".join(
            (
                f"Spearman rho = {panel.spearman_log10:.4f}",
                f"Pearson r (log10) = {panel.pearson_log10:.4f}",
                f"valid={panel.filtered.valid_count}/{panel.filtered.original_count}",
            )
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
    )
    axis.legend(loc="lower right", frameon=True, fontsize=8)
    _style_axis(axis)


def _build_density_figure_from_panel(panel: DensityPanel) -> Figure:
    figure, axis = plt.subplots(figsize=(7.0, 7.0))
    _draw_density_axis(axis, panel)
    figure.tight_layout()
    return figure


def build_density_figure(
    series: PlotSeries,
    settings: DensitySettings,
    *,
    log_limits: tuple[float, float] | None = None,
) -> Figure:
    """Build one expected-versus-observed density figure."""

    return _build_density_figure_from_panel(
        analyze_density_panel(series, settings, log_limits=log_limits)
    )


@contextmanager
def _atomic_output(path: Path) -> Iterator[Path]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=target.parent,
        prefix=f".{target.stem}.",
        suffix=target.suffix,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        yield temporary
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise ValueError(f"density output is empty: {target}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _save_figure(figure: Figure, path: Path, *, dpi: int | None = None) -> Path:
    target = Path(path)
    with _atomic_output(target) as temporary:
        figure.savefig(
            temporary,
            format=target.suffix.removeprefix("."),
            dpi=dpi,
            bbox_inches="tight",
            metadata={"Creator": "UPET ConfidenceHead density plotting"},
        )
    return target


def analysis_payload(panel: DensityPanel, settings: DensitySettings) -> dict[str, Any]:
    """Return the JSON-safe statistics bound to one rendered panel."""

    return {
        "target": panel.series.target,
        "order": panel.series.order,
        "original_count": panel.filtered.original_count,
        "valid_count": panel.filtered.valid_count,
        "excluded": dict(panel.filtered.excluded),
        "spearman_log10": panel.spearman_log10,
        "pearson_log10": panel.pearson_log10,
        "log_limits": list(panel.log_limits),
        "histogram_count": panel.density.histogram_count,
        "contour_levels": list(panel.density.contour_levels),
        "actual_contour_masses": list(panel.density.actual_contour_masses),
        "settings": {
            "scatter_max_points": settings.scatter_max_points,
            "scatter_seed": settings.scatter_seed,
            "grid_size": settings.grid_size,
            "gaussian_sigma": settings.gaussian_sigma,
            "contour_masses": list(settings.contour_masses),
            "log_margin": settings.log_margin,
            "dpi": settings.dpi,
        },
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    encoded = (
        json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    with _atomic_output(target) as temporary:
        temporary.write_bytes(encoded)
    return target


def _write_statistics_csv(path: Path, payload: Mapping[str, Any]) -> Path:
    excluded = payload["excluded"]
    if not isinstance(excluded, Mapping):
        raise ValueError("density exclusion statistics must be a mapping")
    row = {
        "target": payload["target"],
        "order": "" if payload["order"] is None else payload["order"],
        "original_count": payload["original_count"],
        "valid_count": payload["valid_count"],
        "excluded_nan": excluded["nan"],
        "excluded_inf": excluded["inf"],
        "excluded_negative": excluded["negative"],
        "excluded_zero": excluded["zero"],
        "spearman_log10": payload["spearman_log10"],
        "pearson_log10": payload["pearson_log10"],
        "histogram_count": payload["histogram_count"],
    }
    target = Path(path)
    with _atomic_output(target) as temporary:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(row))
            writer.writeheader()
            writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
    return target


def render_density_panel(
    series: PlotSeries,
    settings: DensitySettings,
    output_dir: Path,
    *,
    log_limits: tuple[float, float] | None = None,
) -> tuple[Path, Path, Path, Path]:
    """Render one series and write its audited analysis artifacts."""

    panel = analyze_density_panel(series, settings, log_limits=log_limits)
    figure = _build_density_figure_from_panel(panel)
    root = Path(output_dir)
    stem = root / f"test_{series.target}_expected_vs_observed_density"
    try:
        png_path = _save_figure(figure, stem.with_suffix(".png"), dpi=settings.dpi)
        pdf_path = _save_figure(figure, stem.with_suffix(".pdf"))
    finally:
        plt.close(figure)
    payload = analysis_payload(panel, settings)
    csv_path = _write_statistics_csv(root / "density_statistics.csv", payload)
    json_path = _write_json(root / "density_analysis.json", payload)
    return png_path, pdf_path, csv_path, json_path


def _energy_series_by_order(
    series_by_order: Mapping[int, PlotSeries],
) -> list[PlotSeries]:
    if set(series_by_order) != set(range(1, 9)):
        raise ValueError("energy density comparison requires orders 1 through 8")
    ordered = [series_by_order[order] for order in range(1, 9)]
    if any(
        item.target != "energy" or item.order != order
        for order, item in enumerate(ordered, 1)
    ):
        raise ValueError("energy density comparison contains invalid order metadata")
    return ordered


def build_energy_comparison_figure(
    series_by_order: Mapping[int, PlotSeries],
    settings: DensitySettings,
) -> Figure:
    """Build the 4x2 energy-order comparison with one shared square range."""

    ordered = _energy_series_by_order(series_by_order)
    limits = shared_log_limits(ordered, margin=settings.log_margin)
    figure, axes = plt.subplots(4, 2, figsize=(14.0, 25.0))
    for series, axis in zip(ordered, axes.reshape(-1), strict=True):
        _draw_density_axis(
            axis, analyze_density_panel(series, settings, log_limits=limits)
        )
    figure.suptitle("UPET energy expected error versus observed error", fontsize=16)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.985))
    return figure


def render_energy_comparison(
    series_by_order: Mapping[int, PlotSeries],
    settings: DensitySettings,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Render the dataset-local eight-order energy comparison."""

    figure = build_energy_comparison_figure(series_by_order, settings)
    stem = Path(output_dir) / "combined_energy_expected_vs_observed_density"
    try:
        return (
            _save_figure(figure, stem.with_suffix(".png"), dpi=settings.dpi),
            _save_figure(figure, stem.with_suffix(".pdf")),
        )
    finally:
        plt.close(figure)
