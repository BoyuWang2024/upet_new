"""Matplotlib rendering for deterministic FGE single-dataset panels.

The visual language follows carnet's raw-STD plots while intentionally omitting
all checkpoint-count and cross-dataset comparison curves.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from .artifacts import atomic_write_json, sha256_file, sibling_staging
from .errors import HardFailure
from .plot_analysis import PanelAnalysis, PlotAnalysisResult


_ORANGE = "#f28e2b"
_STEMS = {
    "energy": "energy_uncertainty_vs_absolute_residual",
    "force": "force_uncertainty_vs_absolute_residual",
    "stress": "stress_uncertainty_vs_absolute_residual",
}
_LABELS = {
    "energy": "Energy / atom",
    "force": "Force component",
    "stress": "Stress component",
}


def _analysis_identity(result: PlotAnalysisResult) -> str:
    payload = {
        "dataset_label": result.plot_input.dataset_label,
        "source_kind": result.plot_input.source_kind,
        "source_identity": result.plot_input.source_identity,
        "domains": list(result.domains),
        "settings": asdict(result.settings),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _axis_style(axis: Any) -> None:
    axis.grid(True, which="major", color="#D9D9D9", linewidth=0.65, alpha=0.7)
    axis.grid(True, which="minor", color="#EEEEEE", linewidth=0.4, alpha=0.45)
    for spine in axis.spines.values():
        spine.set_color("#555555")


def build_single_panel_figure(
    panel: PanelAnalysis, result: PlotAnalysisResult
) -> Figure:
    """Build one independent uncertainty-versus-residual panel."""
    settings = result.settings
    figure, axis = plt.subplots(figsize=settings.figure_size)
    low, high = panel.log_limits
    bounds = np.asarray([10.0**low, 10.0**high], dtype=np.float64)
    axis.fill_between(
        bounds,
        bounds[0],
        bounds,
        color="#B8B8B8",
        alpha=0.22,
        label="residual <= uncertainty",
        zorder=0,
    )
    indices = panel.scatter_indices.numpy()
    axis.scatter(
        panel.uncertainty.numpy()[indices],
        panel.absolute_residual.numpy()[indices],
        s=settings.scatter_size,
        alpha=settings.scatter_alpha,
        color=_ORANGE,
        edgecolors="none",
        rasterized=True,
        label="samples",
        zorder=2,
    )
    density = panel.density
    axis.contour(
        np.power(10.0, density.x_centers),
        np.power(10.0, density.y_centers),
        density.grid.T,
        levels=density.contour_levels,
        colors=_ORANGE,
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
    axis.set_xlabel("Population STD uncertainty")
    axis.set_ylabel("Absolute residual")
    axis.set_title(f"{result.plot_input.dataset_label} / {_LABELS[panel.domain]}")
    spearman = "undefined" if panel.spearman is None else f"{panel.spearman:.4f}"
    pearson = (
        "undefined" if panel.pearson_log10 is None else f"{panel.pearson_log10:.4f}"
    )
    axis.text(
        0.04,
        0.96,
        "\n".join(
            (
                f"Spearman rho = {spearman}",
                f"Pearson r (log10) = {pearson}",
                f"valid = {panel.valid_count}/{panel.original_count}",
            )
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#BBBBBB", "alpha": 0.88},
    )
    axis.legend(loc="lower right", frameon=True, fontsize=8)
    _axis_style(axis)
    figure.tight_layout()
    return figure


def _save_panel(
    panel: PanelAnalysis,
    result: PlotAnalysisResult,
    directory: Path,
) -> tuple[Path, Path]:
    figure = build_single_panel_figure(panel, result)
    stem = directory / _STEMS[panel.domain]
    png = stem.with_suffix(".png")
    pdf = stem.with_suffix(".pdf")
    try:
        figure.savefig(
            png,
            format="png",
            dpi=result.settings.dpi,
            metadata={"Software": "UPET FGE plotting"},
        )
        figure.savefig(
            pdf,
            format="pdf",
            dpi=result.settings.dpi,
            metadata={
                "Creator": "UPET FGE plotting",
                "CreationDate": None,
                "ModDate": None,
            },
        )
    finally:
        plt.close(figure)
    return png, pdf


def _panel_statistics(panel: PanelAnalysis) -> dict[str, object]:
    return {
        "domain": panel.domain,
        "original_count": panel.original_count,
        "valid_count": panel.valid_count,
        "excluded": dict(panel.excluded),
        "spearman": panel.spearman,
        "pearson_log10": panel.pearson_log10,
        "log_limits": list(panel.log_limits),
        "scatter_count": panel.scatter_indices.numel(),
        "histogram_count": panel.density.histogram_count,
        "contour_levels": list(panel.density.contour_levels),
        "actual_contour_masses": list(panel.density.actual_contour_masses),
    }


def _expected_names(result: PlotAnalysisResult) -> set[str]:
    names = {
        f"{_STEMS[domain]}.{suffix}"
        for domain in result.domains
        for suffix in ("png", "pdf")
    }
    return names | {"plot_statistics.json", "plot_manifest.json"}


def _validate_existing(result: PlotAnalysisResult, destination: Path) -> None:
    if destination.is_symlink() or not destination.is_dir():
        raise HardFailure("plot publication conflict: destination is not a directory")
    actual = {path.name for path in destination.iterdir()}
    if actual != _expected_names(result):
        raise HardFailure("plot publication conflict: artifact inventory differs")
    try:
        manifest = json.loads(
            (destination / "plot_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, UnicodeError, ValueError) as exc:
        raise HardFailure("plot publication conflict: manifest is invalid") from exc
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema_version") != "upet.fge.plot-result.v1"
        or manifest.get("status") != "PASS"
        or manifest.get("analysis_identity") != _analysis_identity(result)
        or manifest.get("dataset_label") != result.plot_input.dataset_label
        or manifest.get("domains") != list(result.domains)
    ):
        raise HardFailure("plot publication conflict: identity differs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list) or len(outputs) != len(actual) - 1:
        raise HardFailure("plot publication conflict: output inventory differs")
    expected_outputs = actual - {"plot_manifest.json"}
    if {
        entry.get("path") for entry in outputs if isinstance(entry, dict)
    } != expected_outputs:
        raise HardFailure("plot publication conflict: output paths differ")
    for entry in outputs:
        if not isinstance(entry, dict) or set(entry) != {"path", "bytes", "sha256"}:
            raise HardFailure("plot publication conflict: output entry is invalid")
        path = destination / entry["path"]
        if (
            path.is_symlink()
            or not path.is_file()
            or entry["bytes"] != path.stat().st_size
            or entry["sha256"] != sha256_file(path)
        ):
            raise HardFailure(f"plot artifact hash differs: {path.name}")


def render_plot_suite(result: PlotAnalysisResult, output_root: str | Path) -> Path:
    """Publish one immutable per-dataset plot directory and validate reruns."""
    destination = Path(output_root).absolute()
    if destination.exists() or destination.is_symlink():
        _validate_existing(result, destination)
        return destination
    with sibling_staging(destination) as staging:
        images = [
            path
            for panel in result.panels
            for path in _save_panel(panel, result, staging)
        ]
        statistics = {
            "schema_version": "upet.fge.plot-statistics.v1",
            "dataset_label": result.plot_input.dataset_label,
            "source_kind": result.plot_input.source_kind,
            "source_identity": result.plot_input.source_identity,
            "formula": "population_std_unbiased_false",
            "panels": [_panel_statistics(panel) for panel in result.panels],
        }
        statistics_path = staging / "plot_statistics.json"
        atomic_write_json(statistics_path, statistics)
        outputs = [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted((*images, statistics_path), key=lambda item: item.name)
        ]
        atomic_write_json(
            staging / "plot_manifest.json",
            {
                "schema_version": "upet.fge.plot-result.v1",
                "status": "PASS",
                "analysis_identity": _analysis_identity(result),
                "dataset_label": result.plot_input.dataset_label,
                "source_kind": result.plot_input.source_kind,
                "source_identity": result.plot_input.source_identity,
                "domains": list(result.domains),
                "settings": asdict(result.settings),
                "outputs": outputs,
            },
        )
    _validate_existing(result, destination)
    return destination


__all__ = ["build_single_panel_figure", "render_plot_suite"]
