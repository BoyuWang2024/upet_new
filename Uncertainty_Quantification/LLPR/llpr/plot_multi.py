"""Multi-evaluation publication layered on the existing UPET plot helpers."""

from __future__ import annotations

import csv
import os
import re
import shutil
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import yaml
from pydantic import Field, model_validator
from scipy.ndimage import gaussian_filter

from .artifacts import atomic_json_dump, load_verified_manifest, sha256_file
from .config import StrictModel, resolve_repo_path


class PlotEvaluationConfig(StrictModel):
    label: str
    run_root: Path
    evaluation_identity: str | None = None

    @model_validator(mode="after")
    def validate_contract(self) -> "PlotEvaluationConfig":
        if re.fullmatch(r"[a-z0-9][a-z0-9_]*", self.label) is None:
            raise ValueError("label must use lowercase letters, digits, or underscores")
        root = (
            self.run_root.resolve()
            if self.run_root.is_absolute()
            else resolve_repo_path(self.run_root)
        )
        object.__setattr__(self, "run_root", root)
        return self


class PlotStyleConfig(StrictModel):
    grid_size: int = Field(default=160, ge=4)
    gaussian_sigma: float = Field(default=1.2, gt=0)
    contour_masses: tuple[float, ...] = (0.5, 0.7, 0.85, 0.95, 0.99)
    scatter_max_points: int = Field(default=20_000, ge=1)
    random_seed: int = 20260714
    log_margin: float = Field(default=0.05, gt=0)
    figure_size: tuple[float, float] = (7.0, 7.0)
    color: str = "#f28e2b"
    scatter_size: float = Field(default=12.0, gt=0)
    scatter_alpha: float = Field(default=0.04, gt=0, le=1)
    title_font_size: float = Field(default=26.0, gt=0)
    axis_label_font_size: float = Field(default=22.0, gt=0)
    tick_font_size: float = Field(default=18.0, gt=0)
    annotation_font_size: float = Field(default=16.0, gt=0)
    line_width: float = Field(default=3.4, gt=0)
    spine_width: float = Field(default=1.5, gt=0)
    dpi: int = Field(default=300, ge=1)
    formats: tuple[Literal["png", "pdf"], ...] = ("png", "pdf")

    @model_validator(mode="after")
    def validate_contract(self) -> "PlotStyleConfig":
        masses = self.contour_masses
        if not masses or any(value <= 0 or value >= 1 for value in masses):
            raise ValueError("contour masses must be strictly between 0 and 1")
        if any(left >= right for left, right in zip(masses, masses[1:], strict=False)):
            raise ValueError("contour masses must be strictly increasing")
        if self.formats != ("png", "pdf"):
            raise ValueError("plot formats must be exactly png and pdf")
        return self


class PlotConfig(StrictModel):
    evaluations: tuple[PlotEvaluationConfig, ...] = Field(min_length=1)
    output_root: Path
    style: PlotStyleConfig = Field(default_factory=PlotStyleConfig)

    @model_validator(mode="after")
    def validate_contract(self) -> "PlotConfig":
        labels = [item.label for item in self.evaluations]
        if len(labels) != len(set(labels)):
            raise ValueError("evaluation labels must be unique")
        output = (
            self.output_root.resolve()
            if self.output_root.is_absolute()
            else resolve_repo_path(self.output_root)
        )
        for item in self.evaluations:
            if output == item.run_root or item.run_root in output.parents:
                raise ValueError("output_root must be outside every run_root")
        object.__setattr__(self, "output_root", output)
        return self


def load_plot_config(path: Path) -> PlotConfig:
    loaded = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError("plot configuration must contain a YAML mapping")
    return PlotConfig.model_validate(loaded)


@dataclass(frozen=True)
class FilteredPairs:
    uncertainty: np.ndarray
    absolute_error: np.ndarray
    uncertainty_log10: np.ndarray
    absolute_error_log10: np.ndarray
    original_count: int
    valid_count: int
    dropped_count: int
    excluded: dict[str, int]


@dataclass(frozen=True)
class DensityContours:
    x_centers: np.ndarray
    y_centers: np.ndarray
    density: np.ndarray
    levels: np.ndarray


@dataclass(frozen=True)
class PanelAnalysis:
    filtered: FilteredPairs
    density: DensityContours
    scatter_indices: np.ndarray
    statistics: dict[str, object]


@dataclass(frozen=True)
class _EvaluationData:
    identity: str
    details_sha256: str
    panels: dict[str, tuple[np.ndarray, np.ndarray]]


def filter_log_pairs(
    uncertainty: np.ndarray, absolute_error: np.ndarray
) -> FilteredPairs:
    """Extend UPET's existing log filter with categorized exclusions."""
    from .plotting import filter_log_pairs as basic_filter_log_pairs

    basic = basic_filter_log_pairs(uncertainty, absolute_error)
    uncertainty_array = np.asarray(uncertainty, dtype=np.float64).reshape(-1)
    error_array = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    nan = np.isnan(uncertainty_array) | np.isnan(error_array)
    infinite = (~nan) & (np.isinf(uncertainty_array) | np.isinf(error_array))
    remaining = ~(nan | infinite)
    zero = remaining & ((uncertainty_array == 0) | (error_array == 0))
    negative = remaining & (~zero) & ((uncertainty_array < 0) | (error_array < 0))
    valid_count = len(basic.uncertainty)
    return FilteredPairs(
        uncertainty=basic.uncertainty,
        absolute_error=basic.absolute_error,
        uncertainty_log10=np.log10(basic.uncertainty),
        absolute_error_log10=np.log10(basic.absolute_error),
        original_count=basic.original_count,
        valid_count=valid_count,
        dropped_count=basic.dropped_count,
        excluded={
            "nan": int(nan.sum()),
            "inf": int(infinite.sum()),
            "zero": int(zero.sum()),
            "negative": int(negative.sum()),
        },
    )


def shared_square_log_limits(
    pairs: Iterable[tuple[np.ndarray, np.ndarray]], margin: float
) -> tuple[float, float]:
    if margin <= 0:
        raise ValueError("log margin must be positive")
    minima: list[float] = []
    maxima: list[float] = []
    for uncertainty, error in pairs:
        filtered = filter_log_pairs(uncertainty, error)
        if filtered.valid_count == 0:
            raise ValueError("shared log limits require positive finite pairs")
        minima.extend(
            [filtered.uncertainty_log10.min(), filtered.absolute_error_log10.min()]
        )
        maxima.extend(
            [filtered.uncertainty_log10.max(), filtered.absolute_error_log10.max()]
        )
    if not minima:
        raise ValueError("shared log limits require at least one panel")
    low, high = float(min(minima)), float(max(maxima))
    padding = max((high - low) * margin, margin)
    return low - padding, high + padding


def _density(
    filtered: FilteredPairs,
    limits: tuple[float, float],
    grid_size: int,
    sigma: float,
    masses: tuple[float, ...],
) -> DensityContours:
    histogram, x_edges, y_edges = np.histogram2d(
        filtered.uncertainty_log10,
        filtered.absolute_error_log10,
        bins=grid_size,
        range=(limits, limits),
    )
    if histogram.sum() < 2:
        raise ValueError("density histogram requires at least two points")
    density = gaussian_filter(histogram, sigma=sigma, mode="nearest")
    density /= density.sum()
    descending = np.sort(density.reshape(-1))[::-1]
    cumulative = np.cumsum(descending)
    levels = np.unique(
        [
            descending[
                min(
                    int(np.searchsorted(cumulative, mass, side="left")),
                    descending.size - 1,
                )
            ]
            for mass in masses
        ]
    )
    levels = levels[levels > 0]
    if levels.size == 0:
        raise ValueError("density did not produce positive contour levels")
    return DensityContours(
        x_centers=(x_edges[:-1] + x_edges[1:]) * 0.5,
        y_centers=(y_edges[:-1] + y_edges[1:]) * 0.5,
        density=density,
        levels=levels,
    )


def analyze_panel(
    uncertainty: np.ndarray,
    absolute_error: np.ndarray,
    *,
    log_limits: tuple[float, float],
    grid_size: int,
    gaussian_sigma: float,
    contour_masses: tuple[float, ...],
    scatter_max_points: int,
    random_seed: int,
    coverage_thresholds: tuple[float, ...] = (),
) -> PanelAnalysis:
    """Add density analysis to UPET's existing panel statistics."""
    from .plotting import _sample_indices
    from .plotting import analyze_panel as basic_analyze_panel

    basic = basic_analyze_panel(uncertainty, absolute_error)
    filtered = filter_log_pairs(uncertainty, absolute_error)
    if filtered.valid_count < 2:
        raise ValueError("panel requires at least two positive finite pairs")
    statistics: dict[str, object] = {
        "original_count": basic.count + basic.dropped_count,
        "valid_count": basic.count,
        "metric_valid_count": basic.count,
        "excluded": filtered.excluded,
        "pearson_log": basic.pearson_log10,
        "spearman_log": basic.spearman_log10,
        "uncertainty": _summary(filtered.uncertainty),
        "absolute_error": _summary(filtered.absolute_error),
        "ratio_uncertainty_to_error": _summary(
            filtered.uncertainty / filtered.absolute_error
        ),
    }
    for threshold in coverage_thresholds:
        statistics[f"coverage_{threshold:g}"] = float(
            np.mean(filtered.absolute_error <= threshold * filtered.uncertainty)
        )
    return PanelAnalysis(
        filtered=filtered,
        density=_density(
            filtered, log_limits, grid_size, gaussian_sigma, contour_masses
        ),
        scatter_indices=_sample_indices(
            filtered.valid_count, scatter_max_points, random_seed
        ),
        statistics=statistics,
    )


def _summary(values: np.ndarray) -> dict[str, float]:
    q05, q25, median, q75, q95 = np.quantile(values, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "mean": float(np.mean(values)),
        "median": float(median),
        "q05": float(q05),
        "q25": float(q25),
        "q75": float(q75),
        "q95": float(q95),
    }


def _load_evaluation(config: PlotEvaluationConfig) -> _EvaluationData:
    candidates: list[tuple[Path, dict[str, object]]] = []
    for path in sorted((config.run_root / "evaluation").glob("*/*/manifest.json")):
        manifest = load_verified_manifest(path, verify_npz=True)
        identity = str(manifest.get("identity", ""))
        if config.evaluation_identity is None or identity == config.evaluation_identity:
            candidates.append((path.parent, manifest))
    if len(candidates) != 1:
        raise ValueError(
            f"{config.label}: expected exactly one complete evaluation, "
            f"found {len(candidates)}"
        )
    directory, manifest = candidates[0]
    with np.load(directory / "details.npz", allow_pickle=False) as archive:
        panels = {
            "energy": (
                archive["energy_calibrated_std"].copy(),
                np.abs(archive["energy_residual"].copy()),
            ),
            "force": (
                archive["force_calibrated_std_component"].copy(),
                np.abs(archive["force_residual"].copy()),
            ),
        }
    files = manifest.get("files")
    if not isinstance(files, dict) or not isinstance(files.get("details.npz"), str):
        raise ValueError(f"{config.label}: manifest does not bind details.npz")
    return _EvaluationData(
        identity=str(manifest["identity"]),
        details_sha256=str(files["details.npz"]),
        panels=panels,
    )


def _render(
    path: Path,
    analysis: PanelAnalysis,
    limits: tuple[float, float],
    label: str,
    target: str,
    style: PlotStyleConfig,
) -> None:
    import matplotlib.pyplot as plt

    units = "eV/atom" if target == "energy" else "eV/Å"
    figure, axis = plt.subplots(figsize=style.figure_size, layout="constrained")
    try:
        low_log, high_log = limits
        low, high = 10.0**low_log, 10.0**high_log
        diagonal = np.logspace(low_log, high_log, 256)
        axis.fill_between(diagonal, low, diagonal, color="0.88", zorder=0)
        indices = analysis.scatter_indices
        axis.scatter(
            analysis.filtered.uncertainty[indices],
            analysis.filtered.absolute_error[indices],
            s=style.scatter_size,
            alpha=style.scatter_alpha,
            color=style.color,
            edgecolors="none",
            rasterized=True,
        )
        axis.contour(
            10.0**analysis.density.x_centers,
            10.0**analysis.density.y_centers,
            analysis.density.density.T,
            levels=analysis.density.levels,
            colors=style.color,
            linewidths=style.line_width,
        )
        axis.plot(diagonal, diagonal, "--", color="black", linewidth=style.line_width)
        axis.set(xscale="log", yscale="log", xlim=(low, high), ylim=(low, high))
        axis.set_box_aspect(1)
        axis.set_title(
            f"{label.replace('_', ' ').title()} / {target.title()}",
            fontsize=style.title_font_size,
        )
        axis.set_xlabel(f"Uncertainty ({units})", fontsize=style.axis_label_font_size)
        axis.set_ylabel(
            f"Absolute residual ({units})", fontsize=style.axis_label_font_size
        )
        axis.tick_params(labelsize=style.tick_font_size)
        axis.grid(False)
        spearman = analysis.statistics["spearman_log"]
        pearson = analysis.statistics["pearson_log"]
        axis.text(
            0.04,
            0.96,
            (
                f"Spearman rho={float(spearman):.3f}\n"
                f"log10 Pearson rho={float(pearson):.3f}"
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=style.annotation_font_size,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.76},
        )
        for output_format in style.formats:
            figure.savefig(
                path.with_suffix(f".{output_format}"),
                dpi=style.dpi,
                bbox_inches="tight",
            )
    finally:
        plt.close(figure)


def _row(
    label: str,
    target: str,
    analysis: PanelAnalysis,
    limits: tuple[float, float],
) -> dict[str, object]:
    statistics = analysis.statistics
    row: dict[str, object] = {
        "dataset": label,
        "target": target,
        "log_limit_low": limits[0],
        "log_limit_high": limits[1],
        "original_count": statistics["original_count"],
        "log_valid_count": statistics["valid_count"],
        "metric_valid_count": statistics["metric_valid_count"],
        "pearson_log10": statistics["pearson_log"],
        "spearman_log10": statistics["spearman_log"],
    }
    excluded = statistics["excluded"]
    if not isinstance(excluded, Mapping):
        raise TypeError("excluded statistics must be a mapping")
    for name, count in excluded.items():
        row[f"excluded_{name}"] = count
    for group in ("uncertainty", "absolute_error", "ratio_uncertainty_to_error"):
        values = statistics[group]
        if not isinstance(values, Mapping):
            raise TypeError(f"{group} statistics must be a mapping")
        for name, value in values.items():
            row[f"{group}_{name}"] = value
    return row


def _promote(staging: Path, output: Path) -> None:
    backup = output.with_name(f".{output.name}.{uuid.uuid4().hex}.backup")
    existed = output.exists()
    if existed:
        os.replace(output, backup)
    try:
        os.replace(staging, output)
    except BaseException:
        if existed and backup.exists() and not output.exists():
            os.replace(backup, output)
        raise
    if existed:
        shutil.rmtree(backup)


def run_plot(config: PlotConfig) -> Path:
    import matplotlib

    matplotlib.use("Agg", force=True)
    data = {item.label: _load_evaluation(item) for item in config.evaluations}
    limits = {
        target: shared_square_log_limits(
            (item.panels[target] for item in data.values()), config.style.log_margin
        )
        for target in ("energy", "force")
    }
    analyses = {
        (label, target): analyze_panel(
            *item.panels[target],
            log_limits=limits[target],
            grid_size=config.style.grid_size,
            gaussian_sigma=config.style.gaussian_sigma,
            contour_masses=config.style.contour_masses,
            scatter_max_points=config.style.scatter_max_points,
            random_seed=config.style.random_seed,
        )
        for label, item in data.items()
        for target in ("energy", "force")
    }
    output = config.output_root
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir()
    try:
        rows = []
        for label in data:
            for target in ("energy", "force"):
                analysis = analyses[(label, target)]
                _render(
                    staging / f"llpr_{label}_{target}_uncertainty_vs_residual",
                    analysis,
                    limits[target],
                    label,
                    target,
                    config.style,
                )
                rows.append(_row(label, target, analysis, limits[target]))
        with (staging / "plotting_statistics.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        files = {
            path.name: sha256_file(path)
            for path in sorted(staging.iterdir())
            if path.is_file()
        }
        atomic_json_dump(
            staging / "plotting_manifest.json",
            {
                "status": "complete",
                "evaluation_identities": {
                    label: item.identity for label, item in data.items()
                },
                "input_details_sha256": {
                    label: item.details_sha256 for label, item in data.items()
                },
                "shared_log_limits": limits,
                "style": config.style.model_dump(mode="json"),
                "files": files,
            },
        )
        if len(tuple(staging.iterdir())) != 4 * len(data) + 2:
            raise RuntimeError("plot publication contains an unexpected file count")
        _promote(staging, output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return output
