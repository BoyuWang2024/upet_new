"""Deterministic expected-error versus observed-error density plots."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from torch import Tensor

from .metrics import _correlation, _ranks
from .plot_analysis import PlotSeries


@dataclass(frozen=True)
class DensitySettings:
    """Validated settings shared by density analysis and rendering."""

    scatter_max_points: int = 20_000
    scatter_seed: int = 20260714
    grid_size: int = 160
    gaussian_sigma: float = 1.2
    contour_masses: tuple[float, ...] = (0.50, 0.70, 0.85, 0.95, 0.99)
    log_margin: float = 0.05
    dpi: int = 300

    def __post_init__(self) -> None:
        if self.scatter_max_points < 1:
            raise ValueError("scatter_max_points must be positive")
        if self.grid_size < 8:
            raise ValueError("grid_size must be at least 8")
        if self.gaussian_sigma <= 0.0:
            raise ValueError("gaussian_sigma must be positive")
        if self.log_margin <= 0.0:
            raise ValueError("log_margin must be positive")
        if self.dpi < 1:
            raise ValueError("dpi must be positive")
        masses = self.contour_masses
        if not masses or any(not 0.0 < value < 1.0 for value in masses):
            raise ValueError("contour_masses must be inside (0, 1)")
        if any(left >= right for left, right in zip(masses, masses[1:], strict=False)):
            raise ValueError("contour_masses must be strictly increasing")


@dataclass(frozen=True)
class FilteredDensityPairs:
    expected: Tensor
    observed: Tensor
    expected_log10: Tensor
    observed_log10: Tensor
    original_count: int
    valid_count: int
    excluded: dict[str, int]


@dataclass(frozen=True)
class DensityContours:
    x_centers: np.ndarray
    y_centers: np.ndarray
    grid: np.ndarray
    contour_levels: tuple[float, ...]
    actual_contour_masses: tuple[float, ...]
    histogram_count: int


@dataclass(frozen=True)
class DensityPanel:
    series: PlotSeries
    filtered: FilteredDensityPairs
    spearman_log10: float
    pearson_log10: float
    scatter_indices: Tensor
    density: DensityContours
    log_limits: tuple[float, float]


def filter_density_pairs(expected: Tensor, observed: Tensor) -> FilteredDensityPairs:
    """Flatten paired errors and classify values excluded from log-space plots."""

    if not torch.is_tensor(expected) or not torch.is_tensor(observed):
        raise ValueError("expected and observed errors must be tensors")
    if tuple(expected.shape) != tuple(observed.shape):
        raise ValueError("expected and observed errors must have matching shapes")
    expected = expected.detach().cpu().to(torch.float64).reshape(-1)
    observed = observed.detach().cpu().to(torch.float64).reshape(-1)
    if expected.numel() == 0:
        raise ValueError("expected and observed errors must not be empty")

    nan_mask = torch.isnan(expected) | torch.isnan(observed)
    inf_mask = (~nan_mask) & (torch.isinf(expected) | torch.isinf(observed))
    finite_mask = ~(nan_mask | inf_mask)
    zero_mask = finite_mask & ((expected == 0.0) | (observed == 0.0))
    negative_mask = finite_mask & (~zero_mask) & ((expected < 0.0) | (observed < 0.0))
    valid_mask = finite_mask & (~zero_mask) & (~negative_mask)
    valid_expected = expected[valid_mask]
    valid_observed = observed[valid_mask]
    return FilteredDensityPairs(
        expected=valid_expected,
        observed=valid_observed,
        expected_log10=torch.log10(valid_expected),
        observed_log10=torch.log10(valid_observed),
        original_count=int(expected.numel()),
        valid_count=int(valid_expected.numel()),
        excluded={
            "nan": int(nan_mask.sum().item()),
            "inf": int(inf_mask.sum().item()),
            "negative": int(negative_mask.sum().item()),
            "zero": int(zero_mask.sum().item()),
        },
    )


def _sample_indices(count: int, maximum: int, seed: int) -> Tensor:
    if count <= maximum:
        return torch.arange(count, dtype=torch.long)
    generator = np.random.default_rng(seed)
    selected = np.sort(generator.choice(count, size=maximum, replace=False))
    return torch.from_numpy(selected.astype(np.int64, copy=False))


def _limits_from_logs(
    expected_log10: Tensor,
    observed_log10: Tensor,
    margin: float,
) -> tuple[float, float]:
    if expected_log10.numel() < 2:
        raise ValueError("density analysis requires at least two valid positive pairs")
    low = min(float(expected_log10.min()), float(observed_log10.min()))
    high = max(float(expected_log10.max()), float(observed_log10.max()))
    padding = max((high - low) * margin, margin)
    return low - padding, high + padding


def shared_log_limits(
    series: Sequence[PlotSeries],
    *,
    margin: float,
) -> tuple[float, float]:
    """Return one square log range containing all valid pairs in a series set."""

    if not series:
        raise ValueError("shared log limits require at least one series")
    filtered = [filter_density_pairs(item.expected, item.observed) for item in series]
    expected = torch.cat([item.expected_log10 for item in filtered])
    observed = torch.cat([item.observed_log10 for item in filtered])
    return _limits_from_logs(expected, observed, margin)


def _density_contours(
    filtered: FilteredDensityPairs,
    *,
    log_limits: tuple[float, float],
    settings: DensitySettings,
) -> DensityContours:
    low, high = log_limits
    if not math.isfinite(low) or not math.isfinite(high) or low >= high:
        raise ValueError("log limits must be finite and increasing")
    histogram, x_edges, y_edges = np.histogram2d(
        filtered.expected_log10.numpy(),
        filtered.observed_log10.numpy(),
        bins=settings.grid_size,
        range=((low, high), (low, high)),
    )
    histogram_count = int(histogram.sum())
    if histogram_count < 2:
        raise ValueError("density histogram requires at least two in-range points")
    density = gaussian_filter(
        histogram,
        sigma=settings.gaussian_sigma,
        mode="nearest",
    )
    total = float(density.sum())
    if not math.isfinite(total) or total <= 0.0:
        raise ValueError("smoothed density has no finite positive mass")
    density = density / total
    descending = np.sort(density.reshape(-1))[::-1]
    cumulative = np.cumsum(descending)
    thresholds = []
    for mass in settings.contour_masses:
        index = min(
            int(np.searchsorted(cumulative, mass, side="left")),
            descending.size - 1,
        )
        threshold = float(descending[index])
        if threshold > 0.0:
            thresholds.append(threshold)
    levels = tuple(float(value) for value in np.unique(thresholds))
    return DensityContours(
        x_centers=(x_edges[:-1] + x_edges[1:]) * 0.5,
        y_centers=(y_edges[:-1] + y_edges[1:]) * 0.5,
        grid=density,
        contour_levels=levels,
        actual_contour_masses=tuple(
            float(density[density >= level].sum()) for level in levels
        ),
        histogram_count=histogram_count,
    )


def _finite_correlation(name: str, value: float) -> float:
    if not math.isfinite(value) or value < -1.0 - 1e-12 or value > 1.0 + 1e-12:
        raise ValueError(f"{name} correlation is undefined")
    return float(max(-1.0, min(1.0, value)))


def analyze_density_panel(
    series: PlotSeries,
    settings: DensitySettings,
    *,
    log_limits: tuple[float, float] | None = None,
) -> DensityPanel:
    """Analyze one verified ConfidenceHead series in log10 error space."""

    if series.target == "energy":
        if series.order not in range(1, 9):
            raise ValueError("energy density plots require order 1 through 8")
    elif series.force_target_mode != "atom_mean":
        raise ValueError("force density plots require atom_mean semantics")
    filtered = filter_density_pairs(series.expected, series.observed)
    limits = (
        _limits_from_logs(
            filtered.expected_log10,
            filtered.observed_log10,
            settings.log_margin,
        )
        if log_limits is None
        else log_limits
    )
    if filtered.valid_count < 2:
        raise ValueError("density analysis requires at least two valid positive pairs")
    pearson = _finite_correlation(
        "pearson_log10",
        _correlation(filtered.expected_log10, filtered.observed_log10),
    )
    spearman = _finite_correlation(
        "spearman_log10",
        _correlation(
            _ranks(filtered.expected_log10),
            _ranks(filtered.observed_log10),
        ),
    )
    return DensityPanel(
        series=series,
        filtered=filtered,
        spearman_log10=spearman,
        pearson_log10=pearson,
        scatter_indices=_sample_indices(
            filtered.valid_count,
            settings.scatter_max_points,
            settings.scatter_seed,
        ),
        density=_density_contours(
            filtered,
            log_limits=limits,
            settings=settings,
        ),
        log_limits=limits,
    )


# Public rendering API is imported after the analysis types to avoid a cycle.
from .density_rendering import (  # noqa: E402, F401
    build_density_figure,
    build_energy_comparison_figure,
    render_density_panel,
    render_energy_comparison,
)
