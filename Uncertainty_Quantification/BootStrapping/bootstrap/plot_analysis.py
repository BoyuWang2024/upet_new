"""Deterministic Carnet-style analysis for Bootstrap campaign panels."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from scipy.stats import pearsonr, spearmanr

from .campaign import CampaignPlotStyle
from .errors import HardFailure
from .plot_source import PanelKey, PlotSource, load_panel_arrays


@dataclass(frozen=True)
class FilteredPairs:
    """Positive finite uncertainty/residual pairs and exclusive filter counts."""

    uncertainty: NDArray[np.float64]
    residual: NDArray[np.float64]
    uncertainty_log10: NDArray[np.float64]
    residual_log10: NDArray[np.float64]
    original_count: int
    valid_count: int
    excluded: Mapping[str, int]


@dataclass(frozen=True)
class DensityContours:
    """Normalized smoothed histogram and cumulative-mass contour metadata."""

    x_centers: NDArray[np.float64]
    y_centers: NDArray[np.float64]
    grid: NDArray[np.float64]
    contour_levels: tuple[float, ...]
    actual_contour_masses: tuple[float, ...]
    histogram_count: int


@dataclass(frozen=True)
class PanelAnalysis:
    """All deterministic statistics needed to render one campaign panel."""

    key: PanelKey
    filtered: FilteredPairs
    density: DensityContours
    scatter_indices: NDArray[np.int64]
    log_limits: tuple[float, float]
    spearman_log: float
    pearson_log10: float


def _readonly_float(value: NDArray | Sequence[float]) -> NDArray[np.float64]:
    result = np.array(value, dtype=np.float64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _readonly_indices(value: NDArray | Sequence[int]) -> NDArray[np.int64]:
    result = np.array(value, dtype=np.int64, copy=True, order="C")
    result.setflags(write=False)
    return result


def _numeric_array(value: NDArray, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as error:
        raise HardFailure(f"{name} must be a real numeric array: {error}") from error
    if np.iscomplexobj(array):
        raise HardFailure(f"{name} must be a real numeric array, not complex")
    try:
        return np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise HardFailure(f"{name} must be a real numeric array: {error}") from error


def filter_log_pairs(uncertainty: NDArray, residual: NDArray) -> FilteredPairs:
    """Flatten pairs and classify each invalid pair exactly once."""

    uncertainty_array = _numeric_array(uncertainty, "uncertainty")
    residual_array = _numeric_array(residual, "residual")
    if uncertainty_array.shape != residual_array.shape:
        raise HardFailure("uncertainty and residual arrays must have the same shape")
    if uncertainty_array.size == 0:
        raise HardFailure("uncertainty and residual arrays must be nonempty")

    uncertainty_flat = uncertainty_array.reshape(-1)
    residual_flat = residual_array.reshape(-1)
    nan_mask = np.isnan(uncertainty_flat) | np.isnan(residual_flat)
    inf_mask = (~nan_mask) & (np.isinf(uncertainty_flat) | np.isinf(residual_flat))
    finite_mask = ~(nan_mask | inf_mask)
    zero_mask = finite_mask & ((uncertainty_flat == 0.0) | (residual_flat == 0.0))
    negative_mask = (
        finite_mask & (~zero_mask) & ((uncertainty_flat < 0.0) | (residual_flat < 0.0))
    )
    valid_mask = finite_mask & (~zero_mask) & (~negative_mask)

    valid_uncertainty = uncertainty_flat[valid_mask]
    valid_residual = residual_flat[valid_mask]
    return FilteredPairs(
        uncertainty=_readonly_float(valid_uncertainty),
        residual=_readonly_float(valid_residual),
        uncertainty_log10=_readonly_float(np.log10(valid_uncertainty)),
        residual_log10=_readonly_float(np.log10(valid_residual)),
        original_count=int(uncertainty_flat.size),
        valid_count=int(valid_mask.sum()),
        excluded=MappingProxyType(
            {
                "nan": int(nan_mask.sum()),
                "inf": int(inf_mask.sum()),
                "zero": int(zero_mask.sum()),
                "negative": int(negative_mask.sum()),
            }
        ),
    )


def _positive_finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise HardFailure(f"{name} must be a positive finite number")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise HardFailure(f"{name} must be a positive finite number")
    return result


def _validate_margin(value: object) -> float:
    return _positive_finite(value, "plot log margin")


def _validate_settings(settings: CampaignPlotStyle) -> None:
    if not isinstance(settings, CampaignPlotStyle):
        raise HardFailure("plot settings must use CampaignPlotStyle")
    if type(settings.grid_size) is not int or settings.grid_size <= 0:
        raise HardFailure("plot grid_size must be a positive integer")
    _positive_finite(settings.gaussian_sigma, "plot gaussian_sigma")

    masses = settings.contour_masses
    if not isinstance(masses, tuple) or not masses:
        raise HardFailure("plot contour_masses must be a nonempty tuple")
    normalized_masses: list[float] = []
    for mass in masses:
        if isinstance(mass, bool) or not isinstance(mass, Real):
            raise HardFailure("plot contour_masses must contain finite numbers")
        normalized = float(mass)
        if not math.isfinite(normalized) or not 0 < normalized < 1:
            raise HardFailure("plot contour_masses must lie strictly inside (0, 1)")
        normalized_masses.append(normalized)
    if any(
        left >= right
        for left, right in zip(normalized_masses, normalized_masses[1:], strict=False)
    ):
        raise HardFailure("plot contour_masses must be strictly increasing")

    if type(settings.scatter_max_points) is not int or settings.scatter_max_points <= 0:
        raise HardFailure("plot scatter_max_points must be a positive integer")
    if type(settings.scatter_seed) is not int or settings.scatter_seed < 0:
        raise HardFailure("plot scatter_seed must be a nonnegative integer")
    _positive_finite(settings.scatter_size, "plot scatter_size")
    scatter_alpha = _positive_finite(settings.scatter_alpha, "plot scatter_alpha")
    if scatter_alpha > 1:
        raise HardFailure("plot scatter_alpha must not exceed one")
    _positive_finite(settings.log_margin, "plot log_margin")

    figure_size = settings.figure_size
    if not isinstance(figure_size, tuple) or len(figure_size) != 2:
        raise HardFailure("plot figure_size must contain two positive values")
    for value in figure_size:
        _positive_finite(value, "plot figure_size")
    if type(settings.dpi) is not int or settings.dpi <= 0:
        raise HardFailure("plot dpi must be a positive integer")
    if settings.formats != ("png", "pdf"):
        raise HardFailure("plot formats must be exactly png and pdf")


def _panel_log_extrema(source: PlotSource) -> tuple[float, float]:
    if not isinstance(source, PlotSource):
        raise HardFailure("plot source must use PlotSource")
    uncertainty, residual = load_panel_arrays(source)
    filtered = filter_log_pairs(uncertainty, residual)
    del uncertainty, residual
    if filtered.valid_count < 2:
        raise HardFailure(
            "plot panel requires at least two positive finite valid pairs"
        )
    low = float(min(filtered.uncertainty_log10.min(), filtered.residual_log10.min()))
    high = float(max(filtered.uncertainty_log10.max(), filtered.residual_log10.max()))
    del filtered
    if not math.isfinite(low) or not math.isfinite(high):
        raise HardFailure("plot panel log extrema must be finite")
    return low, high


def scan_shared_log_limits(
    sources: Sequence[PlotSource], *, margin: float
) -> dict[str, tuple[float, float]]:
    """First pass: retain only global log extrema for each physical target."""

    normalized_margin = _validate_margin(margin)
    if isinstance(sources, (str, bytes)) or not isinstance(sources, Sequence):
        raise HardFailure("plot sources must be an ordered sequence")
    if not sources:
        raise HardFailure("plot sources must contain at least one panel")

    extrema: dict[str, tuple[float, float]] = {}
    for source in sources:
        low, high = _panel_log_extrema(source)
        target = source.key.target
        if target not in ("energy", "force", "stress"):
            raise HardFailure(f"plot source target is not supported: {target}")
        if target in extrema:
            previous_low, previous_high = extrema[target]
            extrema[target] = min(previous_low, low), max(previous_high, high)
        else:
            extrema[target] = low, high

    limits: dict[str, tuple[float, float]] = {}
    for target, (low, high) in extrema.items():
        padding = max((high - low) * normalized_margin, normalized_margin)
        lower = low - padding
        upper = high + padding
        if not all(math.isfinite(value) for value in (padding, lower, upper)):
            raise HardFailure(f"plot {target} shared limits are not finite")
        limits[target] = lower, upper
    return limits


def _validate_limits(value: object) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise HardFailure("plot log limits must contain two finite numbers")
    if len(value) != 2:
        raise HardFailure("plot log limits must contain two finite numbers")
    low, high = value
    if any(isinstance(item, bool) or not isinstance(item, Real) for item in value):
        raise HardFailure("plot log limits must contain two finite numbers")
    normalized = float(low), float(high)
    if not all(math.isfinite(item) for item in normalized):
        raise HardFailure("plot log limits must be finite")
    if normalized[0] >= normalized[1]:
        raise HardFailure("plot log limits must increase strictly")
    return normalized


def _sample_indices(count: int, maximum: int, seed: int) -> NDArray[np.int64]:
    if count <= maximum:
        return _readonly_indices(np.arange(count, dtype=np.int64))
    generator = np.random.default_rng(seed)
    selected = np.sort(generator.choice(count, size=maximum, replace=False))
    return _readonly_indices(selected)


def _density_contours(
    filtered: FilteredPairs,
    limits: tuple[float, float],
    settings: CampaignPlotStyle,
) -> DensityContours:
    low, high = limits
    histogram, x_edges, y_edges = np.histogram2d(
        filtered.uncertainty_log10,
        filtered.residual_log10,
        bins=settings.grid_size,
        range=((low, high), (low, high)),
    )
    histogram_count = int(histogram.sum())
    if histogram_count != filtered.valid_count:
        raise HardFailure("plot shared limits do not contain every valid pair")
    if histogram_count < 2:
        raise HardFailure("plot density requires at least two in-range valid pairs")

    density = gaussian_filter(
        histogram,
        sigma=settings.gaussian_sigma,
        mode="nearest",
    )
    total = float(density.sum())
    if not math.isfinite(total) or total <= 0:
        raise HardFailure("plot density has no positive finite mass")
    density = np.asarray(density, dtype=np.float64) / total
    if not bool(np.isfinite(density).all()):
        raise HardFailure("plot density contains non-finite mass")

    descending = np.sort(density.reshape(-1))[::-1]
    cumulative = np.cumsum(descending)
    thresholds: list[float] = []
    for mass in settings.contour_masses:
        index = min(
            int(np.searchsorted(cumulative, mass, side="left")),
            descending.size - 1,
        )
        threshold = float(descending[index])
        if not math.isfinite(threshold) or threshold <= 0:
            raise HardFailure("plot density contour level is not positive and finite")
        thresholds.append(threshold)
    levels = tuple(float(value) for value in np.unique(thresholds))
    if len(levels) != len(settings.contour_masses):
        raise HardFailure("plot density did not produce distinct contour levels")
    actual_masses = tuple(float(density[density >= level].sum()) for level in levels)
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in actual_masses):
        raise HardFailure("plot density contour mass is invalid")

    return DensityContours(
        x_centers=_readonly_float((x_edges[:-1] + x_edges[1:]) * 0.5),
        y_centers=_readonly_float((y_edges[:-1] + y_edges[1:]) * 0.5),
        grid=_readonly_float(density),
        contour_levels=levels,
        actual_contour_masses=actual_masses,
        histogram_count=histogram_count,
    )


def _is_constant_or_near_constant(values: NDArray[np.float64]) -> bool:
    if bool(np.all(values == values[0])):
        return True
    centered = values - values.mean()
    scale = float(np.max(np.abs(centered)))
    if scale == 0 or not math.isfinite(scale):
        return True
    norm = scale * float(np.linalg.norm(centered / scale))
    threshold = float(np.finfo(np.float64).eps ** 0.75 * abs(values.mean()))
    return not math.isfinite(norm) or norm < threshold


def analyze_panel_source(
    source: PlotSource,
    log_limits: tuple[float, float],
    settings: CampaignPlotStyle,
) -> PanelAnalysis:
    """Second pass: compute deterministic statistics for one shared-scale panel."""

    if not isinstance(source, PlotSource):
        raise HardFailure("plot source must use PlotSource")
    _validate_settings(settings)
    limits = _validate_limits(log_limits)
    uncertainty, residual = load_panel_arrays(source)
    filtered = filter_log_pairs(uncertainty, residual)
    del uncertainty, residual
    if filtered.valid_count < 2:
        raise HardFailure(
            "plot panel requires at least two positive finite valid pairs"
        )
    if _is_constant_or_near_constant(
        filtered.uncertainty_log10
    ) or _is_constant_or_near_constant(filtered.residual_log10):
        raise HardFailure("plot correlation is undefined for constant input")

    spearman_value = float(
        spearmanr(filtered.uncertainty_log10, filtered.residual_log10).statistic
    )
    pearson_value = float(
        pearsonr(filtered.uncertainty_log10, filtered.residual_log10).statistic
    )
    if (
        not math.isfinite(spearman_value)
        or not math.isfinite(pearson_value)
        or not -1 <= spearman_value <= 1
        or not -1 <= pearson_value <= 1
    ):
        raise HardFailure("plot correlation is undefined")

    return PanelAnalysis(
        key=source.key,
        filtered=filtered,
        density=_density_contours(filtered, limits, settings),
        scatter_indices=_sample_indices(
            filtered.valid_count,
            settings.scatter_max_points,
            settings.scatter_seed,
        ),
        log_limits=limits,
        spearman_log=spearman_value,
        pearson_log10=pearson_value,
    )


__all__ = [
    "DensityContours",
    "FilteredPairs",
    "PanelAnalysis",
    "analyze_panel_source",
    "filter_log_pairs",
    "scan_shared_log_limits",
]
