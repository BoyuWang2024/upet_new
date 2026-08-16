from __future__ import annotations

import gc
import math
import operator
import weakref
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

from Uncertainty_Quantification.BootStrapping.bootstrap import plot_analysis
from Uncertainty_Quantification.BootStrapping.bootstrap.campaign import (
    CampaignPlotStyle,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.plot_analysis import (
    analyze_panel_source,
    filter_log_pairs,
    scan_shared_log_limits,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.plot_source import (
    PanelKey,
    PlotSource,
)


SETTINGS = CampaignPlotStyle(
    grid_size=24,
    gaussian_sigma=1.0,
    contour_masses=(0.5, 0.7, 0.9),
    scatter_max_points=40,
    scatter_seed=20260816,
    scatter_size=4.0,
    scatter_alpha=0.7,
    log_margin=0.05,
    figure_size=(4.0, 4.0),
    dpi=72,
    formats=("png", "pdf"),
)


def _source(
    run_label: str,
    dataset_label: str,
    target: str,
    *,
    storage_key: str | None = None,
) -> PlotSource:
    key = PanelKey(
        run_label=run_label,
        dataset_label=dataset_label,
        storage_key=storage_key or dataset_label,
        target=target,
    )
    root = Path("/synthetic") / run_label / (storage_key or dataset_label)
    return PlotSource(
        key=key,
        targets_path=root / "targets.npz",
        uq_results_path=root / "uncertainty" / "results.npz",
        member_paths=tuple(root / f"member_{index:03d}.npz" for index in range(8)),
        prediction_manifest_sha256="1" * 64,
        uq_manifest_sha256="2" * 64,
        targets_sha256="3" * 64,
    )


def _install_arrays(
    monkeypatch: pytest.MonkeyPatch,
    arrays: dict[
        PanelKey,
        tuple[np.ndarray, np.ndarray],
    ],
) -> None:
    def load_panel_arrays(source: PlotSource) -> tuple[np.ndarray, np.ndarray]:
        uncertainty, residual = arrays[source.key]
        return uncertainty.copy(), residual.copy()

    monkeypatch.setattr(plot_analysis, "load_panel_arrays", load_panel_arrays)


def _curved_pairs(
    *, offset: float = 0.0, count: int = 180
) -> tuple[np.ndarray, np.ndarray]:
    log_uncertainty = np.linspace(-3.0 + offset, 2.0 + offset, count)
    log_residual = (
        0.62 * log_uncertainty
        + 0.24 * np.sin(2.7 * log_uncertainty)
        + 0.11 * np.cos(6.1 * log_uncertainty)
    )
    return np.power(10.0, log_uncertainty), np.power(10.0, log_residual)


def test_filter_log_pairs_uses_exclusive_nan_inf_zero_negative_precedence() -> None:
    uncertainty = np.array(
        [np.nan, np.nan, np.inf, np.inf, 0.0, 0.0, -1.0, 1.0, 1.0, 10.0]
    )
    residual = np.array([np.inf, 2.0, np.nan, 2.0, -1.0, 3.0, 0.0, -2.0, 2.0, 100.0])

    result = filter_log_pairs(uncertainty, residual)

    assert result.original_count == 10
    assert result.valid_count == 2
    assert result.excluded == {"nan": 3, "inf": 1, "zero": 3, "negative": 1}
    assert result.valid_count + sum(result.excluded.values()) == result.original_count


def test_filter_log_pairs_returns_exact_valid_arrays_and_base_ten_logs() -> None:
    uncertainty = np.array([1.0e-3, 0.0, 10.0, -2.0, 100.0, np.inf])
    residual = np.array([1.0e-2, 2.0, 1.0e2, 4.0, 1.0e-1, 8.0])

    result = filter_log_pairs(uncertainty, residual)

    np.testing.assert_array_equal(result.uncertainty, [1.0e-3, 10.0, 100.0])
    np.testing.assert_array_equal(result.residual, [1.0e-2, 1.0e2, 1.0e-1])
    np.testing.assert_allclose(result.uncertainty_log10, [-3.0, 1.0, 2.0])
    np.testing.assert_allclose(result.residual_log10, [-2.0, 2.0, -1.0])


def test_filtered_exclusion_counts_are_runtime_immutable() -> None:
    result = filter_log_pairs(
        np.array([1.0, 0.0, np.nan]),
        np.array([2.0, 3.0, 4.0]),
    )

    with pytest.raises(TypeError):
        operator.setitem(result.excluded, "nan", 99)


@pytest.mark.parametrize(
    ("uncertainty", "residual"),
    [
        (
            np.array([1.0 + 2.0j, 3.0 + 0.0j]),
            np.array([2.0, 4.0]),
        ),
        (
            np.array([1.0, 3.0]),
            np.array([2.0 + 0.0j, 4.0 + 5.0j]),
        ),
    ],
)
def test_filter_log_pairs_rejects_complex_arrays_without_warning_leakage(
    uncertainty: np.ndarray, residual: np.ndarray
) -> None:
    with pytest.raises(HardFailure, match="complex|real"):
        filter_log_pairs(uncertainty, residual)


def test_shared_limits_use_global_target_extrema_and_are_order_independent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    energy_left = _source("run_z", "dataset_a", "energy")
    energy_right = _source("run_a", "dataset_b", "energy")
    force = _source("run_m", "dataset_a", "force")
    stress = _source("run_z", "dataset_b", "stress")
    sources = (energy_left, energy_right, force, stress)
    arrays = {
        energy_left.key: (np.array([1.0, 100.0]), np.array([10.0, 1000.0])),
        energy_right.key: (np.array([0.1, 10.0]), np.array([1.0, 100.0])),
        force.key: (np.array([1.0e-4, 1.0e-3]), np.array([1.0e-2, 1.0e-1])),
        stress.key: (np.array([2.0, 2.0]), np.array([2.0, 2.0])),
    }
    _install_arrays(monkeypatch, arrays)

    forward = scan_shared_log_limits(sources, margin=0.1)
    backward = scan_shared_log_limits(tuple(reversed(sources)), margin=0.1)

    assert set(forward) == {"energy", "force", "stress"}
    assert backward == forward
    assert forward["energy"] == pytest.approx((-1.4, 3.4))
    assert forward["force"] == pytest.approx((-4.3, -0.7))
    center = math.log10(2.0)
    assert forward["stress"] == pytest.approx((center - 0.1, center + 0.1))


def test_every_panel_analysis_uses_its_target_shared_limits_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = tuple(
        _source(run, dataset, target)
        for target in ("energy", "force", "stress")
        for run, dataset in (("run_z", "dataset_a"), ("run_a", "dataset_b"))
    )
    arrays = {
        source.key: _curved_pairs(
            offset={"energy": -1.0, "force": 0.5, "stress": 2.0}[source.key.target]
            + (0.2 if source.key.run_label == "run_a" else 0.0),
            count=96,
        )
        for source in sources
    }
    _install_arrays(monkeypatch, arrays)
    limits = scan_shared_log_limits(sources, margin=0.05)

    analyses = tuple(
        analyze_panel_source(source, limits[source.key.target], SETTINGS)
        for source in sources
    )

    for target in ("energy", "force", "stress"):
        target_limits = {
            analysis.log_limits
            for analysis in analyses
            if analysis.key.target == target
        }
        assert target_limits == {limits[target]}


def test_panel_analysis_is_bitwise_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source("run_z", "dataset_a", "energy")
    arrays = {source.key: _curved_pairs(count=240)}
    _install_arrays(monkeypatch, arrays)
    limits = scan_shared_log_limits((source,), margin=0.05)["energy"]

    left = analyze_panel_source(source, limits, SETTINGS)
    right = analyze_panel_source(source, limits, SETTINGS)

    np.testing.assert_array_equal(left.scatter_indices, right.scatter_indices)
    np.testing.assert_array_equal(left.filtered.uncertainty, right.filtered.uncertainty)
    np.testing.assert_array_equal(left.filtered.residual, right.filtered.residual)
    np.testing.assert_array_equal(left.density.x_centers, right.density.x_centers)
    np.testing.assert_array_equal(left.density.y_centers, right.density.y_centers)
    np.testing.assert_array_equal(left.density.grid, right.density.grid)
    assert left.density.contour_levels == right.density.contour_levels
    assert left.density.actual_contour_masses == right.density.actual_contour_masses
    assert left.spearman_log == right.spearman_log
    assert left.pearson_log10 == right.pearson_log10


def test_all_analysis_array_payloads_are_runtime_read_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source("run_z", "dataset_a", "energy")
    _install_arrays(monkeypatch, {source.key: _curved_pairs(count=160)})
    limits = scan_shared_log_limits((source,), margin=0.05)["energy"]

    analysis = analyze_panel_source(source, limits, SETTINGS)

    arrays = (
        analysis.filtered.uncertainty,
        analysis.filtered.residual,
        analysis.filtered.uncertainty_log10,
        analysis.filtered.residual_log10,
        analysis.density.x_centers,
        analysis.density.y_centers,
        analysis.density.grid,
        analysis.scatter_indices,
    )
    assert all(array.flags.writeable is False for array in arrays)


def test_density_grid_centers_levels_and_cumulative_mass_are_exact(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source("run_z", "dataset_a", "force")
    arrays = {source.key: _curved_pairs(offset=0.3, count=400)}
    _install_arrays(monkeypatch, arrays)
    settings = replace(SETTINGS, grid_size=32, contour_masses=(0.5, 0.8, 0.95))
    limits = scan_shared_log_limits((source,), margin=0.05)["force"]

    analysis = analyze_panel_source(source, limits, settings)
    density = analysis.density

    uncertainty, residual = arrays[source.key]
    expected_histogram, x_edges, y_edges = np.histogram2d(
        np.log10(uncertainty),
        np.log10(residual),
        bins=settings.grid_size,
        range=(limits, limits),
    )
    expected_grid = gaussian_filter(
        expected_histogram,
        sigma=settings.gaussian_sigma,
        mode="nearest",
    )
    expected_grid /= expected_grid.sum()
    descending = np.sort(expected_grid.reshape(-1))[::-1]
    cumulative = np.cumsum(descending)
    expected_thresholds = []
    for mass in settings.contour_masses:
        index = min(
            int(np.searchsorted(cumulative, mass, side="left")),
            descending.size - 1,
        )
        expected_thresholds.append(float(descending[index]))
    expected_levels = tuple(float(value) for value in np.unique(expected_thresholds))

    assert density.grid.shape == (settings.grid_size, settings.grid_size)
    assert density.x_centers.shape == density.y_centers.shape == (settings.grid_size,)
    assert density.histogram_count == analysis.filtered.valid_count
    assert float(density.grid.sum()) == pytest.approx(1.0)
    expected_spacing = (limits[1] - limits[0]) / settings.grid_size
    np.testing.assert_allclose(np.diff(density.x_centers), expected_spacing)
    np.testing.assert_allclose(np.diff(density.y_centers), expected_spacing)
    np.testing.assert_allclose(density.x_centers, (x_edges[:-1] + x_edges[1:]) * 0.5)
    np.testing.assert_allclose(density.y_centers, (y_edges[:-1] + y_edges[1:]) * 0.5)
    np.testing.assert_allclose(density.grid, expected_grid, rtol=0.0, atol=1.0e-15)
    assert limits[0] < density.x_centers[0] < density.x_centers[-1] < limits[1]
    assert limits[0] < density.y_centers[0] < density.y_centers[-1] < limits[1]
    assert len(density.contour_levels) == len(settings.contour_masses)
    assert all(math.isfinite(level) and level > 0 for level in density.contour_levels)
    assert all(
        left < right
        for left, right in zip(
            density.contour_levels, density.contour_levels[1:], strict=False
        )
    )
    assert density.contour_levels == expected_levels
    independent_masses = tuple(
        float(density.grid[density.grid >= level].sum())
        for level in density.contour_levels
    )
    np.testing.assert_allclose(density.actual_contour_masses, independent_masses)
    for actual, requested in zip(
        reversed(density.actual_contour_masses),
        settings.contour_masses,
        strict=True,
    ):
        assert requested <= actual <= 1.0


def test_log_correlations_match_independent_scipy_and_manual_pearson(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source("run_z", "dataset_a", "energy")
    log_uncertainty = np.array([-3.0, -2.0, -1.0, 0.0, 1.0, 2.0])
    log_residual = np.array([-2.0, 0.0, -1.0, 3.0, 2.0, 5.0])
    arrays = {
        source.key: (
            np.power(10.0, log_uncertainty),
            np.power(10.0, log_residual),
        )
    }
    _install_arrays(monkeypatch, arrays)
    limits = scan_shared_log_limits((source,), margin=0.05)["energy"]

    analysis = analyze_panel_source(source, limits, SETTINGS)

    expected_spearman = float(spearmanr(log_uncertainty, log_residual).statistic)
    centered_x = log_uncertainty - log_uncertainty.mean()
    centered_y = log_residual - log_residual.mean()
    expected_pearson = float(
        np.dot(centered_x, centered_y)
        / np.sqrt(np.dot(centered_x, centered_x) * np.dot(centered_y, centered_y))
    )
    assert analysis.spearman_log == pytest.approx(expected_spearman, abs=1.0e-15)
    assert analysis.pearson_log10 == pytest.approx(expected_pearson, abs=1.0e-15)


@pytest.mark.parametrize(
    ("uncertainty", "residual", "message"),
    [
        (np.array([1.0, 0.0]), np.array([2.0, 3.0]), "two|valid"),
        (np.ones(8), np.linspace(1.0, 8.0, 8), "correlation|constant|undefined"),
        (np.linspace(1.0, 8.0, 8), np.ones(8), "correlation|constant|undefined"),
    ],
)
def test_analysis_rejects_too_few_pairs_and_undefined_correlations(
    monkeypatch: pytest.MonkeyPatch,
    uncertainty: np.ndarray,
    residual: np.ndarray,
    message: str,
) -> None:
    source = _source("run_z", "dataset_a", "energy")
    _install_arrays(monkeypatch, {source.key: (uncertainty, residual)})

    with pytest.raises(HardFailure, match=message):
        analyze_panel_source(source, (-2.0, 2.0), SETTINGS)


def test_shared_limit_scan_rejects_no_analyzable_target_or_invalid_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source("run_z", "dataset_a", "energy")
    _install_arrays(
        monkeypatch,
        {source.key: (np.array([1.0, 0.0, np.nan]), np.array([1.0, 2.0, 3.0]))},
    )

    with pytest.raises(HardFailure, match="two|valid|analyzable"):
        scan_shared_log_limits((source,), margin=0.05)
    with pytest.raises(HardFailure, match="source|target|panel"):
        scan_shared_log_limits((), margin=0.05)
    for margin in (0.0, -0.1, float("nan"), float("inf"), True):
        with pytest.raises(HardFailure, match="margin"):
            scan_shared_log_limits((source,), margin=margin)


@pytest.mark.parametrize(
    "limits",
    [
        (0.0, 0.0),
        (1.0, 0.0),
        (float("nan"), 1.0),
        (0.0, float("inf")),
    ],
)
def test_analysis_rejects_invalid_shared_limits(
    monkeypatch: pytest.MonkeyPatch, limits: tuple[float, float]
) -> None:
    source = _source("run_z", "dataset_a", "energy")
    _install_arrays(monkeypatch, {source.key: _curved_pairs(count=80)})

    with pytest.raises(HardFailure, match="limit"):
        analyze_panel_source(source, limits, SETTINGS)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("grid_size", 0),
        ("gaussian_sigma", 0.0),
        ("contour_masses", ()),
        ("contour_masses", (0.8, 0.5)),
        ("scatter_max_points", 0),
        ("scatter_seed", -1),
    ],
)
def test_analysis_rejects_invalid_settings_fail_closed(
    monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    source = _source("run_z", "dataset_a", "energy")
    _install_arrays(monkeypatch, {source.key: _curved_pairs(count=80)})
    settings = replace(SETTINGS, **{field: value})

    with pytest.raises(HardFailure, match="plot|setting|grid|sigma|contour|scatter"):
        analyze_panel_source(source, (-4.0, 3.0), settings)


def test_first_limit_pass_releases_each_panel_arrays_before_loading_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sources = tuple(
        _source(f"run_{index}", f"dataset_{index}", "energy") for index in range(6)
    )
    state = {"live": 0, "peak": 0}
    calls: list[PanelKey] = []

    class TrackedArray(np.ndarray):
        pass

    def released() -> None:
        state["live"] -= 1

    def load_panel_arrays(source: PlotSource) -> tuple[np.ndarray, np.ndarray]:
        gc.collect()
        uncertainty = np.linspace(1.0, 9.0, 64).view(TrackedArray)
        residual = np.linspace(2.0, 10.0, 64).view(TrackedArray)
        for array in (uncertainty, residual):
            state["live"] += 1
            state["peak"] = max(state["peak"], state["live"])
            weakref.finalize(array, released)
        calls.append(source.key)
        return uncertainty, residual

    monkeypatch.setattr(plot_analysis, "load_panel_arrays", load_panel_arrays)

    limits = scan_shared_log_limits(sources, margin=0.05)
    gc.collect()

    assert set(limits) == {"energy"}
    assert calls == [source.key for source in sources]
    assert state["peak"] <= 2
    assert state["live"] == 0


def test_scatter_subsample_is_seeded_sorted_and_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source("run_z", "dataset_a", "energy")
    count = 200
    maximum = 17
    seed = 9127
    arrays = {source.key: _curved_pairs(count=count)}
    _install_arrays(monkeypatch, arrays)
    settings = replace(
        SETTINGS,
        scatter_max_points=maximum,
        scatter_seed=seed,
    )
    limits = scan_shared_log_limits((source,), margin=0.05)["energy"]

    analysis = analyze_panel_source(source, limits, settings)
    expected = np.sort(
        np.random.default_rng(seed).choice(count, size=maximum, replace=False)
    )

    np.testing.assert_array_equal(analysis.scatter_indices, expected)
    assert analysis.scatter_indices.dtype == np.int64
    assert len(analysis.scatter_indices) == maximum
    assert np.all(np.diff(analysis.scatter_indices) > 0)
    assert analysis.scatter_indices[0] >= 0
    assert analysis.scatter_indices[-1] < count


def test_scatter_keeps_every_index_when_panel_is_below_the_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source("run_z", "dataset_a", "energy")
    count = 12
    _install_arrays(monkeypatch, {source.key: _curved_pairs(count=count)})
    limits = scan_shared_log_limits((source,), margin=0.05)["energy"]

    analysis = analyze_panel_source(source, limits, SETTINGS)

    np.testing.assert_array_equal(analysis.scatter_indices, np.arange(count))
