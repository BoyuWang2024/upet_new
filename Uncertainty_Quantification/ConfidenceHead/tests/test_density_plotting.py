from __future__ import annotations

from pathlib import Path

import pytest
import torch

from confidence_head.density_plotting import (
    DensitySettings,
    analyze_density_panel,
    shared_log_limits,
)
from confidence_head.plot_analysis import PlotSeries


def _series(
    observed: torch.Tensor,
    expected: torch.Tensor | None = None,
    *,
    target: str = "energy",
    order: int | None = 1,
) -> PlotSeries:
    expected = observed.clone() if expected is None else expected
    logits = torch.zeros((len(observed), 50), dtype=torch.float64)
    logits[:, 0] = 1.0
    return PlotSeries(
        run_dir=Path("/synthetic") / (f"energy-{order}" if target == "energy" else "force"),
        structure_ids=torch.arange(len(observed)),
        target=target,  # type: ignore[arg-type]
        order=order,
        logits=logits,
        observed=observed.to(torch.float64),
        expected=expected.to(torch.float64),
        representatives=torch.linspace(0.005, 0.495, 50, dtype=torch.float64),
        stored_metrics={"sample_count": len(observed), "pearson": 0.5, "spearman": 0.5},
        force_target_mode="atom_mean" if target == "force" else None,
    )


def _series_with_invalid_pairs() -> PlotSeries:
    observed = torch.tensor([0.02, 0.08, 0.15, 0.24, 0.0, -0.1, float("nan"), float("inf")])
    expected = torch.tensor([0.01, 0.10, 0.13, 0.30, 0.2, 0.2, 0.2, 0.2])
    return _series(observed, expected)


def _large_series(count: int) -> PlotSeries:
    values = torch.logspace(-4, 1, count, dtype=torch.float64)
    return _series(values, values * 1.15)


def test_analyze_density_filters_pairs_and_reports_counts() -> None:
    panel = analyze_density_panel(_series_with_invalid_pairs(), DensitySettings())

    assert panel.filtered.original_count == 8
    assert panel.filtered.valid_count == 4
    assert panel.filtered.excluded == {
        "nan": 1,
        "inf": 1,
        "negative": 1,
        "zero": 1,
    }
    assert panel.density.grid.shape == (160, 160)


def test_sampling_is_deterministic_and_density_uses_all_valid_pairs() -> None:
    settings = DensitySettings(scatter_max_points=20, scatter_seed=17)
    first = analyze_density_panel(_large_series(500), settings)
    second = analyze_density_panel(_large_series(500), settings)

    assert torch.equal(first.scatter_indices, second.scatter_indices)
    assert first.scatter_indices.numel() == 20
    assert first.density.histogram_count == first.filtered.valid_count


def test_shared_log_limits_cover_all_energy_orders() -> None:
    series = tuple(
        _series(
            torch.tensor([10.0 ** (-order)]),
            torch.tensor([2.0 * 10.0 ** (-order)]),
            order=order,
        )
        for order in range(1, 9)
    )

    low, high = shared_log_limits(series, margin=0.05)

    assert low < -8.0
    assert high > 0.0


def test_density_requires_two_valid_pairs() -> None:
    with pytest.raises(ValueError, match="at least two"):
        analyze_density_panel(
            _series(torch.tensor([0.0, float("nan")])), DensitySettings()
        )
