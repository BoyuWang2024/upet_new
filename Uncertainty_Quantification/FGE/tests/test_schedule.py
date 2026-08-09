"""Contract tests for the legacy asymmetric FGE learning-rate schedule."""

from __future__ import annotations

import math

import pytest

from Uncertainty_Quantification.FGE.fge.schedule import asymmetric_triangular_lr


def test_schedule_preserves_legacy_update_index_boundaries() -> None:
    # With six updates, the legacy position is step / (updates - 1).  Therefore
    # step 1 is the exact 0.2 peak and step 3 is halfway down the falling limb.
    expected = {
        0: 1.0e-8,
        1: 1.0e-7,
        3: 5.5e-8,
        5: 1.0e-8,
    }

    for step, value in expected.items():
        assert asymmetric_triangular_lr(
            step=step,
            updates_per_cycle=6,
            lr_min=1.0e-8,
            lr_max=1.0e-7,
            rise_fraction=0.2,
        ) == pytest.approx(value)


@pytest.mark.parametrize("updates_per_cycle", [0, 1, -2, 2.5, True])
def test_schedule_rejects_invalid_update_counts(updates_per_cycle: object) -> None:
    with pytest.raises((TypeError, ValueError), match="updates_per_cycle"):
        asymmetric_triangular_lr(
            step=0,
            updates_per_cycle=updates_per_cycle,  # type: ignore[arg-type]
            lr_min=1.0e-8,
            lr_max=1.0e-7,
            rise_fraction=0.2,
        )


@pytest.mark.parametrize("step", [-1, 6, 1.5, True])
def test_schedule_rejects_steps_outside_the_cycle(step: object) -> None:
    with pytest.raises((TypeError, ValueError), match="step"):
        asymmetric_triangular_lr(
            step=step,  # type: ignore[arg-type]
            updates_per_cycle=6,
            lr_min=1.0e-8,
            lr_max=1.0e-7,
            rise_fraction=0.2,
        )


@pytest.mark.parametrize(
    ("lr_min", "lr_max"),
    [
        (0.0, 1.0e-7),
        (-1.0e-8, 1.0e-7),
        (1.0e-7, 1.0e-7),
        (2.0e-7, 1.0e-7),
        (math.nan, 1.0e-7),
        (1.0e-8, math.inf),
    ],
)
def test_schedule_rejects_invalid_learning_rate_bounds(
    lr_min: float, lr_max: float
) -> None:
    with pytest.raises(ValueError, match="learning rates"):
        asymmetric_triangular_lr(
            step=0,
            updates_per_cycle=6,
            lr_min=lr_min,
            lr_max=lr_max,
            rise_fraction=0.2,
        )


@pytest.mark.parametrize("rise_fraction", [0.0, 1.0, -0.1, 1.1, math.nan])
def test_schedule_rejects_invalid_rise_fraction(rise_fraction: float) -> None:
    with pytest.raises(ValueError, match="rise_fraction"):
        asymmetric_triangular_lr(
            step=0,
            updates_per_cycle=6,
            lr_min=1.0e-8,
            lr_max=1.0e-7,
            rise_fraction=rise_fraction,
        )
