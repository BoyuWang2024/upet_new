"""Stateless legacy FGE learning-rate schedule."""

from __future__ import annotations

import math


def asymmetric_triangular_lr(
    step: int,
    updates_per_cycle: int,
    lr_min: float,
    lr_max: float,
    rise_fraction: float,
) -> float:
    """Return the legacy asymmetric triangular rate for one update index."""
    if isinstance(updates_per_cycle, bool) or not isinstance(updates_per_cycle, int):
        raise TypeError("updates_per_cycle must be an integer")
    if updates_per_cycle < 2:
        raise ValueError("updates_per_cycle must be at least 2")
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError("step must be an integer")
    if not 0 <= step < updates_per_cycle:
        raise ValueError("step must be within the update cycle")
    if not (math.isfinite(lr_min) and math.isfinite(lr_max) and 0.0 < lr_min < lr_max):
        raise ValueError("learning rates must be finite and satisfy 0 < min < max")
    if not math.isfinite(rise_fraction) or not 0.0 < rise_fraction < 1.0:
        raise ValueError("rise_fraction must be finite and lie in (0, 1)")

    cycle_position = step / float(updates_per_cycle - 1)
    if cycle_position <= rise_fraction:
        return lr_min + (lr_max - lr_min) * cycle_position / rise_fraction
    return lr_max - (lr_max - lr_min) * (cycle_position - rise_fraction) / (
        1.0 - rise_fraction
    )
