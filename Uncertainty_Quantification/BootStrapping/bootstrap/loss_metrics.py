"""Dataset-level aggregation for checkpoint-defined multi-term losses."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class LossStatistics:
    weighted_sums: dict[str, float]
    counts: dict[str, int]

    def __post_init__(self) -> None:
        if set(self.weighted_sums) != set(self.counts) or not self.weighted_sums:
            raise ValueError("loss sums and counts must use identical non-empty keys")
        if not all(math.isfinite(value) for value in self.weighted_sums.values()):
            raise ValueError("loss weighted sums must be finite")
        if any(count < 0 for count in self.counts.values()):
            raise ValueError("loss counts must be non-negative")


class LossAccumulator:
    """Accumulate terms independently so results do not depend on batch splits."""

    def __init__(self, reduction: str) -> None:
        if reduction not in {"mean", "sum"}:
            raise ValueError("loss reduction must be mean or sum")
        self.reduction = reduction
        self._sums: dict[str, float] = {}
        self._counts: dict[str, int] = {}

    def update(self, value: LossStatistics) -> None:
        if self._sums and set(value.weighted_sums) != set(self._sums):
            raise ValueError("loss component set changed between batches")
        for name, weighted_sum in value.weighted_sums.items():
            self._sums[name] = self._sums.get(name, 0.0) + weighted_sum
            self._counts[name] = self._counts.get(name, 0) + value.counts[name]

    def total(self) -> float:
        if not self._sums:
            raise RuntimeError("loss accumulator is empty")
        if self.reduction == "mean":
            if any(count == 0 for count in self._counts.values()):
                raise RuntimeError("loss component count is zero")
            components = {
                name: self._sums[name] / self._counts[name] for name in self._sums
            }
        else:
            components = self._sums
        result = sum(components.values())
        if not math.isfinite(result):
            raise RuntimeError("aggregated loss is non-finite")
        return result
