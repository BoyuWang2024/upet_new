"""Pure residual, correlation, and risk-coverage calculations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .errors import HardFailure


@dataclass(frozen=True)
class CorrelationSummary:
    pearson: float
    spearman: float


@dataclass(frozen=True)
class RiskCoverage:
    coverage: NDArray[np.float64]
    risk: NDArray[np.float64]


def _vector(values: NDArray, location: str) -> NDArray[np.float64]:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 1 or result.size < 1 or not bool(np.isfinite(result).all()):
        raise HardFailure(f"{location} must be a non-empty finite vector")
    return result


def absolute_residual(prediction: NDArray, reference: NDArray) -> NDArray[np.float64]:
    predicted = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if predicted.shape != target.shape or not bool(
        np.isfinite(predicted).all() and np.isfinite(target).all()
    ):
        raise HardFailure("prediction and reference must have the same finite layout")
    return np.abs(predicted - target)


def _ranks(values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2 + 1
        start = stop
    return ranks


def _correlation(left: NDArray[np.float64], right: NDArray[np.float64]) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.sqrt(np.sum(left_centered**2) * np.sum(right_centered**2)))
    if denominator == 0:
        raise HardFailure("correlation is undefined for a constant vector")
    return float(np.sum(left_centered * right_centered) / denominator)


def correlation_summary(residual: NDArray, uncertainty: NDArray) -> CorrelationSummary:
    residual_values = _vector(residual, "residual")
    uncertainty_values = _vector(uncertainty, "uncertainty")
    if residual_values.shape != uncertainty_values.shape or len(residual_values) < 2:
        raise HardFailure(
            "correlation inputs must have the same length of at least two"
        )
    return CorrelationSummary(
        pearson=_correlation(residual_values, uncertainty_values),
        spearman=_correlation(_ranks(residual_values), _ranks(uncertainty_values)),
    )


def risk_coverage(residual: NDArray, uncertainty: NDArray) -> RiskCoverage:
    residual_values = _vector(residual, "residual")
    uncertainty_values = _vector(uncertainty, "uncertainty")
    if residual_values.shape != uncertainty_values.shape:
        raise HardFailure("risk-coverage inputs must have the same layout")
    order = np.argsort(uncertainty_values, kind="mergesort")
    ordered_risk = residual_values[order]
    count = len(ordered_risk)
    coverage = np.arange(1, count + 1, dtype=np.float64) / count
    risk = np.cumsum(ordered_risk, dtype=np.float64) / np.arange(1, count + 1)
    return RiskCoverage(coverage=coverage, risk=risk)
