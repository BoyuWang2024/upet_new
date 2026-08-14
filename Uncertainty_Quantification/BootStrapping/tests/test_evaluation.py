from __future__ import annotations

import numpy as np
import pytest


def test_residuals_and_correlations_are_pure_float64() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.evaluation import (
        absolute_residual,
        correlation_summary,
    )

    residual = absolute_residual(np.array([2.0, 4.0]), np.array([1.0, 6.0]))
    summary = correlation_summary(np.array([1.0, 2.0, 3.0]), np.array([2.0, 4.0, 8.0]))

    assert residual == pytest.approx(np.array([1.0, 2.0]))
    assert residual.dtype == np.float64
    assert summary.pearson > 0.98
    assert summary.spearman == pytest.approx(1.0)


def test_risk_coverage_orders_high_uncertainty_first() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.evaluation import (
        risk_coverage,
    )

    curve = risk_coverage(
        residual=np.array([1.0, 9.0, 2.0]),
        uncertainty=np.array([0.1, 0.9, 0.2]),
    )

    assert curve.coverage == pytest.approx(np.array([1 / 3, 2 / 3, 1.0]))
    assert curve.risk == pytest.approx(np.array([1.0, 1.5, 4.0]))
