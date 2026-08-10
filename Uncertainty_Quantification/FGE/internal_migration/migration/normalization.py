"""Schema-only normalization of preserved legacy evaluation values."""

from __future__ import annotations

import math
from collections.abc import Mapping

from ...fge.config import FGEConfig
from ...fge.errors import HardFailure
from .legacy_reader import LegacyRun


_CORRELATION_NAMES = {
    "energy_total_std": "energy_total_std",
    "energy_total_GMD": "energy_total_gmd",
    "energy_atom_std": "energy_per_atom_std",
    "energy_atom_GMD": "energy_per_atom_gmd",
    "force_atom_vector_std": "force_atom_vector_std",
    "force_atom_vector_GMD": "force_atom_vector_gmd",
    "force_component_std": "force_component_std",
    "force_component_GMD": "force_component_gmd",
    "force_structure_mean_std": "force_structure_mean_std",
    "force_structure_max_std": "force_structure_max_std",
    "force_structure_q95_std": "force_structure_q95_std",
}
_RISK_NAMES = {
    "energy_total_std",
    "energy_per_atom_std",
    "force_atom_vector_std",
    "force_structure_q95_std",
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise HardFailure(f"legacy {label} is invalid")
    return value


def _finite(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HardFailure(f"legacy {label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HardFailure(f"legacy {label} is non-finite")
    return result


def _canonical_metrics(
    run: LegacyRun, config: FGEConfig
) -> tuple[dict[str, object], Mapping[str, object]]:
    old = run.legacy_metrics
    base = _mapping(old.get("base_metrics"), "base metrics")
    counts = _mapping(old.get("counts"), "counts")
    old_correlations = _mapping(old.get("correlations"), "correlations")
    if set(old_correlations) != set(_CORRELATION_NAMES):
        raise HardFailure("legacy correlation inventory is invalid")
    correlations: dict[str, dict[str, object]] = {}
    for old_name, new_name in _CORRELATION_NAMES.items():
        diagnostic = _mapping(old_correlations[old_name], "correlation diagnostic")
        if set(diagnostic) != {"pearson", "spearman", "disagreement_collapse", "n"}:
            raise HardFailure("legacy correlation diagnostic schema is invalid")
        collapse = diagnostic["disagreement_collapse"]
        n = diagnostic["n"]
        if (
            type(collapse) is not bool
            or isinstance(n, bool)
            or not isinstance(n, int)
            or n < 1
        ):
            raise HardFailure("legacy correlation diagnostic identity is invalid")
        pearson, spearman = diagnostic["pearson"], diagnostic["spearman"]
        undefined = pearson is None or spearman is None
        if undefined:
            if pearson is not None or spearman is not None:
                raise HardFailure("legacy undefined correlation is inconsistent")
            correlations[new_name] = {
                "status": "undefined_constant_input",
                "pearson": None,
                "spearman": None,
                "uncertainty_constant": collapse,
                "error_constant": True,
                "n": n,
            }
        else:
            correlations[new_name] = {
                "status": "ok",
                "pearson": _finite(pearson, "pearson"),
                "spearman": _finite(spearman, "spearman"),
                "uncertainty_constant": False,
                "error_constant": False,
                "n": n,
            }

    risks = run.legacy_risk_coverage
    if set(risks) != _RISK_NAMES:
        raise HardFailure("legacy risk coverage inventory is invalid")
    coverages = tuple(config.evaluation.risk_coverages)
    statistics = _mapping(run.prediction.get("statistics"), "prediction statistics")
    sizes = {
        "energy_total_std": statistics.get("S"),
        "energy_per_atom_std": statistics.get("S"),
        "force_atom_vector_std": statistics.get("A"),
        "force_structure_q95_std": statistics.get("S"),
    }
    risk_coverage: dict[str, object] = {}
    for name in sorted(_RISK_NAMES):
        rows = risks[name]
        if len(rows) != len(coverages):
            raise HardFailure("legacy risk coverage length is invalid")
        size = sizes[name]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise HardFailure("legacy risk coverage population is invalid")
        normalized = []
        for expected_coverage, row in zip(coverages, rows, strict=False):
            coverage = _finite(row.get("coverage"), "risk coverage")
            if coverage != expected_coverage:
                raise HardFailure("legacy risk coverage order is invalid")
            normalized.append(
                {
                    "coverage": coverage,
                    "kept": max(1, math.ceil(size * coverage)),
                    "risk": _finite(row.get("risk"), "risk value"),
                }
            )
        risk_coverage[name] = normalized
    mae = {
        "energy_total": _finite(base.get("energy_total_mae"), "energy total MAE"),
        "energy_per_atom": _finite(
            base.get("energy_per_atom_mae"), "energy per-atom MAE"
        ),
        "force_component": _finite(
            base.get("force_component_mae"), "force component MAE"
        ),
        "stress_component": _finite(
            base.get("stress_component_mae"), "stress component MAE"
        ),
    }
    atoms = statistics.get("A")
    if isinstance(atoms, bool) or not isinstance(atoms, int) or atoms < 1:
        raise HardFailure("legacy atom count is invalid")
    canonical_counts = {
        "structures": counts.get("structures"),
        "atoms": atoms,
        "force_components": counts.get("force_components"),
        "stress_components": counts.get("stress_components"),
    }
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in canonical_counts.values()
    ):
        raise HardFailure("legacy metric counts are invalid")
    metrics = {
        "schema_version": 4,
        "mae": mae,
        "counts": canonical_counts,
        "correlations": correlations,
        "risk_coverage": risk_coverage,
    }
    report_inputs = {
        "formula_version": "legacy_upet_fge_v1",
        "metric_schema_version": 4,
        "member_count": statistics.get("K"),
        "structure_count": statistics.get("S"),
        "atom_count": statistics.get("A"),
        "coverages": list(coverages),
        "undefined_correlation_count": sum(
            diagnostic["status"] == "undefined_constant_input"
            for diagnostic in correlations.values()
        ),
    }
    return metrics, report_inputs


def evaluation_values(
    run: LegacyRun, config: FGEConfig
) -> tuple[
    dict[str, object], Mapping[str, object], dict[str, object], Mapping[str, object]
]:
    """Normalize names/key trees while retaining every legacy numerical value."""

    legacy = run.legacy_uncertainty
    ensemble = {
        "energy": legacy.get("E_mean"),
        "forces": legacy.get("F_mean"),
        "stress": legacy.get("S_mean"),
    }
    uncertainty = legacy.get("canonical_uncertainty")
    if not isinstance(uncertainty, Mapping):
        uncertainty = {
            "formula_version": "legacy_upet_fge_v1",
            "energy_total": {
                "std": legacy.get("U_E_total_std"),
                "gmd": legacy.get("U_E_total_GMD"),
            },
            "energy_per_atom": {
                "std": legacy.get("U_E_atom_std"),
                "gmd": legacy.get("U_E_atom_GMD"),
            },
            "force_component": {
                "std": legacy.get("U_F_component_std"),
                "gmd": legacy.get("U_F_component_GMD"),
            },
            "force_atom_vector": {
                "std": legacy.get("U_F_atom_vector_std"),
                "gmd": legacy.get("U_F_atom_vector_GMD"),
            },
            "force_structure": {
                "std": {
                    "mean": legacy.get("U_F_structure_mean_std"),
                    "max": legacy.get("U_F_structure_max_std"),
                    "q95": legacy.get("U_F_structure_q95_std"),
                }
            },
        }
    metrics = dict(run.legacy_metrics)
    report_inputs = metrics.pop("report_inputs", None)
    if metrics.get("schema_version") == 4 and isinstance(report_inputs, Mapping):
        return ensemble, uncertainty, metrics, report_inputs
    metrics, report_inputs = _canonical_metrics(run, config)
    return ensemble, uncertainty, metrics, report_inputs
