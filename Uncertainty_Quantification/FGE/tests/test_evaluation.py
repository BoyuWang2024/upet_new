"""Tests for pure, structured evaluation of canonical FGE predictions."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError

import pytest
import torch

from Uncertainty_Quantification.FGE.fge import (
    EvaluationArtifacts,
    evaluate_prediction,
    global_mae,
)


def _payload() -> dict[str, object]:
    force_mean = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [1.0, 1.0, 1.0]],
        dtype=torch.float32,
    )
    return {
        "energy_prediction": torch.tensor(
            [[1.0, 1.0, 1.0], [3.0, 3.0, 5.0]], dtype=torch.float32
        ),
        "forces_prediction": torch.stack(
            (torch.zeros_like(force_mean), 2 * force_mean)
        ),
        "stress_prediction": torch.stack(
            (
                torch.zeros((3, 3, 3), dtype=torch.float32),
                torch.full((3, 3, 3), 2.0, dtype=torch.float32),
            )
        ),
        "energy_reference": torch.zeros(3, dtype=torch.float32),
        "forces_reference": torch.zeros((4, 3), dtype=torch.float32),
        "stress_reference": torch.zeros((3, 3, 3), dtype=torch.float32),
        "n_atoms": torch.tensor([2, 1, 1], dtype=torch.int64),
        "structure_offsets": torch.tensor([0, 2, 3, 4], dtype=torch.int64),
        "member_ids": ("member_001", "member_002"),
    }


def test_global_mae_uses_all_scalar_elements() -> None:
    prediction = torch.tensor([[1.0, -3.0], [5.0, 7.0]], dtype=torch.float64)
    reference = torch.tensor([[0.0, 1.0], [3.0, 10.0]], dtype=torch.float64)

    assert global_mae(prediction, reference) == pytest.approx(10.0 / 4.0)


def test_global_mae_uses_float64_for_finite_float32_extremes() -> None:
    maximum = torch.finfo(torch.float32).max
    prediction = torch.tensor([maximum], dtype=torch.float32)
    reference = torch.tensor([-maximum], dtype=torch.float32)

    result = global_mae(prediction, reference)

    assert math.isfinite(result)
    assert result == pytest.approx(2.0 * float(maximum))


def test_evaluation_builds_equal_weight_ensemble_legacy_uq_and_global_mae() -> None:
    result = evaluate_prediction(
        _payload(), coverages=[1.0, 0.5], constant_tolerance=1e-12
    )

    assert isinstance(result, EvaluationArtifacts)
    torch.testing.assert_close(result.ensemble["energy"], torch.tensor([2.0, 2.0, 3.0]))
    assert set(result.ensemble) == {"energy", "forces", "stress"}
    assert set(result.uncertainty) == {
        "formula_version",
        "energy_total",
        "energy_per_atom",
        "force_component",
        "force_atom_vector",
        "force_structure",
    }
    assert "stress" not in result.uncertainty
    assert set(result.uncertainty["force_structure"]) == {"std"}
    assert set(result.uncertainty["force_structure"]["std"]) == {
        "mean",
        "max",
        "q95",
    }
    torch.testing.assert_close(
        result.uncertainty["energy_total"]["std"],
        torch.tensor([1.0, 1.0, 2.0]),
    )
    torch.testing.assert_close(
        result.uncertainty["energy_total"]["gmd"],
        torch.tensor([1.0, 1.0, 2.0]),
    )
    torch.testing.assert_close(
        result.uncertainty["energy_per_atom"]["std"],
        torch.tensor([0.5, 1.0, 2.0]),
    )
    assert result.metrics["schema_version"] == 4
    assert result.metrics["mae"] == pytest.approx(
        {
            "energy_total": 7.0 / 3.0,
            "energy_per_atom": 2.0,
            "force_component": 9.0 / 12.0,
            "stress_component": 1.0,
        }
    )
    assert result.metrics["counts"] == {
        "structures": 3,
        "atoms": 4,
        "force_components": 12,
        "stress_components": 27,
    }
    assert result.report_inputs["formula_version"] == "legacy_upet_fge_v1"
    assert result.report_inputs["metric_schema_version"] == 4

    assert set(result.metrics["correlations"]) == {
        "energy_total_std",
        "energy_total_gmd",
        "energy_per_atom_std",
        "energy_per_atom_gmd",
        "force_component_std",
        "force_component_gmd",
        "force_atom_vector_std",
        "force_atom_vector_gmd",
        "force_structure_mean_std",
        "force_structure_max_std",
        "force_structure_q95_std",
    }


def test_correlations_use_average_tied_ranks_deterministically() -> None:
    payload = _payload()
    payload["energy_prediction"] = torch.tensor(
        [[0.0, 1.0, 1.0], [4.0, 3.0, 5.0]], dtype=torch.float32
    )
    first = evaluate_prediction(payload, [1.0, 0.5], 1e-12)
    second = evaluate_prediction(payload, [1.0, 0.5], 1e-12)

    correlation = first.metrics["correlations"]["energy_per_atom_std"]
    assert correlation == second.metrics["correlations"]["energy_per_atom_std"]
    assert correlation["status"] == "ok"
    assert correlation["uncertainty_constant"] is False
    assert correlation["error_constant"] is False
    assert correlation["n"] == 3
    assert correlation["spearman"] == pytest.approx(math.sqrt(3.0) / 2.0)


def test_risk_coverage_keeps_lowest_uncertainty_with_global_risk() -> None:
    result = evaluate_prediction(_payload(), [1.0, 0.5], 1e-12)

    rows = result.metrics["risk_coverage"]["energy_per_atom_std"]
    assert rows == pytest.approx(
        [
            {"coverage": 1.0, "kept": 3, "risk": 2.0},
            {"coverage": 0.5, "kept": 2, "risk": 1.5},
        ]
    )


def test_constant_inputs_emit_json_safe_diagnostic_without_python_warning() -> None:
    payload = _payload()
    prediction = torch.tensor([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]], dtype=torch.float32)
    payload["energy_prediction"] = prediction

    result = evaluate_prediction(payload, [1.0], 1e-12)

    correlation = result.metrics["correlations"]["energy_total_std"]
    assert correlation == {
        "status": "undefined_constant_input",
        "pearson": None,
        "spearman": None,
        "uncertainty_constant": True,
        "error_constant": False,
        "n": 3,
    }


def test_evaluation_artifacts_are_frozen() -> None:
    result = evaluate_prediction(_payload(), [1.0], 1e-12)

    with pytest.raises(FrozenInstanceError):
        result.metrics = {}  # type: ignore[misc]


@pytest.mark.parametrize(
    "coverages",
    [[], [0.0], [1.01], [math.nan], [0.5, 1.0], [1.0, 1.0]],
)
def test_evaluation_rejects_invalid_coverages(coverages: list[float]) -> None:
    with pytest.raises(ValueError, match="coverages"):
        evaluate_prediction(_payload(), coverages, 1e-12)


@pytest.mark.parametrize("tolerance", [0.0, -1.0, math.nan, math.inf])
def test_evaluation_rejects_invalid_constant_tolerance(tolerance: float) -> None:
    with pytest.raises(ValueError, match="constant_tolerance"):
        evaluate_prediction(_payload(), [1.0], tolerance)


def test_evaluation_rejects_missing_consumed_payload_fields() -> None:
    missing = _payload()
    del missing["stress_reference"]
    with pytest.raises(ValueError, match="missing.*stress_reference"):
        evaluate_prediction(missing, [1.0], 1e-12)


def test_evaluation_accepts_representative_canonical_metadata_superset() -> None:
    baseline = evaluate_prediction(_payload(), [1.0, 0.5], 1e-12)
    canonical = _payload()
    canonical.update(
        {
            "structure_ids": ("structure_000", "structure_001", "structure_002"),
            "atomic_numbers": torch.tensor([1, 6, 8, 14], dtype=torch.int64),
            "structure_mapping": torch.tensor([0, 0, 1, 2], dtype=torch.int64),
            "target_names": {
                "energy": "energy",
                "forces": "forces",
                "stress": "stress",
            },
            "units": {
                "energy": "eV",
                "forces": "eV/angstrom",
                "stress": "eV/angstrom^3",
            },
            "statistics": {"K": 2, "S": 3, "A": 4},
        }
    )

    enriched = evaluate_prediction(canonical, [1.0, 0.5], 1e-12)

    assert enriched.metrics == baseline.metrics
    assert enriched.report_inputs == baseline.report_inputs
    for target in ("energy", "forces", "stress"):
        torch.testing.assert_close(enriched.ensemble[target], baseline.ensemble[target])


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("energy_prediction", torch.zeros((3, 2)), "K|energy_prediction"),
        ("forces_prediction", torch.zeros((2, 3, 3)), "atom|forces_prediction"),
        ("stress_prediction", torch.zeros((2, 3, 9)), "stress_prediction"),
        ("n_atoms", torch.tensor([2, 0, 2]), "n_atoms"),
        ("structure_offsets", torch.tensor([0, 1, 3, 4]), "offsets|n_atoms"),
        ("member_ids", ("member_001", "member_001"), "member_ids"),
    ],
)
def test_evaluation_rejects_shape_count_mapping_and_member_mismatches(
    field: str, value: object, message: str
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        evaluate_prediction(payload, [1.0], 1e-12)


def test_evaluation_rejects_non_finite_values() -> None:
    payload = _payload()
    reference = payload["forces_reference"]
    assert isinstance(reference, torch.Tensor)
    values = reference.clone()
    values[0, 0] = math.nan
    payload["forces_reference"] = values

    with pytest.raises(ValueError, match="finite"):
        evaluate_prediction(payload, [1.0], 1e-12)


def test_evaluation_rejects_non_finite_derived_uncertainty() -> None:
    payload = _payload()
    maximum = torch.finfo(torch.float32).max
    payload["energy_prediction"] = torch.tensor(
        [[-maximum, 0.0, 0.0], [maximum, 0.0, 0.0]], dtype=torch.float32
    )

    with pytest.raises(ValueError, match="derived.*finite"):
        evaluate_prediction(payload, [1.0], 1e-12)


def test_evaluation_keeps_finite_extreme_energy_mae() -> None:
    payload = _payload()
    maximum = torch.finfo(torch.float32).max
    payload["energy_prediction"] = torch.full((2, 3), maximum, dtype=torch.float32)
    payload["energy_reference"] = torch.full((3,), -maximum, dtype=torch.float32)

    result = evaluate_prediction(payload, [1.0], 1e-12)

    energy_total = result.metrics["mae"]["energy_total"]
    energy_per_atom = result.metrics["mae"]["energy_per_atom"]
    assert math.isfinite(energy_total)
    assert math.isfinite(energy_per_atom)
    assert energy_total == pytest.approx(2.0 * float(maximum))
    assert energy_per_atom == pytest.approx((5.0 / 3.0) * float(maximum))


def test_evaluate_fge_reopens_only_canonical_prediction_and_publishes_artifacts(
    tmp_path,
) -> None:
    """The formal stage must evaluate the stored prediction without model/data I/O."""
    from types import SimpleNamespace

    from Uncertainty_Quantification.FGE.fge.evaluation import evaluate_fge

    root = tmp_path / "upet_fge_n20_cpu"
    prediction = root / "prediction" / "test_raw.pt"
    prediction.parent.mkdir(parents=True)
    torch.save(_payload(), prediction)
    config = SimpleNamespace(
        project=SimpleNamespace(name="upet_fge_n20_cpu"),
        paths=SimpleNamespace(output_root=tmp_path),
        evaluation=SimpleNamespace(risk_coverages=(1.0, 0.5), constant_tolerance=1e-12),
    )

    directory = evaluate_fge(config)

    assert directory == root / "evaluation" / "legacy_equal_weight"
    assert (directory / "ensemble.pt").is_file()
    assert (directory / "uncertainty.pt").is_file()
    assert (directory / "metrics.json").is_file()
    assert (directory / "report.md").is_file()
