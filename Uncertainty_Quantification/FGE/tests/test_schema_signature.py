"""Normalized schemas are invariant to concrete K, S, and A sizes."""

from __future__ import annotations

from pathlib import Path

from Uncertainty_Quantification.FGE.tests.test_validation import make_canonical_result


def _prediction_field_shape(signature: dict[str, object], field: str) -> object:
    tensors = signature["tensors"]
    assert isinstance(tensors, dict)
    prediction = tensors["prediction/test_raw.pt"]
    assert isinstance(prediction, dict)
    field_signature = prediction[field]
    assert isinstance(field_signature, dict)
    return field_signature["shape"]


def test_native_n20_and_migrated_full_results_have_the_same_symbolic_schema(
    tmp_path: Path,
) -> None:
    """K=2 and K=8 are abstracted while artifact relationships stay visible."""
    from Uncertainty_Quantification.FGE.fge.validation import schema_signature

    _, n20 = make_canonical_result(tmp_path / "n20", member_count=2)
    _, migrated = make_canonical_result(
        tmp_path / "migrated", project_name="upet_fge_full", member_count=8
    )

    n20_signature = schema_signature(n20)
    migrated_signature = schema_signature(migrated)

    assert n20_signature == migrated_signature
    assert _prediction_field_shape(n20_signature, "energy_prediction") == ["K", "S"]
    assert _prediction_field_shape(n20_signature, "forces_prediction") == ["K", "A", 3]
