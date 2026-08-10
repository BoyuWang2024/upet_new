"""Normalized schemas are invariant to concrete K, S, and A sizes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

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


def test_schema_signature_includes_completion_roles_not_instance_hashes(
    tmp_path: Path,
) -> None:
    """Completion is structural: role/path references matter, hashes do not."""
    from Uncertainty_Quantification.FGE.fge.validation import (
        schema_signature,
        validate_result,
    )

    config, root = make_canonical_result(tmp_path)
    validate_result(config, root)
    signature = schema_signature(root)

    documents = signature["documents"]
    assert isinstance(documents, dict)
    completion = documents["result_manifest.json"]
    assert isinstance(completion, dict)
    artifacts = completion["artifacts"]
    assert isinstance(artifacts, list)
    assert {"role": "config_resolved", "path": "config_resolved.yaml"} in artifacts
    assert "sha256" not in completion
    assert "bytes" not in completion


def test_schema_signature_rejects_a3_member_schema_drift(
    tmp_path: Path,
) -> None:
    """All members must expose the same formal 12-tensor A3 wire schema."""
    from Uncertainty_Quantification.FGE.fge.errors import HardFailure
    from Uncertainty_Quantification.FGE.fge.validation import schema_signature

    _, root = make_canonical_result(tmp_path)
    member = root / "training" / "members" / "member_002.pt"
    payload = torch.load(member, weights_only=True)
    assert isinstance(payload, dict)
    tensors = payload["tensors"]
    assert isinstance(tensors, list)
    assert isinstance(tensors[0], dict)
    tensors[0]["name"] = "node_last_layers.drift"
    torch.save(payload, member)

    with pytest.raises(HardFailure):
        schema_signature(root)


def test_schema_signature_keeps_formal_versions_and_fixed_a3_dimensions(
    tmp_path: Path,
) -> None:
    """Versions are semantic constants; A3 shapes are not dataset dimensions."""
    from Uncertainty_Quantification.FGE.fge.validation import schema_signature

    _, root = make_canonical_result(tmp_path)
    for name in ("member_001.pt", "member_002.pt"):
        member = root / "training" / "members" / name
        payload = torch.load(member, weights_only=True)
        assert isinstance(payload, dict)
        tensors = payload["tensors"]
        assert isinstance(tensors, list)
        assert isinstance(tensors[0], dict)
        tensors[0]["shape"] = [2]
        tensors[0]["value"] = torch.zeros((2,), dtype=torch.float32)
        torch.save(payload, member)

    signature = schema_signature(root)
    documents = signature["documents"]
    tensors_signature = signature["tensors"]
    assert isinstance(documents, dict)
    assert isinstance(tensors_signature, dict)
    training = documents["training/manifest.json"]
    uncertainty = tensors_signature["evaluation/legacy_equal_weight/uncertainty.pt"]
    a3 = tensors_signature["training/members/member_NNN.pt"]
    assert isinstance(training, dict)
    assert isinstance(uncertainty, dict)
    assert isinstance(a3, dict)
    assert training["schema_version"] == "upet.fge.training.v1"
    assert uncertainty["formula_version"] == "legacy_upet_fge_v1"
    entries = a3["tensors"]
    assert isinstance(entries, list)
    assert isinstance(entries[0], dict)
    assert entries[0]["shape"] == [2]
