"""Normalized schemas are invariant to concrete K, S, and A sizes."""

from __future__ import annotations

import json
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


def test_schema_signature_distinguishes_result_manifest_schema_versions(
    tmp_path: Path,
) -> None:
    """Completion schema version is a semantic contract, not just a string type."""
    from Uncertainty_Quantification.FGE.fge.validation import (
        schema_signature,
        validate_result,
    )

    config, root = make_canonical_result(tmp_path)
    validate_result(config, root)
    baseline = schema_signature(root)
    manifest_path = root / "result_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "upet.fge.result.v999"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert schema_signature(root) != baseline


def test_evaluation_signature_symbolizes_dataset_dimensions() -> None:
    """Evaluation tensors use symbolic S/A dimensions, not instance sizes."""
    from Uncertainty_Quantification.FGE.fge.validation import (
        _evaluation_tensor_signature,
    )

    ensemble = {
        "energy": torch.zeros(20),
        "forces": torch.zeros((143, 3)),
        "stress": torch.zeros((20, 3, 3)),
    }

    signature = _evaluation_tensor_signature(
        ensemble, "evaluation/legacy_equal_weight/ensemble.pt", 2, 20, 143
    )

    energy_signature = signature["energy"]
    forces_signature = signature["forces"]
    stress_signature = signature["stress"]
    assert isinstance(energy_signature, dict)
    assert isinstance(forces_signature, dict)
    assert isinstance(stress_signature, dict)
    assert energy_signature["shape"] == ["S"]
    assert forces_signature["shape"] == ["A", 3]
    assert stress_signature["shape"] == ["S", 3, 3]

    uncertainty = {
        "energy_total": {"std": torch.zeros(20), "gmd": torch.zeros(20)},
        "energy_per_atom": {"std": torch.zeros(20), "gmd": torch.zeros(20)},
        "force_component": {
            "std": torch.zeros((143, 3)),
            "gmd": torch.zeros((143, 3)),
        },
        "force_atom_vector": {"std": torch.zeros(143), "gmd": torch.zeros(143)},
        "force_structure": {
            "std": {
                "mean": torch.zeros(20),
                "max": torch.zeros(20),
                "q95": torch.zeros(20),
            }
        },
    }
    nested = _evaluation_tensor_signature(
        uncertainty,
        "evaluation/legacy_equal_weight/uncertainty.pt",
        2,
        20,
        143,
    )

    energy_total = nested["energy_total"]
    force_component = nested["force_component"]
    force_structure = nested["force_structure"]
    assert isinstance(energy_total, dict)
    assert isinstance(force_component, dict)
    assert isinstance(force_structure, dict)
    energy_std = energy_total["std"]
    force_gmd = force_component["gmd"]
    force_structure_std = force_structure["std"]
    assert isinstance(energy_std, dict)
    assert isinstance(force_gmd, dict)
    assert isinstance(force_structure_std, dict)
    force_q95 = force_structure_std["q95"]
    assert isinstance(force_q95, dict)
    assert energy_std["shape"] == ["S"]
    assert force_gmd["shape"] == ["A", 3]
    assert force_q95["shape"] == ["S"]


def test_document_signature_ignores_values_and_code_identity_availability() -> None:
    """Normalized JSON/YAML signatures retain key trees, not run identities."""
    from Uncertainty_Quantification.FGE.fge.validation import _key_tree

    available = {
        "metric": 1.0,
        "artifact_writer_code_identity": {
            "commit": "a" * 40,
            "dirty_sha256": "b" * 64,
        },
    }
    unavailable = {
        "metric": None,
        "artifact_writer_code_identity": {"status": "unavailable"},
    }

    assert _key_tree(available) == _key_tree(unavailable)
