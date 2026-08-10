from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.prediction import (
    canonical_prediction,
    validate_prediction_payload,
)
from Uncertainty_Quantification.FGE.internal_migration.migration import converter

from .helpers import build_conversion_case


_IDENTITY = {"commit": "a" * 40, "dirty_sha256": "b" * 64}


def _unsorted_payload() -> dict[str, object]:
    return {
        "energy_prediction": torch.zeros((2, 2), dtype=torch.float32),
        "forces_prediction": torch.zeros((2, 2, 3), dtype=torch.float32),
        "stress_prediction": torch.zeros((2, 2, 3, 3), dtype=torch.float32),
        "energy_reference": torch.zeros(2, dtype=torch.float32),
        "forces_reference": torch.zeros((2, 3), dtype=torch.float32),
        "stress_reference": torch.zeros((2, 3, 3), dtype=torch.float32),
        "n_atoms": torch.ones(2, dtype=torch.int64),
        "structure_offsets": torch.tensor([0, 1, 2], dtype=torch.int64),
        "member_ids": ("member_001", "member_002"),
        "structure_ids": ("z", "a"),
        "atomic_numbers": torch.ones(2, dtype=torch.int64),
        "structure_mapping": torch.tensor([0, 1], dtype=torch.int64),
        "target_names": {"energy": "e", "forces": "f", "stress": "s"},
        "units": {"energy": "eV", "forces": "eV/A", "stress": "eV/A3"},
        "statistics": {"K": 2, "S": 2, "A": 2},
    }


def test_prediction_preserves_unique_unsorted_structure_order() -> None:
    payload = canonical_prediction(_unsorted_payload())

    assert validate_prediction_payload(payload).S == 2
    assert payload["structure_ids"] == ("z", "a")


def test_converter_reopens_completed_staging_and_uses_current_identity(
    legacy_tree, config_payload, tmp_path: Path, monkeypatch
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    real_validate = converter.validate_result
    modes: list[str] = []

    def recording_validate(*args, **kwargs):
        report = real_validate(*args, **kwargs)
        modes.append(report.mode)
        return report

    monkeypatch.setattr(converter, "validate_result", recording_validate)
    destination = tmp_path / "published"
    audit_root = tmp_path / "audit"
    converter.convert_legacy_run(
        source,
        destination,
        audit_root,
        config,
        base,
        expected=expected,
        code_identity=_IDENTITY,
    )

    assert modes == ["published", "read_only"]
    training = json.loads((destination / "training/manifest.json").read_text())
    prediction = json.loads((destination / "prediction/manifest.json").read_text())
    result = json.loads((destination / "result_manifest.json").read_text())
    audit = json.loads((audit_root / "published/audit.json").read_text())
    for document in (training, prediction, result, audit):
        assert document["artifact_writer_code_identity"] == _IDENTITY
        assert document["validator_code_identity"] == _IDENTITY
    assert training["training_code_identity"] == {"status": "unavailable"}


@pytest.mark.parametrize("kind", ["destination", "audit"])
def test_converter_rejects_symlink_ancestor_without_pollution(
    legacy_tree, config_payload, tmp_path: Path, kind: str
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    destination = linked / "published" if kind == "destination" else tmp_path / "out"
    audit_root = linked / "audit" if kind == "audit" else tmp_path / "audit"

    with pytest.raises(HardFailure, match="symbolic link"):
        converter.convert_legacy_run(
            source,
            destination,
            audit_root,
            config,
            base,
            expected=expected,
            code_identity=_IDENTITY,
        )
    assert not destination.exists()


@pytest.mark.parametrize("layout", ["inside", "same_parent", "ancestor"])
def test_converter_rejects_overlapping_audit_and_result_paths_without_pollution(
    legacy_tree, config_payload, tmp_path: Path, layout: str
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    destination = (
        tmp_path / "nested/published"
        if layout == "ancestor"
        else tmp_path / "published"
    )
    if layout == "inside":
        audit_root = destination / "audit"
    else:
        audit_root = tmp_path

    with pytest.raises(HardFailure, match="separate"):
        converter.convert_legacy_run(
            source,
            destination,
            audit_root,
            config,
            base,
            expected=expected,
            code_identity=_IDENTITY,
        )
    assert not destination.exists()


def test_converter_rejects_unavailable_migration_identity(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )

    with pytest.raises(HardFailure, match="code identity"):
        converter.convert_legacy_run(
            source,
            tmp_path / "published",
            tmp_path / "audit",
            config,
            base,
            expected=expected,
            code_identity={"status": "unavailable"},
        )
