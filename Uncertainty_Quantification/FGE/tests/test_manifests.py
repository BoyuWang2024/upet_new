"""Tests for source-independent, strict FGE manifest builders."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

import pytest

from Uncertainty_Quantification.FGE.fge import (
    HardFailure,
    build_prediction_manifest,
    build_result_manifest,
    build_training_manifest,
)


def _identity(commit: str) -> dict[str, object]:
    return {"commit": commit, "dirty_sha256": "d" * 64}


def _training_inputs() -> dict[str, object]:
    return {
        "project_name": "upet_fge_full",
        "config_resolved": {
            "paths": {
                "base_checkpoint": {"role": "base_checkpoint", "sha256": "a" * 64},
                "test_data": {"role": "test_data", "sha256": "b" * 64},
            }
        },
        "checkpoint_identity": {"sha256": "a" * 64},
        "data_identities": {"test": {"sha256": "b" * 64}},
        "model_contract": {
            "readout_tensor_count": 12,
            "readout_parameter_count": 13338,
        },
        "frozen_fingerprint_identity": {"sha256": "c" * 64},
        "dependency_snapshot": {"torch": "2.11.0", "metatrain": "2026.3.1"},
        "scientific_flags": {"path_feasibility_only": True, "split_leakage": True},
        "artifact_writer_code_identity": _identity("1" * 40),
        "validator_code_identity": _identity("2" * 40),
        "members": [
            {
                "member_id": "member_001",
                "sha256": "3" * 64,
                "cycle": 1,
                "endpoint_global_step": 174384,
            },
            {
                "member_id": "member_002",
                "sha256": "4" * 64,
                "cycle": 2,
                "endpoint_global_step": 348768,
            },
        ],
    }


def _build_training(
    values: Mapping[str, object], training_code_identity: Mapping[str, object]
) -> dict[str, object]:
    return build_training_manifest(
        project_name=cast(str, values["project_name"]),
        config_resolved=cast(Mapping[str, object], values["config_resolved"]),
        checkpoint_identity=cast(Mapping[str, object], values["checkpoint_identity"]),
        data_identities=cast(Mapping[str, object], values["data_identities"]),
        model_contract=cast(Mapping[str, object], values["model_contract"]),
        frozen_fingerprint_identity=cast(
            Mapping[str, object], values["frozen_fingerprint_identity"]
        ),
        dependency_snapshot=cast(Mapping[str, object], values["dependency_snapshot"]),
        scientific_flags=cast(Mapping[str, object], values["scientific_flags"]),
        artifact_writer_code_identity=cast(
            Mapping[str, object], values["artifact_writer_code_identity"]
        ),
        validator_code_identity=cast(
            Mapping[str, object], values["validator_code_identity"]
        ),
        training_code_identity=training_code_identity,
        members=cast(Sequence[Mapping[str, object]], values["members"]),
    )


def _mapping(value: object) -> Mapping[str, object]:
    assert isinstance(value, Mapping)
    return value


def _artifacts(manifest: Mapping[str, object]) -> list[Mapping[str, object]]:
    value = manifest["artifacts"]
    assert isinstance(value, list)
    assert all(isinstance(item, Mapping) for item in value)
    return [cast(Mapping[str, object], item) for item in value]


def test_native_and_migrated_training_manifests_share_one_key_tree() -> None:
    """Migration may mark training code unavailable but must not fork the schema."""
    values = _training_inputs()
    native = _build_training(values, _identity("5" * 40))
    migrated = _build_training(values, {"status": "unavailable"})

    assert native.keys() == migrated.keys()
    assert native["training_code_identity"] == _identity("5" * 40)
    assert migrated["training_code_identity"] == {"status": "unavailable"}
    assert native["members"] == [
        {
            "member_id": "member_001",
            "sha256": "3" * 64,
            "cycle": 1,
            "endpoint_global_step": 174384,
        },
        {
            "member_id": "member_002",
            "sha256": "4" * 64,
            "cycle": 2,
            "endpoint_global_step": 348768,
        },
    ]
    assert native["model_contract"] == {
        "readout_tensor_count": 12,
        "readout_parameter_count": 13338,
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("artifact_writer_code_identity", {"status": "unavailable"}),
        ("validator_code_identity", {"status": "unavailable"}),
        ("dependency_snapshot", {"torch": math.nan}),
        ("config_resolved", {"path": "/absolute/forbidden"}),
        ("members", [{"member_id": "member_001", "sha256": "3" * 64}]),
    ],
)
def test_training_manifest_rejects_missing_required_identity_or_unsafe_json(
    field: str, value: object
) -> None:
    """Formal provenance is complete, finite JSON without absolute paths."""
    values = _training_inputs()
    values[field] = value

    with pytest.raises(HardFailure):
        _build_training(values, _identity("5" * 40))


def test_prediction_manifest_carries_shape_member_order_and_hashed_artifact(
    tmp_path: Path,
) -> None:
    """The prediction manifest binds a canonical tensor artifact to its members."""
    prediction = tmp_path / "prediction" / "test_raw.pt"
    prediction.parent.mkdir()
    prediction.write_bytes(b"canonical payload")

    manifest = build_prediction_manifest(
        root=tmp_path,
        prediction_path=prediction,
        member_ids=("member_001", "member_002"),
        shape={"K": 2, "S": 2, "A": 3},
        artifact_writer_code_identity=_identity("1" * 40),
        validator_code_identity=_identity("2" * 40),
    )

    assert manifest["member_ids"] == ["member_001", "member_002"]
    assert manifest["shape"] == {"K": 2, "S": 2, "A": 3}
    assert _mapping(manifest["artifact"])["path"] == "prediction/test_raw.pt"
    assert _mapping(manifest["artifact"])["sha256"] == (
        "4a64e29359c7d3f9be9aa5118f928b72226ff181b3123da6cea94b4ef8a1d993"
    )


def test_result_manifest_hashes_existing_formal_artifacts_and_excludes_itself(
    tmp_path: Path,
) -> None:
    """The completion marker describes prior formal files, never itself."""
    config = tmp_path / "config_resolved.yaml"
    prediction = tmp_path / "prediction" / "test_raw.pt"
    validation = tmp_path / "validation.json"
    result = tmp_path / "result_manifest.json"
    prediction.parent.mkdir()
    config.write_text("project: upet_fge_full\n", encoding="utf-8")
    prediction.write_bytes(b"prediction")
    validation.write_text('{"status":"PASS"}\n', encoding="utf-8")
    result.write_text("must not be listed\n", encoding="utf-8")

    manifest = build_result_manifest(
        root=tmp_path,
        project_name="upet_fge_full",
        artifact_writer_code_identity=_identity("1" * 40),
        validator_code_identity=_identity("2" * 40),
    )

    artifacts = _artifacts(manifest)
    assert [artifact["path"] for artifact in artifacts] == [
        "config_resolved.yaml",
        "prediction/test_raw.pt",
        "validation.json",
    ]
    assert [artifact["role"] for artifact in artifacts] == [
        "config_resolved",
        "prediction",
        "validation",
    ]
    assert all(artifact["path"] != "result_manifest.json" for artifact in artifacts)


def test_result_manifest_rejects_absolute_or_escaping_formal_paths(
    tmp_path: Path,
) -> None:
    """An inventory must never bless a path outside its canonical experiment root."""
    outside = tmp_path.parent / "outside.json"
    outside.write_text("{}\n", encoding="utf-8")

    with pytest.raises(HardFailure):
        build_result_manifest(
            root=tmp_path,
            project_name="upet_fge_full",
            artifact_writer_code_identity=_identity("1" * 40),
            validator_code_identity=_identity("2" * 40),
            formal_artifacts={"escape": outside},
        )
