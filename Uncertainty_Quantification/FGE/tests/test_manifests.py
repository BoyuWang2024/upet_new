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
            "schema_version": "upet.fge.v1",
            "project": {"name": "upet_fge_full", "method": "FGE", "backend": "upet"},
            "paths": {
                "base_checkpoint": {"role": "base_checkpoint", "sha256": "a" * 64},
                "train_data": {"role": "train_data", "sha256": "b" * 64},
                "val_data": {"role": "val_data", "sha256": "c" * 64},
                "test_data": {"role": "test_data", "sha256": "d" * 64},
                "output_root": {"role": "output_root"},
            },
            "identity": {
                "base_checkpoint_sha256": "a" * 64,
                "train_data_sha256": "b" * 64,
                "val_data_sha256": "c" * 64,
                "test_data_sha256": "d" * 64,
            },
            "data": {"format": "extxyz"},
            "training": {"mode": "readout_only_official_upet"},
            "fge": {"member_count": 2},
            "ema": {"member_source": "raw_endpoint"},
            "prediction": {"split": "test"},
            "evaluation": {"formula_version": "legacy_upet_fge_v1"},
            "scientific": {"training": {}, "evaluation": {}},
        },
        "config_identity": {"sha256": "e" * 64},
        "checkpoint_identity": {"sha256": "a" * 64},
        "data_identities": {
            "train": {"sha256": "b" * 64},
            "val": {"sha256": "c" * 64},
            "test": {"sha256": "d" * 64},
        },
        "model_contract": {
            "readout_tensor_count": 12,
            "readout_parameter_count": 13338,
        },
        "frozen_fingerprint_identity": {"sha256": "c" * 64},
        "dependency_snapshot": {"torch": "2.11.0", "metatrain": "2026.3.1"},
        "scientific_flags": {
            "path_feasibility_only": True,
            "split_leakage": True,
            "scientific_evaluation": False,
            "inference_only": False,
        },
        "artifact_writer_code_identity": _identity("1" * 40),
        "validator_code_identity": _identity("2" * 40),
        "member_count": 2,
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
        config_identity=cast(Mapping[str, object], values["config_identity"]),
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
        member_count=cast(int, values["member_count"]),
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


def test_training_manifest_rejects_generic_config_and_source_provenance() -> None:
    """Formal training provenance requires the canonical config identity tree."""
    values = _training_inputs()
    values["config_resolved"] = {"source_path": "legacy/run"}

    with pytest.raises(HardFailure):
        _build_training(values, _identity("5" * 40))


def test_result_manifest_ignores_nonformal_runtime_files(tmp_path: Path) -> None:
    """Only named canonical artifact roles enter the completion inventory."""
    (tmp_path / "_work").mkdir()
    (tmp_path / "_work" / "resume.pt").write_bytes(b"work")
    (tmp_path / ".scratch").write_text("temporary\n", encoding="utf-8")
    (tmp_path / "config_resolved.yaml").write_text("project: x\n", encoding="utf-8")
    (tmp_path / "validation.json").write_text('{"status":"PASS"}\n', encoding="utf-8")

    manifest = build_result_manifest(
        root=tmp_path,
        project_name="upet_fge_full",
        artifact_writer_code_identity=_identity("1" * 40),
        validator_code_identity=_identity("2" * 40),
    )

    assert [artifact["path"] for artifact in _artifacts(manifest)] == [
        "config_resolved.yaml",
        "validation.json",
    ]


def test_training_manifest_emits_explicit_config_and_member_bindings() -> None:
    """Canonical training provenance binds its config SHA and exact K."""
    values = _training_inputs()

    manifest = _build_training(values, _identity("5" * 40))

    assert manifest["config_identity"] == {"sha256": "e" * 64}
    assert manifest["member_count"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config_identity", {}),
        ("config_identity", {"sha256": "e" * 64, "unexpected": True}),
        ("member_count", 1),
        (
            "config_resolved",
            {
                "paths": {
                    "base_checkpoint": {
                        "role": "base_checkpoint",
                        "sha256": "a" * 64,
                    },
                    "train_data": {"role": "train_data", "sha256": "b" * 64},
                    "val_data": {"role": "val_data", "sha256": "c" * 64},
                }
            },
        ),
        (
            "data_identities",
            {
                "train": {"sha256": "b" * 64},
                "test": {"sha256": "d" * 64},
            },
        ),
        (
            "scientific_flags",
            {
                "path_feasibility_only": True,
                "split_leakage": True,
                "scientific_evaluation": False,
            },
        ),
        (
            "dependency_snapshot",
            {
                "torch": "2.11.0",
                "metatrain": "2026.3.1",
                "unexpected": "1",
            },
        ),
    ],
)
def test_training_manifest_rejects_incomplete_or_noncanonical_contract_subtrees(
    field: str, value: object
) -> None:
    """Every provenance subtree has one exact, complete canonical shape."""
    values = _training_inputs()
    values[field] = value

    with pytest.raises(HardFailure):
        _build_training(values, _identity("5" * 40))


@pytest.mark.parametrize(
    ("path_role", "identity_role"),
    [
        ("base_checkpoint", "checkpoint"),
        ("train_data", "train"),
        ("val_data", "val"),
        ("test_data", "test"),
    ],
)
def test_training_manifest_rejects_path_identity_sha_cross_binding_mismatch(
    path_role: str, identity_role: str
) -> None:
    """Sanitized logical paths must bind to their declared SHA identities."""
    values = _training_inputs()
    config = cast(dict[str, object], values["config_resolved"])
    paths = cast(dict[str, object], config["paths"])
    path_identity = cast(dict[str, object], paths[path_role])
    path_identity["sha256"] = "f" * 64
    if identity_role == "checkpoint":
        assert path_role == "base_checkpoint"
    else:
        data_identities = cast(dict[str, object], values["data_identities"])
        assert (
            path_identity["sha256"]
            != cast(dict[str, object], data_identities[identity_role])["sha256"]
        )

    with pytest.raises(HardFailure):
        _build_training(values, _identity("5" * 40))


def test_result_manifest_inventories_the_complete_canonical_formal_tree(
    tmp_path: Path,
) -> None:
    """Default inventory includes every formal artifact and no runtime residue."""
    formal_paths = [
        "config_resolved.yaml",
        "preflight/train.json",
        "preflight/predict.json",
        "preflight/evaluate.json",
        "training/manifest.json",
        "training/members/member_001.pt",
        "prediction/manifest.json",
        "prediction/test_raw.pt",
        "evaluation/legacy_equal_weight/ensemble.pt",
        "evaluation/legacy_equal_weight/uncertainty.pt",
        "evaluation/legacy_equal_weight/metrics.json",
        "evaluation/legacy_equal_weight/report.md",
        "validation.json",
    ]
    for relative in formal_paths:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode("utf-8"))
    (tmp_path / "result_manifest.json").write_text("completion\n", encoding="utf-8")
    (tmp_path / "_work").mkdir()
    (tmp_path / "_work" / "resume.pt").write_bytes(b"resume")
    (tmp_path / ".scratch").write_text("temporary\n", encoding="utf-8")

    manifest = build_result_manifest(
        root=tmp_path,
        project_name="upet_fge_full",
        artifact_writer_code_identity=_identity("1" * 40),
        validator_code_identity=_identity("2" * 40),
    )

    assert [artifact["path"] for artifact in _artifacts(manifest)] == sorted(
        formal_paths
    )


def test_training_manifest_accepts_only_the_complete_sanitized_config() -> None:
    """A truncated config must not bypass formal runtime identity binding."""
    values = _training_inputs()
    values["config_resolved"] = {
        "schema_version": "upet.fge.v1",
        "project": {"name": "upet_fge_full", "method": "FGE", "backend": "upet"},
        "paths": {
            "base_checkpoint": {"role": "base_checkpoint", "sha256": "a" * 64},
            "train_data": {"role": "train_data", "sha256": "b" * 64},
            "val_data": {"role": "val_data", "sha256": "c" * 64},
            "test_data": {"role": "test_data", "sha256": "d" * 64},
            "output_root": {"role": "output_root"},
        },
        "identity": {
            "base_checkpoint_sha256": "a" * 64,
            "train_data_sha256": "b" * 64,
            "val_data_sha256": "c" * 64,
            "test_data_sha256": "d" * 64,
        },
        "data": {"format": "extxyz"},
        "training": {"mode": "readout_only_official_upet"},
        "fge": {"member_count": 2},
        "ema": {"member_source": "raw_endpoint"},
        "prediction": {"split": "test"},
        "evaluation": {"formula_version": "legacy_upet_fge_v1"},
        "scientific": {"training": {}, "evaluation": {}},
    }

    manifest = _build_training(values, _identity("5" * 40))

    assert manifest["config_resolved"] == values["config_resolved"]

    values["config_resolved"] = {"paths": {}}
    with pytest.raises(HardFailure):
        _build_training(values, _identity("5" * 40))
