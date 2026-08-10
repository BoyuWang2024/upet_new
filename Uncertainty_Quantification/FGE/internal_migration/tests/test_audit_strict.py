from __future__ import annotations

import json
from pathlib import Path

import pytest

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.internal_migration.migration.converter import (
    convert_legacy_run,
)
from Uncertainty_Quantification.FGE.internal_migration.scripts import audit_results

from .helpers import build_conversion_case


def _converted(legacy_tree, config_payload, tmp_path: Path):
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    destination = tmp_path / "published"
    audit_root = tmp_path / "audits"
    convert_legacy_run(
        source,
        destination,
        audit_root,
        config,
        base,
        expected=expected,
    )
    return source, destination, audit_root


def test_audit_reopens_source_a3_and_completed_destination(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    _, destination, audit_root = _converted(legacy_tree, config_payload, tmp_path)

    audit_results.audit(destination, audit_root)


@pytest.mark.parametrize(
    "mutation",
    ["unknown_key", "destination", "signature", "mapping", "identity"],
)
def test_audit_rejects_schema_or_binding_drift(
    legacy_tree, config_payload, tmp_path: Path, mutation: str
) -> None:
    _, destination, audit_root = _converted(legacy_tree, config_payload, tmp_path)
    path = audit_root / "published/audit.json"
    document = json.loads(path.read_text())
    if mutation == "unknown_key":
        document["extra"] = True
    elif mutation == "destination":
        document["expected_final_destination"] = str(tmp_path / "other")
    elif mutation == "signature":
        document["canonical_staging_signature"] = {}
    elif mutation == "mapping":
        document["source_to_a3"][0]["a3_sha256"] = "0" * 64
    else:
        document["validator_code_identity"] = {
            "commit": "c" * 40,
            "dirty_sha256": "d" * 64,
        }
    path.write_text(json.dumps(document) + "\n")

    with pytest.raises(HardFailure):
        audit_results.audit(destination, audit_root)


def test_audit_rejects_destination_artifact_drift(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    _, destination, audit_root = _converted(legacy_tree, config_payload, tmp_path)
    metrics = destination / "evaluation/legacy_equal_weight/metrics.json"
    metrics.write_text(metrics.read_text() + " ")

    with pytest.raises(HardFailure, match="manifest|artifact|SHA"):
        audit_results.audit(destination, audit_root)


def test_audit_rejects_source_file_replaced_by_symlink(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    source, destination, audit_root = _converted(legacy_tree, config_payload, tmp_path)
    audit = json.loads((audit_root / "published/audit.json").read_text())
    relative = next(iter(audit["source_hashes_before"]))
    original = source / relative
    replacement = tmp_path / "same-bytes"
    replacement.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(replacement)

    with pytest.raises(HardFailure, match="symbolic link|source"):
        audit_results.audit(destination, audit_root)


@pytest.mark.parametrize("mutation", ["truncate", "duplicate", "extra", "wrong_path"])
def test_audit_mapping_exactly_matches_validated_training_members(
    legacy_tree, config_payload, tmp_path: Path, mutation: str
) -> None:
    _, destination, audit_root = _converted(legacy_tree, config_payload, tmp_path)
    path = audit_root / "published/audit.json"
    document = json.loads(path.read_text())
    mapping = document["source_to_a3"]
    if mutation == "truncate":
        mapping.pop()
    elif mutation == "duplicate":
        mapping[1] = dict(mapping[0])
    elif mutation == "extra":
        extra = dict(mapping[-1])
        extra["member_id"] = "member_003"
        extra["a3_path"] = "training/members/member_003.pt"
        mapping.append(extra)
    else:
        mapping[0]["a3_path"] = mapping[1]["a3_path"]
        mapping[0]["a3_sha256"] = mapping[1]["a3_sha256"]
    path.write_text(json.dumps(document) + "\n")

    with pytest.raises(HardFailure):
        audit_results.audit(destination, audit_root)


@pytest.mark.parametrize("layout", ["equal", "audit_ancestor", "destination_ancestor"])
def test_standalone_audit_rejects_containment_before_reading(
    tmp_path: Path, layout: str
) -> None:
    if layout == "equal":
        destination = audit_root = tmp_path / "same"
    elif layout == "audit_ancestor":
        audit_root = tmp_path / "outer"
        destination = audit_root / "result"
    else:
        destination = tmp_path / "outer"
        audit_root = destination / "audit"

    with pytest.raises(HardFailure, match="separate"):
        audit_results.audit(destination, audit_root)


@pytest.mark.parametrize("mutation", ["swap", "duplicate"])
def test_audit_binds_each_member_to_its_unique_legacy_checkpoint(
    legacy_tree, config_payload, tmp_path: Path, mutation: str
) -> None:
    _, destination, audit_root = _converted(legacy_tree, config_payload, tmp_path)
    path = audit_root / "published/audit.json"
    document = json.loads(path.read_text())
    first, second = document["source_to_a3"]
    if mutation == "swap":
        first["source_checkpoint"], second["source_checkpoint"] = (
            second["source_checkpoint"],
            first["source_checkpoint"],
        )
        first["source_sha256"], second["source_sha256"] = (
            second["source_sha256"],
            first["source_sha256"],
        )
    else:
        second["source_checkpoint"] = first["source_checkpoint"]
        second["source_sha256"] = first["source_sha256"]
    path.write_text(json.dumps(document) + "\n")

    with pytest.raises(HardFailure, match="source checkpoint"):
        audit_results.audit(destination, audit_root)
