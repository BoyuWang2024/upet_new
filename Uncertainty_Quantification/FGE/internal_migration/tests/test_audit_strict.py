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


_IDENTITY = {"commit": "a" * 40, "dirty_sha256": "b" * 64}


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
        code_identity=_IDENTITY,
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
