from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.members import load_member
from Uncertainty_Quantification.FGE.fge.validation import validate_result
from Uncertainty_Quantification.FGE.internal_migration.migration.converter import (
    convert_legacy_run,
)

from .helpers import build_conversion_case


def test_converter_publishes_valid_canonical_result_and_audit(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    destination = tmp_path / "published"
    audit_root = tmp_path / "audit"

    result = convert_legacy_run(
        source, destination, audit_root, config, base, expected=expected
    )

    assert result == destination
    assert validate_result(config, destination).mode == "read_only"
    training = json.loads((destination / "training/manifest.json").read_text())
    assert training["training_code_identity"] == {"status": "unavailable"}
    names = tuple(f"node_last_layers.tensor_{index:02d}" for index in range(11)) + (
        "edge_last_layers.tensor_11",
    )
    first = load_member(
        destination / "training/members/member_001.pt",
        config.identity.base_checkpoint_sha256,
        names,
    )
    assert torch.equal(first.tensors[0].value, torch.tensor([1.0]))
    audit = json.loads((audit_root / "published/audit.json").read_text())
    assert audit["publication_authorized"] is True
    assert audit["expected_final_destination"] == str(destination.resolve())
    assert audit["source_hashes_before"] == audit["source_hashes_after"]
    assert [item["member_id"] for item in audit["source_to_a3"]] == [
        "member_001",
        "member_002",
    ]


def test_converter_rejects_non_readout_drift(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    member = source / "successful/members/member_002.ckpt"
    raw = torch.load(member, map_location="cpu", weights_only=True)
    raw["model_state_dict"]["frozen.weight"] += 1
    torch.save(raw, member)
    member_hashes = dict(expected.member_sha256)
    member_hashes["member_002"] = hashlib.sha256(member.read_bytes()).hexdigest()
    expected = type(expected)(**{**expected.__dict__, "member_sha256": member_hashes})

    with pytest.raises(HardFailure, match="non-readout"):
        convert_legacy_run(
            source,
            tmp_path / "published",
            tmp_path / "audit",
            config,
            base,
            expected=expected,
        )


def test_converter_refuses_existing_destination(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    destination = tmp_path / "published"
    destination.mkdir()

    with pytest.raises(HardFailure, match="already exists"):
        convert_legacy_run(
            source,
            destination,
            tmp_path / "audit",
            config,
            base,
            expected=expected,
        )
