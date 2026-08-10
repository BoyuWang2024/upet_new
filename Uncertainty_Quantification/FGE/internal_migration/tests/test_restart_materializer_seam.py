from __future__ import annotations

import hashlib
import inspect
from dataclasses import replace
from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge.members import load_member
from Uncertainty_Quantification.FGE.internal_migration.migration import converter

from .helpers import build_conversion_case


def test_converter_uses_authenticated_restart_materializer_for_all_states(
    legacy_tree, config_payload, tmp_path: Path, monkeypatch
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    raw = torch.load(base, map_location="cpu", weights_only=True)
    raw["model_state_dict"]["raw_schema_only"] = torch.tensor([99.0])
    torch.save(raw, base)
    base_sha = hashlib.sha256(base.read_bytes()).hexdigest()
    config = replace(
        config,
        identity=replace(config.identity, base_checkpoint_sha256=base_sha),
    )
    calls: list[tuple[str, str]] = []

    def materialized(path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
        calls.append((path.name, expected_sha256))
        state = torch.load(path, map_location="cpu", weights_only=True)[
            "model_state_dict"
        ]
        return {
            name: value for name, value in state.items() if name != "raw_schema_only"
        }

    assert (
        "_restart_materializer"
        not in inspect.signature(converter.convert_legacy_run).parameters
    )
    monkeypatch.setattr(converter, "_restart_materialized_state", materialized)
    destination = tmp_path / "published"
    converter.convert_legacy_run(
        source,
        destination,
        tmp_path / "audit",
        config,
        base,
        expected=expected,
    )

    assert [name for name, _ in calls] == [
        "base.ckpt",
        "member_001.ckpt",
        "member_002.ckpt",
    ]
    assert calls[0][1] == base_sha
    assert [digest for _, digest in calls[1:]] == [
        expected.member_sha256["member_001"],
        expected.member_sha256["member_002"],
    ]
    names = tuple(f"node_last_layers.tensor_{index:02d}" for index in range(11)) + (
        "edge_last_layers.tensor_11",
    )
    first = load_member(
        destination / "training/members/member_001.pt",
        base_sha,
        names,
    )
    assert torch.equal(first.tensors[0].value, torch.tensor([1.0]))
