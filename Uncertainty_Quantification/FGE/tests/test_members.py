from __future__ import annotations

import copy
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure


BASE_SHA = "a" * 64
OTHER_SHA = "b" * 64


def test_task_three_interfaces_are_publicly_exported() -> None:
    from Uncertainty_Quantification.FGE import fge

    expected = {
        "CheckpointBundle",
        "LossContract",
        "MemberPayload",
        "ReadoutAudit",
        "TensorFingerprint",
        "apply_member",
        "assert_frozen_unchanged",
        "assert_readout_contract",
        "frozen_fingerprint",
        "load_checkpoint_bundle",
        "load_member",
        "pack_member",
        "readout_tensor_names",
        "recover_loss_contract",
    }
    assert expected <= set(fge.__all__)
    assert all(hasattr(fge, name) for name in expected)


class TinyPET(torch.nn.Module):
    """Twelve readout tensors totaling the formal 13,338 scalars."""

    def __init__(self) -> None:
        super().__init__()
        self.frozen_weight = torch.nn.Parameter(
            torch.tensor([2.0, 3.0]), requires_grad=False
        )
        self.node_last_layers = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.full((1,), float(index))) for index in range(6)]
        )
        self.edge_last_layers = torch.nn.ParameterList(
            [
                *[
                    torch.nn.Parameter(torch.full((1,), float(index + 6)))
                    for index in range(5)
                ],
                torch.nn.Parameter(torch.full((13_327,), 11.0)),
            ]
        )
        self.register_buffer("normalization", torch.tensor([4.0, 5.0]))


def _base_state(model: torch.nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (name, tensor.detach().cpu().clone())
        for name, tensor in model.state_dict().items()
    )


def _member_model(value: float) -> TinyPET:
    model = TinyPET()
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith(("node_last_layers.", "edge_last_layers.")):
                parameter.fill_(value)
    return model


def _saved_payload(value: float = 7.0) -> dict[str, object]:
    from Uncertainty_Quantification.FGE.fge.members import pack_member

    return pack_member(_member_model(value), 1, 1, 40, BASE_SHA)


def _load_payload(payload: dict[str, object], names: tuple[str, ...]):
    from Uncertainty_Quantification.FGE.fge import members

    original_load = torch.load
    try:
        torch.load = lambda *args, **kwargs: payload  # type: ignore[assignment]
        return members.load_member(Path("literal.pt"), BASE_SHA, names)
    finally:
        torch.load = original_load


def test_readout_names_and_audit_are_exact_and_deterministic() -> None:
    from Uncertainty_Quantification.FGE.fge.members import (
        assert_readout_contract,
        readout_tensor_names,
    )

    model = TinyPET()
    names = readout_tensor_names(model)
    audit = assert_readout_contract(model)
    assert names == tuple(
        [f"node_last_layers.{index}" for index in range(6)]
        + [f"edge_last_layers.{index}" for index in range(6)]
    )
    assert audit.names == names
    assert audit.tensor_count == 12
    assert audit.scalar_count == 13_338
    assert audit.shapes == ((1,),) * 11 + ((13_327,),)
    assert audit.dtypes == ("torch.float32",) * 12
    assert audit.all_finite is True


def test_readout_contract_rejects_wrong_counts_and_outside_trainable() -> None:
    from Uncertainty_Quantification.FGE.fge.members import assert_readout_contract

    missing = TinyPET()
    missing.edge_last_layers = torch.nn.ParameterList(
        list(missing.edge_last_layers)[:-1]
    )
    with pytest.raises(HardFailure, match="12 tensors"):
        assert_readout_contract(missing)
    wrong = TinyPET()
    wrong.edge_last_layers[-1] = torch.nn.Parameter(torch.zeros(13_326))
    with pytest.raises(HardFailure, match="13,338"):
        assert_readout_contract(wrong)
    outside = TinyPET()
    outside.frozen_weight.requires_grad_(True)
    with pytest.raises(HardFailure, match="unexpected trainable"):
        assert_readout_contract(outside)


def test_readout_contract_rejects_nonfloating_and_nonfinite() -> None:
    from Uncertainty_Quantification.FGE.fge.members import assert_readout_contract

    nonfloating = TinyPET()
    nonfloating.node_last_layers[0] = torch.nn.Parameter(
        torch.ones(1, dtype=torch.int64), requires_grad=False
    )
    with pytest.raises(HardFailure, match="floating"):
        assert_readout_contract(nonfloating)
    nonfinite = TinyPET()
    with torch.no_grad():
        nonfinite.node_last_layers[0].fill_(float("nan"))
    with pytest.raises(HardFailure, match="finite"):
        assert_readout_contract(nonfinite)


def test_frozen_fingerprint_detects_parameter_and_buffer_drift() -> None:
    from Uncertainty_Quantification.FGE.fge.members import (
        assert_frozen_unchanged,
        frozen_fingerprint,
    )

    model = TinyPET()
    expected = frozen_fingerprint(model)
    assert tuple((item.kind, item.name) for item in expected) == (
        ("parameter", "frozen_weight"),
        ("buffer", "normalization"),
    )
    with torch.no_grad():
        model.frozen_weight[0].add_(1.0)
    with pytest.raises(HardFailure, match="frozen state"):
        assert_frozen_unchanged(model, expected)
    model = TinyPET()
    expected = frozen_fingerprint(model)
    model.normalization[1].add_(1.0)
    with pytest.raises(HardFailure, match="frozen state"):
        assert_frozen_unchanged(model, expected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("member_id", 0),
        ("member_id", True),
        ("cycle", 0),
        ("global_step", -1),
        ("base_sha256", "A" * 64),
        ("base_sha256", "bad"),
    ],
)
def test_pack_member_rejects_invalid_metadata(field: str, value: object) -> None:
    from Uncertainty_Quantification.FGE.fge.members import pack_member

    values = {"member_id": 1, "cycle": 1, "global_step": 0, "base_sha256": BASE_SHA}
    values[field] = value
    with pytest.raises(HardFailure, match=field):
        pack_member(_member_model(6.0), **values)  # type: ignore[arg-type]


def test_pack_member_is_strict_cpu_replacement_schema() -> None:
    payload = _saved_payload()
    assert set(payload) == {
        "schema_version",
        "member_id",
        "cycle",
        "global_step",
        "base_sha256",
        "tensors",
    }
    assert payload["schema_version"] == "upet.fge.member-delta.v1"
    entries = payload["tensors"]
    assert isinstance(entries, list) and len(entries) == 12
    for entry in entries:
        assert set(entry) == {"name", "dtype", "shape", "value"}
        value = entry["value"]
        assert isinstance(value, torch.Tensor) and value.device.type == "cpu"
        assert torch.isfinite(value).all()


def test_load_member_uses_weights_only_cpu_and_checks_base_first(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from Uncertainty_Quantification.FGE.fge.members import (
        load_member,
        readout_tensor_names,
    )

    payload = _saved_payload()
    calls = []

    def fake_load(path: Path, *, weights_only: bool, map_location: object):
        calls.append((path, weights_only, map_location))
        return payload

    monkeypatch.setattr(torch, "load", fake_load)
    path = Path("member.pt")
    assert load_member(path, BASE_SHA, readout_tensor_names(TinyPET())).member_id == 1
    assert calls == [(path, True, "cpu")]
    poisoned = copy.deepcopy(payload)
    poisoned["base_sha256"] = OTHER_SHA
    poisoned["tensors"] = "must-not-be-inspected"
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: poisoned)
    with pytest.raises(HardFailure, match="base_sha256"):
        load_member(path, BASE_SHA, readout_tensor_names(TinyPET()))


@pytest.mark.parametrize(
    "mutation",
    [
        "top_extra",
        "entry_extra",
        "missing",
        "duplicate",
        "order",
        "dtype",
        "shape",
        "nonfinite",
    ],
)
def test_load_member_rejects_schema_mismatches(
    monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    from Uncertainty_Quantification.FGE.fge.members import (
        load_member,
        readout_tensor_names,
    )

    payload = copy.deepcopy(_saved_payload())
    entries = payload["tensors"]
    assert isinstance(entries, list)
    if mutation == "top_extra":
        payload["optimizer"] = {}
    elif mutation == "entry_extra":
        entries[0]["source_path"] = "/legacy/path"
    elif mutation == "missing":
        entries.pop()
    elif mutation == "duplicate":
        entries[1]["name"] = entries[0]["name"]
    elif mutation == "order":
        entries[0], entries[1] = entries[1], entries[0]
    elif mutation == "dtype":
        entries[0]["dtype"] = "torch.float64"
    elif mutation == "shape":
        entries[0]["shape"] = [2]
    else:
        entries[0]["value"].fill_(float("inf"))
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: payload)
    with pytest.raises(HardFailure):
        load_member(Path("bad.pt"), BASE_SHA, readout_tensor_names(TinyPET()))


def test_apply_member_restores_base_and_uses_replacements_not_deltas() -> None:
    from Uncertainty_Quantification.FGE.fge.members import (
        apply_member,
        pack_member,
        readout_tensor_names,
    )

    model = TinyPET()
    base = _base_state(model)
    names = readout_tensor_names(model)
    one = _load_payload(pack_member(_member_model(7.0), 1, 1, 40, BASE_SHA), names)
    two = _load_payload(pack_member(_member_model(-3.0), 2, 2, 80, BASE_SHA), names)
    apply_member(model, base, one)
    assert all(
        torch.equal(p, torch.full_like(p, 7.0))
        for n, p in model.named_parameters()
        if n in names
    )
    with torch.no_grad():
        model.frozen_weight.add_(100.0)
        model.normalization.add_(100.0)
        for name, parameter in model.named_parameters():
            if name in names:
                parameter.fill_(float("nan"))
    apply_member(model, base, two)
    assert all(
        torch.equal(p, torch.full_like(p, -3.0))
        for n, p in model.named_parameters()
        if n in names
    )
    assert torch.equal(model.frozen_weight, base["frozen_weight"])
    assert torch.equal(model.normalization, base["normalization"])


@pytest.mark.parametrize(
    "mutation", ["base_missing", "base_extra", "base_dtype", "member_shape"]
)
def test_apply_member_rejects_mismatches(mutation: str) -> None:
    from Uncertainty_Quantification.FGE.fge.members import (
        apply_member,
        pack_member,
        readout_tensor_names,
    )

    model = TinyPET()
    base = _base_state(model)
    names = readout_tensor_names(model)
    member = _load_payload(pack_member(_member_model(7.0), 1, 1, 40, BASE_SHA), names)
    if mutation == "base_missing":
        base.pop("normalization")
    elif mutation == "base_extra":
        base["obsolete"] = torch.tensor(1.0)
    elif mutation == "base_dtype":
        base["frozen_weight"] = base["frozen_weight"].double()
    else:
        object.__setattr__(member.tensors[0], "value", torch.zeros(2))
    with pytest.raises(HardFailure):
        apply_member(model, base, member)
