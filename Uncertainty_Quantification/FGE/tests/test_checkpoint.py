from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure


LOSS_NAMES = (
    "energy",
    "forces",
    "virial",
    "non_conservative_forces",
    "non_conservative_stress",
)


def _raw_checkpoint() -> dict[str, object]:
    return {
        "model_state_dict": {"weight": torch.tensor([1.0])},
        "best_model_state_dict": {"weight": torch.tensor([99.0])},
        "train_hypers": {
            "loss": {
                "type": {
                    "huber": {
                        "deltas": dict(
                            zip(
                                LOSS_NAMES,
                                (0.015, 0.04, 0.03, 0.02, 0.004),
                                strict=True,
                            )
                        )
                    }
                },
                "weights": dict(
                    zip(LOSS_NAMES, (1.0, 2.0, 3.0, 0.1, 0.2), strict=True)
                ),
                "reduction": "mean",
                "sliding_factor": None,
            },
            "per_structure_targets": ["non_conservative_stress"],
            "grad_clip_norm": 1.0,
        },
    }


def test_load_bundle_selects_restart_state_and_passes_explicit_load_options(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _raw_checkpoint()
    calls: list[tuple[Path, object, bool]] = []

    def fake_load(path: Path, *, map_location: object, weights_only: bool):
        calls.append((path, map_location, weights_only))
        return raw

    monkeypatch.setattr(torch, "load", fake_load)

    from Uncertainty_Quantification.FGE.fge.checkpoint import load_checkpoint_bundle

    path = Path("literal.ckpt")
    bundle = load_checkpoint_bundle(path)

    assert calls == [(path, "cpu", False)]
    assert bundle.model_state_dict is raw["model_state_dict"]
    assert bundle.model_state_dict is not raw["best_model_state_dict"]


def test_load_bundle_round_trips_actual_checkpoint_file(tmp_path: Path) -> None:
    from Uncertainty_Quantification.FGE.fge.checkpoint import load_checkpoint_bundle

    path = tmp_path / "checkpoint.ckpt"
    raw = _raw_checkpoint()
    torch.save(raw, path)
    bundle = load_checkpoint_bundle(path)
    assert torch.equal(bundle.model_state_dict["weight"], torch.tensor([1.0]))
    assert tuple(term.name for term in bundle.loss_contract.terms) == LOSS_NAMES


def test_recover_loss_contract_preserves_exact_five_term_configuration() -> None:
    from Uncertainty_Quantification.FGE.fge.checkpoint import recover_loss_contract

    contract = recover_loss_contract(_raw_checkpoint())

    assert tuple(term.name for term in contract.terms) == LOSS_NAMES
    assert tuple(term.delta for term in contract.terms) == (
        0.015,
        0.04,
        0.03,
        0.02,
        0.004,
    )
    assert tuple(term.weight for term in contract.terms) == (1.0, 2.0, 3.0, 0.1, 0.2)
    assert all(term.loss_type == "huber" for term in contract.terms)
    assert contract.reduction == "mean"
    assert contract.sliding_factor is None
    assert contract.per_structure_targets == ("non_conservative_stress",)
    assert contract.grad_clip_norm == 1.0

    with pytest.raises((AttributeError, TypeError)):
        contract.reduction = "sum"  # type: ignore[misc]


@pytest.mark.parametrize("state", [None, [], {"weight": "not-a-tensor"}])
def test_load_bundle_rejects_missing_or_malformed_restart_state(
    monkeypatch: pytest.MonkeyPatch, state: object
) -> None:
    raw = _raw_checkpoint()
    if state is None:
        raw.pop("model_state_dict")
    else:
        raw["model_state_dict"] = state
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: raw)

    from Uncertainty_Quantification.FGE.fge.checkpoint import load_checkpoint_bundle

    with pytest.raises(HardFailure, match="restart model_state_dict"):
        load_checkpoint_bundle(Path("bad.ckpt"))


@pytest.mark.parametrize("section", ["deltas", "weights"])
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_loss_contract_requires_exact_term_keys(section: str, mutation: str) -> None:
    from Uncertainty_Quantification.FGE.fge.checkpoint import recover_loss_contract

    raw = _raw_checkpoint()
    loss = raw["train_hypers"]["loss"]  # type: ignore[index]
    mapping = (
        loss["type"]["huber"][section]  # type: ignore[index]
        if section == "deltas"
        else loss[section]  # type: ignore[index]
    )
    if mutation == "missing":
        mapping.pop("virial")
    else:
        mapping["obsolete"] = 1.0

    with pytest.raises(HardFailure, match="loss contract"):
        recover_loss_contract(raw)


@pytest.mark.parametrize("mutation", ["competing_loss", "obsolete_huber_key"])
def test_loss_contract_rejects_extra_loss_type_and_huber_keys(mutation: str) -> None:
    from Uncertainty_Quantification.FGE.fge.checkpoint import recover_loss_contract

    raw = _raw_checkpoint()
    loss_type = raw["train_hypers"]["loss"]["type"]  # type: ignore[index]
    if mutation == "competing_loss":
        loss_type["mae"] = {}  # type: ignore[index]
    else:
        loss_type["huber"]["legacy_delta"] = 0.1  # type: ignore[index]
    with pytest.raises(HardFailure, match="loss contract"):
        recover_loss_contract(raw)


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("deltas", "energy", True),
        ("deltas", "forces", "0.04"),
        ("deltas", "virial", 0.0),
        ("deltas", "energy", math.inf),
        ("weights", "energy", -1.0),
        ("weights", "forces", math.nan),
    ],
)
def test_loss_contract_rejects_mistyped_nonfinite_or_invalid_values(
    section: str, name: str, value: object
) -> None:
    from Uncertainty_Quantification.FGE.fge.checkpoint import recover_loss_contract

    raw = _raw_checkpoint()
    loss = raw["train_hypers"]["loss"]  # type: ignore[index]
    mapping = (
        loss["type"]["huber"][section]  # type: ignore[index]
        if section == "deltas"
        else loss[section]  # type: ignore[index]
    )
    mapping[name] = value

    with pytest.raises(HardFailure, match="loss contract"):
        recover_loss_contract(raw)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("reduction", "median"),
        ("sliding_factor", 0.5),
    ],
)
def test_loss_contract_rejects_inconsistent_loss_fields(
    field: str, value: object
) -> None:
    from Uncertainty_Quantification.FGE.fge.checkpoint import recover_loss_contract

    raw = _raw_checkpoint()
    raw["train_hypers"]["loss"][field] = value  # type: ignore[index]

    with pytest.raises(HardFailure, match="loss contract"):
        recover_loss_contract(raw)


@pytest.mark.parametrize("value", [None, True, 0.0, -1.0, math.inf, "1.0"])
def test_loss_contract_rejects_invalid_gradient_clipping(value: object) -> None:
    from Uncertainty_Quantification.FGE.fge.checkpoint import recover_loss_contract

    raw = _raw_checkpoint()
    raw["train_hypers"]["grad_clip_norm"] = value  # type: ignore[index]

    with pytest.raises(HardFailure, match="loss contract"):
        recover_loss_contract(raw)


def test_loss_contract_rejects_non_string_or_duplicate_structure_targets() -> None:
    from Uncertainty_Quantification.FGE.fge.checkpoint import recover_loss_contract

    for invalid in (
        ["non_conservative_stress", "non_conservative_stress"],
        ["non_conservative_stress", 7],
        "non_conservative_stress",
    ):
        raw = copy.deepcopy(_raw_checkpoint())
        raw["train_hypers"]["per_structure_targets"] = invalid  # type: ignore[index]
        with pytest.raises(HardFailure, match="loss contract"):
            recover_loss_contract(raw)
