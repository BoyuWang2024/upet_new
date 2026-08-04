from __future__ import annotations

import math

import pytest
import torch
from torch.nn import functional as F

from Uncertainty_Quantification.ConfidenceHead.confidence_head.adapters import (
    LocalToGlobalCumulantAdapter,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.heads import (
    ComponentConfidenceHead,
    ConfidenceHead,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.losses import (
    confidence_loss,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.model import (
    ConfidenceModel,
)


def test_model_uses_strictly_separate_force_and_energy_readouts() -> None:
    model = ConfidenceModel(
        force_input_dim=2,
        energy_input_dim=2,
        force_hidden_dims=(7,),
        energy_hidden_dims=(11, 5),
        force_dropout=0.1,
        energy_dropout=0.2,
        force_num_bins=13,
        energy_num_bins=17,
        cumulant_order=1,
        signed_root=True,
        force_target_mode="component",
    )
    assert model.force_head.heads[0].network[0].out_features == 7
    assert model.energy_head.network[0].out_features == 11
    force_features = torch.zeros(3, 2)
    energy_features = torch.full((3, 2), 7.0)
    offsets = torch.tensor([0, 1, 3])
    energy_head_inputs: list[torch.Tensor] = []

    def capture_energy_input(
        _module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        energy_head_inputs.append(inputs[0].detach().clone())

    handle = model.energy_head.register_forward_pre_hook(capture_energy_input)
    try:
        output = model(force_features, energy_features, offsets)
    finally:
        handle.remove()

    assert output.force_logits.shape == (3, 3, 13)
    assert output.energy_logits.shape == (2, 17)
    assert len(energy_head_inputs) == 1
    torch.testing.assert_close(
        energy_head_inputs[0],
        torch.full((2, 2), 7.0),
    )


def _model_kwargs() -> dict[str, object]:
    return {
        "force_input_dim": 2,
        "energy_input_dim": 3,
        "force_hidden_dims": (4,),
        "energy_hidden_dims": (5,),
        "force_dropout": 0.0,
        "energy_dropout": 0.0,
        "force_num_bins": 7,
        "energy_num_bins": 11,
        "cumulant_order": 2,
        "signed_root": True,
        "force_target_mode": "atom_mean",
    }


def test_force_only_model_has_no_energy_modules_or_parameters() -> None:
    model = ConfidenceModel(
        **_model_kwargs(),
        force_active=True,
        energy_active=False,
    )

    assert model.force_head is not None
    assert model.energy_adapter is None
    assert model.energy_head is None
    assert all("energy_" not in name for name, _ in model.named_parameters())
    output = model(
        force_features=torch.randn(5, 2),
        energy_features=None,
        offsets=None,
    )
    assert output.force_logits is not None
    assert output.energy_logits is None


def test_energy_only_model_has_no_force_modules_or_parameters() -> None:
    model = ConfidenceModel(
        **_model_kwargs(),
        force_active=False,
        energy_active=True,
    )

    assert model.force_head is None
    assert model.energy_adapter is not None
    assert model.energy_head is not None
    assert all("force_head" not in name for name, _ in model.named_parameters())
    output = model(
        force_features=None,
        energy_features=torch.randn(5, 3),
        offsets=torch.tensor([0, 2, 5]),
    )
    assert output.force_logits is None
    assert output.energy_logits is not None
    assert output.energy_logits.shape == (2, 11)


def test_model_rejects_disabling_both_targets() -> None:
    with pytest.raises(ValueError, match="at least one"):
        ConfidenceModel(
            **_model_kwargs(),
            force_active=False,
            energy_active=False,
        )


def test_force_only_loss_is_supervised_weighted_total() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])

    result = confidence_loss(
        logits,
        labels,
        None,
        None,
        force_target_mode="atom_mean",
        force_weight=0.5,
        energy_weight=0.0,
    )

    expected = F.cross_entropy(logits, labels)
    torch.testing.assert_close(result.total, 0.5 * expected)
    torch.testing.assert_close(result.force, expected)
    assert result.energy is None
    assert result.force_count == 2
    assert result.energy_count == 0


def test_energy_only_loss_is_supervised_weighted_total() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1])

    result = confidence_loss(
        None,
        None,
        logits,
        labels,
        force_target_mode="atom_mean",
        force_weight=0.0,
        energy_weight=0.3,
    )

    expected = F.cross_entropy(logits, labels)
    torch.testing.assert_close(result.total, 0.3 * expected)
    assert result.force is None
    torch.testing.assert_close(
        result.energy,
        expected,
    )
    assert result.force_count == 0
    assert result.energy_count == 2


@pytest.mark.parametrize(
    ("mode", "expected_shape", "head_type"),
    [
        ("atom_mean", (3, 5), ConfidenceHead),
        ("component", (3, 3, 5), ComponentConfidenceHead),
    ],
)
def test_model_selects_force_head_from_explicit_mode(
    mode: str,
    expected_shape: tuple[int, ...],
    head_type: type[torch.nn.Module],
) -> None:
    model = ConfidenceModel(
        force_input_dim=2,
        energy_input_dim=2,
        force_hidden_dims=(4,),
        energy_hidden_dims=(4,),
        force_dropout=0.0,
        energy_dropout=0.0,
        force_num_bins=5,
        energy_num_bins=7,
        cumulant_order=1,
        signed_root=True,
        force_target_mode=mode,
    )

    assert isinstance(model.force_head, head_type)
    output = model(
        torch.zeros(3, 2),
        torch.ones(3, 2),
        torch.tensor([0, 3]),
    )
    assert output.force_logits.shape == expected_shape


@pytest.mark.parametrize(
    ("order", "first_structure"),
    [
        (1, [1.5]),
        (2, [1.5, 0.5]),
        (3, [1.5, 0.5, 0.0]),
        (4, [1.5, 0.5, 0.0, -math.pow(0.125, 0.25)]),
    ],
)
def test_cumulant_adapter_matches_hand_calculation(
    order: int, first_structure: list[float]
) -> None:
    adapter = LocalToGlobalCumulantAdapter(
        input_dim=1,
        order=order,
        signed_root=True,
    )
    features = torch.tensor([[1.0], [2.0], [3.0]])
    offsets = torch.tensor([0, 2, 3])

    result = adapter(features, offsets)

    expected = torch.tensor(
        [
            first_structure,
            [3.0, *([0.0] * (order - 1))],
        ]
    )
    torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("order", [0, 9])
def test_cumulant_adapter_rejects_orders_outside_one_to_eight(order: int) -> None:
    with pytest.raises(ValueError, match="1.*8"):
        LocalToGlobalCumulantAdapter(
            input_dim=2,
            order=order,
            signed_root=True,
        )


@pytest.mark.parametrize(
    "offsets",
    [
        torch.tensor([[0, 2]]),
        torch.tensor([0.0, 2.0]),
        torch.tensor([1, 2]),
        torch.tensor([0, 1]),
        torch.tensor([0, 2, 2]),
        torch.tensor([0]),
    ],
)
def test_cumulant_adapter_rejects_invalid_offsets(offsets: torch.Tensor) -> None:
    adapter = LocalToGlobalCumulantAdapter(1, 2, signed_root=False)

    with pytest.raises(ValueError, match="offsets"):
        adapter(torch.tensor([[1.0], [2.0]]), offsets)


def test_component_head_has_three_independent_parameter_sets() -> None:
    head = ComponentConfidenceHead(
        input_dim=2,
        hidden_dims=(4,),
        dropout=0.0,
        num_bins=5,
    )

    parameter_ids = [
        {id(parameter) for parameter in subhead.parameters()} for subhead in head.heads
    ]

    assert len(parameter_ids) == 3
    assert parameter_ids[0].isdisjoint(parameter_ids[1])
    assert parameter_ids[0].isdisjoint(parameter_ids[2])
    assert parameter_ids[1].isdisjoint(parameter_ids[2])
    assert head(torch.zeros(6, 2)).shape == (6, 3, 5)


@pytest.mark.parametrize("head_type", [ConfidenceHead, ComponentConfidenceHead])
@pytest.mark.parametrize(
    "arguments",
    [
        {"input_dim": 0, "hidden_dims": (4,), "dropout": 0.0, "num_bins": 5},
        {"input_dim": 2, "hidden_dims": (0,), "dropout": 0.0, "num_bins": 5},
        {"input_dim": 2, "hidden_dims": (4,), "dropout": 0.0, "num_bins": 0},
        {"input_dim": 2, "hidden_dims": (4,), "dropout": -0.1, "num_bins": 5},
        {"input_dim": 2, "hidden_dims": (4,), "dropout": 1.0, "num_bins": 5},
    ],
)
def test_heads_reject_invalid_construction_arguments(
    head_type: type[torch.nn.Module],
    arguments: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        head_type(**arguments)


def test_component_head_rejects_non_matrix_features() -> None:
    head = ComponentConfidenceHead(2, (4,), 0.0, 5)

    with pytest.raises(ValueError, match="features"):
        head(torch.zeros(3, 1, 2))


@pytest.mark.parametrize(
    ("force_features", "energy_features", "offsets"),
    [
        (torch.zeros(3, 1, 2), torch.ones(3, 2), torch.tensor([0, 3])),
        (torch.zeros(3, 2), torch.ones(3, 1, 2), torch.tensor([0, 3])),
        (torch.zeros(2, 2), torch.ones(3, 2), torch.tensor([0, 3])),
    ],
)
def test_model_rejects_invalid_readout_feature_shapes(
    force_features: torch.Tensor,
    energy_features: torch.Tensor,
    offsets: torch.Tensor,
) -> None:
    model = ConfidenceModel(
        force_input_dim=2,
        energy_input_dim=2,
        force_hidden_dims=(4,),
        energy_hidden_dims=(4,),
        force_dropout=0.0,
        energy_dropout=0.0,
        force_num_bins=5,
        energy_num_bins=5,
        cumulant_order=1,
        signed_root=True,
        force_target_mode="component",
    )

    with pytest.raises(ValueError, match="features"):
        model(force_features, energy_features, offsets)


def test_confidence_loss_flattens_force_components_and_weights_means() -> None:
    force_logits = torch.tensor(
        [
            [[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]],
            [[0.0, 2.0], [2.0, 0.0], [3.0, 1.0]],
        ]
    )
    force_labels = torch.tensor([[0, 1, 0], [1, 0, 1]])
    energy_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    energy_labels = torch.tensor([0, 0])

    result = confidence_loss(
        force_logits,
        force_labels,
        energy_logits,
        energy_labels,
        force_target_mode="component",
    )

    expected_force = F.cross_entropy(
        force_logits.reshape(-1, 2),
        force_labels.reshape(-1),
    )
    expected_energy = F.cross_entropy(energy_logits, energy_labels)
    torch.testing.assert_close(result.force, expected_force)
    torch.testing.assert_close(result.energy, expected_energy)
    torch.testing.assert_close(
        result.total,
        1.0 * expected_force + 1.5 * expected_energy,
    )
    assert result.force_count == force_labels.numel()
    assert result.energy_count == energy_labels.numel()


def test_atom_mean_loss_counts_atoms_and_keeps_supervised_total() -> None:
    force_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    force_labels = torch.tensor([0, 1])
    energy_logits = torch.tensor([[2.0, 0.0]])
    energy_labels = torch.tensor([1])

    result = confidence_loss(
        force_logits,
        force_labels,
        energy_logits,
        energy_labels,
        force_target_mode="atom_mean",
    )

    expected_force = F.cross_entropy(force_logits, force_labels)
    expected_energy = F.cross_entropy(energy_logits, energy_labels)
    torch.testing.assert_close(result.force, expected_force)
    torch.testing.assert_close(result.energy, expected_energy)
    torch.testing.assert_close(
        result.total,
        expected_force + 1.5 * expected_energy,
    )
    assert result.force_count == 2
    assert result.energy_count == 1


@pytest.mark.parametrize(
    ("mode", "force_logits", "force_labels"),
    [
        (
            "atom_mean",
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, dtype=torch.long),
        ),
        (
            "component",
            torch.zeros(2, 4),
            torch.zeros(2, dtype=torch.long),
        ),
    ],
)
def test_confidence_loss_rejects_force_shape_from_other_mode(
    mode: str,
    force_logits: torch.Tensor,
    force_labels: torch.Tensor,
) -> None:
    with pytest.raises(ValueError, match="force"):
        confidence_loss(
            force_logits,
            force_labels,
            torch.zeros(1, 4),
            torch.zeros(1, dtype=torch.long),
            force_target_mode=mode,
        )


def test_confidence_loss_accepts_independent_branch_bin_counts() -> None:
    force_logits = torch.zeros(2, 3, 4)
    force_labels = torch.zeros(2, 3, dtype=torch.long)
    energy_logits = torch.zeros(2, 5)
    energy_labels = torch.zeros(2, dtype=torch.long)

    result = confidence_loss(
        force_logits,
        force_labels,
        energy_logits,
        energy_labels,
        force_target_mode="component",
    )

    assert result.force_count == force_labels.numel()
    assert result.energy_count == energy_labels.numel()


@pytest.mark.parametrize(
    ("force_logits", "force_labels", "energy_logits", "energy_labels"),
    [
        (
            torch.zeros(2, 2, 4),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 4),
            torch.zeros(2, dtype=torch.long),
        ),
        (
            torch.zeros(2, 3, 4),
            torch.zeros(3, 2, dtype=torch.long),
            torch.zeros(2, 4),
            torch.zeros(2, dtype=torch.long),
        ),
        (
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 1, 4),
            torch.zeros(2, dtype=torch.long),
        ),
        (
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 4),
            torch.zeros(2, 1, dtype=torch.long),
        ),
        (
            torch.zeros(0, 3, 4),
            torch.zeros(0, 3, dtype=torch.long),
            torch.zeros(2, 4),
            torch.zeros(2, dtype=torch.long),
        ),
        (
            torch.zeros(2, 3, 4),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(0, 4),
            torch.zeros(0, dtype=torch.long),
        ),
        (
            torch.zeros(2, 3, 0),
            torch.zeros(2, 3, dtype=torch.long),
            torch.zeros(2, 0),
            torch.zeros(2, dtype=torch.long),
        ),
    ],
)
def test_confidence_loss_rejects_invalid_tensor_contracts(
    force_logits: torch.Tensor,
    force_labels: torch.Tensor,
    energy_logits: torch.Tensor,
    energy_labels: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        confidence_loss(
            force_logits,
            force_labels,
            energy_logits,
            energy_labels,
            force_target_mode="component",
        )
