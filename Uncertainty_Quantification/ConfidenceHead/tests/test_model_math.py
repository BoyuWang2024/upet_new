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
        hidden_dims=(4,),
        num_bins=5,
        cumulant_order=1,
        signed_root=True,
        dropout=0.0,
    )
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

    assert output.force_logits.shape == (3, 3, 5)
    assert output.energy_logits.shape == (2, 5)
    assert len(energy_head_inputs) == 1
    torch.testing.assert_close(
        energy_head_inputs[0],
        torch.full((2, 2), 7.0),
    )


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
