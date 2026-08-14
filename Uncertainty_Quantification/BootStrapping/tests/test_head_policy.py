from __future__ import annotations

import pytest
import torch


class _TinyPET(torch.nn.Module):
    def __init__(self, *, node_count: int = 10_000, edge_count: int = 3_338):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.node_last_layers = torch.nn.Parameter(torch.zeros(node_count))
        self.edge_last_layers = torch.nn.Parameter(torch.zeros(edge_count))


def test_pet_last_layer_policy_freezes_everything_else() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.head_policy import (
        apply_pet_last_layer_policy,
    )

    model = _TinyPET()
    audit = apply_pet_last_layer_policy(model)

    assert audit.trainable_parameter_count == 13_338
    assert audit.trainable_names == ("node_last_layers", "edge_last_layers")
    assert not model.backbone.weight.requires_grad
    assert not model.backbone.bias.requires_grad
    assert model.node_last_layers.requires_grad
    assert model.edge_last_layers.requires_grad


def test_pet_last_layer_policy_rejects_parameter_count_drift() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.bootstrap.head_policy import (
        apply_pet_last_layer_policy,
    )

    with pytest.raises(HardFailure, match="13,338"):
        apply_pet_last_layer_policy(_TinyPET(edge_count=3_337))
