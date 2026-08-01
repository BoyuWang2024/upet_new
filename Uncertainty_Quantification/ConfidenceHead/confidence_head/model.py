from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adapters import LocalToGlobalCumulantAdapter
from .config import ForceTargetMode
from .heads import ComponentConfidenceHead, ConfidenceHead


@dataclass(frozen=True)
class ConfidenceOutput:
    force_logits: torch.Tensor
    energy_logits: torch.Tensor


class ConfidenceModel(nn.Module):
    """Separate force and energy confidence readouts."""

    def __init__(
        self,
        force_input_dim: int,
        energy_input_dim: int,
        force_hidden_dims: tuple[int, ...],
        energy_hidden_dims: tuple[int, ...],
        force_dropout: float,
        energy_dropout: float,
        force_num_bins: int,
        energy_num_bins: int,
        cumulant_order: int,
        signed_root: bool,
        force_target_mode: ForceTargetMode,
    ) -> None:
        super().__init__()
        self.force_input_dim = force_input_dim
        self.energy_input_dim = energy_input_dim
        self.force_target_mode = force_target_mode
        force_head_type = (
            ConfidenceHead
            if force_target_mode == "atom_mean"
            else ComponentConfidenceHead
        )
        self.force_head = force_head_type(
            force_input_dim,
            force_hidden_dims,
            force_dropout,
            force_num_bins,
        )
        self.energy_adapter = LocalToGlobalCumulantAdapter(
            energy_input_dim,
            cumulant_order,
            signed_root,
        )
        self.energy_head = ConfidenceHead(
            energy_input_dim * cumulant_order,
            energy_hidden_dims,
            energy_dropout,
            energy_num_bins,
        )

    def forward(
        self,
        force_features: torch.Tensor,
        energy_features: torch.Tensor,
        offsets: torch.Tensor,
    ) -> ConfidenceOutput:
        if force_features.ndim != 2 or force_features.shape[1] != self.force_input_dim:
            raise ValueError(
                f"force features must have shape [N, {self.force_input_dim}]"
            )
        if (
            energy_features.ndim != 2
            or energy_features.shape[1] != self.energy_input_dim
        ):
            raise ValueError(
                f"energy features must have shape [N, {self.energy_input_dim}]"
            )
        if force_features.shape[0] != energy_features.shape[0]:
            raise ValueError("force and energy features must have matching atom counts")
        if force_features.data_ptr() == energy_features.data_ptr():
            raise ValueError(
                "force and energy readout features must be distinct tensors"
            )

        force_logits = self.force_head(force_features)
        energy_features = self.energy_adapter(energy_features, offsets)
        energy_logits = self.energy_head(energy_features)
        return ConfidenceOutput(
            force_logits=force_logits,
            energy_logits=energy_logits,
        )
