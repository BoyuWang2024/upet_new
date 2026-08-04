from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .adapters import LocalToGlobalCumulantAdapter
from .config import ForceTargetMode
from .heads import ComponentConfidenceHead, ConfidenceHead


@dataclass(frozen=True)
class ConfidenceOutput:
    force_logits: torch.Tensor | None
    energy_logits: torch.Tensor | None


class ConfidenceModel(nn.Module):
    """Independent force and energy confidence readouts with branch elision."""

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
        force_active: bool = True,
        energy_active: bool = True,
    ) -> None:
        super().__init__()
        if not force_active and not energy_active:
            raise ValueError("at least one confidence target must be active")
        self.force_input_dim = force_input_dim
        self.energy_input_dim = energy_input_dim
        self.force_target_mode = force_target_mode
        self.force_active = bool(force_active)
        self.energy_active = bool(energy_active)

        if self.force_active:
            force_head_type = (
                ConfidenceHead
                if force_target_mode == "atom_mean"
                else ComponentConfidenceHead
            )
            self.force_head: nn.Module | None = force_head_type(
                force_input_dim,
                force_hidden_dims,
                force_dropout,
                force_num_bins,
            )
        else:
            self.force_head = None

        if self.energy_active:
            self.energy_adapter: LocalToGlobalCumulantAdapter | None = (
                LocalToGlobalCumulantAdapter(
                    energy_input_dim,
                    cumulant_order,
                    signed_root,
                )
            )
            self.energy_head: ConfidenceHead | None = ConfidenceHead(
                energy_input_dim * cumulant_order,
                energy_hidden_dims,
                energy_dropout,
                energy_num_bins,
            )
        else:
            self.energy_adapter = None
            self.energy_head = None

    def forward(
        self,
        force_features: torch.Tensor | None,
        energy_features: torch.Tensor | None,
        offsets: torch.Tensor | None,
    ) -> ConfidenceOutput:
        force_logits: torch.Tensor | None = None
        energy_logits: torch.Tensor | None = None
        if self.force_active:
            if force_features is None or self.force_head is None:
                raise ValueError("active force target requires force features")
            if (
                force_features.ndim != 2
                or force_features.shape[1] != self.force_input_dim
            ):
                raise ValueError(
                    f"force features must have shape [N, {self.force_input_dim}]"
                )
            force_logits = self.force_head(force_features)

        if self.energy_active:
            if (
                energy_features is None
                or offsets is None
                or self.energy_adapter is None
                or self.energy_head is None
            ):
                raise ValueError(
                    "active energy target requires energy features and offsets"
                )
            if (
                energy_features.ndim != 2
                or energy_features.shape[1] != self.energy_input_dim
            ):
                raise ValueError(
                    f"energy features must have shape [N, {self.energy_input_dim}]"
                )
            if self.force_active and force_features is not None:
                if force_features.shape[0] != energy_features.shape[0]:
                    raise ValueError(
                        "force and energy features must have matching atom counts"
                    )
                if force_features.data_ptr() == energy_features.data_ptr():
                    raise ValueError(
                        "force and energy readout features must be distinct tensors"
                    )
            global_features = self.energy_adapter(energy_features, offsets)
            energy_logits = self.energy_head(global_features)

        return ConfidenceOutput(
            force_logits=force_logits,
            energy_logits=energy_logits,
        )
