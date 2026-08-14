"""Native PET restart runtime for readout-only bootstrap training."""

from __future__ import annotations

import copy
import math
import warnings
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from .config import BootstrapConfig
from .errors import HardFailure
from .head_policy import apply_pet_last_layer_policy
from .loss_metrics import LossAccumulator, LossStatistics
from .occurrence import BootstrapOccurrenceCollate, BootstrapOccurrenceDataset
from .sampling import MemberSeeds

_TERMS = (
    "energy",
    "forces",
    "virial",
    "non_conservative_forces",
    "non_conservative_stress",
)
_LOSS_NAMES = {
    "energy": "energy",
    "energy_grad_positions": "forces",
    "energy_grad_strain": "virial",
    "non_conservative_forces": "non_conservative_forces",
    "non_conservative_stress": "non_conservative_stress",
}


def _mask_values(extra: Any, target_name: str) -> torch.Tensor | None:
    mask = extra.get(f"{target_name}_mask")
    return None if mask is None else mask.block().values


def _valid_count(values: torch.Tensor, mask: torch.Tensor | None = None) -> int:
    valid = torch.isfinite(values)
    if mask is not None:
        if mask.numel() != values.numel():
            raise HardFailure("loss mask and target shapes are incompatible")
        valid &= mask.reshape_as(values).bool()
    return int(valid.sum().item())


def _component_counts(targets: Any, extra: Any) -> dict[str, int]:
    energy = targets["energy"].block()
    energy_mask_map = extra.get("energy_mask")
    energy_mask = None if energy_mask_map is None else energy_mask_map.block()
    counts = {
        "energy": _valid_count(
            energy.values, None if energy_mask is None else energy_mask.values
        ),
        "forces": _valid_count(
            energy.gradient("positions").values,
            None if energy_mask is None else energy_mask.gradient("positions").values,
        ),
        "virial": _valid_count(
            energy.gradient("strain").values,
            None if energy_mask is None else energy_mask.gradient("strain").values,
        ),
    }
    for name in ("non_conservative_forces", "non_conservative_stress"):
        counts[name] = _valid_count(
            targets[name].block().values, _mask_values(extra, name)
        )
    return counts


def _targets(path: Path) -> dict[str, object]:
    source = str(path)
    return {
        "energy": {
            "quantity": "energy",
            "read_from": source,
            "reader": "ase",
            "key": "energy",
            "unit": "eV",
            "type": "scalar",
            "sample_kind": "system",
            "per_atom": False,
            "num_subtargets": 1,
            "forces": {"read_from": source, "key": "forces"},
            "stress": {"read_from": source, "key": "stress"},
            "virial": False,
        },
        "non_conservative_forces": {
            "quantity": "force",
            "read_from": source,
            "reader": "ase",
            "key": "forces",
            "unit": "eV/A",
            "type": {"cartesian": {"rank": 1}},
            "sample_kind": "atom",
            "per_atom": True,
            "num_subtargets": 1,
        },
        "non_conservative_stress": {
            "quantity": "pressure",
            "read_from": source,
            "reader": "ase",
            "key": "stress",
            "unit": "eV/A^3",
            "type": {"cartesian": {"rank": 2}},
            "sample_kind": "system",
            "per_atom": False,
            "num_subtargets": 1,
        },
    }


def _loss_contract(
    raw: Mapping[str, object],
) -> tuple[dict[str, object], tuple[str, ...], float, str]:
    try:
        train = raw["train_hypers"]
        if not isinstance(train, Mapping):
            raise TypeError("train_hypers")
        loss = train["loss"]
        if not isinstance(loss, Mapping):
            raise TypeError("loss")
        loss_type = loss["type"]
        if not isinstance(loss_type, Mapping) or set(loss_type) != {"huber"}:
            raise TypeError("loss.type")
        huber = loss_type["huber"]
        if not isinstance(huber, Mapping) or set(huber) != {"deltas"}:
            raise TypeError("loss.type.huber")
        deltas, weights = huber["deltas"], loss["weights"]
        if not isinstance(deltas, Mapping) or not isinstance(weights, Mapping):
            raise TypeError("loss terms")
        if set(deltas) != set(_TERMS) or set(weights) != set(_TERMS):
            raise ValueError("loss term names")
        reduction = loss["reduction"]
        if reduction not in {"mean", "sum"} or loss["sliding_factor"] is not None:
            raise ValueError("loss reduction")
        per_structure = tuple(train["per_structure_targets"])
        if any(not isinstance(name, str) for name in per_structure):
            raise TypeError("per_structure_targets")
        gradient_clip = float(train["grad_clip_norm"])
        delta_values = {name: float(deltas[name]) for name in _TERMS}
        weight_values = {name: float(weights[name]) for name in _TERMS}
    except (KeyError, TypeError, ValueError) as error:
        raise HardFailure(f"invalid checkpoint loss contract: {error}") from error
    numbers = (*delta_values.values(), *weight_values.values(), gradient_clip)
    if not all(math.isfinite(value) for value in numbers):
        raise HardFailure("checkpoint loss contract contains non-finite values")
    if any(value <= 0 for value in delta_values.values()) or gradient_clip <= 0:
        raise HardFailure("checkpoint loss deltas and gradient clip must be positive")
    if any(value < 0 for value in weight_values.values()):
        raise HardFailure("checkpoint loss weights must be non-negative")

    def term(name: str) -> dict[str, object]:
        return {
            "type": "huber",
            "delta": delta_values[name],
            "weight": weight_values[name],
            "reduction": reduction,
            "gradients": {},
        }

    config = {
        "energy": term("energy"),
        "non_conservative_forces": term("non_conservative_forces"),
        "non_conservative_stress": term("non_conservative_stress"),
    }
    config["energy"]["gradients"] = {
        "positions": term("forces"),
        "strain": term("virial"),
    }
    return config, per_structure, gradient_clip, reduction


class PETTrainingRuntime:
    """Real PET data/model/loss runtime implementing the training protocol."""

    def __init__(
        self,
        *,
        model: torch.nn.Module,
        loss_fn: Any,
        train_loader: Any,
        val_loader: Any,
        helpers: tuple[Any, Any, Any, Any],
        loader_generator: torch.Generator,
        per_structure_targets: tuple[str, ...],
        gradient_clip: float,
        loss_reduction: str,
    ) -> None:
        self.model = model
        self._loss_fn = loss_fn
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._loader_generator = loader_generator
        (
            self._unpack_batch,
            self._batch_to,
            self._evaluate_model,
            self._average_by_num_atoms,
        ) = helpers
        self._per_structure_targets = per_structure_targets
        self._gradient_clip = gradient_clip
        self._loss_reduction = loss_reduction
        self._frozen = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        }

    @classmethod
    def from_config(
        cls, config: BootstrapConfig, indices: Any, seeds: MemberSeeds
    ) -> PETTrainingRuntime:
        """Build the official restart, transform, target, and loss path."""
        try:
            from metatrain.utils.additive import get_remove_additive_transform
            from metatrain.utils.augmentation import RotationalAugmenter
            from metatrain.utils.data import (
                CollateFn,
                CombinedDataLoader,
                Dataset,
                read_systems,
                read_targets,
                unpack_batch,
            )
            from metatrain.utils.evaluate_model import evaluate_model
            from metatrain.utils.io import model_from_checkpoint
            from metatrain.utils.loss import LossAggregator
            from metatrain.utils.neighbor_lists import (
                get_requested_neighbor_lists,
                get_system_with_neighbor_lists_transform,
            )
            from metatrain.utils.per_atom import average_by_num_atoms
            from metatrain.utils.scaler import get_remove_scale_transform
            from metatrain.utils.transfer import batch_to
            from omegaconf import OmegaConf
            from torch.utils.data import DataLoader
        except ImportError as error:
            raise HardFailure(
                "PET training runtime dependencies are unavailable"
            ) from error
        if config.training.device != "cpu" or config.training.precision != "float32":
            raise HardFailure("native PET training requires CPU float32")
        try:
            raw = torch.load(
                config.checkpoint.base_path, map_location="cpu", weights_only=False
            )
            if not isinstance(raw, Mapping):
                raise HardFailure("base checkpoint root must be a mapping")
            model = model_from_checkpoint(raw, context="restart").to(
                device="cpu", dtype=torch.float32
            )
        except HardFailure:
            raise
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise HardFailure(
                f"could not construct PET training model: {error}"
            ) from error
        apply_pet_last_layer_policy(model)
        loss_config, per_structure, checkpoint_clip, reduction = _loss_contract(raw)
        if not math.isclose(
            config.training.gradient_clip, checkpoint_clip, rel_tol=0.0, abs_tol=0.0
        ):
            raise HardFailure(
                "configured gradient_clip differs from base checkpoint contract"
            )
        for additive in model.additive_models:
            additive.to(dtype=torch.float64)
        model.scaler.to(dtype=torch.float64)
        model.additive_models[0].weights_to(device="cpu", dtype=torch.float64)
        additives = copy.deepcopy(
            model.additive_models.to(device="cpu", dtype=torch.float64)
        )
        model.scaler.scales_to(device="cpu", dtype=torch.float64)
        scaler = copy.deepcopy(model.scaler.to(device="cpu", dtype=torch.float64))
        model.additive_models.to(device="cpu", dtype=torch.float32)
        model.additive_models[0].weights_to(device="cpu", dtype=torch.float32)
        model.scaler.to(device="cpu", dtype=torch.float32)
        model.scaler.scales_to(device="cpu", dtype=torch.float32)

        def dataset(path: Path):
            systems = read_systems(str(path))
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        r"the name of 'non_conservative_(forces|stress)' "
                        r"resembles.*"
                    ),
                    category=UserWarning,
                )
                values, info = read_targets(OmegaConf.create(_targets(path)))
            expected = model.dataset_info.targets
            if set(info) != set(expected) or any(
                not info[name].is_compatible_with(expected[name]) for name in expected
            ):
                raise HardFailure("dataset targets are incompatible with checkpoint")
            return Dataset.from_dict({"system": systems, **values})

        train_data = BootstrapOccurrenceDataset(dataset(config.data.train), indices)
        val_data = dataset(config.data.val)
        target_info = model.dataset_info.targets
        fixed = [
            get_system_with_neighbor_lists_transform(
                get_requested_neighbor_lists(model)
            ),
            get_remove_additive_transform(additives, target_info),
            get_remove_scale_transform(scaler),
        ]
        train_collate = BootstrapOccurrenceCollate(
            CollateFn(
                target_keys=list(target_info),
                callables=[
                    RotationalAugmenter(
                        target_info_dict=target_info,
                        extra_data_info_dict=model.dataset_info.extra_data,
                    ).apply_random_augmentations,
                    *fixed,
                ],
            )
        )
        val_collate = CollateFn(target_keys=list(target_info), callables=fixed)
        generator = torch.Generator().manual_seed(seeds.loader)
        train_loader = CombinedDataLoader(
            [
                DataLoader(
                    train_data,
                    batch_size=config.training.batch_size,
                    shuffle=True,
                    drop_last=False,
                    collate_fn=train_collate,
                    num_workers=config.training.num_workers,
                    generator=generator,
                )
            ],
            shuffle=True,
        )
        val_loader = CombinedDataLoader(
            [
                DataLoader(
                    val_data,
                    batch_size=config.training.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=val_collate,
                    num_workers=config.training.num_workers,
                )
            ],
            shuffle=False,
        )
        return cls(
            model=model,
            loss_fn=LossAggregator(targets=target_info, config=loss_config),
            train_loader=train_loader,
            val_loader=val_loader,
            loader_generator=generator,
            helpers=(unpack_batch, batch_to, evaluate_model, average_by_num_atoms),
            per_structure_targets=per_structure,
            gradient_clip=checkpoint_clip,
            loss_reduction=reduction,
        )

    def _loss(
        self, batch: object, training: bool
    ) -> tuple[torch.Tensor, LossStatistics]:
        systems, targets, extra = self._unpack_batch(batch)
        systems, targets, extra = self._batch_to(
            systems, targets, extra, dtype=torch.float32, device=torch.device("cpu")
        )
        requested = {name: self.model.dataset_info.targets[name] for name in targets}
        predicted = self._evaluate_model(
            self.model, systems, requested, is_training=training
        )
        predicted = self._average_by_num_atoms(
            predicted, systems, self._per_structure_targets
        )
        targets = self._average_by_num_atoms(
            targets, systems, self._per_structure_targets
        )
        loss = self._loss_fn(predicted, targets, extra)
        if (
            not isinstance(loss, torch.Tensor)
            or loss.numel() != 1
            or not bool(torch.isfinite(loss).item())
        ):
            raise HardFailure("checkpoint-defined PET loss is invalid")
        counts = _component_counts(targets, extra)
        components = {
            _LOSS_NAMES[key]: float(
                (term.weight * term.compute(predicted, targets, extra)).detach().item()
            )
            for key, term in self._loss_fn.losses.items()
            if key in _LOSS_NAMES
        }
        if set(components) != set(counts):
            raise HardFailure("checkpoint loss component set is incomplete")
        weighted = components
        if self._loss_reduction == "mean":
            weighted = {
                name: value * counts[name] for name, value in components.items()
            }
        return loss, LossStatistics(weighted, counts)

    def capture_state(self) -> dict[str, torch.Tensor]:
        return {"loader_generator": self._loader_generator.get_state().clone()}

    def restore_state(self, state: Mapping[str, object]) -> None:
        value = state.get("loader_generator")
        if not isinstance(value, torch.Tensor):
            raise HardFailure("resume runtime state lacks loader generator")
        self._loader_generator.set_state(value.detach().cpu())

    def _assert_frozen(self) -> None:
        parameters = dict(self.model.named_parameters())
        for name, expected in self._frozen.items():
            if not torch.equal(parameters[name].detach().cpu(), expected):
                raise HardFailure(f"frozen PET parameter changed: {name}")

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        del epoch
        self.model.train()
        losses: list[float] = []
        trainable = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]
        for batch in self._train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _ = self._loss(batch, True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(trainable, self._gradient_clip)
            if not bool(torch.isfinite(norm).item()):
                raise HardFailure("PET training gradient norm is non-finite")
            optimizer.step()
            losses.append(float(loss.detach().item()))
        if not losses:
            raise HardFailure("PET training loader is empty")
        self._assert_frozen()
        return sum(losses) / len(losses)

    def validation_loss(self, epoch: int) -> float:
        del epoch
        self.model.eval()
        accumulator = LossAccumulator(self._loss_reduction)
        with torch.no_grad():
            for batch in self._val_loader:
                _, statistics = self._loss(batch, False)
                accumulator.update(statistics)
        try:
            return accumulator.total()
        except (RuntimeError, ValueError) as error:
            raise HardFailure(
                f"could not aggregate validation loss: {error}"
            ) from error
