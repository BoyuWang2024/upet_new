"""Pure single-target ConfidenceHead prediction without artifact publication."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch

from .binning import BinningSpec, expected_error, labels_from_thresholds
from .errors import energy_per_atom_error, force_error, force_error_definition
from .metrics import classification_metrics


@dataclass(frozen=True)
class SingleTargetResult:
    """Pure tensors and metrics produced by one active confidence branch."""

    target: str
    predictions: dict[str, Any]
    metrics: dict[str, Any]


def collect_single_target_predictions(
    *,
    model: Any,
    loader: Any,
    device: torch.device,
    config: Any,
    force_spec: BinningSpec,
    energy_spec: BinningSpec,
    to_device: Callable[[Mapping[str, Any], torch.device], dict[str, Any]],
    offsets: Callable[[list[torch.Tensor]], torch.Tensor],
    include_raw: bool = False,
) -> SingleTargetResult:
    """Run exactly one active head without publishing or mutating a run."""

    force_active = bool(model.force_active)
    energy_active = bool(model.energy_active)
    if force_active == energy_active:
        raise ValueError("single-target prediction requires exactly one active target")
    target = "force" if force_active else "energy"
    names = [
        "structure_ids",
        f"{target}_logits",
        f"{target}_labels",
        f"{target}_observed_errors",
        f"{target}_expected_errors",
    ]
    if include_raw:
        names.extend((f"{target}_prediction", f"{target}_reference"))
    collected: dict[str, list[torch.Tensor]] = {name: [] for name in names}
    atom_counts: list[torch.Tensor] = []
    with torch.inference_mode():
        for raw_batch in loader:
            batch = to_device(raw_batch, device)
            output = model(
                batch["force_features"] if force_active else None,
                batch["energy_features"] if energy_active else None,
                batch["num_atoms"] if energy_active else None,
            )
            values: dict[str, torch.Tensor] = {"structure_ids": batch["structure_ids"]}
            if force_active:
                if output.force_logits is None:
                    raise RuntimeError("active force model returned no logits")
                observed = force_error(
                    batch["force_prediction"],
                    batch["force_reference"],
                    config.model.force.target_mode,
                )
                values.update(
                    {
                        "force_logits": output.force_logits,
                        "force_labels": labels_from_thresholds(
                            observed, force_spec.thresholds
                        ),
                        "force_observed_errors": observed,
                        "force_expected_errors": expected_error(
                            output.force_logits, force_spec.representatives
                        ),
                    }
                )
                if include_raw:
                    values.update(
                        {
                            "force_prediction": batch["force_prediction"],
                            "force_reference": batch["force_reference"],
                        }
                    )
            else:
                if output.energy_logits is None:
                    raise RuntimeError("active energy model returned no logits")
                observed = energy_per_atom_error(
                    batch["energy_prediction"],
                    batch["energy_reference"],
                    batch["num_atoms"],
                )
                values.update(
                    {
                        "energy_logits": output.energy_logits,
                        "energy_labels": labels_from_thresholds(
                            observed, energy_spec.thresholds
                        ),
                        "energy_observed_errors": observed,
                        "energy_expected_errors": expected_error(
                            output.energy_logits, energy_spec.representatives
                        ),
                    }
                )
                if include_raw:
                    values.update(
                        {
                            "energy_prediction": batch["energy_prediction"],
                            "energy_reference": batch["energy_reference"],
                        }
                    )
            for name, tensor in values.items():
                collected[name].append(tensor.detach().cpu())
            atom_counts.append(batch["num_atoms"].detach().cpu())

    if not atom_counts:
        raise ValueError("prediction dataset produced no batches")
    predictions: dict[str, Any] = {
        name: torch.cat(parts) for name, parts in collected.items()
    }
    predictions["atom_offsets"] = offsets(atom_counts)
    spec = force_spec if force_active else energy_spec
    predictions[f"{target}_representatives"] = spec.representatives
    if force_active:
        predictions["force_target_mode"] = config.model.force.target_mode
        predictions["force_error_definition"] = force_error_definition(
            config.model.force.target_mode
        )
    logits = predictions[f"{target}_logits"]
    labels = predictions[f"{target}_labels"]
    observed = predictions[f"{target}_observed_errors"]
    metrics = {
        target: classification_metrics(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            observed.reshape(-1),
            spec.representatives,
        )
    }
    return SingleTargetResult(target, predictions, metrics)
