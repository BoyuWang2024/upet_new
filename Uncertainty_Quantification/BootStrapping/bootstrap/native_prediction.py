"""Native PET member inference into the canonical prediction store."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from numpy.typing import NDArray

from .artifacts import atomic_write_json, sha256_file
from .checkpoint import load_checkpoint_branch
from .config import BootstrapConfig
from .errors import HardFailure
from .head_policy import TrainablePolicyAudit, apply_pet_last_layer_policy
from .prediction import PredictionArrays, PredictionStore, TargetArrays


_UNITS = {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"}


def apply_member_state(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> TrainablePolicyAudit:
    """Apply exactly one audited PET last-layer state without touching frozen tensors."""

    audit = apply_pet_last_layer_policy(model)
    if set(state) != set(audit.trainable_names):
        missing = sorted(set(audit.trainable_names) - set(state))
        unexpected = sorted(set(state) - set(audit.trainable_names))
        raise HardFailure(
            f"member state keys differ; missing={missing}, unexpected={unexpected}"
        )
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name in audit.trainable_names:
            source = state[name]
            destination = parameters[name]
            if source.shape != destination.shape or source.dtype != destination.dtype:
                raise HardFailure(f"member tensor metadata differs for {name}")
            if not bool(torch.isfinite(source).all()):
                raise HardFailure(f"member tensor contains non-finite values: {name}")
            destination.copy_(source.to(device=destination.device))
    return audit


def load_pet_member_model(
    base_checkpoint: str | Path,
    member_checkpoint: str | Path,
    *,
    mode: str,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    """Build the base PET model and apply one canonical or supported v1 member."""

    try:
        import metatomic.torch  # noqa: F401
        from metatrain.utils.io import model_from_checkpoint

        base = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(base, dict):
            raise HardFailure("base PET checkpoint root must be a mapping")
        model = model_from_checkpoint(base, context="restart")
    except HardFailure:
        raise
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(f"could not construct PET model: {error}") from error
    model.to(device=device, dtype=dtype)
    apply_member_state(model, load_checkpoint_branch(member_checkpoint, mode))
    return model.eval()


def _read_atoms(path: Path, limit: int | None) -> list[Any]:
    try:
        from ase.io import read

        value = read(str(path), index=":" if limit is None else f":{limit}")
    except (ImportError, OSError, ValueError) as error:
        raise HardFailure(
            f"could not read prediction dataset {path}: {error}"
        ) from error
    atoms = value if isinstance(value, list) else [value]
    if not atoms:
        raise HardFailure(f"prediction dataset is empty: {path}")
    return atoms


def _targets(atoms: list[Any], split: str) -> TargetArrays:
    try:
        counts = np.asarray([len(item) for item in atoms], dtype=np.int64)
        offsets = np.concatenate(
            [np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)]
        )
        return TargetArrays(
            structure_ids=np.asarray(
                [
                    str(item.info.get("structure_id", f"{split}-{index}"))
                    for index, item in enumerate(atoms)
                ]
            ),
            num_atoms=counts,
            atom_offsets=offsets,
            energy=np.asarray(
                [float(item.get_potential_energy()) for item in atoms],
                dtype=np.float64,
            ),
            forces=np.concatenate([item.get_forces() for item in atoms]).astype(
                np.float64, copy=False
            ),
            stress=np.stack([item.get_stress(voigt=False) for item in atoms]).astype(
                np.float64, copy=False
            ),
        )
    except (KeyError, RuntimeError, ValueError) as error:
        raise HardFailure(f"could not extract {split} references: {error}") from error


def _predict_batch(
    model: torch.nn.Module,
    atoms: list[Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, NDArray[np.float64]]:
    try:
        from metatomic.torch import ModelOutput, systems_to_torch
        from metatrain.utils.neighbor_lists import (
            get_requested_neighbor_lists,
            get_system_with_neighbor_lists,
        )

        outputs = {
            "energy": ModelOutput(quantity="energy", unit="eV", per_atom=False),
            "non_conservative_forces": ModelOutput(
                quantity="force", unit="eV/A", per_atom=True
            ),
            "non_conservative_stress": ModelOutput(
                quantity="pressure", unit="eV/A^3", per_atom=False
            ),
        }
        requested = get_requested_neighbor_lists(model)
        systems = []
        for system in systems_to_torch(atoms, dtype=torch.float64):
            system = system.to(device=device, dtype=dtype)
            systems.append(get_system_with_neighbor_lists(system, requested))
        with torch.no_grad():
            result = model(systems, outputs)
        tensors = {
            "energy": result["energy"].block().values.reshape(-1),
            "forces": result["non_conservative_forces"].block().values.squeeze(-1),
            "stress": result["non_conservative_stress"].block().values.squeeze(-1),
        }
    except (ImportError, KeyError, RuntimeError, TypeError, ValueError) as error:
        raise HardFailure(f"PET prediction failed: {error}") from error
    return {
        name: tensor.detach().cpu().numpy().astype(np.float64, copy=False)
        for name, tensor in tensors.items()
    }


def _predict_dataset(
    model: torch.nn.Module,
    atoms: list[Any],
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> PredictionArrays:
    collected: dict[str, list[NDArray[np.float64]]] = {
        "energy": [],
        "forces": [],
        "stress": [],
    }
    for start in range(0, len(atoms), batch_size):
        values = _predict_batch(
            model, atoms[start : start + batch_size], device=device, dtype=dtype
        )
        for name in collected:
            collected[name].append(values[name])
    return PredictionArrays(
        energy=np.concatenate(collected["energy"]),
        forces=np.concatenate(collected["forces"]),
        stress=np.concatenate(collected["stress"]),
    )


def predict_run(
    config: BootstrapConfig,
    run_root: str | Path,
    *,
    splits: Iterable[str] | None = None,
    structure_limit: int | None = None,
) -> tuple[Path, ...]:
    """Run configured native PET inference and publish canonical member arrays."""

    root = Path(run_root).expanduser().resolve()
    selected_splits = tuple(config.prediction.splits if splits is None else splits)
    if not selected_splits or any(
        split not in config.prediction.splits for split in selected_splits
    ):
        raise HardFailure("requested prediction split is not enabled")
    if structure_limit is not None and structure_limit < 1:
        raise HardFailure("prediction structure_limit must be positive")
    device = torch.device(config.prediction.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise HardFailure("prediction requests CUDA but CUDA is unavailable")
    try:
        dtype = getattr(torch, config.training.precision)
    except AttributeError as error:
        raise HardFailure("unsupported prediction precision") from error
    publications: list[Path] = []
    for split in selected_splits:
        atoms = _read_atoms(getattr(config.data, split), structure_limit)
        store = PredictionStore(root / "predictions", split=split, units=_UNITS)
        store.write_targets(_targets(atoms, split))
        records = []
        for index in range(config.bootstrap.ensemble_size):
            checkpoint = (
                root / "members" / f"member_{index:03d}" / "checkpoints" / "best.pt"
            )
            for mode in config.prediction.parameter_modes:
                model = load_pet_member_model(
                    config.checkpoint.base_path,
                    checkpoint,
                    mode=mode,
                    device=device,
                    dtype=dtype,
                )
                values = _predict_dataset(
                    model,
                    atoms,
                    batch_size=config.prediction.batch_size,
                    device=device,
                    dtype=dtype,
                )
                publication = store.write_member(index, mode, values)
                records.append(
                    {
                        "member_index": index,
                        "mode": mode,
                        "path": str(publication.path.relative_to(store.split_root)),
                        "sha256": publication.sha256,
                        "shapes": publication.shapes,
                        "dtypes": publication.dtypes,
                    }
                )
                del model
        manifest = atomic_write_json(
            store.split_root / "manifest.json",
            {
                "schema": "upet.bootstrap.predictions/v1",
                "split": split,
                "units": _UNITS,
                "member_count": config.bootstrap.ensemble_size,
                "targets": {
                    "structure_limit": structure_limit,
                    "path": "targets.npz",
                    "sha256": sha256_file(store.targets_path),
                },
                "members": records,
            },
        )
        publications.append(manifest)
    return tuple(publications)
