"""Build the immutable raw UPET readout cache from configured datasets."""

from __future__ import annotations

import importlib.metadata
import itertools
import platform
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch

from ..artifacts import sha256_file
from ..cache import (
    SCHEMA_VERSION,
    RawStructure,
    build_raw_cache,
    prepare_raw_cache,
)
from ..checkpoint import load_upet_checkpoint
from ..config import ConfidenceConfig
from ..data import ConfidenceSample, DatasetIdentity, dataset_identity, iter_samples
from ..features import extract_readouts


def _version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _git_revision(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def _verified_identities(config: ConfidenceConfig) -> dict[str, DatasetIdentity]:
    identities: dict[str, DatasetIdentity] = {}
    for split in ("train", "validation", "test"):
        configured = getattr(config.data, split)
        identity = dataset_identity(
            configured.path, expected_sha256=configured.expected_sha256
        )
        if identity.sha256 != configured.expected_sha256:
            raise ValueError(
                f"dataset {split} SHA mismatch: "
                f"{identity.sha256} != {configured.expected_sha256}"
            )
        identities[split] = identity
    return identities


def _batches(
    samples: Iterator[ConfidenceSample],
    batch_size: int,
) -> Iterator[list[ConfidenceSample]]:
    batch: list[ConfidenceSample] = []
    for sample in samples:
        batch.append(sample)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _execution_policy(config: ConfidenceConfig) -> dict[str, Any]:
    device = torch.device(config.run.device)
    if config.run.amp and device.type != "cuda":
        raise ValueError("amp=true requires a CUDA device")
    return {
        "device": str(device),
        "model_dtype": "float32",
        "system_dtype": "float32",
        "autocast": bool(config.run.amp),
        "autocast_dtype": "float16" if config.run.amp else None,
    }


def _autocast_context(execution: dict[str, Any]) -> Any:
    if not execution["autocast"]:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True)


def _build_systems(
    samples: list[ConfidenceSample], model: Any, device: torch.device
) -> list[Any]:
    from metatomic.torch.systems_to_torch import systems_to_torch
    from vesin.metatomic import NeighborList

    systems = [
        systems_to_torch(
            sample.atoms,
            dtype=torch.float32,
            device=device,
            positions_requires_grad=False,
            cell_requires_grad=False,
        )
        for sample in samples
    ]
    neighbor_options = tuple(model.requested_neighbor_lists())
    for system in systems:
        for options in neighbor_options:
            NeighborList(
                options=options,
                length_unit="angstrom",
                check_consistency=False,
            ).add_neighbor_list(system, copy=True)
    return systems


def _verify_outputs(model: Any, config: ConfidenceConfig) -> dict[str, str]:
    outputs = model.supported_outputs()
    if not isinstance(outputs, Mapping):
        raise ValueError("model supported_outputs() must return a mapping")
    keys = {
        "energy_prediction": config.readouts.energy_prediction,
        "force_prediction": config.readouts.force_prediction,
        "energy_features": config.readouts.energy_features,
        "force_features": config.readouts.force_features,
    }
    missing = [key for key in keys.values() if key not in outputs]
    if missing:
        raise ValueError(f"checkpoint is missing required outputs: {missing}")
    return keys


def _raw_structures(
    *,
    path: Path,
    model: Any,
    config: ConfidenceConfig,
    device: torch.device,
    execution: dict[str, Any],
    expected_sha256: str,
) -> Iterator[RawStructure]:
    for samples in _batches(iter_samples(path), config.cache.batch_size):
        systems = _build_systems(samples, model, device)
        structure_ids = torch.tensor(
            [sample.index for sample in samples],
            dtype=torch.int64,
            device=device,
        )
        atom_counts = torch.tensor(
            [len(sample.atoms) for sample in samples],
            dtype=torch.int64,
            device=device,
        )
        with torch.inference_mode(), _autocast_context(execution):
            readouts = extract_readouts(
                model,
                systems,
                structure_ids,
                atom_counts,
                config.readouts,
            )
        for local_index, sample in enumerate(samples):
            start = int(readouts.atom_offsets[local_index].item())
            stop = int(readouts.atom_offsets[local_index + 1].item())
            yield RawStructure(
                structure_id=sample.index,
                atomic_numbers=torch.as_tensor(
                    sample.atoms.numbers,
                    dtype=torch.int64,
                ),
                force_prediction=readouts.force_prediction[start:stop],
                force_reference=torch.as_tensor(sample.force_reference),
                energy_prediction=readouts.energy_prediction[local_index],
                energy_reference=sample.energy_reference_total,
                force_features=readouts.force_features[start:stop],
                energy_features=readouts.energy_features[start:stop],
            )
    sha_after = sha256_file(path)
    if sha_after != expected_sha256:
        raise ValueError(
            f"dataset SHA changed during extraction: {sha_after} != {expected_sha256}"
        )


def _identity_payload(
    config: ConfidenceConfig,
    checkpoint_sha256: str,
    identities: dict[str, DatasetIdentity],
    outputs: dict[str, str],
    execution: dict[str, Any],
    feature_dims: tuple[int, int],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {"sha256": checkpoint_sha256},
        "splits": {
            split: {
                "sha256": identity.sha256,
                "structure_count": identity.structure_count,
                "atom_count": identity.atom_count,
                "force_component_count": identity.force_component_count,
            }
            for split, identity in identities.items()
        },
        "outputs": outputs,
        "features": {
            "force_dim": feature_dims[0],
            "energy_dim": feature_dims[1],
            "dtype": "float32",
        },
        "cache": {
            "batch_size": config.cache.batch_size,
            "shard_max_atoms": config.cache.shard_max_atoms,
        },
        "execution": execution,
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "metatomic": _version("metatomic-torch"),
            "metatrain": _version("metatrain"),
            "upet": _version("upet"),
            "upet_git": _git_revision(repo_root),
        },
    }


def build_cache(config: ConfidenceConfig) -> Path:
    """Verify inputs, extract raw readouts, and atomically publish the cache."""
    output_root = config.run.output_root / "cache"
    staging = prepare_raw_cache(output_root)
    execution = _execution_policy(config)
    device = torch.device(execution["device"])
    checkpoint = load_upet_checkpoint(
        config.checkpoint.path,
        expected_sha256=config.checkpoint.expected_sha256,
        device=device,
        dtype=torch.float32,
    )
    outputs = _verify_outputs(checkpoint.model, config)
    identities = _verified_identities(config)
    streams = {
        split: _raw_structures(
            path=getattr(config.data, split).path,
            model=checkpoint.model,
            config=config,
            device=device,
            execution=execution,
            expected_sha256=identities[split].sha256,
        )
        for split in ("train", "validation", "test")
    }
    try:
        first = {split: next(stream) for split, stream in streams.items()}
    except StopIteration as error:
        raise ValueError(
            "all cache splits must contain at least one structure"
        ) from error
    feature_dims = (
        int(first["train"].force_features.shape[1]),
        int(first["train"].energy_features.shape[1]),
    )
    for split, structure in first.items():
        dims = (
            int(structure.force_features.shape[1]),
            int(structure.energy_features.shape[1]),
        )
        if dims != feature_dims:
            raise ValueError(f"{split}: feature dimensions disagree with train")
    identity_payload = _identity_payload(
        config, checkpoint.sha256, identities, outputs, execution, feature_dims
    )
    return build_raw_cache(
        output_root=output_root,
        split_structures={
            split: itertools.chain([first[split]], stream)
            for split, stream in streams.items()
        },
        identity_payload=identity_payload,
        shard_max_atoms=config.cache.shard_max_atoms,
        staging=staging,
    )
