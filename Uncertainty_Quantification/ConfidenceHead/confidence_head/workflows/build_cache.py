"""Build the immutable raw UPET readout cache from configured datasets."""

from __future__ import annotations

import importlib.metadata
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import metatomic.torch as mta
import torch

from ..cache import SCHEMA_VERSION, RawStructure, build_raw_cache
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
        identity = dataset_identity(configured.path)
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


def _raw_structures(
    *,
    path: Path,
    model: Any,
    config: ConfidenceConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Iterator[RawStructure]:
    for samples in _batches(iter_samples(path), config.cache.batch_size):
        systems = mta.systems_to_torch([sample.atoms for sample in samples])
        systems = [system.to(device=device, dtype=dtype) for system in systems]
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
        with torch.inference_mode():
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


def _identity_payload(
    config: ConfidenceConfig,
    checkpoint_sha256: str,
    identities: dict[str, DatasetIdentity],
) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[4]
    return {
        "schema_version": SCHEMA_VERSION,
        "checkpoint": {"sha256": checkpoint_sha256},
        "splits": {
            split: {
                "sha256": identity.sha256,
                "structures": identity.structure_count,
                "atoms": identity.atom_count,
                "force_components": identity.force_component_count,
            }
            for split, identity in identities.items()
        },
        "readouts": config.readouts.model_dump(),
        "cache": {
            "batch_size": config.cache.batch_size,
            "shard_max_atoms": config.cache.shard_max_atoms,
        },
        "execution": {
            "device": config.run.device,
            "amp": config.run.amp,
        },
        "versions": {
            "torch": torch.__version__,
            "metatomic": _version("metatomic-torch"),
            "metatrain": _version("metatrain"),
            "upet_git": _git_revision(repo_root),
        },
    }


def build_cache(config: ConfidenceConfig) -> Path:
    """Verify all inputs, extract raw readouts, and build the configured cache."""
    identities = _verified_identities(config)
    device = torch.device(config.run.device)
    dtype = torch.float32 if config.run.amp else torch.float64
    checkpoint = load_upet_checkpoint(
        config.checkpoint.path,
        expected_sha256=config.checkpoint.expected_sha256,
        device=device,
        dtype=dtype,
    )
    identity_payload = _identity_payload(config, checkpoint.sha256, identities)
    split_structures = {
        split: _raw_structures(
            path=getattr(config.data, split).path,
            model=checkpoint.model,
            config=config,
            device=device,
            dtype=dtype,
        )
        for split in ("train", "validation", "test")
    }
    return build_raw_cache(
        output_root=config.run.output_root / "cache",
        split_structures=split_structures,
        identity_payload=identity_payload,
        shard_max_atoms=config.cache.shard_max_atoms,
    )
