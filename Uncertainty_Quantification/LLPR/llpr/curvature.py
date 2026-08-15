"""Huber-weighted last-layer curvature and resumable build stage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .artifacts import (
    RunPaths,
    atomic_json_dump,
    atomic_npz_save,
    load_verified_manifest,
    sha256_file,
    stage_identity,
)
from .checkpoint import load_checkpoint
from .config import LLPRConfig
from .curvature_progress import load_validated_curvature_progress
from .data import build_system, dataset_identity, iter_samples
from .observables import (
    compute_structure_jacobians,
    huber_curvature,
)
from .readout import discover_readout_layout
from .reuse import materialize_reused_stage


@dataclass
class CurvatureAccumulator:
    energy: torch.Tensor
    force: torch.Tensor
    structure_count: int = 0
    atom_count: int = 0
    force_component_count: int = 0

    @classmethod
    def zeros(cls, energy_dim: int, force_dim: int) -> "CurvatureAccumulator":
        return cls(
            energy=torch.zeros((energy_dim, energy_dim), dtype=torch.float64),
            force=torch.zeros((force_dim, force_dim), dtype=torch.float64),
        )

    def add_structure(
        self,
        *,
        energy_gradient: torch.Tensor,
        energy_residual: torch.Tensor,
        force_gradients: torch.Tensor,
        force_residuals: torch.Tensor,
        num_atoms: int,
        energy_weight: float,
        force_weight: float,
        energy_delta: float,
        force_delta: float,
    ) -> None:
        if num_atoms <= 0:
            raise ValueError("num_atoms must be positive")
        energy_gradient = energy_gradient.detach().to(dtype=torch.float64, device="cpu")
        force_gradients = force_gradients.detach().to(dtype=torch.float64, device="cpu")
        energy_residual = energy_residual.detach().to(dtype=torch.float64, device="cpu")
        force_residuals = (
            force_residuals.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        )
        if energy_gradient.numel() != self.energy.shape[0]:
            raise ValueError("energy gradient dimension mismatch")
        if force_gradients.shape != (
            force_residuals.numel(),
            self.force.shape[0],
        ):
            raise ValueError("force gradient dimension mismatch")
        if force_residuals.numel() != 3 * num_atoms:
            raise ValueError("force residual count must equal 3N")
        ce = huber_curvature(energy_residual, energy_delta)
        cf = huber_curvature(force_residuals, force_delta)
        self.energy.add_(
            energy_weight * ce * torch.outer(energy_gradient, energy_gradient)
        )
        weighted_force = force_gradients * cf[:, None]
        self.force.add_(
            (force_weight / (3 * num_atoms)) * weighted_force.T @ force_gradients
        )
        self.structure_count += 1
        self.atom_count += num_atoms
        self.force_component_count += 3 * num_atoms


def save_curvature_progress(
    path: Path,
    accumulator: CurvatureAccumulator,
    *,
    next_structure_index: int,
    identity: str,
) -> None:
    atomic_npz_save(
        path,
        {
            "energy": accumulator.energy.numpy(),
            "force": accumulator.force.numpy(),
            "structure_count": np.array(accumulator.structure_count),
            "atom_count": np.array(accumulator.atom_count),
            "force_component_count": np.array(accumulator.force_component_count),
            "next_structure_index": np.array(next_structure_index),
            "identity": np.array(identity),
        },
    )


def load_curvature_progress(
    path: Path,
    *,
    expected_identity: str,
    expected_energy_dimension: int,
    expected_force_dimension: int,
    expected_structure_count: int,
) -> tuple[CurvatureAccumulator, int]:
    progress = load_validated_curvature_progress(
        path,
        expected_identity=expected_identity,
        expected_energy_dimension=expected_energy_dimension,
        expected_force_dimension=expected_force_dimension,
        expected_structure_count=expected_structure_count,
    )
    accumulator = CurvatureAccumulator(
        energy=torch.from_numpy(progress.energy).to(torch.float64),
        force=torch.from_numpy(progress.force).to(torch.float64),
        structure_count=progress.structure_count,
        atom_count=progress.atom_count,
        force_component_count=progress.force_component_count,
    )
    return accumulator, progress.next_structure_index


def _curvature_identity(config: LLPRConfig, build_sha256: str) -> dict[str, object]:
    return stage_identity(
        "curvature",
        {
            "checkpoint_sha256": config.checkpoint.expected_sha256,
            "build_sha256": build_sha256,
            "curvature": config.curvature.model_dump(mode="json"),
            "matrix_dtype": config.runtime.matrix_dtype,
            "jacobian_backend": config.runtime.jacobian_backend,
            "force_component_chunk_size": (config.runtime.force_component_chunk_size),
        },
    )


def resolve_curvature_stage(config: LLPRConfig) -> Path:
    """Return a verified curvature stage, computing it only when not reused."""
    reuse = config.reuse.curvature if config.reuse is not None else None
    if reuse is None:
        return run_build(config)
    root = config.output.root / config.output.experiment
    return materialize_reused_stage(
        reuse.path,
        RunPaths(root).curvature,
        stage="curvature",
        identity=reuse.identity,
        expected_payload={
            "checkpoint_sha256": config.checkpoint.expected_sha256,
            "curvature": config.curvature.model_dump(mode="json"),
            "matrix_dtype": config.runtime.matrix_dtype,
            "jacobian_backend": config.runtime.jacobian_backend,
            "force_component_chunk_size": config.runtime.force_component_chunk_size,
        },
    )


def run_build(config: LLPRConfig) -> Path:
    """Compute or resume the canonical curvature for the build dataset."""
    if config.runtime.matrix_dtype != "float64":
        raise ValueError("LLPR matrices require float64")
    build_path = config.data.build
    if build_path is None:
        raise ValueError("numerical curvature build requires build data")
    data_identity = dataset_identity(build_path)
    if (
        config.data.build_expected_sha256 is not None
        and data_identity.sha256 != config.data.build_expected_sha256
    ):
        raise ValueError("build dataset SHA mismatch")
    identity = _curvature_identity(config, data_identity.sha256)
    identity_value = str(identity["identity"])
    root = config.output.root / config.output.experiment
    paths = RunPaths(root)
    stage_dir = paths.curvature / identity_value
    manifest_path = stage_dir / "manifest.json"
    if manifest_path.exists():
        load_verified_manifest(
            manifest_path, {"identity": identity_value}, verify_npz=True
        )
        return stage_dir
    stage_dir.mkdir(parents=True, exist_ok=True)
    progress_path = stage_dir / "progress.npz"

    device = torch.device(config.runtime.device)
    loaded = load_checkpoint(config.checkpoint, device=device, dtype=torch.float64)
    layout = discover_readout_layout(loaded.model)
    if progress_path.exists():
        accumulator, next_index = load_curvature_progress(
            progress_path,
            expected_identity=identity_value,
            expected_energy_dimension=layout.energy.dimension,
            expected_force_dimension=layout.force.dimension,
            expected_structure_count=data_identity.structure_count,
        )
    else:
        accumulator = CurvatureAccumulator.zeros(
            layout.energy.dimension, layout.force.dimension
        )
        next_index = 0

    for sample in iter_samples(build_path):
        if sample.index < next_index:
            continue
        system = build_system(sample, loaded.model, device=device, dtype=torch.float64)
        jacobians = compute_structure_jacobians(
            loaded.model,
            system,
            layout,
            config.runtime.jacobian_backend,
            config.runtime.force_component_chunk_size,
        )
        atom_count = len(sample.atoms)
        energy_reference_per_atom = sample.energy_reference_total / atom_count
        accumulator.add_structure(
            energy_gradient=jacobians.energy_jacobian,
            energy_residual=(
                jacobians.energy_pred_per_atom - energy_reference_per_atom
            ),
            force_gradients=jacobians.force_jacobian,
            force_residuals=(
                jacobians.force_pred
                - torch.from_numpy(sample.force_reference.reshape(-1))
            ),
            num_atoms=atom_count,
            energy_weight=config.curvature.energy_loss_weight,
            force_weight=config.curvature.force_loss_weight,
            energy_delta=config.curvature.energy_huber_delta,
            force_delta=config.curvature.force_huber_delta,
        )
        next_index = sample.index + 1
        if accumulator.structure_count % config.curvature.checkpoint_interval == 0:
            save_curvature_progress(
                progress_path,
                accumulator,
                next_structure_index=next_index,
                identity=identity_value,
            )

    accumulator.energy.copy_((accumulator.energy + accumulator.energy.T) / 2)
    accumulator.force.copy_((accumulator.force + accumulator.force.T) / 2)
    matrix_path = stage_dir / "curvature.npz"
    atomic_npz_save(
        matrix_path,
        {
            "energy": accumulator.energy.numpy(),
            "force": accumulator.force.numpy(),
        },
    )
    diagnostics_path = stage_dir / "diagnostics.json"
    atomic_json_dump(
        diagnostics_path,
        {
            "energy_dimension": layout.energy.dimension,
            "force_dimension": layout.force.dimension,
            "total_dimension": layout.total_dimension,
            "structure_count": accumulator.structure_count,
            "atom_count": accumulator.atom_count,
            "force_component_count": accumulator.force_component_count,
            "energy_symmetry_error": float(
                torch.linalg.matrix_norm(accumulator.energy - accumulator.energy.T)
            ),
            "force_symmetry_error": float(
                torch.linalg.matrix_norm(accumulator.force - accumulator.force.T)
            ),
        },
    )
    manifest = {
        **identity,
        "status": "complete",
        "layout_hash": layout.layout_hash,
        "dataset_sha256": data_identity.sha256,
        "files": {
            matrix_path.name: sha256_file(matrix_path),
            diagnostics_path.name: sha256_file(diagnostics_path),
        },
    }
    atomic_json_dump(manifest_path, manifest)
    progress_path.unlink(missing_ok=True)
    return stage_dir
