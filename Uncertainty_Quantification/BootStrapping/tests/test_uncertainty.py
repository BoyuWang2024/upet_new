from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def _write_member(path: Path, energy: np.ndarray) -> Path:
    np.savez(
        path,
        energy=energy,
        forces=np.repeat(energy[:, None], 3, axis=1),
        stress=np.repeat(energy[:, None, None], 9, axis=1).reshape(-1, 3, 3),
    )
    return path


def test_sample_std_uses_ddof_one(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import (
        streaming_mean_std,
    )

    paths = [
        _write_member(tmp_path / "a.npz", np.array([1.0])),
        _write_member(tmp_path / "b.npz", np.array([3.0])),
    ]
    result = streaming_mean_std(paths, "energy")

    assert result.member_count == 2
    assert result.mean == pytest.approx(np.array([2.0]))
    assert result.std == pytest.approx(np.array([np.sqrt(2.0)]))
    assert result.mean.dtype == np.float64


def test_gmd_excludes_self_and_counts_unordered_pairs(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import (
        pairwise_gmd,
    )

    paths = [
        _write_member(tmp_path / "a.npz", np.array([0.0])),
        _write_member(tmp_path / "b.npz", np.array([2.0])),
        _write_member(tmp_path / "c.npz", np.array([5.0])),
    ]

    assert pairwise_gmd(paths, "energy") == pytest.approx(np.array([10.0 / 3.0]))


def test_scalar_rms_reductions_follow_atom_layout() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.uncertainty import (
        scalar_rms_reductions,
    )

    num_atoms = np.array([2, 1])
    offsets = np.array([0, 2, 3])
    energy = scalar_rms_reductions(
        np.array([4.0, 3.0]), "energy", num_atoms=num_atoms, atom_offsets=offsets
    )
    forces = scalar_rms_reductions(
        np.array([[3.0, 4.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 6.0]]),
        "forces",
        num_atoms=num_atoms,
        atom_offsets=offsets,
    )
    stress = scalar_rms_reductions(
        np.stack([np.eye(3), np.ones((3, 3))]),
        "stress",
        num_atoms=num_atoms,
        atom_offsets=offsets,
    )

    assert energy == pytest.approx(np.array([2.0, 3.0]))
    assert forces == pytest.approx(np.array([np.sqrt(25 / 3), 1.0, np.sqrt(12)]))
    assert stress == pytest.approx(np.array([np.sqrt(1 / 3), 1.0]))


def test_compute_store_uncertainty_requires_publication_api() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import uncertainty

    assert callable(uncertainty.compute_store_uncertainty)
