from pathlib import Path

import numpy as np
import pytest

from Uncertainty_Quantification.LLPR.llpr.curvature import load_curvature_progress


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("next_mismatch", "next index.*structure count"),
        ("next_too_large", "next index.*build size"),
        ("force_count", "force-component count"),
        ("energy_shape", "energy matrix shape"),
        ("non_finite", "finite"),
        ("non_symmetric", "symmetric"),
    ],
)
def test_progress_rejects_corrupted_state(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    path = tmp_path / "progress.npz"
    energy = np.eye(2, dtype=np.float64)
    force = np.eye(3, dtype=np.float64)
    structure_count = 2
    atom_count = 5
    force_component_count = 15
    next_index = 2
    expected_structure_count = 4

    if mutation == "next_mismatch":
        next_index = 3
    elif mutation == "next_too_large":
        structure_count = 5
        next_index = 5
    elif mutation == "force_count":
        force_component_count = 14
    elif mutation == "energy_shape":
        energy = np.eye(3, dtype=np.float64)
    elif mutation == "non_finite":
        energy[0, 0] = np.nan
    elif mutation == "non_symmetric":
        energy[0, 1] = 1.0
    else:
        raise AssertionError(f"unknown mutation {mutation}")

    np.savez_compressed(
        path,
        energy=energy,
        force=force,
        structure_count=np.array(structure_count),
        atom_count=np.array(atom_count),
        force_component_count=np.array(force_component_count),
        next_structure_index=np.array(next_index),
        identity=np.array("abc"),
    )

    with pytest.raises(ValueError, match=message):
        load_curvature_progress(
            path,
            expected_identity="abc",
            expected_energy_dimension=2,
            expected_force_dimension=3,
            expected_structure_count=expected_structure_count,
        )
