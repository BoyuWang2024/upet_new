from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.LLPR.llpr.calibration_progress import (
    load_calibration_progress,
    save_calibration_progress,
)


def test_calibration_progress_round_trip_and_resume_equivalence(
    tmp_path: Path,
) -> None:
    path = tmp_path / "progress.npz"
    energy = [torch.tensor([1.0, 2.0], dtype=torch.float64)]
    force = [torch.tensor([3.0, 4.0, 5.0], dtype=torch.float64)]
    q_values = {
        "energy": [
            [torch.tensor([0.1, 0.2], dtype=torch.float64)],
            [torch.tensor([0.3, 0.4], dtype=torch.float64)],
        ],
        "force": [[torch.tensor([0.5, 0.6, 0.7], dtype=torch.float64)]],
    }
    save_calibration_progress(
        path,
        identity="cal",
        next_structure_index=2,
        energy_residuals=energy,
        force_residuals=force,
        q_values=q_values,
    )

    loaded_energy, loaded_force, loaded_q, next_index = load_calibration_progress(
        path,
        expected_identity="cal",
        candidate_counts={"energy": 2, "force": 1},
    )
    loaded_energy.append(torch.tensor([6.0], dtype=torch.float64))
    loaded_force.append(torch.tensor([7.0], dtype=torch.float64))
    loaded_q["energy"][0].append(torch.tensor([0.8], dtype=torch.float64))

    assert next_index == 2
    torch.testing.assert_close(
        torch.cat(loaded_energy),
        torch.tensor([1.0, 2.0, 6.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.cat(loaded_force),
        torch.tensor([3.0, 4.0, 5.0, 7.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        torch.cat(loaded_q["energy"][0]),
        torch.tensor([0.1, 0.2, 0.8], dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        load_calibration_progress(
            path,
            expected_identity="different",
            candidate_counts={"energy": 2, "force": 1},
        )
