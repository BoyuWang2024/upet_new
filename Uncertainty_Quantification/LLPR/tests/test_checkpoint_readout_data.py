from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from Uncertainty_Quantification.LLPR.llpr.artifacts import sha256_file
from Uncertainty_Quantification.LLPR.llpr.checkpoint import load_checkpoint
from Uncertainty_Quantification.LLPR.llpr.config import FileIdentityConfig
from Uncertainty_Quantification.LLPR.llpr.data import (
    dataset_identity,
    iter_samples,
)
from Uncertainty_Quantification.LLPR.llpr.readout import discover_readout_layout


class FakeReadout(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.energy = torch.nn.Linear(2, 1)
        self.force = torch.nn.Linear(2, 3)
        self.last_layer_parameter_names = {
            "energy": ["energy.weight"],
            "non_conservative_forces": ["force.weight"],
        }


def test_layout_includes_weight_and_bias_in_stable_order() -> None:
    layout = discover_readout_layout(FakeReadout())

    assert [entry.name for entry in layout.energy.entries] == [
        "energy.weight",
        "energy.bias",
    ]
    assert [entry.name for entry in layout.force.entries] == [
        "force.weight",
        "force.bias",
    ]
    assert [entry.offset for entry in layout.energy.entries] == [0, 2]
    assert [entry.offset for entry in layout.force.entries] == [0, 6]
    assert layout.energy.dimension == 3
    assert layout.force.dimension == 9
    assert layout.total_dimension == 12
    assert layout.force.source_target == "non_conservative_forces"
    assert layout.force.target == "non_conservative_force"


def test_layout_rejects_missing_bias() -> None:
    model = FakeReadout()
    model.force.bias = None

    with pytest.raises(ValueError, match="bias.*force.bias"):
        discover_readout_layout(model)


def _write_two_structure_dataset(path: Path) -> list[tuple[float, np.ndarray]]:
    references: list[tuple[float, np.ndarray]] = []
    structures = [
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.7]]),
        Atoms("H2O", positions=[[0, 0, 0], [0.8, 0, 0], [0, 0.8, 0]]),
    ]
    for index, atoms in enumerate(structures):
        energy = -1.5 - index
        forces = np.arange(3 * len(atoms), dtype=np.float64).reshape(-1, 3) / 10
        atoms.info["source_index"] = 10 + index
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        references.append((energy, forces))
    write(path, structures, format="extxyz")
    return references


def test_dataset_iteration_is_deterministic_and_preserves_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "two.extxyz"
    references = _write_two_structure_dataset(path)

    samples = list(iter_samples(path))

    assert [sample.index for sample in samples] == [0, 1]
    assert [len(sample.atoms) for sample in samples] == [2, 3]
    for sample, (energy, forces) in zip(samples, references, strict=True):
        assert sample.energy_reference_total == pytest.approx(energy)
        np.testing.assert_allclose(sample.force_reference, forces)
        assert sample.force_reference.shape == (len(sample.atoms), 3)


def test_dataset_identity_binds_bytes_and_counts(tmp_path: Path) -> None:
    path = tmp_path / "two.extxyz"
    _write_two_structure_dataset(path)

    identity = dataset_identity(path)

    assert identity.sha256 == sha256_file(path)
    assert identity.structure_count == 2
    assert identity.atom_count == 5
    assert identity.force_component_count == 15


def test_checkpoint_hash_mismatch_fails_before_model_load(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"not a model")
    config = FileIdentityConfig(path=checkpoint, expected_sha256="0" * 64)

    with pytest.raises(ValueError, match="checkpoint SHA mismatch"):
        load_checkpoint(config, torch.device("cpu"), torch.float64)


@pytest.mark.llpr_n20
@pytest.mark.skipif(
    not bool(__import__("os").environ.get("UPET_RUN_LLPR_N20")),
    reason="set UPET_RUN_LLPR_N20=1 to load the real checkpoint",
)
def test_real_checkpoint_readout_dimensions() -> None:
    path = Path(
        "/home/lilong/code/UQ/upet_new/data/checkpoint/pet-omatpes-l-v0.1.0.ckpt"
    )
    loaded = load_checkpoint(
        FileIdentityConfig(
            path=path,
            expected_sha256=(
                "879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004"
            ),
        ),
        torch.device("cpu"),
        torch.float64,
    )
    layout = discover_readout_layout(loaded.model)

    assert layout.energy.dimension == 1026
    assert layout.force.dimension == 3078
    assert layout.total_dimension == 4104
