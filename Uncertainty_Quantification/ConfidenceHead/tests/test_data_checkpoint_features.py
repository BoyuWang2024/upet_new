from __future__ import annotations

import hashlib
import sys
import types
from collections.abc import Mapping, Sequence
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from Uncertainty_Quantification.ConfidenceHead.confidence_head.checkpoint import (
    LoadedCheckpoint,
    load_upet_checkpoint,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import (
    ReadoutConfig,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.data import (
    ENERGY_LABEL_CANDIDATES,
    FORCE_LABEL_CANDIDATES,
    ConfidenceSample,
    DatasetIdentity,
    dataset_identity,
    iter_samples,
    require_energy_label,
    require_force_label,
)
from Uncertainty_Quantification.ConfidenceHead.confidence_head.features import (
    ExtractedReadouts,
    extract_readouts,
)


class FakeSamples:
    def __init__(self, names: tuple[str, ...], values: list[list[int]]) -> None:
        self.names = names
        self.values = torch.tensor(values, dtype=torch.int64)


class FakeBlock:
    def __init__(
        self,
        values: torch.Tensor,
        sample_names: tuple[str, ...],
        samples: list[list[int]],
    ) -> None:
        self.values = values
        self.samples = FakeSamples(sample_names, samples)


class FakeTensorMap:
    def __init__(self, block: FakeBlock) -> None:
        self._block = block

    def block(self) -> FakeBlock:
        return self._block


def _atom_samples(atom_counts: tuple[int, ...]) -> list[list[int]]:
    return [
        [system, atom]
        for system, count in enumerate(atom_counts)
        for atom in range(count)
    ]


def _valid_outputs(
    config: ReadoutConfig,
    *,
    atom_counts: tuple[int, ...] = (2, 1),
) -> dict[str, FakeTensorMap]:
    atoms = _atom_samples(atom_counts)
    atom_count = sum(atom_counts)
    return {
        config.energy_prediction: FakeTensorMap(
            FakeBlock(
                torch.arange(len(atom_counts), dtype=torch.float64).reshape(-1, 1),
                ("system",),
                [[system] for system in range(len(atom_counts))],
            )
        ),
        config.force_prediction: FakeTensorMap(
            FakeBlock(
                torch.arange(3 * atom_count, dtype=torch.float64).reshape(-1, 3),
                ("system", "atom"),
                atoms,
            )
        ),
        config.energy_features: FakeTensorMap(
            FakeBlock(
                torch.arange(2 * atom_count, dtype=torch.float64).reshape(-1, 2),
                ("system", "atom"),
                atoms,
            )
        ),
        config.force_features: FakeTensorMap(
            FakeBlock(
                torch.arange(4 * atom_count, dtype=torch.float64).reshape(-1, 4),
                ("system", "atom"),
                atoms,
            )
        ),
    }


class FakeModel:
    def __init__(
        self,
        outputs: dict[str, FakeTensorMap],
        *,
        joint_error: BaseException | None = None,
    ) -> None:
        self.output_values = outputs
        self.capability_outputs = {
            key: object()
            for key in (
                ReadoutConfig().energy_prediction,
                ReadoutConfig().force_prediction,
                ReadoutConfig().energy_features,
                ReadoutConfig().force_features,
            )
        }
        self.calls: list[dict[str, object]] = []
        self.joint_error = joint_error

    def capabilities(self) -> types.SimpleNamespace:
        return types.SimpleNamespace(outputs=self.capability_outputs)

    def __call__(
        self,
        systems: list[object],
        outputs: dict[str, object],
        selected_atoms: object | None = None,
    ) -> dict[str, FakeTensorMap]:
        del systems, selected_atoms
        self.calls.append(outputs)
        if len(outputs) == 4 and self.joint_error is not None:
            raise self.joint_error
        return {key: self.output_values[key] for key in outputs}


def _extract(
    model: FakeModel,
    *,
    structure_ids: torch.Tensor | None = None,
    atom_counts: torch.Tensor | None = None,
    systems: list[object] | None = None,
    config: ReadoutConfig | None = None,
) -> ExtractedReadouts:
    return extract_readouts(
        model,
        [object(), object()] if systems is None else systems,
        (
            torch.tensor([41, 99], dtype=torch.int64)
            if structure_ids is None
            else structure_ids
        ),
        (
            torch.tensor([2, 1], dtype=torch.int64)
            if atom_counts is None
            else atom_counts
        ),
        ReadoutConfig() if config is None else config,
    )


def test_label_candidate_constants_are_exact_and_ordered() -> None:
    assert ENERGY_LABEL_CANDIDATES == (
        "energy",
        "free_energy",
        "dft_energy",
        "REF_energy",
    )
    assert FORCE_LABEL_CANDIDATES == (
        "forces",
        "force",
        "dft_forces",
        "REF_forces",
    )


def test_labels_use_candidate_order_across_supported_containers() -> None:
    atoms = Atoms("H2")
    calculator_forces = np.full((2, 3), 3.0)
    atoms.calc = SinglePointCalculator(atoms, energy=-2.0, forces=calculator_forces)
    atoms.info["free_energy"] = -1.0
    atoms.arrays["force"] = np.full((2, 3), 4.0)

    assert require_energy_label(atoms, index=7) == -2.0
    np.testing.assert_array_equal(
        require_force_label(atoms, index=7),
        calculator_forces,
    )


@pytest.mark.parametrize(
    "value, message",
    [
        (np.array([1.0, 2.0]), "scalar"),
        (np.inf, "finite"),
    ],
)
def test_energy_label_must_be_scalar_and_finite(
    value: object,
    message: str,
) -> None:
    atoms = Atoms("H")
    atoms.info["energy"] = value

    with pytest.raises(ValueError, match=rf"structure 12.*energy.*{message}"):
        require_energy_label(atoms, index=12)


@pytest.mark.parametrize(
    "value, message",
    [
        (np.zeros(6), r"\(2, 3\)"),
        (np.zeros((3, 3)), r"\(2, 3\)"),
        (np.array([[0.0, 0.0, 0.0], [0.0, np.nan, 0.0]]), "finite"),
    ],
)
def test_force_label_must_have_exact_shape_and_be_finite(
    value: np.ndarray,
    message: str,
) -> None:
    atoms = Atoms("H2")
    atoms.arrays["forces"] = value

    with pytest.raises(ValueError, match=rf"structure 13.*force.*{message}"):
        require_force_label(atoms, index=13)


def _write_dataset(path: Path) -> list[tuple[float, np.ndarray]]:
    references: list[tuple[float, np.ndarray]] = []
    structures = [Atoms("H2"), Atoms("H2O")]
    for index, atoms in enumerate(structures):
        energy = -1.25 - index
        forces = np.arange(3 * len(atoms), dtype=np.float64).reshape(-1, 3)
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        references.append((energy, forces))
    write(path, structures, format="extxyz")
    return references


def test_iter_samples_is_deterministic_and_returns_frozen_samples(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset.extxyz"
    references = _write_dataset(path)

    first = list(iter_samples(path))
    second = list(iter_samples(path))

    assert all(isinstance(sample, ConfidenceSample) for sample in first)
    assert [sample.index for sample in first] == [0, 1]
    assert [sample.index for sample in second] == [0, 1]
    assert [len(sample.atoms) for sample in first] == [2, 3]
    for sample, (energy, forces) in zip(first, references, strict=True):
        assert sample.energy_reference_total == pytest.approx(energy)
        np.testing.assert_array_equal(sample.force_reference, forces)
    with pytest.raises(FrozenInstanceError):
        first[0].index = 9  # type: ignore[misc]


def test_dataset_identity_binds_file_bytes_and_all_counts(tmp_path: Path) -> None:
    path = tmp_path / "dataset.extxyz"
    _write_dataset(path)

    identity = dataset_identity(path)

    assert isinstance(identity, DatasetIdentity)
    assert identity.path == path.resolve()
    assert identity.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    assert identity.structure_count == 2
    assert identity.atom_count == 5
    assert identity.force_component_count == 15
    with pytest.raises(FrozenInstanceError):
        identity.atom_count = 6  # type: ignore[misc]


def test_checkpoint_hash_mismatch_precedes_loader_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"not a checkpoint")
    imported: list[str] = []
    original_import = __import__

    def recording_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] = (),
        level: int = 0,
    ) -> Any:
        if name.startswith("metatrain"):
            imported.append(name)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", recording_import)

    with pytest.raises(ValueError, match="checkpoint SHA mismatch"):
        load_upet_checkpoint(
            checkpoint,
            "0" * 64,
            torch.device("cpu"),
            torch.float64,
        )

    assert imported == []


def test_checkpoint_loader_freezes_and_places_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_bytes(b"fake checkpoint")
    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    load_calls: list[str] = []

    class TinyModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))

    model = TinyModel()

    def fake_load_model(path: str) -> TinyModel:
        load_calls.append(path)
        return model

    metatrain = types.ModuleType("metatrain")
    metatrain_utils = types.ModuleType("metatrain.utils")
    metatrain_io = types.ModuleType("metatrain.utils.io")
    metatrain_io.load_model = fake_load_model  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "metatrain", metatrain)
    monkeypatch.setitem(sys.modules, "metatrain.utils", metatrain_utils)
    monkeypatch.setitem(sys.modules, "metatrain.utils.io", metatrain_io)

    loaded = load_upet_checkpoint(
        checkpoint,
        expected_sha,
        torch.device("cpu"),
        torch.float64,
    )

    assert isinstance(loaded, LoadedCheckpoint)
    assert loaded.model is model
    assert loaded.sha256 == expected_sha
    assert load_calls == [str(checkpoint)]
    assert model.training is False
    assert model.weight.device == torch.device("cpu")
    assert model.weight.dtype == torch.float64
    assert all(not parameter.requires_grad for parameter in model.parameters())
    with pytest.raises(FrozenInstanceError):
        loaded.sha256 = "f" * 64  # type: ignore[misc]


def test_extract_readouts_jointly_requests_exact_capability_outputs() -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    model = FakeModel(outputs)

    extracted = _extract(model, config=config)

    assert len(model.calls) == 1
    assert list(model.calls[0]) == [
        config.energy_prediction,
        config.force_prediction,
        config.energy_features,
        config.force_features,
    ]
    assert all(
        requested is model.capability_outputs[key]
        for key, requested in model.calls[0].items()
    )
    assert isinstance(extracted, ExtractedReadouts)
    assert extracted.structure_ids.dtype == torch.int64
    assert extracted.atom_counts.dtype == torch.int64
    assert extracted.atom_offsets.dtype == torch.int64
    torch.testing.assert_close(extracted.structure_ids, torch.tensor([41, 99]))
    torch.testing.assert_close(extracted.atom_counts, torch.tensor([2, 1]))
    torch.testing.assert_close(extracted.atom_offsets, torch.tensor([0, 2, 3]))
    assert extracted.energy_prediction.shape == (2,)
    assert extracted.force_prediction.shape == (3, 3)
    assert extracted.energy_features.shape == (3, 2)
    assert extracted.force_features.shape == (3, 4)
    assert (
        extracted.energy_features.data_ptr()
        == outputs[config.energy_features].block().values.data_ptr()
    )
    assert (
        extracted.force_features.data_ptr()
        == outputs[config.force_features].block().values.data_ptr()
    )
    with pytest.raises(FrozenInstanceError):
        extracted.energy_prediction = torch.zeros(2)  # type: ignore[misc]


@pytest.mark.parametrize(
    "missing_attribute",
    [
        "energy_prediction",
        "force_prediction",
        "energy_features",
        "force_features",
    ],
)
def test_extract_readouts_reports_each_missing_capability(
    missing_attribute: str,
) -> None:
    config = ReadoutConfig()
    model = FakeModel(_valid_outputs(config))
    missing_key = getattr(config, missing_attribute)
    del model.capability_outputs[missing_key]

    with pytest.raises(ValueError, match=rf"missing required outputs.*{missing_key}"):
        _extract(model, config=config)

    assert model.calls == []


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_extract_readouts_falls_back_to_two_paired_requests(
    error_type: type[BaseException],
) -> None:
    config = ReadoutConfig()
    model = FakeModel(
        _valid_outputs(config),
        joint_error=error_type("combined outputs are unsupported"),
    )

    extracted = _extract(model, config=config)

    assert extracted.energy_prediction.shape == (2,)
    assert extracted.force_prediction.shape == (3, 3)
    assert [list(call) for call in model.calls] == [
        [
            config.energy_prediction,
            config.force_prediction,
            config.energy_features,
            config.force_features,
        ],
        [config.energy_prediction, config.energy_features],
        [config.force_prediction, config.force_features],
    ]


def test_extract_readouts_does_not_fallback_for_unexpected_exceptions() -> None:
    config = ReadoutConfig()
    model = FakeModel(
        _valid_outputs(config),
        joint_error=TypeError("programming error"),
    )

    with pytest.raises(TypeError, match="programming error"):
        _extract(model, config=config)

    assert len(model.calls) == 1


def test_extract_readouts_rejects_shared_feature_storage() -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    shared = torch.arange(6, dtype=torch.float64).reshape(3, 2)
    outputs[config.energy_features]._block.values = shared
    outputs[config.force_features]._block.values = shared

    with pytest.raises(ValueError, match="energy_features.*force_features.*storage"):
        _extract(FakeModel(outputs), config=config)


@pytest.mark.parametrize(
    "output_attribute, values, message",
    [
        ("energy_prediction", torch.zeros(2, 2), r"\[S\]"),
        ("force_prediction", torch.zeros(3, 2), r"\[N, ?3\]"),
        ("energy_features", torch.zeros(3, 0), "positive"),
        ("force_features", torch.zeros(3), r"\[N, ?D"),
    ],
)
def test_extract_readouts_validates_every_output_shape(
    output_attribute: str,
    values: torch.Tensor,
    message: str,
) -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    output_key = getattr(config, output_attribute)
    outputs[output_key]._block.values = values

    with pytest.raises(ValueError, match=rf"{output_key}.*{message}"):
        _extract(FakeModel(outputs), config=config)


def test_extract_readouts_reports_original_id_for_structure_sample_order() -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    outputs[config.energy_prediction]._block.samples = FakeSamples(
        ("system",),
        [[0], [0]],
    )

    with pytest.raises(
        ValueError,
        match=rf"{config.energy_prediction}.*structure 99",
    ):
        _extract(FakeModel(outputs), config=config)


def test_extract_readouts_reports_original_id_for_wrong_structure_atom_count() -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    outputs[config.force_features]._block.samples = FakeSamples(
        ("system", "atom"),
        [[0, 0], [0, 1], [0, 2]],
    )

    with pytest.raises(
        ValueError,
        match=rf"{config.force_features}.*structure 99",
    ):
        _extract(FakeModel(outputs), config=config)


@pytest.mark.parametrize(
    "systems, structure_ids, atom_counts, message",
    [
        (
            [object(), object()],
            torch.tensor([[41, 99]], dtype=torch.int64),
            torch.tensor([2, 1], dtype=torch.int64),
            "structure_ids.*one-dimensional",
        ),
        (
            [object(), object()],
            torch.tensor([41, 99], dtype=torch.int64),
            torch.tensor([[2, 1]], dtype=torch.int64),
            "atom_counts.*one-dimensional",
        ),
        (
            [object()],
            torch.tensor([41, 99], dtype=torch.int64),
            torch.tensor([2, 1], dtype=torch.int64),
            "systems.*structure_ids",
        ),
        (
            [object(), object()],
            torch.tensor([41], dtype=torch.int64),
            torch.tensor([2, 1], dtype=torch.int64),
            "structure_ids.*atom_counts",
        ),
        (
            [object(), object()],
            torch.tensor([41, 99], dtype=torch.int64),
            torch.tensor([2, 0], dtype=torch.int64),
            "atom_counts.*positive",
        ),
    ],
)
def test_extract_readouts_validates_batch_metadata_before_model_call(
    systems: list[object],
    structure_ids: torch.Tensor,
    atom_counts: torch.Tensor,
    message: str,
) -> None:
    config = ReadoutConfig()
    model = FakeModel(_valid_outputs(config))

    with pytest.raises(ValueError, match=message):
        _extract(
            model,
            systems=systems,
            structure_ids=structure_ids,
            atom_counts=atom_counts,
            config=config,
        )

    assert model.calls == []
