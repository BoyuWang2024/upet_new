from __future__ import annotations

import hashlib
import sys
import traceback
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
        *,
        components: list[FakeSamples] | None = None,
        properties: FakeSamples | None = None,
    ) -> None:
        self.values = values
        self.samples = FakeSamples(sample_names, samples)
        self.components = [] if components is None else components
        self.properties = (
            FakeSamples(("property",), [[0]]) if properties is None else properties
        )


class FakeTensorMap:
    def __init__(self, blocks: FakeBlock | list[FakeBlock]) -> None:
        self._blocks = blocks if isinstance(blocks, list) else [blocks]
        self._block = self._blocks[0]

    def __len__(self) -> int:
        return len(self._blocks)

    def block(self) -> FakeBlock:
        return self._block


def _feature_properties(size: int) -> FakeSamples:
    return FakeSamples(
        ("feature",),
        [[feature] for feature in range(size)],
    )


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
                torch.arange(1, atom_count + 1, dtype=torch.float64).reshape(-1, 1),
                ("system", "atom"),
                atoms,
            )
        ),
        config.force_prediction: FakeTensorMap(
            FakeBlock(
                torch.arange(3 * atom_count, dtype=torch.float64).reshape(-1, 3, 1),
                ("system", "atom"),
                atoms,
                components=[
                    FakeSamples(("xyz",), [[0], [1], [2]]),
                ],
            )
        ),
        config.energy_features: FakeTensorMap(
            FakeBlock(
                torch.arange(2 * atom_count, dtype=torch.float64).reshape(-1, 2),
                ("system", "atom"),
                atoms,
                properties=_feature_properties(2),
            )
        ),
        config.force_features: FakeTensorMap(
            FakeBlock(
                torch.arange(4 * atom_count, dtype=torch.float64).reshape(-1, 4),
                ("system", "atom"),
                atoms,
                properties=_feature_properties(4),
            )
        ),
    }


class FakeModel:
    def __init__(
        self,
        outputs: dict[str, FakeTensorMap],
        *,
        joint_error: BaseException | None = None,
        fallback_error: BaseException | None = None,
    ) -> None:
        self.output_values = outputs
        self.supported_output_descriptors = {key: object() for key in outputs}
        self.supported_outputs_calls = 0
        self.calls: list[dict[str, object]] = []
        self.joint_error = joint_error
        self.fallback_error = fallback_error

    def supported_outputs(self) -> dict[str, object]:
        self.supported_outputs_calls += 1
        return self.supported_output_descriptors

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
        if len(outputs) == 2 and self.fallback_error is not None:
            raise self.fallback_error
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


def test_readout_config_uses_real_non_conservative_force_key() -> None:
    assert ReadoutConfig().force_prediction == "non_conservative_forces"


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


def test_dataset_identity_rejects_bytes_changed_during_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "dataset.extxyz"
    _write_dataset(path)
    calls: list[Path] = []
    hashes = iter(("a" * 64, "b" * 64))

    def changing_sha256(candidate: Path) -> str:
        calls.append(candidate)
        return next(hashes)

    monkeypatch.setattr(
        "Uncertainty_Quantification.ConfidenceHead.confidence_head.data.sha256_file",
        changing_sha256,
    )

    with pytest.raises(ValueError, match="changed|SHA"):
        dataset_identity(path)

    assert calls == [path, path]


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


def test_extract_readouts_uses_supported_outputs_and_direct_descriptors() -> None:
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
    assert model.supported_outputs_calls == 1
    assert all(
        requested is model.supported_output_descriptors[key]
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
    torch.testing.assert_close(
        extracted.energy_prediction,
        torch.tensor([3.0, 3.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        extracted.force_prediction,
        outputs[config.force_prediction].block().values.squeeze(-1),
    )
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
def test_extract_readouts_reports_each_missing_supported_output(
    missing_attribute: str,
) -> None:
    config = ReadoutConfig()
    model = FakeModel(_valid_outputs(config))
    missing_key = getattr(config, missing_attribute)
    del model.supported_output_descriptors[missing_key]

    with pytest.raises(ValueError, match=rf"missing required outputs.*{missing_key}"):
        _extract(model, config=config)

    assert model.calls == []


@pytest.mark.parametrize(
    "first, second",
    [
        ("energy_prediction", "force_prediction"),
        ("energy_prediction", "energy_features"),
        ("energy_prediction", "force_features"),
        ("force_prediction", "energy_features"),
        ("force_prediction", "force_features"),
        ("energy_features", "force_features"),
    ],
)
def test_extract_readouts_rejects_duplicate_keys_before_model_call(
    first: str,
    second: str,
) -> None:
    defaults = ReadoutConfig()
    config = defaults.model_copy(
        update={second: getattr(defaults, first)},
    )
    model = FakeModel(_valid_outputs(config))

    with pytest.raises(ValueError, match="distinct|unique"):
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


@pytest.mark.parametrize("joint_type", [ValueError, RuntimeError])
def test_fallback_failure_preserves_joint_and_fallback_errors(
    joint_type: type[BaseException],
) -> None:
    config = ReadoutConfig()
    model = FakeModel(
        _valid_outputs(config),
        joint_error=joint_type("joint readouts failed"),
        fallback_error=RuntimeError("paired fallback failed"),
    )

    with pytest.raises(BaseException) as caught:
        _extract(model, config=config)

    rendered = "".join(
        traceback.TracebackException.from_exception(caught.value).format()
    )
    assert "joint readouts failed" in rendered
    assert "paired fallback failed" in rendered
    assert len(model.calls) == 2


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
    outputs[config.force_features].block().properties = _feature_properties(2)

    with pytest.raises(ValueError, match="energy_features.*force_features.*storage"):
        _extract(FakeModel(outputs), config=config)


def test_force_prediction_normalizes_real_component_and_property_axes() -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    expected = torch.tensor(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
        dtype=torch.float64,
    )
    outputs[config.force_prediction].block().values = expected.unsqueeze(-1)

    extracted = _extract(FakeModel(outputs), config=config)

    torch.testing.assert_close(extracted.force_prediction, expected)


@pytest.mark.parametrize("schema_part", ["components", "properties"])
def test_force_prediction_rejects_invalid_tensor_map_schema(
    schema_part: str,
) -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    block = outputs[config.force_prediction].block()
    if schema_part == "components":
        block.components = [
            FakeSamples(("xyz",), [[0], [2], [1]]),
        ]
    else:
        block.properties = FakeSamples(("property",), [[0], [1]])
        block.values = torch.zeros(3, 3, 2)

    with pytest.raises(
        ValueError,
        match=rf"{config.force_prediction}.*{schema_part}|"
        rf"{config.force_prediction}.*schema",
    ):
        _extract(FakeModel(outputs), config=config)


def test_extract_readouts_rejects_multiple_blocks_with_output_key() -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    key = config.energy_features
    block = outputs[key].block()
    outputs[key] = FakeTensorMap([block, block])

    with pytest.raises(ValueError, match=rf"{key}.*one block|{key}.*single block"):
        _extract(FakeModel(outputs), config=config)


def test_extract_readouts_rejects_empty_batch_before_model_call() -> None:
    config = ReadoutConfig()
    model = FakeModel(_valid_outputs(config))

    with pytest.raises(ValueError, match="empty|non-empty"):
        _extract(
            model,
            systems=[],
            structure_ids=torch.empty(0, dtype=torch.int64),
            atom_counts=torch.empty(0, dtype=torch.int64),
            config=config,
        )

    assert model.calls == []
    assert model.supported_outputs_calls == 0


@pytest.mark.parametrize(
    "output_attribute, values, message",
    [
        ("energy_prediction", torch.zeros(3, 2), r"\[N, ?1\]"),
        ("force_prediction", torch.zeros(3, 2, 1), r"\[N, ?3, ?1\]"),
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
        ("system", "atom"),
        [[0, 0], [0, 1], [0, 2]],
    )

    with pytest.raises(
        ValueError,
        match=rf"{config.energy_prediction}.*structure 99",
    ):
        _extract(FakeModel(outputs), config=config)


def test_structure_sample_prefix_mismatch_reports_first_expected_id() -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    outputs[config.energy_prediction]._block.samples = FakeSamples(
        ("system", "atom"),
        [[1, 0]],
    )
    outputs[config.energy_prediction]._block.values = torch.ones(1, 1)

    with pytest.raises(
        ValueError,
        match=rf"{config.energy_prediction}.*structure 41",
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


def test_atom_sample_prefix_mismatch_reports_first_expected_id_and_key() -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    outputs[config.force_prediction]._block.samples = FakeSamples(
        ("system", "atom"),
        [[0, 1], [1, 0]],
    )
    outputs[config.force_prediction]._block.values = torch.zeros(2, 3, 1)

    with pytest.raises(
        ValueError,
        match=rf"{config.force_prediction}.*structure 41",
    ):
        _extract(FakeModel(outputs), config=config)


def test_extract_readouts_rejects_different_views_of_same_storage() -> None:
    config = ReadoutConfig()
    outputs = _valid_outputs(config)
    shared = torch.arange(7, dtype=torch.float64)
    energy_view = shared[:6].reshape(3, 2)
    force_view = shared[1:7].reshape(3, 2)
    assert energy_view.data_ptr() != force_view.data_ptr()
    assert (
        energy_view.untyped_storage().data_ptr()
        == force_view.untyped_storage().data_ptr()
    )
    outputs[config.energy_features].block().values = energy_view
    outputs[config.force_features].block().values = force_view
    outputs[config.force_features].block().properties = _feature_properties(2)

    with pytest.raises(ValueError, match="energy_features.*force_features.*storage"):
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
