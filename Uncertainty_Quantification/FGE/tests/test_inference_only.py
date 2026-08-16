from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import cast

import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write

from Uncertainty_Quantification.FGE.fge import training
from Uncertainty_Quantification.FGE.fge.artifacts import sha256_file
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.inference_config import (
    ChunkPolicy,
    DatasetSourceConfig,
    EnsembleSourceConfig,
    InferenceConfig,
    InferenceOutputConfig,
    RuntimeInferenceConfig,
)
from Uncertainty_Quantification.FGE.fge.inference_data import DatasetChunk
from Uncertainty_Quantification.FGE.fge.inference_only import (
    predict_inference_dataset,
    validate_prediction_chunk,
)
from Uncertainty_Quantification.FGE.tests.test_inference_authority import (
    _completed_root,
)


class SimulatedStop(RuntimeError):
    pass


class LiteralChunkRuntime:
    def __init__(self, *, stop_on_chunk: int | None = None) -> None:
        self.stop_on_chunk = stop_on_chunk
        self.prepare_calls: list[str] = []
        self.applied_members: list[tuple[int, str]] = []
        self.current_chunk: DatasetChunk | None = None
        self.current_member = ""

    def load_reused_base(
        self,
        base_checkpoint: Path,
        expected_sha256: str,
        members_directory: Path,
    ) -> object:
        del base_checkpoint, expected_sha256, members_directory
        return object()

    def prepare_chunk(self, base: object, chunk: DatasetChunk) -> DatasetChunk:
        del base
        if chunk.chunk_index == self.stop_on_chunk:
            raise SimulatedStop
        self.prepare_calls.append(chunk.chunk_id)
        self.current_chunk = chunk
        return chunk

    def restore_and_apply(self, base: object, member_id: str) -> None:
        del base
        assert self.current_chunk is not None
        self.current_member = member_id
        self.applied_members.append((self.current_chunk.chunk_index, member_id))

    def infer_prepared(
        self, base: object, prepared: DatasetChunk
    ) -> dict[str, torch.Tensor]:
        del base
        member = int(self.current_member.removeprefix("member_"))
        return {
            "energy": torch.full(
                (len(prepared.records),), float(member), dtype=torch.float32
            ),
            "forces": torch.full(
                (prepared.atom_count, 3), float(member), dtype=torch.float32
            ),
            "stress": torch.full(
                (len(prepared.records), 3, 3),
                float(member),
                dtype=torch.float32,
            ),
        }


def _dataset(path: Path) -> None:
    systems: list[Atoms] = []
    for index, symbols in enumerate(("H", "HeH", "Li")):
        atom_count = 2 if symbols == "HeH" else 1
        atoms = Atoms(
            symbols,
            positions=[[float(atom), 0.0, 0.0] for atom in range(atom_count)],
            cell=[8.0, 8.0, 8.0],
            pbc=True,
        )
        atoms.info["structure_id"] = f"structure_{index:03d}"
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=float(index),
            forces=torch.zeros((len(atoms), 3)).numpy(),
            stress=torch.eye(3).numpy() * index,
        )
        systems.append(atoms)
    write(path, systems, format="extxyz")


def _config(tmp_path: Path) -> InferenceConfig:
    authority_root, result_sha = _completed_root(tmp_path / "authority")
    dataset_path = tmp_path / "dataset.extxyz"
    _dataset(dataset_path)
    return InferenceConfig(
        schema_version="upet.fge.inference.v1",
        ensemble=EnsembleSourceConfig(
            root=authority_root,
            result_manifest_sha256=result_sha,
            base_checkpoint=tmp_path / "base.ckpt",
            base_checkpoint_sha256="a" * 64,
            member_count=8,
        ),
        dataset=DatasetSourceConfig(
            label="matpes_train",
            path=dataset_path,
            expected_sha256=sha256_file(dataset_path),
            split="train",
            reference_availability={
                "energy": True,
                "forces": True,
                "stress": True,
            },
        ),
        chunking=ChunkPolicy(max_structures=2, max_atoms=3),
        output=InferenceOutputConfig(root=tmp_path / "inference-output"),
        runtime=RuntimeInferenceConfig(device="cpu", torch_threads=2),
    )


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_predicts_each_chunk_once_with_all_eight_members(tmp_path: Path) -> None:
    runtime = LiteralChunkRuntime()

    manifest_path = predict_inference_dataset(_config(tmp_path), runtime=runtime)
    manifest = _json(manifest_path)

    assert runtime.prepare_calls == ["chunk_000000", "chunk_000001"]
    assert runtime.applied_members == [
        (chunk, f"member_{index:03d}") for chunk in (0, 1) for index in range(1, 9)
    ]
    assert manifest["chunk_count"] == 2
    assert manifest["member_ids"] == [f"member_{index:03d}" for index in range(1, 9)]


def test_resume_reuses_only_hash_verified_chunks(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(SimulatedStop):
        predict_inference_dataset(
            config,
            runtime=LiteralChunkRuntime(stop_on_chunk=1),
        )

    resumed = LiteralChunkRuntime()
    predict_inference_dataset(config, runtime=resumed)

    assert resumed.prepare_calls == ["chunk_000001"]
    assert resumed.applied_members == [
        (1, f"member_{index:03d}") for index in range(1, 9)
    ]


def test_resume_rejects_a_tampered_orphan_chunk(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(SimulatedStop):
        predict_inference_dataset(
            config,
            runtime=LiteralChunkRuntime(stop_on_chunk=1),
        )
    chunk = config.output.root / "prediction" / "chunks" / "chunk_000000.pt"
    payload = torch.load(chunk, weights_only=True, map_location="cpu")
    assert isinstance(payload, dict)
    energy = payload["energy_prediction"]
    assert isinstance(energy, torch.Tensor)
    energy[0, 0] += 1.0
    torch.save(payload, chunk)

    with pytest.raises(HardFailure, match="chunk"):
        predict_inference_dataset(config, runtime=LiteralChunkRuntime())


def test_chunk_validator_rejects_wrong_member_order_and_shape(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest_path = predict_inference_dataset(
        config,
        runtime=LiteralChunkRuntime(),
    )
    manifest = _json(manifest_path)
    chunks = manifest["chunks"]
    assert isinstance(chunks, list)
    first = chunks[0]
    assert isinstance(first, dict)
    payload = torch.load(
        config.output.root / cast(str, first["path"]),
        weights_only=True,
        map_location="cpu",
    )
    assert isinstance(payload, dict)
    expected_identity = copy.deepcopy(payload["identity"])
    identity = payload["identity"]
    assert isinstance(identity, dict)
    identity["member_ids"] = tuple(reversed(identity["member_ids"]))

    with pytest.raises(HardFailure, match="identity"):
        validate_prediction_chunk(payload, expected_identity)

    identity.clear()
    identity.update(expected_identity)
    payload["energy_prediction"] = torch.zeros((7, 2), dtype=torch.float32)
    with pytest.raises(HardFailure, match="shape"):
        validate_prediction_chunk(payload, expected_identity)


def test_existing_conflicting_manifest_or_range_is_rejected(tmp_path: Path) -> None:
    config = _config(tmp_path)
    manifest_path = predict_inference_dataset(
        config,
        runtime=LiteralChunkRuntime(),
    )
    manifest = _json(manifest_path)
    chunks = manifest["chunks"]
    assert isinstance(chunks, list)
    first = chunks[0]
    assert isinstance(first, dict)
    first["stop_structure"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(HardFailure, match="manifest"):
        predict_inference_dataset(config, runtime=LiteralChunkRuntime())


def test_prediction_path_never_calls_training_or_backward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("training path was called")

    monkeypatch.setattr(training, "train_fge", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)

    predict_inference_dataset(_config(tmp_path), runtime=LiteralChunkRuntime())
