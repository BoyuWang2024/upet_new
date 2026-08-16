from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.artifacts import sha256_file
from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.fge.inference_authority import (
    open_ensemble_authority,
)
from Uncertainty_Quantification.FGE.fge.inference_config import (
    ChunkPolicy,
    DatasetSourceConfig,
    EnsembleSourceConfig,
    InferenceConfig,
    InferenceOutputConfig,
    RuntimeInferenceConfig,
)
from Uncertainty_Quantification.FGE.fge.members import ReadoutAudit
from Uncertainty_Quantification.FGE.fge.prediction import (
    PETBase,
    PETPredictionRuntime,
    PreparedPETChunk,
)
from Uncertainty_Quantification.FGE.fge.validation import validate_result
from Uncertainty_Quantification.FGE.tests.test_validation import (
    make_canonical_result,
)


def _inference_config(root: Path, manifest_sha256: str) -> InferenceConfig:
    return InferenceConfig(
        schema_version="upet.fge.inference.v1",
        ensemble=EnsembleSourceConfig(
            root=root,
            result_manifest_sha256=manifest_sha256,
            base_checkpoint=root.parent / "base.ckpt",
            base_checkpoint_sha256="a" * 64,
            member_count=8,
        ),
        dataset=DatasetSourceConfig(
            label="mad_test",
            path=root.parent / "mad-test.xyz",
            expected_sha256="d" * 64,
            split="test",
            reference_availability={
                "energy": True,
                "forces": True,
                "stress": False,
            },
        ),
        chunking=ChunkPolicy(max_structures=32, max_atoms=1024),
        output=InferenceOutputConfig(root=root.parent / "inference"),
        runtime=RuntimeInferenceConfig(device="cpu", torch_threads=2),
    )


def _completed_root(tmp_path: Path) -> tuple[Path, str]:
    config, root = make_canonical_result(
        tmp_path,
        project_name="upet_fge_full",
        member_count=8,
    )
    resolved_path = root / "config_resolved.yaml"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["project"] = {"name": "upet_fge_full"}
    resolved_path.write_text(json.dumps(resolved), encoding="utf-8")
    encoded = json.dumps(
        resolved,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    config_identity = {"sha256": hashlib.sha256(encoded).hexdigest()}
    training_path = root / "training" / "manifest.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    training["config_resolved"] = resolved
    training["config_identity"] = config_identity
    training_path.write_text(json.dumps(training), encoding="utf-8")
    prediction_path = root / "prediction" / "manifest.json"
    prediction = json.loads(prediction_path.read_text(encoding="utf-8"))
    prediction["config_identity"] = config_identity
    prediction_path.write_text(json.dumps(prediction), encoding="utf-8")
    for stage in ("train", "predict", "evaluate"):
        preflight_path = root / "preflight" / f"{stage}.json"
        preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
        preflight["config_identity"] = config_identity
        preflight_path.write_text(json.dumps(preflight), encoding="utf-8")
    config.sanitized = lambda: dict(resolved)
    validate_result(config, root)
    return root, sha256_file(root / "result_manifest.json")


def test_authority_reopens_completed_result_and_binds_all_members(
    tmp_path: Path,
) -> None:
    root, manifest_sha256 = _completed_root(tmp_path)
    config = _inference_config(root, manifest_sha256)

    authority = open_ensemble_authority(config)

    assert authority.member_ids == tuple(f"member_{index:03d}" for index in range(1, 9))
    assert len(authority.member_sha256) == 8
    assert authority.result_manifest_sha256 == manifest_sha256
    assert authority.base_checkpoint_sha256 == "a" * 64
    assert authority.members_directory == root / "training" / "members"
    assert str(root) not in repr(authority.canonical_identity())


def test_authority_rejects_tampered_member(tmp_path: Path) -> None:
    root, manifest_sha256 = _completed_root(tmp_path)
    member = root / "training" / "members" / "member_004.pt"
    member.write_bytes(member.read_bytes() + b"tampered")

    with pytest.raises(HardFailure, match="member"):
        open_ensemble_authority(_inference_config(root, manifest_sha256))


def test_authority_rejects_wrong_result_manifest_identity(tmp_path: Path) -> None:
    root, _ = _completed_root(tmp_path)

    with pytest.raises(HardFailure, match="result manifest"):
        open_ensemble_authority(_inference_config(root, "0" * 64))


def test_load_reused_base_checks_identity_before_deserialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "base.ckpt"
    checkpoint.write_bytes(b"not authenticated")
    called = False

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal called
        del args, kwargs
        called = True
        raise AssertionError("checkpoint must not be deserialized")

    monkeypatch.setattr(torch, "load", forbidden)

    with pytest.raises(HardFailure, match="checkpoint SHA256"):
        PETPredictionRuntime().load_reused_base(
            checkpoint,
            "0" * 64,
            tmp_path / "verified-members",
        )
    assert called is False


class _FakeSystem:
    def __init__(self, name: str) -> None:
        self.name = name

    def to(self, *, dtype: torch.dtype) -> _FakeSystem:
        assert dtype == torch.float32
        return self


class _FakeTensorMap:
    def __init__(self, values: torch.Tensor) -> None:
        self._values = values

    def block(self) -> SimpleNamespace:
        return SimpleNamespace(values=self._values)


def _empty_audit() -> ReadoutAudit:
    return ReadoutAudit(
        names=(),
        tensor_count=0,
        scalar_count=0,
        shapes=(),
        dtypes=(),
        all_finite=True,
    )


def test_pet_runtime_prepares_one_ase_chunk_then_evaluates_three_targets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import metatomic.torch
    import metatrain.utils.evaluate_model
    import metatrain.utils.neighbor_lists

    converted = [_FakeSystem("a"), _FakeSystem("b")]
    conversion_calls: list[tuple[object, ...]] = []
    neighbor_calls: list[str] = []

    def systems_to_torch(atoms: list[object], *, dtype: torch.dtype) -> list[object]:
        assert dtype == torch.float32
        conversion_calls.append(tuple(atoms))
        return cast(list[object], converted)

    def requested(model: object) -> str:
        del model
        neighbor_calls.append("requested")
        return "neighbors"

    def prepared(system: _FakeSystem, request: str) -> tuple[str, str]:
        assert request == "neighbors"
        return ("prepared", system.name)

    def evaluated(
        model: object,
        systems: list[object],
        targets: dict[str, object],
        *,
        is_training: bool,
    ) -> dict[str, _FakeTensorMap]:
        del model
        assert systems == [("prepared", "a"), ("prepared", "b")]
        assert set(targets) == {
            "energy",
            "non_conservative_forces",
            "non_conservative_stress",
        }
        assert is_training is False
        return {
            "energy": _FakeTensorMap(torch.tensor([[1.0], [2.0]])),
            "non_conservative_forces": _FakeTensorMap(
                torch.arange(9, dtype=torch.float32).reshape(3, 3, 1)
            ),
            "non_conservative_stress": _FakeTensorMap(
                torch.arange(18, dtype=torch.float32).reshape(2, 3, 3, 1)
            ),
        }

    monkeypatch.setattr(metatomic.torch, "systems_to_torch", systems_to_torch)
    monkeypatch.setattr(
        metatrain.utils.neighbor_lists,
        "get_requested_neighbor_lists",
        requested,
    )
    monkeypatch.setattr(
        metatrain.utils.neighbor_lists,
        "get_system_with_neighbor_lists",
        prepared,
    )
    monkeypatch.setattr(
        metatrain.utils.evaluate_model,
        "evaluate_model",
        evaluated,
    )
    model = SimpleNamespace(
        dataset_info=SimpleNamespace(
            targets={
                "energy": object(),
                "non_conservative_forces": object(),
                "non_conservative_stress": object(),
            }
        ),
        eval=lambda: None,
    )
    base = PETBase(
        model=cast(Any, model),
        state_dict={},
        audit=_empty_audit(),
        members_directory=tmp_path / "verified-members",
        base_sha256="a" * 64,
    )
    runtime = PETPredictionRuntime()
    atoms = (object(), object())

    prepared_chunk = runtime.prepare_ase_chunk(base, atoms)
    output = runtime.infer_prepared(base, prepared_chunk)

    assert isinstance(prepared_chunk, PreparedPETChunk)
    assert conversion_calls == [atoms]
    assert neighbor_calls == ["requested"]
    assert tuple(cast(torch.Tensor, output["energy"]).shape) == (2,)
    assert tuple(cast(torch.Tensor, output["forces"]).shape) == (3, 3)
    assert tuple(cast(torch.Tensor, output["stress"]).shape) == (2, 3, 3)


def test_authority_reads_exact_training_manifest_member_hashes(tmp_path: Path) -> None:
    root, manifest_sha256 = _completed_root(tmp_path)
    training = json.loads(
        (root / "training" / "manifest.json").read_text(encoding="utf-8")
    )

    authority = open_ensemble_authority(_inference_config(root, manifest_sha256))

    assert authority.member_sha256 == tuple(
        member["sha256"] for member in training["members"]
    )
