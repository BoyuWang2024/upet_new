"""Contract tests for native FGE training orchestration.

Each test names the production regression it protects.  The fake runtime uses
real ``torch.nn.Parameter`` values so optimizer and frozen-state behavior is
observable without an external PET checkpoint or dataset.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from Uncertainty_Quantification.FGE.fge import (
    FGEConfig,
    HardFailure,
    asymmetric_triangular_lr,
    load_config,
)
from Uncertainty_Quantification.FGE.fge.members import pack_member

# RED contract: Task 6 supplies this native orchestration module.  Do not add
# production code until this import has failed once on the remote test checkout.
from Uncertainty_Quantification.FGE.fge.training import (  # noqa: F401
    PETTrainingRuntime,
    TrainingBatchResult,
    TrainingRuntime,
    train_fge,
)
from Uncertainty_Quantification.FGE.tests.conftest import SHA_BASE, SHA_TEST, SHA_TRAIN


class _ReadoutModel(torch.nn.Module):
    """Exactly 12 readout tensors / 13,338 trainable scalars plus frozen state."""

    def __init__(self) -> None:
        super().__init__()
        self.node_last_layers = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.ones(1)) for _ in range(11)]
            + [torch.nn.Parameter(torch.ones(13_327))]
        )
        self.frozen = torch.nn.Parameter(torch.tensor([3.0]), requires_grad=False)
        self.register_buffer("frozen_buffer", torch.tensor([5.0]))


class _TorchRuntime:
    """Small real-torch boundary for the orchestration contract.

    A missing optimizer step, skipped loss term, incorrectly applied EMA, or
    absent endpoint reload check changes these externally visible records.
    """

    def __init__(self, batches: int = 5) -> None:
        self.model = _ReadoutModel()
        # Runtime loaders yield already-collated batches.  This fixture makes
        # the incomplete tail absent before the orchestration loop begins.
        self.train_loader: Iterable[object] = tuple(range(batches // 4))
        self.lrs: list[float] = []
        self.loss_term_sets: list[tuple[str, ...]] = []
        self.frozen_checks = 0
        self.ema_validations = 0
        self.raw_validations = 0
        self.endpoint_paths: list[Path] = []
        self.resume_identity_value = {
            "config": "c" * 64,
            "code": "d" * 64,
            "base": SHA_BASE,
            "data": SHA_TRAIN,
        }

    def train_batch(self, batch: object, *, lr: float) -> TrainingBatchResult:
        del batch
        self.lrs.append(lr)
        self.loss_term_sets.append(
            (
                "energy",
                "forces",
                "virial",
                "non_conservative_forces",
                "non_conservative_stress",
            )
        )
        loss = sum(parameter.square().sum() for parameter in self.model.parameters())
        return TrainingBatchResult(loss=loss, loss_terms=self.loss_term_sets[-1])

    def validate(self, *, use_ema: bool) -> dict[str, float]:
        if use_ema:
            self.ema_validations += 1
        else:
            self.raw_validations += 1
        return {"loss_total": 0.0}

    def assert_frozen(self) -> None:
        assert self.model.frozen.item() == 3.0
        assert self.model.frozen_buffer.item() == 5.0
        self.frozen_checks += 1

    def reload_and_smoke(self, member_path: Path) -> bool:
        self.endpoint_paths.append(member_path)
        return member_path.is_file()

    def resume_identity(self) -> dict[str, str]:
        return dict(self.resume_identity_value)

    def manifest_metadata(self) -> dict[str, dict[str, str]]:
        return {
            "dependency_snapshot": {
                "torch": "test-torch",
                "metatrain": "test-metatrain",
            },
            "training_code_identity": {
                "commit": "a" * 40,
                "dirty_sha256": "b" * 64,
            },
            "artifact_writer_code_identity": {
                "commit": "c" * 40,
                "dirty_sha256": "d" * 64,
            },
            "validator_code_identity": {
                "commit": "e" * 40,
                "dirty_sha256": "f" * 64,
            },
        }


def _config(tmp_path: Path, *, resume: bool = False) -> Any:
    """Use a tiny n20-shaped runtime scale with independent on-disk identities."""

    return SimpleNamespace(
        project=SimpleNamespace(name="upet_fge_n20_cpu"),
        paths=SimpleNamespace(output_root=tmp_path),
        identity=SimpleNamespace(
            base_checkpoint_sha256=SHA_BASE,
            train_data_sha256=SHA_TRAIN,
            val_data_sha256=SHA_TRAIN,
            test_data_sha256=SHA_TEST,
        ),
        training=SimpleNamespace(
            batch_size=4,
            drop_last=True,
            weight_decay=0.0,
            resume=resume,
            expected_readout_tensor_count=12,
            expected_readout_parameter_count=13_338,
        ),
        fge=SimpleNamespace(
            member_count=2,
            cycles=2,
            epochs_per_cycle=2,
            lr_min=1e-8,
            lr_max=1e-7,
            rise_fraction=0.2,
        ),
        ema=SimpleNamespace(enabled=True, decay=0.999, member_source="raw_endpoint"),
        scientific=SimpleNamespace(
            training=SimpleNamespace(
                path_feasibility_only=True,
                split_leakage=True,
                scientific_evaluation=False,
                inference_only=False,
            )
        ),
        sanitized=lambda: {
            "schema_version": "upet.fge.v1",
            "project": {
                "name": "upet_fge_n20_cpu",
                "method": "FGE",
                "backend": "upet",
            },
            "paths": {
                "base_checkpoint": {"role": "base_checkpoint", "sha256": SHA_BASE},
                "train_data": {"role": "train_data", "sha256": SHA_TRAIN},
                "val_data": {"role": "val_data", "sha256": SHA_TRAIN},
                "test_data": {"role": "test_data", "sha256": SHA_TEST},
                "output_root": {"role": "output_root"},
            },
            "identity": {
                "base_checkpoint_sha256": SHA_BASE,
                "train_data_sha256": SHA_TRAIN,
                "val_data_sha256": SHA_TRAIN,
                "test_data_sha256": SHA_TEST,
            },
            "data": {"format": "extxyz"},
            "training": {"mode": "readout_only_official_upet"},
            "fge": {"member_count": 2},
            "ema": {"member_source": "raw_endpoint"},
            "prediction": {"split": "test"},
            "evaluation": {"formula_version": "legacy_upet_fge_v1"},
            "scientific": {
                "training": {
                    "path_feasibility_only": True,
                    "split_leakage": True,
                    "scientific_evaluation": False,
                    "inference_only": False,
                },
                "evaluation": {
                    "path_feasibility_only": True,
                    "split_leakage": True,
                    "scientific_evaluation": False,
                    "inference_only": False,
                },
            },
        },
    )


def test_training_uses_one_optimizer_and_raw_endpoint_members_across_cycles(
    tmp_path: Path,
) -> None:
    """Creating one optimizer per cycle or applying EMA at endpoints is a bug."""

    runtime = _TorchRuntime(batches=5)
    config = _config(tmp_path)

    manifest_path = train_fge(config, runtime=runtime)

    # floor(5 / 4) * 2 epochs * 2 cycles: exact drop_last update contract.
    assert len(runtime.lrs) == 4
    assert (
        runtime.loss_term_sets
        == [
            (
                "energy",
                "forces",
                "virial",
                "non_conservative_forces",
                "non_conservative_stress",
            )
        ]
        * 4
    )
    assert runtime.raw_validations == 4
    assert runtime.ema_validations == 4
    assert runtime.frozen_checks == 6  # four epochs plus two endpoints
    assert runtime.endpoint_paths == [
        tmp_path / "upet_fge_n20_cpu" / "training" / "members" / "member_001.pt",
        tmp_path / "upet_fge_n20_cpu" / "training" / "members" / "member_002.pt",
    ]
    assert manifest_path == tmp_path / "upet_fge_n20_cpu" / "training" / "manifest.json"
    assert not (tmp_path / "upet_fge_n20_cpu" / "_work").exists()
    manifest = json.loads(manifest_path.read_text())
    resolved = config.sanitized()
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    assert "runtime_resume_identity" not in manifest
    assert set(manifest) == {
        "schema_version",
        "project_name",
        "config_resolved",
        "config_identity",
        "checkpoint_identity",
        "data_identities",
        "model_contract",
        "frozen_fingerprint_identity",
        "dependency_snapshot",
        "scientific_flags",
        "training_code_identity",
        "artifact_writer_code_identity",
        "validator_code_identity",
        "member_count",
        "members",
    }
    assert manifest["config_resolved"] == resolved
    assert manifest["config_identity"] == {
        "sha256": hashlib.sha256(canonical).hexdigest()
    }
    assert manifest["model_contract"] == {
        "readout_tensor_count": 12,
        "readout_parameter_count": 13_338,
    }
    assert manifest["scientific_flags"] == vars(config.scientific.training)


def test_training_sets_the_cycle_lr_before_every_optimizer_update(
    tmp_path: Path,
) -> None:
    """Applying LR only per epoch would silently alter the legacy trajectory."""

    runtime = _TorchRuntime(batches=8)
    config = _config(tmp_path)

    train_fge(config, runtime=runtime)

    cycle_steps = 2 * (8 // 4)
    expected = [
        asymmetric_triangular_lr(
            step,
            cycle_steps,
            config.fge.lr_min,
            config.fge.lr_max,
            config.fge.rise_fraction,
        )
        for _cycle in range(2)
        for step in range(cycle_steps)
    ]
    assert runtime.lrs == expected


def test_resume_rejects_any_identity_mismatch_before_training(tmp_path: Path) -> None:
    """Resuming against a different config/code/base/data identity is forbidden."""

    runtime = _TorchRuntime(batches=4)
    config = _config(tmp_path, resume=True)
    work = tmp_path / "upet_fge_n20_cpu" / "_work"
    work.mkdir(parents=True)
    torch.save(
        {
            "resume_identity": {
                "config": "0" * 64,
                "code": "d" * 64,
                "base": SHA_BASE,
                "data": SHA_TRAIN,
            }
        },
        work / "native_resume.pt",
    )

    with pytest.raises(HardFailure, match="resume identity"):
        train_fge(config, runtime=runtime)

    assert runtime.lrs == []
    assert (work / "native_resume.pt").is_file()


def test_failed_training_retains_identity_matched_resume_state(tmp_path: Path) -> None:
    """Only successful training publication may consume its own resume state."""
    runtime = _TorchRuntime(batches=4)
    runtime.reload_and_smoke = lambda _: False  # type: ignore[method-assign]

    with pytest.raises(HardFailure, match="reload smoke"):
        train_fge(_config(tmp_path), runtime=runtime)

    assert (tmp_path / "upet_fge_n20_cpu" / "_work" / "native_resume.pt").is_file()


def test_training_constructs_the_native_pet_runtime_when_not_injected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The public train path must bind the real PET runtime by default."""

    runtime = _TorchRuntime(batches=4)
    seen: dict[str, FGEConfig] = {}

    def build(config: FGEConfig) -> _TorchRuntime:
        seen["config"] = config
        return runtime

    monkeypatch.setattr(PETTrainingRuntime, "from_config", staticmethod(build))

    train_fge(_config(tmp_path))

    assert seen["config"].project.name == "upet_fge_n20_cpu"
    assert len(runtime.lrs) == 4


def test_pet_runtime_marks_unavailable_code_identities() -> None:
    """A runtime without a git revision must use the builder's exact fallback."""

    runtime = object.__new__(PETTrainingRuntime)

    metadata = runtime.manifest_metadata()

    assert set(metadata["dependency_snapshot"]) == {"torch", "metatrain"}
    assert metadata["training_code_identity"] == {"status": "unavailable"}
    assert metadata["artifact_writer_code_identity"] == {"status": "unavailable"}
    assert metadata["validator_code_identity"] == {"status": "unavailable"}


_REAL_CHECKPOINT = Path("/home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt")
_REAL_N20 = Path("/home/bywang/code/UQ/upet_new/data/dataset/matpes_n20.extxyz")


@pytest.mark.skipif(
    not (_REAL_CHECKPOINT.is_file() and _REAL_N20.is_file()),
    reason="requires the remote UPET n20 PET fixture",
)
def test_native_pet_runtime_runs_one_n20_batch_validation_and_member_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real PET API compatibility: batch, validation, and weights-only reload."""

    for name, value in {
        "UPET_FGE_BASE_CHECKPOINT": _REAL_CHECKPOINT,
        "UPET_FGE_TRAIN_DATA": _REAL_N20,
        "UPET_FGE_VAL_DATA": _REAL_N20,
        "UPET_FGE_TEST_DATA": _REAL_N20,
        "UPET_FGE_OUTPUT_ROOT": tmp_path,
    }.items():
        monkeypatch.setenv(name, str(value))
    config = load_config(
        Path(__file__).parents[1] / "configs" / "upet_fge_n20_cpu.yaml"
    )
    runtime = PETTrainingRuntime.from_config(config)
    result = runtime.train_batch(next(iter(runtime.train_loader)), lr=1e-8)
    result.loss.backward()
    assert tuple(result.loss_terms) == (
        "energy",
        "forces",
        "virial",
        "non_conservative_forces",
        "non_conservative_stress",
    )
    assert runtime.validate(use_ema=False)["loss_total"] >= 0.0
    member_path = tmp_path / "member_001.pt"
    torch.save(pack_member(runtime.model, 1, 1, 1, SHA_BASE), member_path)
    assert runtime.reload_and_smoke(member_path)


def test_success_cleanup_rejects_symlinked_work_ancestor(tmp_path: Path) -> None:
    """Successful cleanup must never unlink a matching file outside the result root."""
    from Uncertainty_Quantification.FGE.fge.artifacts import ExperimentLayout
    from Uncertainty_Quantification.FGE.fge.training import (
        _consume_resume_after_success,
    )

    root = tmp_path / "result"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    resume = outside / "native_resume.pt"
    runtime = _TorchRuntime(batches=4)
    torch.save({"resume_identity": runtime.resume_identity()}, resume)
    try:
        (root / "_work").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlink creation is unavailable")

    with pytest.raises(HardFailure, match="symlink"):
        _consume_resume_after_success(ExperimentLayout(root), runtime)

    assert resume.is_file()


def test_success_cleanup_ignores_empty_directory_removal_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A benign rmdir race cannot reverse an already published training success."""
    from Uncertainty_Quantification.FGE.fge.artifacts import ExperimentLayout
    from Uncertainty_Quantification.FGE.fge.training import (
        _consume_resume_after_success,
    )

    root = tmp_path / "result"
    work = root / "_work"
    work.mkdir(parents=True)
    resume = work / "native_resume.pt"
    runtime = _TorchRuntime(batches=4)
    torch.save({"resume_identity": runtime.resume_identity()}, resume)

    def raced_rmdir(_: Path) -> None:
        raise OSError("directory changed concurrently")

    monkeypatch.setattr(Path, "rmdir", raced_rmdir)

    _consume_resume_after_success(ExperimentLayout(root), runtime)

    assert not resume.exists()
