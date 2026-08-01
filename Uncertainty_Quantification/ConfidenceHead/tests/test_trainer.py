from __future__ import annotations

import copy
import random
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from Uncertainty_Quantification.ConfidenceHead.confidence_head.trainer import (
    CHECKPOINT_SCHEMA_VERSION,
    ControlUpdate,
    EarlyStoppingState,
    LossAccumulator,
    TrainingIdentity,
    advance_validation_epoch,
    build_plateau_scheduler,
    capture_training_snapshot,
    commit_epoch_checkpoints,
    restore_training_snapshot,
)


def _identity(**overrides: str) -> TrainingIdentity:
    values = {
        "config_id": "config-123",
        "cache_id": "cache-123",
        "binning_id": "binning-123",
        "model_loss_id": "model-loss-123",
        "force_target_mode": "atom_mean",
        "force_error_definition": "abs_cartesian_component_mean_v1",
    }
    values.update(overrides)
    return TrainingIdentity(**values)


def _model_optimizer_scheduler() -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
]:
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = build_plateau_scheduler(
        optimizer,
        factor=0.5,
        patience=5,
        threshold=1e-4,
        threshold_mode="abs",
        cooldown=0,
        min_lr=1e-6,
    )
    return model, optimizer, scheduler


def _take_optimizer_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    optimizer.zero_grad()
    loss = model(torch.ones(2, 2)).square().mean()
    loss.backward()
    optimizer.step()


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray)
        assert np.array_equal(left, right)
    else:
        assert left == right


def test_public_state_dataclasses_have_the_exact_control_fields() -> None:
    assert [field.name for field in fields(EarlyStoppingState)] == [
        "ema",
        "best",
        "best_epoch",
        "bad_epochs",
        "stopped",
        "stop_reason",
    ]
    assert [field.name for field in fields(ControlUpdate)] == [
        "state",
        "improved",
        "should_save_best",
        "should_stop",
    ]
    assert [field.name for field in fields(TrainingIdentity)] == [
        "config_id",
        "cache_id",
        "binning_id",
        "model_loss_id",
        "force_target_mode",
        "force_error_definition",
    ]


def test_loss_accumulator_uses_branch_sample_sums_and_counts() -> None:
    accumulator = LossAccumulator()
    accumulator.update(
        force_loss_sum=6.0,
        force_count=2,
        energy_loss_sum=8.0,
        energy_count=1,
    )
    accumulator.update(
        force_loss_sum=3.0,
        force_count=1,
        energy_loss_sum=4.0,
        energy_count=3,
    )

    force, energy, total = accumulator.result(
        force_coefficient=1.0,
        energy_coefficient=1.5,
    )

    assert force == pytest.approx(3.0)
    assert energy == pytest.approx(3.0)
    assert total == pytest.approx(7.5)


@pytest.mark.parametrize(
    ("force_count", "energy_count", "message"),
    [(0, 1, "force"), (1, 0, "energy")],
)
def test_loss_accumulator_rejects_an_empty_branch(
    force_count: int,
    energy_count: int,
    message: str,
) -> None:
    accumulator = LossAccumulator(
        force_sum=1.0,
        force_count=force_count,
        energy_sum=1.0,
        energy_count=energy_count,
    )

    with pytest.raises(ValueError, match=message):
        accumulator.result(1.0, 1.5)


def test_first_validation_epoch_initializes_ema_from_raw_total_loss() -> None:
    _, optimizer, scheduler = _model_optimizer_scheduler()

    update = advance_validation_epoch(
        state=EarlyStoppingState(),
        raw_total_loss=2.5,
        epoch=0,
        scheduler=scheduler,
        beta=0.95,
        min_delta=1e-4,
        patience=15,
        min_epochs=3,
    )

    assert update.state.ema == pytest.approx(2.5)
    assert update.state.best == pytest.approx(2.5)
    assert update.state.best_epoch == 0
    assert update.state.bad_epochs == 0
    assert update.improved is True
    assert update.should_save_best is True
    assert update.should_stop is False
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_each_complete_validation_epoch_applies_one_ema_update() -> None:
    _, _, scheduler = _model_optimizer_scheduler()
    calls: list[float] = []
    original_step = scheduler.step

    def record_step(metric: float) -> None:
        calls.append(float(metric))
        original_step(metric)

    scheduler.step = record_step  # type: ignore[method-assign]
    state = EarlyStoppingState(ema=2.0, best=1.0, best_epoch=0)

    update = advance_validation_epoch(
        state=state,
        raw_total_loss=4.0,
        epoch=1,
        scheduler=scheduler,
        beta=0.95,
        min_delta=1e-4,
        patience=15,
        min_epochs=3,
    )

    expected = 0.95 * 2.0 + 0.05 * 4.0
    assert update.state.ema == pytest.approx(expected)
    assert calls == pytest.approx([expected])


@pytest.mark.parametrize(
    ("raw", "improved"),
    [(0.9998, True), (0.9999, False), (0.99991, False)],
)
def test_improvement_is_strictly_below_best_minus_min_delta(
    raw: float,
    improved: bool,
) -> None:
    _, _, scheduler = _model_optimizer_scheduler()

    update = advance_validation_epoch(
        state=EarlyStoppingState(ema=1.0, best=1.0, best_epoch=0),
        raw_total_loss=raw,
        epoch=1,
        scheduler=scheduler,
        beta=0.0,
        min_delta=1e-4,
        patience=15,
        min_epochs=3,
    )

    assert update.improved is improved
    assert update.should_save_best is improved


def test_early_stopping_occurs_on_the_fifteenth_consecutive_bad_epoch() -> None:
    _, _, scheduler = _model_optimizer_scheduler()
    state = EarlyStoppingState(ema=1.0, best=1.0, best_epoch=0)

    for epoch in range(1, 15):
        update = advance_validation_epoch(
            state=state,
            raw_total_loss=1.0,
            epoch=epoch,
            scheduler=scheduler,
            beta=0.95,
            min_delta=1e-4,
            patience=15,
            min_epochs=3,
        )
        state = update.state
        assert update.should_stop is False

    update = advance_validation_epoch(
        state=state,
        raw_total_loss=1.0,
        epoch=15,
        scheduler=scheduler,
        beta=0.95,
        min_delta=1e-4,
        patience=15,
        min_epochs=3,
    )

    assert update.state.bad_epochs == 15
    assert update.state.stopped is True
    assert update.state.stop_reason == "early_stopping"
    assert update.should_stop is True


def test_min_epochs_prevents_an_early_stop() -> None:
    _, _, scheduler = _model_optimizer_scheduler()

    update = advance_validation_epoch(
        state=EarlyStoppingState(
            ema=1.0,
            best=1.0,
            best_epoch=0,
            bad_epochs=14,
        ),
        raw_total_loss=1.0,
        epoch=1,
        scheduler=scheduler,
        beta=0.95,
        min_delta=1e-4,
        patience=15,
        min_epochs=3,
    )

    assert update.state.bad_epochs == 15
    assert update.state.stopped is False
    assert update.state.stop_reason is None
    assert update.should_stop is False


def test_plateau_scheduler_uses_validated_configuration() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    scheduler = build_plateau_scheduler(
        optimizer,
        factor=0.25,
        patience=3,
        threshold=1e-5,
        threshold_mode="abs",
        cooldown=2,
        min_lr=1e-7,
    )

    assert scheduler.mode == "min"
    assert scheduler.factor == pytest.approx(0.25)
    assert scheduler.patience == 3
    assert scheduler.threshold == pytest.approx(1e-5)
    assert scheduler.threshold_mode == "abs"
    assert scheduler.cooldown == 2
    assert scheduler.min_lrs == pytest.approx([1e-7])
    assert scheduler.optimizer is optimizer


def test_snapshot_contains_complete_training_and_reproducibility_state() -> None:
    model, optimizer, scheduler = _model_optimizer_scheduler()
    _take_optimizer_step(model, optimizer)
    scheduler.step(0.9)
    sampler = torch.Generator().manual_seed(71)
    control = EarlyStoppingState(
        ema=0.9,
        best=0.8,
        best_epoch=2,
        bad_epochs=3,
        stopped=True,
        stop_reason="early_stopping",
    )

    snapshot = capture_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=5,
        global_step=19,
        control_state=control,
        best_step=11,
        identity=_identity(),
        sampler_generator=sampler,
    )

    assert set(snapshot) == {
        "schema_version",
        "model",
        "optimizer",
        "scheduler",
        "epoch",
        "global_step",
        "learning_rate",
        "ema",
        "best",
        "best_epoch",
        "best_step",
        "bad_epochs",
        "stopped",
        "stop_reason",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_state",
        "sampler_rng_state",
        "config_id",
        "cache_id",
        "binning_id",
        "model_loss_id",
        "force_target_mode",
        "force_error_definition",
    }
    assert snapshot["force_target_mode"] == "atom_mean"
    assert snapshot["force_error_definition"] == "abs_cartesian_component_mean_v1"
    assert snapshot["schema_version"] == CHECKPOINT_SCHEMA_VERSION
    assert snapshot["epoch"] == 5
    assert snapshot["global_step"] == 19
    assert snapshot["learning_rate"] == pytest.approx(1e-3)
    assert snapshot["best_step"] == 11
    assert snapshot["stop_reason"] == "early_stopping"
    assert isinstance(snapshot["python_rng_state"], tuple)
    assert isinstance(snapshot["numpy_rng_state"], tuple)
    assert isinstance(snapshot["torch_cpu_rng_state"], torch.Tensor)
    assert isinstance(snapshot["torch_cuda_rng_state"], list)
    assert torch.equal(snapshot["sampler_rng_state"], sampler.get_state())


def test_restore_is_exact_and_resumes_at_the_next_epoch() -> None:
    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    model, optimizer, scheduler = _model_optimizer_scheduler()
    _take_optimizer_step(model, optimizer)
    scheduler.step(0.8)
    sampler = torch.Generator().manual_seed(10)
    snapshot = capture_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=4,
        global_step=23,
        control_state=EarlyStoppingState(
            ema=0.8,
            best=0.7,
            best_epoch=2,
            bad_epochs=2,
        ),
        best_step=13,
        identity=_identity(),
        sampler_generator=sampler,
    )
    expected_model = copy.deepcopy(model.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    expected_scheduler = copy.deepcopy(scheduler.state_dict())

    random.random()
    np.random.random()
    torch.rand(3)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    optimizer.param_groups[0]["lr"] = 0.25
    scheduler.step(5.0)
    sampler.manual_seed(999)

    restored = restore_training_snapshot(
        snapshot=snapshot,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_identity=_identity(),
        sampler_generator=sampler,
    )

    _assert_nested_equal(model.state_dict(), expected_model)
    _assert_nested_equal(optimizer.state_dict(), expected_optimizer)
    _assert_nested_equal(scheduler.state_dict(), expected_scheduler)
    assert restored.epoch == 4
    assert restored.next_epoch == 5
    assert restored.global_step == 23
    assert restored.learning_rate == pytest.approx(1e-3)
    assert restored.control_state == EarlyStoppingState(
        ema=0.8,
        best=0.7,
        best_epoch=2,
        bad_epochs=2,
    )
    assert restored.best_step == 13
    assert torch.equal(sampler.get_state(), snapshot["sampler_rng_state"])


@pytest.mark.parametrize(
    ("field", "wrong"),
    [
        ("force_target_mode", "component"),
        ("force_error_definition", "abs_cartesian_component_v1"),
        ("config_id", "config-wrong"),
        ("cache_id", "cache-wrong"),
        ("binning_id", "binning-wrong"),
        ("model_loss_id", "model-loss-wrong"),
    ],
)
def test_resume_rejects_each_upstream_identity_mismatch(
    field: str,
    wrong: str,
) -> None:
    model, optimizer, scheduler = _model_optimizer_scheduler()
    sampler = torch.Generator().manual_seed(1)
    snapshot = capture_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
        control_state=EarlyStoppingState(),
        best_step=None,
        identity=_identity(),
        sampler_generator=sampler,
    )

    with pytest.raises(ValueError, match=field):
        restore_training_snapshot(
            snapshot=snapshot,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=_identity(**{field: wrong}),
            sampler_generator=sampler,
        )


def test_legacy_checkpoint_without_force_semantics_is_component_only() -> None:
    model, optimizer, scheduler = _model_optimizer_scheduler()
    sampler = torch.Generator().manual_seed(1)
    component_identity = _identity(
        force_target_mode="component",
        force_error_definition="abs_cartesian_component_v1",
    )
    snapshot = capture_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=0,
        control_state=EarlyStoppingState(),
        best_step=None,
        identity=component_identity,
        sampler_generator=sampler,
    )
    snapshot.pop("force_target_mode")
    snapshot.pop("force_error_definition")

    restored = restore_training_snapshot(
        snapshot=snapshot,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_identity=component_identity,
        sampler_generator=sampler,
    )
    assert restored.next_epoch == 1

    atom_mean_identity = _identity()
    with pytest.raises(ValueError, match="force target semantics"):
        restore_training_snapshot(
            snapshot=snapshot,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=atom_mean_identity,
            sampler_generator=sampler,
        )


def test_best_and_last_are_independent_atomic_checkpoints(tmp_path: Path) -> None:
    model, optimizer, scheduler = _model_optimizer_scheduler()
    sampler = torch.Generator().manual_seed(2)
    best_snapshot = capture_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=4,
        control_state=EarlyStoppingState(ema=0.8, best=0.8, best_epoch=1),
        best_step=4,
        identity=_identity(),
        sampler_generator=sampler,
    )
    commit_epoch_checkpoints(
        checkpoint_dir=tmp_path,
        snapshot=best_snapshot,
        save_best=True,
        profile="production",
        stop_after_epoch=None,
        max_epochs=200,
    )
    best_bytes = (tmp_path / "best.pt").read_bytes()

    last_snapshot = dict(best_snapshot)
    last_snapshot.update(
        epoch=2,
        global_step=8,
        ema=0.9,
        bad_epochs=1,
        stop_reason="max_epochs",
    )
    commit_epoch_checkpoints(
        checkpoint_dir=tmp_path,
        snapshot=last_snapshot,
        save_best=False,
        profile="production",
        stop_after_epoch=None,
        max_epochs=200,
    )

    assert (tmp_path / "best.pt").read_bytes() == best_bytes
    last = torch.load(tmp_path / "last.pt", map_location="cpu", weights_only=False)
    best = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert last["epoch"] == 2
    assert last["stop_reason"] == "max_epochs"
    assert best["epoch"] == 1
    assert best["stop_reason"] is None
    assert not list(tmp_path.glob("*.tmp"))


def test_smoke_stop_happens_only_after_last_is_committed(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _model_optimizer_scheduler()
    sampler = torch.Generator().manual_seed(3)
    snapshot = capture_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=5,
        control_state=EarlyStoppingState(),
        best_step=None,
        identity=_identity(),
        sampler_generator=sampler,
    )

    should_stop = commit_epoch_checkpoints(
        checkpoint_dir=tmp_path,
        snapshot=snapshot,
        save_best=False,
        profile="smoke",
        stop_after_epoch=1,
        max_epochs=3,
    )

    assert should_stop is True
    assert (tmp_path / "last.pt").is_file()
    persisted = torch.load(
        tmp_path / "last.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert persisted["epoch"] == 1
    assert persisted["stop_reason"] == "external_stop_after_epoch"
    assert persisted["config_id"] == "config-123"
    assert "stop_after_epoch" not in persisted
    assert "max_epochs" not in persisted


def test_smoke_stop_is_transient_and_does_not_change_max_epochs(
    tmp_path: Path,
) -> None:
    model, optimizer, scheduler = _model_optimizer_scheduler()
    sampler = torch.Generator().manual_seed(4)
    snapshot = capture_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=1,
        global_step=5,
        control_state=EarlyStoppingState(),
        best_step=None,
        identity=_identity(),
        sampler_generator=sampler,
    )
    max_epochs = 3
    commit_epoch_checkpoints(
        checkpoint_dir=tmp_path,
        snapshot=snapshot,
        save_best=False,
        profile="smoke",
        stop_after_epoch=1,
        max_epochs=max_epochs,
    )
    persisted = torch.load(
        tmp_path / "last.pt",
        map_location="cpu",
        weights_only=False,
    )

    restored = restore_training_snapshot(
        snapshot=persisted,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        expected_identity=_identity(),
        sampler_generator=sampler,
    )

    assert restored.next_epoch == 2
    assert restored.control_state.stop_reason is None
    assert restored.control_state.stopped is False
    assert max_epochs == 3


def test_stop_after_epoch_is_smoke_only(tmp_path: Path) -> None:
    model, optimizer, scheduler = _model_optimizer_scheduler()
    snapshot = capture_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=0,
        global_step=1,
        control_state=EarlyStoppingState(),
        best_step=None,
        identity=_identity(),
        sampler_generator=torch.Generator(),
    )

    with pytest.raises(ValueError, match="smoke"):
        commit_epoch_checkpoints(
            checkpoint_dir=tmp_path,
            snapshot=snapshot,
            save_best=False,
            profile="production",
            stop_after_epoch=0,
            max_epochs=200,
        )


def _runtime_state(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    sampler: torch.Generator,
) -> dict[str, Any]:
    return {
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
        "scheduler": copy.deepcopy(scheduler.state_dict()),
        "python": copy.deepcopy(random.getstate()),
        "numpy": copy.deepcopy(np.random.get_state()),
        "torch": torch.get_rng_state().clone(),
        "sampler": sampler.get_state().clone(),
    }


def _assert_runtime_state_unchanged(
    before: dict[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau,
    sampler: torch.Generator,
) -> None:
    after = _runtime_state(model, optimizer, scheduler, sampler)
    _assert_nested_equal(before, after)


def _valid_snapshot_for_rejection() -> tuple[
    dict[str, Any],
    torch.nn.Module,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    torch.Generator,
]:
    model, optimizer, scheduler = _model_optimizer_scheduler()
    sampler = torch.Generator().manual_seed(17)
    snapshot = capture_training_snapshot(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=7,
        control_state=EarlyStoppingState(
            ema=1.0,
            best=0.9,
            best_epoch=1,
            bad_epochs=1,
        ),
        best_step=4,
        identity=_identity(),
        sampler_generator=sampler,
    )
    return snapshot, model, optimizer, scheduler, sampler


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ema", "bad", "ema"),
        ("best", float("nan"), "best"),
        ("best_epoch", -1, "best_epoch"),
        ("stopped", "false", "stopped"),
        ("stop_reason", 4, "stop_reason"),
        ("ema", None, "ema and best"),
        ("best", None, "ema and best"),
    ],
)
def test_restore_rejects_invalid_control_scalars_without_mutation(
    field: str,
    value: Any,
    message: str,
) -> None:
    snapshot, model, optimizer, scheduler, sampler = _valid_snapshot_for_rejection()
    snapshot[field] = value
    before = _runtime_state(model, optimizer, scheduler, sampler)

    with pytest.raises(ValueError, match=message):
        restore_training_snapshot(
            snapshot=snapshot,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=_identity(),
            sampler_generator=sampler,
        )

    _assert_runtime_state_unchanged(before, model, optimizer, scheduler, sampler)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("python_rng_state", ("bad",), "python_rng_state"),
        ("numpy_rng_state", ("bad",), "numpy_rng_state"),
        ("torch_cpu_rng_state", torch.tensor([1]), "torch_cpu_rng_state"),
        ("sampler_rng_state", torch.tensor([1]), "sampler_rng_state"),
    ],
)
def test_restore_rejects_invalid_rng_without_mutation(
    field: str,
    value: Any,
    message: str,
) -> None:
    snapshot, model, optimizer, scheduler, sampler = _valid_snapshot_for_rejection()
    snapshot[field] = value
    before = _runtime_state(model, optimizer, scheduler, sampler)

    with pytest.raises(ValueError, match=message):
        restore_training_snapshot(
            snapshot=snapshot,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=_identity(),
            sampler_generator=sampler,
        )

    _assert_runtime_state_unchanged(before, model, optimizer, scheduler, sampler)


def test_restore_rejects_learning_rate_conflict_without_mutation() -> None:
    snapshot, model, optimizer, scheduler, sampler = _valid_snapshot_for_rejection()
    snapshot["learning_rate"] = 0.25
    before = _runtime_state(model, optimizer, scheduler, sampler)

    with pytest.raises(ValueError, match="learning_rate"):
        restore_training_snapshot(
            snapshot=snapshot,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_identity=_identity(),
            sampler_generator=sampler,
        )

    _assert_runtime_state_unchanged(before, model, optimizer, scheduler, sampler)
