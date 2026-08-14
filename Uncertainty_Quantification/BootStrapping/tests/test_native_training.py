from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch


class _FakeRuntime:
    def __init__(self) -> None:
        self.model = torch.nn.Module()
        self.model.register_parameter(
            "node_last_layers", torch.nn.Parameter(torch.zeros(13_338))
        )

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        optimizer.zero_grad(set_to_none=True)
        loss = (self.model.node_last_layers.mean() - epoch).square()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def validation_loss(self, epoch: int) -> float:
        return float(self.model.node_last_layers.square().mean().detach())


def _config(tmp_path: Path):
    return SimpleNamespace(
        data=SimpleNamespace(train=tmp_path / "train.extxyz"),
        bootstrap=SimpleNamespace(
            ensemble_size=2,
            base_seed=17,
            sample_size=None,
            replacement=True,
            save_indices=True,
            save_oob=True,
        ),
        prediction=SimpleNamespace(parameter_modes=("raw",)),
        training=SimpleNamespace(
            max_epochs=1,
            ema_decay=0.9,
            optimizer=SimpleNamespace(
                name="Adam", learning_rate=0.01, weight_decay=0.0
            ),
        ),
    )


def test_train_run_publishes_audited_best_final_latest(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.checkpoint import (
        audit_checkpoint,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.native_training import (
        train_run,
    )

    config = _config(tmp_path)
    root = tmp_path / "run"

    result = train_run(
        config,
        root,
        training_size=4,
        runtime_factory=lambda config, indices, seeds: _FakeRuntime(),
    )

    assert result == root.resolve()
    for index in range(2):
        checkpoint_root = root / f"members/member_{index:03d}/checkpoints"
        for kind in ("best", "final", "latest"):
            audit = audit_checkpoint(
                checkpoint_root / f"{kind}.pt", expected_parameter_count=13_338
            )
            assert audit.inference_ready is True
        assert audit_checkpoint(checkpoint_root / "latest.pt").resume_ready is True
        member_root = root / f"members/member_{index:03d}"
        assert (member_root / "bootstrap_indices.npz").is_file()
        assert (member_root / "manifest.json").is_file()
    assert (root / "run_manifest.json").is_file()


def test_train_run_resumes_latest_and_matches_continuous(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.checkpoint import (
        load_checkpoint_branch,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.native_training import (
        train_run,
    )

    config = _config(tmp_path)
    interrupted = tmp_path / "interrupted"
    train_run(
        config,
        interrupted,
        training_size=4,
        runtime_factory=lambda config, indices, seeds: _FakeRuntime(),
    )
    for index in range(2):
        member = interrupted / f"members/member_{index:03d}"
        (member / "checkpoints/final.pt").unlink()
        (member / "checkpoints/best.pt").unlink()
        (member / "manifest.json").unlink()
    config.training.max_epochs = 2
    train_run(
        config,
        interrupted,
        training_size=4,
        runtime_factory=lambda config, indices, seeds: _FakeRuntime(),
    )

    continuous = tmp_path / "continuous"
    train_run(
        config,
        continuous,
        training_size=4,
        runtime_factory=lambda config, indices, seeds: _FakeRuntime(),
    )
    for index in range(2):
        resumed = load_checkpoint_branch(
            interrupted / f"members/member_{index:03d}/checkpoints/final.pt", "raw"
        )
        expected = load_checkpoint_branch(
            continuous / f"members/member_{index:03d}/checkpoints/final.pt", "raw"
        )
        assert resumed.keys() == expected.keys()
        assert all(torch.equal(resumed[name], expected[name]) for name in resumed)
