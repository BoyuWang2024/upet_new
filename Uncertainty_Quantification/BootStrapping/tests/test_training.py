from __future__ import annotations

import random

import numpy as np
import torch


class _FakeRuntime:
    def __init__(self, validation_losses: list[float] | None = None) -> None:
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.zero_()
        self._validation_losses = validation_losses

    def train_epoch(self, optimizer: torch.optim.Optimizer, epoch: int) -> float:
        optimizer.zero_grad(set_to_none=True)
        noise = torch.rand(()) + float(np.random.random()) + random.random()
        loss = (self.model.weight.sum() - noise - epoch).square()
        loss.backward()
        optimizer.step()
        return float(loss.detach())

    def validation_loss(self, epoch: int) -> float:
        if self._validation_losses is not None:
            return self._validation_losses[epoch - 1]
        return float(self.model.weight.detach().square().sum())


def _seed_all() -> None:
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)


def test_best_uses_raw_validation_and_pairs_same_epoch() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.training import fit_runtime

    _seed_all()
    runtime = _FakeRuntime([0.4, 0.2, 0.3])
    optimizer = torch.optim.Adam(runtime.model.parameters(), lr=0.03)
    result = fit_runtime(runtime, optimizer, max_epochs=3, ema_decay=0.9)

    assert result.best_epoch == 2
    assert result.history[1].raw_validation_loss == 0.2
    assert result.best_raw["epoch"].item() == 2
    assert result.best_ema["epoch"].item() == 2


def test_resume_restores_optimizer_epoch_ema_and_rng() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.training import fit_runtime

    _seed_all()
    continuous_runtime = _FakeRuntime()
    continuous_optimizer = torch.optim.Adam(
        continuous_runtime.model.parameters(), lr=0.03
    )
    continuous = fit_runtime(
        continuous_runtime,
        continuous_optimizer,
        max_epochs=3,
        ema_decay=0.9,
    )

    _seed_all()
    first_runtime = _FakeRuntime()
    first_optimizer = torch.optim.Adam(first_runtime.model.parameters(), lr=0.03)
    interrupted = fit_runtime(
        first_runtime,
        first_optimizer,
        max_epochs=1,
        ema_decay=0.9,
    )
    random.random()
    np.random.random()
    torch.rand(())
    resumed_runtime = _FakeRuntime()
    resumed_optimizer = torch.optim.Adam(resumed_runtime.model.parameters(), lr=0.03)
    resumed = fit_runtime(
        resumed_runtime,
        resumed_optimizer,
        max_epochs=3,
        ema_decay=0.9,
        resume=interrupted.resume,
    )

    assert resumed.history == continuous.history
    assert torch.equal(resumed.final_raw["weight"], continuous.final_raw["weight"])
    assert torch.equal(resumed.final_ema["weight"], continuous.final_ema["weight"])
