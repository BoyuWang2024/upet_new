from __future__ import annotations

from types import SimpleNamespace

import torch

from confidence_head.binning import fixed_linear_binning
from confidence_head.model import ConfidenceOutput
from confidence_head.single_target_evaluation import collect_single_target_predictions


class _EnergyModel:
    force_active = False
    energy_active = True

    def __call__(self, force_features, energy_features, atom_counts):
        assert force_features is None
        assert energy_features is ENERGY_FEATURES
        torch.testing.assert_close(atom_counts, torch.tensor([2, 1]))
        return ConfidenceOutput(
            force_logits=None,
            energy_logits=torch.tensor(
                [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]], dtype=torch.float32
            ),
        )


class _ForceModel:
    force_active = True
    energy_active = False

    def __call__(self, force_features, energy_features, atom_counts):
        assert force_features is FORCE_FEATURES
        assert energy_features is None
        assert atom_counts is None
        return ConfidenceOutput(
            force_logits=torch.tensor(
                [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]], dtype=torch.float32
            ),
            energy_logits=None,
        )


FORCE_FEATURES = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
ENERGY_FEATURES = torch.tensor([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])


def _batch() -> dict[str, torch.Tensor]:
    return {
        "force_features": FORCE_FEATURES,
        "energy_features": ENERGY_FEATURES,
        "force_prediction": torch.zeros((2, 3)),
        "force_reference": torch.tensor([[1.0, 2.0, 3.0], [3.0, 3.0, 3.0]]),
        "energy_prediction": torch.tensor([4.0, 6.0]),
        "energy_reference": torch.tensor([6.0, 7.0]),
        "structure_ids": torch.tensor([11, 12]),
        "num_atoms": torch.tensor([2, 1]),
    }


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(force=SimpleNamespace(target_mode="atom_mean"))
    )


def _to_device(batch, _device):
    return batch


def _offsets(counts):
    merged = torch.cat(counts)
    result = torch.zeros(len(merged) + 1, dtype=torch.int64)
    result[1:] = torch.cumsum(merged, dim=0)
    return result


def test_collect_energy_is_per_atom_and_can_include_raw_values() -> None:
    result = collect_single_target_predictions(
        model=_EnergyModel(),
        loader=[_batch()],
        device=torch.device("cpu"),
        config=_config(),
        force_spec=fixed_linear_binning(3, 3.0),
        energy_spec=fixed_linear_binning(3, 3.0),
        to_device=_to_device,
        offsets=_offsets,
        include_raw=True,
    )

    assert result.target == "energy"
    torch.testing.assert_close(
        result.predictions["energy_observed_errors"], torch.tensor([1.0, 1.0])
    )
    torch.testing.assert_close(
        result.predictions["energy_prediction"], torch.tensor([4.0, 6.0])
    )
    torch.testing.assert_close(
        result.predictions["energy_reference"], torch.tensor([6.0, 7.0])
    )
    assert "force_prediction" not in result.predictions


def test_collect_force_uses_atom_mean_and_only_force_features() -> None:
    result = collect_single_target_predictions(
        model=_ForceModel(),
        loader=[_batch()],
        device=torch.device("cpu"),
        config=_config(),
        force_spec=fixed_linear_binning(3, 3.0),
        energy_spec=fixed_linear_binning(3, 3.0),
        to_device=_to_device,
        offsets=_offsets,
        include_raw=True,
    )

    assert result.target == "force"
    torch.testing.assert_close(
        result.predictions["force_observed_errors"], torch.tensor([2.0, 3.0])
    )
    assert result.predictions["force_target_mode"] == "atom_mean"
    assert result.predictions["force_error_definition"] == (
        "abs_cartesian_component_mean_v1"
    )
    torch.testing.assert_close(
        result.predictions["force_prediction"], torch.zeros((2, 3))
    )
    assert "energy_prediction" not in result.predictions


def test_collect_without_raw_preserves_existing_evaluation_field_set() -> None:
    result = collect_single_target_predictions(
        model=_EnergyModel(),
        loader=[_batch()],
        device=torch.device("cpu"),
        config=_config(),
        force_spec=fixed_linear_binning(3, 3.0),
        energy_spec=fixed_linear_binning(3, 3.0),
        to_device=_to_device,
        offsets=_offsets,
        include_raw=False,
    )

    assert set(result.predictions) == {
        "energy_logits",
        "energy_labels",
        "energy_observed_errors",
        "energy_expected_errors",
        "structure_ids",
        "atom_offsets",
        "energy_representatives",
    }
