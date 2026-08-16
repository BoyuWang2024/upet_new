from __future__ import annotations

import json
from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge.inference_evaluation import (
    _domain_metrics,
    evaluate_inference_dataset,
    evaluate_prediction_chunk,
)
from Uncertainty_Quantification.FGE.fge.inference_only import (
    predict_inference_dataset,
)
from Uncertainty_Quantification.FGE.fge.uncertainty import (
    tensor_to_voigt_symmetric,
)
from Uncertainty_Quantification.FGE.tests.test_inference_only import (
    LiteralChunkRuntime,
    _config,
)


def _literal_prediction(
    *, stress_reference: torch.Tensor | None = None
) -> dict[str, object]:
    return {
        "energy_prediction": torch.tensor(
            [[2.0, 8.0], [4.0, 12.0]], dtype=torch.float32
        ),
        "forces_prediction": torch.tensor(
            [
                [[1.0, 2.0, 3.0], [2.0, 4.0, 6.0]],
                [[3.0, 4.0, 5.0], [4.0, 6.0, 8.0]],
            ],
            dtype=torch.float32,
        ),
        "stress_prediction": torch.tensor(
            [
                [
                    [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                    torch.eye(3).tolist(),
                ],
                [
                    [[3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
                    (torch.eye(3) * 3.0).tolist(),
                ],
            ],
            dtype=torch.float32,
        ),
        "energy_reference": torch.tensor([3.0, 8.0], dtype=torch.float32),
        "forces_reference": torch.zeros((2, 3), dtype=torch.float32),
        "stress_reference": stress_reference,
        "n_atoms": torch.tensor([2, 4], dtype=torch.int64),
    }


def test_stress_is_symmetrized_in_fixed_voigt_order() -> None:
    stress = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])

    assert torch.equal(
        tensor_to_voigt_symmetric(stress),
        torch.tensor([[1.0, 5.0, 9.0, 7.0, 5.0, 3.0]]),
    )


def test_energy_uses_population_std_after_per_atom_conversion() -> None:
    result = evaluate_prediction_chunk(_literal_prediction())
    expected = torch.std(
        torch.tensor([[1.0, 2.0], [2.0, 3.0]]),
        dim=0,
        unbiased=False,
    )

    assert torch.equal(result["energy_per_atom_std"], expected)
    assert result["force_component_std"].shape == (6,)
    assert result["stress_component_std"].shape == (12,)


def test_mad_omits_stress_residual_but_keeps_unsupervised_std() -> None:
    result = evaluate_prediction_chunk(_literal_prediction(stress_reference=None))

    assert "stress_component_std" in result
    assert "stress_component_absolute_residual" not in result


def test_zero_uncertainty_records_an_empty_ratio_summary() -> None:
    metrics = _domain_metrics(torch.zeros(3), torch.ones(3))

    assert metrics["residual_to_uncertainty"] is None


def test_dataset_evaluation_publishes_chunk_uq_and_global_metrics(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    predict_inference_dataset(config, runtime=LiteralChunkRuntime())

    manifest_path = evaluate_inference_dataset(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = json.loads((config.output.root / "metrics.json").read_text("utf-8"))

    assert manifest["formula_version"] == "legacy_upet_fge_v1"
    assert manifest["chunk_count"] == 2
    assert set(metrics["domains"]) == {"energy", "force", "stress"}
    assert metrics["domains"]["energy"]["count_total"] == 3
    assert metrics["domains"]["force"]["count_total"] == 12
    assert metrics["domains"]["stress"]["count_total"] == 18
    assert (config.output.root / "run_manifest.json").is_file()
