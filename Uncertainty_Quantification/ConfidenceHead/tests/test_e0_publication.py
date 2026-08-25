from __future__ import annotations

from pathlib import Path

import pytest
import torch
from confidence_head.binning import fixed_linear_binning, labels_from_thresholds
from confidence_head.e0_calibration import E0Dataset, energy_observed_errors
from confidence_head.e0_publication import (
    E0_VARIANTS,
    E0CampaignInputs,
    publish_e0_campaign,
    verify_e0_campaign,
)
from confidence_head.external_prediction import _publish_external_result
from confidence_head.single_target_prediction import SingleTargetResult


def _dataset() -> tuple[E0Dataset, torch.Tensor, torch.Tensor, torch.Tensor]:
    composition = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        dtype=torch.float64,
    )
    model_e0 = torch.tensor([-1.0, -3.0], dtype=torch.float64)
    mad_e0 = torch.tensor([-2.0, -5.0], dtype=torch.float64)
    atomization = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    raw = atomization + composition @ model_e0
    target = atomization + composition @ mad_e0
    dataset = E0Dataset(
        structure_ids=torch.tensor([0, 1, 2], dtype=torch.int64),
        atomic_numbers=torch.tensor([1, 6, 1, 6], dtype=torch.int64),
        atom_offsets=torch.tensor([0, 1, 2, 4], dtype=torch.int64),
        composition=composition,
        target_energy_r2scan=target,
        atomization_energy=atomization,
    )
    return dataset, raw, model_e0, mad_e0


def _source(
    runs_root: Path,
    *,
    dataset_name: str,
    order: int,
    dataset: E0Dataset,
    raw: torch.Tensor,
) -> Path:
    run = runs_root / f"energy-{order}"
    run.mkdir(parents=True, exist_ok=True)
    (run / "manifest.json").write_text(
        f'{{"run_id":"energy-{order}","status":"complete"}}\n',
        encoding="utf-8",
    )
    spec = fixed_linear_binning(3, 3.0)
    observed = energy_observed_errors(
        raw,
        dataset.target_energy_r2scan,
        dataset.atom_counts,
    ).to(torch.float32)
    logits = torch.tensor(
        [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        dtype=torch.float32,
    )
    result = SingleTargetResult(
        target="energy",
        predictions={
            "structure_ids": dataset.structure_ids,
            "atom_offsets": dataset.atom_offsets,
            "energy_prediction": raw.to(torch.float32),
            "energy_reference": dataset.target_energy_r2scan.to(torch.float32),
            "energy_logits": logits + order / 100.0,
            "energy_labels": labels_from_thresholds(observed, spec.thresholds),
            "energy_observed_errors": observed,
            "energy_expected_errors": torch.tensor(
                [0.2, 1.0, 2.2],
                dtype=torch.float32,
            )
            + order / 100.0,
            "energy_representatives": spec.representatives,
        },
        metrics={"energy": {"sample_count": 3}},
    )
    return _publish_external_result(
        run_dir=run,
        dataset_name=dataset_name,
        result=result,
        spec=spec,
        identity_payload={
            "dataset": {"name": dataset_name, "sha256": "a" * 64},
            "source_run_id": f"energy-{order}",
            "head_checkpoint_sha256": "b" * 64,
            "input_cache_id": f"cache-{dataset_name}",
            "training_cache_id": "cache-training",
            "binning_id": "binning",
            "target": "energy",
            "order": order,
        },
    )


def _inputs(
    tmp_path: Path,
    *,
    mismatched_order: int | None = None,
    checkpoint_sha256: str = "c" * 64,
) -> E0CampaignInputs:
    dataset, raw, model_e0, _ = _dataset()
    validation_sources = {}
    test_sources = {}
    for order in range(1, 9):
        validation_sources[order] = _source(
            tmp_path / "validation" / "runs",
            dataset_name="mad_r2scan_val",
            order=order,
            dataset=dataset,
            raw=raw,
        )
        test_raw = raw.clone()
        if order == mismatched_order:
            test_raw[0] += 0.25
        test_sources[order] = _source(
            tmp_path / "test" / "runs",
            dataset_name="mad_r2scan_test",
            order=order,
            dataset=dataset,
            raw=test_raw,
        )
    return E0CampaignInputs(
        checkpoint_sha256=checkpoint_sha256,
        validation_sha256="d" * 64,
        test_sha256="e" * 64,
        atomic_types=(1, 6),
        model_e0=model_e0,
        validation=dataset,
        test=dataset,
        validation_sources=validation_sources,
        test_sources=test_sources,
    )


def test_campaign_references_raw_uq_without_copying_it(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    source_predictions = inputs.test_sources[1] / "predictions.pt"
    source_before = source_predictions.read_bytes()

    root = publish_e0_campaign(inputs, tmp_path / "campaign")
    manifest = verify_e0_campaign(root / "manifest.json", full=True)

    assert set(manifest["variants"]) == set(E0_VARIANTS)
    assert source_predictions.read_bytes() == source_before
    assert not list(root.rglob("*logits*"))
    assert not list(root.rglob("*expected_errors*"))
    assert not list(root.rglob("*force*"))
    assert not list(root.parent.glob(".staging-*"))

    direct = torch.load(
        root / "variants/direct_mad_e0_test_informed/energy_data.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert torch.allclose(
        direct["model_energy_corrected"],
        inputs.test.target_energy_r2scan,
        atol=1e-6,
        rtol=0,
    )
    assert torch.allclose(
        direct["energy_observed_errors"],
        torch.zeros(3, dtype=torch.float64),
        atol=1e-6,
        rtol=0,
    )


def test_campaign_rejects_cross_order_raw_energy_mismatch(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path, mismatched_order=8)

    with pytest.raises(ValueError, match="raw energy.*order 8"):
        publish_e0_campaign(inputs, tmp_path / "campaign")


def test_campaign_reuses_identity_and_rejects_conflicting_output(
    tmp_path: Path,
) -> None:
    inputs = _inputs(tmp_path)
    first = publish_e0_campaign(inputs, tmp_path / "campaign")
    second = publish_e0_campaign(inputs, tmp_path / "campaign")
    assert second == first

    conflicting = E0CampaignInputs(
        **{
            **inputs.__dict__,
            "checkpoint_sha256": "f" * 64,
        }
    )
    with pytest.raises(ValueError, match="identity"):
        publish_e0_campaign(conflicting, tmp_path / "campaign")


def test_verifier_detects_internal_artifact_tampering(tmp_path: Path) -> None:
    root = publish_e0_campaign(_inputs(tmp_path), tmp_path / "campaign")
    metrics = root / "variants/uncorrected/metrics.json"
    metrics.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA mismatch"):
        verify_e0_campaign(root / "manifest.json", full=True)
