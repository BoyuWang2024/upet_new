"""Stage orchestration for MAD r2SCAN E0 postprocessing."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Literal

import torch

from ..artifacts import sha256_file
from ..checkpoint import load_upet_checkpoint
from ..e0_calibration import (
    E0Dataset,
    extract_model_e0,
    load_e0_dataset,
)
from ..e0_config import (
    E0PostprocessingConfig,
    ExpectedDatasetCounts,
)
from ..e0_plotting import publish_e0_density_campaign
from ..e0_publication import (
    E0CampaignInputs,
    publish_e0_campaign,
    verify_e0_campaign,
)
from ..external_cache import build_external_dataset_cache
from ..external_config import (
    ExternalPredictionConfig,
    load_external_config,
)
from ..external_prediction import (
    discover_energy_runs,
    predict_dataset_run,
    verify_external_prediction,
)


E0Stage = Literal["predict", "postprocess", "plot", "all"]
_ORDERS = tuple(range(1, 9))


def _load_inputs(
    config: E0PostprocessingConfig,
) -> tuple[ExternalPredictionConfig, Mapping[int, Path]]:
    external = load_external_config(config.external_config)
    names = (config.validation_dataset, config.test_dataset)
    if set(external.datasets) != set(names):
        raise ValueError(
            "E0 external config must contain only validation and test datasets"
        )
    for name in names:
        if getattr(external.datasets[name], "source", None) != "extxyz":
            raise ValueError(f"E0 dataset {name!r} must use an extxyz source")
    runs = discover_energy_runs(external.runs_root)
    return external, runs


def _prediction_sources(
    runs: Mapping[int, Path],
    dataset_name: str,
) -> dict[int, Path]:
    return {
        order: Path(runs[order]) / "predictions" / dataset_name for order in _ORDERS
    }


def predict_e0_inputs(
    config: E0PostprocessingConfig,
) -> tuple[Path, ...]:
    """Build two shared caches and run only the eight completed energy heads."""

    external, runs = _load_inputs(config)
    outputs: list[Path] = []
    for name in (config.validation_dataset, config.test_dataset):
        cache = build_external_dataset_cache(external, name)
        for order in _ORDERS:
            outputs.append(
                predict_dataset_run(
                    runs[order],
                    name,
                    cache,
                    batch_size=external.batch_size,
                    device_name=external.device,
                )
            )
    return tuple(outputs)


def _verify_sha(path: Path, expected: str, label: str) -> None:
    actual = sha256_file(path)
    if actual != expected:
        raise ValueError(f"{label} SHA mismatch: {actual} != {expected}")


def _verify_counts(
    dataset: E0Dataset,
    expected: ExpectedDatasetCounts,
    label: str,
) -> None:
    observed = {
        "structures": int(dataset.structure_ids.numel()),
        "atoms": int(dataset.atom_offsets[-1].item()),
        "elements": int(torch.unique(dataset.atomic_numbers).numel()),
    }
    required = {
        "structures": int(expected.structures),
        "atoms": int(expected.atoms),
        "elements": int(expected.elements),
    }
    if observed != required:
        raise ValueError(f"{label} dataset counts mismatch: {observed} != {required}")


def _preflight_raw_predictions(sources: dict[int, Path]) -> None:
    for order in _ORDERS:
        verify_external_prediction(
            sources[order] / "manifest.json",
            full=True,
        )


def postprocess_e0_from_config(
    config: E0PostprocessingConfig,
) -> Path:
    """Verify all inputs, calibrate E0 variants, and publish one campaign."""

    external, runs = _load_inputs(config)
    validation_source = external.datasets[config.validation_dataset]
    test_source = external.datasets[config.test_dataset]
    _verify_sha(
        external.checkpoint.path,
        external.checkpoint.expected_sha256,
        "checkpoint",
    )
    _verify_sha(
        validation_source.path,
        validation_source.expected_sha256,
        "validation dataset",
    )
    _verify_sha(
        test_source.path,
        test_source.expected_sha256,
        "test dataset",
    )
    validation_predictions = _prediction_sources(
        runs,
        config.validation_dataset,
    )
    test_predictions = _prediction_sources(runs, config.test_dataset)
    _preflight_raw_predictions(validation_predictions)
    _preflight_raw_predictions(test_predictions)

    checkpoint = load_upet_checkpoint(
        external.checkpoint.path,
        expected_sha256=external.checkpoint.expected_sha256,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    atomic_types, model_e0 = extract_model_e0(checkpoint.model)
    validation = load_e0_dataset(
        validation_source.path,
        expected_sha256=validation_source.expected_sha256,
        atomic_types=atomic_types,
    )
    test = load_e0_dataset(
        test_source.path,
        expected_sha256=test_source.expected_sha256,
        atomic_types=atomic_types,
    )
    _verify_counts(validation, config.validation_expected, "validation")
    _verify_counts(test, config.test_expected, "test")
    return publish_e0_campaign(
        E0CampaignInputs(
            checkpoint_sha256=checkpoint.sha256,
            validation_sha256=validation_source.expected_sha256,
            test_sha256=test_source.expected_sha256,
            atomic_types=atomic_types,
            model_e0=model_e0,
            validation=validation,
            test=test,
            validation_sources=validation_predictions,
            test_sources=test_predictions,
        ),
        config.output_root,
    )


def plot_e0_from_config(config: E0PostprocessingConfig) -> Path:
    """Plot only a complete campaign and its SHA-bound raw predictions."""

    verify_e0_campaign(config.output_root / "manifest.json", full=True)
    return publish_e0_density_campaign(
        config.output_root,
        config.plots_root,
        config.plot.settings(),
    )


def dispatch_e0_stage(
    config: E0PostprocessingConfig,
    stage: str,
) -> tuple[Path, ...]:
    """Run one explicit stage or the complete predict/postprocess/plot sequence."""

    if stage == "predict":
        return predict_e0_inputs(config)
    if stage == "postprocess":
        return (postprocess_e0_from_config(config),)
    if stage == "plot":
        return (plot_e0_from_config(config),)
    if stage == "all":
        return (
            *predict_e0_inputs(config),
            postprocess_e0_from_config(config),
            plot_e0_from_config(config),
        )
    raise ValueError("stage must be predict, postprocess, plot, or all")
