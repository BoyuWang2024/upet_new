"""Configuration-driven wrappers for external prediction and plotting."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..external_config import ExternalPredictionConfig
from ..external_plotting import (
    load_dataset_series,
    plot_cross_dataset_correlations,
    plot_dataset_suite,
)
from ..external_prediction import discover_external_runs, predict_external_datasets


def _selected_names(
    config: ExternalPredictionConfig,
    names: Sequence[str],
) -> tuple[str, ...]:
    selected = tuple(names) if names else tuple(config.datasets)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected external datasets must be unique and non-empty")
    missing = [name for name in selected if name not in config.datasets]
    if missing:
        raise ValueError(f"selected external dataset is not configured: {missing[0]}")
    return selected


def predict_external_from_config(
    config: ExternalPredictionConfig,
    names: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Predict selected external datasets using the exact completed run set."""

    return predict_external_datasets(config, names=_selected_names(config, names))


def plot_external_from_config(
    config: ExternalPredictionConfig,
    names: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Plot verified evaluation or dataset-scoped predictions without inference."""

    selected = _selected_names(config, names)
    runs = discover_external_runs(config.runs_root)
    publications = []
    for name in selected:
        source = config.datasets[name]
        if source.source == "existing_evaluation":
            directories = tuple(run / "evaluation" for run in runs.ordered())
        else:
            directories = tuple(run / "predictions" / name for run in runs.ordered())
        dataset = load_dataset_series(directories, name)
        publications.append(plot_dataset_suite(dataset, config.plots_root))
    manifests = [publication.manifest for publication in publications]
    if len(publications) == 3:
        manifests.append(
            plot_cross_dataset_correlations(
                publications,
                config.plots_root / "comparisons",
            )
        )
    return tuple(manifests)
