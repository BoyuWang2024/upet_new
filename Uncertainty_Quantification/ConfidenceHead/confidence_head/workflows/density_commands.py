"""Configuration-driven plotting of completed ConfidenceHead predictions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from ..density_comparisons import publish_density_cross_dataset
from ..density_config import DensityPlotConfig
from ..density_publication import publish_density_dataset
from ..external_config import ExternalPredictionConfig, load_external_config
from ..external_plotting import load_dataset_series
from ..external_prediction import discover_external_runs


def _selected_names(
    config: ExternalPredictionConfig,
    names: Sequence[str],
) -> tuple[str, ...]:
    selected = tuple(names) if names else tuple(config.datasets)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected density datasets must be unique and non-empty")
    missing = [name for name in selected if name not in config.datasets]
    if missing:
        raise ValueError(f"selected density dataset is not configured: {missing[0]}")
    return selected


def plot_density_from_config(
    config: DensityPlotConfig,
    names: Sequence[str] = (),
) -> tuple[Path, ...]:
    """Plot verified existing results without running model inference."""

    external = load_external_config(config.external_config)
    selected = _selected_names(external, names)
    runs = discover_external_runs(external.runs_root)
    settings = config.settings()
    publications = []
    for name in selected:
        source = external.datasets[name]
        if source.source == "existing_evaluation":
            directories = tuple(run / "evaluation" for run in runs.ordered())
        else:
            directories = tuple(run / "predictions" / name for run in runs.ordered())
        dataset = load_dataset_series(directories, name)
        publications.append(
            publish_density_dataset(dataset, config.output_root, settings)
        )
    manifests = [publication.manifest for publication in publications]
    if len(publications) == 3:
        manifests.append(
            publish_density_cross_dataset(
                publications,
                settings,
                config.output_root / "comparisons",
            )
        )
    return tuple(manifests)
