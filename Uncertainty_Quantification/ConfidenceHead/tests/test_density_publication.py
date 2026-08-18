from __future__ import annotations

import json
from pathlib import Path

import torch

from confidence_head.density_plotting import DensitySettings
from confidence_head.density_publication import (
    DensityPublication,
    publish_density_dataset,
    verify_density_publication,
)
from confidence_head.external_plotting import DatasetSeries
from confidence_head.plot_analysis import PlotSeries


def _series(root: Path, *, target: str, order: int | None) -> PlotSeries:
    observed = torch.logspace(-4, -1, 64, dtype=torch.float64)
    expected = observed * (1.05 if target == "force" else 1.0 + 0.01 * (order or 1))
    logits = torch.zeros((len(observed), 50), dtype=torch.float64)
    logits[:, 0] = 1.0
    return PlotSeries(
        run_dir=root / (f"energy-{order}" if target == "energy" else "force"),
        structure_ids=torch.arange(len(observed)),
        target=target,  # type: ignore[arg-type]
        order=order,
        logits=logits,
        observed=observed,
        expected=expected,
        representatives=torch.linspace(0.005, 0.495, 50, dtype=torch.float64),
        stored_metrics={"sample_count": len(observed), "pearson": 0.9, "spearman": 0.9},
        force_target_mode="atom_mean" if target == "force" else None,
    )


def _dataset(root: Path, name: str) -> DatasetSeries:
    return DatasetSeries(
        name=name,
        force=_series(root, target="force", order=None),
        energy_by_order={
            order: _series(root, target="energy", order=order)
            for order in range(1, 9)
        },
        input_manifests={f"run-{index}": f"{index:064x}" for index in range(9)},
    )


def test_density_publication_is_complete_and_preserves_legacy_outputs(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "argmax_bin_boxplots" / "old.png"
    legacy.parent.mkdir()
    legacy.write_bytes(b"legacy")

    publication = publish_density_dataset(
        _dataset(tmp_path / "runs", "mad_test"),
        tmp_path / "density_scatter",
        DensitySettings(grid_size=32, dpi=72),
    )

    assert isinstance(publication, DensityPublication)
    manifest = verify_density_publication(publication.manifest, full=True)
    assert manifest["schema_version"] == "upet_confidence_density_plots_v1"
    assert manifest["status"] == "complete"
    files = [path for path in publication.root.rglob("*") if path.is_file()]
    assert sum(path.suffix == ".png" for path in files) == 10
    assert sum(path.suffix == ".pdf" for path in files) == 10
    assert sum(path.suffix == ".csv" for path in files) == 10
    assert sum(path.suffix == ".json" for path in files) == 11
    assert legacy.read_bytes() == b"legacy"


def test_density_publication_reuses_identical_and_rejects_conflicting_inputs(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path / "runs", "mad_test")
    first = publish_density_dataset(
        dataset, tmp_path / "density_scatter", DensitySettings(grid_size=32, dpi=72)
    )
    second = publish_density_dataset(
        dataset, tmp_path / "density_scatter", DensitySettings(grid_size=32, dpi=72)
    )
    assert second.manifest == first.manifest

    changed = DatasetSeries(
        name=dataset.name,
        force=dataset.force,
        energy_by_order=dataset.energy_by_order,
        input_manifests={**dataset.input_manifests, "run-0": "f" * 64},
    )
    try:
        publish_density_dataset(
            changed, tmp_path / "density_scatter", DensitySettings(grid_size=32, dpi=72)
        )
    except ValueError as error:
        assert "identity" in str(error)
    else:
        raise AssertionError("conflicting density identity was accepted")


def test_density_manifest_contains_input_identity_payload(tmp_path: Path) -> None:
    publication = publish_density_dataset(
        _dataset(tmp_path / "runs", "mad_test"),
        tmp_path / "density_scatter",
        DensitySettings(grid_size=32, dpi=72),
    )
    manifest = json.loads(publication.manifest.read_text(encoding="utf-8"))
    assert manifest["identity_payload"]["dataset"] == "mad_test"
    assert len(manifest["identity_payload"]["inputs"]) == 9
