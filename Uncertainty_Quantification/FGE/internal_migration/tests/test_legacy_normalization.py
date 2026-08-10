from __future__ import annotations

import csv
import json
from dataclasses import replace
from pathlib import Path

import torch

from Uncertainty_Quantification.FGE.fge.validation import validate_result
from Uncertainty_Quantification.FGE.internal_migration.migration.converter import (
    convert_legacy_run,
)

from .helpers import build_conversion_case


def test_converter_normalizes_preserved_real_legacy_uq_metrics_and_risks(
    legacy_tree, config_payload, tmp_path: Path
) -> None:
    source, base, config, expected = build_conversion_case(
        legacy_tree, config_payload, tmp_path
    )
    uncertainty_path = source / expected.uncertainty_path
    old = torch.load(uncertainty_path, map_location="cpu", weights_only=True)
    canonical = old.pop("canonical_uncertainty")
    old.update(
        {
            "U_E_total_std": canonical["energy_total"]["std"],
            "U_E_total_GMD": canonical["energy_total"]["gmd"],
            "U_E_atom_std": canonical["energy_per_atom"]["std"],
            "U_E_atom_GMD": canonical["energy_per_atom"]["gmd"],
            "U_F_component_std": canonical["force_component"]["std"],
            "U_F_component_GMD": canonical["force_component"]["gmd"],
            "U_F_atom_vector_std": canonical["force_atom_vector"]["std"],
            "U_F_atom_vector_GMD": canonical["force_atom_vector"]["gmd"],
            "U_F_structure_mean_std": canonical["force_structure"]["std"]["mean"],
            "U_F_structure_max_std": canonical["force_structure"]["std"]["max"],
            "U_F_structure_q95_std": canonical["force_structure"]["std"]["q95"],
        }
    )
    torch.save(old, uncertainty_path)

    metrics_path = source / expected.metrics_path
    canonical_metrics = json.loads(metrics_path.read_text())
    canonical_metrics.pop("report_inputs")
    reverse_names = {
        "energy_per_atom_std": "energy_atom_std",
        "energy_per_atom_gmd": "energy_atom_GMD",
        "energy_total_gmd": "energy_total_GMD",
        "force_atom_vector_gmd": "force_atom_vector_GMD",
        "force_component_gmd": "force_component_GMD",
    }
    legacy_correlations = {}
    for name, diagnostic in canonical_metrics["correlations"].items():
        legacy_correlations[reverse_names.get(name, name)] = {
            "pearson": diagnostic["pearson"],
            "spearman": diagnostic["spearman"],
            "disagreement_collapse": diagnostic["uncertainty_constant"],
            "n": diagnostic["n"],
        }
    mae = canonical_metrics["mae"]
    legacy_metrics = {
        "base_metrics": {
            "energy_total_mae": mae["energy_total"],
            "energy_per_atom_mae": mae["energy_per_atom"],
            "force_component_mae": mae["force_component"],
            "stress_component_mae": mae["stress_component"],
        },
        "correlations": legacy_correlations,
        "counts": {
            "structures": canonical_metrics["counts"]["structures"],
            "force_components": canonical_metrics["counts"]["force_components"],
            "stress_components": canonical_metrics["counts"]["stress_components"],
        },
        "mae": {
            "mae_e": mae["energy_per_atom"],
            "mae_f": mae["force_component"],
            "mae_s": mae["stress_component"],
        },
    }
    metrics_path.write_text(json.dumps(legacy_metrics) + "\n", encoding="utf-8")
    risk_paths = {}
    for name, rows in canonical_metrics["risk_coverage"].items():
        path = source / "matpes_test_full/evaluation" / f"risk_{name}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=("coverage", "risk"))
            writer.writeheader()
            for row in rows:
                writer.writerow({"coverage": row["coverage"], "risk": row["risk"]})
        risk_paths[name] = path.relative_to(source).as_posix()
    expected = replace(expected, risk_coverage_paths=risk_paths)

    destination = tmp_path / "published"
    convert_legacy_run(
        source,
        destination,
        tmp_path / "audit",
        config,
        base,
        expected=expected,
    )
    assert validate_result(config, destination).mode == "read_only"
