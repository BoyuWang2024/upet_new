import json
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import yaml

from Uncertainty_Quantification.LLPR.llpr.artifacts import sha256_file, verify_run
from Uncertainty_Quantification.LLPR.llpr.legacy import import_legacy


def _write_legacy_tree(root: Path) -> Path:
    results = root / "results"
    formal = results / "LLPR"
    fit = formal / "fit"
    fit.mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "configs").mkdir()
    (root / "scripts" / "compute.py").write_text("print('old')\n")
    (root / "configs" / "run.yaml").write_text("eta: 1e-6\n")
    (results / "run.log").write_text("complete\n")
    (results / "llpr_test_details.npz").write_bytes(b"legacy smoke")
    (fit / "reliability_matpes_linear_fit_summary.json").write_text("{}\n")
    (formal / "reliability_matpes_llpr.png").write_bytes(b"orphan")

    h_e = np.zeros((3, 3), dtype=np.float64)
    h_f = np.zeros((3, 3), dtype=np.float64)
    h_e[:2, :2] = np.diag([2.0, 3.0])
    h_f[2:, 2:] = np.array([[4.0]])
    np.savez_compressed(results / "H_E_full_run.npz", H=h_e)
    np.savez_compressed(results / "H_F_full_run.npz", H=h_f)
    np.savez_compressed(results / "H_EF_full_run.npz", H=h_e + h_f)

    details = {
        "structure_index": np.array([0], dtype=np.int64),
        "num_atoms": np.array([1], dtype=np.int64),
        "energy_pred_total": np.array([2.0]),
        "energy_true_total": np.array([1.5]),
        "energy_pred_per_atom": np.array([2.0]),
        "energy_true_per_atom": np.array([1.5]),
        "energy_residual": np.array([0.5]),
        "energy_raw_var": np.array([0.25]),
        "energy_calibrated_var": np.array([1.0]),
        "energy_calibrated_std": np.array([1.0]),
        "energy_inverse_variance": np.array([1.0]),
        "energy_raw_var_total_derived": np.array([0.25]),
        "energy_calibrated_var_total_derived": np.array([1.0]),
        "force_offsets": np.array([0, 3], dtype=np.int64),
        "force_structure_index": np.array([0, 0, 0], dtype=np.int64),
        "force_component_index_within_structure": np.array([0, 1, 2], dtype=np.int64),
        "force_atom_index": np.array([0, 0, 0], dtype=np.int64),
        "force_cartesian_index": np.array([0, 1, 2], dtype=np.int64),
        "force_pred": np.array([0.1, -0.2, 0.3]),
        "force_true": np.zeros(3),
        "force_residual": np.array([0.1, -0.2, 0.3]),
        "force_raw_var_component": np.array([1.0, 4.0, 9.0]),
        "force_calibrated_var_component": np.array([0.25, 1.0, 2.25]),
        "force_calibrated_std_component": np.array([0.5, 1.0, 1.5]),
        "force_inverse_variance_component": np.array([4.0, 1.0, 4 / 9]),
        "force_raw_var_atom_mean": np.array([14 / 3]),
        "force_calibrated_var_atom_mean": np.array([3.5 / 3]),
        "force_calibrated_std_atom_rms": np.array([np.sqrt(3.5 / 3)]),
        "force_raw_var_component_mean_structure": np.array([14 / 3]),
        "force_calibrated_var_component_mean_structure": np.array([3.5 / 3]),
        "force_calibrated_std_component_rms_structure": np.array([np.sqrt(3.5 / 3)]),
    }
    np.savez_compressed(formal / "llpr_test_full_gpu_details.npz", **details)
    summary = {
        "alpha_energy": 2.0,
        "alpha_force": 0.5,
        "alpha_energy_sq": 4.0,
        "alpha_force_sq": 0.25,
        "damping_eta": 1.0e-6,
        "dim_theta_E": 2,
        "dim_theta_F": 1,
        "dim_theta_total": 3,
        "num_structures_processed": 1,
        "num_structures_skipped": 0,
        "total_force_components": 3,
    }
    (formal / "llpr_test_full_gpu_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )
    return root


def _write_config(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / "import.yaml"
    inputs = tmp_path / "inputs"
    inputs.mkdir(exist_ok=True)
    input_paths = {}
    for name in ("checkpoint", "build", "validation", "test"):
        input_path = inputs / f"{name}.bin"
        input_path.write_bytes(f"audited-{name}".encode())
        input_paths[name] = input_path

    path.write_text(
        yaml.safe_dump(
            {
                "source_root": str(source),
                "destination_root": str(tmp_path / "outputs"),
                "experiment": "legacy",
                "checkpoint_path": str(input_paths["checkpoint"]),
                "build_path": str(input_paths["build"]),
                "validation_path": str(input_paths["validation"]),
                "test_path": str(input_paths["test"]),
                "checkpoint_sha256": sha256_file(input_paths["checkpoint"]),
                "build_sha256": sha256_file(input_paths["build"]),
                "validation_sha256": sha256_file(input_paths["validation"]),
                "test_sha256": sha256_file(input_paths["test"]),
                "expected_eta": {"energy": 1.0e-6, "force": 1.0e-6},
                "expected_alpha": {"energy": 2.0, "force": 0.5},
                "expected_dimensions": {
                    "energy": 2,
                    "force": 1,
                    "total": 3,
                },
                "expected_counts": {
                    "structures": 1,
                    "atoms": 1,
                    "force_components": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class ModuleThatRaisesOnAccess(ModuleType):
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"model access attempted: {name}")


def test_import_is_model_free_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _write_legacy_tree(tmp_path / "old")
    module = ModuleThatRaisesOnAccess("metatrain.utils.io")
    monkeypatch.setitem(sys.modules, "metatrain.utils.io", module)
    config = _write_config(tmp_path, source)

    first = import_legacy(config)
    details = next(first.rglob("details.npz"))
    first_mtime = details.stat().st_mtime_ns
    second = import_legacy(config)

    assert first == second
    assert details.stat().st_mtime_ns == first_mtime
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["origin"] == "legacy_import"
    assert manifest["legacy_fixed_ridge"] is True
    classifications = {
        row["classification"]
        for row in json.loads((first / "inventory.json").read_text())["files"]
    }
    assert classifications == {
        "authoritative",
        "legacy_smoke",
        "incomplete",
        "orphan",
    }


def test_tampered_joint_matrix_fails(tmp_path: Path) -> None:
    source = _write_legacy_tree(tmp_path / "old")
    path = source / "results/H_EF_full_run.npz"
    with np.load(path) as archive:
        matrix = archive["H"].copy()
    matrix[0, 0] += 1.0
    np.savez_compressed(path, H=matrix)

    with pytest.raises(ValueError, match="H_EF.*H_E.*H_F"):
        import_legacy(_write_config(tmp_path, source))


def test_changed_alpha_fails(tmp_path: Path) -> None:
    source = _write_legacy_tree(tmp_path / "old")
    path = source / "results/LLPR/llpr_test_full_gpu_summary.json"
    summary = json.loads(path.read_text())
    summary["alpha_force"] = 0.6
    path.write_text(json.dumps(summary))

    with pytest.raises(ValueError, match="Alpha"):
        import_legacy(_write_config(tmp_path, source))


@pytest.mark.parametrize("name", ["checkpoint", "build", "validation", "test"])
def test_changed_input_identity_fails(tmp_path: Path, name: str) -> None:
    source = _write_legacy_tree(tmp_path / "old")
    config = _write_config(tmp_path, source)
    raw = yaml.safe_load(config.read_text())
    raw[f"{name}_sha256"] = "0" * 64
    config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=f"{name}.*SHA mismatch"):
        import_legacy(config)


@pytest.mark.parametrize("mutation", ["tamper", "delete"])
def test_full_verify_protects_legacy_raw(tmp_path: Path, mutation: str) -> None:
    source = _write_legacy_tree(tmp_path / "old")
    config = _write_config(tmp_path, source)
    destination = import_legacy(config)
    raw = destination / "legacy_raw/scripts/compute.py"
    if mutation == "tamper":
        raw.write_text("changed\n", encoding="utf-8")
    else:
        raw.unlink()

    with pytest.raises(ValueError, match="legacy raw"):
        verify_run(destination, level="full")
    with pytest.raises(ValueError, match="legacy raw"):
        import_legacy(config)


def test_existing_different_identity_is_not_overwritten(tmp_path: Path) -> None:
    source = _write_legacy_tree(tmp_path / "old")
    destination = tmp_path / "outputs/legacy"
    destination.mkdir(parents=True)
    (destination / "sentinel").write_text("keep")
    (destination / "manifest.json").write_text(
        '{"status":"complete","identity":"different"}'
    )

    with pytest.raises(ValueError, match="identity collision"):
        import_legacy(_write_config(tmp_path, source))
    assert (destination / "sentinel").read_text() == "keep"
