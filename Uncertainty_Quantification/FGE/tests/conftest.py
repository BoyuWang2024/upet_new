from __future__ import annotations

import copy

import pytest


SHA_BASE = "879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004"
SHA_TRAIN = "12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec"
SHA_TEST = "1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d"


@pytest.fixture
def config_payload() -> dict[str, object]:
    """Return the complete formal FGE configuration contract."""
    return {
        "schema_version": "upet.fge.v1",
        "project": {"name": "upet_fge_full", "method": "FGE", "backend": "upet"},
        "paths": {
            "base_checkpoint": "models/base.ckpt",
            "train_data": "data/train.extxyz",
            "val_data": "data/val.extxyz",
            "test_data": "data/test.extxyz",
            "output_root": "outputs",
        },
        "identity": {
            "base_checkpoint_sha256": SHA_BASE,
            "train_data_sha256": SHA_TRAIN,
            "val_data_sha256": SHA_TRAIN,
            "test_data_sha256": SHA_TEST,
        },
        "data": {
            "format": "extxyz",
            "energy_target": "energy",
            "forces_target": "non_conservative_forces",
            "stress_target": "non_conservative_stress",
            "energy_unit": "eV",
            "forces_unit": "eV/angstrom",
            "stress_unit": "eV/angstrom^3",
        },
        "training": {
            "mode": "readout_only_official_upet",
            "restart_state": "model_state_dict",
            "trainable_prefixes": ["node_last_layers.", "edge_last_layers."],
            "expected_readout_tensor_count": 12,
            "expected_readout_parameter_count": 13338,
            "seed": 2026,
            "batch_size": 16,
            "validation_batch_size": 16,
            "drop_last": True,
            "weight_decay": 0.0,
            "dtype": "float32",
            "device": "cuda",
            "num_workers": 0,
            "resume": True,
        },
        "fge": {
            "member_count": 8,
            "cycles": 8,
            "epochs_per_cycle": 8,
            "schedule": "asymmetric_triangular",
            "lr_min": 1e-8,
            "lr_max": 1e-7,
            "rise_fraction": 0.2,
        },
        "ema": {
            "enabled": True,
            "decay": 0.999,
            "member_source": "raw_endpoint",
            "usage": "validation_only",
        },
        "prediction": {"split": "test", "batch_size": 16, "compute_stress": True},
        "evaluation": {
            "formula_version": "legacy_upet_fge_v1",
            "metric_schema_version": 4,
            "structure_quantile": 0.95,
            "constant_tolerance": 1e-12,
            "risk_coverages": [1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1],
        },
        "scientific": {
            "training": {
                "path_feasibility_only": True,
                "split_leakage": True,
                "scientific_evaluation": False,
                "inference_only": False,
            },
            "evaluation": {
                "path_feasibility_only": False,
                "split_leakage": False,
                "scientific_evaluation": True,
                "inference_only": True,
            },
        },
    }


@pytest.fixture
def copy_config(config_payload: dict[str, object]):
    return lambda: copy.deepcopy(config_payload)
