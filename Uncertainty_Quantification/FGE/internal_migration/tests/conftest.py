from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.tests.conftest import (
    config_payload as config_payload,  # noqa: F401
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def legacy_tree(tmp_path: Path):
    root = tmp_path / "legacy"
    member_root = root / "successful" / "members"
    prediction_root = root / "matpes_test_full" / "predictions" / "test"
    member_root.mkdir(parents=True)

    member_hashes: dict[str, str] = {}
    for index in (2, 1):
        member_id = f"member_{index:03d}"
        checkpoint = member_root / f"{member_id}.ckpt"
        checkpoint.write_bytes(f"checkpoint-{index}".encode())
        member_hashes[member_id] = _sha(checkpoint)

    n_atoms = (1, 2, 1)
    structure_ids = ("001", "002", "003")
    chunks = []
    member_chunks: dict[str, list[str]] = {}
    for member_index in (1, 2):
        member_id = f"member_{member_index:03d}"
        paths: list[str] = []
        for chunk_index, count in enumerate(n_atoms):
            path = prediction_root / member_id / f"chunk_{chunk_index:03d}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            value = float(member_index * 10 + chunk_index)
            payload = {
                "member_id": member_id,
                "split": "test",
                "chunk_index": chunk_index,
                "structure_start": chunk_index,
                "structure_ids": [structure_ids[chunk_index]],
                "n_atoms": torch.tensor([count], dtype=torch.int64),
                "atom_offsets": torch.tensor([0, count], dtype=torch.int64),
                "atom_to_structure": torch.zeros(count, dtype=torch.int64),
                "atomic_numbers": torch.arange(1, count + 1, dtype=torch.int64),
                "E_member": torch.tensor([value], dtype=torch.float32),
                "F_member": torch.full((count, 3), value, dtype=torch.float32),
                "S_member": torch.full((1, 3, 3), value, dtype=torch.float32),
                "E_ref": torch.tensor([float(chunk_index)], dtype=torch.float32),
                "F_ref": torch.full(
                    (count, 3), float(chunk_index), dtype=torch.float32
                ),
                "S_ref": torch.full((1, 3, 3), float(chunk_index), dtype=torch.float32),
            }
            torch.save(payload, path)
            paths.append(path.relative_to(prediction_root).as_posix())
            if member_index == 1:
                chunks.append(
                    {
                        "chunk_index": chunk_index,
                        "structure_start": chunk_index,
                        "structure_count": 1,
                    }
                )
        member_chunks[member_id] = paths

    summary = {
        "split": "test",
        "n_structures": 3,
        "n_atoms": list(n_atoms),
        "structure_ids": list(structure_ids),
        "member_ids": ["member_001", "member_002"],
        "inference_only": True,
        "dataset_identity": {
            "path": "/legacy/test.extxyz",
            "size_bytes": 123,
            "mtime_ns": 456,
            "sha256": "a" * 64,
            "n_structures": 3,
        },
        "chunks": chunks,
        "member_chunks": member_chunks,
    }
    _write_json(prediction_root / "prediction_summary.json", summary)

    members = []
    for index in (1, 2):
        member_id = f"member_{index:03d}"
        members.append(
            {
                "member_id": member_id,
                "checkpoint_path": str((member_root / f"{member_id}.ckpt").resolve()),
                "cycle_index": index,
                "global_step_after_optimizer": index * 10,
                "accepted": True,
                "status": "valid",
                "reload_check": "passed",
                "finite_check": "passed",
                "frozen_drift": 0.0,
            }
        )
    _write_json(
        root / "matpes_test_full" / "fge_manifest.json",
        {
            "inference_only": True,
            "members": members,
            "datasets": {"test": summary["dataset_identity"]},
        },
    )
    uncertainty = {
        "member_ids": ["member_001", "member_002"],
        "E_mean": torch.tensor([15.0, 16.0, 17.0]),
        "marker": torch.tensor([7.0]),
    }
    uncertainty_path = (
        root / "matpes_test_full" / "uncertainty" / "test" / "uncertainty.pt"
    )
    uncertainty_path.parent.mkdir(parents=True)
    torch.save(uncertainty, uncertainty_path)
    metrics = {"mae": {"mae_e": 1.25}, "metric_schema_version": 4}
    _write_json(
        root / "matpes_test_full" / "evaluation" / "test_final_metrics.json",
        metrics,
    )

    return root, member_hashes, uncertainty, metrics
