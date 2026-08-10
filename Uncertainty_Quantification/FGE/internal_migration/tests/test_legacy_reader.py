from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from Uncertainty_Quantification.FGE.fge.errors import HardFailure
from Uncertainty_Quantification.FGE.internal_migration.migration.legacy_reader import (
    LegacyExpectations,
    read_legacy_run,
    verify_source_unchanged,
)


def _expectations(member_hashes: dict[str, str]) -> LegacyExpectations:
    return LegacyExpectations(
        member_count=2,
        chunk_count=3,
        member_sha256=member_hashes,
        test_data_sha256="a" * 64,
        manifest_path="matpes_test_full/fge_manifest.json",
        members_directory="successful/members",
        prediction_directory="matpes_test_full/predictions/test",
        uncertainty_path="matpes_test_full/uncertainty/test/uncertainty.pt",
        metrics_path="matpes_test_full/evaluation/test_final_metrics.json",
    )


def test_reader_merges_members_and_chunks_in_numeric_order(legacy_tree) -> None:
    root, member_hashes, uncertainty, metrics = legacy_tree

    run = read_legacy_run(root, _expectations(member_hashes))

    assert tuple(member.member_id for member in run.members) == (
        "member_001",
        "member_002",
    )
    assert tuple(member.global_step for member in run.members) == (10, 20)
    assert run.prediction["member_ids"] == ("member_001", "member_002")
    assert run.prediction["structure_ids"] == ("001", "002", "003")
    assert torch.equal(
        run.prediction["energy_prediction"],
        torch.tensor([[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]),
    )
    assert torch.equal(
        run.prediction["energy_reference"], torch.tensor([0.0, 1.0, 2.0])
    )
    assert torch.equal(run.prediction["n_atoms"], torch.tensor([1, 2, 1]))
    assert torch.equal(run.prediction["structure_offsets"], torch.tensor([0, 1, 3, 4]))
    assert torch.equal(run.prediction["structure_mapping"], torch.tensor([0, 1, 1, 2]))
    assert torch.equal(run.legacy_uncertainty["marker"], uncertainty["marker"])
    assert run.legacy_metrics == metrics
    verify_source_unchanged(run.source_snapshot)


@pytest.mark.parametrize(
    "defect",
    ["crash", "hash", "missing_chunk", "reference", "path_escape"],
)
def test_reader_rejects_invalid_legacy_source(legacy_tree, defect: str) -> None:
    root, member_hashes, _, _ = legacy_tree
    manifest_path = root / "matpes_test_full" / "fge_manifest.json"
    summary_path = (
        root / "matpes_test_full" / "predictions" / "test" / "prediction_summary.json"
    )
    if defect == "crash":
        manifest = json.loads(manifest_path.read_text())
        manifest["members"][0]["status"] = "crashed"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif defect == "hash":
        member_hashes = dict(member_hashes)
        member_hashes["member_001"] = "f" * 64
    elif defect == "missing_chunk":
        (root / "matpes_test_full/predictions/test/member_002/chunk_001.pt").unlink()
    elif defect == "reference":
        path = root / "matpes_test_full/predictions/test/member_002/chunk_001.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["E_ref"] = payload["E_ref"] + 1
        torch.save(payload, path)
    else:
        summary = json.loads(summary_path.read_text())
        summary["member_chunks"]["member_001"][0] = "../outside.pt"
        summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(HardFailure):
        read_legacy_run(root, _expectations(member_hashes))


def test_source_snapshot_detects_post_read_mutation(legacy_tree) -> None:
    root, member_hashes, _, _ = legacy_tree
    run = read_legacy_run(root, _expectations(member_hashes))
    path = root / "matpes_test_full/evaluation/test_final_metrics.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(HardFailure, match="source changed"):
        verify_source_unchanged(run.source_snapshot)


def test_reader_rejects_symlink_source_escape(legacy_tree, tmp_path: Path) -> None:
    root, member_hashes, _, _ = legacy_tree
    outside = tmp_path / "outside.ckpt"
    outside.write_bytes(b"checkpoint-1")
    member = root / "successful/members/member_001.ckpt"
    member.unlink()
    member.symlink_to(outside)

    with pytest.raises(HardFailure):
        read_legacy_run(root, _expectations(member_hashes))
