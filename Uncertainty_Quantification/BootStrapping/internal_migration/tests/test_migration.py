from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _config(member_count: int = 2):
    return SimpleNamespace(
        bootstrap=SimpleNamespace(ensemble_size=member_count),
        prediction=SimpleNamespace(splits=("val", "test"), parameter_modes=("raw",)),
        data=SimpleNamespace(
            units=SimpleNamespace(
                energy="eV", forces="eV/Angstrom", stress="eV/Angstrom^3"
            )
        ),
    )


def _write_legacy_tree(root: Path) -> Path:
    for member in range(1, 3):
        member_id = f"member_{member:02d}"
        member_root = root / member_id
        member_root.mkdir(parents=True)
        checkpoint = {
            "schema_version": 1,
            "member_index": member,
            "epoch": 4,
            "raw_state": {"head": torch.tensor([float(member)])},
            "ema_state": {"head": torch.tensor([float(member) + 0.1])},
        }
        for kind in ("best", "final", "latest"):
            torch.save(checkpoint, member_root / f"{kind}.pt")
        for split in ("val", "test"):
            for chunk_index, (start, atoms) in enumerate(((0, 2), (1, 1))):
                chunk_root = root / "predictions" / split / member_id
                chunk_root.mkdir(parents=True, exist_ok=True)
                energy = torch.tensor([member * 10.0 + start], dtype=torch.float64)
                payload = {
                    "identity": {
                        "split": split,
                        "branch": "raw",
                        "member_id": member_id,
                        "chunk_index": chunk_index,
                        "structure_start": start,
                        "structure_count": 1,
                    },
                    "structure_ids": [f"{split}-{start}"],
                    "n_atoms": torch.tensor([atoms]),
                    "atom_offsets": torch.tensor([0, atoms]),
                    "E_member": energy,
                    "F_member": torch.full((atoms, 3), float(member + start)),
                    "S_member": torch.full((1, 3, 3), float(member + start)),
                    "E_ref": torch.tensor([-float(start + 1)]),
                    "F_ref": torch.full((atoms, 3), -float(start + 1)),
                    "S_ref": torch.full((1, 3, 3), -float(start + 1)),
                }
                torch.save(payload, chunk_root / f"chunk_{chunk_index:05d}.pt")
    return root


def test_reader_rejects_missing_or_reordered_chunks(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.internal_migration.migration.legacy_reader import (
        inspect_legacy_run,
    )

    source = _write_legacy_tree(tmp_path / "source")
    missing = source / "predictions/val/member_02/chunk_00001.pt"
    missing.unlink()
    with pytest.raises(HardFailure, match="chunk"):
        inspect_legacy_run(source, _config())


def test_converter_copies_checkpoints_and_normalizes_predictions_without_compute(
    tmp_path: Path, monkeypatch
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import prediction, training
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
        load_prediction_arrays,
        load_target_arrays,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap import uncertainty
    from Uncertainty_Quantification.BootStrapping.internal_migration.migration.converter import (
        convert_legacy_run,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("migration called compute")

    monkeypatch.setattr(torch.nn.Module, "forward", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(training, "train_ensemble", forbidden, raising=False)
    monkeypatch.setattr(prediction, "predict_members", forbidden, raising=False)
    monkeypatch.setattr(uncertainty, "compute_uncertainty", forbidden, raising=False)
    source = _write_legacy_tree(tmp_path / "source")
    destination = tmp_path / "published"
    audit_root = tmp_path / "audit"

    publication = convert_legacy_run(source, destination, _config(), audit_root)

    assert publication.member_count == 2
    for kind in ("best", "final", "latest"):
        assert (
            destination / f"members/member_000/checkpoints/{kind}.pt"
        ).read_bytes() == (source / f"member_01/{kind}.pt").read_bytes()
    targets = load_target_arrays(destination / "predictions/test/targets.npz")
    member = load_prediction_arrays(
        destination / "predictions/test/members/member_000/raw.npz"
    )
    assert targets.structure_ids.tolist() == ["test-0", "test-1"]
    assert targets.atom_offsets.tolist() == [0, 2, 3]
    assert member.energy.tolist() == [10.0, 11.0]
    assert not (destination / "uncertainty").exists()
    assert (audit_root / "migration_audit.json").is_file()
    formal_text = "".join(
        path.read_text(encoding="utf-8") for path in destination.rglob("*.json")
    ).lower()
    assert "legacy" not in formal_text
    assert "migration" not in formal_text


def test_converter_rejects_overlapping_source_and_destination(tmp_path: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
    from Uncertainty_Quantification.BootStrapping.internal_migration.migration.converter import (
        convert_legacy_run,
    )

    source = _write_legacy_tree(tmp_path / "source")
    with pytest.raises(HardFailure, match="separate"):
        convert_legacy_run(source, source / "published", _config(), tmp_path / "audit")
