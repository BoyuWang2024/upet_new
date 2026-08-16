from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from Uncertainty_Quantification.BootStrapping.bootstrap.campaign import (
    CampaignConfig,
    CampaignDataset,
    CampaignPrediction,
    CampaignRun,
)
from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure
from Uncertainty_Quantification.BootStrapping.bootstrap.prediction import (
    PredictionArrays,
    PredictionStore,
    TargetArrays,
)


UNITS = {"energy": "eV", "forces": "eV/Angstrom", "stress": "eV/Angstrom^3"}


@dataclass
class FakeRuntime:
    monkeypatch: pytest.MonkeyPatch
    fail_member: int | None = None

    def __post_init__(self) -> None:
        from Uncertainty_Quantification.BootStrapping.bootstrap import native_prediction

        self.monkeypatch.setattr(native_prediction, "_read_atoms", self.read_atoms)
        self.monkeypatch.setattr(
            native_prediction, "load_pet_member_model", self.load_member
        )
        self.monkeypatch.setattr(
            native_prediction, "_predict_dataset", self.predict_member
        )

    def request(self, root: Path):
        from Uncertainty_Quantification.BootStrapping.bootstrap.native_prediction import (
            DatasetPredictionRequest,
        )

        return DatasetPredictionRequest(
            run=SimpleNamespace(
                run_root=root,
                config=SimpleNamespace(checkpoint=SimpleNamespace(base_path="base.ckpt")),
            ),
            dataset=CampaignDataset(
                label="mad_test",
                storage_key="mad_test",
                path=root / "mad-test.xyz",
                reference_targets=("energy", "forces"),
            ),
            mode="raw",
            member_count=2,
            device=torch.device("cpu"),
            dtype=torch.float64,
            batch_size=4,
        )

    @staticmethod
    def read_atoms(path: Path, limit: int | None) -> list[object]:
        class MadAtoms:
            def __init__(self, index: int, atom_count: int) -> None:
                self.info = {"structure_id": f"mad-{index}"}
                self.atom_count = atom_count

            def __len__(self) -> int:
                return self.atom_count

            def get_potential_energy(self) -> float:
                return float(-self.atom_count)

            def get_forces(self) -> np.ndarray:
                return np.zeros((self.atom_count, 3), dtype=np.float64)

        del path, limit
        return [MadAtoms(0, 2), MadAtoms(1, 1)]

    def load_member(self, *args: object, **kwargs: object) -> object:
        member_path = Path(args[1])
        member_index = int(member_path.parts[-3].split("_")[1])
        if member_index == self.fail_member:
            raise HardFailure(f"member {member_index} failed")
        return member_index

    @staticmethod
    def predict_member(
        model: object,
        atoms: list[object],
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> PredictionArrays:
        del atoms, batch_size, device, dtype
        return PredictionArrays(
            energy=np.array([-1.0, -2.0], dtype=np.float64) + int(model),
            forces=np.zeros((3, 3), dtype=np.float64),
            stress=np.zeros((2, 3, 3), dtype=np.float64),
        )


@pytest.fixture
def fake_runtime(monkeypatch: pytest.MonkeyPatch) -> FakeRuntime:
    return FakeRuntime(monkeypatch)


def _mad_targets() -> TargetArrays:
    return TargetArrays(
        structure_ids=np.array(["m0", "m1"]),
        num_atoms=np.array([2, 1], dtype=np.int64),
        atom_offsets=np.array([0, 2, 3], dtype=np.int64),
        energy=np.array([-1.0, -2.0], dtype=np.float64),
        forces=np.zeros((3, 3), dtype=np.float64),
        stress=None,
    )


def _member_values() -> PredictionArrays:
    return PredictionArrays(
        energy=np.array([-1.0, -2.0], dtype=np.float64),
        forces=np.zeros((3, 3), dtype=np.float64),
        stress=np.zeros((2, 3, 3), dtype=np.float64),
    )


def _write_existing_v1_test_publication(run_root: Path) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.artifacts import (
        atomic_write_json,
        sha256_file,
    )

    store = PredictionStore.at_split_root(
        run_root / "predictions" / "test", split="test", units=UNITS
    )
    store.write_targets(
        TargetArrays(
            structure_ids=np.array(["s0", "s1"]),
            num_atoms=np.array([2, 1], dtype=np.int64),
            atom_offsets=np.array([0, 2, 3], dtype=np.int64),
            energy=np.array([-1.0, -2.0]),
            forces=np.zeros((3, 3)),
            stress=np.zeros((2, 3, 3)),
        )
    )
    records = []
    for index in range(2):
        publication = store.write_member(index, "raw", _member_values())
        records.append(
            {
                "member_index": index,
                "mode": "raw",
                "path": str(publication.path.relative_to(store.split_root)),
                "sha256": publication.sha256,
                "shapes": publication.shapes,
                "dtypes": publication.dtypes,
            }
        )
    atomic_write_json(
        store.split_root / "manifest.json",
        {
            "schema": "upet.bootstrap.predictions/v1",
            "split": "test",
            "units": UNITS,
            "member_count": 2,
            "targets": {
                "structure_limit": None,
                "path": "targets.npz",
                "sha256": sha256_file(store.targets_path),
            },
            "members": records,
        },
    )


@pytest.fixture
def campaign(tmp_path: Path) -> CampaignConfig:
    run_root = tmp_path / "full_remote_b8_e8"
    return CampaignConfig(
        schema_version=1,
        runs=(
            CampaignRun(
                label="full_remote_b8_e8",
                config_path=tmp_path / "run.yaml",
                run_root=run_root,
                config=SimpleNamespace(),
            ),
        ),
        datasets=(
            CampaignDataset(
                label="matpes_test",
                storage_key="test",
                path=tmp_path / "matpes-test.xyz",
                reference_targets=("energy", "forces", "stress"),
            ),
        ),
        prediction=CampaignPrediction(
            mode="raw", member_count=2, device="cpu", batch_size=4
        ),
        plot=SimpleNamespace(),
        output_root=tmp_path / "output",
        source_path=tmp_path / "campaign.yaml",
    )


def test_predict_dataset_publishes_manifest_last(
    tmp_path: Path, fake_runtime: FakeRuntime
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.native_prediction import (
        predict_dataset,
    )

    manifest = predict_dataset(fake_runtime.request(tmp_path))

    document = json.loads(manifest.read_text())
    assert document["schema"] == "upet.bootstrap.predictions/v2"
    assert document["dataset_label"] == "mad_test"
    assert document["reference_targets"] == ["energy", "forces"]
    assert document["member_count"] == 2
    assert len(document["members"]) == 2
    assert all(
        (manifest.parent / record["path"]).is_file() for record in document["members"]
    )


def test_predict_campaign_reuses_complete_test_without_loader_calls(
    campaign: CampaignConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap import native_prediction
    from Uncertainty_Quantification.BootStrapping.bootstrap.native_prediction import (
        predict_campaign,
    )

    _write_existing_v1_test_publication(campaign.runs[0].run_root)

    def fail_if_called(*args: object, **kwargs: object) -> object:
        raise AssertionError("a valid existing publication must be reused")

    monkeypatch.setattr(native_prediction, "load_pet_member_model", fail_if_called)
    publications = predict_campaign(
        campaign, ("full_remote_b8_e8",), ("matpes_test",)
    )

    assert publications[0].skipped is True


def test_member_failure_does_not_publish_destination(
    tmp_path: Path, fake_runtime: FakeRuntime
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.native_prediction import (
        predict_dataset,
    )

    fake_runtime.fail_member = 1
    with pytest.raises(HardFailure, match="member 1"):
        predict_dataset(fake_runtime.request(tmp_path))

    assert not (tmp_path / "predictions" / "mad_test").exists()


def test_validation_rejects_an_extra_declared_member_record(
    tmp_path: Path, fake_runtime: FakeRuntime
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.native_prediction import (
        predict_dataset,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.prediction_publication import (
        validate_prediction_publication,
    )

    manifest = predict_dataset(fake_runtime.request(tmp_path))
    document = json.loads(manifest.read_text())
    document["members"].append(document["members"][0])
    manifest.unlink()
    manifest.write_text(json.dumps(document))

    with pytest.raises(HardFailure, match="member records"):
        validate_prediction_publication(
            manifest.parent,
            dataset_key="mad_test",
            mode="raw",
            member_count=2,
            reference_targets=("energy", "forces"),
        )
