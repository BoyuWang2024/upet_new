from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


def test_dataset_layout_uses_configured_files_without_fingerprint(
    tmp_path: Path,
) -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.data import DatasetLayout
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure

    paths = {name: tmp_path / f"{name}.extxyz" for name in ("train", "val", "test")}
    for path in paths.values():
        path.write_text("fixture", encoding="utf-8")
    layout = DatasetLayout(**paths)

    assert layout.for_split("train") == paths["train"]
    assert layout.for_split("val") == paths["val"]
    assert layout.for_split("test") == paths["test"]
    assert not hasattr(layout, "fingerprint")
    with pytest.raises(HardFailure, match="split"):
        layout.for_split("holdout")


def test_occurrence_ids_distinguish_duplicate_source_rows() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.data import (
        OccurrenceDataset,
    )

    view = OccurrenceDataset(["a", "b"], np.array([0, 0, 1], dtype=np.int64))

    assert [view[index].value for index in range(len(view))] == ["a", "a", "b"]
    assert view[0].source_index == view[1].source_index == 0
    assert view[0].occurrence_id != view[1].occurrence_id
    assert view[0].occurrence_id == "source_000000000_occurrence_000000000"


def test_occurrence_dataset_rejects_out_of_bounds_index() -> None:
    from Uncertainty_Quantification.BootStrapping.bootstrap.data import (
        OccurrenceDataset,
    )
    from Uncertainty_Quantification.BootStrapping.bootstrap.errors import HardFailure

    with pytest.raises(HardFailure, match="out of bounds"):
        OccurrenceDataset(["a"], np.array([1], dtype=np.int64))
