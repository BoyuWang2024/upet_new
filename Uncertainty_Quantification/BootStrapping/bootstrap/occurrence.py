"""Occurrence-aware dataset views for replacement bootstrap samples."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from metatensor.torch import Labels, TensorBlock, TensorMap
from torch.utils.data import Dataset


def relabel_system_samples(labels: Labels, occurrence_id: int) -> Labels:
    """Replace a labels ``system`` column with one occurrence identifier."""

    if "system" not in labels.names:
        return labels
    values = labels.values.clone()
    values[:, labels.names.index("system")] = occurrence_id
    return Labels(labels.names, values)


def relabel_system_ids(tensor_map: TensorMap, occurrence_id: int) -> TensorMap:
    """Rebuild a TensorMap with one occurrence ID in all system columns."""

    blocks: list[TensorBlock] = []
    for _, old_block in tensor_map.items():  # noqa: PERF102
        new_block = TensorBlock(
            values=old_block.values,
            samples=relabel_system_samples(old_block.samples, occurrence_id),
            components=old_block.components,
            properties=old_block.properties,
        )
        for gradient_name in old_block.gradients_list():
            old_gradient = old_block.gradient(gradient_name)
            new_block.add_gradient(
                gradient_name,
                TensorBlock(
                    values=old_gradient.values,
                    samples=relabel_system_samples(old_gradient.samples, occurrence_id),
                    components=old_gradient.components,
                    properties=old_gradient.properties,
                ),
            )
        blocks.append(new_block)
    return TensorMap(keys=tensor_map.keys, blocks=blocks)


def _is_tensor_map(value: Any) -> bool:
    return isinstance(value, torch.ScriptObject) and value._type().name() == "TensorMap"


def relabel_sample_system_ids(sample: Any, occurrence_id: int) -> Any:
    """Relabel every TensorMap field in one dataset sample."""

    if not hasattr(sample, "_fields"):
        return sample
    values = [
        relabel_system_ids(value, occurrence_id) if _is_tensor_map(value) else value
        for value in sample
    ]
    return type(sample)(*values)


class BootstrapOccurrenceDataset(Dataset):
    """Map occurrence positions to source rows with unique sample IDs."""

    def __init__(self, base_dataset: Any, source_indices: Sequence[int]) -> None:
        self.base_dataset = base_dataset
        self.source_indices = tuple(int(index) for index in source_indices)

    def __len__(self) -> int:
        return len(self.source_indices)

    def __getitem__(self, occurrence_id: int):
        if occurrence_id < 0 or occurrence_id >= len(self):
            raise IndexError("bootstrap occurrence index is outside dataset")
        sample = self.base_dataset[self.source_indices[occurrence_id]]
        return relabel_sample_system_ids(sample, occurrence_id)


class BootstrapOccurrenceCollate:
    """Map shuffled occurrence labels to contiguous batch-local system IDs."""

    def __init__(self, base_collate: Any) -> None:
        self.base_collate = base_collate

    def __call__(self, batch: list[Any]) -> Any:
        relabeled = [
            relabel_sample_system_ids(sample, batch_id)
            for batch_id, sample in enumerate(batch)
        ]
        return self.base_collate(relabeled)
