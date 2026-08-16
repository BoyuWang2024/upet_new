"""Equal-weight uncertainty reductions matching the legacy UPET FGE formula."""

from __future__ import annotations

import math

import torch


FORMULA_VERSION = "legacy_upet_fge_v1"


def _validate_members(members: torch.Tensor, *, minimum_rank: int, dim: int = 0) -> int:
    if not isinstance(members, torch.Tensor):
        raise TypeError("members must be a torch.Tensor")
    if members.device.type != "cpu":
        raise ValueError("members must be CPU tensors")
    if members.ndim < minimum_rank:
        raise ValueError(f"members tensor rank must be at least {minimum_rank}")
    if not members.is_floating_point():
        raise TypeError("members must have a floating dtype")
    if not bool(torch.isfinite(members).all().item()):
        raise ValueError("members must contain only finite values")
    if not isinstance(dim, int) or isinstance(dim, bool):
        raise TypeError("dim must be an integer")
    normalized_dim = dim if dim >= 0 else members.ndim + dim
    if not 0 <= normalized_dim < members.ndim:
        raise ValueError("dim is outside the members tensor rank")
    if members.shape[normalized_dim] < 2:
        raise ValueError("at least two members are required")
    return normalized_dim


def population_std(members: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Return population standard deviation along the member dimension."""
    member_dim = _validate_members(members, minimum_rank=1, dim=dim)
    return torch.std(members, dim=member_dim, unbiased=False)


def scalar_gmd(members: torch.Tensor) -> torch.Tensor:
    """Return ordered-pair scalar GMD along the leading member dimension."""
    _validate_members(members, minimum_rank=1)
    differences = torch.abs(members.unsqueeze(0) - members.unsqueeze(1))
    return differences.mean(dim=(0, 1))


def vector_gmd(members: torch.Tensor) -> torch.Tensor:
    """Return ordered-pair L2-vector GMD along the leading member dimension."""
    _validate_members(members, minimum_rank=2)
    differences = members.unsqueeze(0) - members.unsqueeze(1)
    return torch.linalg.vector_norm(differences, dim=-1).mean(dim=(0, 1))


def tensor_to_voigt_symmetric(stress: torch.Tensor) -> torch.Tensor:
    """Symmetrize trailing 3x3 tensors and return xx,yy,zz,yz,xz,xy."""
    if not isinstance(stress, torch.Tensor):
        raise TypeError("stress must be a torch.Tensor")
    if stress.device.type != "cpu":
        raise ValueError("stress must be stored on CPU")
    if not stress.is_floating_point():
        raise TypeError("stress must have a floating dtype")
    if stress.ndim < 2 or tuple(stress.shape[-2:]) != (3, 3):
        raise ValueError("stress must end in a 3x3 tensor")
    if not bool(torch.isfinite(stress).all()):
        raise ValueError("stress must contain only finite values")
    symmetric = 0.5 * (stress + stress.transpose(-1, -2))
    return torch.stack(
        (
            symmetric[..., 0, 0],
            symmetric[..., 1, 1],
            symmetric[..., 2, 2],
            symmetric[..., 1, 2],
            symmetric[..., 0, 2],
            symmetric[..., 0, 1],
        ),
        dim=-1,
    )


def reduce_force_by_structure(
    values: torch.Tensor, offsets: torch.Tensor, quantile: float = 0.95
) -> dict[str, torch.Tensor]:
    """Reduce per-atom force quantities into mean, max, and quantile by structure."""
    if not isinstance(values, torch.Tensor) or not isinstance(offsets, torch.Tensor):
        raise TypeError("values and offsets must be torch.Tensor instances")
    if values.device.type != "cpu" or offsets.device.type != "cpu":
        raise ValueError("values and offsets must be CPU tensors")
    if values.ndim != 1:
        raise ValueError("values must have rank 1")
    if offsets.ndim != 1:
        raise ValueError("offsets must have rank 1")
    if not values.is_floating_point():
        raise TypeError("values must have a floating dtype")
    if offsets.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError("offsets must have an integer dtype")
    if not bool(torch.isfinite(values).all().item()):
        raise ValueError("values must contain only finite values")
    if isinstance(quantile, bool) or not isinstance(quantile, (float, int)):
        raise TypeError("quantile must be a real number")
    quantile_value = float(quantile)
    if not math.isfinite(quantile_value) or not 0.0 <= quantile_value <= 1.0:
        raise ValueError("quantile must be finite and lie in [0, 1]")
    if values.numel() == 0 or offsets.numel() < 2:
        raise ValueError("at least one non-empty structure is required")
    if int(offsets[0].item()) != 0 or int(offsets[-1].item()) != values.numel():
        raise ValueError("offsets must span all values from zero")
    if not bool((offsets[1:] > offsets[:-1]).all().item()):
        raise ValueError("offsets must define strictly non-empty structures")

    means: list[torch.Tensor] = []
    maxima: list[torch.Tensor] = []
    quantiles: list[torch.Tensor] = []
    for start_tensor, end_tensor in zip(offsets[:-1], offsets[1:], strict=True):
        structure = values[int(start_tensor.item()) : int(end_tensor.item())]
        means.append(structure.mean())
        maxima.append(structure.max())
        quantiles.append(torch.quantile(structure, quantile_value))
    return {
        "mean": torch.stack(means),
        "max": torch.stack(maxima),
        "q95": torch.stack(quantiles),
    }
