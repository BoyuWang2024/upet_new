"""Foundational FGE configuration and artifact interfaces."""

from .artifacts import (
    ExperimentLayout,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_yaml,
    normalize_artifact_path,
    sha256_file,
    sibling_staging,
)
from .config import FGEConfig, load_config
from .errors import HardFailure


__all__ = [
    "ExperimentLayout",
    "FGEConfig",
    "HardFailure",
    "atomic_torch_save",
    "atomic_write_json",
    "atomic_write_yaml",
    "load_config",
    "normalize_artifact_path",
    "sha256_file",
    "sibling_staging",
]
