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
from .evaluation import EvaluationArtifacts, evaluate_prediction, global_mae
from .schedule import asymmetric_triangular_lr
from .uncertainty import (
    FORMULA_VERSION,
    population_std,
    reduce_force_by_structure,
    scalar_gmd,
    vector_gmd,
)


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
    "EvaluationArtifacts",
    "FORMULA_VERSION",
    "asymmetric_triangular_lr",
    "evaluate_prediction",
    "global_mae",
    "population_std",
    "reduce_force_by_structure",
    "scalar_gmd",
    "vector_gmd",
]
