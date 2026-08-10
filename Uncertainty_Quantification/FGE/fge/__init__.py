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
from .checkpoint import (
    CheckpointBundle,
    LossContract,
    load_checkpoint_bundle,
    recover_loss_contract,
)
from .config import FGEConfig, load_config
from .data import DatasetIdentity
from .errors import HardFailure
from .evaluation import EvaluationArtifacts, evaluate_prediction, global_mae
from .manifests import (
    build_prediction_manifest,
    build_result_manifest,
    build_training_manifest,
)
from .members import (
    MemberPayload,
    ReadoutAudit,
    TensorFingerprint,
    apply_member,
    assert_frozen_unchanged,
    assert_readout_contract,
    frozen_fingerprint,
    load_member,
    pack_member,
    readout_tensor_names,
)
from .prediction import (
    PredictionShape,
    canonical_prediction,
    predict_members,
    validate_prediction_payload,
)
from .schedule import asymmetric_triangular_lr
from .uncertainty import (
    FORMULA_VERSION,
    population_std,
    reduce_force_by_structure,
    scalar_gmd,
    vector_gmd,
)


__all__ = [
    "CheckpointBundle",
    "DatasetIdentity",
    "LossContract",
    "MemberPayload",
    "PredictionShape",
    "ReadoutAudit",
    "TensorFingerprint",
    "ExperimentLayout",
    "FGEConfig",
    "HardFailure",
    "apply_member",
    "assert_frozen_unchanged",
    "assert_readout_contract",
    "atomic_torch_save",
    "atomic_write_json",
    "atomic_write_yaml",
    "build_prediction_manifest",
    "build_result_manifest",
    "build_training_manifest",
    "canonical_prediction",
    "predict_members",
    "frozen_fingerprint",
    "load_checkpoint_bundle",
    "load_config",
    "load_member",
    "normalize_artifact_path",
    "pack_member",
    "readout_tensor_names",
    "recover_loss_contract",
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
    "validate_prediction_payload",
]
