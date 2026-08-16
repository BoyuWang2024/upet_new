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
from .evaluation import (
    EvaluationArtifacts,
    evaluate_fge,
    evaluate_prediction,
    global_mae,
)
from .inference_config import InferenceConfig, load_inference_config
from .inference_evaluation import evaluate_inference_dataset
from .inference_only import predict_inference_dataset
from .inference_validation import (
    InferenceValidationReport,
    validate_inference_result,
)
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
from .plot_analysis import (
    PlotAnalysisResult,
    PlotConfig,
    PlotInput,
    PlotSettings,
    analyze_plot_input,
    load_completed_fge_plot_input,
    load_inference_plot_input,
    load_plot_config,
)
from .plot_rendering import render_plot_suite
from .prediction import (
    PredictionShape,
    canonical_prediction,
    predict_members,
    validate_prediction_payload,
)
from .preflight import run_preflight
from .schedule import asymmetric_triangular_lr
from .training import PETTrainingRuntime, train_fge
from .uncertainty import (
    FORMULA_VERSION,
    population_std,
    reduce_force_by_structure,
    scalar_gmd,
    vector_gmd,
)
from .validation import ValidationReport, schema_signature, validate_result


__all__ = [
    "CheckpointBundle",
    "DatasetIdentity",
    "LossContract",
    "MemberPayload",
    "PredictionShape",
    "ReadoutAudit",
    "TensorFingerprint",
    "ValidationReport",
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
    "run_preflight",
    "schema_signature",
    "validate_result",
    "EvaluationArtifacts",
    "FORMULA_VERSION",
    "asymmetric_triangular_lr",
    "evaluate_fge",
    "evaluate_prediction",
    "global_mae",
    "population_std",
    "reduce_force_by_structure",
    "scalar_gmd",
    "vector_gmd",
    "train_fge",
    "PETTrainingRuntime",
    "validate_prediction_payload",
    "InferenceConfig",
    "InferenceValidationReport",
    "PlotAnalysisResult",
    "PlotConfig",
    "PlotInput",
    "PlotSettings",
    "analyze_plot_input",
    "evaluate_inference_dataset",
    "load_completed_fge_plot_input",
    "load_inference_config",
    "load_inference_plot_input",
    "load_plot_config",
    "predict_inference_dataset",
    "render_plot_suite",
    "validate_inference_result",
]
