# UPET LLPR Legacy Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a current-UPET LLPR pipeline that can recompute curvature, calibration, evaluation, and plots, while importing the audited legacy fixed-eta results without rerunning the full datasets.

**Architecture:** A modular computation package writes one stage-aware artifact protocol. The recomputation path and an isolated legacy importer converge on that protocol; verification and plotting consume only complete canonical artifacts. Fixed eta exactly preserves the legacy definition, while fitted eta independently selects energy and force values on validation data.

**Tech Stack:** Python 3.11, PyTorch, NumPy, ASE, metatrain/metatomic, SciPy, Matplotlib, PyYAML, Pydantic v2, pytest, tox.

## Global Constraints

- Activate the runtime with `conda activate upet_new`.
- Do not modify `/home/lilong/code/UQ/upet/UQ_LLPR`.
- Do not run full train/validation/test model computation during this migration.
- The only model-based end-to-end calculation allowed in this migration is `data/dataset/matpes_n20.extxyz`.
- Preserve legacy formal results with `eta_energy=eta_force=1.0e-6`.
- Preserve `alpha_energy=1.1467388818005693` and `alpha_force=0.2095766082027508`.
- Preserve 19,374 test structures, 149,321 atoms, and 447,963 force components.
- Energy curvature uses per-atom energy, Huber delta `0.015`, and loss weight `1.0`.
- Force curvature uses component residuals, Huber delta `0.01`, and per-structure prefactor `0.1/(3N)`.
- Fixed eta warns on excessive condition number but never changes eta or adds hidden jitter.
- Fitted eta selects energy and force independently using validation Gaussian NLL; test data is never read during selection.
- Use `float64` for matrix accumulation, spectra, factorization, q, Alpha, and calibration statistics.
- Use canonical output name `non_conservative_force`; isolate any plural legacy alias in the checkpoint/readout adapter.
- Use schema `upet-llpr-artifact-v1` and formula version `upet-llpr-huber-readout-v1`.
- Large outputs remain under `Uncertainty_Quantification/LLPR/outputs/` and are not committed.
- Preserve the user's existing `.gitignore` modification; do not stage or rewrite it.
- Treat warnings as test failures unless `pyproject.toml` contains a narrowly targeted allowlist entry.
- Use TDD for every code task: failing test, observed failure, minimal implementation, passing test, commit.

---

## File Structure

### Create

```text
Uncertainty_Quantification/__init__.py
Uncertainty_Quantification/LLPR/__init__.py
Uncertainty_Quantification/LLPR/README.md
Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml
Uncertainty_Quantification/LLPR/configs/cpu_n20_fit.yaml
Uncertainty_Quantification/LLPR/configs/gpu_full_fixed.yaml
Uncertainty_Quantification/LLPR/configs/gpu_full_fit.yaml
Uncertainty_Quantification/LLPR/configs/import_legacy.yaml
Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml
Uncertainty_Quantification/LLPR/llpr/__init__.py
Uncertainty_Quantification/LLPR/llpr/__main__.py
Uncertainty_Quantification/LLPR/llpr/config.py
Uncertainty_Quantification/LLPR/llpr/artifacts.py
Uncertainty_Quantification/LLPR/llpr/checkpoint.py
Uncertainty_Quantification/LLPR/llpr/readout.py
Uncertainty_Quantification/LLPR/llpr/data.py
Uncertainty_Quantification/LLPR/llpr/observables.py
Uncertainty_Quantification/LLPR/llpr/curvature.py
Uncertainty_Quantification/LLPR/llpr/ridge.py
Uncertainty_Quantification/LLPR/llpr/calibration.py
Uncertainty_Quantification/LLPR/llpr/inference.py
Uncertainty_Quantification/LLPR/llpr/legacy.py
Uncertainty_Quantification/LLPR/llpr/plotting.py
Uncertainty_Quantification/LLPR/llpr/cli.py
Uncertainty_Quantification/LLPR/tests/conftest.py
Uncertainty_Quantification/LLPR/tests/test_config.py
Uncertainty_Quantification/LLPR/tests/test_artifacts.py
Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py
Uncertainty_Quantification/LLPR/tests/test_curvature.py
Uncertainty_Quantification/LLPR/tests/test_ridge_calibration.py
Uncertainty_Quantification/LLPR/tests/test_inference.py
Uncertainty_Quantification/LLPR/tests/test_legacy.py
Uncertainty_Quantification/LLPR/tests/test_plotting.py
Uncertainty_Quantification/LLPR/tests/test_cli.py
Uncertainty_Quantification/LLPR/tests/test_n20.py
Uncertainty_Quantification/LLPR/MIGRATION_REPORT.md
```

### Modify

```text
tox.ini
pyproject.toml
```

### Responsibilities

- `config.py`: typed YAML contracts and repository-root path resolution.
- `artifacts.py`: canonical JSON, hashes, stage identities, atomic writes, manifests, and verification.
- `checkpoint.py`: current checkpoint loading and training-metadata validation.
- `readout.py`: stable energy/force parameter layouts including bias.
- `data.py`: deterministic extxyz identities, labels, iteration, and metatomic system construction.
- `observables.py`: current-output forward pass and scalar/batched parameter Jacobians.
- `curvature.py`: Huber-weighted energy/force accumulation and resumable build.
- `ridge.py`: fixed/fitted eta candidates, spectra, Cholesky, and q.
- `calibration.py`: validation-only Alpha/NLL selection and resumable calibration.
- `inference.py`: test evaluation, NPZ shards, aggregation, and final details.
- `legacy.py`: immutable legacy snapshot, audit, canonical conversion, and idempotency.
- `plotting.py`: numeric diagnostics and atomic figure publication.
- `cli.py`: command parsing and orchestration only.

---

### Task 1: Package, Configuration Contracts, and Test Environment

**Files:**
- Create: `Uncertainty_Quantification/__init__.py`
- Create: `Uncertainty_Quantification/LLPR/__init__.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/__init__.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/config.py`
- Create: `Uncertainty_Quantification/LLPR/tests/conftest.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_config.py`
- Modify: `tox.ini`
- Modify: `pyproject.toml`

**Interfaces:**
- Produces: `REPO_ROOT: Path`
- Produces: `TargetEta`, `FileIdentityConfig`, `DataConfig`, `CurvatureConfig`, `RidgeFitConfig`, `RidgeConfig`, `RuntimeConfig`, `OutputConfig`, `LLPRConfig`
- Produces: `load_llpr_config(path: Path) -> LLPRConfig`
- Produces: `resolve_repo_path(raw: str | Path) -> Path`
- Consumes: YAML files relative to repository root.

- [ ] **Step 1: Add the isolated LLPR tox environment and lint scope**

Modify `tox.ini` so `lint_folders` includes
`"{toxinidir}/Uncertainty_Quantification/LLPR/"`, and add:

```ini
[testenv:llpr-tests]
description = Run UPET LLPR tests
deps =
    pytest
    pydantic>=2
    pyyaml
    scipy
    matplotlib
changedir = {toxinidir}
commands =
    pytest Uncertainty_Quantification/LLPR/tests {posargs}
```

Modify `pyproject.toml`:

```toml
[tool.ruff.lint.isort]
known-first-party = ["upet", "Uncertainty_Quantification"]

[tool.pytest.ini_options]
markers = [
    "llpr_n20: runs the real checkpoint on the 20-structure smoke dataset",
    "llpr_legacy: reads the audited external legacy result tree",
]
```

- [ ] **Step 2: Write failing configuration tests**

Create tests that prove repository-root resolution and fixed/fit validation:

```python
from pathlib import Path

import pytest
import yaml

from Uncertainty_Quantification.LLPR.llpr.config import (
    REPO_ROOT,
    load_llpr_config,
)


def test_relative_paths_resolve_from_repository_root(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "checkpoint": {
                    "path": "data/checkpoint/pet-omatpes-l-v0.1.0.ckpt",
                    "expected_sha256": "a" * 64,
                },
                "data": {
                    "build": "data/dataset/matpes_n20.extxyz",
                    "calibration": "data/dataset/matpes_n20.extxyz",
                    "test": "data/dataset/matpes_n20.extxyz",
                },
                "curvature": {
                    "energy_loss_weight": 1.0,
                    "force_loss_weight": 0.1,
                    "energy_huber_delta": 0.015,
                    "force_huber_delta": 0.01,
                },
                "calibration": {
                    "ridge": {
                        "mode": "fixed",
                        "max_condition_number": 1.0e10,
                        "eta": {"energy": 1.0e-6, "force": 1.0e-6},
                    }
                },
                "runtime": {"device": "cpu", "matrix_dtype": "float64"},
                "output": {"root": "Uncertainty_Quantification/LLPR/outputs", "experiment": "test"},
            }
        ),
        encoding="utf-8",
    )
    config = load_llpr_config(config_path)
    assert config.checkpoint.path == REPO_ROOT / "data/checkpoint/pet-omatpes-l-v0.1.0.ckpt"
    assert config.calibration.ridge.eta.energy == pytest.approx(1.0e-6)


def test_fit_mode_rejects_fixed_eta(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="fit mode must not define eta"):
        load_llpr_config(tmp_path / "invalid.yaml")
```

Use a helper in `conftest.py` to write the invalid complete YAML; do not let this
test fail because required unrelated fields are absent.

- [ ] **Step 3: Run the tests and observe the expected import failure**

Run:

```bash
conda activate upet_new
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_config.py -q
```

Expected: collection fails because `llpr.config` does not exist.

- [ ] **Step 4: Implement typed configuration**

Use Pydantic v2 with explicit forbidden extras:

```python
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


REPO_ROOT = Path(__file__).resolve().parents[3]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetEta(StrictModel):
    energy: float = Field(gt=0)
    force: float = Field(gt=0)


class RidgeFitConfig(StrictModel):
    candidate_multipliers: tuple[float, ...] = (1, 3, 10, 30, 100, 300, 1000)
    score: Literal["gaussian_nll"] = "gaussian_nll"


class RidgeConfig(StrictModel):
    mode: Literal["fixed", "fit"]
    max_condition_number: float = Field(default=1.0e10, gt=1)
    eta: TargetEta | None = None
    fit: RidgeFitConfig | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "RidgeConfig":
        if self.mode == "fixed" and (self.eta is None or self.fit is not None):
            raise ValueError("fixed mode requires eta and must not define fit")
        if self.mode == "fit" and (self.fit is None or self.eta is not None):
            raise ValueError("fit mode must define fit and must not define eta")
        return self
```

Define the remaining models with the exact fields in the design. In
`load_llpr_config`, parse YAML, normalize scalar fixed eta to both targets,
resolve all file paths with `resolve_repo_path`, and return a frozen model.

- [ ] **Step 5: Run configuration tests and lint**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_config.py -q
tox -e lint
```

Expected: configuration tests pass; lint reports no LLPR configuration errors.

- [ ] **Step 6: Commit Task 1**

```bash
git add tox.ini pyproject.toml Uncertainty_Quantification/__init__.py \
  Uncertainty_Quantification/LLPR/__init__.py \
  Uncertainty_Quantification/LLPR/llpr/__init__.py \
  Uncertainty_Quantification/LLPR/llpr/config.py \
  Uncertainty_Quantification/LLPR/tests/conftest.py \
  Uncertainty_Quantification/LLPR/tests/test_config.py
git commit -m "test: establish UPET LLPR configuration contracts"
```

---

### Task 2: Canonical Artifacts, Stage Identities, and Atomic Publication

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/artifacts.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_artifacts.py`

**Interfaces:**
- Consumes: `LLPRConfig`
- Produces: `RunPaths`
- Produces: `canonical_json(value: object) -> str`
- Produces: `stable_id(value: object, length: int = 16) -> str`
- Produces: `sha256_file(path: Path) -> str`
- Produces: `stage_identity(stage: str, payload: Mapping[str, object]) -> dict[str, object]`
- Produces: `atomic_json_dump(path: Path, value: Mapping[str, object]) -> None`
- Produces: `atomic_npz_save(path: Path, arrays: Mapping[str, np.ndarray]) -> None`
- Produces: `load_complete_manifest(path: Path, expected_identity: Mapping[str, object] | None = None) -> dict[str, object]`
- Produces: `verify_run(root: Path, level: Literal["metadata", "full"]) -> dict[str, object]`

- [ ] **Step 1: Write failing artifact tests**

```python
def test_stage_identity_is_order_independent() -> None:
    left = stable_id({"stage": "curvature", "payload": {"b": 2, "a": 1}})
    right = stable_id({"payload": {"a": 1, "b": 2}, "stage": "curvature"})
    assert left == right


def test_atomic_npz_does_not_publish_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "matrix.npz"
    monkeypatch.setattr(np, "savez_compressed", raising_savez)
    with pytest.raises(RuntimeError, match="injected write failure"):
        atomic_npz_save(destination, {"matrix": np.eye(2)})
    assert not destination.exists()


def test_incomplete_manifest_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"status": "in_progress"}', encoding="utf-8")
    with pytest.raises(ValueError, match="status.*complete"):
        load_complete_manifest(path)
```

- [ ] **Step 2: Run the tests and observe the expected import failure**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_artifacts.py -q
```

Expected: collection fails because `llpr.artifacts` does not exist.

- [ ] **Step 3: Implement canonical identities and atomic writers**

Use sorted compact JSON and same-directory temporary files:

```python
SCHEMA_VERSION = "upet-llpr-artifact-v1"
FORMULA_VERSION = "upet-llpr-huber-readout-v1"


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_id(value: object, length: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()
    return digest[:length]


def atomic_json_dump(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
```

`atomic_npz_save` must open the temporary path as a binary handle so NumPy
does not append an unexpected `.npz` suffix. `RunPaths` must expose root,
resolved-config, curvature, calibration, evaluation, plots, and legacy-raw
directories without creating them in its constructor.

- [ ] **Step 4: Implement stage manifests and verification**

Root identity contains input identities and formula/schema. Stage identities
contain only stage-relevant configuration:

```python
def stage_identity(stage: str, payload: Mapping[str, object]) -> dict[str, object]:
    identity_payload = {
        "schema_version": SCHEMA_VERSION,
        "formula_version": FORMULA_VERSION,
        "stage": stage,
        "payload": dict(payload),
    }
    return {**identity_payload, "identity": stable_id(identity_payload)}
```

`verify_run(metadata)` checks hashes, status, and dependency identities.
`verify_run(full)` additionally loads arrays and delegates numerical checks
registered by later tasks.

- [ ] **Step 5: Run artifact tests**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_artifacts.py -q
```

Expected: all artifact tests pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add Uncertainty_Quantification/LLPR/llpr/artifacts.py \
  Uncertainty_Quantification/LLPR/tests/test_artifacts.py
git commit -m "feat: add LLPR artifact identity and atomic writes"
```

---

### Task 3: Current Checkpoint, Readout Layout, and Deterministic Data

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/checkpoint.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/readout.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/data.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py`

**Interfaces:**
- Produces: `CheckpointIdentity`, `LoadedCheckpoint`
- Produces: `load_checkpoint(config: FileIdentityConfig, device: torch.device, dtype: torch.dtype) -> LoadedCheckpoint`
- Produces: `ParameterEntry`, `ReadoutLayout`
- Produces: `discover_readout_layout(model: torch.nn.Module) -> ReadoutLayout`
- Produces: `DatasetIdentity`, `LLPRSample`
- Produces: `dataset_identity(path: Path) -> DatasetIdentity`
- Produces: `iter_samples(path: Path) -> Iterator[LLPRSample]`
- Produces: `build_system(sample: LLPRSample, model: torch.nn.Module, device: torch.device, dtype: torch.dtype)`

- [ ] **Step 1: Write failing readout and dataset tests**

Use a fake model whose `last_layer_parameter_names` lists only weights:

```python
class FakeReadout(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.energy = torch.nn.Linear(2, 1)
        self.force = torch.nn.Linear(2, 3)
        self.last_layer_parameter_names = {
            "energy": ["energy.weight"],
            "non_conservative_forces": ["force.weight"],
        }


def test_layout_includes_weight_and_bias_in_stable_order() -> None:
    layout = discover_readout_layout(FakeReadout())
    assert [entry.name for entry in layout.energy.entries] == [
        "energy.weight",
        "energy.bias",
    ]
    assert [entry.name for entry in layout.force.entries] == [
        "force.weight",
        "force.bias",
    ]
    assert layout.energy.dimension == 3
    assert layout.force.dimension == 9
```

Write a two-structure extxyz fixture and assert deterministic indices, total
energy labels, force shape `(N, 3)`, SHA, and no shuffle.

- [ ] **Step 2: Run tests and observe missing modules**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py -q
```

Expected: collection fails because checkpoint/readout/data modules do not exist.

- [ ] **Step 3: Implement checkpoint loading and metadata validation**

```python
@dataclass(frozen=True)
class CheckpointIdentity:
    path: Path
    sha256: str
    model_class: str
    loss_reduction: str
    energy_loss_weight: float
    force_loss_weight: float
    energy_huber_delta: float
    force_huber_delta: float


def load_checkpoint(
    config: FileIdentityConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> LoadedCheckpoint:
    actual_sha = sha256_file(config.path)
    if actual_sha != config.expected_sha256:
        raise ValueError(f"checkpoint SHA mismatch: {actual_sha}")
    model = load_model(str(config.path)).eval().to(device=device, dtype=dtype)
    identity = inspect_training_metadata(config.path, model, actual_sha)
    validate_legacy_loss_contract(identity)
    return LoadedCheckpoint(model=model, identity=identity)
```

`inspect_training_metadata` must read the same checkpoint fields audited by
the old `check_model.py`. It must require reduction `mean`, weights `1.0/0.1`,
and deltas `0.015/0.01` for the formal configs.

- [ ] **Step 4: Implement readout discovery**

Resolve the force mapping from
`("non_conservative_force", "non_conservative_forces")`, but store canonical
target `non_conservative_force`. Include a bias immediately after each listed
weight, calculate offsets, and hash the serializable entry list. Reject
missing weights, duplicate names, absent biases, empty groups, and overlapping
energy/force parameters.

- [ ] **Step 5: Implement deterministic data and system construction**

```python
@dataclass(frozen=True)
class LLPRSample:
    index: int
    atoms: Atoms
    energy_reference_total: float
    force_reference: np.ndarray


def iter_samples(path: Path) -> Iterator[LLPRSample]:
    for index, atoms in enumerate(iread(path, index=":")):
        energy = require_energy_label(atoms, index)
        forces = require_force_label(atoms, index)
        if forces.shape != (len(atoms), 3):
            raise ValueError(
                f"test structure {index}: force shape {forces.shape} != {(len(atoms), 3)}"
            )
        yield LLPRSample(index, atoms, energy, forces)
```

Build metatomic systems with `systems_to_torch`, requested neighbor lists,
`_compute_ase_neighbors`, and `register_autograd_neighbors`, preserving the
working old implementation while isolating the private API in this file.

- [ ] **Step 6: Add the real checkpoint layout smoke assertion**

Mark the test `@pytest.mark.llpr_n20` and require
`UPET_RUN_LLPR_N20=1`. Assert checkpoint SHA and dimensions:

```python
assert layout.energy.dimension == 1026
assert layout.force.dimension == 3078
assert layout.total_dimension == 4104
```

- [ ] **Step 7: Run fast tests**

Run:

```bash
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py \
  -m "not llpr_n20" -q
```

Expected: all fast tests pass.

- [ ] **Step 8: Commit Task 3**

```bash
git add Uncertainty_Quantification/LLPR/llpr/checkpoint.py \
  Uncertainty_Quantification/LLPR/llpr/readout.py \
  Uncertainty_Quantification/LLPR/llpr/data.py \
  Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py
git commit -m "feat: load UPET LLPR checkpoint and readout layout"
```

---

### Task 4: Jacobians and Resumable Huber Curvature

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/observables.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/curvature.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_curvature.py`

**Interfaces:**
- Consumes: `LoadedCheckpoint`, `ReadoutLayout`, `LLPRSample`, `CurvatureConfig`, `RunPaths`
- Produces: `StructureJacobians`
- Produces: `compute_structure_jacobians(model, system, layout, backend, force_component_chunk_size) -> StructureJacobians`
- Produces: `huber_curvature(residual: torch.Tensor, delta: float) -> torch.Tensor`
- Produces: `CurvatureAccumulator`
- Produces: `run_build(config: LLPRConfig) -> Path`

- [ ] **Step 1: Write failing hand-calculated curvature tests**

Use explicit gradients and residuals:

```python
def test_huber_curvature_and_legacy_scaling() -> None:
    accumulator = CurvatureAccumulator.zeros(energy_dim=2, force_dim=2)
    accumulator.add_structure(
        energy_gradient=torch.tensor([1.0, 2.0]),
        energy_residual=torch.tensor(0.01),
        force_gradients=torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
        force_residuals=torch.tensor([0.001, 0.02, -0.001]),
        num_atoms=1,
        energy_weight=1.0,
        force_weight=0.1,
        energy_delta=0.015,
        force_delta=0.01,
    )
    torch.testing.assert_close(
        accumulator.energy,
        torch.tensor([[1.0, 2.0], [2.0, 4.0]], dtype=torch.float64),
    )
    expected_force = (0.1 / 3.0) * torch.tensor(
        [[2.0, 1.0], [1.0, 1.0]], dtype=torch.float64
    )
    torch.testing.assert_close(accumulator.force, expected_force)
```

Add tests that energy is divided by `N`, force weight is applied once, all
outliers yield zero Huber curvature, and completed structures are not
double-counted after resume.

- [ ] **Step 2: Write failing scalar/batched Jacobian parity test**

Use a toy module with independent energy and force heads and assert:

```python
torch.testing.assert_close(
    scalar.force_jacobian,
    batched.force_jacobian,
    rtol=1.0e-10,
    atol=1.0e-12,
)
```

- [ ] **Step 3: Run tests and observe missing implementations**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_curvature.py -q
```

Expected: collection fails because observables/curvature modules do not exist.

- [ ] **Step 4: Implement current-output forward and Jacobians**

Request only canonical outputs:

```python
outputs = model(
    [system],
    {
        "energy": ModelOutput(unit="eV", per_atom=False),
        "non_conservative_force": ModelOutput(unit="eV/A", per_atom=True),
    },
)
```

Flatten gradients with the exact `ReadoutLayout`. Implement:

- scalar reference: one `torch.autograd.grad` call per force component;
- batched backend: identity grad outputs in chunks with
  `is_grads_batched=True`;
- finite checks with structure/component context;
- return total energy, per-atom energy, flattened force, energy Jacobian, and
  force Jacobian.

- [ ] **Step 5: Implement curvature accumulation**

```python
def huber_curvature(residual: Tensor, delta: float) -> Tensor:
    return (residual.abs() <= delta).to(dtype=torch.float64)


def add_structure(self, *, energy_gradient, energy_residual, force_gradients,
                  force_residuals, num_atoms, energy_weight, force_weight,
                  energy_delta, force_delta) -> None:
    ce = huber_curvature(energy_residual, energy_delta)
    cf = huber_curvature(force_residuals, force_delta)
    self.energy.add_(energy_weight * ce * torch.outer(energy_gradient, energy_gradient))
    weighted_force = force_gradients * cf[:, None]
    self.force.add_(
        (force_weight / (3 * num_atoms)) * weighted_force.T @ force_gradients
    )
```

Convert all contributions to CPU float64 before accumulation.

- [ ] **Step 6: Implement resumable build**

After each complete structure, update counters. Every configured interval,
atomically save `progress.npz` with energy/force blocks, next structure index,
counts, and curvature identity. On resume, require identity equality. At
completion, symmetrize blocks, write canonical NPZ/manifests/diagnostics,
mark `complete`, and retain no ambiguous `H + eta^2 I` diagnostic.

- [ ] **Step 7: Run curvature tests**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_curvature.py -q
```

Expected: all hand-calculated, parity, and resume tests pass.

- [ ] **Step 8: Commit Task 4**

```bash
git add Uncertainty_Quantification/LLPR/llpr/observables.py \
  Uncertainty_Quantification/LLPR/llpr/curvature.py \
  Uncertainty_Quantification/LLPR/tests/test_curvature.py
git commit -m "feat: compute resumable UPET LLPR curvature"
```

---

### Task 5: Fixed and Validation-Fitted Eta Calibration

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/ridge.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/calibration.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_ridge_calibration.py`

**Interfaces:**
- Consumes: complete curvature, validation samples/Jacobians, `RidgeConfig`
- Produces: `Spectrum`, `RidgeCandidate`, `CalibrationRecord`
- Produces: `prepare_spectrum(matrix: torch.Tensor) -> Spectrum`
- Produces: `fixed_candidate(target: str, eta: float, spectrum: Spectrum, max_condition_number: float) -> RidgeCandidate`
- Produces: `fit_candidates(target: str, spectrum: Spectrum, multipliers: Sequence[float], max_condition_number: float) -> tuple[RidgeCandidate, ...]`
- Produces: `quadratic_forms(cholesky: torch.Tensor, gradients: torch.Tensor) -> torch.Tensor`
- Produces: `moment_alpha(residuals: torch.Tensor, q: torch.Tensor) -> float`
- Produces: `gaussian_nll(residuals: torch.Tensor, variance: torch.Tensor) -> float`
- Produces: `select_candidate(records: Sequence[CalibrationRecord]) -> CalibrationRecord`
- Produces: `run_calibrate(config: LLPRConfig) -> Path`

- [ ] **Step 1: Write failing fixed-eta tests**

```python
def test_fixed_eta_matches_direct_inverse() -> None:
    h = torch.tensor([[2.0, 0.5], [0.5, 1.0]], dtype=torch.float64)
    gradients = torch.tensor([[1.0, 2.0], [0.5, -1.0]], dtype=torch.float64)
    eta = 1.0e-6
    cholesky = torch.linalg.cholesky(h + eta * torch.eye(2))
    q = quadratic_forms(cholesky, gradients)
    expected = torch.einsum(
        "bi,ij,bj->b", gradients, torch.linalg.inv(h + eta * torch.eye(2)), gradients
    )
    torch.testing.assert_close(q, expected)
```

Assert fixed mode keeps `1e-6` even when condition number exceeds `1e10`, and
records `condition_warning=True`.

- [ ] **Step 2: Write failing fitted-eta tests**

Test the exact floor:

```python
eta_condition = max(
    0.0,
    (mu_max - max_condition_number * mu_min) / (max_condition_number - 1.0),
)
```

Assert candidate multipliers are deterministic, energy and force selection
are independent, Alpha equals `sqrt(mean(residual**2/q))`, minimum validation
NLL wins, and exact NLL ties select smaller eta.

Create a `PoisonTestDataset` whose iterator raises immediately and assert
`run_calibrate` never touches it.

- [ ] **Step 3: Run tests and observe missing implementations**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_ridge_calibration.py -q
```

Expected: collection fails because ridge/calibration modules do not exist.

- [ ] **Step 4: Implement spectrum and eta candidates**

```python
spectral_eps = torch.finfo(torch.float64).eps * max(1.0, abs(mu_max))
eta_pd = max(0.0, -mu_min) + spectral_eps
eta_cond = max(
    0.0,
    (mu_max - max_condition_number * mu_min) / (max_condition_number - 1.0),
)
eta_floor = np.nextafter(max(eta_pd, eta_cond), np.inf)
etas = tuple(sorted({float(eta_floor * multiplier) for multiplier in multipliers}))
```

Fixed candidates skip condition rejection but record it. Fit candidates reject
non-finite eta, Cholesky failure, non-positive q, and condition number above
the configured maximum.

- [ ] **Step 5: Implement validation calibration**

For each target and candidate, stream validation structures, calculate q, and
accumulate `sum(residual**2/q)` plus row counts. Finalize Alpha, variance,
Gaussian NLL, coverage at 1/2/3 sigma, and standardized-residual summaries.
Store all records in `candidates.json`; store separate selected energy/force
records in `summary.json`.

Persist progress only after a complete structure. Identity must include
curvature identity, validation SHA, ridge mode, candidates, and formula
version, but must exclude test data.

- [ ] **Step 6: Run ridge/calibration tests**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_ridge_calibration.py -q
```

Expected: fixed and fit tests pass with no test-data access.

- [ ] **Step 7: Commit Task 5**

```bash
git add Uncertainty_Quantification/LLPR/llpr/ridge.py \
  Uncertainty_Quantification/LLPR/llpr/calibration.py \
  Uncertainty_Quantification/LLPR/tests/test_ridge_calibration.py
git commit -m "feat: calibrate fixed and fitted LLPR ridge"
```

---

### Task 6: Evaluation, Aggregation, and NPZ Shards

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/inference.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_inference.py`

**Interfaces:**
- Consumes: complete curvature/calibration and test samples/Jacobians.
- Produces: `EvaluationBuffers`
- Produces: `evaluate_structure(sample: LLPRSample, jacobians: StructureJacobians, calibration: Mapping[str, CalibrationRecord], solvers: Mapping[str, torch.Tensor]) -> dict[str, np.ndarray]`
- Produces: `summarize_evaluation(details: Mapping[str, np.ndarray]) -> dict[str, object]`
- Produces: `run_evaluate(config: LLPRConfig) -> Path`

- [ ] **Step 1: Write failing structure-level inference tests**

```python
def test_evaluate_structure_uses_per_atom_energy_and_component_forces() -> None:
    result = evaluate_structure(sample, jacobians, calibration, solvers)
    np.testing.assert_allclose(
        result["energy_residual"],
        result["energy_pred_per_atom"] - result["energy_true_per_atom"],
    )
    np.testing.assert_allclose(
        result["energy_calibrated_var"],
        calibration["energy"].alpha_sq * result["energy_raw_var"],
    )
    np.testing.assert_allclose(
        result["force_calibrated_var_component"],
        calibration["force"].alpha_sq * result["force_raw_var_component"],
    )
    assert result["force_offsets"].tolist() == [0, 3 * len(sample.atoms)]
```

Add tests for atom-major x/y/z indices, atom variance means, structure force
RMSE, positive variance, inverse variance, and `N**2` total-energy variance.

- [ ] **Step 2: Write failing resume/shard tests**

Interrupt after one structure, resume, and compare every merged array with an
uninterrupted run. Corrupt a shard row count and assert resume fails.

- [ ] **Step 3: Run tests and observe missing inference**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_inference.py -q
```

Expected: collection fails because `llpr.inference` does not exist.

- [ ] **Step 4: Implement structure evaluation**

Use Cholesky solves from the selected target-specific eta. Produce the
canonical old-compatible fields:

```text
structure_index, num_atoms,
energy_pred_total, energy_true_total,
energy_pred_per_atom, energy_true_per_atom, energy_residual,
energy_raw_var, energy_calibrated_var, energy_calibrated_std,
energy_inverse_variance, energy_raw_var_total_derived,
energy_calibrated_var_total_derived,
force_offsets, force_structure_index,
force_component_index_within_structure, force_atom_index,
force_cartesian_index, force_pred, force_true, force_residual,
force_raw_var_component, force_calibrated_var_component,
force_calibrated_std_component, force_inverse_variance_component,
force_raw_var_atom_mean, force_calibrated_var_atom_mean,
force_calibrated_std_atom_rms,
force_raw_var_component_mean_structure,
force_calibrated_var_component_mean_structure,
force_calibrated_std_component_rms_structure
```

Reject any non-finite or non-positive q/variance with full sample context.

- [ ] **Step 5: Implement transactional shards and finalization**

Write one compressed shard per configured complete-structure batch. Progress
records the next structure index, shard hashes, and row counts. Finalization
verifies all shards, concatenates arrays, recomputes summary statistics,
writes `details.npz`, `summary.json`, `preview.json`, and a complete evaluation
manifest, then keeps or removes shards according to explicit config.

- [ ] **Step 6: Run inference tests**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_inference.py -q
```

Expected: all formula, indexing, aggregation, and resume tests pass.

- [ ] **Step 7: Commit Task 6**

```bash
git add Uncertainty_Quantification/LLPR/llpr/inference.py \
  Uncertainty_Quantification/LLPR/tests/test_inference.py
git commit -m "feat: evaluate and aggregate LLPR uncertainty"
```

---

### Task 7: Audited Legacy Importer

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/legacy.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_legacy.py`
- Create: `Uncertainty_Quantification/LLPR/configs/import_legacy.yaml`

**Interfaces:**
- Consumes: old root `/home/lilong/code/UQ/upet/UQ_LLPR/matpes_r2/Hef`
- Consumes: new checkpoint and dataset identities.
- Produces: `LegacyImportConfig`, `InventoryEntry`, `LegacyClassification`
- Produces: `build_source_inventory(source: Path) -> tuple[InventoryEntry, ...]`
- Produces: `validate_legacy_matrices(paths: Mapping[str, Path]) -> dict[str, object]`
- Produces: `validate_legacy_calibration(summary: Mapping[str, object]) -> dict[str, object]`
- Produces: `validate_legacy_evaluation(details: Mapping[str, np.ndarray], summary: Mapping[str, object]) -> dict[str, object]`
- Produces: `import_legacy(config_path: Path) -> Path`

- [ ] **Step 1: Write a synthetic legacy fixture**

Build a tiny old-format tree with 2+1 parameter blocks, one validation record,
one test structure, and raw files classified as authoritative, smoke,
incomplete, and orphan. Record hashes before import.

- [ ] **Step 2: Write failing importer tests**

```python
def test_import_is_model_free_and_idempotent(
    legacy_tree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(
        sys.modules,
        "metatrain.utils.io",
        ModuleTypeThatRaisesOnAttributeAccess(),
    )
    first = import_legacy(write_import_config(legacy_tree))
    second = import_legacy(write_import_config(legacy_tree))
    assert first == second
    assert json.loads((first / "manifest.json").read_text())["origin"] == "legacy_import"


def test_tampered_joint_matrix_fails(legacy_tree: Path) -> None:
    tamper_joint_matrix(legacy_tree)
    with pytest.raises(ValueError, match="H_EF.*H_E.*H_F"):
        import_legacy(write_import_config(legacy_tree))
```

Add failures for changed checkpoint/data SHA, asymmetry, non-finite values,
wrong offsets, changed Alpha, missing formal files, and target identity
collision.

- [ ] **Step 3: Run tests and observe missing legacy importer**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_legacy.py -q
```

Expected: collection fails because `llpr.legacy` does not exist.

- [ ] **Step 4: Implement inventory and classifications**

Use `shutil.copy2` into a staging `legacy_raw`. Inventory rows include source
absolute path, destination relative path, bytes, SHA-256, classification, and
formal-source boolean. Preserve every old script/config/log/result/test file.

Classify:

- full H/Alpha/test/reference plotting as `authoritative`;
- root two-structure LLPR as `legacy_smoke`;
- linear-fit summary with missing PNG as `incomplete`;
- combined plot without current generator as `orphan`.

- [ ] **Step 5: Implement strict numerical conversion**

Load old full matrices, verify finite/symmetric/block-zero/additive invariants,
and extract:

```python
energy_block = old_h_e[:1026, :1026]
force_block = old_h_f[1026:4104, 1026:4104]
```

Verify exact Alpha, eta, counts, offsets, residual identities, calibrated
variance, standard deviation, inverse variance, and all summary counts. Write
canonical curvature/calibration/evaluation stage manifests with
`origin="legacy_import"` and `legacy_fixed_ridge=True`.

Recompute diagnostics with `H + eta I`; preserve the old `H + eta**2 I`
diagnostic only inside raw provenance notes.

- [ ] **Step 6: Implement staging publication and idempotency**

Import into a same-filesystem temporary directory. Run full verification
before rename. If the completed destination has the same root/stage
identities, verify and return it unchanged. If identities differ, fail without
overwriting.

- [ ] **Step 7: Run importer tests**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_legacy.py -q
```

Expected: all synthetic audit, tamper, model-free, and idempotency tests pass.

- [ ] **Step 8: Commit Task 7**

```bash
git add Uncertainty_Quantification/LLPR/llpr/legacy.py \
  Uncertainty_Quantification/LLPR/tests/test_legacy.py \
  Uncertainty_Quantification/LLPR/configs/import_legacy.yaml
git commit -m "feat: import audited legacy LLPR artifacts"
```

---

### Task 8: Canonical Plotting and Diagnostics

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/plotting.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_plotting.py`
- Create: `Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml`

**Interfaces:**
- Consumes: complete canonical evaluation.
- Produces: `PlotConfig`, `FilteredPairs`, `PanelAnalysis`, `ReliabilityBin`
- Produces: `filter_log_pairs(uncertainty: np.ndarray, absolute_error: np.ndarray) -> FilteredPairs`
- Produces: `analyze_panel(uncertainty: np.ndarray, absolute_error: np.ndarray) -> PanelAnalysis`
- Produces: `reliability_bins(std: np.ndarray, residual: np.ndarray, bin_count: int) -> tuple[ReliabilityBin, ...]`
- Produces: `standardized_residual_cdf(std: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray]`
- Produces: `run_plot(config: PlotConfig) -> Path`

- [ ] **Step 1: Write failing numeric plotting tests**

```python
def test_panel_statistics_are_deterministic() -> None:
    uncertainty = np.array([0.1, 0.2, 0.4, 0.8])
    error = np.array([0.08, 0.3, 0.2, 1.0])
    first = analyze_panel(uncertainty, error)
    second = analyze_panel(uncertainty, error)
    assert first == second
    assert first.count == 4
    assert np.isfinite(first.pearson_log10)
    assert np.isfinite(first.spearman_log10)
```

Test non-positive filtering, deterministic sampling, shared log limits,
coverage, standardized CDF, and no test-time Alpha rescaling.

- [ ] **Step 2: Write failing transactional publication test**

Inject a failure after the first image render. Assert the prior complete plot
directory and manifest remain byte-identical and no partial new images are
published.

- [ ] **Step 3: Run tests and observe missing plotting**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_plotting.py -q
```

Expected: collection fails because `llpr.plotting` does not exist.

- [ ] **Step 4: Implement numeric analysis**

Use NumPy/SciPy, not pandas. Compute:

- Pearson and Spearman on valid `log10` pairs;
- deterministic sampled scatter/density;
- 1/2/3 sigma coverage;
- reliability bins;
- standardized residual empirical CDF;
- Gaussian NLL and quantile summaries.

Energy uses per-atom residual/std. Force uses component residual/std.

- [ ] **Step 5: Implement atomic rendering**

Render energy, force-component, reliability, and standardized-residual figures
to a staging directory. Write PNG, PDF, statistics CSV, and manifest hashes.
Verify every declared file before atomically promoting the directory.

- [ ] **Step 6: Run plotting tests**

Run:

```bash
MPLBACKEND=Agg tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_plotting.py -q
```

Expected: numeric and transactional plotting tests pass.

- [ ] **Step 7: Commit Task 8**

```bash
git add Uncertainty_Quantification/LLPR/llpr/plotting.py \
  Uncertainty_Quantification/LLPR/tests/test_plotting.py \
  Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml
git commit -m "feat: plot canonical LLPR diagnostics"
```

---

### Task 9: CLI, Runnable Configurations, and Documentation

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/cli.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/__main__.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_cli.py`
- Create: `Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml`
- Create: `Uncertainty_Quantification/LLPR/configs/cpu_n20_fit.yaml`
- Create: `Uncertainty_Quantification/LLPR/configs/gpu_full_fixed.yaml`
- Create: `Uncertainty_Quantification/LLPR/configs/gpu_full_fit.yaml`
- Create: `Uncertainty_Quantification/LLPR/README.md`

**Interfaces:**
- Consumes: all stage runners.
- Produces: `build_parser() -> argparse.ArgumentParser`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Write failing CLI dispatch tests**

```python
@pytest.mark.parametrize(
    ("command", "runner_name"),
    [
        ("build", "run_build"),
        ("calibrate", "run_calibrate"),
        ("evaluate", "run_evaluate"),
        ("import-legacy", "import_legacy"),
        ("verify", "verify_run"),
        ("plot", "run_plot"),
    ],
)
def test_cli_dispatches_once(
    command: str, runner_name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[Path] = []
    monkeypatch.setattr(cli, runner_name, lambda path: calls.append(Path(path)))
    assert cli.main([command, "--config", "config.yaml"]) == 0
    assert calls == [Path("config.yaml")]
```

Test `run` calls build, calibrate, evaluate in order and stops immediately on
failure.

- [ ] **Step 2: Run tests and observe missing CLI**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_cli.py -q
```

Expected: collection fails because `llpr.cli` does not exist.

- [ ] **Step 3: Implement thin CLI**

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run":
        config = load_llpr_config(args.config)
        run_build(config)
        run_calibrate(config)
        run_evaluate(config)
        return 0
    return dispatch_single_command(args)
```

Configure logging and nonzero exit codes, but keep all numerical logic in the
stage modules.

- [ ] **Step 4: Write exact runnable configs**

All configs bind checkpoint SHA
`879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004`.

`cpu_n20_fixed.yaml` uses n20 for all splits, CPU, scalar Jacobians, fixed
eta `1e-6`, and experiment `n20_smoke`.

`cpu_n20_fit.yaml` uses the same curvature identity, CPU, scalar Jacobians,
condition maximum `1e10`, multipliers `[1,3,10,30,100,300,1000]`, and
Gaussian NLL.

`gpu_full_fixed.yaml` binds full train/validation/test hashes and fixed eta,
but its header comment states that this migration must not execute it.

`gpu_full_fit.yaml` binds the same full splits and fitted eta, but its header
comment states that it is a future recomputation config with no migrated
formal output.

- [ ] **Step 5: Write README**

Document:

- environment activation;
- command sequence;
- fixed versus fitted eta;
- exact legacy formulas;
- output tree and stage identities;
- resume behavior;
- n20 functional-only status;
- prohibition on treating imported fixed eta as numerically robust;
- explicit statement that import/verify/plot do not run the model.

- [ ] **Step 6: Run CLI tests and config parsing**

Run:

```bash
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_cli.py \
  Uncertainty_Quantification/LLPR/tests/test_config.py -q
python -m Uncertainty_Quantification.LLPR.llpr --help
```

Expected: tests pass and help lists all seven commands.

- [ ] **Step 7: Commit Task 9**

```bash
git add Uncertainty_Quantification/LLPR/llpr/cli.py \
  Uncertainty_Quantification/LLPR/llpr/__main__.py \
  Uncertainty_Quantification/LLPR/tests/test_cli.py \
  Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml \
  Uncertainty_Quantification/LLPR/configs/cpu_n20_fit.yaml \
  Uncertainty_Quantification/LLPR/configs/gpu_full_fixed.yaml \
  Uncertainty_Quantification/LLPR/configs/gpu_full_fit.yaml \
  Uncertainty_Quantification/LLPR/README.md
git commit -m "docs: expose runnable UPET LLPR workflow"
```

---

### Task 10: n20 Full Path, Formal Legacy Migration, and Final Acceptance

**Files:**
- Create: `Uncertainty_Quantification/LLPR/tests/test_n20.py`
- Create: `Uncertainty_Quantification/LLPR/MIGRATION_REPORT.md`
- Modify: any LLPR file only when a failing acceptance test identifies a defect.

**Interfaces:**
- Consumes: all completed tasks and audited source files.
- Produces: completed n20 fixed/fit outputs.
- Produces: completed imported legacy output and plots.
- Produces: a tracked migration report with commands, identities, counts, and verification evidence.

- [ ] **Step 1: Write the gated n20 acceptance test**

```python
@pytest.mark.llpr_n20
def test_n20_fixed_and_fit_full_paths() -> None:
    fixed = load_llpr_config(CONFIGS / "cpu_n20_fixed.yaml")
    fitted = load_llpr_config(CONFIGS / "cpu_n20_fit.yaml")
    fixed_root = run_all(fixed)
    fitted_root = run_all(fitted)
    assert fixed_root == fitted_root
    assert load_curvature_identity(fixed_root) == load_curvature_identity(fitted_root)
    assert load_selected_eta(fixed_root, "fixed") == {
        "energy": 1.0e-6,
        "force": 1.0e-6,
    }
    assert_all_positive_finite_variances(fixed_root, "fixed")
    assert_all_positive_finite_variances(fitted_root, "fit")
    verify_run(fixed_root, level="full")
```

Gate collection unless `UPET_RUN_LLPR_N20=1`.

- [ ] **Step 2: Run the complete fast suite**

Run:

```bash
conda activate upet_new
tox -e llpr-tests -- -m "not llpr_n20 and not llpr_legacy" -q
```

Expected: all fast tests pass with zero warnings.

- [ ] **Step 3: Run n20 fixed and fitted full paths**

Run:

```bash
export UPET_RUN_LLPR_N20=1
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_n20.py \
  Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py \
  -m llpr_n20 -v
```

Expected:

- checkpoint/readout dimensions 1026/3078/4104;
- one shared complete curvature;
- fixed eta exactly `1e-6/1e-6`;
- distinct fitted energy/force candidate records;
- finite positive q/variance;
- no skipped structures/components;
- fixed and fitted plots complete;
- full verification passes.

- [ ] **Step 4: Run the formal legacy import**

Run:

```bash
python -m Uncertainty_Quantification.LLPR.llpr import-legacy \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml
```

Expected:

- no model-loading log entries;
- raw inventory covers all old scripts/configs/logs/results/tests;
- authoritative, smoke, incomplete, and orphan classifications are present;
- canonical output origin is `legacy_import`;
- fixed eta and Alpha exactly match audited values.

- [ ] **Step 5: Run full verification twice**

Run:

```bash
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml \
  --level full
python -m Uncertainty_Quantification.LLPR.llpr import-legacy \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml \
  --level full
```

Expected: both verifications pass; the second import reports an identity-matched
no-op; formal result file hashes and mtimes remain unchanged.

- [ ] **Step 6: Plot imported formal results**

Run:

```bash
MPLBACKEND=Agg python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml
```

Expected:

- energy and force-component plots, reliability, standardized residuals,
  statistics CSV, and plotting manifest are complete;
- imported reference correlations and fit statistics match the audited old
  numeric values within `rtol=1e-12`, `atol=1e-14`;
- image bytes are not required to match old Matplotlib output.

- [ ] **Step 7: Prove no full recomputation occurred**

Inspect manifests and logs:

```bash
rg -n '"origin": "recomputed"' \
  Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
rg -n 'Processing structure|model forward|build_system' \
  Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
```

Expected: the formal migrated tree contains no recomputed origin and no model
forward/build logs. Recomputed origin is allowed only under the n20 experiment.

- [ ] **Step 8: Run lint and full LLPR verification**

Run:

```bash
tox -e lint
tox -e llpr-tests -- -m "not llpr_n20 and not llpr_legacy" -q
```

Expected: both commands exit zero.

- [ ] **Step 9: Write the migration report**

Record in `MIGRATION_REPORT.md`:

- source and destination absolute paths;
- source commit and inventory hash;
- checkpoint/train/validation/test hashes;
- canonical root and stage identities;
- exact fixed eta and Alpha;
- matrix dimensions and condition warnings;
- formal structure/atom/component counts;
- n20 fixed/fit selected eta and test counts;
- exact commands and exit results;
- statement that full model computation was not run;
- orphan/incomplete classifications;
- final Git commit.

- [ ] **Step 10: Commit Task 10**

Do not stage `outputs/` or the existing user `.gitignore` change:

```bash
git add Uncertainty_Quantification/LLPR/tests/test_n20.py \
  Uncertainty_Quantification/LLPR/MIGRATION_REPORT.md
git commit -m "test: verify UPET LLPR migration end to end"
```

- [ ] **Step 11: Final repository-state audit**

Run:

```bash
git status --short
git log --oneline --max-count=12
```

Expected: only the pre-existing user `.gitignore` modification remains
uncommitted; all LLPR source, config, tests, README, and migration report are
committed; large outputs are ignored.

---

## Plan Self-Review Checklist

- Every design section maps to at least one task.
- Fixed and fitted eta have separate tests and share curvature identity.
- Test leakage is prevented by an explicit poison-test dataset.
- Legacy import has a model-free test and a full-result acceptance command.
- Old raw files are copied unchanged and hashed before conversion.
- Large formal outputs are not staged.
- n20 is the only real model-based calculation in this migration.
- Stage identity boundaries permit fixed/fit curvature reuse.
- Recovery is tested for build and evaluation; calibration progress identity is tested with fitted candidates.
- Plotting is transactional and does not recalibrate.
- No task modifies the old repository or the user's `.gitignore` change.
