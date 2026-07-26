# UPET LLPR Legacy Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a current-UPET LLPR `build -> calibrate -> evaluate -> plot` pipeline with fixed and validation-fitted eta modes, then migrate the audited legacy fixed-eta results without rerunning the full datasets.

**Architecture:** A modular calculation package writes one versioned artifact protocol. A model-free legacy adapter validates and converts the old NPZ/JSON/log tree into that protocol, so plotting and verification consume the same interfaces for migrated and recomputed results. Stage-specific identity manifests allow fixed and fitted calibration runs to share curvature safely.

**Tech Stack:** Python 3.11, PyTorch, NumPy, ASE, Metatomic/Metatrain, Pydantic v2, PyYAML, SciPy, Matplotlib, pytest, tox.

## Global Constraints

- Work in `/home/lilong/code/UQ/upet_new` after `conda activate upet_new`.
- Do not modify `/home/lilong/code/UQ/upet/UQ_LLPR`.
- Do not run full train/validation/test model calculations in this implementation.
- The only permitted model-based end-to-end calculation is `data/dataset/matpes_n20.extxyz`.
- Preserve the migrated formal result as `legacy_fixed_ridge` with `eta_energy=eta_force=1.0e-6`.
- Fixed mode must warn about excessive condition numbers but must not change eta or add hidden jitter.
- Fit mode must choose energy and force eta independently using validation Gaussian NLL; test data must not participate.
- Preserve energy-per-atom, force-component, Huber, `w_E=1.0`, and `w_F/(3N)=0.1/(3N)` semantics exactly.
- Use `float64` for curvature accumulation, spectra, Cholesky, q, Alpha, and statistics.
- Use current singular `non_conservative_force` output names at the API boundary; preserve actual checkpoint parameter names in the readout manifest.
- Use `upet-llpr-artifact-v1` as the artifact schema and `upet-llpr-huber-readout-v1` as the formula version.
- Default invalid-sample policy is error; do not skip missing, non-finite, or malformed samples.
- Do not introduce pandas; use NumPy, csv, and json for tabular artifacts.
- Keep large `outputs/` artifacts out of Git.
- Preserve the user's existing `.gitignore` change; do not stage or amend it unless separately requested.
- Every task follows red-green TDD and ends in a focused commit.

---

## File Structure

Create or modify the following files:

```text
Uncertainty_Quantification/
├── __init__.py
└── LLPR/
    ├── __init__.py
    ├── README.md
    ├── MIGRATION_REPORT.md                 # created after formal import
    ├── configs/
    │   ├── cpu_n20_fixed.yaml
    │   ├── cpu_n20_fit.yaml
    │   ├── gpu_full_fixed.yaml
    │   ├── gpu_full_fit.yaml
    │   ├── import_legacy.yaml
    │   └── plot_legacy.yaml
    ├── llpr/
    │   ├── __init__.py
    │   ├── __main__.py
    │   ├── config.py                       # validated YAML contracts
    │   ├── artifacts.py                    # hashes, identities, atomic I/O
    │   ├── checkpoint.py                   # current UPET checkpoint adapter
    │   ├── readout.py                      # stable energy/force layouts
    │   ├── data.py                         # extxyz identity and iteration
    │   ├── observables.py                  # forward and Jacobians
    │   ├── curvature.py                    # Huber H_E/H_F accumulation
    │   ├── ridge.py                        # spectra, fixed/fit eta candidates
    │   ├── calibration.py                  # validation Alpha and NLL selection
    │   ├── inference.py                    # test LLPR and aggregation
    │   ├── legacy.py                       # model-free legacy import
    │   ├── plotting.py                     # audited plots and diagnostics
    │   └── cli.py                          # thin command dispatcher
    └── tests/
        ├── conftest.py
        ├── test_config.py
        ├── test_artifacts.py
        ├── test_checkpoint_readout_data.py
        ├── test_observables_curvature.py
        ├── test_ridge_calibration.py
        ├── test_inference.py
        ├── test_legacy.py
        ├── test_plotting.py
        ├── test_cli.py
        └── test_n20.py

tox.ini                                      # LLPR test env and lint paths
pyproject.toml                               # first-party import and pytest markers
```

Focused responsibilities are fixed by the approved design. Do not merge calculation,
artifact, legacy, or plotting responsibilities into a single script.

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
- Produces: Pydantic models `FileIdentityConfig`, `DataConfig`,
  `CurvatureConfig`, `TargetEta`, `RidgeFitConfig`, `RidgeConfig`,
  `RuntimeConfig`, `OutputConfig`, and `LLPRConfig`
- Produces: `load_llpr_config(path: Path) -> LLPRConfig`
- Produces: `resolved_config_payload(config: LLPRConfig) -> dict[str, Any]`
- Consumes: no earlier task interfaces

- [ ] **Step 1: Add failing config and environment tests**

Create tests that prove paths resolve relative to the repository root, a scalar eta
expands to both targets, fixed and fit fields are mutually valid, and malformed
configs fail:

```python
def test_scalar_fixed_eta_expands_to_both_targets(tmp_path: Path) -> None:
    path = write_yaml(tmp_path, base_config(ridge={"mode": "fixed", "eta": 1e-6}))
    config = load_llpr_config(path)
    assert config.calibration.ridge.fixed_eta() == TargetEta(
        energy=1e-6,
        force=1e-6,
    )


def test_fit_rejects_fixed_eta(tmp_path: Path) -> None:
    payload = base_config(
        ridge={
            "mode": "fit",
            "eta": 1e-6,
            "max_condition_number": 1e10,
            "fit": {"candidate_multipliers": [1, 10], "score": "gaussian_nll"},
        }
    )
    with pytest.raises(ValidationError, match="eta is only valid in fixed mode"):
        load_llpr_config(write_yaml(tmp_path, payload))
```

- [ ] **Step 2: Run the tests and verify the import failure**

Run:

```bash
conda activate upet_new
python -m pytest Uncertainty_Quantification/LLPR/tests/test_config.py -v
```

Expected: FAIL because `Uncertainty_Quantification.LLPR.llpr.config` does not exist.

- [ ] **Step 3: Add the package and validated configuration models**

Implement Pydantic v2 models with `extra="forbid"` and exact discriminated behavior:

```python
REPO_ROOT = Path(__file__).resolve().parents[3]


class TargetEta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    energy: PositiveFloat
    force: PositiveFloat


class RidgeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["fixed", "fit"]
    max_condition_number: float = 1.0e10
    eta: PositiveFloat | TargetEta | None = None
    fit: RidgeFitConfig | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> "RidgeConfig":
        if self.mode == "fixed" and self.eta is None:
            raise ValueError("eta is required in fixed mode")
        if self.mode == "fit" and self.eta is not None:
            raise ValueError("eta is only valid in fixed mode")
        if self.mode == "fit" and self.fit is None:
            raise ValueError("fit configuration is required in fit mode")
        return self

    def fixed_eta(self) -> TargetEta:
        if self.mode != "fixed" or self.eta is None:
            raise ValueError("fixed_eta is only available in fixed mode")
        if isinstance(self.eta, TargetEta):
            return self.eta
        return TargetEta(energy=float(self.eta), force=float(self.eta))
```

`load_llpr_config()` must call `yaml.safe_load`, reject non-mapping roots, and
normalize all configured paths against `REPO_ROOT`, never against `Path.cwd()`.

- [ ] **Step 4: Add the dedicated tox environment and lint coverage**

Add `Uncertainty_Quantification/LLPR/` to `lint_folders`, add:

```ini
[testenv:llpr-tests]
description = Run UPET LLPR unit and integration tests
deps =
    pytest
    pydantic>=2
    pyyaml
    scipy
    matplotlib
commands =
    pytest {toxinidir}/Uncertainty_Quantification/LLPR/tests {posargs}
```

Add `"Uncertainty_Quantification"` to Ruff `known-first-party`, and register
`llpr_n20` and `llpr_legacy` pytest markers.

- [ ] **Step 5: Run focused tests and lint**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_config.py -v
tox -e lint
```

Expected: config tests PASS; lint reports no errors in the new package skeleton.

- [ ] **Step 6: Commit**

```bash
git add tox.ini pyproject.toml Uncertainty_Quantification/__init__.py \
  Uncertainty_Quantification/LLPR/__init__.py \
  Uncertainty_Quantification/LLPR/llpr/__init__.py \
  Uncertainty_Quantification/LLPR/llpr/config.py \
  Uncertainty_Quantification/LLPR/tests/conftest.py \
  Uncertainty_Quantification/LLPR/tests/test_config.py
git commit -m "test: establish LLPR configuration contracts"
```

### Task 2: Versioned Artifacts, Stage Identities, and Atomic I/O

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/artifacts.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_artifacts.py`

**Interfaces:**
- Consumes: `REPO_ROOT`, `LLPRConfig`, `resolved_config_payload`
- Produces: `SCHEMA_VERSION`, `FORMULA_VERSION`
- Produces: `canonical_json(value: Any) -> str`
- Produces: `stable_id(value: Any, length: int = 16) -> str`
- Produces: `sha256_file(path: Path) -> str`
- Produces: `RunPaths.from_config(config: LLPRConfig, checkpoint_sha256: str) -> RunPaths`
- Produces: `stage_identity(stage: str, payload: Mapping[str, Any]) -> dict[str, Any]`
- Produces: `atomic_json_dump`, `atomic_npz_save`, `load_complete_manifest`,
  `require_identity`, and `promote_staging_directory`

- [ ] **Step 1: Add failing artifact contract tests**

Cover deterministic JSON, stage-specific identity, atomic replacement, incomplete
artifact rejection, and non-overwrite behavior:

```python
def test_curvature_identity_ignores_calibration_config() -> None:
    fixed = stage_identity("curvature", curvature_payload(calibration="fixed"))
    fitted = stage_identity("curvature", curvature_payload(calibration="fit"))
    assert fixed == fitted


def test_require_complete_rejects_in_progress(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text('{"status":"in_progress"}', encoding="utf-8")
    with pytest.raises(ValueError, match="status must be complete"):
        load_complete_manifest(path)
```

- [ ] **Step 2: Run tests and verify missing interfaces**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_artifacts.py -v
```

Expected: FAIL because `artifacts.py` is missing.

- [ ] **Step 3: Implement canonical identities and run paths**

Use these constants and structures:

```python
SCHEMA_VERSION = "upet-llpr-artifact-v1"
FORMULA_VERSION = "upet-llpr-huber-readout-v1"


@dataclass(frozen=True)
class RunPaths:
    root: Path
    resolved_configs: Path
    curvature: Path
    calibration: Path
    evaluation: Path
    plots: Path
    legacy_raw: Path
```

`stage_identity()` must whitelist stage inputs rather than hash the entire config:

```python
STAGE_KEYS = {
    "curvature": (
        "checkpoint",
        "train",
        "readout",
        "formula",
        "curvature",
        "jacobian_backend",
        "matrix_dtype",
    ),
    "calibration": (
        "curvature_identity",
        "validation",
        "ridge",
        "formula",
    ),
    "evaluation": (
        "curvature_identity",
        "calibration_identity",
        "test",
        "formula",
    ),
}
```

- [ ] **Step 4: Implement atomic file and directory publication**

Write temporary files in the destination filesystem, `fsync` file handles, validate
the staged payload, and call `os.replace`. `atomic_npz_save()` must use a real
`.npz` temporary suffix so NumPy does not append a second suffix.

`promote_staging_directory()` must refuse to replace a complete directory with a
different identity. For the same identity it returns without changing bytes.

- [ ] **Step 5: Run focused and cumulative tests**

Run:

```bash
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_config.py \
  Uncertainty_Quantification/LLPR/tests/test_artifacts.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/artifacts.py \
  Uncertainty_Quantification/LLPR/tests/test_artifacts.py
git commit -m "feat: add LLPR artifact identities"
```

### Task 3: Current UPET Checkpoint, Readout, and Dataset Adapters

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/checkpoint.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/readout.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/data.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py`

**Interfaces:**
- Consumes: `FileIdentityConfig`, `DataConfig`, `sha256_file`
- Produces: `CheckpointIdentity`, `LoadedCheckpoint`, `load_checkpoint`
- Produces: `ParameterEntry`, `ReadoutLayout`, `discover_readout_layout`
- Produces: `DatasetIdentity`, `LLPRSample`, `dataset_identity`, `iter_samples`,
  `build_system`

- [ ] **Step 1: Add failing fake-model and temporary-extxyz tests**

The fake model must expose plural checkpoint parameter keys while the adapter returns
canonical target names:

```python
def test_readout_includes_weight_and_bias_in_stable_order() -> None:
    model = FakeUPETModel()
    layouts = discover_readout_layout(model)
    assert layouts["energy"].names == (
        "node_last_layers.energy.0.energy___0.weight",
        "node_last_layers.energy.0.energy___0.bias",
    )
    assert layouts["non_conservative_force"].dimension == 6


def test_dataset_identity_and_labels(tmp_path: Path) -> None:
    path = write_two_structure_extxyz(tmp_path)
    identity = dataset_identity(path)
    samples = list(iter_samples(path))
    assert identity.structure_count == 2
    assert samples[0].energy_ref_total == pytest.approx(-1.25)
    assert samples[0].force_ref.shape == (2, 3)
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py -v
```

Expected: FAIL because checkpoint/readout/data modules do not exist.

- [ ] **Step 3: Implement checkpoint identity and strict loading**

Use:

```python
@dataclass(frozen=True)
class CheckpointIdentity:
    path: str
    sha256: str
    model_class: str
    training_loss: dict[str, Any]


@dataclass
class LoadedCheckpoint:
    model: torch.nn.Module
    identity: CheckpointIdentity
```

`load_checkpoint()` must:

1. compare SHA before loading;
2. call `metatrain.utils.io.load_model(str(path))`;
3. set `.eval().to(device=device, dtype=dtype)`;
4. inspect raw checkpoint training metadata;
5. require energy weight 1.0, force weight 0.1, energy Huber 0.015,
   force Huber 0.01, energy divide-by-atoms, and mean force reduction;
6. report exact conflicting fields on failure.

- [ ] **Step 4: Implement stable readout discovery**

Resolve force keys from `("non_conservative_force", "non_conservative_forces")`, but
publish `target="non_conservative_force"`. For every weight from
`model.last_layer_parameter_names[target_key]`, append its matching bias immediately.
Reject missing parameters, duplicate names, zero dimensions, and layout drift.

Expose:

```python
@dataclass(frozen=True)
class ParameterEntry:
    name: str
    shape: tuple[int, ...]
    numel: int
    offset: int


@dataclass
class ReadoutLayout:
    target: str
    entries: tuple[ParameterEntry, ...]
    parameters: tuple[torch.nn.Parameter, ...]
    dimension: int
    layout_sha256: str

    def flatten(self, gradients: Sequence[Tensor]) -> Tensor: ...
```

- [ ] **Step 5: Implement deterministic extxyz iteration and system construction**

Extract labels only from audited candidates and require both labels:

```python
@dataclass(frozen=True)
class LLPRSample:
    structure_index: int
    atoms: Atoms
    energy_ref_total: float
    force_ref: np.ndarray
```

Use `ase.io.iread(path, index=":")`, `systems_to_torch`,
`model.requested_neighbor_lists()`, `_compute_ase_neighbors`, and
`register_autograd_neighbors`. Paths and identities include SHA and structure count.

- [ ] **Step 6: Run tests**

Run:

```bash
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py -v
```

Expected: fake-model and extxyz tests PASS without loading the formal checkpoint.

- [ ] **Step 7: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/checkpoint.py \
  Uncertainty_Quantification/LLPR/llpr/readout.py \
  Uncertainty_Quantification/LLPR/llpr/data.py \
  Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py
git commit -m "feat: adapt current UPET inputs for LLPR"
```

### Task 4: Observables, Jacobian Backends, and Huber Curvature

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/observables.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/curvature.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_observables_curvature.py`

**Interfaces:**
- Consumes: `LoadedCheckpoint`, `ReadoutLayout`, `LLPRSample`, `build_system`,
  artifact identities and atomic I/O
- Produces: `StructureJacobians`, `compute_structure_jacobians`
- Produces: `huber_curvature`
- Produces: `CurvatureAccumulator`, `run_build(config: LLPRConfig) -> Path`

- [ ] **Step 1: Add failing hand-calculated curvature tests**

Use a tiny model whose energy and direct force heads are linear:

```python
def test_curvature_matches_hand_calculation() -> None:
    jac = StructureJacobians(
        energy_total=torch.tensor(4.0),
        energy_per_atom=torch.tensor(2.0),
        force=torch.tensor([0.2, -0.4, 0.1]),
        energy_gradient=torch.tensor([1.0, 2.0]),
        force_gradients=torch.tensor([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]]),
    )
    acc = CurvatureAccumulator.zeros(energy_dim=2, force_dim=2)
    acc.add(
        jac,
        energy_residual_per_atom=0.01,
        force_residuals=np.array([0.001, 0.02, -0.001]),
        num_atoms=1,
        energy_weight=1.0,
        force_weight=0.1,
        energy_delta=0.015,
        force_delta=0.01,
    )
    assert_close(acc.energy, torch.tensor([[1.0, 2.0], [2.0, 4.0]]))
    expected_force = (0.1 / 3.0) * torch.tensor([[2.0, 1.0], [1.0, 1.0]])
    assert_close(acc.force, expected_force)
```

Add a backend parity test comparing scalar and batched VJP on the same toy model.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_observables_curvature.py -v
```

Expected: FAIL because observables and curvature modules are missing.

- [ ] **Step 3: Implement current-output forward and Jacobians**

Request only:

```python
outputs = model(
    [system],
    {
        "energy": ModelOutput(unit="eV", per_atom=False),
        "non_conservative_force": ModelOutput(unit="eV/A", per_atom=True),
    },
)
```

Define:

```python
@dataclass
class StructureJacobians:
    energy_total: Tensor
    energy_per_atom: Tensor
    force: Tensor
    energy_gradient: Tensor
    force_gradients: Tensor
```

`scalar_reference` loops over components. `batched_vjp` uses
`torch.autograd.grad(..., is_grads_batched=True)` in configured chunks. Both return
atom-major x/y/z order and must agree within `rtol=1e-5`, `atol=2e-6`.

- [ ] **Step 4: Implement exact Huber accumulation**

Use:

```python
def huber_curvature(residual: float, delta: float) -> float:
    return 1.0 if abs(residual) <= delta else 0.0
```

Energy uses `energy_gradient = grad(E_total / N)` and multiplies `w_E`.
Force uses `w_F/(3N)` once per component. Accumulate CPU float64 matrices only after
all outputs, labels, and Jacobians for a structure are finite.

- [ ] **Step 5: Add build progress and final diagnostics**

`run_build()` must write atomic progress after a configurable number of complete
structures. Progress binds curvature identity, next structure index, H matrices,
counts, and layout. Resume rejects any identity mismatch.

Final diagnostics include symmetry error, Frobenius norm, trace, eigenvalue bounds,
condition estimate, Huber inlier counts, and processed structure/component counts.

- [ ] **Step 6: Run focused tests**

Run:

```bash
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_observables_curvature.py -v
```

Expected: all math, backend parity, failure, and resume tests PASS.

- [ ] **Step 7: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/observables.py \
  Uncertainty_Quantification/LLPR/llpr/curvature.py \
  Uncertainty_Quantification/LLPR/tests/test_observables_curvature.py
git commit -m "feat: build UPET LLPR curvature"
```

### Task 5: Fixed and Validation-Fitted Eta Calibration

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/ridge.py`
- Create: `Uncertainty_Quantification/LLPR/llpr/calibration.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_ridge_calibration.py`

**Interfaces:**
- Consumes: complete curvature artifact, validation samples and Jacobians,
  `RidgeConfig`
- Produces: `Spectrum`, `RidgeCandidate`, `CalibrationRecord`
- Produces: `prepare_spectrum`, `fixed_candidate`, `fit_candidates`,
  `quadratic_forms`, `moment_alpha`, `gaussian_nll`, `select_candidate`
- Produces: `run_calibrate(config: LLPRConfig) -> Path`

- [ ] **Step 1: Add failing fixed and fit calibration tests**

Use small diagonal matrices with hand-computable q:

```python
def test_fixed_candidate_preserves_explicit_eta() -> None:
    h = torch.diag(torch.tensor([1.0, 4.0], dtype=torch.float64))
    candidate = fixed_candidate(h, eta=1e-6, max_condition_number=2.0)
    assert candidate.eta == 1e-6
    assert candidate.condition_warning is True


def test_fit_selects_energy_and_force_independently() -> None:
    result = calibrate_targets(
        energy_case_preferring_small_eta(),
        force_case_preferring_large_eta(),
        fit_config(),
    )
    assert result.energy.eta != result.force.eta
```

Add a sentinel `ForbiddenTestDataset` whose iterator raises. Calibration must finish
without touching it.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_ridge_calibration.py -v
```

Expected: FAIL because ridge/calibration modules do not exist.

- [ ] **Step 3: Implement spectrum validation and fixed eta**

Define:

```python
@dataclass(frozen=True)
class Spectrum:
    eigenvalues: Tensor
    eigenvectors: Tensor
    minimum: float
    maximum: float


@dataclass(frozen=True)
class RidgeCandidate:
    eta: float
    condition_number: float
    condition_warning: bool
    exclusion_reason: str | None = None
```

Symmetrize with `(H + H.T)/2`, reject materially negative eigenvalues, and retain
tiny roundoff negatives only in diagnostics. Fixed mode attempts exactly `H+eta I`;
it must never adjust eta.

- [ ] **Step 4: Implement deterministic fit candidates**

Use the approved formulas:

```python
spectral_eps = np.finfo(np.float64).eps * max(1.0, abs(mu_max))
eta_pd = max(0.0, -mu_min) + spectral_eps
eta_cond = max(0.0, (mu_max - kappa * mu_min) / (kappa - 1.0))
eta_floor = np.nextafter(max(eta_pd, eta_cond), np.inf)
candidates = sorted({eta_floor * multiplier for multiplier in multipliers})
```

Reject non-finite multipliers, multipliers below 1, and candidates that do not meet
the configured condition limit after floating-point evaluation.

- [ ] **Step 5: Implement q, Alpha, metrics, and selection**

Use Cholesky for formal q:

```python
factor = torch.linalg.cholesky(h + eta * torch.eye(h.shape[0], dtype=h.dtype))
solved = torch.cholesky_solve(gradients.T, factor).T
q = torch.sum(gradients * solved, dim=1)
alpha_sq = torch.mean(residuals.square() / q)
nll = 0.5 * torch.mean(
    torch.log(2.0 * torch.pi * alpha_sq * q)
    + residuals.square() / (alpha_sq * q)
)
```

Require finite positive q and Alpha. Sort valid candidates by `(nll, eta)` and save
all candidates, including exclusions and 1/2/3-sigma coverage.

- [ ] **Step 6: Implement streaming calibration and resume**

Process validation one complete structure at a time. Keep candidate-specific running
sums and, when needed for exact NLL after Alpha is known, deterministic NPZ shards of
residual/q pairs. Resume binds curvature identity, validation SHA, ridge config, and
committed structure count.

- [ ] **Step 7: Run tests**

Run:

```bash
tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_ridge_calibration.py -v
```

Expected: fixed, fit, no-test-leakage, invalid-candidate, and resume tests PASS.

- [ ] **Step 8: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/ridge.py \
  Uncertainty_Quantification/LLPR/llpr/calibration.py \
  Uncertainty_Quantification/LLPR/tests/test_ridge_calibration.py
git commit -m "feat: calibrate LLPR eta and alpha"
```

### Task 6: Test Inference, Indexing, Aggregation, and Resume

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/inference.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_inference.py`

**Interfaces:**
- Consumes: complete curvature and calibration manifests, test samples/Jacobians
- Produces: `EvaluationChunk`, `evaluate_structure`, `merge_evaluation_chunks`
- Produces: `run_evaluate(config: LLPRConfig) -> Path`

- [ ] **Step 1: Add failing formula and index tests**

```python
def test_evaluate_structure_preserves_atom_major_xyz_order() -> None:
    result = evaluate_structure(synthetic_two_atom_case(), calibration())
    assert result.force_atom_index.tolist() == [0, 0, 0, 1, 1, 1]
    assert result.force_cartesian_index.tolist() == [0, 1, 2, 0, 1, 2]
    assert result.force_offsets.tolist() == [0, 6]
    assert_allclose(result.energy_calibrated_var, ALPHA_E_SQ * result.energy_raw_var)
    assert_allclose(
        result.force_calibrated_var_component,
        ALPHA_F_SQ * result.force_raw_var_component,
    )
```

Add tests for total-energy derived variance, atom mean variance, structure RMS std,
inverse variance, non-positive q rejection, shard merge order, and resume.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_inference.py -v
```

Expected: FAIL because `inference.py` is missing.

- [ ] **Step 3: Implement per-structure evaluation**

Use fixed definitions:

```python
energy_residual_per_atom = energy_pred_total / n - energy_ref_total / n
energy_variance = alpha_energy_sq * energy_q
energy_total_variance_derived = n**2 * energy_variance
force_variance = alpha_force_sq * force_q
force_rigidity = 1.0 / force_variance
```

Force arrays are flattened atom-major x/y/z. Atom aggregation is the mean of three
component variances; structure aggregation is the mean across all `3N` components.

- [ ] **Step 4: Implement transactional NPZ shards and final merge**

Commit one structure per shard or bounded group of complete structures. Each shard
contains its inclusive/exclusive structure range, row counts, and identity. Merge
requires contiguous ranges, sequential structure indices, exact `3N` offsets, and
matching identities before publishing `details.npz`.

- [ ] **Step 5: Implement summary regeneration**

Generate counts, finite/non-positive diagnostics, residual/error metrics, q/variance
statistics, 1/2/3-sigma coverage, and standardized residual summaries directly from
the merged details. Never trust cached summary values during verification.

- [ ] **Step 6: Run tests**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_inference.py -v
```

Expected: formula, indexing, aggregation, merge, and resume tests PASS.

- [ ] **Step 7: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/inference.py \
  Uncertainty_Quantification/LLPR/tests/test_inference.py
git commit -m "feat: evaluate LLPR uncertainty artifacts"
```

### Task 7: Strict Model-Free Legacy Importer

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/legacy.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_legacy.py`
- Create: `Uncertainty_Quantification/LLPR/configs/import_legacy.yaml`

**Interfaces:**
- Consumes: artifact I/O and identity functions; no checkpoint, data model, or
  observables imports
- Produces: `LegacyImportConfig`, `SourceInventoryEntry`
- Produces: `build_source_inventory`, `classify_legacy_path`,
  `validate_legacy_matrices`, `validate_legacy_calibration`,
  `validate_legacy_evaluation`, `import_legacy`

- [ ] **Step 1: Add failing synthetic legacy-tree tests**

Build a tiny legacy fixture with full zero-padded matrices and exact result arrays:

```python
def test_import_legacy_extracts_active_blocks_without_loading_model(
    legacy_tree: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "metatrain.utils.io", ForbiddenModelModule())
    output = import_legacy(legacy_config(legacy_tree))
    energy = np.load(output / "curvature/energy_block.npz")["H"]
    force = np.load(output / "curvature/force_block.npz")["H"]
    assert energy.shape == (2, 2)
    assert force.shape == (3, 3)
```

Add tampering tests for matrix symmetry, `H_EF=H_E+H_F`, Alpha, offsets, residual,
variance, manifest hash, destination identity, and second-import no-op.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_legacy.py -v
```

Expected: FAIL because `legacy.py` is missing.

- [ ] **Step 3: Implement inventory and classification**

Every copied source file records:

```python
@dataclass(frozen=True)
class SourceInventoryEntry:
    source_path: str
    archived_path: str
    size_bytes: int
    sha256: str
    classification: Literal[
        "authoritative", "legacy_smoke", "incomplete", "orphan", "supporting"
    ]
    canonical_source: bool
```

Copy with `shutil.copy2` into staging and verify source/destination SHA. Do not copy
checkpoint or datasets; record their repository paths and expected SHA.

- [ ] **Step 4: Implement legacy numerical validation and conversion**

Require:

- full matrix shapes `(4104, 4104)`;
- energy slice `[0:1026]`, force slice `[1026:4104]`;
- finite exact symmetry;
- exact zero outside active blocks;
- exact `H_EF=H_E+H_F`;
- Alpha values `1.1467388818005693` and `0.2095766082027508`;
- fixed eta `1.0e-6`;
- 19,374 structures, 149,321 atoms, 447,963 force components;
- sequential indices and exact `3N` offsets;
- exact residual and calibrated-variance identities.

Publish extracted active blocks, canonical calibration summary, canonical evaluation
details/summary, root/stage manifests, and the `legacy_fixed_ridge` condition warning.
Recompute diagnostics using `H+eta I`; retain the old `H+eta^2 I` diagnostic only in
the raw archive.

- [ ] **Step 5: Add the formal import configuration**

Use exact source and target paths:

```yaml
legacy:
  source_root: /home/lilong/code/UQ/upet/UQ_LLPR/matpes_r2/Hef
  expected_source_commit: 4e0614c8309ed707c9cf93d04dab75274221119a
checkpoint:
  path: data/checkpoint/pet-omatpes-l-v0.1.0.ckpt
  expected_sha256: 879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
output:
  root: Uncertainty_Quantification/LLPR/outputs
  experiment: matpes_r2_legacy
```

Include the three exact dataset hashes from the approved design.

- [ ] **Step 6: Run tests**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_legacy.py -v
```

Expected: all synthetic import, tampering, model-isolation, atomicity, and idempotency
tests PASS.

- [ ] **Step 7: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/legacy.py \
  Uncertainty_Quantification/LLPR/tests/test_legacy.py \
  Uncertainty_Quantification/LLPR/configs/import_legacy.yaml
git commit -m "feat: import audited legacy LLPR artifacts"
```

### Task 8: Result-Only Plotting and Statistical Diagnostics

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/plotting.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_plotting.py`
- Create: `Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml`

**Interfaces:**
- Consumes: complete evaluation manifest and `details.npz`
- Produces: `FilteredPairs`, `PanelAnalysis`, `ReliabilityBin`
- Produces: `filter_log_pairs`, `deterministic_sample_indices`,
  `analyze_panel`, `reliability_bins`, `standardized_residual_cdf`,
  `run_plot`

- [ ] **Step 1: Add failing plotting-kernel and transaction tests**

```python
def test_analyze_panel_matches_known_statistics() -> None:
    uncertainty = np.array([1.0, 2.0, 4.0])
    error = np.array([1.0, 4.0, 2.0])
    result = analyze_panel(uncertainty, error)
    assert result.count == 3
    assert result.pearson_log10 == pytest.approx(0.5)
    assert result.coverage_1sigma == pytest.approx(2 / 3)


def test_plot_failure_preserves_previous_directory(
    complete_evaluation: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    previous_hashes = tree_hashes(existing_plot_directory())
    monkeypatch.setattr(plt.Figure, "savefig", raise_after_first_save)
    with pytest.raises(RuntimeError):
        run_plot(plot_config(complete_evaluation))
    assert tree_hashes(existing_plot_directory()) == previous_hashes
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_plotting.py -v
```

Expected: FAIL because `plotting.py` is missing.

- [ ] **Step 3: Implement numerical plotting kernels**

Use NumPy and SciPy only. Filter log plots to finite positive uncertainty and error,
use deterministic sampling with `np.random.default_rng(seed)`, calculate Pearson and
Spearman on `log10`, reliability coverage, standardized residual empirical CDF, and
Gaussian NLL. Preserve all full-data statistics even when plots are sampled.

- [ ] **Step 4: Render and publish audited outputs**

Create:

- energy-per-atom uncertainty vs absolute residual PNG/PDF;
- force-component uncertainty vs absolute residual PNG/PDF;
- reliability/coverage PNG/PDF;
- standardized-residual CDF PNG/PDF;
- `plotting_statistics.csv`;
- `plotting_manifest.json`.

Stage the entire plot directory and promote it only after every file and manifest hash
validates. Plotting must not import checkpoint, data, observables, or curvature modules.

- [ ] **Step 5: Add formal legacy plot configuration**

Point to `matpes_r2_legacy`, fixed calibration ID, seed 2026, explicit output formats,
and fixed axis/statistical settings. Do not include any Alpha multiplier.

- [ ] **Step 6: Run tests**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_plotting.py -v
```

Expected: numerical and transactional plotting tests PASS.

- [ ] **Step 7: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/plotting.py \
  Uncertainty_Quantification/LLPR/tests/test_plotting.py \
  Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml
git commit -m "feat: plot canonical LLPR results"
```

### Task 9: CLI, Runnable Configurations, and User Documentation

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
- Consumes: `run_build`, `run_calibrate`, `run_evaluate`, `import_legacy`,
  artifact verification, and `run_plot`
- Produces: `build_parser() -> argparse.ArgumentParser`
- Produces: `main(argv: Sequence[str] | None = None) -> int`

- [ ] **Step 1: Add failing CLI dispatch tests**

```python
@pytest.mark.parametrize(
    ("command", "target"),
    [
        ("build", "run_build"),
        ("calibrate", "run_calibrate"),
        ("evaluate", "run_evaluate"),
        ("import-legacy", "import_legacy"),
        ("plot", "run_plot"),
    ],
)
def test_cli_dispatches_one_stage(command: str, target: str, monkeypatch) -> None:
    calls = install_stage_spies(monkeypatch)
    assert main([command, "--config", str(CONFIG)]) == 0
    assert calls == [target]
```

Test that `run` calls build/calibrate/evaluate in order and stops on the first failure.
Test `verify --level metadata|full`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_cli.py -v
```

Expected: FAIL because CLI modules are missing.

- [ ] **Step 3: Implement the thin CLI**

Use subparsers with a required `--config` and no duplicated math:

```python
COMMANDS = ("build", "calibrate", "evaluate", "run", "import-legacy", "verify", "plot")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return dispatch(args)
    except Exception:
        logging.getLogger("upet.llpr").exception("LLPR command failed")
        return 1
```

`python -m Uncertainty_Quantification.LLPR.llpr` must call `main()`.

- [ ] **Step 4: Add exact n20 and future full configurations**

Both n20 configs use:

```yaml
checkpoint:
  path: data/checkpoint/pet-omatpes-l-v0.1.0.ckpt
data:
  build: data/dataset/matpes_n20.extxyz
  calibration: data/dataset/matpes_n20.extxyz
  test: data/dataset/matpes_n20.extxyz
runtime:
  device: cpu
  jacobian_backend: scalar_reference
  matrix_dtype: float64
output:
  root: Uncertainty_Quantification/LLPR/outputs
  experiment: n20_smoke
```

Fixed sets scalar eta `1e-6`; fit uses the approved condition limit and multipliers.
The GPU full configs point to the exact full datasets, use `cuda` and `batched_vjp`,
and contain an explicit comment and README warning that they are not run in this task.

- [ ] **Step 5: Write README usage and safety boundaries**

Document environment activation, every command, artifact tree, fixed/fit meanings,
resume, verification, legacy import, n20-only implementation validation, and the
explicit prohibition on presenting n20 as formal statistics.

- [ ] **Step 6: Run tests and CLI help**

Run:

```bash
tox -e llpr-tests -- Uncertainty_Quantification/LLPR/tests/test_cli.py -v
conda activate upet_new
python -m Uncertainty_Quantification.LLPR.llpr --help
```

Expected: dispatch tests PASS and help lists all seven commands.

- [ ] **Step 7: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/cli.py \
  Uncertainty_Quantification/LLPR/llpr/__main__.py \
  Uncertainty_Quantification/LLPR/tests/test_cli.py \
  Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml \
  Uncertainty_Quantification/LLPR/configs/cpu_n20_fit.yaml \
  Uncertainty_Quantification/LLPR/configs/gpu_full_fixed.yaml \
  Uncertainty_Quantification/LLPR/configs/gpu_full_fit.yaml \
  Uncertainty_Quantification/LLPR/README.md
git commit -m "feat: expose the UPET LLPR workflow"
```

### Task 10: n20 Full-Path Proof, Formal Legacy Migration, and Acceptance Report

**Files:**
- Create: `Uncertainty_Quantification/LLPR/tests/test_n20.py`
- Create after successful import: `Uncertainty_Quantification/LLPR/MIGRATION_REPORT.md`
- Modify if verification exposes a defect: only the responsible LLPR module/test

**Interfaces:**
- Consumes: all earlier public interfaces and configurations
- Produces: verified n20 fixed/fit artifacts, migrated formal artifacts, plots, and
  a tracked acceptance report

- [ ] **Step 1: Add the gated n20 end-to-end test**

```python
@pytest.mark.llpr_n20
@pytest.mark.skipif(
    os.environ.get("UPET_RUN_LLPR_N20") != "1",
    reason="set UPET_RUN_LLPR_N20=1 for the real-checkpoint smoke path",
)
def test_n20_fixed_and_fit_share_curvature(tmp_path: Path) -> None:
    fixed = run_config("cpu_n20_fixed.yaml", output_root=tmp_path)
    fitted = run_config("cpu_n20_fit.yaml", output_root=tmp_path)
    assert fixed.curvature_identity == fitted.curvature_identity
    assert fixed.energy_eta == pytest.approx(1e-6)
    assert fixed.force_eta == pytest.approx(1e-6)
    assert fitted.energy_eta > 0
    assert fitted.force_eta > 0
    verify_run(fixed.root, level="full")
    verify_run(fitted.root, level="full")
```

Also assert 20 energy records, 429 force-component records, no skipped samples,
finite positive variances, complete plots, and model-free plotting imports.

- [ ] **Step 2: Run all fast tests first**

Run:

```bash
tox -e llpr-tests -- -m "not llpr_n20 and not llpr_legacy" -v
```

Expected: all fast tests PASS; gated real-data tests are deselected or skipped.

- [ ] **Step 3: Run the fixed n20 chain**

Run:

```bash
conda activate upet_new
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml
python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml \
  --level full
```

Expected: build/calibrate/evaluate/plot complete; eta remains exactly `1e-6`.

- [ ] **Step 4: Run the fitted n20 chain**

Run:

```bash
python -m Uncertainty_Quantification.LLPR.llpr calibrate \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fit.yaml
python -m Uncertainty_Quantification.LLPR.llpr evaluate \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fit.yaml
python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fit.yaml
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fit.yaml \
  --level full
```

Expected: the existing curvature is reused; both fitted targets select valid eta;
all candidate records and plots are complete.

- [ ] **Step 5: Prove resume and idempotency**

Run the n20 fixed command again and compare all final artifact hashes except
explicit runtime timestamps. Exercise an interrupted temporary run using the gated
test fixture and prove its final arrays equal the uninterrupted arrays.

- [ ] **Step 6: Import the formal legacy results without model computation**

Run:

```bash
python -m Uncertainty_Quantification.LLPR.llpr import-legacy \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml \
  --level full
```

Expected:

```text
alpha_energy = 1.1467388818005693
alpha_force  = 0.2095766082027508
structures   = 19374
atoms        = 149321
force rows   = 447963
origin       = legacy_import
```

Confirm the old source tree Git status remains clean.

- [ ] **Step 7: Prove formal import idempotency**

Run the same `import-legacy` command again. Expected: verified no-op with identical
canonical and raw archive hashes.

- [ ] **Step 8: Plot the migrated formal result**

Run:

```bash
python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml
```

Expected reference correlations:

```text
energy Pearson(log10)  = 0.14123766265
energy Spearman(log10) = 0.13657688943
force Pearson(log10)   = 0.41166804002
force Spearman(log10)  = 0.45650505212
```

Use numeric tolerances, not PNG byte equality.

- [ ] **Step 9: Write the migration acceptance report**

Record:

- source and destination paths;
- design and implementation commit IDs;
- checkpoint/data/source inventory hashes;
- imported counts and Alpha;
- curvature dimensions and condition diagnostics;
- n20 fixed/fit selected eta and test counts;
- commands executed and their exit status;
- confirmation that full datasets were not recomputed;
- confirmation that `.gitignore` remained outside LLPR commits;
- incomplete/orphan files and their archive classification.

- [ ] **Step 10: Run final verification**

Run:

```bash
tox -e llpr-tests
tox -e lint
git diff --check
git status --short
```

Expected: LLPR tests and lint PASS; no whitespace errors; only intentionally
uncommitted user changes remain.

- [ ] **Step 11: Commit the gated test and acceptance report**

```bash
git add Uncertainty_Quantification/LLPR/tests/test_n20.py \
  Uncertainty_Quantification/LLPR/MIGRATION_REPORT.md
git commit -m "test: verify LLPR migration end to end"
```

## Plan Self-Review

- Spec coverage: Tasks 1-10 cover configuration, current API adaptation, readout
  discovery, Huber curvature, fixed/fit eta, inference, artifact identity, resume,
  model-free legacy import, plotting, CLI, n20 validation, formal migration, and
  documentation.
- Scope: The plan is one dependency-ordered deliverable. Legacy import and plotting
  are not independent products because both consume the canonical artifact protocol.
- Type consistency: `LLPRConfig`, `ReadoutLayout`, `StructureJacobians`,
  `RidgeCandidate`, stage identities, and run entry points are defined before their
  consumers.
- Data leakage: calibration consumes only validation; Task 5 includes an explicit
  test that raises if test data is accessed.
- Numerical compatibility: fixed eta is immutable; fitted eta is target-specific;
  all formal linear algebra is float64.
- Execution boundary: only n20 runs the model; formal full results enter through
  `import-legacy`.
- Placeholder scan: the plan contains no deferred implementation fields or
  unspecified error-handling steps.
