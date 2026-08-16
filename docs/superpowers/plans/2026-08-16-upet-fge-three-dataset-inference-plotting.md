# UPET FGE Three-Dataset Inference and Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the verified K=8 FGE ensemble to publish resumable prediction/UQ results for MAD test and MATPES train, reuse the existing MATPES test prediction, and generate separate reference-style raw uncertainty-versus-residual plots for all available domains.

**Architecture:** Add a path-independent inference-only contract beside the existing formal train/predict/evaluate contract. A verified completed FGE result supplies the immutable ensemble; a streaming reader creates bounded chunks, a narrow runtime applies the existing A3 members, and prediction/UQ/plot stages publish independently verified artifacts. Existing MATPES test artifacts are adapted read-only, while plotting ports only the analysis/rendering algorithms from `carnet_new` and continues to use the current descriptor-bound FGE artifact writer.

**Tech Stack:** Python 3.11, PyTorch, ASE, metatrain/metatomic, PyYAML, NumPy, SciPy, Matplotlib, pytest, Ruff, mypy.

## Global Constraints

- Work on the current local `Plots` branch; do not create or switch branches.
- Preserve every unrelated tracked or untracked file. In particular, never stage `.idea/`, ConfidenceHead test files, or plans created by concurrent work.
- Ensemble authority is the remote completed result `upet-FGE-CKPT-UQ-v1.0-full-v3`, result-manifest SHA `b1f4a3c5c7713b4be369b467e98e62d59bc23a13aefceffbd49ac51773b6dc86`.
- K is exactly 8 and ordered member IDs are exactly `member_001..member_008`.
- Do not call training, backward, or optimizer steps; MATPES test must not call model inference.
- MAD test SHA is `d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b`; it has energy/forces references but no stress reference.
- MATPES train SHA is `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec`; it has energy/forces/stress references.
- Preserve source-file order. Generate MAD IDs as `mad_test:{index:08d}` without changing the source file.
- Use current FGE population STD (`unbiased=False`, denominator K), not the carnet sample STD.
- Plot domains are Energy per atom, Force Cartesian component, and symmetrized Stress Voigt-6 `[xx, yy, zz, yz, xz, xy]`.
- Produce separate dataset directories. MAD has only Energy/Force. Never generate checkpoint-count plots or sweep tables.
- Large prediction/UQ artifacts remain remote and ignored; sync only PNG/PDF/statistics/manifest files locally.
- All filesystem publication must use current FGE secure atomic writers; no `Path.write_*`, `unlink`, `rmtree`, pathname `replace`, or symlink traversal.
- Every production change follows RED -> minimal GREEN -> focused regression -> commit.

---

### Task 1: Strict inference-only configuration

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/inference_config.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_inference_config.py`

**Interfaces:**
- Produces: `load_inference_config(path: str | Path) -> InferenceConfig`
- Produces immutable dataclasses `EnsembleSourceConfig`, `DatasetSourceConfig`, `ChunkPolicy`, `InferenceOutputConfig`, and `InferenceConfig`.
- `InferenceConfig.sanitized() -> dict[str, object]` returns path-neutral identity data; runtime paths remain under `runtime` and never enter the canonical run ID.

- [ ] **Step 1: Write the failing strict-schema tests**

```python
def test_inference_config_loads_exact_contract(tmp_path: Path) -> None:
    path = write_inference_config(tmp_path, stress_reference=False)
    config = load_inference_config(path)
    assert config.dataset.label == "mad_test"
    assert config.dataset.expected_sha256 == "d9a128" + "0" * 58
    assert config.dataset.reference_availability == {
        "energy": True, "forces": True, "stress": False
    }
    assert config.chunking.max_structures == 128
    assert config.chunking.max_atoms == 4096
    assert config.ensemble.member_count == 8


@pytest.mark.parametrize("field", ["member_count", "result_manifest_sha256"])
def test_inference_config_rejects_wrong_ensemble_contract(
    tmp_path: Path, field: str
) -> None:
    path = write_inference_config(tmp_path, stress_reference=False, mutate=field)
    with pytest.raises(HardFailure, match="ensemble"):
        load_inference_config(path)
```

- [ ] **Step 2: Run the test and confirm RED**

Run:

```bash
.tox/fge-tests/bin/python -m pytest \
  Uncertainty_Quantification/FGE/tests/test_inference_config.py -q
```

Expected: collection fails because `fge.inference_config` does not exist.

- [ ] **Step 3: Implement strict dataclasses and loader**

Implement these exact public shapes:

```python
@dataclass(frozen=True)
class DatasetSourceConfig:
    label: str
    path: Path
    expected_sha256: str
    split: str
    reference_availability: Mapping[str, bool]


@dataclass(frozen=True)
class ChunkPolicy:
    max_structures: int
    max_atoms: int


@dataclass(frozen=True)
class EnsembleSourceConfig:
    root: Path
    result_manifest_sha256: str
    base_checkpoint: Path
    base_checkpoint_sha256: str
    member_count: int


@dataclass(frozen=True)
class InferenceConfig:
    schema_version: str
    ensemble: EnsembleSourceConfig
    dataset: DatasetSourceConfig
    chunking: ChunkPolicy
    output: InferenceOutputConfig
    runtime: RuntimeInferenceConfig

    def sanitized(self) -> dict[str, object]: ...
```

Require exact YAML keys, lowercase SHA-256, labels matching `[a-z0-9][a-z0-9_]*`, `member_count == 8`, positive chunk limits, CPU device, and exactly three boolean availability keys. Resolve relative paths against the config file directory.

- [ ] **Step 4: Run focused tests and static checks**

Run:

```bash
.tox/fge-tests/bin/python -m pytest \
  Uncertainty_Quantification/FGE/tests/test_inference_config.py -q
python -m ruff format --check \
  Uncertainty_Quantification/FGE/fge/inference_config.py \
  Uncertainty_Quantification/FGE/tests/test_inference_config.py
python -m ruff check \
  Uncertainty_Quantification/FGE/fge/inference_config.py \
  Uncertainty_Quantification/FGE/tests/test_inference_config.py
python -m mypy Uncertainty_Quantification/FGE/fge/inference_config.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/FGE/fge/inference_config.py \
  Uncertainty_Quantification/FGE/tests/test_inference_config.py
git commit -m "feat(fge): define inference-only configuration"
```

### Task 2: Streaming dataset identity and chunk planning

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/inference_data.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_inference_data.py`

**Interfaces:**
- Consumes: `DatasetSourceConfig`, `ChunkPolicy` from Task 1.
- Produces: `DatasetRecord`, `DatasetScan`, `DatasetChunk` dataclasses.
- Produces: `scan_dataset(config, *, reader=None) -> DatasetScan`.
- Produces: `iter_dataset_chunks(config, scan, *, reader=None) -> Iterator[DatasetChunk]`.
- A private `_iter_ase_records(path, label, availability)` is the only ASE-reading seam; unit tests inject literal records and do not require ASE.

- [ ] **Step 1: Write failing tests for IDs, labels, and dual limits**

```python
def test_mad_ids_are_path_neutral_and_preserve_file_order() -> None:
    records = literal_records(atom_counts=(2, 5, 3), structure_ids=(None, None, None))
    scan = scan_dataset(mad_config(), reader=lambda _: iter(records))
    assert scan.structure_ids == (
        "mad_test:00000000", "mad_test:00000001", "mad_test:00000002"
    )


def test_chunking_obeys_structure_and_atom_limits() -> None:
    records = literal_records(atom_counts=(3, 4, 8, 2), structure_ids=("a", "b", "c", "d"))
    chunks = tuple(iter_dataset_chunks(train_config(max_structures=2, max_atoms=6),
                                       scan_for(records), reader=lambda _: iter(records)))
    assert [(c.start_structure, c.stop_structure, c.atom_count) for c in chunks] == [
        (0, 1, 3), (1, 2, 4), (2, 3, 8), (3, 4, 2)
    ]
    assert chunks[2].oversize_single_structure is True


def test_mixed_stress_availability_is_rejected() -> None:
    records = literal_records(stress=(tensor33(), None))
    with pytest.raises(HardFailure, match="mixed stress"):
        scan_dataset(train_config(), reader=lambda _: iter(records))
```

- [ ] **Step 2: Run and confirm RED**

Run the new test file. Expected: import failure for `inference_data`.

- [ ] **Step 3: Implement record normalization and two-pass streaming**

`scan_dataset` performs a first pass that hashes the ordinary file, validates all labels and IDs, counts structures/atoms, and records only compact ordered IDs/count metadata. `iter_dataset_chunks` performs a second pass and verifies every record against the scan while yielding bounded `DatasetChunk` objects. It must compare SHA/size/mtime before and after each pass.

Use this exact MAD ID rule:

```python
def canonical_structure_id(label: str, index: int, supplied: str | None) -> str:
    if supplied is not None:
        value = supplied.strip()
        if not value:
            raise HardFailure("dataset structure ID is empty")
        return value
    if label != "mad_test":
        raise HardFailure("dataset structure is missing structure_id")
    return f"mad_test:{index:08d}"
```

- [ ] **Step 4: Verify GREEN plus existing dataset tests**

Run:

```bash
.tox/fge-tests/bin/python -m pytest \
  Uncertainty_Quantification/FGE/tests/test_inference_data.py \
  Uncertainty_Quantification/FGE/tests/test_data.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/FGE/fge/inference_data.py \
  Uncertainty_Quantification/FGE/tests/test_inference_data.py
git commit -m "feat(fge): stream inference dataset chunks"
```

### Task 3: Verified ensemble authority and reusable PET runtime

**Files:**
- Modify: `Uncertainty_Quantification/FGE/fge/prediction.py`
- Create: `Uncertainty_Quantification/FGE/fge/inference_authority.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_inference_authority.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_prediction.py`

**Interfaces:**
- Consumes: `InferenceConfig` from Task 1.
- Produces: `EnsembleAuthority` with result/training manifest SHA, ordered member records, base SHA, writer/validator identities, and member directory.
- Produces: `open_ensemble_authority(config: InferenceConfig) -> EnsembleAuthority`.
- Adds `PETPredictionRuntime.load_reused_base(base_checkpoint, expected_sha256, members_directory) -> PETBase`.
- Adds immutable `PreparedPETChunk` and `PETPredictionRuntime.prepare_ase_chunk(base, atoms) -> PreparedPETChunk` plus `infer_prepared(base, prepared) -> Mapping[str, Tensor]`.
- Existing `load_base(FGEConfig)` and `infer_member(...)` remain behavior-compatible and delegate to the extracted primitives.

- [ ] **Step 1: Write RED tests for completed authority and path-independent member loading**

```python
def test_authority_reopens_completed_result_and_binds_all_members(
    completed_fge_root: Path,
) -> None:
    config = inference_config_for(completed_fge_root)
    authority = open_ensemble_authority(config)
    assert authority.member_ids == tuple(f"member_{i:03d}" for i in range(1, 9))
    assert len(authority.member_sha256) == 8
    assert authority.result_manifest_sha256 == config.ensemble.result_manifest_sha256


def test_authority_rejects_tampered_member(completed_fge_root: Path) -> None:
    mutate_member(completed_fge_root, "member_004")
    with pytest.raises(HardFailure, match="member"):
        open_ensemble_authority(inference_config_for(completed_fge_root))
```

Add a prediction regression proving `load_reused_base` reads members from the supplied verified directory instead of `output_root/project/training/members`.

- [ ] **Step 2: Run and confirm RED**

Expected: `open_ensemble_authority` and `load_reused_base` are missing.

- [ ] **Step 3: Implement authority reopening**

Call `validate_completed_result(root)` first. Then reopen `result_manifest.json` and `training/manifest.json`, validate exact schemas already defined by current validators, recompute all declared member hashes, and bind the exact member list. Reject source/candidate symlinks using current path helpers. Do not save the absolute ensemble path in `EnsembleAuthority.canonical_identity()`.

- [ ] **Step 4: Refactor PET runtime without changing existing formal prediction**

Extract the current base-checkpoint and inference logic so both entry points use the same code. `prepare_ase_chunk` converts only the supplied chunk to float32 metatomic systems and adds requested neighbor lists once. `infer_prepared` calls `evaluate_model` for energy/non-conservative forces/non-conservative stress and does not wrap the energy path in `torch.no_grad()` because metatrain derives gradients for force/stress outputs.

- [ ] **Step 5: Run focused and full prediction tests**

Run:

```bash
.tox/fge-tests/bin/python -m pytest \
  Uncertainty_Quantification/FGE/tests/test_inference_authority.py \
  Uncertainty_Quantification/FGE/tests/test_prediction.py -q
```

Expected: pass; the real-checkpoint test remains conditionally skipped locally.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/FGE/fge/prediction.py \
  Uncertainty_Quantification/FGE/fge/inference_authority.py \
  Uncertainty_Quantification/FGE/tests/test_inference_authority.py \
  Uncertainty_Quantification/FGE/tests/test_prediction.py
git commit -m "feat(fge): verify reusable ensemble authority"
```

### Task 4: Resumable K=8 chunk prediction publication

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/inference_only.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_inference_only.py`

**Interfaces:**
- Consumes: Tasks 1-3.
- Defines `ChunkInferenceRuntime` protocol with `load_reused_base`, `prepare_chunk`, `restore_and_apply`, and `infer_prepared`.
- Produces: `predict_inference_dataset(config: InferenceConfig, *, runtime: object | None = None) -> Path` returning `prediction/manifest.json`.
- Produces: `validate_prediction_chunk(payload, expected_identity) -> PredictionChunkShape`.

- [ ] **Step 1: Write RED tests for chunk-major K=8, resume, and guards**

```python
def test_predicts_each_chunk_once_with_all_eight_members(tmp_path: Path) -> None:
    runtime = LiteralChunkRuntime()
    manifest = predict_inference_dataset(config(tmp_path), runtime=runtime)
    assert runtime.prepare_calls == ["chunk_000000", "chunk_000001"]
    assert runtime.applied_members == [
        (chunk, f"member_{index:03d}")
        for chunk in (0, 1) for index in range(1, 9)
    ]
    assert json_load(manifest)["chunk_count"] == 2


def test_resume_reuses_only_hash_verified_chunks(tmp_path: Path) -> None:
    first = LiteralChunkRuntime(stop_after=1)
    with pytest.raises(SimulatedStop):
        predict_inference_dataset(config(tmp_path), runtime=first)
    second = LiteralChunkRuntime()
    predict_inference_dataset(config(tmp_path), runtime=second)
    assert second.prepare_calls == ["chunk_000001"]


def test_prediction_path_never_calls_training(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(training, "train_fge", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    predict_inference_dataset(config(), runtime=LiteralChunkRuntime())
```

Add tamper, wrong K/order, gap/overlap, dataset SHA mismatch, and existing conflicting-manifest tests.

- [ ] **Step 2: Run and confirm RED**

Expected: import failure for `inference_only`.

- [ ] **Step 3: Implement exact chunk schema and publication**

Each `.pt` is self-describing and contains canonical input identity, range metadata, references/availability, K=8 predictions, topology tensors, dtypes/shapes, and formula-free statistics. Publish with `atomic_torch_save`; its existence counts as reusable only after reopening with `weights_only=True` and complete validation. Publish `prediction/manifest.json` last with the exact ordered chunk path/SHA/shape/range inventory.

Do not use a two-file chunk transaction or delete orphans. An orphan whose embedded identity validates may be adopted; a conflicting orphan hard-fails.

- [ ] **Step 4: Implement real runtime adapter**

The default adapter loads `PETPredictionRuntime.load_reused_base(...)` once per process, calls `prepare_ase_chunk` once per chunk, loops over the eight authority members, and stacks outputs. References and topology come from `DatasetChunk`, not from the model output.

- [ ] **Step 5: Run focused tests and full existing prediction tests**

Run the new file plus `test_prediction.py`, `test_artifacts.py`, and `test_failure_policy.py`. Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/FGE/fge/inference_only.py \
  Uncertainty_Quantification/FGE/tests/test_inference_only.py
git commit -m "feat(fge): publish resumable dataset predictions"
```

### Task 5: Chunk UQ, optional stress references, and read-only validator

**Files:**
- Modify: `Uncertainty_Quantification/FGE/fge/uncertainty.py`
- Create: `Uncertainty_Quantification/FGE/fge/inference_evaluation.py`
- Create: `Uncertainty_Quantification/FGE/fge/inference_validation.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_inference_evaluation.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_inference_validation.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_uncertainty.py`

**Interfaces:**
- Adds `tensor_to_voigt_symmetric(stress: torch.Tensor) -> torch.Tensor` using `[xx, yy, zz, yz, xz, xy]`.
- Produces: `evaluate_prediction_chunk(payload: Mapping[str, object]) -> dict[str, torch.Tensor]`.
- Produces: `evaluate_inference_dataset(config: InferenceConfig) -> Path`.
- Produces: `validate_inference_result(root: str | Path, *, read_only: bool = True) -> InferenceValidationReport`.
- Uncertainty chunk keys are exact: `energy_per_atom_std`, `force_component_std`, `stress_component_std`, and matching `*_absolute_residual` only when the reference is available.

- [ ] **Step 1: Write formula RED tests**

```python
def test_stress_is_symmetrized_in_fixed_voigt_order() -> None:
    stress = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
    assert torch.equal(
        tensor_to_voigt_symmetric(stress),
        torch.tensor([[1., 5., 9., 7., 5., 3.]])
    )


def test_energy_uses_population_std_after_per_atom_conversion() -> None:
    payload = literal_prediction(energy=[[2., 8.], [4., 12.]], n_atoms=[2, 4])
    result = evaluate_prediction_chunk(payload)
    expected = torch.std(torch.tensor([[1., 2.], [2., 3.]]), dim=0, unbiased=False)
    assert torch.equal(result["energy_per_atom_std"], expected)


def test_mad_omits_stress_residual_but_keeps_unsupervised_std() -> None:
    result = evaluate_prediction_chunk(literal_prediction(stress_reference=None))
    assert "stress_component_std" in result
    assert "stress_component_absolute_residual" not in result
```

- [ ] **Step 2: Write validator RED tests**

Cover missing/duplicate chunks, noncontiguous structure/atom ranges, bad prediction hash, wrong formula version, mixed reference availability, metrics that differ from a fresh global recomputation, and read-only bytes/mtime invariance.

- [ ] **Step 3: Run and confirm RED**

Expected: missing modules/functions.

- [ ] **Step 4: Implement UQ and global statistics**

For each prediction chunk, compute population STD and residual tensors and atomically publish one uncertainty chunk. Reopen every published chunk. Concatenate only compact one-dimensional domain arrays when calculating dataset-level Spearman, Pearson(log10), exclusions, and summaries; never average per-chunk correlations. Publish `uncertainty/manifest.json` and `metrics.json` last.

- [ ] **Step 5: Implement complete read-only validation**

Validator independently reopens authority identities, prediction chunks, uncertainty chunks, manifests, and metrics. It recomputes every UQ chunk and global metric, checks exact file inventory, rejects symlinks/residue, and writes nothing in read-only mode.

- [ ] **Step 6: Run focused/full UQ tests and commit**

```bash
.tox/fge-tests/bin/python -m pytest \
  Uncertainty_Quantification/FGE/tests/test_uncertainty.py \
  Uncertainty_Quantification/FGE/tests/test_inference_evaluation.py \
  Uncertainty_Quantification/FGE/tests/test_inference_validation.py -q
git add Uncertainty_Quantification/FGE/fge/uncertainty.py \
  Uncertainty_Quantification/FGE/fge/inference_evaluation.py \
  Uncertainty_Quantification/FGE/fge/inference_validation.py \
  Uncertainty_Quantification/FGE/tests/test_uncertainty.py \
  Uncertainty_Quantification/FGE/tests/test_inference_evaluation.py \
  Uncertainty_Quantification/FGE/tests/test_inference_validation.py
git commit -m "feat(fge): evaluate inference-only uncertainty"
```

### Task 6: Reference-style single-panel plotting

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/plot_analysis.py`
- Create: `Uncertainty_Quantification/FGE/fge/plot_rendering.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_plot_analysis.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_plot_rendering.py`
- Modify: `tox.ini`

**Interfaces:**
- Produces immutable `PlotSettings`, `PanelAnalysis`, `PlotInput`, and `PlotAnalysisResult`.
- Produces immutable `PlotConfig` and `load_plot_config(path: str | Path) -> PlotConfig`.
- Produces `analyze_plot_input(input: PlotInput, settings: PlotSettings) -> PlotAnalysisResult`.
- Produces `render_plot_suite(result: PlotAnalysisResult, output_root: Path) -> Path`.
- Produces adapters `load_completed_fge_plot_input(root)` and `load_inference_plot_input(root)`.

- [ ] **Step 1: Port the reference analysis tests as RED tests**

Use non-circular hard-coded fixtures for log filtering, density mass levels, deterministic sampling, and correlations. Add explicit assertions that population STD values come from input artifacts and plot analysis never calls `sample_std`.

```python
def test_mad_domains_are_exactly_energy_force() -> None:
    result = analyze_plot_input(mad_plot_input(), PlotSettings())
    assert result.domains == ("energy", "force")


def test_plotting_rejects_checkpoint_sweep_fields() -> None:
    with pytest.raises(HardFailure, match="checkpoint"):
        load_plot_config(config_with_checkpoint_sweep=True)
```

- [ ] **Step 2: Port rendering tests as RED tests**

Assert exact image names/counts, PNG dimensions/DPI, PDF readability, manifest SHA inventory, identical rerun no-op, conflicting output rejection, and that MAD has no stress files.

- [ ] **Step 3: Add plotting test dependencies**

Add `numpy`, `scipy`, and `matplotlib` to `[testenv:fge-tests].deps` in `tox.ini`. Do not add checkpoint sweep dependencies or runtime coupling to carnet.

- [ ] **Step 4: Implement analysis by adapting carnet algorithms**

Port the positive finite filter, 160x160 histogram, `scipy.ndimage.gaussian_filter(..., sigma=1.2, mode="nearest")`, contour-mass thresholds, fixed sample seed `20260714`, Spearman, and Pearson(log10). Preserve source attribution in module docstrings. The adapters must validate source manifests before reading tensors.

- [ ] **Step 5: Implement rendering and secure publication**

Port only `build_single_panel_figure` behavior. Fixed style: 7x7 inches, 300 DPI, `#f28e2b`, size 2.0, alpha 0.035, max 20,000 samples, grey `residual <= uncertainty`, dashed `y=x`, and five contour masses. Use current `sibling_staging`, `atomic_write_json`, and exact final inventory; do not port carnet's pathname replacement/deletion helpers.

- [ ] **Step 6: Run focused tests and visual QA**

Run both plot test files. Open the three synthetic PNGs and compare against the reference dimensions/layout; verify labels and log limits are not clipped. Expected exact outputs are 8 files for MATPES-like input and 6 for MAD.

- [ ] **Step 7: Commit**

```bash
git add tox.ini \
  Uncertainty_Quantification/FGE/fge/plot_analysis.py \
  Uncertainty_Quantification/FGE/fge/plot_rendering.py \
  Uncertainty_Quantification/FGE/tests/test_plot_analysis.py \
  Uncertainty_Quantification/FGE/tests/test_plot_rendering.py
git commit -m "feat(fge): render raw uncertainty residual plots"
```

### Task 7: Thin stage CLIs and formal configs

**Files:**
- Create: `Uncertainty_Quantification/FGE/scripts/predict_dataset.py`
- Create: `Uncertainty_Quantification/FGE/scripts/evaluate_dataset.py`
- Create: `Uncertainty_Quantification/FGE/scripts/plot_dataset.py`
- Create: `Uncertainty_Quantification/FGE/configs/inference_mad_test.yaml`
- Create: `Uncertainty_Quantification/FGE/configs/inference_matpes_train.yaml`
- Create: `Uncertainty_Quantification/FGE/configs/plot_matpes_test.yaml`
- Create: `Uncertainty_Quantification/FGE/configs/plot_mad_test.yaml`
- Create: `Uncertainty_Quantification/FGE/configs/plot_matpes_train.yaml`
- Create: `Uncertainty_Quantification/FGE/tests/test_inference_scripts.py`
- Modify: `Uncertainty_Quantification/FGE/fge/__init__.py`

**Interfaces:**
- Each CLI has `main(argv: list[str] | None = None) -> int` and exactly one required `--config` argument.
- HardFailure exits 2 and prints one concise stderr line.
- No CLI exposes `run-all`, `train`, `checkpoint-sweep`, or arbitrary runtime override flags.

- [ ] **Step 1: Write direct-path CLI RED tests**

```python
@pytest.mark.parametrize("script", [
    "predict_dataset.py", "evaluate_dataset.py", "plot_dataset.py"
])
def test_cli_runs_directly_and_accepts_only_config(script: str) -> None:
    result = subprocess.run([sys.executable, SCRIPTS / script, "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--config" in result.stdout
    assert "run-all" not in result.stdout
    assert "train" not in result.stdout
```

Add callback monkeypatch tests proving each CLI calls only its named stage.

- [ ] **Step 2: Run and confirm RED**

Expected: script files are missing.

- [ ] **Step 3: Implement scripts and exact YAML files**

Use the verified local/remote hashes from the design. YAML runtime paths use `${UPET_FGE_*}` variables only where the existing config policy already permits environment injection; otherwise use explicit remote config variants and keep local test fixtures separate. Plot configs point to one result each and one dataset-specific output directory.

- [ ] **Step 4: Export stable APIs and run all script/config tests**

Run new CLI tests, existing `test_scripts.py`, config tests, and public import checks. Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/FGE/scripts/predict_dataset.py \
  Uncertainty_Quantification/FGE/scripts/evaluate_dataset.py \
  Uncertainty_Quantification/FGE/scripts/plot_dataset.py \
  Uncertainty_Quantification/FGE/configs/inference_mad_test.yaml \
  Uncertainty_Quantification/FGE/configs/inference_matpes_train.yaml \
  Uncertainty_Quantification/FGE/configs/plot_matpes_test.yaml \
  Uncertainty_Quantification/FGE/configs/plot_mad_test.yaml \
  Uncertainty_Quantification/FGE/configs/plot_matpes_train.yaml \
  Uncertainty_Quantification/FGE/fge/__init__.py \
  Uncertainty_Quantification/FGE/tests/test_inference_scripts.py
git commit -m "feat(fge): add inference and plotting commands"
```

### Task 8: Complete local verification and independent code review

**Files:**
- Modify only files required by verified review findings.

- [ ] **Step 1: Run the complete FGE suite**

```bash
.tox/fge-tests/bin/python -m Uncertainty_Quantification.FGE.tests.runner
```

Expected: all collected tests pass; only existing environment-gated real PET tests may skip.

- [ ] **Step 2: Run formatting, lint, types, and diff checks**

```bash
python -m ruff format --check Uncertainty_Quantification/FGE
python -m ruff check Uncertainty_Quantification/FGE
python -m mypy Uncertainty_Quantification/FGE
git diff --check
git status --short
```

Verify only task files are staged; preserve unrelated untracked files.

- [ ] **Step 3: Request independent review**

Review against the approved design with special attention to: no-training guards, authority identity, chunk coverage, optional stress, population-vs-sample STD, secure publication, plot input read-only behavior, and forbidden sweep artifacts.

- [ ] **Step 4: Address findings through separate RED/GREEN commits**

For each Critical/Important finding, first add a focused failing regression, prove RED, implement the minimum fix, run focused and full gates, and commit only the affected files. Repeat independent review until Critical=0 and Important=0.

### Task 9: Remote real-data execution and artifact delivery

**Files:**
- Create: `docs/superpowers/reports/2026-08-16-upet-fge-three-dataset-inference-plotting.md`
- Generate and sync: `Uncertainty_Quantification/Plots/FGE/{matpes_test,mad_test,matpes_train}/`

**Interfaces:**
- Remote checkout: `/home/bywang/code/UQ/upet_new_fge_test`.
- Remote ensemble root: `/home/bywang/code/UQ/upet_new_fge_test/Uncertainty_Quantification/FGE/outputs/upet-FGE-CKPT-UQ-v1.0-full-v3`.
- Remote MAD data: `/home/bywang/code/UQ/mace_new-plots/data/dataset/mad-test.xyz`.
- Remote train data: `/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_train.extxyz`.

- [ ] **Step 1: Deploy exact committed code and run preflight**

Archive/sync tracked committed files only. Verify local commit, remote tracked hashes, conda environment, data SHA, ensemble result-manifest SHA, member inventory, and output absence/identity. Record a before snapshot of all source hashes/size/mtime.

- [ ] **Step 2: Prove MATPES test is plot-only**

Install guards that raise on PET load/inference/training/backward/optimizer. Run `plot_dataset.py` with `plot_matpes_test.yaml`. Validate PNG/PDF/statistics/manifest and prove source result bytes/mtime unchanged.

- [ ] **Step 3: Run one real MAD chunk smoke**

Use the production runtime and one bounded chunk. Record thread count, structures, atoms, wall time, peak RSS, tensor shapes, references, and K=8 member order. Delete nothing; smoke output uses a distinct run ID.

- [ ] **Step 4: Run complete MAD predict, evaluate, and plot**

Run only the three named scripts in sequence. Validate read-only result, prove `stress_reference=false`, and verify exactly Energy/Force plot files.

- [ ] **Step 5: Run one real MATPES train chunk benchmark**

Run a distinct smoke identity. Select final CPU thread/chunk values only from measured peak RSS and wall time; update the formal config and commit that exact parameter change before the full run.

- [ ] **Step 6: Run complete resumable MATPES train predict**

Start the formal prediction. Monitor progress through verified chunk inventory, not log silence. If interrupted, rerun the same command and prove already verified chunks were not recomputed. Then run evaluate and plot.

- [ ] **Step 7: Perform deep read-only acceptance**

Verify before/after source hashes, every chunk range, K/order, dataset coverage, random independent UQ recomputation, global metrics, output inventories, PNG/PDF readability, plot SHA manifests, no sweep files, no training calls, and repeated read-only bytes/mtime invariance.

- [ ] **Step 8: Sync only final plot artifacts and write report**

Copy the three verified plot directories and no prediction/UQ chunks. Report exact remote roots, elapsed times, counts, hashes, active domains, formula version, tests/static gates, and reviewer result.

- [ ] **Step 9: Commit plots and report**

```bash
git add Uncertainty_Quantification/Plots/FGE/matpes_test \
  Uncertainty_Quantification/Plots/FGE/mad_test \
  Uncertainty_Quantification/Plots/FGE/matpes_train \
  docs/superpowers/reports/2026-08-16-upet-fge-three-dataset-inference-plotting.md
git commit -m "feat(fge): publish three-dataset uncertainty plots"
```

### Task 10: Final release verification

**Files:**
- No production changes unless a fresh release review finds a regression.

- [ ] **Step 1: Fresh final gates**

Run complete FGE tests, Ruff format/check, mypy, plot manifest verifier, remote inference validator, and repeated completed-result/read-only validation.

- [ ] **Step 2: Fresh independent release review**

Reviewer checks committed objects, remote artifact hashes, source immutability, exact plot inventory, no forbidden outputs, and the final report. Approval requires Critical=0, Important=0.

- [ ] **Step 3: Report completion**

Provide the implementation commits, remote result paths, local plot paths, test counts, exact final hashes, and any retained failed staging artifacts. Do not claim completion until every gate in this task has fresh evidence.
