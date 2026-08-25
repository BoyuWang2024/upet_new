# ConfidenceHead MAD r2SCAN E0 Postprocessing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a configuration-driven, audited ConfidenceHead workflow that preserves UPET/ConfidenceHead inference while publishing uncorrected, test-informed direct-E0, and val-calibrated model-aware energy-error variants plus continuous density plots.

**Architecture:** Existing external prediction code produces immutable raw energy artifacts for the filtered MAD r2SCAN val/test datasets and the eight completed energy heads. New focused modules read checkpoint composition weights and extxyz reference fields in float64, fit the two E0 calibrations, publish shared energy variants that reference rather than copy logits/expected error, and adapt those variants to the existing density renderer. The workflow is fail-closed, identity-addressed, atomic, and does not touch force artifacts.

**Tech Stack:** Python 3.11, PyTorch, NumPy, ASE, Pydantic v2, matplotlib, pytest, existing ConfidenceHead artifact/density helpers.

---

## File Map

- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_config.py`: strict paths, dataset names/counts, and plotting settings.
- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_calibration.py`: extxyz field extraction, composition matrices, checkpoint E0 extraction, SVD fitting, correction, metrics.
- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_publication.py`: raw prediction audit, three immutable variants, calibration tables/manifests, verification.
- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_plotting.py`: energy-only `PlotSeries` adapters and atomic density publication.
- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/e0_commands.py`: prediction reuse, calibration/publication, and plotting orchestration.
- Create `Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_r2scan_e0.py`: command-line stage dispatcher.
- Create `Uncertainty_Quantification/ConfidenceHead/configs/predict_r2scan_e0_gpu.yaml`: filtered val/test raw prediction inputs.
- Create `Uncertainty_Quantification/ConfidenceHead/configs/e0_postprocessing_gpu.yaml`: formal calibration/output/plot settings.
- Create `Uncertainty_Quantification/ConfidenceHead/run/submit_e0_postprocessing.sh`: Slurm stages `predict`, `postprocess`, `plot`, and `all`.
- Modify `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py`: add the narrowly validated `postprocessing` external prediction profile.
- Modify `Uncertainty_Quantification/ConfidenceHead/EXTERNAL_PREDICTION.md`: document E0 semantics and exact commands.
- Create focused tests `test_e0_config.py`, `test_e0_calibration.py`, `test_e0_publication.py`, `test_e0_plotting.py`, and `test_e0_commands.py`.
- Modify `test_external_config.py` and `test_external_scripts.py`: cover the new profile and shipped formal files.

### Task 1: Strict Campaign Configuration

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_config.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py`
- Test: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_config.py`
- Test: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py`

- [ ] **Step 1: Write failing strict-config tests**

```python
def test_e0_config_resolves_paths_and_requires_distinct_datasets(tmp_path: Path) -> None:
    config = load_e0_config(_write_config(tmp_path))
    assert config.external_config == (tmp_path / "prediction.yaml").absolute()
    assert config.validation_dataset == "mad_r2scan_val"
    assert config.test_dataset == "mad_r2scan_test"
    assert config.validation_expected.structures == 16098
    assert config.test_expected.atoms == 311657

def test_postprocessing_profile_requires_exact_val_test_sources(tmp_path: Path) -> None:
    config = load_external_config(_write_prediction_config(tmp_path))
    assert config.profile == "postprocessing"
    assert set(config.datasets) == {"mad_r2scan_val", "mad_r2scan_test"}
```

- [ ] **Step 2: Run tests and observe missing config/profile failures**

Run:
```bash
pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_e0_config.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py
```

Expected: import/profile validation failures.

- [ ] **Step 3: Implement strict models and duplicate-key-safe loader**

```python
class ExpectedDatasetCounts(StrictModel):
    structures: int = Field(gt=0)
    atoms: int = Field(gt=0)
    elements: int = Field(gt=0)

class E0PostprocessingConfig(StrictModel):
    external_config: Path
    validation_dataset: str
    test_dataset: str
    validation_expected: ExpectedDatasetCounts
    test_expected: ExpectedDatasetCounts
    output_root: Path
    plots_root: Path
    plot: DensityPlotSettings = Field(default_factory=DensityPlotSettings)

    @model_validator(mode="after")
    def distinct_datasets(self) -> "E0PostprocessingConfig":
        if self.validation_dataset == self.test_dataset:
            raise ValueError("validation_dataset and test_dataset must differ")
        return self
```

Extend only `ExternalPredictionConfig.profile` with literal `postprocessing`; require exactly the two safe names above and two `ExtXYZSource` values. Keep existing production/smoke rules unchanged.

- [ ] **Step 4: Run config tests**

Expected: all selected tests pass.

### Task 2: Float64 E0 Numerical Core

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_calibration.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_calibration.py`

- [ ] **Step 1: Write failing synthetic-data tests**

```python
def test_full_rank_svd_recovers_direct_and_model_aware_weights() -> None:
    composition = torch.tensor([[1, 0], [0, 1], [1, 1]], dtype=torch.float64)
    mad_e0 = torch.tensor([-2.0, -5.0], dtype=torch.float64)
    atomization = torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64)
    target = atomization + composition @ mad_e0
    fitted = solve_full_rank_svd(composition, target - atomization)
    assert torch.allclose(fitted.solution, mad_e0, atol=1e-12, rtol=0)

def test_model_aware_uses_total_energy_residual_without_atom_weighting() -> None:
    fitted = fit_model_aware(composition, target, raw_prediction)
    assert torch.allclose(fitted.solution, known_delta, atol=1e-12, rtol=0)

def test_rank_deficient_and_uncovered_elements_fail_closed() -> None:
    with pytest.raises(ValueError, match="rank deficient"):
        solve_full_rank_svd(torch.ones((3, 2), dtype=torch.float64), torch.ones(3))
```

Also test explicit calculator `energy` versus `atoms.info["atomization_energy"]`, unsupported element rejection, correction signs, per-atom absolute errors, mean signed error, MAE/RMSE/P95, and label regeneration.

- [ ] **Step 2: Run the numerical tests and observe import failures**

- [ ] **Step 3: Implement immutable numeric records and functions**

```python
@dataclass(frozen=True)
class E0Dataset:
    structure_ids: torch.Tensor
    atomic_numbers: torch.Tensor
    atom_offsets: torch.Tensor
    composition: torch.Tensor
    target_energy_r2scan: torch.Tensor
    atomization_energy: torch.Tensor

@dataclass(frozen=True)
class SvdSolution:
    solution: torch.Tensor
    rank: int
    singular_values: torch.Tensor
    condition_number: float
    residual_rmse: float
    residual_max_abs: float

def solve_full_rank_svd(matrix: Tensor, rhs: Tensor) -> SvdSolution:
    u, singular_values, vh = torch.linalg.svd(matrix.to(torch.float64), full_matrices=False)
    rank = int(torch.linalg.matrix_rank(matrix).item())
    if rank != matrix.shape[1]:
        raise ValueError("composition matrix is rank deficient")
    solution = vh.mT @ ((u.mT @ rhs) / singular_values)
    residual = matrix @ solution - rhs
    return SvdSolution(
        solution=solution,
        rank=rank,
        singular_values=singular_values,
        condition_number=float(singular_values[0] / singular_values[-1]),
        residual_rmse=float(torch.sqrt(torch.mean(residual.square()))),
        residual_max_abs=float(torch.max(torch.abs(residual))),
    )
```

Load extxyz with ASE, read absolute energy only from calculator results/`info.energy`, read atomization energy only from `info.atomization_energy`, bind SHA before/after, and preserve structure order.

Extract checkpoint E0 by finding exactly one energy CompositionModel, calling `sync_tensor_maps()`, and reading `model.weights["energy"]` by its `center_type` samples. Reject missing, duplicate, non-scalar, non-finite, or mismatched atomic types.

- [ ] **Step 4: Run numerical tests**

Expected: all numerical tests pass without loading the 772 MB checkpoint.

### Task 3: Raw Prediction Audit and Atomic Variant Publication

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_publication.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_publication.py`

- [ ] **Step 1: Write failing publication tests**

```python
def test_campaign_references_raw_uq_without_copying_it(tmp_path: Path) -> None:
    result = publish_e0_campaign(inputs, output_root)
    manifest = verify_e0_campaign(result / "manifest.json", full=True)
    assert set(manifest["variants"]) == set(E0_VARIANTS)
    assert not list(result.rglob("*logits*"))
    assert not list(result.rglob("*expected_errors*"))
    assert source_predictions.read_bytes() == source_before

def test_campaign_rejects_cross_order_raw_energy_mismatch(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="raw energy.*order"):
        publish_e0_campaign(mismatched_inputs, output_root)

def test_campaign_reuses_identity_and_rejects_conflicting_existing_output(
    tmp_path: Path,
) -> None:
    first = publish_e0_campaign(_campaign_inputs("1" * 64), tmp_path / "campaign")
    second = publish_e0_campaign(_campaign_inputs("1" * 64), tmp_path / "campaign")
    assert second == first
    with pytest.raises(ValueError, match="identity"):
        publish_e0_campaign(_campaign_inputs("2" * 64), tmp_path / "campaign")
```

Cover raw manifest SHA validation, structure IDs/order/offsets/composition identity, all-eight-order completeness, finite values, threshold consistency, unchanged raw SHA, staging cleanup, and no force artifact output.

- [ ] **Step 2: Run tests and observe missing publication API**

- [ ] **Step 3: Implement audited loading and derivation**

For each energy order, use `verify_external_prediction(source / "manifest.json", full=True)` and `load_verified_torch`. Validate raw prediction/reference/structure identity across orders and against the float64 extxyz dataset.

Compute once:
```python
direct = raw_test - test.composition @ model_e0 + test.composition @ mad_e0_test
model_aware = raw_test + test.composition @ delta_e0
observed = torch.abs(corrected - test.target_energy_r2scan) / test.atom_counts
labels = labels_from_thresholds(observed, shared_thresholds)
```

Publish one shared `energy_data.pt` per variant, metrics JSON, calibration JSON/CSV, and a manifest that binds every raw manifest/artifact SHA. Do not copy logits, representatives, expected errors, force data, logs, or checkpoints. Write into a sibling `.staging-<uuid>`, verify, then `os.replace`; reuse only an identical complete identity.

- [ ] **Step 4: Run publication tests**

Expected: all publication tests pass.

### Task 4: Energy-Only Continuous Density Publication

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/e0_plotting.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_plotting.py`

- [ ] **Step 1: Write failing adapter/render tests**

```python
def test_variant_series_preserves_expected_and_replaces_only_observed() -> None:
    series = build_variant_series(raw_series, corrected_observed)
    assert torch.equal(series.logits, raw_series.logits)
    assert torch.equal(series.expected, raw_series.expected)
    assert torch.equal(series.representatives, raw_series.representatives)
    assert torch.equal(series.observed, corrected_observed)

def test_density_publication_contains_three_variants_eight_orders_no_force(
    tmp_path: Path,
) -> None:
    path = publish_e0_density_campaign(_density_inputs(), tmp_path / "plots", settings)
    manifest = verify_e0_density_publication(path, full=True)
    assert len(list(path.rglob("test_energy_expected_vs_observed_density.png"))) == 24
    assert not list(path.rglob("*force*"))
    assert not list(path.rglob("*boxplot*"))
```

- [ ] **Step 2: Run tests and observe missing plotting API**

- [ ] **Step 3: Reuse existing density analysis/rendering**

Use `dataclasses.replace(raw_series, observed=variant_observed)`, `render_density_panel`, `render_energy_comparison`, `analyze_density_panel`, and shared log limits across all three variants for a given comparison. Publish PNG/PDF/CSV/JSON per order and a comparison correlation CSV/JSON/PNG/PDF with explicit `test_informed` and `val_calibrated` labels.

- [ ] **Step 4: Run plotting tests**

Expected: plots decode, PDFs are non-empty, hashes verify, no force/boxplot artifacts exist, and staging is removed.

### Task 5: Workflow, CLI, Formal Configs, and Slurm

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/e0_commands.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/postprocess_r2scan_e0.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/predict_r2scan_e0_gpu.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/e0_postprocessing_gpu.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_e0_postprocessing.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_e0_commands.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts.py`

- [ ] **Step 1: Write failing command/script tests**

```python
def test_predict_stage_runs_only_eight_energy_heads(monkeypatch, config) -> None:
    outputs = predict_e0_inputs(config)
    assert len(outputs) == 16
    assert all("force" not in str(path) for path in outputs)

def test_cli_dispatches_explicit_stage(monkeypatch, tmp_path) -> None:
    assert module.main(["--config", str(path), "--stage", "postprocess"]) == 0
    assert seen == ["postprocess"]
```

Verify the shipped configs bind checkpoint SHA `879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004`, val/test SHA `4f4d4807592d75cfedda4a157850d60fc1428e44762e8debf37c012a4fc060aa`/`499b479499eb56d0792360cb8bcb3397b566ac99c4e290ce0e866382c7f4d2ed`, expected counts 16,098/310,432 and 16,072/311,657, remote bywang paths, and the dedicated output roots. Verify the Slurm script invokes only the E0 command and contains no training/W&B command.

- [ ] **Step 2: Run command tests and observe missing workflow**

- [ ] **Step 3: Implement orchestration**

`predict` discovers the exact eight energy runs, builds/reuses one cache for each val/test dataset, and calls existing `predict_dataset_run` only for those energy runs. `postprocess` verifies all inputs, loads the checkpoint E0 once, fits both calibrations, and publishes the campaign. `plot` reads only the complete campaign and raw predictions. `all` executes those stages in order.

- [ ] **Step 4: Add formal YAML and Slurm stage dispatcher**

Use `/home/bywang/code/UQ/upet_new` paths, `/home/bywang/.conda/envs/upet_new/bin/python`, and a single GPU. No temporary order-specific configs are created.

- [ ] **Step 5: Run command/config/script tests**

Expected: all pass.

### Task 6: Documentation and Full Verification

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/EXTERNAL_PREDICTION.md`

- [ ] **Step 1: Document the exact energy semantics**

State that raw UPET and ConfidenceHead logits/expected error remain unchanged; direct replacement is test-informed/oracle; model-aware is unweighted val-only total-energy OLS; observed error is eV/atom; forces are neither recomputed nor republished.

- [ ] **Step 2: Run focused ConfidenceHead tests**

```bash
conda run -n upet_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests
```

Expected: zero failures.

- [ ] **Step 3: Run static checks on touched Python files**

```bash
conda run -n upet_new ruff format --check Uncertainty_Quantification/ConfidenceHead/confidence_head Uncertainty_Quantification/ConfidenceHead/scripts Uncertainty_Quantification/ConfidenceHead/tests
conda run -n upet_new ruff check Uncertainty_Quantification/ConfidenceHead/confidence_head Uncertainty_Quantification/ConfidenceHead/scripts Uncertainty_Quantification/ConfidenceHead/tests
```

Expected: zero errors.

- [ ] **Step 4: Run a local synthetic CPU end-to-end publication/render test**

Run the dedicated E0 tests with `-vv`; expected output includes complete campaign and density manifests with no staging directories.

- [ ] **Step 5: Commit only task files**

Use explicit pathspecs and inspect `git diff --cached --name-status` before committing so the pre-existing LLPR/plot staging remains untouched.

### Task 7: Remote Full Campaign and Audit

**Files:** No tracked source changes; generated outputs remain ignored.

- [ ] **Step 1: Push the current branch and update the bywang checkout**

Confirm the remote checkout revision exactly matches the implementation commit.

- [ ] **Step 2: Submit/run the formal `all` stage**

```bash
cd /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/run
STAGE=all sbatch submit_e0_postprocessing.sh
```

- [ ] **Step 3: Verify full counts and artifacts**

Run the verification CLI against the published manifests. Require val/test counts 16,098/16,072, 89 elements, 3 variants, 8 orders, 24 PNG and 24 PDF order panels, complete comparison artifacts, finite metrics, and no staging directories.

- [ ] **Step 4: Audit invariants**

Re-hash all referenced raw `predictions.pt`, logits/expected error sources, and force artifacts before/after. Require exact equality and report direct as test-informed and model-aware as val-calibrated.
