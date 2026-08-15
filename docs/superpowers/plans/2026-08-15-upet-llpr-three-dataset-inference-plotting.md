# UPET LLPR Three-Dataset Inference and Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse the verified MATPES LLPR curvature and calibration artifacts, compute only the required MAD and MATPES-train stages on the remote GPU server, and publish six carnet-style uncertainty plots plus the agreed local result set.

**Architecture:** Add strict optional stage-reuse configuration and a transactional reuse materializer, then route the existing calibration and evaluation entry points through stage resolvers so numerical work is skipped when a verified stage is supplied. Add a streaming semantic dataset fingerprint for the non-byte-identical remote MATPES train file, replace the single-evaluation diagnostic plotter with a multi-evaluation carnet-style publisher, and keep remote-only large results separate from the local summary bundle.

**Tech Stack:** Python 3.11, Pydantic v2, NumPy, PyTorch, ASE, SciPy, Matplotlib, PyYAML, pytest, Ruff, mypy, Conda `upet_new`, SSH, Slurm.

## Global Constraints

- Work only on the existing local `Plots` branch; do not modify or commit `.idea/`.
- Activate the local environment with `source /home/lilong/miniforge3/etc/profile.d/conda.sh && conda activate upet_new`.
- Never change the 11 files under `Uncertainty_Quantification/LLPR/outputs/matpes_r2`; its curvature, calibration, and evaluation identities remain `981cc8bdf7f8b820`, `169d1d75c0dc1190`, and `519237da91c72b73`.
- The checkpoint SHA-256 is `879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004`.
- MAD uses fixed energy and force eta `1.0e-6`, recalculates Alpha on `mad-val-compatible.xyz`, and evaluates `mad-test-compatible.xyz`.
- MATPES train reuses Alpha values `1.1467388818005693` and `0.2095766082027508`; it must not rebuild curvature or recalibrate.
- Do not touch `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new`; deploy only to `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_llpr_codex`.
- Accept SSH host fingerprint `SHA256:SPxE4NOiUvxqOCvLRGKSMb1URKFcLmVJoKiJSOVvH2g` only; retry another load-balanced endpoint instead of disabling host verification.
- Run model work through Slurm partition `ai`, without `--gres`, using one node, one task, 8 CPUs, 64 GiB, and `CUDA_VISIBLE_DEVICES=0`.
- Run a real small-data remote full-chain test before formal computation, then remove all small-test outputs and temporary subsets.
- Keep complete `mad_test` locally; keep complete `matpes_train` remotely and do not copy its `details.npz` locally.
- The final plot directory contains exactly 12 PNG/PDF figures, `plotting_statistics.csv`, and `plotting_manifest.json`.
- Follow red-green-refactor for every behavior change and make focused commits after each independently verified task.

---

### Task 1: Strict Stage-Reuse Configuration

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/config.py:32-159`
- Modify: `Uncertainty_Quantification/LLPR/tests/conftest.py:8-44`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_config.py:17-75`

**Interfaces:**
- Produces: `StageReuseConfig(path: Path, identity: str)` and `ReuseConfig(curvature, calibration)`.
- Produces: `LLPRConfig.reuse: ReuseConfig | None`.
- Guarantees: `data.build` is optional only with curvature reuse; `data.calibration` is optional only with calibration reuse; `data.test` always exists.

- [ ] **Step 1: Write failing configuration tests**

```python
def test_curvature_reuse_allows_missing_build(write_llpr_config):
    path = write_llpr_config({
        "reuse": {"curvature": {"path": "base/curvature/id", "identity": "a" * 16}}
    })
    raw = yaml.safe_load(path.read_text())
    raw["data"].pop("build")
    path.write_text(yaml.safe_dump(raw))
    config = load_llpr_config(path)
    assert config.data.build is None
    assert config.reuse.curvature.path == REPO_ROOT / "base/curvature/id"

def test_calibration_reuse_requires_curvature_reuse(write_llpr_config):
    path = write_llpr_config({
        "reuse": {"calibration": {"path": "base/calibration/id", "identity": "b" * 16}}
    })
    with pytest.raises(ValueError, match="calibration reuse requires curvature reuse"):
        load_llpr_config(path)
```

- [ ] **Step 2: Verify the new tests fail for the missing `reuse` field**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_config.py -q`

Expected: the reuse cases fail because `LLPRConfig` rejects the unknown `reuse` key or still requires the omitted data path.

- [ ] **Step 3: Implement the minimal strict models and cross-field validation**

```python
class StageReuseConfig(StrictModel):
    path: Path
    identity: str = Field(pattern=r"^[0-9a-f]{16}$")

class ReuseConfig(StrictModel):
    curvature: StageReuseConfig | None = None
    calibration: StageReuseConfig | None = None

class DataConfig(StrictModel):
    build: Path | None = None
    calibration: Path | None = None
    test: Path

class LLPRConfig(StrictModel):
    reuse: ReuseConfig | None = None

    @model_validator(mode="after")
    def validate_stage_inputs(self) -> "LLPRConfig":
        reuse = self.reuse or ReuseConfig()
        if reuse.curvature is None and self.data.build is None:
            raise ValueError("build data is required without curvature reuse")
        if reuse.calibration is None and self.data.calibration is None:
            raise ValueError("calibration data is required without calibration reuse")
        if reuse.calibration is not None and reuse.curvature is None:
            raise ValueError("calibration reuse requires curvature reuse")
        return self
```

Resolve `reuse.curvature.path` and `reuse.calibration.path` with `resolve_repo_path`, just like checkpoint, data, and output paths.

- [ ] **Step 4: Run the focused configuration tests**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_config.py -q`

Expected: all configuration tests pass, including missing-path rejection, reuse-path resolution, identity format rejection, and legacy configuration compatibility.

- [ ] **Step 5: Commit the configuration contract**

```bash
git add Uncertainty_Quantification/LLPR/llpr/config.py Uncertainty_Quantification/LLPR/tests/conftest.py Uncertainty_Quantification/LLPR/tests/test_config.py
git commit -m "feat(llpr): configure verified stage reuse"
```

### Task 2: Transactional Reuse Validation and Materialization

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/reuse.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_reuse.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/artifacts.py:153-205`

**Interfaces:**
- Consumes: `load_verified_manifest`, `SCHEMA_VERSION`, `FORMULA_VERSION`, `canonical_json`, and declared manifest file hashes.
- Produces: `materialize_reused_stage(source, destination_root, stage, identity, expected_payload, expected_curvature_identity=None) -> Path`.
- Produces: `validate_curvature_layout(stage_dir, manifest, layout) -> None`.

- [ ] **Step 1: Write failing tests for source verification and target safety**

```python
def test_reuse_falls_back_to_copy_on_cross_device_link(
    curvature_source, tmp_path, monkeypatch
):
    def cross_device_link(source, destination):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(os, "link", cross_device_link)
    target = materialize_reused_stage(
        curvature_source,
        tmp_path / "target/curvature",
        stage="curvature",
        identity="a" * 16,
        expected_payload={"checkpoint_sha256": "b" * 64},
    )
    assert (target / "curvature.npz").read_bytes() == (
        curvature_source / "curvature.npz"
    ).read_bytes()
    assert load_verified_manifest(target / "manifest.json", verify_npz=True)
```

Each fixture writes a real complete manifest and real small NPZ/JSON artifacts; the cross-device test patches `os.link` to raise `OSError(errno.EXDEV)` and verifies copied bytes and hashes.

- [ ] **Step 2: Run the new tests and observe the import failure**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_reuse.py -q`

Expected: collection fails because `llpr.reuse` does not exist.

- [ ] **Step 3: Implement strict manifest validation**

```python
def materialize_reused_stage(
    source: Path,
    destination_root: Path,
    *,
    stage: Literal["curvature", "calibration"],
    identity: str,
    expected_payload: Mapping[str, object],
    expected_curvature_identity: str | None = None,
) -> Path:
    manifest = load_verified_manifest(
        source / "manifest.json",
        {
            "stage": stage,
            "identity": identity,
            "schema_version": SCHEMA_VERSION,
            "formula_version": FORMULA_VERSION,
        },
        verify_npz=True,
    )
```

Require the stage-specific files (`curvature.npz` and `diagnostics.json`, or `summary.json` and `candidates.json`), compare every requested payload key with canonical JSON, and compare calibration `curvature_identity` when supplied.

- [ ] **Step 4: Implement atomic hardlink/copy materialization**

Create a unique sibling staging directory. For each declared file and `manifest.json`, try `os.link`; on `errno.EXDEV`, `errno.EPERM`, or `errno.EACCES`, use `shutil.copy2`. Re-run complete verification in staging, rename staging to `destination_root / identity`, and remove staging on every failure. If the target already exists, verify it and either return it unchanged or raise; never overwrite it.

- [ ] **Step 5: Implement curvature layout validation**

Read `diagnostics.json` and `curvature.npz`; require square finite energy/force matrices, dimensions equal to `ReadoutLayout.energy.dimension` and `.force.dimension`, diagnostics dimensions equal to the arrays, and `manifest["layout_hash"] == layout.layout_hash`.

- [ ] **Step 6: Run reuse and artifact tests**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_reuse.py Uncertainty_Quantification/LLPR/tests/test_artifacts.py -q`

Expected: all tests pass and injected failures leave no staging or partial target directory.

- [ ] **Step 7: Commit transactional reuse**

```bash
git add Uncertainty_Quantification/LLPR/llpr/artifacts.py Uncertainty_Quantification/LLPR/llpr/reuse.py Uncertainty_Quantification/LLPR/tests/test_reuse.py
git commit -m "feat(llpr): materialize verified reusable stages"
```

### Task 3: Route Calibration and Evaluation Through Reusable Stages

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/curvature.py:135-268`
- Modify: `Uncertainty_Quantification/LLPR/llpr/calibration.py:155-313`
- Modify: `Uncertainty_Quantification/LLPR/llpr/inference.py:266-441`
- Modify: `Uncertainty_Quantification/LLPR/llpr/cli.py:10-89`
- Create: `Uncertainty_Quantification/LLPR/tests/test_stage_resolution.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_cli.py:1-77`

**Interfaces:**
- Produces: `resolve_curvature_stage(config: LLPRConfig) -> Path`.
- Produces: `resolve_calibration_stage(config: LLPRConfig) -> Path`.
- Existing `run_build`, `run_calibrate`, and `run_evaluate` retain their public signatures for non-reuse configurations.

- [ ] **Step 1: Write failing no-compute and binding tests**

```python
def test_curvature_reuse_does_not_call_run_build(
    reused_curvature_config, monkeypatch
):
    def fail_numerical_build(config):
        raise AssertionError("numerical stage called")

    monkeypatch.setattr(curvature, "run_build", fail_numerical_build)
    resolved = curvature.resolve_curvature_stage(reused_curvature_config)
    assert resolved.name == reused_curvature_config.reuse.curvature.identity
    assert load_verified_manifest(resolved / "manifest.json", verify_npz=True)

def test_calibration_reuse_does_not_call_run_calibrate(
    reused_calibration_config, monkeypatch
):
    def fail_numerical_calibration(config):
        raise AssertionError("numerical stage called")

    monkeypatch.setattr(calibration, "run_calibrate", fail_numerical_calibration)
    resolved = calibration.resolve_calibration_stage(reused_calibration_config)
    assert resolved.name == reused_calibration_config.reuse.calibration.identity
    assert load_verified_manifest(resolved / "manifest.json")
```

The patched compute functions raise `AssertionError("numerical stage called")`; reuse paths must return without triggering them.

- [ ] **Step 2: Run tests and verify the resolver imports fail**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_stage_resolution.py -q`

Expected: collection fails because the two resolver functions are absent.

- [ ] **Step 3: Implement curvature resolution**

```python
def resolve_curvature_stage(config: LLPRConfig) -> Path:
    if config.reuse is None or config.reuse.curvature is None:
        return run_build(config)
    root = config.output.root / config.output.experiment
    return materialize_reused_stage(
        config.reuse.curvature.path,
        RunPaths(root).curvature,
        stage="curvature",
        identity=config.reuse.curvature.identity,
        expected_payload={
            "checkpoint_sha256": config.checkpoint.expected_sha256,
            "curvature": config.curvature.model_dump(mode="json"),
            "matrix_dtype": config.runtime.matrix_dtype,
            "jacobian_backend": config.runtime.jacobian_backend,
            "force_component_chunk_size": config.runtime.force_component_chunk_size,
        },
    )
```

- [ ] **Step 4: Implement calibration resolution and update callers**

`run_calibrate` calls `resolve_curvature_stage`. `resolve_calibration_stage` materializes a configured calibration with expected curvature identity and exact ridge payload, otherwise calls `run_calibrate`. `run_evaluate` calls both resolvers and checks that the chosen calibration manifest binds the chosen curvature. CLI `build` and `calibrate` wrappers call the resolvers, so a reuse configuration never enters numerical build/calibration code.

- [ ] **Step 5: Bind reused curvature to the loaded model layout**

Immediately after `discover_readout_layout` in calibration and evaluation, call `validate_curvature_layout`. This validates both reused and newly computed curvature before quadratic forms are evaluated.

- [ ] **Step 6: Run focused and full LLPR unit tests**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_stage_resolution.py Uncertainty_Quantification/LLPR/tests/test_cli.py Uncertainty_Quantification/LLPR/tests/test_n20.py -q`

Then: `python -m pytest Uncertainty_Quantification/LLPR/tests -q`

Expected: all available tests pass; real-checkpoint tests may retain only their pre-existing environment-based skips.

- [ ] **Step 7: Commit stage orchestration**

```bash
git add Uncertainty_Quantification/LLPR/llpr/curvature.py Uncertainty_Quantification/LLPR/llpr/calibration.py Uncertainty_Quantification/LLPR/llpr/inference.py Uncertainty_Quantification/LLPR/llpr/cli.py Uncertainty_Quantification/LLPR/tests/test_stage_resolution.py Uncertainty_Quantification/LLPR/tests/test_cli.py
git commit -m "feat(llpr): resolve reused stages without recomputation"
```

### Task 4: Streaming MATPES Semantic Fingerprint

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/dataset_fingerprint.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_dataset_fingerprint.py`

**Interfaces:**
- Produces: `SemanticDatasetFingerprint(sha256, structure_count, atom_count, force_component_count, quantum)`.
- Produces: `semantic_dataset_fingerprint(path: Path, quantum: float = 1.0e-6) -> SemanticDatasetFingerprint`.
- Produces: module CLI `python -m Uncertainty_Quantification.LLPR.llpr.dataset_fingerprint --dataset PATH --output JSON`.

- [ ] **Step 1: Write failing semantic-equivalence tests**

Create real extxyz fixtures with the same ordered structures but `id` versus `structure_id`, different ignored metadata, and sub-quantum formatting differences. Assert equal fingerprints. Add separate tests proving changed structure order, atomic order, PBC, cell, positions, energy, or force above `1.0e-6` changes the fingerprint, and missing identifiers or labels raise a structure-indexed error.

- [ ] **Step 2: Run the new tests and observe the missing module failure**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_dataset_fingerprint.py -q`

Expected: collection fails because `dataset_fingerprint.py` does not exist.

- [ ] **Step 3: Implement a platform-stable streaming hash**

For each `iter_samples(path)` item, append explicit tagged, length-prefixed little-endian bytes for structure index, normalized identifier, atomic numbers/order, PBC, quantized cell, quantized positions, quantized total energy, and quantized forces. Quantize with `np.rint(values / quantum).astype("<i8")`; append final structure/atom/component counts. Ignore stress and unrelated textual metadata.

- [ ] **Step 4: Implement deterministic JSON output**

The module CLI writes sorted, newline-terminated JSON atomically. It includes only the semantic digest, counts, and quantum, not an absolute source path.

- [ ] **Step 5: Run fingerprint tests and a two-structure CLI smoke test**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_dataset_fingerprint.py -q`

Expected: equality and inequality cases pass, and two CLI invocations produce byte-identical JSON for semantically identical fixtures.

- [ ] **Step 6: Commit the audit tool**

```bash
git add Uncertainty_Quantification/LLPR/llpr/dataset_fingerprint.py Uncertainty_Quantification/LLPR/tests/test_dataset_fingerprint.py
git commit -m "feat(llpr): fingerprint extxyz datasets semantically"
```

### Task 5: Multi-Dataset Carnet-Style Plot Publisher

**Files:**
- Replace: `Uncertainty_Quantification/LLPR/llpr/plotting.py`
- Replace: `Uncertainty_Quantification/LLPR/tests/test_plotting.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_n20.py:95-118`

**Interfaces:**
- Produces: `PlotEvaluationConfig(label, run_root, evaluation_identity=None)`.
- Produces: `PlotStyleConfig` with the approved grid, smoothing, contour, sampling, color, size, and DPI values.
- Produces: `PlotConfig(evaluations, output_root, style)` supporting one or more evaluations.
- Produces: `shared_square_log_limits`, `smoothed_histogram_contours`, `analyze_panel`, and `run_plot`.

- [ ] **Step 1: Replace old reliability/CDF tests with failing publication tests**

```python
def test_run_plot_publishes_exactly_fourteen_files(tmp_path):
    evaluations = tuple(
        _write_evaluation(tmp_path / label, label, scale)
        for label, scale in (
            ("matpes_test", 1.0),
            ("mad_test", 10.0),
            ("matpes_train", 100.0),
        )
    )
    destination = run_plot(
        PlotConfig(evaluations=evaluations, output_root=tmp_path / "plots")
    )
    assert {path.name for path in destination.iterdir()} == {
        *(f"llpr_{label}_{target}_uncertainty_vs_residual.{suffix}"
          for label in ("matpes_test", "mad_test", "matpes_train")
          for target in ("energy", "force")
          for suffix in ("png", "pdf")),
        "plotting_statistics.csv",
        "plotting_manifest.json",
    }
```

Each fake evaluation contains a verified `details.npz` with the real UPET keys `energy_calibrated_std`, `energy_residual`, `force_calibrated_std_component`, and `force_residual`.

- [ ] **Step 2: Verify tests fail against the single-evaluation plotter**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_plotting.py -q`

Expected: failures show the missing `evaluations` configuration and missing 14-file output contract.

- [ ] **Step 3: Implement strict multi-input loading and analysis**

Resolve exactly one complete evaluation per configured label, verify its manifest and `details.npz`, reject duplicate/unsafe labels, and map energy to `(energy_calibrated_std, abs(energy_residual))` and force to `(force_calibrated_std_component, abs(force_residual))`. Compute one square log range across all energy inputs and one across all force inputs.

- [ ] **Step 4: Port the approved carnet analysis parameters**

Use grid `160`, Gaussian sigma `1.2`, contour masses `(0.5, 0.7, 0.85, 0.95, 0.99)`, sample maximum `20000`, seed `20260714`, log margin `0.05`, size `(7.0, 7.0)`, orange `#f28e2b`, scatter size `12.0`, alpha `0.04`, and DPI `300`. Record Spearman, log10 Pearson, quantiles, excluded-value categories, and uncertainty/error ratios.

- [ ] **Step 5: Render and transactionally publish the exact filenames**

For each label and `energy`/`force`, render `llpr_<label>_<target>_uncertainty_vs_residual.{png,pdf}` with square log axes, gray region, dashed `y=x`, density contours, deterministic scatter, correlations, title, and physical units. Write six CSV rows and `plotting_manifest.json`; publish via a sibling staging directory with rollback-safe replacement.

- [ ] **Step 6: Run plotting and n20 tests**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_plotting.py Uncertainty_Quantification/LLPR/tests/test_n20.py -q`

Expected: all tests pass, every PNG/PDF is nonempty, and manifest hashes verify.

- [ ] **Step 7: Commit the plot publisher**

```bash
git add Uncertainty_Quantification/LLPR/llpr/plotting.py Uncertainty_Quantification/LLPR/tests/test_plotting.py Uncertainty_Quantification/LLPR/tests/test_n20.py
git commit -m "feat(llpr): publish carnet-style multi-dataset plots"
```

### Task 6: Formal Configurations, Remote-Only Summary Export, and Chinese Documentation

**Files:**
- Create: `Uncertainty_Quantification/LLPR/llpr/export.py`
- Create: `Uncertainty_Quantification/LLPR/tests/test_export.py`
- Create: `Uncertainty_Quantification/LLPR/configs/gpu_mad_test_fixed.yaml`
- Create: `Uncertainty_Quantification/LLPR/configs/gpu_matpes_train_fixed.yaml`
- Create: `Uncertainty_Quantification/LLPR/configs/plot_three_datasets.yaml`
- Modify: `Uncertainty_Quantification/LLPR/README.md`

**Interfaces:**
- Produces: `export_evaluation_summary(evaluation_dir: Path, destination: Path) -> Path`.
- Produces: a locally verifiable summary bundle containing `summary.json`, `preview.json`, and `manifest.json`, without `details.npz`.

- [ ] **Step 1: Write a failing summary-export test**

Create a verified fake evaluation, export it, and assert the destination contains exactly the three summary files, records the source evaluation identity and `details.npz` SHA, declares only the copied files, omits absolute paths, and rejects a conflicting existing destination.

- [ ] **Step 2: Run the export test and observe the missing module failure**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_export.py -q`

- [ ] **Step 3: Implement atomic summary export**

Verify the complete source evaluation first, copy only `summary.json` and `preview.json` into staging, write a stage `evaluation-summary` manifest with their hashes plus the evaluation identity and remote `details.npz` hash, verify the bundle, then atomically publish it.

- [ ] **Step 4: Add real formal YAML configurations**

MAD config uses source curvature `outputs/matpes_r2/curvature/981cc8bdf7f8b820`, validation SHA `13f381d87dc20c56454ddf49f4958da6654220e904ad37f57cfb9a34593612e3`, test SHA `d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b`, fixed eta `1.0e-6`, and experiment `mad_test`. MATPES train config reuses both named stages and initially binds the existing remote byte SHA `42bc5b908fbd70da740175f824fd87169dcc4bf62258e5096c3eda7e3372eae1`; if semantic audit fails, replace it with the uploaded local SHA `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec` before execution and commit that factual adjustment.

- [ ] **Step 5: Add the three-evaluation plotting YAML**

Use labels `matpes_test`, `mad_test`, and `matpes_train`; pin the known MATPES-test evaluation identity and let the two new run roots resolve their sole complete evaluation. Output directly to `Uncertainty_Quantification/Plots/LLPR` with all approved style values explicit.

- [ ] **Step 6: Rewrite the Chinese README around the publishable workflow**

Document every code/config/result file by purpose, the two reuse modes, exact commands, small-data validation, formal remote execution, plot filenames, local/remote retention boundary, and the fact that MATPES-train full details remain in the isolated remote workspace. Do not describe this as a legacy migration and do not include private key material.

- [ ] **Step 7: Run export, config, and documentation checks**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_export.py Uncertainty_Quantification/LLPR/tests/test_config.py -q`

Run: `python -m Uncertainty_Quantification.LLPR.llpr --help`

Expected: configs load strictly, export verifies, and documentation commands match the real CLI.

- [ ] **Step 8: Commit configs and documentation**

```bash
git add Uncertainty_Quantification/LLPR/llpr/export.py Uncertainty_Quantification/LLPR/tests/test_export.py Uncertainty_Quantification/LLPR/configs/gpu_mad_test_fixed.yaml Uncertainty_Quantification/LLPR/configs/gpu_matpes_train_fixed.yaml Uncertainty_Quantification/LLPR/configs/plot_three_datasets.yaml Uncertainty_Quantification/LLPR/README.md
git commit -m "docs(llpr): add formal reuse and plotting workflows"
```

### Task 7: Local Full Verification and Reuse Smoke Chain

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/tests/test_n20.py`
- Modify only if required by a reproduced failure: the source file responsible for that failure

**Interfaces:**
- Validates all code paths before remote deployment.

- [ ] **Step 1: Add a real n20 reuse-chain integration test**

Build/calibrate/evaluate one n20 base experiment, then create a second experiment that reuses the base curvature and calibration. Assert the second run has materialized verified stages, its predictions and UQ arrays equal the base evaluation, and patched numerical build/calibration functions are never reached.

- [ ] **Step 2: Run the integration test once red and once green**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests/test_n20.py -k reuse -q`

Expected before final wiring: failure at the first missing or incorrect reuse behavior. Expected after correction: pass.

- [ ] **Step 3: Run the complete local LLPR suite**

Run: `python -m pytest Uncertainty_Quantification/LLPR/tests -q`

Expected: all non-environment-skipped tests pass with no new warnings.

- [ ] **Step 4: Run format, lint, type, and repository checks**

```bash
python -m ruff format --check Uncertainty_Quantification/LLPR
python -m ruff check Uncertainty_Quantification/LLPR
python -m mypy Uncertainty_Quantification/LLPR/llpr
git diff --check
```

Also run `tox -e lint` if the environment has the repository lint dependencies; report any pre-existing scope-external failure separately.

- [ ] **Step 5: Commit the verified integration coverage**

```bash
git add Uncertainty_Quantification/LLPR/tests/test_n20.py
git commit -m "test(llpr): cover artifact reuse end to end"
```

### Task 8: Safe Remote Audit, Small-Data Test, Formal Computation, and Result Return

**Files:**
- Runtime only: `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_llpr_codex/run/*.sbatch`
- Generate locally: `Uncertainty_Quantification/LLPR/outputs/mad_test/` (Git-ignored formal result)
- Generate locally: `Uncertainty_Quantification/LLPR/summaries/matpes_train/`
- Generate locally: `Uncertainty_Quantification/Plots/LLPR/`
- Modify after measured results: `Uncertainty_Quantification/LLPR/README.md`

**Interfaces:**
- Consumes: the exact committed `Plots` SHA and trusted SSH identity.
- Produces: verified remote `mad_test`, `matpes_train`, and plots; approved local subset without MATPES-train `details.npz`.

- [ ] **Step 1: Record immutable pre-run evidence**

Record the local commit SHA, `git status --short`, all 11 `matpes_r2` hashes, checkpoint hash, local dataset hashes, and free space. Abort if tracked files are dirty or any base hash changed.

- [ ] **Step 2: Deploy without touching the existing remote checkout**

Create only `upet_new_llpr_codex`, transfer the committed tree, the base `matpes_r2`, both MAD compatible datasets, and the required checkpoint or a verified link. Record deployed commit SHA and verify every transferred formal input hash.

- [ ] **Step 3: Compute and compare complete MATPES semantic fingerprints**

Run the fingerprint module over local `mace_new/data/dataset/matpes_train.extxyz` and remote `/code/upet/matpes_train.extxyz`. Compare digest, structure count `348780`, atom count, force-component count, and quantum exactly. If equal, link the existing remote file into the isolated data directory. If unequal, stop this path, upload the local file, change the formal config to its byte SHA, rerun config tests, and commit the factual config change before continuing.

- [ ] **Step 4: Submit a Slurm hardware and import probe**

On partition `ai`, request one node/task, 8 CPUs, 64 GiB, set `CUDA_VISIBLE_DEVICES=0`, activate `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new`, run `nvidia-smi`, assert `torch.cuda.is_available()`, import LLPR, and verify the checkpoint SHA. Do not submit model work unless this exits zero.

- [ ] **Step 5: Run the real small-data full-chain test**

Create temporary complete-structure MAD validation/test subsets and use remote `matpes_n20.extxyz`. Run MAD curvature reuse plus new fixed-eta Alpha plus UQ, and MATPES curvature/calibration reuse plus UQ. Run full manifest/NPZ verification, check finite positive Alpha/variance/std, exercise resume from a stopped shard, and generate a temporary one-or-more-evaluation plot. Delete the temporary subsets, test outputs, test plots, staging, and completed shards after all assertions pass; retain only remote test logs until final acceptance.

- [ ] **Step 6: Submit formal jobs with `afterok` dependencies**

Submit `MAD Alpha + MAD UQ`, then `MATPES train UQ`, then unified plot/full verification. Poll Slurm and logs without launching duplicate jobs. On failure, validate and resume the recorded progress/shards rather than removing them or starting a second output tree.

- [ ] **Step 7: Verify formal numerical contracts remotely**

Run full verification on all three result roots. Assert exact curvature/calibration identities, fixed eta, exact reused MATPES Alpha, positive finite MAD Alpha, correct dataset structure/atom/component counts, finite arrays, positive variance/std, consistent offsets/order/lengths, exact 14 plot files, shared axis limits, CSV/manifest count agreement, and all declared hashes.

- [ ] **Step 8: Export and return only the approved local result set**

Export the MATPES-train summary bundle remotely. Copy back complete `mad_test`, the three-file MATPES-train summary, and all 14 plots. Do not transfer MATPES-train `details.npz`. Verify hashes again locally and confirm the remote complete file still exists and passes full verification.

- [ ] **Step 9: Visually inspect all six PNGs**

Open each PNG at original resolution and check readable labels, correct units/titles, square shared axes, visible `y=x`, contours/scatter, non-clipped annotations, and consistent energy/force ranges. If rendering changes are needed, first add or adjust a test, rerun red-green, regenerate all plots, and reverify the manifest.

- [ ] **Step 10: Remove only temporary test material and update measured documentation**

Delete the remote small-test outputs/subsets, temporary staging, completed formal shards, and unneeded caches. Preserve formal remote results and Slurm formal logs. Update README with actual new identities, counts, Alpha values, plot statistics location, and exact remote retention directory.

- [ ] **Step 11: Run final fresh verification before the result commit**

```bash
python -m pytest Uncertainty_Quantification/LLPR/tests -q
python -m ruff format --check Uncertainty_Quantification/LLPR
python -m ruff check Uncertainty_Quantification/LLPR
python -m mypy Uncertainty_Quantification/LLPR/llpr
python -m Uncertainty_Quantification.LLPR.llpr verify --config Uncertainty_Quantification/LLPR/outputs/mad_test
git diff --check
git status --short
```

Separately verify `matpes_r2` against the pre-run 11-file hash record and verify `Plots/LLPR/plotting_manifest.json` plus the MATPES summary manifest.

- [ ] **Step 12: Commit the publishable documentation and returned lightweight artifacts**

```bash
git add Uncertainty_Quantification/LLPR/README.md Uncertainty_Quantification/LLPR/summaries/matpes_train Uncertainty_Quantification/Plots/LLPR
git commit -m "feat(llpr): publish three-dataset uncertainty plots"
```

Do not force-add Git-ignored full output directories. Finish with a clean tracked worktree apart from the pre-existing untracked `.idea/`, and report every commit SHA plus the remote full-result path.
