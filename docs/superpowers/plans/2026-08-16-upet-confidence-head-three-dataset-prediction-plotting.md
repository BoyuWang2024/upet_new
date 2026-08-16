# UPET ConfidenceHead Three-Dataset Prediction and Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不重新训练、不覆盖现有 evaluation 的前提下，复用 MATPES cache 与九个 ConfidenceHead `best.pt`，完成 MATPES test、MAD test、MATPES train 的 prediction/UQ 和 Carnet 风格绘图。

**Architecture:** 新增独立的 strict external-prediction config 和 dataset-scoped artifact 流程；MATPES test 适配现有 evaluation，MATPES train 读取 run 声明的 cache train split，MAD test 使用现有 raw-cache primitives 构建单 split cache。将单目标张量计算提取为纯函数供现有 evaluate 与外部预测共同调用，并扩展 `plot_analysis.py` 读取三个数据集和绘制跨数据集相关性。

**Tech Stack:** Python 3.11、PyTorch、Pydantic v2、ASE、NumPy memmap、Matplotlib、PyYAML、pytest、Slurm。

## Global Constraints

- 当前本地分支固定为 `Plots`；不创建 worktree，不修改或提交用户的 `.idea/`。
- 不重新训练 UPET 或 ConfidenceHead，不修改九个 run 的训练配置、checkpoint、manifest 或现有 evaluation。
- 能量误差固定为逐原子 `abs(E_pred-E_ref)/N`；力固定为每原子三个 Cartesian component 绝对误差均值。
- energy head 与 force head 的 readout 输入严格分离。
- 固定线性 50-bin、原训练 thresholds/representatives 与 order 1–8 保持不变。
- MATPES test 不重新推理；MATPES train 不重新构建基础 UPET cache；只为 MAD test 提取一次基础 UPET 特征。
- 新结果 no-clobber、原子发布、SHA/identity 可验证；推理不初始化 W&B。
- 本地只运行静态检查、单元测试和合成/小数据测试；真实全量推理在远端 Slurm 节点执行。
- 每个 production change 必须先有对应失败测试并观察 RED，再做最小实现并观察 GREEN。

---

## File Structure

**Create**

- `confidence_head/external_config.py`：外部数据源、运行选择和输出的 strict config。
- `confidence_head/external_cache.py`：单数据集 cache 构建、现有 split 解析和 feature compatibility。
- `confidence_head/external_prediction.py`：run/checkpoint 校验、dataset-scoped prediction/UQ 原子发布与验证。
- `confidence_head/external_plotting.py`：三数据集输入适配、跨数据集相关性统计和绘图发布。
- `scripts/predict_external_datasets.py`：预测/UQ 薄 CLI。
- `scripts/plot_prediction_datasets.py`：绘图薄 CLI。
- `configs/predict_external_gpu.yaml`：远端正式配置。
- `configs/predict_external_smoke.yaml`：远端 n20 smoke 配置。
- `run/submit_external_prediction.sh`：Slurm 参数化提交模板，不含九份复制脚本。
- `tests/test_external_config.py`
- `tests/test_external_cache.py`
- `tests/test_external_prediction.py`
- `tests/test_external_plotting.py`
- `tests/test_external_scripts.py`

**Modify**

- `confidence_head/single_target_evaluation.py`：提取可复用的单目标纯推理函数。
- `confidence_head/plot_analysis.py`：允许显式 prediction/manifest 输入和共享箱线图尺度。
- `confidence_head/workflows/commands.py`：增加外部预测和绘图 orchestration wrapper。
- `README.md`：增加中文远端预测/UQ/绘图说明。

---

### Task 1: Strict external-prediction configuration

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py`
- Test: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py`

**Interfaces:**
- Produces: `ExternalPredictionConfig`, `ExistingEvaluationSource`, `CacheSplitSource`, `ExtXYZSource`, `load_external_config(path: Path, repo_root: Path | None = None) -> ExternalPredictionConfig`。
- Later tasks consume `config.datasets`, `config.runs_root`, `config.cache_root`, `config.plots_root`, `config.batch_size`, `config.device`。

- [ ] **Step 1: Write failing config tests**

```python
def test_external_config_loads_three_mutually_exclusive_sources(tmp_path: Path):
    config = load_external_config(_write_valid_config(tmp_path), repo_root=tmp_path)
    assert config.datasets["matpes_test"].source == "existing_evaluation"
    assert config.datasets["matpes_train"].source == "cache_split"
    assert config.datasets["mad_test"].source == "extxyz"

def test_external_config_rejects_unknown_and_mixed_source_fields(tmp_path: Path):
    path = _write_config(tmp_path, mad={"source": "extxyz", "path": "x.xyz", "split": "train"})
    with pytest.raises(ValueError, match="split|Extra inputs"):
        load_external_config(path, repo_root=tmp_path)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py -q`

Expected: collection fails because `confidence_head.external_config` does not exist.

- [ ] **Step 3: Implement strict discriminated config**

```python
class ExistingEvaluationSource(StrictModel):
    source: Literal["existing_evaluation"]
    expected_sha256: Sha256

class CacheSplitSource(StrictModel):
    source: Literal["cache_split"]
    split: str = "train"
    expected_sha256: Sha256

class ExtXYZSource(StrictModel):
    source: Literal["extxyz"]
    path: Path
    expected_sha256: Sha256

DatasetSource = Annotated[
    ExistingEvaluationSource | CacheSplitSource | ExtXYZSource,
    Field(discriminator="source"),
]

class ExternalPredictionConfig(StrictModel):
    checkpoint: CheckpointConfig
    datasets: dict[str, DatasetSource]
    runs_root: Path
    cache_root: Path
    plots_root: Path
    batch_size: int = Field(gt=0)
    device: str
```

Require dataset names to be safe path components and exactly include `matpes_test`, `matpes_train`, `mad_test` for production; resolve relative paths against repo root; reject duplicate YAML keys via the existing `UniqueKeySafeLoader`.

- [ ] **Step 4: Run config tests and existing config tests**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/external_config.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_config.py
git commit -m "feat(confidence-head): configure external dataset prediction"
```

### Task 2: Single-dataset cache and verified cache-split reuse

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_cache.py`
- Test: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_cache.py`

**Interfaces:**
- Consumes: `ExternalPredictionConfig`, existing `build_raw_cache`, `CachedSplitDataset`, `_raw_structures`, `_identity_payload` primitives.
- Produces: `DatasetCache`, `build_external_dataset_cache(config, name) -> DatasetCache`, `resolve_declared_cache_split(run_dir, source) -> DatasetCache`, `feature_compatibility(manifest, split) -> FeatureCompatibility`。

- [ ] **Step 1: Write failing cache-reuse tests**

```python
def test_resolve_cache_split_binds_run_declared_cache_and_dataset_sha(cache_run):
    resolved = resolve_declared_cache_split(cache_run.run_dir, cache_run.source)
    assert resolved.split == "train"
    assert resolved.dataset_sha256 == cache_run.source.expected_sha256

def test_resolve_cache_split_rejects_sha_even_when_shapes_match(cache_run):
    source = cache_run.source.model_copy(update={"expected_sha256": "0" * 64})
    with pytest.raises(ValueError, match="dataset SHA"):
        resolve_declared_cache_split(cache_run.run_dir, source)
```

- [ ] **Step 2: Run cache tests and verify RED**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_cache.py -q`

Expected: import failure for missing `external_cache`.

- [ ] **Step 3: Implement cache compatibility and existing split resolver**

```python
@dataclass(frozen=True)
class FeatureCompatibility:
    checkpoint_sha256: str
    readouts: Mapping[str, str]
    force_dim: int
    energy_dim: int
    dtype: str

@dataclass(frozen=True)
class DatasetCache:
    manifest_path: Path
    cache_id: str
    split: str
    dataset_sha256: str
    compatibility: FeatureCompatibility
```

Resolve the cache path only from the run manifest `cache_id`, confine it below `outputs/cache`, validate the complete cache, and compare `identity_payload.splits[split].sha256` exactly.

- [ ] **Step 4: Add failing external cache build test**

```python
def test_build_external_cache_publishes_one_dataset_split_and_reuses_identity(monkeypatch, external_config):
    monkeypatch.setattr(external_cache, "_raw_structures", lambda **_: iter([RAW]))
    first = build_external_dataset_cache(external_config, "mad_test")
    second = build_external_dataset_cache(external_config, "mad_test")
    assert first == second
    assert json.loads(first.manifest_path.read_text())["splits"].keys() == {"dataset"}
```

- [ ] **Step 5: Run the new test and verify RED**

Expected: fails because `build_external_dataset_cache` is not implemented.

- [ ] **Step 6: Implement single-dataset cache using existing raw-cache writer**

Build identity payload with one `dataset` split, the verified extxyz counts/SHA, checkpoint SHA, readouts, feature dimensions, execution policy and versions. Call:

```python
build_raw_cache(
    output_root=config.cache_root,
    split_structures={"dataset": itertools.chain([first], stream)},
    identity_payload=payload,
    staging=prepare_raw_cache(config.cache_root),
)
```

Do not duplicate memmap writing or checkpoint/readout extraction logic.

- [ ] **Step 7: Run cache suites**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_cache.py Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py Uncertainty_Quantification/ConfidenceHead/tests/test_cache_workflow.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/external_cache.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_cache.py
git commit -m "feat(confidence-head): reuse and build external dataset caches"
```

### Task 3: Reusable single-target inference kernel

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/single_target_evaluation.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`
- Test: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_prediction.py`

**Interfaces:**
- Produces: `collect_single_target_predictions(model, loader, device, config, force_spec, energy_spec, to_device, offsets, include_raw=False) -> SingleTargetResult`。
- `SingleTargetResult` contains `target`, `predictions`, `metrics`, and `bin_summary` inputs; existing evaluate remains byte-schema compatible.

- [ ] **Step 1: Write failing pure-kernel tests**

```python
def test_collect_energy_is_per_atom_and_can_include_raw_values(fake_energy_loader):
    result = collect_single_target_predictions(..., include_raw=True)
    torch.testing.assert_close(result.predictions["energy_observed_errors"], torch.tensor([1.0]))
    assert "energy_prediction" in result.predictions
    assert "force_prediction" not in result.predictions

def test_collect_force_uses_atom_mean_and_only_force_features(fake_force_loader):
    result = collect_single_target_predictions(..., include_raw=True)
    assert result.predictions["force_observed_errors"].shape == (2,)
    assert result.predictions["force_target_mode"] == "atom_mean"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_prediction.py -k collect -q`

Expected: missing function failure.

- [ ] **Step 3: Extract the computation without changing evaluation publication**

Move the loop currently inside `evaluate_single_target` into the pure function. Preserve existing field names and tensor shapes. With `include_raw=True`, include only the active target's prediction/reference; with false, existing `test_predictions.pt` keys remain unchanged.

- [ ] **Step 4: Run workflow and external kernel tests**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_prediction.py -k collect Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py -q`

Expected: PASS and existing evaluation tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/single_target_evaluation.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_prediction.py
git commit -m "refactor(confidence-head): share single-target prediction kernel"
```

### Task 4: Dataset-scoped prediction/UQ publication

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_prediction.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_prediction.py`

**Interfaces:**
- Consumes: `DatasetCache`, `collect_single_target_predictions`, existing run/checkpoint/binning validation.
- Produces: `discover_external_runs(runs_root) -> CompletedRunSet`, `predict_dataset_run(run_dir, dataset_name, cache) -> Path`, `verify_external_prediction(path, full=True) -> dict`, `predict_external_datasets(config, names=()) -> tuple[Path, ...]`。

- [ ] **Step 1: Write failing run discovery and compatibility tests**

```python
def test_discovery_requires_exact_energy_orders_and_one_atom_mean_force(nine_runs):
    runs = discover_external_runs(nine_runs)
    assert set(runs.energy_by_order) == set(range(1, 9))
    assert runs.force.force_target_mode == "atom_mean"

def test_external_cache_id_need_not_equal_training_cache_but_features_must_match(run, cache):
    assert validate_external_compatibility(run, cache).energy_dim == cache.compatibility.energy_dim
    with pytest.raises(ValueError, match="energy feature dimension"):
        validate_external_compatibility(run, replace(cache, compatibility=BAD_DIM))
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_prediction.py -k 'discovery or compatibility' -q`

Expected: missing APIs.

- [ ] **Step 3: Implement discovery, original-run validation and compatibility signature**

Reuse `_declared_artifact`, `_checkpoint_identity`, `_specs`, `ConfidenceModel`, and current completed-run classification rules. Check checkpoint against original run/cache identities, then separately compare external cache compatibility.

- [ ] **Step 4: Write failing atomic publication/no-clobber tests**

```python
def test_predict_run_publishes_dataset_manifest_without_mutating_run_manifest(run, cache):
    before = (run / "manifest.json").read_bytes()
    result = predict_dataset_run(run, "mad_test", cache)
    assert verify_external_prediction(result / "manifest.json", full=True)["status"] == "complete"
    assert (run / "manifest.json").read_bytes() == before

def test_predict_run_refuses_existing_different_identity(run, cache):
    _write_conflicting_target(run / "predictions" / "mad_test")
    with pytest.raises(ValueError, match="identity|refus"):
        predict_dataset_run(run, "mad_test", cache)
```

- [ ] **Step 5: Run and verify RED**

Expected: publication function missing.

- [ ] **Step 6: Implement prediction, metrics, CSV and manifest publication**

Write to `.staging-<uuid>` below `run/predictions`, verify all files, then `os.replace` to `<dataset-name>`. Identity payload includes dataset/cache/run/checkpoint/binning/target semantics. An identical complete target is reused; a conflicting target is never overwritten.

- [ ] **Step 7: Add MATPES-test existing-evaluation adapter test**

```python
def test_existing_evaluation_adapter_does_not_call_prediction(monkeypatch, nine_runs):
    monkeypatch.setattr(external_prediction, "predict_dataset_run", forbidden)
    outputs = predict_external_datasets(CONFIG_WITH_EXISTING_TEST, names=("matpes_test",))
    assert len(outputs) == 9
```

- [ ] **Step 8: Run external prediction and verification tests**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_prediction.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py -q`

Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/external_prediction.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_prediction.py
git commit -m "feat(confidence-head): publish external prediction and uq"
```

### Task 5: Dataset-scoped and cross-dataset plotting

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/external_plotting.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py`
- Test: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_plotting.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_rendering.py`

**Interfaces:**
- Produces: `load_dataset_series(source, dataset_name) -> PlotSeries`, `plot_dataset_suite(dataset, runs, output_root) -> PlotPublication`, `plot_cross_dataset_correlations(publications, output_root) -> Path`, `verify_plot_publication(path, full=True)`。

- [ ] **Step 1: Write failing generic-source and shared-scale tests**

```python
@pytest.mark.parametrize("source_kind", ["evaluation", "external_prediction"])
def test_load_dataset_series_accepts_both_verified_sources(source_kind, source_factory):
    series = load_dataset_series(source_factory(source_kind), "dataset")
    assert series.dataset_name == "dataset"

def test_combined_boxplots_share_nonnegative_symlog_scale(eight_energy_series):
    figure = build_combined_energy_figure(eight_energy_series)
    assert len({axis.get_ylim() for axis in figure.axes}) == 1
    assert all(axis.get_ylim()[0] == 0 for axis in figure.axes)
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_plotting.py -k 'load or shared' -q`

Expected: missing APIs.

- [ ] **Step 3: Generalize plot input and shared scale**

Keep `load_plot_series(run_dir)` backward compatible. Add an explicit verified prediction path/manifest adapter and calculate one shared nonnegative symlog `y_max` for every combined figure.

- [ ] **Step 4: Write failing suite and cross-dataset tests**

```python
def test_dataset_suite_writes_nine_png_pdf_csv_and_comparisons(tmp_path, completed_series):
    publication = plot_dataset_suite("mad_test", completed_series, tmp_path)
    assert publication.single_run_count == 9
    assert verify_plot_publication(publication.manifest, full=True)["status"] == "complete"

def test_cross_dataset_correlations_have_energy_orders_and_force_row(tmp_path, three_publications):
    manifest = plot_cross_dataset_correlations(three_publications, tmp_path)
    rows = list(csv.DictReader((manifest.parent / "cross_dataset_energy_correlations.csv").open()))
    assert {(row["dataset"], int(row["order"])) for row in rows} == {
        (name, order) for name in DATASETS for order in range(1, 9)
    }
```

- [ ] **Step 5: Run and verify RED**

Expected: suite functions missing.

- [ ] **Step 6: Implement atomic PNG/PDF/CSV suites and cross-dataset plots**

Reuse `_draw_boxplot`, `bin_rows`, `energy_correlation`, atomic figure save and current Carnet colors. Add force Pearson/Spearman computation using the same finite/nonconstant rules. Publish one plot manifest containing input and output SHA values.

- [ ] **Step 7: Run all plotting tests**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_plotting.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_rendering.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_orchestration.py -q`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/external_plotting.py Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_plotting.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_rendering.py
git commit -m "feat(confidence-head): plot three external datasets"
```

### Task 6: Commands, scripts, formal configs and Slurm template

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/commands.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/predict_external_datasets.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_prediction_datasets.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/predict_external_gpu.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/predict_external_smoke.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_external_prediction.sh`
- Test: `Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts.py`

**Interfaces:**
- Produces CLI commands in the design and a Slurm template selected by `STAGE={mad-cache,mad-predict,matpes-train-predict,plot}`.

- [ ] **Step 1: Write failing CLI delegation tests**

```python
def test_predict_script_delegates_to_external_workflow(monkeypatch, config_path):
    called = []
    monkeypatch.setattr(commands, "predict_external_from_config", lambda config, names: called.append(names) or ())
    assert predict_main(["--config", str(config_path), "--dataset", "mad_test"]) == 0
    assert called == [("mad_test",)]

def test_plot_script_only_reads_completed_artifacts(monkeypatch, config_path):
    monkeypatch.setattr(commands, "plot_external_from_config", lambda config: Path("manifest.json"))
    assert plot_main(["--config", str(config_path)]) == 0
```

- [ ] **Step 2: Run and verify RED**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts.py -q`

Expected: script imports fail.

- [ ] **Step 3: Implement thin wrappers and configs**

Scripts only parse `--config`, repeatable `--dataset`, and optional explicit run paths, then call workflow functions. The formal config records the three accepted SHAs; the smoke config uses `matpes_n20.extxyz`. Slurm script invokes Python only and contains no duplicated config values.

- [ ] **Step 4: Add static assertions for release boundary**

```python
def test_only_formal_and_smoke_external_configs_are_committed():
    assert sorted(path.name for path in CONFIGS.glob("predict_external*.yaml")) == [
        "predict_external_gpu.yaml", "predict_external_smoke.yaml"
    ]

def test_slurm_template_has_no_training_or_wandb_command():
    text = SUBMIT.read_text()
    assert "train.py" not in text
    assert "wandb" not in text.lower()
```

- [ ] **Step 5: Run scripts/config tests**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts.py Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/commands.py Uncertainty_Quantification/ConfidenceHead/scripts/predict_external_datasets.py Uncertainty_Quantification/ConfidenceHead/scripts/plot_prediction_datasets.py Uncertainty_Quantification/ConfidenceHead/configs/predict_external_gpu.yaml Uncertainty_Quantification/ConfidenceHead/configs/predict_external_smoke.yaml Uncertainty_Quantification/ConfidenceHead/run/submit_external_prediction.sh Uncertainty_Quantification/ConfidenceHead/tests/test_external_scripts.py
git commit -m "feat(confidence-head): add external prediction commands"
```

### Task 7: Documentation and complete local verification

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/README.md`
- Modify: tests only if verification reveals a reproduced defect; every defect first receives a failing regression test.

**Interfaces:**
- Documents exact remote preflight, smoke, Slurm dependency and plotting commands.

- [ ] **Step 1: Add README command and artifact documentation**

Document:

```bash
python Uncertainty_Quantification/ConfidenceHead/scripts/predict_external_datasets.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/predict_external_gpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/plot_prediction_datasets.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/predict_external_gpu.yaml
```

State MATPES reuse behavior, MAD-only cache extraction, energy/force semantics, no W&B, no-clobber and output directories.

- [ ] **Step 2: Run full ConfidenceHead tests**

Run: `python -m pytest Uncertainty_Quantification/ConfidenceHead/tests -q`

Expected: all tests PASS with no warnings.

- [ ] **Step 3: Run repository lint for touched files**

Run: `tox -e lint`

Expected: PASS. If the full environment is unavailable, run the exact ruff/mypy commands from `tox.ini` and report the limitation; do not claim full lint passed.

- [ ] **Step 4: Verify source release boundary and diff**

Run: `git diff --check && git status --short && git diff --stat HEAD~6..HEAD`

Expected: no whitespace errors; only the intended design/plan/source/test/config/doc files and the pre-existing untracked `.idea/` appear.

- [ ] **Step 5: Commit README or verification fixes**

```bash
git add Uncertainty_Quantification/ConfidenceHead/README.md
git commit -m "docs(confidence-head): document external prediction workflow"
```

### Task 8: Remote smoke, deployment and formal Slurm execution

**Files:**
- No new local source files unless a remote failure is first reproduced by a local failing test.
- Remote runtime artifacts remain ignored and uncommitted.

**Interfaces:**
- Consumes the commits from Tasks 1–7.
- Produces remote prediction/UQ manifests and plots only after smoke verification.

- [ ] **Step 1: Push current branch and deploy only task commits**

```bash
git push origin Plots
ssh -p 55801 bywang@121.48.164.204 \
  'cd /home/bywang/code/UQ/upet_new && git fetch origin Plots'
```

On remote `ConfidenceHead`, cherry-pick only the commits created by this plan. Before and after, record `git status --short`; preserve all existing dirty user files.

- [ ] **Step 2: Stage and verify MAD data**

Transfer `data/dataset/mad-test.xyz` to a unique staging filename, verify:

```text
d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b
```

then atomically rename it to the configured remote path. Do not transfer MATPES train because its cache split SHA already matches.

- [ ] **Step 3: Run remote n20 smoke**

Use `/home/bywang/.conda/envs/upet_new/bin/python` and `predict_external_smoke.yaml`; build the small cache, run one energy and one force head, verify artifacts, plot, then rerun to prove identity reuse.

Expected: both repetitions exit 0; second repetition does not invoke feature extraction or overwrite outputs.

- [ ] **Step 4: Submit formal dependency chain**

Submit `mad-cache`, `mad-predict`, `matpes-train-predict`, then the CPU plot job with `afterok` dependencies exactly as specified in the design. Record all Slurm job IDs.

- [ ] **Step 5: Verify formal prediction results**

Run the full verifier for all 18 new prediction directories. Confirm MATPES test still points to nine existing evaluations, MATPES train uses existing cache IDs, and MAD uses the new dataset cache.

- [ ] **Step 6: Verify plots and original run immutability**

Hash the original nine `best.pt`, existing evaluation artifacts and run manifests against the preflight inventory. Validate all plot manifests, PNG/PDF readability, 50-bin CSV counts, energy orders 1–8, and independently recomputed Pearson/Spearman values.

- [ ] **Step 7: Final local commit state check**

Run: `git status --short && git log --oneline --decorate -12`

Expected: only the pre-existing `.idea/` remains untracked locally; all task source changes are committed.
