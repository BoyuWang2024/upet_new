# UPET BootStrapping 整理与结果适配 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在当前分支实现可发布的 UPET BootStrapping 工作流，远端无重训练地标准化三组既有 checkpoint/raw prediction，并按 carnet_new 定义重算 UQ。

**Architecture:** 正式 `bootstrap` 包负责严格配置、采样、最后层训练、checkpoint、prediction、UQ、artifact 与验证；`internal_migration` 单向调用正式 writer/validator，将旧 chunks 写成同一标准 schema。所有正式代码本地开发并提交，所有测试、CPU 小数据计算、全量适配与 UQ 均在远端 `upet_new` 环境执行。

**Tech Stack:** Python 3.11、PyTorch、NumPy、ASE、metatrain/metatomic、PyYAML、pytest、Ruff、mypy。

## Global Constraints

- 直接使用当前本地分支；不创建 worktree，不触碰用户现有 ConfidenceHead 修改与 `.idea`。
- 本地只编辑与提交代码，不运行代码测试、训练、推理、结果适配或 UQ。
- 远端仓库为 `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new`；远端已有 ConfidenceHead 工作区修改，不切换、不提交、不覆盖。
- 远端测试使用隔离同步目录；正式 outputs 写到远端 `Uncertainty_Quantification/BootStrapping/outputs` 并由 `.gitignore` 排除。
- 远端 checkpoint 固定读取 `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/pet-omatpes-l-v0.1.0.ckpt`。
- 远端数据读取 `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_train.extxyz`、`matpes_val.extxyz`、`matpes_test.extxyz`；CPU smoke 使用 `upet_new/data/dataset/matpes_n20.extxyz` 或远端等价小数据。
- 三组运行全部处理：`full_remote_b8_e8`、`lr_1e-4`、`lr_1e-6`。
- `full_remote_b8_e8` 的 training/prediction batch 默认 64/64；两个 LR 运行默认 16/8；解析器接受任意正整数。
- 迁移 `best.pt`、`final.pt`、`latest.pt` 与 val/test raw prediction；不迁移旧 UQ、evaluation、plots、indices、logs 或 W&B。
- 既有全量 checkpoint 不重新训练，既有全量 prediction 不重新推理。
- 新 UQ 使用 sample STD (`ddof=1`) 与 distinct unordered-pair GMD。
- 数据路径由配置提供，不使用数据集 SHA 构造身份；artifact SHA 只用于文件完整性。
- `internal_migration` 留在开发仓库但不得进入发布包或正式公共依赖图。
- 每个任务遵循 TDD：先写失败测试，同步到远端确认失败，再实现并在远端确认通过，最后仅提交该任务文件。

---

## File Map

### 正式包

- `Uncertainty_Quantification/BootStrapping/bootstrap/errors.py`：领域异常。
- `.../bootstrap/config.py`：严格 YAML dataclass 配置。
- `.../bootstrap/artifacts.py`：安全路径、SHA、原子 no-clobber 写入和 sibling staging。
- `.../bootstrap/manifests.py`：run/member/prediction/UQ manifest schema 与审计。
- `.../bootstrap/data.py`：extxyz layout、targets 与训练数据适配。
- `.../bootstrap/sampling.py`：确定性 bootstrap seeds、indices、OOB 与 occurrence view。
- `.../bootstrap/head_policy.py`：PET 最后层选择与冻结策略。
- `.../bootstrap/checkpoint.py`：best/final/latest payload、raw/EMA 审计与分支加载。
- `.../bootstrap/members.py`：成员状态、resume decision 与 member store。
- `.../bootstrap/training.py`：成员训练和 ensemble orchestration。
- `.../bootstrap/prediction.py`：标准 PredictionArrays、store 与正式推理。
- `.../bootstrap/uncertainty.py`：streaming sample STD、pairwise GMD 与 UQ 发布。
- `.../bootstrap/evaluation.py`：正式但本轮不执行的 residual/UQ 指标接口。
- `.../bootstrap/preflight.py`：stage-specific preflight。
- `.../bootstrap/validation.py`：标准结果只读验证。
- `.../scripts/*.py`：薄 CLI。

### 内部适配

- `.../internal_migration/migration/legacy_reader.py`：旧目录只读解析。
- `.../internal_migration/migration/normalization.py`：chunk 到标准数组的流式规范化。
- `.../internal_migration/migration/converter.py`：staging、正式 writer、validator 与外部 audit 编排。
- `.../internal_migration/scripts/*.py`：inspect/migrate/validate/audit CLI。

### 测试与配置

- `.../tests/`：正式包测试。
- `.../internal_migration/tests/`：内部适配与 no-compute 测试。
- `.../configs/*.yaml`：三组远端正式配置和一个 CPU smoke 配置。
- `.../outputs/.gitignore`：忽略所有生成物，仅跟踪自身。
- `.../README.md`：正式用户文档，不依赖旧仓库。

---

### Task 1: 包骨架、严格配置与 CLI 约定

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/__init__.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/__init__.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/errors.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/config.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/__init__.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/_cli.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/__init__.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_config.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_scripts.py`

**Interfaces:**
- Produces: `HardFailure`, `BootstrapConfig`, `load_config(path)`, `run_stage(parser, argv, stage)`。
- Consumes: PyYAML、dataclasses、pathlib。

- [ ] **Step 1: 写失败配置测试**

```python
def test_load_config_resolves_paths_and_keeps_batch_configurable(tmp_path):
    path = write_minimal_config(tmp_path, training_batch=7, prediction_batch=5)
    config = load_config(path)
    assert config.training.batch_size == 7
    assert config.prediction.batch_size == 5
    assert config.data.train.is_absolute()

def test_load_config_rejects_unknown_key(tmp_path):
    path = write_minimal_config(tmp_path, extra={"unexpected": True})
    with pytest.raises(HardFailure, match="unknown key"):
        load_config(path)
```

- [ ] **Step 2: 远端运行失败测试**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests/test_config.py Uncertainty_Quantification/BootStrapping/tests/test_scripts.py -q`

Expected: FAIL，因为 `bootstrap.config` 与 CLI 尚不存在。

- [ ] **Step 3: 实现严格 dataclass 配置**

```python
@dataclass(frozen=True)
class BootstrapConfig:
    schema_version: int
    experiment: ExperimentConfig
    checkpoint: CheckpointConfig
    data: DataConfig
    bootstrap: BootstrapSettings
    training: TrainingConfig
    prediction: PredictionConfig
    uncertainty: UncertaintyConfig
    source_path: Path

def load_config(path: str | Path) -> BootstrapConfig:
    """Load one strict schema_version=1 YAML and resolve relative paths."""
```

只允许 `${UPET_BOOTSTRAP_*}` 白名单环境变量；正整数、parameter mode、split、单位和 `ddof=1` 必须显式验证。

- [ ] **Step 4: 实现 CLI 错误转换并远端复测**

```python
def run_stage(parser, argv, stage):
    try:
        args = parser.parse_args(argv)
        stage(load_config(args.config))
    except HardFailure as error:
        print(str(error), file=sys.stderr)
        return 2
    return 0
```

Expected: 两个测试文件 PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping
git commit -m "feat: add strict bootstrap configuration"
```

### Task 2: Artifact 安全、目录布局与 manifests

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/artifacts.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/manifests.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_artifacts.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_manifests.py`

**Interfaces:**
- Consumes: `HardFailure`, `BootstrapConfig`。
- Produces: `ExperimentLayout`, `sha256_file`, `atomic_write_json`, `atomic_write_yaml`, `atomic_write_npz`, `copy_file_exact`, `sibling_staging`, `build_run_manifest`, `validate_manifest`。

- [ ] **Step 1: 写失败的 no-clobber/path 测试**

```python
def test_atomic_write_rejects_different_existing_payload(tmp_path):
    target = tmp_path / "artifact.json"
    atomic_write_json(target, {"value": 1})
    with pytest.raises(HardFailure, match="already exists"):
        atomic_write_json(target, {"value": 2})

def test_layout_rejects_symlink_escape(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (tmp_path / "root").symlink_to(outside, target_is_directory=True)
    with pytest.raises(HardFailure, match="symlink"):
        ExperimentLayout(tmp_path / "root")
```

- [ ] **Step 2: 远端确认失败**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests/test_artifacts.py Uncertainty_Quantification/BootStrapping/tests/test_manifests.py -q`

Expected: FAIL，因为 artifact 接口不存在。

- [ ] **Step 3: 实现安全 artifact primitives**

```python
@dataclass(frozen=True)
class ExperimentLayout:
    root: Path
    def member_dir(self, member_index: int) -> Path: ...
    def split_dir(self, split: str) -> Path: ...

@contextmanager
def sibling_staging(destination: str | Path) -> Iterator[Path]:
    """Yield same-filesystem staging and publish with no-replace rename."""
```

写入后必须 flush/fsync/reload；存在相同字节时幂等返回，存在不同字节时硬失败。

- [ ] **Step 4: 实现 manifest 递归审计并远端复测**

```python
def validate_manifest(path: str | Path, *, expected_schema: str) -> dict[str, object]:
    """Reject escapes, missing artifacts, non-regular files and SHA drift."""
```

Expected: artifact/manifests 测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat: add immutable bootstrap artifacts"
```

### Task 3: 数据布局、确定性采样与最后层策略

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/data.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/sampling.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/head_policy.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_data.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_sampling.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_head_policy.py`

**Interfaces:**
- Consumes: `BootstrapConfig`, ASE extxyz, PET model named parameters。
- Produces: `DatasetLayout`, `TargetArrays`, `MemberSeeds`, `BootstrapSample`, `OccurrenceDataset`, `apply_pet_last_layer_policy(model)`。

- [ ] **Step 1: 写 deterministic sampling 与 occurrence 失败测试**

```python
def test_member_samples_are_reproducible_and_member_specific():
    a = draw_bootstrap_sample(10, derive_member_seeds(2026, 0).sampling)
    b = draw_bootstrap_sample(10, derive_member_seeds(2026, 0).sampling)
    c = draw_bootstrap_sample(10, derive_member_seeds(2026, 1).sampling)
    assert np.array_equal(a.indices, b.indices)
    assert not np.array_equal(a.indices, c.indices)
    assert np.array_equal(a.oob, np.setdiff1d(np.arange(10), np.unique(a.indices)))

def test_occurrence_ids_are_unique_for_duplicate_source_rows():
    view = OccurrenceDataset(["a", "b"], np.array([0, 0, 1]))
    assert view[0].occurrence_id != view[1].occurrence_id
```

- [ ] **Step 2: 远端确认失败**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests/test_data.py Uncertainty_Quantification/BootStrapping/tests/test_sampling.py Uncertainty_Quantification/BootStrapping/tests/test_head_policy.py -q`

- [ ] **Step 3: 实现 seeds/sample/layout**

```python
@dataclass(frozen=True)
class MemberSeeds:
    sampling: int
    loader: int
    python: int
    torch: int

def derive_member_seeds(base_seed: int, member_index: int) -> MemberSeeds: ...
def draw_bootstrap_sample(dataset_size: int, seed: int, sample_size: int | None = None) -> BootstrapSample: ...
```

使用 `np.random.SeedSequence([base_seed, member_index]).spawn(4)` 与 PCG64，保存 indices/OOB 仅用于新训练。

- [ ] **Step 4: 实现 PET 最后层策略并远端复测**

```python
def apply_pet_last_layer_policy(model: torch.nn.Module) -> TrainablePolicyAudit:
    """Freeze all parameters except node_last_layers/edge_last_layers and require 13,338 parameters."""
```

Expected: 三个测试文件 PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat: add deterministic bootstrap sampling"
```

### Task 4: Checkpoint、成员状态与真实 resume 能力

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/checkpoint.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/members.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_checkpoint.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_members.py`

**Interfaces:**
- Consumes: artifact primitives、13,338 参数 policy。
- Produces: `CheckpointAudit`, `audit_checkpoint`, `load_checkpoint_branch`, `MemberStore`, `ResumeDecision`, `decide_resume`。

- [ ] **Step 1: 写 raw/EMA 与 capability 失败测试**

```python
def test_checkpoint_requires_matching_raw_and_ema_keys(tmp_path):
    path = save_checkpoint(tmp_path, raw={"w": tensor(1)}, ema={"x": tensor(1)})
    with pytest.raises(HardFailure, match="raw and EMA"):
        audit_checkpoint(path)

def test_latest_without_optimizer_is_inference_only(tmp_path):
    audit = audit_checkpoint(save_checkpoint(tmp_path, include_optimizer=False))
    assert audit.inference_ready is True
    assert audit.resume_ready is False
```

- [ ] **Step 2: 远端确认失败**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests/test_checkpoint.py Uncertainty_Quantification/BootStrapping/tests/test_members.py -q`

- [ ] **Step 3: 实现审计和 branch loader**

```python
@dataclass(frozen=True)
class CheckpointAudit:
    path: Path
    sha256: str
    epoch: int
    validation_loss: float | None
    raw_keys: tuple[str, ...]
    ema_keys: tuple[str, ...]
    parameter_count: int
    inference_ready: bool
    resume_ready: bool

def load_checkpoint_branch(path: str | Path, mode: Literal["raw", "ema"]) -> dict[str, Tensor]: ...
```

- [ ] **Step 4: 实现成员状态机并远端复测**

```python
def decide_resume(store: MemberStore) -> ResumeDecision:
    """Return SKIP_VALID, RESUME_LATEST, START_EMPTY, or HARD_FAIL after re-audit."""
```

Expected: checkpoint/member 测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat: add bootstrap checkpoint contracts"
```

### Task 5: 最后层训练工作流

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/training.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_training.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/train.py`

**Interfaces:**
- Consumes: config、sampling、data、head policy、checkpoint/member store、旧实现中已核实的 metatrain loss/transform 逻辑。
- Produces: `train_member(config, member_index) -> MemberTrainingResult`、`train_ensemble(config) -> tuple[MemberTrainingResult, ...]`。

- [ ] **Step 1: 写 best pairing 与 resume 失败测试**

```python
def test_best_uses_raw_validation_and_pairs_same_epoch():
    result = run_fake_training(val_losses=[0.4, 0.2, 0.3])
    assert result.best_epoch == 2
    assert result.best_raw["epoch"].item() == 2
    assert result.best_ema["epoch"].item() == 2

def test_running_member_resumes_optimizer_epoch_and_rng():
    resumed = run_fake_training(interrupt_after_epoch=1, resume=True)
    uninterrupted = run_fake_training(interrupt_after_epoch=None)
    assert_state_equal(resumed.final_raw, uninterrupted.final_raw)
```

- [ ] **Step 2: 远端确认失败**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests/test_training.py -q`

- [ ] **Step 3: 移植并整理正式训练逻辑**

```python
def train_member(config: BootstrapConfig, member_index: int) -> MemberTrainingResult:
    """Train one deterministic PET-last-layer member with paired raw/EMA checkpoints."""

def train_ensemble(config: BootstrapConfig) -> tuple[MemberTrainingResult, ...]:
    return tuple(train_member(config, index) for index in range(config.bootstrap.ensemble_size))
```

保留 occurrence ID、metatrain transforms、checkpoint 恢复 loss、force/virial gradients、strict `<` best selection；移除 2026.1 精确版本门，改为 API/schema 能力检查。

- [ ] **Step 4: 远端运行 fake training 测试**

Expected: PASS；恢复结果与不中断结果相同，valid 成员只读跳过。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/scripts Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat: add bootstrap head training"
```

### Task 6: 标准 Prediction Store 与正式推理

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/prediction.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_prediction.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/predict.py`

**Interfaces:**
- Consumes: config、data targets、checkpoint branch loader、artifacts。
- Produces: `PredictionArrays`, `TargetArrays`, `PredictionStore`, `load_prediction_arrays`, `predict_members`。

- [ ] **Step 1: 写 round-trip/layout 失败测试**

```python
def test_prediction_store_round_trip_and_single_targets(tmp_path):
    store = PredictionStore(tmp_path, split="test", units=UNITS)
    store.write_targets(TARGETS)
    publication = store.write_member(0, "raw", PREDICTIONS)
    assert load_prediction_arrays(publication.path) == PREDICTIONS
    assert len(list((tmp_path / "test").glob("targets.npz"))) == 1

def test_prediction_store_rejects_layout_mismatch(tmp_path):
    with pytest.raises(HardFailure, match="layout"):
        store.write_member(1, "raw", mismatched_predictions())
```

- [ ] **Step 2: 远端确认失败**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests/test_prediction.py -q`

- [ ] **Step 3: 实现 canonical arrays/store**

```python
@dataclass(frozen=True)
class PredictionArrays:
    energy: np.ndarray
    forces: np.ndarray
    stress: np.ndarray

class PredictionStore:
    def write_targets(self, targets: TargetArrays) -> Path: ...
    def write_member(self, member_index: int, mode: str, values: PredictionArrays) -> PredictionPublication: ...
```

- [ ] **Step 4: 实现 chunked inference 并远端复测**

```python
def predict_members(config: BootstrapConfig) -> tuple[PredictionPublication, ...]:
    """Run configured splits/modes from best checkpoints and publish after full validation."""
```

使用磁盘临时数组累计；stress 始终为完整 `[N,3,3]`；raw/EMA 不混合。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/scripts Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat: add bootstrap prediction store"
```

### Task 7: carnet_new UQ 与正式 evaluation 接口

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/uncertainty.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/evaluation.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_uncertainty.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_evaluation.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/compute_uq.py`

**Interfaces:**
- Consumes: standard prediction NPZ 与 targets。
- Produces: `streaming_mean_std`, `pairwise_gmd`, `scalar_rms_reductions`, `compute_uncertainty`, `evaluate_uncertainty`。

- [ ] **Step 1: 写公式失败测试**

```python
def test_sample_std_uses_ddof_one(tmp_path):
    paths = write_member_field(tmp_path, [np.array([1.0]), np.array([3.0])])
    result = streaming_mean_std(paths, "energy")
    assert result.std == pytest.approx(np.array([np.sqrt(2.0)]))

def test_gmd_excludes_self_and_counts_unordered_pairs(tmp_path):
    paths = write_member_field(tmp_path, [np.array([0.0]), np.array([2.0]), np.array([5.0])])
    assert pairwise_gmd(paths, "energy") == pytest.approx(np.array([10.0 / 3.0]))
```

- [ ] **Step 2: 远端确认失败**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests/test_uncertainty.py Uncertainty_Quantification/BootStrapping/tests/test_evaluation.py -q`

- [ ] **Step 3: 移植 carnet_new bounded-memory 公式**

```python
def streaming_mean_std(member_paths: Iterable[Path], field: str) -> StreamingMeanStd: ...
def pairwise_gmd(member_paths: Iterable[Path], field: str) -> np.ndarray: ...
def compute_uncertainty(config: BootstrapConfig) -> tuple[UncertaintyPublication, ...]: ...
```

STD 使用 Welford 与 `B-1`；GMD 遍历 `i<j`；所有计算使用 float64。

- [ ] **Step 4: 实现 evaluation 纯函数并远端复测**

`evaluation.py` 只提供 residual、Pearson/Spearman、coverage/risk 纯函数；本轮不生成全量 evaluation artifact。

Expected: UQ/evaluation 测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/scripts Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat: add carnet bootstrap uncertainty"
```

### Task 8: Preflight、标准 validator 与 CLI 完整链路

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/preflight.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/validation.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_preflight.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_validation.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/preflight.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/validate.py`

**Interfaces:**
- Consumes: 所有正式 manifest、checkpoint、prediction、UQ 接口。
- Produces: `preflight(config, stage) -> PreflightReport`、`validate_run(config) -> ValidationReport`。

- [ ] **Step 1: 写 stage/corruption 失败测试**

```python
def test_preflight_records_versions_without_exact_metatrain_pin(config):
    report = preflight(config, "predict")
    assert report.versions["metatrain"]

def test_validator_rejects_prediction_sha_drift(valid_run):
    corrupt(valid_run / "predictions/test/members/member_000/raw.npz")
    with pytest.raises(HardFailure, match="sha256"):
        validate_run(valid_run.config)
```

- [ ] **Step 2: 远端确认失败**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests/test_preflight.py Uncertainty_Quantification/BootStrapping/tests/test_validation.py -q`

- [ ] **Step 3: 实现 stage-specific preflight 与只读 validator**

```python
def preflight(config: BootstrapConfig, stage: Literal["train", "predict", "uq", "validate"]) -> PreflightReport: ...
def validate_run(config: BootstrapConfig) -> ValidationReport: ...
```

validator 不修复正式结果，只报告 PASS 或抛出 HardFailure。

- [ ] **Step 4: 完成 CLI 与远端全正式测试集**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests -q`

Expected: 全部 PASS，warnings-as-errors。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap Uncertainty_Quantification/BootStrapping/scripts Uncertainty_Quantification/BootStrapping/tests
git commit -m "feat: validate bootstrap workflows"
```

### Task 9: 隔离的 internal migration

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/internal_migration/README.md`
- Create: `Uncertainty_Quantification/BootStrapping/internal_migration/__init__.py`
- Create: `.../internal_migration/migration/__init__.py`
- Create: `.../internal_migration/migration/legacy_reader.py`
- Create: `.../internal_migration/migration/normalization.py`
- Create: `.../internal_migration/migration/converter.py`
- Create: `.../internal_migration/scripts/_cli.py`
- Create: `.../internal_migration/scripts/inspect_results.py`
- Create: `.../internal_migration/scripts/migrate_results.py`
- Create: `.../internal_migration/scripts/validate_migration.py`
- Create: `.../internal_migration/scripts/audit_results.py`
- Create: `.../internal_migration/tests/`

**Interfaces:**
- Consumes: 正式 config、artifact writer、checkpoint audit、PredictionStore、validate_run。
- Produces: `inspect_legacy_run`, `convert_legacy_run`, `validate_migrated_run`, `audit_migrated_run`。

- [ ] **Step 1: 写 reader/no-compute/path 失败测试**

```python
def test_reader_rejects_missing_or_reordered_chunks(legacy_tree): ...

def test_converter_never_calls_compute(monkeypatch, legacy_tree):
    monkeypatch.setattr(torch.nn.Module, "forward", forbidden)
    monkeypatch.setattr(torch.Tensor, "backward", forbidden)
    monkeypatch.setattr(training, "train_ensemble", forbidden)
    monkeypatch.setattr(prediction, "predict_members", forbidden)
    monkeypatch.setattr(uncertainty, "compute_uncertainty", forbidden)
    convert_legacy_run(legacy_tree.source, legacy_tree.destination, legacy_tree.config)
```

- [ ] **Step 2: 远端确认失败**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/internal_migration/tests -q`

- [ ] **Step 3: 实现固定旧 schema reader 与流式 normalization**

```python
def inspect_legacy_run(source: Path, config: BootstrapConfig) -> LegacyRunAudit: ...
def concatenate_legacy_chunks(chunks: Sequence[LegacyChunk], target: Path) -> PredictionArrays: ...
```

reader 仅接受三个已确认运行的 8 成员、76 chunks/member/split 结构；源路径与目标路径必须分离。

- [ ] **Step 4: 实现 staging converter/audit 并远端复测**

```python
def convert_legacy_run(source: Path, destination: Path, config: BootstrapConfig, audit_root: Path) -> MigrationPublication: ...
```

复制 checkpoint 后逐个 SHA 比较；prediction 与每个旧 chunk 对应切片 `np.array_equal`；正式 validator PASS 后才原子发布。

- [ ] **Step 5: 提交**

```bash
git add Uncertainty_Quantification/BootStrapping/internal_migration
git commit -m "feat: add internal bootstrap result adapter"
```

### Task 10: 配置、README、outputs 与发布边界

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/configs/upet_bootstrap_full_remote_b8_e8.yaml`
- Create: `Uncertainty_Quantification/BootStrapping/configs/upet_bootstrap_lr_1e-4.yaml`
- Create: `Uncertainty_Quantification/BootStrapping/configs/upet_bootstrap_lr_1e-6.yaml`
- Create: `Uncertainty_Quantification/BootStrapping/configs/upet_bootstrap_n20_cpu.yaml`
- Create: `Uncertainty_Quantification/BootStrapping/outputs/.gitignore`
- Create: `Uncertainty_Quantification/BootStrapping/README.md`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_publication.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: 完整 CLI/config schema。
- Produces: 可运行配置、公开说明和发布排除测试。

- [ ] **Step 1: 写发布边界失败测试**

```python
def test_formal_package_never_imports_internal_migration():
    for path in formal_python_files():
        assert "internal_migration" not in path.read_text()

def test_release_inventory_excludes_internal_and_outputs():
    inventory = build_release_inventory(BOOTSTRAP_ROOT)
    assert not any("internal_migration" in item for item in inventory)
    assert not any("outputs" in item for item in inventory)
```

- [ ] **Step 2: 远端确认失败**

Run: `python -m pytest Uncertainty_Quantification/BootStrapping/tests/test_publication.py -q`

- [ ] **Step 3: 写四份配置与公开 README**

正式三份配置使用用户给定的远端 checkpoint/data 路径；README 只描述标准 train/predict/UQ/validate，不要求旧仓库存在。

- [ ] **Step 4: 加入 ignore 与发布 inventory 并远端复测**

`outputs/.gitignore` 内容固定为：

```gitignore
*
!.gitignore
```

Expected: publication 测试 PASS。

- [ ] **Step 5: 提交**

```bash
git add .gitignore Uncertainty_Quantification/BootStrapping
git commit -m "docs: publish bootstrap workflow"
```

### Task 11: 远端完整静态与单元验证

**Files:**
- Modify only if failures prove necessary: BootStrapping code/tests from Tasks 1-10。

**Interfaces:**
- Consumes: 当前完整实现。
- Produces: 远端测试报告，不产生 Git 跟踪 outputs。

- [ ] **Step 1: 将当前 BootStrapping 工作树同步到远端隔离测试区**

只同步 `Uncertainty_Quantification/BootStrapping` 与必要配置，不覆盖远端 ConfidenceHead 文件；远端测试路径固定为 `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_bootstrap_codex`。

- [ ] **Step 2: 在远端激活环境并运行格式/静态检查**

Run:

```bash
conda activate upet_new
ruff format --check Uncertainty_Quantification/BootStrapping
ruff check Uncertainty_Quantification/BootStrapping
mypy Uncertainty_Quantification/BootStrapping/bootstrap
```

Expected: 全部退出 0。

- [ ] **Step 3: 运行正式与内部测试**

```bash
python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests -q
python -m pytest -W error Uncertainty_Quantification/BootStrapping/internal_migration/tests -q
```

Expected: 全部 PASS。

- [ ] **Step 4: 对失败执行最小修复并重复完整验证**

每个修复先新增或收紧复现测试；不得通过宽泛 warning ignore 或降低 schema 验证使测试变绿。

- [ ] **Step 5: 提交验证修复**

```bash
git add Uncertainty_Quantification/BootStrapping
git commit -m "test: harden bootstrap workflows"
```

### Task 12: 远端 CPU 小数据原生/适配等价验证

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/internal_migration/tests/test_schema_equivalence.py`
- Generated remote only: `Uncertainty_Quantification/BootStrapping/outputs/n20_*`

**Interfaces:**
- Consumes: 正式 train/predict/UQ/validate 和 internal converter。
- Produces: 原生与适配 schema signature 比较报告。

- [ ] **Step 1: 写 schema signature 失败测试**

```python
def test_native_and_migrated_outputs_share_schema(native_result, migrated_result):
    assert schema_signature(native_result) == schema_signature(migrated_result)
    assert validate_result(native_result).status == "PASS"
    assert validate_result(migrated_result).status == "PASS"
```

- [ ] **Step 2: 远端 CPU 运行 B=2、1 epoch 原生流程**

Run: preflight -> train -> predict raw val/test -> compute UQ -> validate，设备全部为 CPU。

Expected: 2 个成员均产生 best/final/latest，原生结果 PASS。

- [ ] **Step 3: 远端构建小型旧格式 fixture 并适配**

Run: inspect -> migrate -> validate -> audit；no-compute guard 保持启用。

Expected: 适配结果 PASS。

- [ ] **Step 4: 比较 schema 而非独立 NPZ 容器 SHA**

比较目录、manifest keys、array names、dtype category、rank、loader return type 和 validator 状态；同源 prediction 数值使用 `np.array_equal`。

- [ ] **Step 5: 提交测试与必要修复**

```bash
git add Uncertainty_Quantification/BootStrapping
git commit -m "test: verify bootstrap schema equivalence"
```

### Task 13: 三组远端全量 checkpoint/prediction 适配

**Files:**
- Generated remote only: `Uncertainty_Quantification/BootStrapping/outputs/upet-bootstrap-head-posttrain-v1/*`
- Generated remote only: external migration audit directory。

**Interfaces:**
- Consumes: 旧结果根、三份正式配置、internal migration CLI。
- Produces: 3 run directories、72 checkpoint、48 raw prediction NPZ、6 targets NPZ。

- [ ] **Step 1: 对三个源运行执行只读 inspect**

源根：`/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/Ensemble/results/BootStrapping/upet-bootstrap-head-posttrain-v1`。

Expected: 每个运行 8 valid members、24 checkpoint、val/test 各 608 chunks。

- [ ] **Step 2: 逐运行执行 migrate**

顺序：`full_remote_b8_e8` -> `lr_1e-4` -> `lr_1e-6`。每个运行独立 staging；一个失败时停止该运行，不删除 staging，不影响已发布运行。

- [ ] **Step 3: 对每个运行执行 validate 与 audit**

Expected:

- 72 checkpoint SHA 与源逐一相同；
- 3,648 chunks 与 48 NPZ 对应切片 `np.array_equal`；
- val/test counts 为 19,370/19,374；
- atoms 为 152,959/149,321；
- 所有正式 validator PASS；
- 源关键 SHA 未变化。

- [ ] **Step 4: 检查目标不含禁迁内容**

Expected: 无旧 UQ、evaluation、plots、W&B、training logs 或 bootstrap indices。

- [ ] **Step 5: 保存远端 audit，不提交 outputs**

确认 `git status --short Uncertainty_Quantification/BootStrapping/outputs` 为空。

### Task 14: 三组远端全量 UQ 重算

**Files:**
- Generated remote only: six `uncertainty/<split>/raw` result sets。

**Interfaces:**
- Consumes: Task 13 的 48 standard raw prediction NPZ。
- Produces: 六组 sample-STD/distinct-GMD UQ。

- [ ] **Step 1: 逐运行、逐 split 执行正式 compute_uq**

组合为 3 runs x val/test，parameter mode 固定 raw。

- [ ] **Step 2: 用独立 NumPy 公式全量交叉验证**

Expected:

- mean：`rtol=1e-13, atol=1e-15`；
- STD/GMD：`rtol=1e-12, atol=1e-14`；
- 所有输出 finite。

- [ ] **Step 3: 对共同字段做旧 UQ 只读 sanity check**

Expected: `new_std ~= old_std * sqrt(8/7)`，`new_gmd ~= old_gmd * 8/7`；该检查不复制旧 UQ。

- [ ] **Step 4: 重新运行正式 validator 与 audit**

Expected: 三个 run 均 PASS，六组 UQ manifest 完整。

- [ ] **Step 5: 确认 outputs 仍被 Git 忽略**

Run: `git status --short --ignored Uncertainty_Quantification/BootStrapping/outputs`

Expected: 仅显示 ignored 状态，无 staged/untracked 结果文件。

### Task 15: 最终远端验证、代码审查与收尾提交

**Files:**
- Modify only when final evidence exposes a defect。
- Update: `Uncertainty_Quantification/BootStrapping/README.md` only if actual commands differ from documented commands。

**Interfaces:**
- Consumes: 全部代码与远端结果。
- Produces: 可发布提交序列和最终验证证据。

- [ ] **Step 1: 运行最终远端验证矩阵**

```bash
ruff format --check Uncertainty_Quantification/BootStrapping
ruff check Uncertainty_Quantification/BootStrapping
mypy Uncertainty_Quantification/BootStrapping/bootstrap
python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests -q
python -m pytest -W error Uncertainty_Quantification/BootStrapping/internal_migration/tests -q
```

Expected: 全部退出 0。

- [ ] **Step 2: 验证发布边界**

构建发布 inventory 并确认无 `internal_migration`、outputs、checkpoint、prediction、UQ、logs、W&B、plots 或 `__pycache__`。

- [ ] **Step 3: 检查 Git diff 和用户原有改动**

Run: `git status --short`、`git diff --check`、`git log --oneline`。

Expected: 仅保留用户既有 ConfidenceHead/.idea 工作树改动；BootStrapping 实现均已提交。

- [ ] **Step 4: 执行最终代码审查并修复发现的问题**

逐项对照设计文档 16 个章节，特别检查 no-compute、no-clobber、data-path configuration、raw-only migration 和 UQ formulas。

- [ ] **Step 5: 提交最终修复**

```bash
git add Uncertainty_Quantification/BootStrapping
git commit -m "fix: finalize bootstrap result adaptation"
```

如果没有最终修复，不创建空提交；以最后一个有内容的任务提交作为完成提交。
