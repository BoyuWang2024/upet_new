# UPET ConfidenceHead Configurable Workflows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 UPET ConfidenceHead 增加与 `carnet_new` 同格式的完整 YAML、四阶段薄脚本、独立双头超参数、可配置 total-loss EMA 控制和 W&B 记录，并完成本地与远端 n20 CPU 全链路验证。

**Architecture:** 保留现有内容寻址 cache、事务式训练和产物身份验证，把用户可调参数收敛到严格 Pydantic 配置。新增运行命名/产物发现层和 W&B 辅助记录层，脚本只负责加载 YAML 并调用高层工作流；JSONL、checkpoint 和 manifest 仍是权威产物。

**Tech Stack:** Python 3.10+、PyTorch 2.11.0+cu128（目标服务器）、Pydantic 2、PyYAML、pytest、Weights & Biases、UPET/metatrain/metatomic。

## Global Constraints

- 不复制或迁移旧 ConfidenceHead 代码、配置、日志或结果；全部结果由新代码重新计算。
- 力标签必须是逐笛卡尔分量绝对误差；能量标签必须是逐原子绝对误差。
- 力 head 只消费 `force_features`，能量 head 只消费 `energy_features`；两者允许不同输入维度、隐藏层、dropout 和分箱数。
- 固定线性分箱默认 `force_max_error: 0.5`、`energy_max_error: 0.3`。
- best checkpoint、ReduceLROnPlateau 和 early stopping 必须共同监督 `val/total_loss_ema`。
- 监督总 loss 必须为 `force_coefficient * force_loss + energy_coefficient * energy_loss`。
- n20 local/remote 使用 W&B offline；full GPU 默认 W&B online。
- 不实现绘图，不执行 GPU 全数据训练。
- 本地先执行静态/单元测试和 n20 W&B offline 小数据全链路；远端只执行 CPU n20 全链路。
- 所有实现提交到当前 `ConfidenceHead` 分支，推送 GitHub 后由目标服务器拉取。

## File Structure

### Create

- `Uncertainty_Quantification/ConfidenceHead/configs/n20_local_cpu.yaml`：本地 CPU n20 完整配置，仓库相对数据路径，W&B offline。
- `Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml`：目标服务器 CPU n20 完整配置，用户给定绝对路径，W&B offline。
- `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu.yaml`：目标服务器完整数据 GPU 配置，W&B online，仅静态验证。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/run_naming.py`：可读 run name、唯一 cache manifest 和 run directory 解析。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/tracking.py`：W&B 可选初始化、逐 epoch 日志、summary、恢复和容错。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/commands.py`：从一份配置串接现有低层工作流。
- `Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py`：缓存 CLI。
- `Uncertainty_Quantification/ConfidenceHead/scripts/train.py`：训练 CLI。
- `Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py`：评估 CLI。
- `Uncertainty_Quantification/ConfidenceHead/scripts/verify.py`：验证 CLI。
- `Uncertainty_Quantification/ConfidenceHead/tests/test_run_naming.py`：运行命名和 cache/run 发现测试。
- `Uncertainty_Quantification/ConfidenceHead/tests/test_tracking.py`：W&B 生命周期、指标与故障降级测试。
- `Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py`：高层命令与薄脚本测试。
- `Uncertainty_Quantification/ConfidenceHead/README.md`：中文配置字段与四阶段运行说明。

### Modify

- `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`：独立双头、训练 batch、运行命名、恢复与 logging 配置。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py`：力/能量分支使用各自隐藏层、dropout 和 bins。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py`：scheduler 构造函数消费经过验证的配置参数。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`：训练 batch、独立模型参数、可配置 scheduler 与 W&B 生命周期。
- `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py`：从独立双头配置恢复模型。
- `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py`：新 schema 与三份 YAML 契约。
- `Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py`：不同分支结构测试。
- `Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py`：可配置 scheduler 测试。
- `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`：现有夹具迁移、batch、双头、W&B 与 resume 回归。
- `Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt`：加入 `wandb`。

---

### Task 1: 扩展严格配置并交付三份完整 YAML

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/n20_local_cpu.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu.yaml`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py`

**Interfaces:**
- Consumes: 现有 `StrictModel`、`FileIdentityConfig`、`load_config(path, repo_root)`。
- Produces: `BranchModelConfig`、`EnergyModelConfig`、`ModelConfig.force`、`ModelConfig.energy`、`TrainerConfig.batch_size`、`TrainerConfig.resume_from`、`RunConfig.name_prefix`、`LoggingConfig`。

- [ ] **Step 1: 写新 schema 的失败测试**

在 `test_config_identity_artifacts.py` 增加测试，断言独立分支、scheduler 可调、logging、训练 batch 和 resume path 正确解析：

```python
def test_config_exposes_independent_heads_scheduler_batch_and_logging(
    tmp_path: Path,
) -> None:
    raw = _raw_config(tmp_path)
    raw["model"] = {
        "force": {"enabled": True, "hidden_dims": [8], "dropout": 0.1, "num_bins": 31},
        "energy": {
            "enabled": True,
            "hidden_dims": [12, 6],
            "dropout": 0.2,
            "num_bins": 17,
            "cumulant_order": 4,
            "signed_root": False,
        },
    }
    raw["scheduler"] = {
        "name": "reduce_lr_on_plateau",
        "monitor": "val/total_loss_ema",
        "factor": 0.25,
        "patience": 3,
        "threshold": 1e-5,
        "threshold_mode": "abs",
        "cooldown": 2,
        "min_lr": 1e-7,
    }
    raw["trainer"] = {
        "batch_size": 64,
        "max_epochs": 9,
        "ema_beta": 0.9,
        "early_stopping_patience": 4,
        "min_delta": 1e-5,
        "min_epochs": 2,
        "monitor": "val/total_loss_ema",
        "resume_from": str(tmp_path / "outputs/runs/demo/checkpoints/last.pt"),
    }
    raw["run"]["name_prefix"] = "demo"
    raw["logging"] = {
        "jsonl": True,
        "wandb": True,
        "wandb_mode": "offline",
        "wandb_project": "upet-confidence-head",
    }
    config = ConfidenceConfig.model_validate(raw)
    assert config.model.force.hidden_dims == (8,)
    assert config.model.energy.hidden_dims == (12, 6)
    assert config.trainer.batch_size == 64
    assert config.scheduler.factor == 0.25
    assert config.logging.wandb_mode == "offline"
```

另加参数化测试拒绝非法 `name_prefix`、`wandb_mode`、空 hidden dims、`enabled: false`（本次双分支都必须开启）、非正 batch、非 `adamw` optimizer 和非固定 monitor。

- [ ] **Step 2: 运行配置测试并确认旧 schema 无法满足**

Run:

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py -q
```

Expected: 新测试因 `model.force`、`trainer.batch_size`、`run.name_prefix` 或 `logging` 尚不存在而失败。

- [ ] **Step 3: 实现最小严格配置类型**

在 `config.py` 定义并接入：

```python
class BranchModelConfig(StrictModel):
    enabled: Literal[True] = True
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    dropout: float = Field(default=0.0, ge=0, lt=1, allow_inf_nan=False)
    num_bins: int = Field(default=50, ge=3)

class EnergyModelConfig(BranchModelConfig):
    cumulant_order: int = Field(default=3, ge=1, le=8)
    signed_root: bool = True

class ModelConfig(StrictModel):
    force: BranchModelConfig = Field(default_factory=BranchModelConfig)
    energy: EnergyModelConfig = Field(default_factory=EnergyModelConfig)

class LoggingConfig(StrictModel):
    jsonl: Literal[True] = True
    wandb: bool = True
    wandb_mode: Literal["offline", "online"] = "offline"
    wandb_project: str = Field(default="upet-confidence-head", min_length=1)
```

同时把 `OptimizerConfig.lr` 改为 YAML 名称 `learning_rate`，给 `SchedulerConfig` 增加固定 `name`/`monitor` 但移除常数强制校验；给 `TrainerConfig` 增加 `batch_size` 与 `resume_from`；给 `RunConfig` 增加安全格式 `name_prefix`。在 `_resolve_config_paths` 中解析 `trainer.resume_from`（仅当非空）。保留 extra-forbid、frozen、重复键拒绝、路径锚定和 split 身份规则。

- [ ] **Step 4: 创建三份完整 YAML**

按批准设计逐字段写满三份配置。远端文件身份必须逐字使用：

```yaml
checkpoint_sha256: 879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
local_n20_sha256: c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d
remote_n20_sha256: f6516b4066f5c05b0b74061ef8f332c58c397098d575ec6b99fcf177faa15d8b
train_sha256: 42bc5b908fbd70da740175f824fd87169dcc4bf62258e5096c3eda7e3372eae1
validation_sha256: 5f267bbe8a6690196470ee86c04cc7a3ee2839ccbda542a7bff4bea88b9e9078
test_sha256: 76d0c38064af8a904eac1b736b07e2ed952838d4311574d83664a4967ba6ec98
```

YAML 中使用实际字段 `expected_sha256`，上面的标签只用于计划核对。`n20_local_cpu.yaml` 使用仓库相对路径，并将 `run.output_root` 设置为仓库相对路径 `Uncertainty_Quantification/ConfidenceHead/outputs`。`n20_cpu.yaml` 与 `full_gpu.yaml` 使用设计文档第 4 节给出的输入绝对路径，并将 `run.output_root` 设置为 `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new/Uncertainty_Quantification/ConfidenceHead/outputs`。n20 的三个 split 使用 YAML anchor 指向同一对象并设置 `allow_identical_splits: true`。

- [ ] **Step 5: 增加三份配置的契约测试**

```python
@pytest.mark.parametrize(
    ("name", "profile", "device", "wandb_mode"),
    [
        ("n20_local_cpu.yaml", "smoke", "cpu", "offline"),
        ("n20_cpu.yaml", "smoke", "cpu", "offline"),
        ("full_gpu.yaml", "production", "cuda", "online"),
    ],
)
def test_shipped_config_contracts(
    name: str, profile: str, device: str, wandb_mode: str
) -> None:
    config = load_config(CONFIDENCE_ROOT / "configs" / name)
    assert config.profile == profile
    assert config.run.device == device
    assert config.logging.wandb_mode == wandb_mode
    assert config.binning.force_max_error == 0.5
    assert config.binning.energy_max_error == 0.3
    assert config.trainer.monitor == "val/total_loss_ema"
```

- [ ] **Step 6: 运行配置测试并提交**

Run:

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py -q
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py Uncertainty_Quantification/ConfidenceHead/configs Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py
git commit -m "feat: add confidence workflow configurations"
```

Expected: 配置测试全部通过，提交只包含 schema、YAML 和对应测试。

---

### Task 2: 独立双头结构、训练 batch 与可配置 scheduler

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

**Interfaces:**
- Consumes: Task 1 的 `config.model.force`、`config.model.energy`、`config.trainer.batch_size`、`config.scheduler`。
- Produces: `ConfidenceModel(..., force_hidden_dims, energy_hidden_dims, force_dropout, energy_dropout, force_num_bins, energy_num_bins, cumulant_order, signed_root)`；`build_plateau_scheduler(optimizer, *, factor, patience, threshold, threshold_mode, cooldown, min_lr)`。

- [ ] **Step 1: 写独立网络结构与训练 batch 的失败测试**

在 `test_model_math.py` 构造不同分支：

```python
model = ConfidenceModel(
    force_input_dim=2,
    energy_input_dim=3,
    force_hidden_dims=(7,),
    energy_hidden_dims=(11, 5),
    force_dropout=0.1,
    energy_dropout=0.2,
    force_num_bins=13,
    energy_num_bins=17,
    cumulant_order=2,
    signed_root=True,
)
assert model.force_head.network[0].out_features == 7
assert model.energy_head.network[0].out_features == 11
```

在 `test_workflows.py` monkeypatch `_loader` 或检查 DataLoader，断言训练/验证 loader 使用 `config.trainer.batch_size`，而 cache workflow 仍使用 `config.cache.batch_size`。

- [ ] **Step 2: 写 scheduler 参数传递失败测试**

将原先“唯一批准常数”的测试替换为：

```python
def test_plateau_scheduler_uses_validated_configuration() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    scheduler = build_plateau_scheduler(
        optimizer,
        factor=0.25,
        patience=3,
        threshold=1e-5,
        threshold_mode="abs",
        cooldown=2,
        min_lr=1e-7,
    )
    assert scheduler.factor == 0.25
    assert scheduler.patience == 3
    assert scheduler.threshold == 1e-5
    assert scheduler.cooldown == 2
    assert scheduler.min_lrs == [1e-7]
```

- [ ] **Step 3: 运行目标测试并确认失败**

Run:

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py -q
```

Expected: 新构造签名、独立隐藏层、训练 batch 或 scheduler 参数测试失败。

- [ ] **Step 4: 实现独立模型参数并迁移训练/评估构造**

将 `ConfidenceModel` 的共享 `hidden_dims`/`dropout` 移除，分别传给 `ComponentConfidenceHead` 与 `ConfidenceHead`。在训练和评估中统一使用：

```python
model = ConfidenceModel(
    force_input_dim=force_dim,
    energy_input_dim=energy_dim,
    force_hidden_dims=config.model.force.hidden_dims,
    energy_hidden_dims=config.model.energy.hidden_dims,
    force_dropout=config.model.force.dropout,
    energy_dropout=config.model.energy.dropout,
    force_num_bins=config.model.force.num_bins,
    energy_num_bins=config.model.energy.num_bins,
    cumulant_order=config.model.energy.cumulant_order,
    signed_root=config.model.energy.signed_root,
).to(device)
```

分箱规格、`model_loss_id` 和 evaluation 也改读 branch `num_bins`，防止训练与评估身份漂移。

- [ ] **Step 5: 实现训练 batch 与 scheduler 参数传递**

把训练 `_loader` 的 batch size 改为 `config.trainer.batch_size`。`build_plateau_scheduler` 接受显式关键字并原样传给 PyTorch，训练调用读取 `config.scheduler`。保留 `advance_validation_epoch` 只对同一个 EMA 调用一次 `scheduler.step()` 的现有逻辑。

- [ ] **Step 6: 运行回归并提交**

Run:

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py -q
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py
git commit -m "feat: configure independent confidence heads"
```

Expected: 三个测试文件全部通过，既有逐分量力误差、逐原子能量误差和 resume 测试保持绿色。

---

### Task 3: 可读运行命名、唯一缓存发现与配置驱动命令

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/run_naming.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/commands.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_run_naming.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py`

**Interfaces:**
- Consumes: `ConfidenceConfig`、`build_cache`、`train_run`、`evaluate_run`、`verify_run`、完整 cache/run manifest。
- Produces: `build_run_name(config) -> str`、`resolve_unique_cache_manifest(config) -> Path`、`resolve_run_dir(config) -> Path`、`build_cache_from_config(config) -> Path`、`train_from_config(config) -> Path`、`evaluate_from_config(config) -> Path`、`verify_from_config(config) -> dict[str, Any]`。

- [ ] **Step 1: 写运行名确定性与安全字符失败测试**

```python
def test_run_name_encodes_branch_semantics(config: ConfidenceConfig) -> None:
    assert build_run_name(config) == (
        "demo_fixed-linear_f3-fmax0.5-fw1-fmlp4_"
        "e3-emax0.3-ew1.5-emlp4-order2"
    )

def test_run_name_changes_when_energy_head_changes(config: ConfidenceConfig) -> None:
    changed = config.model_copy(
        update={"model": config.model.model_copy(
            update={"energy": config.model.energy.model_copy(update={"hidden_dims": (9,)})}
        )}
    )
    assert build_run_name(changed) != build_run_name(config)
```

- [ ] **Step 2: 写 cache 发现的零个、一个、多个匹配测试**

创建两个测试 manifest，要求函数只接受 `status: complete` 且 checkpoint SHA、三个 split SHA、四个 readout 和执行策略均与配置相同。断言零匹配报 `matching cache manifest was not found`，多匹配报 `multiple matching cache manifests`，恰好一个时返回其解析后路径。

- [ ] **Step 3: 写四个高层命令的失败测试**

用 monkeypatch 替换低层工作流，验证：

```python
def test_train_from_config_passes_derived_inputs(config, monkeypatch) -> None:
    seen: dict[str, object] = {}
    monkeypatch.setattr(commands, "resolve_unique_cache_manifest", lambda value: Path("/cache/manifest.json"))
    monkeypatch.setattr(commands, "train_run", lambda value, **kwargs: seen.update(kwargs) or Path("/run"))
    result = commands.train_from_config(config)
    assert result == Path("/run")
    assert seen["cache_manifest_path"] == Path("/cache/manifest.json")
    assert seen["run_name"] == build_run_name(config)
    assert seen["resume_from"] == config.trainer.resume_from
```

另测 evaluate 默认 best checkpoint、verify 使用 `full=True`，以及 resume checkpoint 必须位于派生 run directory。

- [ ] **Step 4: 运行新测试确认接口尚不存在**

Run:

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_run_naming.py Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py -q
```

Expected: import `run_naming` 或 `workflows.commands` 失败。

- [ ] **Step 5: 实现运行命名与缓存发现**

`build_run_name` 只使用已验证字段并通过统一数字格式去掉无意义尾零。`resolve_unique_cache_manifest` 扫描 `config.run.output_root / "cache" / "*" / "manifest.json"`，读取完整 manifest 后比较输入身份；不得只按目录名猜测。`resolve_run_dir` 返回：

```python
return (config.run.output_root / "runs" / build_run_name(config)).resolve()
```

并验证结果仍位于 `<output_root>/runs` 内。

- [ ] **Step 6: 实现四个配置驱动命令**

`build_cache_from_config` 直接调用 `build_cache` 并返回 manifest；其他三个命令先解析唯一 cache 和派生 run directory，再调用现有低层工作流。保持低层接口可供现有测试和程序调用，不把 CLI 解析混入工作流。

- [ ] **Step 7: 运行测试并提交**

Run:

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_run_naming.py Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py -q
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/run_naming.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/commands.py Uncertainty_Quantification/ConfidenceHead/tests/test_run_naming.py Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py
git commit -m "feat: add config driven workflow commands"
```

Expected: 新测试全部通过，运行目录不覆盖、不逃逸，cache 匹配严格且唯一。

---

### Task 4: W&B offline/online 记录与恢复语义

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/tracking.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_tracking.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt`

**Interfaces:**
- Consumes: Task 1 的 `LoggingConfig`、resolved config、派生 run name、每轮 metric record、resume manifest。
- Produces: `WandbTracker.start(...) -> WandbTracker`、`tracker.log(metrics)`、`tracker.finish(summary, *, status)`、`tracker.run_id: str | None`。

- [ ] **Step 1: 写 fake W&B 生命周期失败测试**

用注入的 fake module 测试 mode/project/name/config、指标、summary 和 finish：

```python
def test_tracker_records_metrics_summary_and_finishes(fake_wandb, config) -> None:
    tracker = WandbTracker.start(
        config.logging,
        run_name="demo",
        resolved_config=config.model_dump(mode="json"),
        resume_id=None,
        wandb_module=fake_wandb,
    )
    tracker.log({"epoch": 0, "val/total_loss_ema": 1.25})
    tracker.finish(
        {"best_epoch": 0, "best_metric": 1.25, "stop_reason": "max_epochs"},
        status="success",
    )
    assert fake_wandb.init_kwargs["mode"] == "offline"
    assert fake_wandb.run.logged == [{"epoch": 0, "val/total_loss_ema": 1.25}]
    assert fake_wandb.run.summary["best_epoch"] == 0
    assert fake_wandb.run.finished is True
```

另测 `resume_id="abc123"` 传递 `id="abc123", resume="allow"`；init/log/finish 分别抛异常时产生 `RuntimeWarning` 且训练侧调用不抛出。

- [ ] **Step 2: 写训练工作流 W&B 指标与 manifest 失败测试**

在 synthetic complete cache 测试中注入 tracker factory，断言每个完成 epoch 恰好一次 `log`，字段集合包含九项训练/验证指标、学习率、epoch、global_step；初次运行 manifest 保存 run ID；恢复运行传回同一 run ID；最终 summary 与 manifest 的 best/stop reason 相同。

- [ ] **Step 3: 运行目标测试确认失败**

Run:

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_tracking.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py -q
```

Expected: `tracking.py` 不存在或训练尚未启动 tracker，新增测试失败。

- [ ] **Step 4: 实现容错 tracker**

`tracking.py` 延迟导入 `wandb`，使未启用时不要求依赖。启用后调用：

```python
run = wandb_module.init(
    project=logging.wandb_project,
    name=run_name,
    mode=logging.wandb_mode,
    config=resolved_config,
    id=resume_id,
    resume="allow" if resume_id is not None else None,
)
```

所有 W&B 边界异常转换为包含阶段名称的 `RuntimeWarning`；`log` 或 `finish` 失败不改变 JSONL、checkpoint 和 manifest 的事务结果。即使训练抛异常，也在 `finally` 路径调用安全 finish。

- [ ] **Step 5: 接入训练、manifest 与 resume**

初次运行在发布 incomplete manifest 前取得 W&B run ID并写入：

```python
"tracking": {
    "wandb_enabled": config.logging.wandb,
    "wandb_mode": config.logging.wandb_mode,
    "wandb_project": config.logging.wandb_project,
    "wandb_run_id": tracker.run_id,
}
```

恢复运行从经过完整性预检的旧 manifest 读取 run ID。每轮先原子更新 JSONL/检查点，再把同一 `record` 发给 W&B，确保 W&B 永远不是权威提交点。训练成功后用 complete manifest 标量结束 tracker。

- [ ] **Step 6: 加入依赖并运行测试**

在 `requirements-remote.txt` 加入受约束依赖 `wandb>=0.19,<1`，然后运行：

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_tracking.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py -q
```

Expected: W&B fake 测试和完整工作流回归全部通过，无网络访问。

- [ ] **Step 7: 提交 W&B 集成**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/tracking.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py Uncertainty_Quantification/ConfidenceHead/tests/test_tracking.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt
git commit -m "feat: track confidence training with wandb"
```

---

### Task 5: 四个薄脚本与中文使用说明

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/train.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/verify.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/README.md`

**Interfaces:**
- Consumes: Task 3 四个 `*_from_config` 函数和 `load_config(Path)`。
- Produces: 每个脚本统一 `main(argv: Sequence[str] | None = None) -> int`，命令行仅要求 `--config PATH`。

- [ ] **Step 1: 写脚本 `--help` 与分派失败测试**

```python
@pytest.mark.parametrize("script", ["build_cache.py", "train.py", "evaluate.py", "verify.py"])
def test_script_help_exposes_only_config(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / script), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--config" in result.stdout

def test_train_script_loads_config_and_dispatches(monkeypatch, config_path) -> None:
    module = _load_script("train.py")
    seen: list[object] = []
    monkeypatch.setattr(module, "train_from_config", lambda config: seen.append(config) or Path("/run"))
    assert module.main(["--config", str(config_path)]) == 0
    assert len(seen) == 1
```

- [ ] **Step 2: 运行测试确认脚本不存在**

Run:

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py -q
```

Expected: 四个脚本路径不存在或无法导入。

- [ ] **Step 3: 实现统一薄脚本模式**

每个脚本使用相同骨架，只替换导入函数：

```python
def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    artifact = train_from_config(load_config(args.config))
    print(artifact)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

脚本将 ConfidenceHead 根目录加入 `sys.path`，保证从仓库根目录和模块目录运行时行为一致。不得在脚本中复制路径发现、训练或验证逻辑。

- [ ] **Step 4: 编写中文 README**

README 必须说明：三份配置用途、数据路径、SHA 校验、配置字段、四阶段命令、输出目录、W&B offline/online、resume 配置示例、n20 相同 split 仅作冒烟测试、full GPU 本次不运行、无绘图步骤。

- [ ] **Step 5: 运行脚本测试、语法检查并提交**

Run:

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py -q
conda run -n upet_new python -m compileall -q Uncertainty_Quantification/ConfidenceHead/confidence_head Uncertainty_Quantification/ConfidenceHead/scripts
git add Uncertainty_Quantification/ConfidenceHead/scripts Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py Uncertainty_Quantification/ConfidenceHead/README.md
git commit -m "feat: add confidence workflow scripts"
```

Expected: 所有脚本 `--help` 成功，分派测试通过，Python 编译检查无输出。

---

### Task 6: 完整回归与本地 n20 W&B offline 全链路

**Files:**
- Modify if required by evidence: only files already listed in Tasks 1–5
- Runtime outputs only: `Uncertainty_Quantification/ConfidenceHead/outputs/`

**Interfaces:**
- Consumes: Tasks 1–5 的完整配置化四阶段流程。
- Produces: 本地 n20 完整 cache、run、evaluation、verify 和 W&B offline 证据。

- [ ] **Step 1: 运行 ConfidenceHead 全测试集**

```bash
conda run -n upet_new python -m pytest Uncertainty_Quantification/ConfidenceHead/tests -q
```

Expected: 全部通过，无 warning（pytest 将 warning 视为错误时，测试中的预期 W&B warning 必须用 `pytest.warns` 捕获）。

- [ ] **Step 2: 运行仓库静态检查**

```bash
conda run -n upet_new python -m ruff format --check Uncertainty_Quantification/ConfidenceHead
conda run -n upet_new python -m ruff check Uncertainty_Quantification/ConfidenceHead
conda run -n upet_new python -m mypy Uncertainty_Quantification/ConfidenceHead/confidence_head Uncertainty_Quantification/ConfidenceHead/scripts
```

Expected: 三项均退出 0。若仓库现有配置不覆盖该独立目录，则记录实际命令和输出，不静默跳过。

- [ ] **Step 3: 确认本地依赖和输入身份**

```bash
conda run -n upet_new python -c "import torch, wandb; print(torch.__version__); print(wandb.__version__)"
sha256sum data/checkpoint/pet-omatpes-l-v0.1.0.ckpt data/dataset/matpes_n20.extxyz
```

Expected: W&B 可导入；checkpoint SHA 为 `879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004`，本地 n20 SHA 为 `c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d`。若 W&B 未安装，使用 `requirements-remote.txt` 安装到 `upet_new` 环境后重新检查。

- [ ] **Step 4: 清理目标范围内的本次测试输出**

只在确认解析后的目标严格位于 `Uncertainty_Quantification/ConfidenceHead/outputs` 且属于本次 `upet_n20_local_cpu` run 后，移走同名旧冒烟测试目录到该 outputs 下的时间戳备份目录；不得删除其他实验结果。

- [ ] **Step 5: 顺序执行本地四阶段**

```bash
cd Uncertainty_Quantification/ConfidenceHead
conda run -n upet_new python scripts/build_cache.py --config configs/n20_local_cpu.yaml
conda run -n upet_new python scripts/train.py --config configs/n20_local_cpu.yaml
conda run -n upet_new python scripts/evaluate.py --config configs/n20_local_cpu.yaml
conda run -n upet_new python scripts/verify.py --config configs/n20_local_cpu.yaml
```

Expected: 四个命令退出 0；训练只运行 1 epoch；不访问 W&B 网络。

- [ ] **Step 6: 核验本地产物**

检查并记录：cache manifest `status=complete`；run manifest `status=complete`；`best.pt` 与 `last.pt` 均存在；JSONL 恰有 1 条且包含 `val/total_loss_ema`；evaluation manifest 完整；verify 返回成功；W&B 目录中存在 `offline-run-*` 且 run summary/历史包含训练指标。

- [ ] **Step 7: 修复仅由验证证据发现的问题并重复受影响检查**

任何失败都先使用 `superpowers:systematic-debugging` 定位根因，再补一个能稳定复现该问题的失败测试，实施最小修复并重跑该测试、相关测试文件和全测试集。

- [ ] **Step 8: 提交验证阶段产生的源代码修复**

若 Step 7 产生源代码或测试修改：

```bash
git add Uncertainty_Quantification/ConfidenceHead
git commit -m "fix: harden configurable confidence workflow"
```

运行产物不加入 Git。

---

### Task 7: 最终审查、GitHub 推送与目标服务器 CPU n20 验证

**Files:**
- Read/verify: all files changed in Tasks 1–6
- Remote checkout: `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new`
- Remote runtime outputs: configured ConfidenceHead output root

**Interfaces:**
- Consumes: 本地测试通过且工作树干净的 `ConfidenceHead` 分支。
- Produces: GitHub 上同一提交、目标服务器拉取后的同一 SHA、远端 CPU n20 四阶段验证记录。

- [ ] **Step 1: 执行最终 diff 审查**

```bash
git status --short
git diff HEAD~5..HEAD --check
git diff HEAD~5..HEAD --stat
git log --oneline -8
```

Expected: 工作树干净；无空白错误；提交按配置、模型、命令、W&B、脚本/文档和必要修复分层。

- [ ] **Step 2: 使用 verification-before-completion 复核关键证据**

重新运行 Task 6 Step 1–2 的全测试和静态检查，并读取最新输出后才能声明本地完成。不得引用较早运行结果代替最新证据。

- [ ] **Step 3: 推送当前分支到 GitHub**

```bash
git push origin ConfidenceHead
git rev-parse HEAD
git ls-remote origin refs/heads/ConfidenceHead
```

Expected: 本地 HEAD 与远端分支 SHA 完全一致。

- [ ] **Step 4: 目标服务器拉取同一提交**

通过已确认 SSH 身份连接目标服务器，在 `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new` 执行非破坏性 fetch/fast-forward pull，并确认：

```bash
git status --short
git branch --show-current
git rev-parse HEAD
```

Expected: 当前分支为 `ConfidenceHead`，工作树干净，SHA 与 GitHub 相同。若远端存在未提交改动，停止并报告，不覆盖。

- [ ] **Step 5: 核验远端环境与输入**

```bash
python -c "import torch, wandb; print(torch.__version__); print(wandb.__version__)"
sha256sum /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/pet-omatpes-l-v0.1.0.ckpt /XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_n20.extxyz
```

Expected: 使用项目环境；torch 为 `2.11.0+cu128`；W&B 可导入；SHA 与配置一致。

- [ ] **Step 6: 顺序执行远端 CPU n20 四阶段**

```bash
cd /XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new/Uncertainty_Quantification/ConfidenceHead
python scripts/build_cache.py --config configs/n20_cpu.yaml
python scripts/train.py --config configs/n20_cpu.yaml
python scripts/evaluate.py --config configs/n20_cpu.yaml
python scripts/verify.py --config configs/n20_cpu.yaml
```

Expected: 四个命令退出 0；设备为 CPU；W&B 为 offline；只使用 n20；不启动 GPU 训练。

- [ ] **Step 7: 核验远端完整产物并记录摘要**

检查 cache/run/evaluation manifest 状态、best/last SHA、JSONL epoch 记录、W&B offline run、verify 输出、git commit 与依赖版本。最终报告必须给出运行目录、manifest 路径、best epoch、best metric、stop reason 和远端 HEAD。

- [ ] **Step 8: 保持 full GPU 未执行**

只对 `configs/full_gpu.yaml` 执行 `load_config` 静态解析。最终报告明确说明 GPU 全数据训练未启动、未生成图、未迁移旧结果。

