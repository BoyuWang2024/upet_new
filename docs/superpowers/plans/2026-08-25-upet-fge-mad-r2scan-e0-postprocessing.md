# FGE MAD R2SCAN E0 Post-processing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改变或重复 MAD R2SCAN test prediction 的前提下，实现 test-derived direct E0 与 val-calibrated model-aware E0 两种 float64 后处理，生成可审计 sidecar、统计和四组图。

**Architecture:** 复用现有 inference config、prediction chunk、成员身份、population STD、artifact 与绘图基础设施。新增严格 correction 配置、纯数值核心、float64 EXTXYZ/只读 prediction 适配器、三个隔离的 campaign 阶段和专用绘图发布器；方法 A 只读 test，方法 B 只在 val 拟合后封存。

**Tech Stack:** Python 3.11+、NumPy、PyTorch、ASE/extxyz、metatensor/metatrain（仅生产环境 checkpoint E0 解码）、PyYAML、Matplotlib、pytest、tox、Ruff、mypy。

---

## 文件结构

- Create: `Uncertainty_Quantification/FGE/fge/energy_correction_config.py` — 严格加载 correction/plot YAML，并复用两份 inference config。
- Create: `Uncertainty_Quantification/FGE/fge/energy_correction.py` — float64 SVD、E0 提取、两种修正和指标纯函数。
- Create: `Uncertainty_Quantification/FGE/fge/energy_correction_io.py` — EXTXYZ float64 reference、只读 prediction chunk、SHA/size/mtime 快照与 NPZ schema。
- Create: `Uncertainty_Quantification/FGE/fge/energy_correction_campaign.py` — 三阶段不可变发布与 result manifest 验证。
- Create: `Uncertainty_Quantification/FGE/fge/energy_correction_plot.py` — 三张 Energy 与一张 Force 图的共享分析/发布。
- Modify: `Uncertainty_Quantification/FGE/fge/inference_evaluation.py` — 将现有 verified prediction iterator 提升为公共只读接口。
- Modify: `Uncertainty_Quantification/FGE/fge/plot_analysis.py` — 暴露可选固定 log limits 的单 panel 分析接口。
- Modify: `Uncertainty_Quantification/FGE/fge/plot_rendering.py` — 复用现有 figure builder，不改变旧 plot suite 行为。
- Create: `Uncertainty_Quantification/FGE/scripts/correct_energy.py` 与 `plot_energy_correction.py`。
- Create: 三份 config：`inference_mad_r2scan_val.yaml`、`mad_r2scan_energy_correction.yaml`、`plot_mad_r2scan_energy_correction.yaml`。
- Create: 六份 `test_energy_correction*.py` 测试。
- Create: `docs/validation/2026-08-25-fge-mad-r2scan-e0-postprocessing-acceptance.md`。

### Task 1: 严格配置与 split 隔离

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/energy_correction_config.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_energy_correction_config.py`

- [ ] **Step 1: 写失败测试，固定 correction 配置契约**

```python
def test_correction_config_binds_test_and_val_without_swapping(tmp_path: Path) -> None:
    test_config = write_inference_config(tmp_path, label=mad_r2scan_test, split=test)
    val_config = write_inference_config(tmp_path, label=mad_r2scan_val, split=val)
    path = write_correction_config(tmp_path, test_config, val_config)
    config = load_energy_correction_config(path)
    assert config.test.dataset.split == test
    assert config.val.dataset.split == val
    assert config.elements == tuple([*range(1, 84), *range(89, 95)])
    assert config.test.ensemble == config.val.ensemble
```

同时覆盖未知键、test/val 互换、不同 checkpoint/result manifest、重复元素、非 89 元素和非固定容差，均抛 `HardFailure`。

- [ ] **Step 2: 运行测试确认红灯**

Run: `tox -e fge-tests -- -q Uncertainty_Quantification/FGE/tests/test_energy_correction_config.py`

Expected: FAIL during collection with `ModuleNotFoundError: ...energy_correction_config`。

- [ ] **Step 3: 实现最小严格配置加载器**

实现不可变的 `CorrectionTolerances`、`EnergyCorrectionConfig` 和
`EnergyCorrectionPlotConfig`。复用 `load_inference_config`，通过 `HardFailure`
拒绝未知键；严格校验 test/val split、共同 checkpoint/result manifest、K=8 成员、
固定元素顺序 `1..83, 89..94` 和设计中的固定容差。相对路径相对 YAML 文件解析。

- [ ] **Step 4: 运行配置测试确认绿灯**

Run: `tox -e fge-tests -- -q Uncertainty_Quantification/FGE/tests/test_energy_correction_config.py`

Expected: PASS。

- [ ] **Step 5: 提交配置契约**

暂存且仅暂存配置模块和对应测试，提交信息：
`feat(fge): define strict energy correction config`。

### Task 2: float64 数值核心与 checkpoint E0

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/energy_correction.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_energy_correction.py`

- [ ] **Step 1: 写失败测试固定 SVD 与修正语义**

用小型合成组成矩阵覆盖：`fit_elemental_offsets` 的 float64、满秩、奇异值与条件数；
欠秩/非有限输入硬失败；`fit_direct_test_e0` 从 `raw-atomization` 恢复 E0；
`fit_model_aware_delta` 只使用 ensemble mean；两种 apply 给所有成员增加共同 shift；
direct 恒等式、population STD 不变以及误差、胜率与 correlation 指标。

- [ ] **Step 2: 运行数值测试确认红灯**

Run: `tox -e fge-tests -- -q Uncertainty_Quantification/FGE/tests/test_energy_correction.py`

Expected: FAIL during collection，模块尚不存在。

- [ ] **Step 3: 实现最小纯函数核心**

`np.linalg.lstsq` 使用
`rcond=max(x.shape) * np.finfo(np.float64).eps`。实现
`fit_elemental_offsets`、`fit_direct_test_e0`、`fit_model_aware_delta`、
`apply_direct_e0`、`apply_model_aware`、`energy_uq_mev_per_atom` 与指标函数。
要求 89 列时 rank=89，所有计算显式 float64，严禁静默截断或广播。

- [ ] **Step 4: 写失败测试固定真实 checkpoint E0 解码边界**

构造最小 checkpoint bundle 和序列化 metatensor buffer，验证只读取
`additive_models.0.energy_composition_buffer`，严格检查一个 energy block、
`center_type` samples、`energy` property、`(89, 1)` values 和固定元素顺序；
缺键、错序、多 block、非有限值均硬失败。

- [ ] **Step 5: 实现并验证 `extract_matpes_e0`**

惰性导入 `metatensor.torch.load_buffer`，复用 `load_checkpoint_bundle`；不构造模型、
不 forward。运行同一测试文件，Expected: PASS。

- [ ] **Step 6: 提交数值核心**

仅暂存数值核心与测试，提交信息：
`feat(fge): add float64 energy correction core`。

### Task 3: float64 reference 与只读 prediction adapter

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/energy_correction_io.py`
- Modify: `Uncertainty_Quantification/FGE/fge/inference_evaluation.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_energy_correction_io.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_inference_evaluation.py`

- [ ] **Step 1: 写失败测试固定只读数据契约**

覆盖 ASE `iread` 的 float64 `energy`/`atomization_energy`、结构 ID、原子数、89 列组成；
覆盖缺字段、错 split、未知元素、错 SHA、重复 ID，并验证文件 SHA/size/mtime 前后不变。

- [ ] **Step 2: 写 prediction 对齐失败测试并确认红灯**

将 `inference_evaluation._verified_prediction_chunks` 提升为公共 iterator；测试要求逐 chunk
校验 manifest SHA、chunk identity、structure IDs、`n_atoms`、`atomic_numbers`、offsets、
K=8 member IDs 与 reference 顺序。任一错位必须 `HardFailure`，不得按长度猜测。

Run: `tox -e fge-tests -- -q Uncertainty_Quantification/FGE/tests/test_energy_correction_io.py Uncertainty_Quantification/FGE/tests/test_inference_evaluation.py`

Expected: 新测试 FAIL，公共接口/adapter 尚不存在。

- [ ] **Step 3: 实现 float64 reader、快照与 verified iterator**

新增只服务 E0 实验的 reader，保持既有 inference reader 的 float32 语义不变；预测读取后
转 float64。提供原始树 SHA/size/mtime 清单和比较器，以及原子级映射检查。

- [ ] **Step 4: 实现轻量 NPZ sidecar schema**

sidecar 使用原子写入，包含结构 ID、原 chunk identity、natoms、composition、成员能量、
ensemble mean、UQ 和 residual；manifest 对每个文件记录 SHA、size、mtime。reader 严格拒绝
pickle/object arrays 和未知字段。

- [ ] **Step 5: 运行相关测试确认绿灯并提交**

重复 Step 2 命令，Expected: PASS。仅暂存上述四个文件，提交信息：
`feat(fge): add read-only correction data adapters`。

### Task 4: 三阶段 campaign、sealed manifest 与 CLI

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/energy_correction_campaign.py`
- Create: `Uncertainty_Quantification/FGE/scripts/correct_energy.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_energy_correction_campaign.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_energy_correction_scripts.py`

- [ ] **Step 1: 写失败测试固定方法隔离**

`calibrate-model-aware` 只能接收 val config/prediction/reference，使用 ensemble mean 拟合一套
公共 delta，并最后发布 `calibration_manifest.json`；其 API 与 manifest 不得出现 test 路径。
`extract-direct-e0` 只能读取 test EXTXYZ 的 raw/atomization，不得读取 val 或 prediction。

- [ ] **Step 2: 写失败测试固定 apply 与 sealed 输入**

`apply` 必须重新散列两个 sealed manifest，绑定 test prediction 的 129 个 chunk identities，
发布 baseline/direct/model-aware/shared-force，且最后原子发布 `result_manifest.json`。
覆盖篡改校准、错数据 SHA、错 member identity、错结构顺序、部分输出和重复发布。

- [ ] **Step 3: 运行 campaign/CLI 测试确认红灯**

Run: `tox -e fge-tests -- -q Uncertainty_Quantification/FGE/tests/test_energy_correction_campaign.py Uncertainty_Quantification/FGE/tests/test_energy_correction_scripts.py`

Expected: FAIL during collection，campaign/CLI 尚不存在。

- [ ] **Step 4: 实现三个隔离阶段与 validate**

复用 `atomic_write_json`、`atomic_torch_save`/NPZ 等价原子发布、`sibling_staging` 和
artifact 身份 helpers。所有 stage 先完整验证再发布，已有正式完整目录只读复用；不完整或
冲突目录硬失败，禁止删除。`validate` 全量重算 manifest 链和数值不变量但不写文件。

- [ ] **Step 5: 实现薄 CLI 并加 no-compute guard**

CLI subcommands 为 `calibrate-model-aware`、`extract-direct-e0`、`apply`、`validate`。
测试 monkeypatch 训练、predict、backward、optimizer 入口为立即失败，证明后处理不会调用它们。

- [ ] **Step 6: 运行绿灯、回归并提交**

重复 Step 3；再运行：
`tox -e fge-tests -- -q Uncertainty_Quantification/FGE/tests/test_completed_read_only.py Uncertainty_Quantification/FGE/tests/test_inference_only.py`。

Expected: 全部 PASS。仅暂存本 Task 文件，提交信息：
`feat(fge): publish sealed energy correction campaign`。

### Task 5: 专用绘图与固定坐标范围

**Files:**
- Create: `Uncertainty_Quantification/FGE/fge/energy_correction_plot.py`
- Modify: `Uncertainty_Quantification/FGE/fge/plot_analysis.py`
- Create: `Uncertainty_Quantification/FGE/scripts/plot_energy_correction.py`
- Create: `Uncertainty_Quantification/FGE/tests/test_energy_correction_plot.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_plot_analysis.py`

- [ ] **Step 1: 写失败测试固定四图语义**

测试输出三张 Energy 和一张共享 Force 的 PNG/PDF；direct 与 model-aware 的 x/y log limits
逐位相同，baseline 独立并标注 `unmatched energy-zero baseline`。标题必须分别包含
`test-derived direct E0` 与 `val-calibrated model-aware E0`；不生成 correlation/sweep 图。

- [ ] **Step 2: 写固定 limits 的 plot analysis 失败测试**

为单 panel 分析接口增加可选显式 limits，验证输入点不变、越界硬失败、旧接口结果不变。
复用 `PlotPanelInput`、现有统计与 `build_single_panel_figure`。

- [ ] **Step 3: 运行绘图测试确认红灯**

Run: `tox -e fge-tests -- -q Uncertainty_Quantification/FGE/tests/test_energy_correction_plot.py Uncertainty_Quantification/FGE/tests/test_plot_analysis.py`

Expected: 新测试 FAIL。

- [ ] **Step 4: 实现绘图、统计与 manifest 发布**

读取并验证 `result_manifest.json`，生成 Energy 误差指标、UQ Spearman/log-Pearson 统计、
胜率与 head-to-head；Force 完全复用 shared force 语义。图和 JSON 经 staging 原子发布，
`plot_manifest.json` 记录所有输入/输出 SHA、size、mtime。

- [ ] **Step 5: 运行绿灯、旧绘图回归并提交**

重复 Step 3，再运行既有 `test_plot_rendering.py`。Expected: 全部 PASS。
仅暂存本 Task 文件，提交信息：`feat(fge): plot MAD energy correction comparison`。

### Task 6: 正式配置与本地静态/单元验证

**Files:**
- Create: `Uncertainty_Quantification/FGE/configs/inference_mad_r2scan_val.yaml`
- Create: `Uncertainty_Quantification/FGE/configs/mad_r2scan_energy_correction.yaml`
- Create: `Uncertainty_Quantification/FGE/configs/plot_mad_r2scan_energy_correction.yaml`
- Modify: `Uncertainty_Quantification/FGE/tests/test_energy_correction_config.py`
- Modify: `Uncertainty_Quantification/FGE/tests/test_energy_correction_scripts.py`

- [ ] **Step 1: 写失败的正式配置与 CLI dry-run 测试**

断言固定 test/val/checkpoint/result SHA、结构数、原子数、K=8、89 元素与输出目录；
val 配置复用 test 的过滤和成员身份。dry-run 只能打印解析后的身份，不得创建输出。

- [ ] **Step 2: 添加三份配置并确认绿灯**

运行配置/CLI 测试，Expected: PASS；再运行 `tox -e fge-tests` 和 `tox -e lint`。
若 lint 被无关工作树改动阻断，另外对本任务文件运行 Ruff format/check 与 mypy，并在报告中
保留全量命令的真实失败证据，不修改无关文件。

- [ ] **Step 3: 提交配置**

仅暂存三份配置和必要测试更新，提交信息：
`config(fge): define MAD R2SCAN correction campaign`。

### Task 7: 远端一次性 val predict 与正式后处理

**Scope:** 所有大数据、prediction、sidecar 计算与绘图均只在远端
`/home/bywang/code/UQ/upet_new_fge_test` 执行。

- [ ] **Step 1: 只同步小型代码/config 与 filtered val 输入**

先比较远端文件身份；远端缺少 filtered val 时只上传该必要输入一次。禁止把任何 EXTXYZ、
checkpoint、prediction chunk、NPZ sidecar 或其他大文件拉回本地。

- [ ] **Step 2: 只读验证 test，检查 val 是否已完成**

test 只允许验证既有
`outputs/inference_mad_r2scan_test_k8/prediction/manifest.json`（SHA
`39da4c760bcdd8d4930a66970c83bc04f71bde937b26240e44eaf6b85c307fef`）；
严禁再次运行 test predict。若 val 的完整、身份匹配 manifest 已存在则复用；否则继续 Step 3。

- [ ] **Step 3: 只启动一次 val K=8 predict**

使用远端 Python `/home/bywang/.conda/envs/upet_new/bin/python` 和正式 val config，先做 preflight，
记录 PID/log/启动时间。启动前再次确认不存在同输出的进程或完整 manifest；绝不并发或重启。
只读监控到完整 manifest，失败时系统诊断且不删除任何文件。

- [ ] **Step 4: 按隔离顺序运行三阶段 correction**

先 `calibrate-model-aware` 并记录 sealed calibration SHA；再独立运行
`extract-direct-e0`；最后 `apply`。每阶段前后记录输入树 SHA/size/mtime，证明 test prediction
未变；禁止因 test 指标返回重拟合 delta。

- [ ] **Step 5: 双重只读 validate 与绘图**

连续两次运行 `validate`，两次之间不写输入；比较原 test/val prediction 与两个 sealed
manifest 的 SHA/size/mtime 完全一致。随后运行正式 plot config，验证 8 个 PNG/PDF、统计和
plot manifest 的 SHA 与 source manifest 绑定。

- [ ] **Step 6: 仅同步小型最终产物**

只将最终图目录中的 PNG/PDF/JSON、小型校准 JSON/manifest 证据和远端命令摘要同步回当前分支；
明确排除 raw/filtered dataset、prediction chunks、reference/energy NPZ sidecars 和 checkpoint。

### Task 8: 验收报告、全量验证与最终 task-only commit

**Files:**
- Create: `docs/validation/2026-08-25-fge-mad-r2scan-e0-postprocessing-acceptance.md`
- Create: `Uncertainty_Quantification/Plots/FGE/mad_r2scan_test_energy_correction/*`

- [ ] **Step 1: 编写证据驱动的验收报告**

记录远端环境、所有输入 SHA/计数、val predict 的唯一运行证据、两种公式、SVD rank/condition、
重构与 identity/UQ 容差、Energy/Force/UQ 指标、双重只读验证、源树未变证明和同步排除清单。
清楚标注 direct 为 test 内口径对齐，model-aware 为 val-to-test held-out。

- [ ] **Step 2: 运行最终新鲜验证**

依次运行：

```bash
tox -e fge-tests
tox -e lint
git diff --check
```

核对 plot manifest 中 8 个图文件及 JSON 的 SHA，核对本地仅有允许的小型远端产物。

- [ ] **Step 3: 只暂存 task files 并审计 staged diff**

显式列出本计划中的代码、测试、配置、图和报告路径执行 `git add`；运行
`git diff --cached --name-status` 与 `git diff --cached --check`。确认不含 `.idea`、LLPR、
BootStrapping、ConfidenceHead 或既有无关 Plots 改动。

- [ ] **Step 4: 最终提交并核验提交范围**

提交信息：`feat(fge): compare MAD R2SCAN E0 corrections`。运行
`git diff-tree --no-commit-id --name-status -r HEAD`，逐项确认只含 task files；保留所有无关
working-tree 改动原样不动。
