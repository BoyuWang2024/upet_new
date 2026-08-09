# UPET FGE 代码与既有结果迁移设计

状态：Precheck 与五个新版设计部分均已由用户确认。本文只定义技术设计与验收合同；在用户复核本文并批准后续实施计划之前，不实现 FGE 代码、不调整环境，也不执行结果迁移。

## 1. 决策摘要

本任务在当前 `FGE` 分支中，为 `/home/lilong/code/UQ/upet_new/Uncertainty_Quantification/FGE` 设计一套完整、可发布、可测试的 UPET-native FGE 子系统，并在远端把旧仓库 `/home/bywang/code/UQ/upet/Ensemble` 中已经完成的 8-member FGE 结果一次性导入同一套 canonical result schema。

已确认的核心决策是：

1. full 既有结果不重新训练，也不重新执行 full test 模型预测；迁移只读取、验证、重组和规范化旧 artifacts。
2. 正式代码提供 `preflight`、`train`、`predict`、`evaluate`、`validate` 五个独立阶段，未来仍可执行新的 FGE 训练。
3. 迁移使用“外部 base checkpoint + 12 个完整 readout endpoint replacement tensors”的 A3 member 表示，显著消除 8 个旧 full checkpoints 中重复的模型状态。
4. 正式代码只消费 canonical schema；旧格式识别仅存在于受测但不发布的 `internal_migration/`。
5. 旧 UPET 数值语义原样保留并版本化为 `legacy_upet_fge_v1`，不替换为 MACE/CarNet 的 sample STD、unordered-pair GMD、weighted ensemble 或 raw/EMA 双分支。
6. 本地只开发和提交代码；所有测试、n20 原生流程、A3 等价推理、full migration 和 full validation 都在远端执行。
7. BootStrapping、W&B 历史、日志、plots、失败结果、误标 train-derived prediction 和旧续训状态均不迁移。

总体方案是“UPET 原生数值实现 + MACE 式发布边界”。主要组织参考 `/home/lilong/code/UQ/mace_new/Uncertainty_Quantification/FGE`，但所有 UPET 模型、checkpoint、loss、readout、member 与 UQ 语义独立实现。

## 2. 自包含 Precheck 基线

### 2.1 当前目标仓库

- 当前分支：`FGE`。
- 当前 HEAD：`11036e7e36d7f7e87e4e8d75ae52e1707c7db3b5`。
- 工作树：设计开始时干净。
- 当前 `Uncertainty_Quantification/` 只有 ConfidenceHead 与 LLPR，没有 FGE 目录。
- 前一轮设计提交 `9f4715a` 和计划提交 `797fba7` 仍存在于 Git 对象库，但已从分支历史 reset；本轮不恢复旧实施计划，而是在 `FGE` 分支重新提交准确设计。

目标仓库实际依赖约束以 `pyproject.toml` 为准：

- Python `>=3.11`；
- `metatrain>=2026.3.1,<2026.4`；
- `metatomic-ase` 由项目依赖管理。

仓库 AGENTS.md 中“metatrain `>=2026.2,<2026.3`”的描述与当前 `pyproject.toml` 不一致，视为过期说明；设计和实现不得因此降级环境。

### 2.2 旧代码身份与训练语义

旧 FGE 代码位于 `/home/lilong/code/UQ/upet/Ensemble` 和远端 `/home/bywang/code/UQ/upet/Ensemble`。两端 tracked FGE tree（`src/FGE`、`scripts/FGE`、`configs/FGE`、`tests/FGE`）SHA256 均为：

```text
7565832c4e89172439173439b15074c280be6decaa0ad68cf3761f87c8cacbae
```

旧代码的权威计算合同为：

- base 使用 checkpoint 的 restart `model_state_dict`，不使用 `best_model_state_dict`；
- targets 为 `energy`、`non_conservative_forces`、`non_conservative_stress`；
- loss 从 checkpoint 恢复 energy、conservative forces、virial、non-conservative forces、non-conservative stress 五项 Huber 配置，`sliding_factor=None`；
- optimizer 为 Adam，weight decay 为 0；旧 scheduler 不恢复，由 FGE LR 独占；
- 只训练 PET `node_last_layers.*` 与 `edge_last_layers.*` 的 12 个 tensor，共 13,338 个标量参数；
- composition model 与 scaler 不重新拟合；
- EMA decay 为 0.999，只用于 validation，不发布 EMA member；
- 正式 run 为 K=8、8 epochs/cycle、batch size 16、`drop_last=true`；
- 348,780 个训练结构产生 21,798 updates/epoch、174,384 updates/cycle、64 epochs 和 1,395,072 total updates；
- asymmetric triangular LR 为 `lr_min=1e-8`、`lr_max=1e-7`、rise fraction 0.2，每个 cycle endpoint 回到约 `lr_min` 并发布一个 member。

历史 run 名称 `n20_official_readout_schema_v4_path_validation` 具有误导性：实际使用 348,780 个结构，且 train 与 validation 是同一文件。该 run 可作为 member 来源和训练路径证据，但 validation 指标不能作为泛化结论。

### 2.3 Base checkpoint 与数据身份

远端正式 base checkpoint：

```text
/home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt
size:   771,692,879 bytes
sha256: 879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
```

数据身份：

```text
train/legacy validation:
  structures: 348,780
  size:       397,915,595 bytes
  sha256:     12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec

independent matpes_test_full:
  structures: 19,374
  atoms:      149,321
  size:       21,726,599 bytes
  sha256:     1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d
```

train unique IDs 为 348,780，test unique IDs 为 19,374，二者 overlap 为 0；test duplicate IDs 为 0。

### 2.4 旧正式结果清单

远端权威结果根：

```text
/home/bywang/code/UQ/upet/Ensemble/results/FGE/upet-FGE-CKPT-UQ-v1.0
```

总量约 20 GB。活动迁移范围只包括成功 source run 中的 8 个 members，以及独立 `matpes_test_full` prediction/UQ/evaluation。明确排除：

- `old/` OOM failure run；
- `train_prediction/` 中误标为 test 的 train-derived artifacts；
- `wandb/`、`metrics.jsonl`、`wandb_run.json`、`wandb_status.json`；
- `run_state.pt`、optimizer、scheduler、EMA resume state；
- plots 和本地 `remote-plots/` 副本。

8 个旧 member checkpoint 均为 1,543,343,010 bytes，旧 manifest 均标记 accepted、valid、reload/finite passed、frozen drift 0。其 SHA256 为：

| Member | Endpoint global step | SHA256 |
| --- | ---: | --- |
| `member_001` | 174,384 | `c9376c9d06aaef652602cfd30c20aaeb2c20b547f2cdc2aded7b7554231e44cb` |
| `member_002` | 348,768 | `96239b6766252168da329435fa19f78ec0801903aae5f192cf42561f845b08a7` |
| `member_003` | 523,152 | `6afaaec932d44eea9e3d7f5460b2d4fb16142a7acca5668a324ff5c98d4c8ef3` |
| `member_004` | 697,536 | `c6e688442a7afef456de36282d3a6d998d15ac9b328242d9c27a8d8d64b865e0` |
| `member_005` | 871,920 | `415306b16c60aa400ce75d2e18fabec9f61cd19d7ced4ec2f380055fe00b1546` |
| `member_006` | 1,046,304 | `68c4a04418022b08d7120eae03862fdb47c42beabc3ac63adf1168065a290cbd` |
| `member_007` | 1,220,688 | `1a99ac819c408d65e8e0d3e411f023b5642a285d7fbbe661b09894c9f708980a` |
| `member_008` | 1,395,072 | `5b91297faa2a1f0ec95d48f728e33687efd4fe3e6f6773c5de2f985ac6671521` |

独立 test 包含 969 chunks/member，共 7,752 个 chunk 文件。已全量验证 chunk 连续性、shape、dtype、finite、structure/atom coverage、跨 member reference 一致性和 mapping 完整性。

### 2.5 旧结果数值完整性

独立 test 最终 MAE：

```text
energy: 0.01294630383014235 eV/atom
force:  0.08171965440375559 eV/angstrom       (component MAE)
stress: 0.00215389879550793 eV/angstrom^3     (3x3 component MAE)
```

逐 chunk 独立重算旧 ensemble、population STD、ordered-pair `N^2` GMD、force structure mean/max/q95、reference 和 MAE 后：

- 核心 UQ 项精确相等或最大绝对差不超过约 `3.7e-9`；
- E/F/stress mean 因 float32 reduction 顺序不同，最大绝对差分别约 `2.44140625e-4`、`1.52587890625e-5`、`2.384185791015625e-7`；
- 最大相对差不超过约 `1.71e-6`；
- 使用 `rtol=2e-6, atol=2e-7` 时全量通过；
- 最终三项 MAE 与旧 JSON 精确一致。

因此旧代码与旧结果在其自身 `legacy_upet_fge_v1` 语义下完整、一致，可作为迁移权威来源；这不等于 ensemble 已良好校准，也不消除历史 train=validation 的科学局限。

### 2.6 已确认环境

本地与远端统一使用 `conda activate upet_new`，用户允许在实施阶段按需调整环境。

```text
local:
  Python 3.11.15
  torch 2.13.0+cpu
  metatrain 2026.3.1
  metatomic 0.1.16
  pytest 9.1.1

remote:
  Python 3.11.15
  torch 2.11.0+cu128
  metatrain 2026.3.1
  metatomic 0.1.16
  pytest 9.1.1
```

editable package version string 不作为代码身份；实际 Git commit、dirty digest 与依赖快照共同构成运行身份。

## 3. 目标、非目标与执行边界

### 3.1 目标

1. 提供完整、可独立执行的 UPET FGE 五阶段正式流程。
2. 为原生新训练和旧结果导入定义唯一 canonical result schema。
3. 把 8 个旧 full checkpoints 转换成紧凑 A3 members，并证明重建预测等价。
4. 在远端通过只读 source、sibling staging、完整 audit 和原子 rename 发布 full canonical result。
5. 用远端 CPU n20 完整原生流程证明正式代码可生成与 migrated full 相同的 schema。
6. 让正式 artifact 不依赖或暴露旧目录布局，同时保留结果树外的完整迁移审计。

### 3.2 非目标

- 不迁移或修改 BootStrapping。
- 不重新训练 full FGE，不重新执行 full test 模型预测。
- 不迁移 W&B、日志、plots、失败结果、误标 train-derived artifacts 或旧续训状态。
- 不提供旧数字脚本兼容入口，不维护双格式 reader。
- 不伪造 EMA member，不增加 weighted ensemble。
- 不增加 stress UQ。
- 不在本地复制远端 full 数据、大 checkpoints 或 full results。
- 不对 ConfidenceHead、LLPR、`src/upet/` 或其他无关代码做重构。

### 3.3 执行边界

- 本地只进行代码、测试代码、配置、README 和设计/计划文档开发与提交。
- 所有可执行测试都在远端专用 clean checkout/worktree 中运行。
- full migration 只在远端执行，且不依赖 GPU；A3 等价测试和 n20 原生流程使用远端 CPU。
- 所有 `outputs/`、audit、临时 runtime config 和测试证据保持 ignored，不进入 Git。

## 4. 架构与发布边界

### 4.1 目录结构

```text
Uncertainty_Quantification/FGE/
├── README.md
├── __init__.py
├── publication_files.txt
├── configs/
│   ├── upet_fge_full.yaml
│   └── upet_fge_n20_cpu.yaml
├── fge/
│   ├── __init__.py
│   ├── errors.py
│   ├── config.py
│   ├── artifacts.py
│   ├── manifests.py
│   ├── checkpoint.py
│   ├── members.py
│   ├── schedule.py
│   ├── data.py
│   ├── preflight.py
│   ├── training.py
│   ├── prediction.py
│   ├── uncertainty.py
│   ├── evaluation.py
│   └── validation.py
├── scripts/
│   ├── preflight.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── validate.py
├── tests/
├── internal_migration/
│   ├── README.md
│   ├── __init__.py
│   ├── migration/
│   │   ├── __init__.py
│   │   ├── legacy_reader.py
│   │   └── converter.py
│   ├── scripts/
│   └── tests/
└── outputs/
    └── .gitignore
```

相较上一版设计，正式模块清单显式增加 `errors.py`、`manifests.py`、`data.py` 和 `preflight.py`。这些职责在实现中不可避免，提前列入设计可避免把错误、manifest、数据合同或 preflight 逻辑塞入无关模块。

### 4.2 单向依赖

- `fge/` 是正式、source-independent 的领域层，只识别 canonical config/member/prediction/evaluation contracts。
- `scripts/` 是五个薄 CLI，每个只加载一个 YAML 并调用一个正式阶段；不提供 `run-all`。
- `internal_migration/` 识别旧布局，可调用正式 artifact writer、A3 packer 与 validator。
- `fge/`、正式 scripts 和正式 tests 禁止 import `internal_migration`。
- 正式代码不扫描、猜测或兼容旧 `Ensemble/results/FGE` 目录。

### 4.3 Publication allowlist

`publication_files.txt` 固定为：

```text
README.md
__init__.py
publication_files.txt
configs/
fge/
scripts/
tests/
```

`internal_migration/` 提交并测试，但不进入正式发布；`outputs/` 只跟踪用于忽略运行内容的 `.gitignore`，其余内容既不发布也不跟踪。正式发布内容不得包含旧绝对路径、旧 run 名、旧 full member 的 source checkpoint SHA（正式 base checkpoint SHA 除外）、迁移标记、W&B、日志、plots 或 BootStrapping 内容。

## 5. 配置、身份与科学状态

### 5.1 严格配置

`fge/config.py` 对 YAML 做拒绝未知字段的严格解析。运行时 YAML 可包含环境相关路径，但写入 canonical root 的 `config_resolved.yaml` 必须清洗路径，只保留逻辑角色和内容身份。

正式训练合同固定为：

- float32，seed 2026；
- restart `model_state_dict`；
- 12 readout tensors、13,338 trainable scalars；
- checkpoint 五项 Huber loss 与 gradient clipping contract；
- Adam、weight decay 0；
- K=8、8 epochs/cycle、batch size 16、`drop_last=true`；
- `lr_min=1e-8`、`lr_max=1e-7`、rise fraction 0.2；
- EMA decay 0.999，仅 validation；
- metric schema 4、structure quantile 0.95、constant tolerance `1e-12`；
- risk coverage `[1.0, 0.95, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]`。

### 5.2 身份语义

正式 artifacts 区分三种代码身份，避免把迁移代码错误声明为历史训练代码：

- `artifact_writer_code_identity`：实际写出 canonical artifacts 的目标仓库 commit/dirty digest；必填。
- `validator_code_identity`：实际验证结果的目标仓库 commit/dirty digest；必填。
- `training_code_identity`：新原生训练时记录实际训练 commit；对历史 migrated members 明确记录状态 `unavailable`，不得伪造为当前代码。

所有 native/migrated manifests 使用相同 key/schema；只允许字段值不同。正式结果还记录 checkpoint/data/config SHA、模型 contract、dependency snapshot 和 canonical artifact hashes。

旧绝对路径、旧 member SHA 和 source-to-A3 映射不进入这些字段，只进入 external audit。

### 5.3 科学状态

- 历史 training provenance：`path_feasibility_only=true`、`split_leakage=true`、`training_code_identity_status=unavailable`。
- 独立 test evaluation：`inference_only=true`、`path_feasibility_only=false`、`split_leakage=false`、`scientific_evaluation=true`。
- n20 smoke：`path_feasibility_only=true`、`split_leakage=true`、`scientific_evaluation=false`。

## 6. Canonical result contract

### 6.1 结果结构

```text
outputs/<experiment>/
├── config_resolved.yaml
├── preflight/
│   ├── train.json
│   ├── predict.json
│   └── evaluate.json
├── training/
│   ├── manifest.json
│   └── members/
│       ├── member_001.pt
│       └── ...
├── prediction/
│   ├── manifest.json
│   └── test.pt
├── evaluation/
│   └── legacy_equal_weight/
│       ├── ensemble.pt
│       ├── uncertainty.pt
│       ├── metrics.json
│       └── report.md
├── validation.json
└── result_manifest.json
```

`result_manifest.json` 是唯一完成标记，最后写入，列出此前所有正式文件的相对路径、角色、大小和 SHA256。缺失它、validation 未通过或任一 hash 不匹配时，结果均视为未完成。

已完成结果不可覆盖或原地修改。对已有 `result_manifest.json` 的结果再次运行 validator 时，validator 必须进入完全只读模式，只验证并返回结论，不重写 `validation.json`、manifest 或任何 artifact。

### 6.2 Preflight reports

三份 report 使用相同 schema，至少包含 `stage`、`status`、`basis`、identity checks 与 scientific flags：

- native preflight 的 `basis=runtime_inputs`；
- migrated existing result 的 `basis=canonical_artifacts`；
- 两者都使用 `status=PASS`，不写旧路径、迁移 source 或 migration 标记。

这避免声称历史训练曾由新 preflight 执行，同时保持 native/migrated schema 一致。

### 6.3 A3 member contract

schema 名称固定为 `upet.fge.member-delta.v1`。这里的 “delta” 指相对于外部 base checkpoint 的稀疏可变状态集合；保存的是完整 replacement values，不是 `member - base` 算术差值。

每个 `member_NNN.pt` 必须：

- 可由 `torch.load(..., weights_only=True, map_location="cpu")` 加载；
- 包含 schema version、member ID、cycle、global step、base checkpoint SHA256；
- 包含确定顺序的 12 个 tensor name/dtype/shape/CPU value；
- 不包含模型对象、optimizer、scheduler、EMA、RNG、absolute path 或 source metadata。

加载时先流式验证 base SHA，使用 restart state 恢复模型，再严格覆盖 12 个 tensors。切换 member 前必须恢复同一 base snapshot，防止成员串写。

`training/manifest.json` 记录 K、member 顺序、canonical member hashes、endpoint steps、12-tensor/13,338 contract、base identity、frozen fingerprint、科学状态和代码身份状态。所有 non-readout parameters/buffers 必须与 base 精确相等。

### 6.4 Canonical prediction

`prediction/test.pt` 使用单一 CPU tensor payload：

- energy prediction：float32 `[K,S]`；
- forces prediction：float32 `[K,A,3]`；
- stress prediction：float32 `[K,S,3,3]`；
- energy/forces/stress reference：各一份；
- structure IDs、member IDs、atomic numbers、`n_atoms`、structure offsets/mapping；
- target names、units 与 K/S/A statistics。

迁移合并旧 chunks 时，只改变容器和重复 reference 的组织方式。E/F/stress/reference 元素、member/chunk/structure/atom 顺序、ID、dtype 与 mapping 必须保持不变。

### 6.5 Evaluation artifacts

只发布 `legacy_equal_weight`：

- `ensemble.pt`：E/F/stress equal-weight means；
- `uncertainty.pt`：旧 energy/force UQ，不含 stress UQ；
- `metrics.json`：旧 metric schema 4 的 MAE、correlation、risk-coverage 和质量诊断；
- `report.md`：从结构化 artifacts 生成的摘要，不是运行日志。

迁移保留并规范化既有 ensemble/UQ/metrics 值。validator 可以独立复算并比较，但不得以复算结果覆盖迁移值。

## 7. 原生五阶段流程

### 7.1 Preflight

Preflight 检查：

- strict config 与固定公式版本；
- checkpoint/data 文件、SHA、target、unit 与 statistics；
- restart state 可解析；
- 精确 12 tensors、13,338 parameters；
- K、cycle、batch、drop-last、device/dtype；
- 磁盘空间、输出不可覆盖、路径不逃逸；
- code/dirty/dependency identity 可记录；
- scientific flags 与数据身份一致。

Preflight 不执行 forward、backward 或 optimizer step。首次 native train preflight 原子写 sanitized `config_resolved.yaml`；后续阶段验证 identity 一致，不修改已完成结果。

### 7.2 Train

训练从 restart state 开始，使用旧完整五项 loss contract。训练 loader 保留随机旋转、neighbor list、additive removal 和 scale removal；validation 不使用随机旋转。composition/scaler 不重拟合。

一个 Adam optimizer 与一个 readout-only EMA 跨越全部 cycles。每个 optimizer update 使用旧 asymmetric triangular LR。每个 epoch 和 endpoint 检查 frozen state；每个 endpoint 检查 trainable finite、原子写 A3、base+A3 reload 和 E/F/stress finite smoke。只有全部通过的 endpoint 才加入 manifest。

原生 resume 只保存在 ignored `_work/`，且 config/code/base/data identity 必须全部匹配。旧 `run_state.pt` 永不导入。

可选 W&B 只允许未来 native training 使用，所有文件只能进入 `_work/`。W&B 初始化、记录或结束失败只产生 warning，不影响 optimizer、resume、member 接受或完成状态。n20 和 migration 强制关闭。

### 7.3 Predict

Predict 只读 canonical training artifacts 和运行时 base path。加载 base 一次，对每个 member 执行“恢复 base → 覆盖 12 tensors → 推理”，按 manifest 顺序写 E/F/stress。member 缺失、顺序变化、base mismatch 或 reference/mapping 不一致均硬失败；禁止跳过成员或缩小 K。

### 7.4 Evaluate

Evaluate 只读取 canonical prediction，不加载模型或 dataset。它生成 equal-weight ensemble、`legacy_upet_fge_v1` energy/force UQ、metric schema 4 和报告。没有独立 `uq.py` CLI。

### 7.5 Validate

Validate 从磁盘重新打开所有 artifacts，不复用 writer 的内存值。首次完成验证通过后原子写 `validation.json`，最后写 `result_manifest.json`。已完成结果的再次验证严格只读。

## 8. 一次性旧结果迁移

### 8.1 正式结果与审计分离

```text
outputs/<experiment>/                         # 正式 canonical result
outputs/_internal_migration/<experiment>/     # ignored external audit
```

正式结果 source-independent；external audit 才保存旧路径、旧 hashes、旧清单和 source-to-A3 mapping。

### 8.2 迁移步骤

1. 读取并验证旧成功 source manifest、8 个 member hashes、test manifest、969 chunks/member、旧 UQ/metrics 和完成状态。
2. 对每个旧 member 与 base 做全 state 对比；任何 non-readout 漂移都硬失败。
3. 提取 12 个完整 endpoint replacements，写 A3 并重新严格加载。
4. 按 numeric member/chunk 顺序流式合并 `matpes_test_full`，精确保持 prediction/reference/mapping。
5. 规范化既有 ensemble/UQ/metrics，不调用正式 train、predict 或 evaluate。
6. 正式 validator 从 canonical staging 独立复算 mean、UQ 与 MAE并比较，不覆盖迁移值。
7. 重新计算旧 source critical hashes，要求与迁移前完全相同。
8. 在 canonical rename 之前，原子写完整 external audit。audit 至少记录 source paths/hashes、8-member source-to-A3 mapping、source hashes before/after、artifact writer/validator identities、canonical staging signature、expected final destination 和 `publication_authorized=true`。
9. audit 写入和 validator 任一失败都禁止发布。全部通过后，把同文件系统 sibling staging 原子 rename 为最终 canonical root。

迁移程序禁止调用 forward、backward、optimizer、正式 train/predict/evaluate。A3 预测等价的模型 forward 属于单独的 n20 acceptance test，不属于 converter。

如果最终 rename 失败，允许 external audit 保留授权证据，但不存在带 `result_manifest.json` 的正式目标，因此不会被误识别为已完成结果。

## 9. Legacy UQ 与指标语义

公式版本固定为 `legacy_upet_fge_v1`。设 K 个成员值为 `x_1 ... x_K`：

```text
mean(x) = (1/K) * sum_i x_i

population_std(x)
  = sqrt((1/K) * sum_i (x_i - mean(x))^2)
  = torch.std(x, unbiased=False)

scalar_gmd(x)
  = (1/K^2) * sum_i sum_j abs(x_i - x_j)

vector_gmd(x)
  = (1/K^2) * sum_i sum_j ||x_i - x_j||_2
```

GMD 使用所有有序成员对并包含 `i=j`。禁止替换为 `K-1` sample STD 或 `i<j` unordered pairs。

energy 保留 total 与 per-atom STD/GMD。force 保留 component、atom-vector、structure mean/max/q95 聚合。stress 不产生 UQ。

最终 MAE：

- energy：所有结构 `abs(mean_energy/n_atoms - reference_energy/n_atoms)` 的全局平均；
- force：所有原子、三个 Cartesian components 的绝对误差全局平均；
- stress：所有结构、3×3 components 的绝对误差全局平均。

指标均是绝对误差总和除以元素总数，不是 chunk MAE 的平均。

## 10. 等价、结构签名与验证

### 10.1 三层等价

1. key、identity、member order、structure/atom mapping、dtype、shape 必须精确相等。
2. 旧 chunks 合并后的 E/F/stress/reference tensor elements 必须精确相等。
3. 重新 reduction 或 A3 forward 的派生浮点比较使用 `rtol=2e-6, atol=2e-7`，并报告 max absolute 与 relative difference。

重组后的 `.pt` 容器不使用 byte equality 验收。

### 10.2 Normalized schema signature

同一 verifier 对 n20 native 与 full migrated result 生成 signature，包含：

- artifact roles 和完成关系；
- YAML/JSON key trees、schema/formula versions；
- tensor keys、dtypes、ranks 和 symbolic shapes；
- manifest 引用关系；
- K/S/A 的符号化维度。

signature 排除具体 K/S/A、tensor values、hashes、run/data identity 和 metrics。n20 K=2 与 full K=8 必须得到完全相同的 normalized signature。

## 11. 错误、warning、恢复与原子性

### 11.1 硬失败

- config/schema/formula/target/unit/scientific status 不匹配；
- checkpoint/data/artifact SHA 不匹配；
- base 或模型 contract 不匹配；
- trainable scope 不是精确 12 tensors/13,338 parameters；
- non-readout parameter 或 buffer 漂移；
- K、member order、endpoint step 或 chunk continuity 不匹配；
- structure/atom mapping、reference、dtype、shape 或 key 不匹配；
- NaN/Inf；
- A3 reconstruction 超出容差；
- native/migrated schema signature 不同；
- 正式 artifacts 含旧路径/source marker、W&B/log/plot/run-state 或 BootStrapping；
- destination 已存在、空间不足、staging 跨文件系统或原子 rename 不可用；
- external audit 无法在 publish 前完整写入。

禁止通过跳过 member、缩小 K、忽略 chunk、重排 structure 或降级公式继续。

### 11.2 Warning

- 可选 W&B 不可用；
- 常量输入导致 correlation undefined；
- diversity、risk-coverage 或 calibration 较弱，但 artifact contract 与数值正确。

这些 warning 必须进入结构化 diagnostics，不得被描述为良好校准结论。

### 11.3 原子性与恢复

- 单文件：同目录 temp → flush → fsync → `os.replace`。
- native incomplete root：没有 `result_manifest.json`，只可由 identity-matched `_work/` resume。
- native completed root：immutable；validator 只读。
- migration：同文件系统 sibling staging；validation 与 external audit 完成后一次 rename。
- migration failure：不得产生 completed formal root；只可保留 ignored failure/audit evidence。

## 12. 测试与远端验收

### 12.1 仓库测试入口

在 `tox.ini` 增加 `fge-tests` 环境，覆盖正式 tests 与 internal migration tests，并把 FGE 加入 lint 路径。远端至少运行：

```text
tox -e fge-tests
tox -e lint
tox -e upet-tests
```

pytest warnings 继续按项目配置视为错误；不得在实现内静默吞掉新 warning。

### 12.2 单元与契约测试

至少覆盖：

- strict config、unknown keys、sanitized paths、scientific states；
- checkpoint restart/best 选择和五项 loss contract；
- 12 tensors、13,338 parameters 与 frozen fingerprints；
- asymmetric triangular LR 起点、峰值、下降段和 endpoint；
- A3 weights-only pack/load、base/name/dtype/shape mismatch；
- population STD、ordered-pair `N^2` scalar/vector GMD、全局 E/F/stress MAE；
- canonical prediction keys/dtypes/shapes/order/mapping/reference；
- atomic file write、manifest-last、completed read-only validation；
- source/path/publication boundary；
- migration fault injection 和 no-compute guard。

### 12.3 n20 CPU 原生流程

远端使用同一个 20-structure n20 文件作为 train/validation/test：

- K=2；
- 2 cycles、2 epochs/cycle；
- batch size 4、`drop_last=true`；
- 5 updates/epoch、20 total optimizer updates；
- 正式 LR、EMA validation、frozen checks、A3 publish 和 reload checks 全部启用；
- W&B disabled；
- `path_feasibility_only=true`、`split_leakage=true`、`scientific_evaluation=false`。

按五个独立 CLI 执行 `preflight → train → predict → evaluate → validate`，得到完整 result manifest。该结果只证明路径与合同可用，不产生科学性能结论。

### 12.4 8-member A3 等价

在远端 CPU 上对全部 8 个旧 full members：

1. 校验 old/base SHA 并提取 A3；
2. 证明 non-readout state 精确相等；
3. 对同一 n20 输入分别执行旧 full checkpoint 与 base+A3 inference；
4. 比较 energy、forces、stress，使用 `rtol=2e-6, atol=2e-7`；
5. 报告每个 member/target 的最大绝对与相对差。

8/8 通过才允许 full publish。

### 12.5 Full migration validation

只包含 8 个正式 members 与独立 `matpes_test_full`。必须证明：

- canonical prediction elements 与旧 chunks 精确相等；
- 旧 ensemble/UQ/metrics 被保留；
- validator 独立复算通过；
- 三项 MAE 与 Precheck 精确一致；
- full migrated 与 n20 native schema signature 精确相等；
- formal root 不含旧 source、W&B、日志、plots、run-state、失败或 BootStrapping 内容；
- old source hashes before/after 精确一致；
- BootStrapping 与远端其他未提交内容未触碰。

## 13. 完成定义

只有全部满足以下条件，任务才可宣告完成：

1. 本设计与重新生成的 implementation plan 均经用户复核；
2. 正式 FGE、tests、README、configs 和隔离的 internal migration 已提交；
3. publication allowlist 与 dependency direction 检查通过；
4. 远端 `fge-tests`、lint 和 UPET regression 通过；
5. n20 CPU 五阶段完整通过；
6. 8/8 A3 n20 E/F/stress 等价通过；
7. full migration 原子发布，external audit 完整；
8. prediction/UQ/metrics/MAE/schema signature 全部通过；
9. old source hashes 不变，BootStrapping 与无关文件未触碰；
10. Git 中不存在 outputs、大数据、checkpoint、W&B、日志、plots 或 failure artifacts；
11. 最终本地与远端验证工作树没有本任务引入的未提交 tracked changes。

## 14. 相对前一版设计的准确性修正

本轮保留此前确认的技术决策，但明确修正以下内容：

1. 目标分支由旧文档中的 `ConfidenceHead` 更正为当前 `FGE`。
2. metatrain 约束以当前 `pyproject.toml` 的 `>=2026.3.1,<2026.4` 为准，不采用过期 AGENTS.md 描述。
3. 正式模块清单补充 errors、manifests、data、preflight 的独立职责。
4. 明确区分 artifact writer、validator 和历史 training code identity；不伪造旧训练代码身份。
5. preflight report 增加 `basis`，准确区分 runtime-input 检查与 canonical-artifact 检查，同时保持 schema 一致。
6. 明确 completed result 的再次 validation 完全只读。
7. external migration audit 必须在 canonical rename 前完整可用；audit 写入失败禁止发布。
8. 明确 A3 等价 forward 属于独立 acceptance test，不属于禁止计算的 converter。

本文书面复核通过后，下一步只能调用 `superpowers:writing-plans` 重新生成实施计划。旧实施计划 `797fba7` 不恢复、不复用为执行授权。
