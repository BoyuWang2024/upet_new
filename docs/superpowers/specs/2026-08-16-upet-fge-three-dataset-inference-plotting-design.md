# UPET FGE 三数据集复用推理、UQ 与绘图设计

日期：2026-08-16

状态：已完成逐节确认，等待书面审阅

## 1. 目标

复用已经完成并通过正式验证的 K=8 FGE ensemble，在不重新训练、不修改既有结果的前提下完成以下工作：

1. 直接使用现有 `matpes_test` prediction/evaluation 生成参考风格图；
2. 在远端仅对 `mad-test.xyz` 执行 prediction、UQ 与绘图；
3. 在远端仅对 `matpes_train.extxyz` 执行 prediction、UQ 与绘图；
4. 每个数据集独立发布图，不制作跨数据集组合图；
5. 不生成 checkpoint/member 数量相关性曲线；
6. 最大限度复用当前 FGE 的 checkpoint、member、prediction、UQ、validation 和安全 artifact 模块，以及 `carnet_new` 中已经验证的 FGE 单面板绘图算法。

本地开发固定使用当前 `Plots` 分支。已有未跟踪文件（当前包括 `.idea/` 和 `Uncertainty_Quantification/ConfidenceHead/tests/test_external_prediction_publication.py`）属于用户，不修改、不暂存、不提交。

## 2. 已确认的基础事实

### 2.1 现有 FGE ensemble 与 MATPES test 结果

远端已完成且通过独立 release review 的正式结果为：

```text
/home/bywang/code/UQ/upet_new_fge_test/Uncertainty_Quantification/FGE/outputs/
upet-FGE-CKPT-UQ-v1.0-full-v3
```

它包含：

- K=8 的 `member_001.pt` 至 `member_008.pt`；
- training manifest、base checkpoint 身份和全部 member SHA；
- 已完成的 `matpes_test` prediction；
- 已完成的 ensemble、uncertainty、metrics 与 validation；
- `PASS/read_only` completed-result 验证链。

正式结果的 `result_manifest.json` SHA-256 为：

```text
b1f4a3c5c7713b4be369b467e98e62d59bc23a13aefceffbd49ac51773b6dc86
```

该目录是本设计唯一可信的 ensemble 来源，必须保持只读。`matpes_test` 数据 SHA-256 为：

```text
1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d
```

### 2.2 新推理数据

用户指定的本地文件和远端已有副本逐字节一致，因此无需上传或复制数据。

#### MAD test

本地：

```text
/home/lilong/code/UQ/upet_new/data/dataset/mad-test.xyz
```

远端：

```text
/home/bywang/code/UQ/mace_new-plots/data/dataset/mad-test.xyz
```

身份与结构：

| 字段 | 值 |
|---|---|
| SHA-256 | `d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b` |
| 文件大小 | `30,505,970` bytes |
| 结构数 | `8,255` |
| reference | energy、forces |
| 缺失 reference | stress |
| 结构 ID | 源文件无 `id`/`structure_id` |

#### MATPES train

本地：

```text
/home/lilong/code/UQ/upet_new/data/dataset/matpes_train.extxyz
```

远端：

```text
/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_train.extxyz
```

身份与结构：

| 字段 | 值 |
|---|---|
| SHA-256 | `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec` |
| 文件大小 | `397,915,595` bytes |
| 结构数 | `348,780` |
| reference | energy、forces、stress |
| 结构 ID | `structure_id` |

### 2.3 远端运行资源

远端主机有 112 个逻辑 CPU、约 250 GiB RAM，但 NVIDIA 驱动不可用。因此正式推理采用 CPU 分块模式；不得依赖 CUDA，也不得把 348,780 个结构一次性物化为完整 ASE/metatrain system 列表。

### 2.4 参考图与原始实现

参考图位于：

```text
/home/lilong/code/carnet_new/Uncertainty_Quantification/Plots/FGE
```

对应的原始实现和测试已经定位到：

```text
/home/lilong/code/carnet_new/Uncertainty_Quantification/FGE/fge/plot_analysis.py
/home/lilong/code/carnet_new/Uncertainty_Quantification/FGE/fge/plot_rendering.py
/home/lilong/code/carnet_new/Uncertainty_Quantification/FGE/tests/
```

本设计采用该实现的 raw STD 分析粒度和单面板视觉参数，但不运行时依赖另一个仓库，也不移植 checkpoint sweep。

## 3. 范围与非目标

### 3.1 本次范围

- 为当前 FGE 增加只读复用既有 ensemble 的 inference-only 运行模式；
- 新增可恢复的 CPU 分块 prediction；
- 支持 reference availability，允许 MAD 缺失 stress reference；
- 对新 prediction 计算 K=8 raw population STD 与 residual；
- 从现有 MATPES test 正式 prediction 直接分析和绘图，不重复推理；
- 为三个数据集分别生成 raw Energy/Force/Stress（可用时）图；
- 原子发布 prediction chunks、UQ、统计、图和 manifest；
- 本地测试、远端真实 chunk smoke、正式远端运行、最终只读验收和图文件同步。

### 3.2 非目标

- 不运行 `train_fge`；
- 不执行 backward、optimizer step 或任何参数更新；
- 不重新生成 `matpes_test` prediction；
- 不复制、硬链接、改写或重新打包 K=8 member 文件；
- 不修改 `upet-FGE-CKPT-UQ-v1.0-full-v3`；
- 不制作多数据集叠加图；
- 不生成 checkpoint/member 数量相关性图；
- 不为 MAD 伪造 stress reference、零张量或 NaN 占位；
- 不把大型 prediction/UQ chunk 文件同步回本地；
- 不在运行时导入 `carnet_new`。

## 4. 方案选择

### 4.1 采用：共享 ensemble 的 inference-only 运行

新运行只读引用一个已经验证的 ensemble authority，同时为每个数据集发布独立 prediction/UQ 产物。优势是：

- 不复制 member；
- 不伪装成新的 training result；
- 明确绑定 ensemble manifest、member SHA、数据 SHA 和代码身份；
- 可为大数据集提供分块恢复；
- `matpes_test` 可以走纯只读适配器。

### 4.2 不采用：复制完整 training 树

复制 training manifest 和 member 会制造重复文件，并且既有 training manifest 与原始完整 config 身份绑定。为了换 test 数据而重写 training manifest 会破坏来源语义，因此不采用。

### 4.3 不采用：一次性远端脚本

一次性脚本缺少正式 manifest、分块恢复、身份审计和完整验证，难以复现，不采用。

## 5. 总体架构

```text
已验证 K=8 FGE ensemble（只读）
        │
        ├─ matpes_test ──复用现有 prediction──分析/UQ补充──绘图
        │
        ├─ mad_test ─────分块 predict──UQ聚合──Energy/Force 绘图
        │
        └─ matpes_train ─分块 predict──UQ聚合──Energy/Force/Stress 绘图
```

推理、UQ、绘图为三个独立阶段：

- predict 可以读取 checkpoint/member 和数据；
- UQ 只能读取 prediction artifacts；
- plot 只能读取已验证 UQ/residual artifacts，不能加载模型。

三个数据集使用独立输出目录，互不覆盖。`matpes_test` 的现有正式目录继续作为 completed result，不物化为新的 prediction chunks。

## 6. Ensemble 复用契约

inference-only 配置必须显式包含：

```yaml
ensemble:
  root: /absolute/remote/path/to/upet-FGE-CKPT-UQ-v1.0-full-v3
  result_manifest_sha256: b1f4a3c5c7713b4be369b467e98e62d59bc23a13aefceffbd49ac51773b6dc86
```

运行前必须：

1. 对 ensemble root 执行 completed read-only validation；
2. 验证 result manifest SHA；
3. 重开 training manifest；
4. 验证 base checkpoint SHA；
5. 验证 member ID 精确为 `member_001..member_008`；
6. 重算 8 个 member SHA；
7. 验证 A3 readout contract；
8. 记录当前 validator/writer 代码身份。

inference-only manifest 只保存规范身份、哈希和相对 member 角色，不保存可迁移性差的来源绝对路径。运行时绝对路径只出现在 resolved execution record 中，不进入跨路径 identity。

任何来源身份差异都硬失败，禁止“找到八个同名文件就继续”。

## 7. 数据预检与结构 ID

### 7.1 全量流式预检

正式 predict 前逐结构流式检查：

- energy 和 forces 每个结构都存在且 finite；
- force 形状精确为 `[n_atoms, 3]`；
- stress 必须全有或全无，禁止 mixed availability；
- 原子数为正；
- 原子序数有效；
- cell、positions、PBC 和标签均可读取；
- 数据 SHA 与配置完全一致；
- 预检前后文件 SHA、size、mtime 不变。

### 7.2 ID 规则

- `matpes_train` 使用源文件中的 `structure_id`，按源文件顺序保留，要求非空且唯一；
- `mad_test` 不修改源文件，使用文件顺序生成：

```text
mad_test:00000000
mad_test:00000001
...
```

该 ID 方案与数据 SHA、split=`mad_test` 和结构总数共同绑定。相同索引但不同数据 SHA 不构成相同数据身份。

新 inference-only schema 明确允许“有顺序且唯一”的 ID，不强制为了字典序而一次性加载/排序全部结构。

## 8. CPU 分块 Prediction

### 8.1 Chunk 规划

每个 chunk 同时受两个配置约束：

- 最大结构数，初始默认 `128`；
- 最大原子数，初始默认 `4096`。

只要加入下一个结构会超过任一上限，就结束当前 chunk。单个结构原子数超过 atom 上限时允许形成单结构 chunk，但必须显式记录 oversize 状态；不得丢弃结构。

正式参数在两个新数据集各自的真实单 chunk smoke 后锁定。调整只允许发生在正式 run identity 建立之前。

### 8.2 Chunk-major 执行

每个正式 predict 进程只加载并验证一次 base model；恢复后的进程重新加载一次。每个 chunk：

1. 读取一组 ASE/metatrain systems；
2. 构建一次请求的 neighbor lists；
3. 复用该进程已经加载并验证的 base model；
4. 按 `member_001..member_008` 顺序恢复并应用 A3 readout；
5. 对同一 prepared systems 依次完成 8 次 forward；
6. 组装 K=8 chunk prediction；
7. 重新验证 schema、shape、member 顺序和 reference；
8. 原子发布 chunk 与 chunk manifest。

member 应用很小，chunk-major 可以避免对 398 MB 数据做八次完整解析，也避免一次性持有全数据集系统对象。

### 8.3 Chunk payload

每个 prediction chunk 至少包含：

- `energy_prediction: [K, S_chunk]`；
- `forces_prediction: [K, A_chunk, 3]`；
- `stress_prediction: [K, S_chunk, 3, 3]`；
- 可用的 energy/forces/stress references；
- `reference_availability`；
- `n_atoms`、`structure_offsets`、`atomic_numbers`、`structure_mapping`；
- ordered member IDs 与 ordered structure IDs；
- chunk index、全局结构范围、全局原子范围；
- dataset identity、ensemble identity、代码身份；
- tensor dtype/shape 和文件 SHA。

缺失 stress reference 使用 `reference_availability.stress=false` 表示；不保存形状兼容的假张量。

### 8.4 输出布局

```text
Uncertainty_Quantification/FGE/outputs/inference/<run-id>/
├── run_manifest.json
├── prediction/
│   ├── manifest.json
│   └── chunks/
│       ├── chunk_000000.pt
│       └── ...
├── uncertainty/
│   ├── manifest.json
│   └── chunks/
│       ├── chunk_000000.pt
│       └── ...
└── metrics.json
```

`run-id` 至少绑定 dataset label、dataset SHA、ensemble result manifest SHA、K、chunk policy 和代码身份。

### 8.5 恢复

- chunk 文件、chunk record 和 SHA 全部验证完成后才算成功；
- 重启时逐 chunk 重开；
- 完全一致的 chunk 只读复用；
- 缺失、损坏、范围不连续或身份不一致时硬失败；
- 失败临时文件不删除、不纳入 manifest；
- 不允许通过覆盖冲突 chunk“修复”结果；必须使用新的 run identity 或明确恢复同一未完成 run。

## 9. UQ 与 residual 口径

### 9.1 Raw population STD

只使用 equal-weight K=8 raw member predictions：

- Energy uncertainty：先将每个 member 的总能量除以该结构原子数，再对 K 维计算 population STD；
- Force uncertainty：对每个原子每个 Cartesian 分量在 K 维计算 population STD；
- Stress uncertainty：先对每个 3×3 stress 做 `0.5 * (S + S^T)` 对称化，再按 `[xx, yy, zz, yz, xz, xy]` 映射为 Voigt-6，最后对每个分量在 K 维计算 population STD。

公式必须调用或提升当前 FGE 的公共纯函数，不在绘图模块中复制。

这里有意沿用当前 `upet_new` 正式 FGE 的 population STD（分母 `K`、`unbiased=False`），不采用参考 `carnet_new` 绘图代码中的 sample STD（分母 `K-1`）。参考工程只作为粒度、过滤、分析与视觉样式的来源。公式版本必须写入 UQ manifest 和 plot manifest，防止两种 STD 被混淆。

### 9.2 Residual

- Energy residual：`abs(mean(E_total)/n_atoms - E_reference/n_atoms)`；
- Force residual：`abs(mean(F_component) - F_reference_component)`；
- Stress residual：`abs(mean(S_voigt_component) - S_reference_voigt_component)`。

Energy、Force 和 Stress 的 uncertainty/residual 必须形状完全一致，flatten 只改变视图，不改变配对顺序。

MAD 没有 stress reference：

- 可以保留模型 stress prediction 和无监督 stress STD；
- 不生成 stress residual；
- 不把 stress 纳入相关性或图；
- manifest 显式记录 stress plot unavailable 的原因。

### 9.3 聚合

UQ 阶段逐个读取 prediction chunk，生成对应 uncertainty chunk。数据集级 `metrics.json` 保存：

- 每域总样本数；
- finite/positive log-valid 数；
- NaN、Inf、zero、negative 排除数；
- Spearman rho；
- Pearson r(log10)；
- uncertainty、residual 及二者比值的描述统计。

计算全局统计时可以拼接数值数组或使用经证明等价的流式算法，但不得平均 chunk 相关系数来冒充全局相关性。

## 10. 绘图设计

### 10.1 输入适配

`plot_analysis.py` 支持两种只读输入：

1. 现有正式 FGE completed result（用于 `matpes_test`）；
2. 新 inference-only result（用于 `mad_test`、`matpes_train`）。

两种输入都适配为相同的 `(uncertainty, absolute residual)` domain 数据。`matpes_test` 所需 stress STD 可以直接从现有 K=8 `prediction/test_raw.pt` 重算；不重新推理、不修改既有 evaluation 目录。

### 10.2 参考实现复用

从 `carnet_new` 移植并适配：

- log filter；
- Spearman/log10 Pearson；
- 固定随机抽样；
- 2D density histogram；
- Gaussian smoothing 与概率质量等高线；
- 单面板 Matplotlib 样式。

不运行时导入另一个仓库。移植代码保留来源说明并迁移对应测试。文件发布不使用参考仓库的旧实现，而使用当前 FGE 已审查的 descriptor-bound、原子、持久化 artifact writer。

### 10.3 视觉参数

| 参数 | 固定值 |
|---|---|
| figure size | `7 × 7` inch |
| PNG DPI | `300` |
| formats | `png`, `pdf` |
| axis scale | x/y log |
| log margin | `0.05` |
| color | `#f28e2b` |
| scatter size | `2.0` |
| scatter alpha | `0.035` |
| scatter maximum | `20,000` |
| random seed | `20260714` |
| density grid | `160 × 160` |
| Gaussian sigma | `1.2` |
| contour masses | `0.50, 0.70, 0.85, 0.95, 0.99` |

每张图：

- x/y 使用同一低/高界限，保持正方形；
- 灰色区域表示 `absolute residual <= uncertainty`；
- 黑色虚线表示 `absolute residual = uncertainty`；
- 标注 Spearman rho、Pearson r(log10) 和 valid/original count；
- 抽样只影响散点显示，不影响统计、过滤、密度或等高线。

### 10.4 输出

```text
Uncertainty_Quantification/Plots/FGE/
├── matpes_test/
│   ├── raw_energy_uncertainty_vs_residual.png
│   ├── raw_energy_uncertainty_vs_residual.pdf
│   ├── raw_force_uncertainty_vs_residual.png
│   ├── raw_force_uncertainty_vs_residual.pdf
│   ├── raw_stress_uncertainty_vs_residual.png
│   ├── raw_stress_uncertainty_vs_residual.pdf
│   ├── raw_single_stats.json
│   └── plot_manifest.json
├── mad_test/
│   ├── raw_energy_uncertainty_vs_residual.png
│   ├── raw_energy_uncertainty_vs_residual.pdf
│   ├── raw_force_uncertainty_vs_residual.png
│   ├── raw_force_uncertainty_vs_residual.pdf
│   ├── raw_single_stats.json
│   └── plot_manifest.json
└── matpes_train/
    ├── raw_energy_uncertainty_vs_residual.png
    ├── raw_energy_uncertainty_vs_residual.pdf
    ├── raw_force_uncertainty_vs_residual.png
    ├── raw_force_uncertainty_vs_residual.pdf
    ├── raw_stress_uncertainty_vs_residual.png
    ├── raw_stress_uncertainty_vs_residual.pdf
    ├── raw_single_stats.json
    └── plot_manifest.json
```

不得生成 `raw_std_correlation_vs_checkpoint_count.*` 或任何 checkpoint sweep CSV/JSON。

`raw_single_stats.json` 保存完整分析审计。`plot_manifest.json` 绑定：

- 数据身份；
- ensemble/result manifest 身份；
- prediction/UQ 输入 SHA；
- active domains；
- 绘图参数与随机种子；
- 代码身份；
- 每个输出文件的 path、size、SHA。

输出存在且完全一致时只读返回；存在冲突或篡改时硬失败，不覆盖。

## 11. 代码组织

### 11.1 直接复用

- `fge/artifacts.py`：安全路径、SHA、原子发布、fsync；
- `fge/members.py`：A3 member 加载、验证和应用；
- `fge/prediction.py`：PET base 加载、member 恢复和模型输出转换；
- `fge/uncertainty.py`：STD 等纯公式；
- `fge/evaluation.py`：residual、相关性和统计定义；
- `fge/validation.py`：现有 completed result 只读验证。

必要时只把私有纯函数提升为稳定内部接口，不复制公式。

### 11.2 新增文件

```text
Uncertainty_Quantification/FGE/
├── fge/
│   ├── inference_only.py
│   ├── inference_validation.py
│   ├── plot_analysis.py
│   └── plot_rendering.py
├── scripts/
│   ├── predict_dataset.py
│   ├── evaluate_dataset.py
│   └── plot_dataset.py
└── configs/
    ├── inference_mad_test.yaml
    ├── inference_matpes_train.yaml
    ├── plot_matpes_test.yaml
    ├── plot_mad_test.yaml
    └── plot_matpes_train.yaml
```

职责：

- `inference_only.py`：数据扫描、chunk 规划、ensemble 复用、分块预测和恢复；
- `inference_validation.py`：验证 chunk 连续性、identity、SHA、optional reference 与 UQ；
- `plot_analysis.py`：把两种结果 schema 适配为统一 domain 分析；
- `plot_rendering.py`：纯渲染，不读取 checkpoint；
- 三个 CLI 分别执行 predict、UQ、plot，不提供会跨阶段的 `run-all`。

绘图使用独立严格配置，不向已有 `FGEConfig` 追加字段，避免改变正式 train/predict/evaluate identity。

## 12. 远端执行设计

### 12.1 部署

只部署已经提交并通过本地测试的当前分支代码到远端隔离 checkout：

```text
/home/bywang/code/UQ/upet_new_fge_test
```

不得切换、覆盖或清理用户的其他远端工作区。每次运行前核对本地 commit 与远端 tracked 文件 SHA。

大型 prediction/UQ artifacts 保留在远端 ignored `FGE/outputs/inference/`。只有最终 PNG/PDF、统计 JSON 和 plot manifest 在完整验收后同步回当前本地 `Uncertainty_Quantification/Plots/FGE/`。

### 12.2 CPU 参数

- 单个模型进程；
- 明确设置 Torch/OMP/MKL 线程；
- 先对每个新数据集运行一个真实 chunk；
- 记录 wall time、峰值 RSS、结构数和原子数；
- smoke 通过后才冻结 chunk/线程参数并建立正式 run identity；
- 不通过登录 shell 的“无错误输出”推断成功，必须检查 manifest 和 exit code。

### 12.3 阶段顺序

```text
1. 验证 ensemble 与三个数据身份
2. matpes_test 现有结果只读验证
3. matpes_test UQ适配与绘图
4. mad_test 单chunk smoke
5. mad_test 完整predict
6. mad_test UQ与绘图
7. matpes_train 单chunk基准
8. matpes_train 完整可恢复predict
9. matpes_train UQ与绘图
10. 三个结果最终只读验收与图文件同步
```

`matpes_test` 阶段安装 no-inference guard；两个新数据集安装 no-training guard。任何被禁止调用都会立即硬失败。

## 13. 验证与失败策略

### 13.1 单元与集成测试

采用 TDD，至少覆盖：

- MAD 确定性索引 ID；
- mixed stress availability 拒绝；
- 双 chunk 上限和 oversize 单结构；
- chunk 缺失、重复、乱序、重叠、gap 和篡改；
- ensemble/result/member 身份不一致；
- 中断后只复用已验证 chunk；
- K=8 member 顺序；
- chunk UQ 与单体 UQ 等价；
- Energy per-atom、Force component、Stress Voigt-6 粒度；
- log filter/exclusion count；
- 参考渲染参数与固定抽样；
- MAD 精确不生成 stress 图；
- 精确不生成 checkpoint-count 图；
- MATPES test 路径不加载模型；
- no-training、no-backward、no-optimizer；
- 输出 manifest 与磁盘文件清单精确相等。

### 13.2 正式验收

正式结果必须满足：

1. 数据、ensemble、base checkpoint 和 8 个 member 的 SHA/size/mtime 前后不变；
2. MATPES test 推理调用次数为零；
3. 两个新数据集每个源结构恰好出现一次，无遗漏、重复或重排；
4. 所有 chunk 均为 K=8 且 member 顺序精确；
5. chunk 全局结构/原子范围连续；
6. 抽样 chunk 独立重算 mean/std/residual 与产物一致；
7. synthetic/小切片上的分块与单体计算等价；
8. MAD `stress_reference=false`，没有 stress residual 图；
9. plot 阶段不导入/加载模型；
10. PNG/PDF 可打开，数量、文件名、size、SHA 与 manifest 一致；
11. 重复只读验证不改变任何正式文件 bytes/mtime；
12. 完整 FGE 测试、Ruff format/check 和 mypy 通过；
13. 独立 reviewer 给出无 Critical/Important 的批准结论。

### 13.3 失败处理

- 任何 identity、SHA、schema、shape、finite 或路径安全错误都硬失败；
- 不用 NaN、空字符串、零张量或缺省图掩盖错误；
- 不自动删除失败 staging/chunk；
- 失败结果没有 completed manifest，不得作为后续输入；
- 输出冲突不得覆盖；
- 长任务中断后只通过正式 chunk manifest 恢复；
- 未经测试、commit 和独立 review 的修复不得直接应用到正式远端结果。

## 14. 完成标准

本任务只有在以下全部成立时完成：

- 两个新数据集的 predict 和 UQ 均由已验证 K=8 ensemble 产生；
- 没有训练或参数更新；
- MATPES test 没有重复推理；
- 三个独立绘图目录按约定发布；
- MAD 只有 Energy/Force 图；
- MATPES test/train 均有 Energy/Force/Stress 图；
- 没有 checkpoint-count 图；
- 远端大型产物保留，本地只获得最终图和审计 JSON；
- 所有验证与独立审查通过；
- 设计、实现计划、代码和最终报告均已提交到当前分支。
