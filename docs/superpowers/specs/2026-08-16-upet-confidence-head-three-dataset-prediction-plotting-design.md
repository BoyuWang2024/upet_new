# UPET ConfidenceHead 三数据集预测、UQ 与绘图设计

日期：2026-08-16

状态：已完成逐节确认，等待书面审阅

## 1. 背景与目标

远端 `bywang@121.48.164.204` 的 UPET ConfidenceHead 已完成九个正式任务：

- 一个 force-only 任务，`force_coefficient=1`、`energy_coefficient=0`，力目标为 `atom_mean`；
- 八个 energy-only 任务，`force_coefficient=0`、`energy_coefficient=1`，能量 cumulant order 为 1–8。

本次不重新训练 ConfidenceHead，只复用九个任务的 `best.pt` 和现有 UPET 特征/预测，完成三个数据集的预测、UQ 统计与 Carnet 风格绘图：

1. `matpes_test`：直接复用现有九份 evaluation 绘图，不重新推理；
2. `mad-test.xyz`：在远端提取一次公共 UPET 特征和基础预测，再运行九个 ConfidenceHead；
3. `matpes_train.extxyz`：复用远端现有训练 cache 的 `train` split，再运行九个 ConfidenceHead；
4. 为三个数据集分别生成逐任务图、数据集内部汇总图，并生成跨数据集相关性汇总图。

用户最初给出的 `matpes_train.json` 不存在，已经确认改用：

```text
/home/lilong/code/UQ/upet_new/data/dataset/matpes_train.extxyz
```

## 2. 已确认事实

### 2.1 数据身份

| 数据集 | 本地路径 | SHA-256 |
|---|---|---|
| MATPES test | `data/dataset/matpes_test.extxyz` | `1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d` |
| MATPES train | `data/dataset/matpes_train.extxyz` | `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec` |
| MAD test | `data/dataset/mad-test.xyz` | `d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b` |

两个新推理数据集均包含标量总能量 `energy` 与形状为 `[N, 3]` 的 `forces`，可计算基础模型残差、ConfidenceHead UQ 与相关性。

### 2.2 远端已有 cache

远端 ConfidenceHead cache 中已经存在完整 MATPES 三分割：

| split | structures | atoms | SHA-256 |
|---|---:|---:|---|
| train | 348780 | 2753112 | `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec` |
| validation | 19370 | 152959 | `5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef` |
| test | 19374 | 149321 | `1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d` |

远端存在两个 complete cache identity，二者包含相同的数据分割。正式复用时不凭文件名任选其一，而是对每个 run 使用其 manifest 声明的原始 `cache_id`，并验证对应 split 的数据 SHA、checkpoint、readout 名称、特征维度和 dtype。

### 2.3 九个正式 run

远端 `Uncertainty_Quantification/ConfidenceHead/outputs/runs` 中恰好存在所需的八个 energy-only order 1–8 run 和一个 force-only run。九个 run 均有 `checkpoints/best.pt` 和现有 `evaluation/manifest.json`。

### 2.4 分支与部署边界

本地当前分支为 `Plots`，并包含远端 `ConfidenceHead` 当前 commit `11036e7`，但本地分支还包含其他功能提交。实现必须形成独立提交；远端只 cherry-pick 本次提交，不整体切换到 `Plots`，不触碰远端已有未跟踪配置、Slurm 文件、日志、W&B 文件和训练结果。

## 3. 范围与非目标

### 3.1 本次范围

- 新增严格配置驱动的外部数据集 prediction/UQ 工作流；
- 复用现有 `adapters`、`features`、`cache`、`checkpoint`、`model`、`binning`、`metrics`、artifact writer 与 verifier；
- 复用 MATPES test evaluation 和 MATPES train cache；
- 只为 MAD test 新建一次基础 UPET cache；
- 对两个新数据集运行八个 energy head 和一个 force head；
- 扩展现有 `plot_analysis.py`，支持 dataset-scoped prediction；
- 输出单任务、单数据集汇总和跨数据集汇总图；
- 本地进行静态验证和合成数据测试，远端进行真实数据推理与绘图。

### 3.2 非目标

- 不重新训练或微调 UPET/ConfidenceHead；
- 不修改九个任务的超参数、checkpoint 或训练身份；
- 不覆盖或重写 `evaluation/test_predictions.pt`；
- 不为 MATPES train 重新提取 UPET 特征；
- 不为 MATPES test 重复推理；
- 不引入 ensemble、bootstrap、LLPR 或新的 UQ 公式；
- 不绘制训练曲线或置信区间；
- 不上传推理日志到 W&B；
- 不复制九份临时训练配置。

## 4. 方案选择

采用独立的外部数据集预测工作流。

不采用“替换训练配置中的 test split 后调用现有 evaluate”，因为现有 evaluate 会严格绑定训练配置、三分割 cache、run identity 与 checkpoint；替换数据后合法地无法通过校验。也不采用全局跨方法推理数据库，因为本次范围不需要新的查询层和统一数据库。

新工作流必须同时满足两类身份：

1. **训练身份**：`best.pt` 必须继续与原 run manifest、原训练 cache ID、binning ID、model/loss ID 一致；
2. **外部输入兼容性**：新数据 cache 不伪装成训练 cache，只验证基础 checkpoint SHA、readout 名称、特征维度、dtype、张量布局和目标语义与该 head 兼容。

这样既不绕过训练产物校验，也允许一个训练好的 head 合法预测新数据。

## 5. 总体数据流

```text
extxyz 数据集
    ↓
基础 UPET checkpoint
    ↓
复用 adapters/features/cache 提取逻辑
    ├── energy prediction
    ├── force prediction
    ├── energy readout features
    ├── force readout features
    ├── energy reference
    └── force reference
    ↓ 每个新数据集最多提取一次
九个已完成 ConfidenceHead best.pt
    ├── energy order 1–8
    └── force atom-mean
    ↓
logits、argmax bin、expected error、observed error、metrics
    ↓
dataset-scoped prediction artifacts
    ↓
复用 plot_analysis.py
    ↓
单任务图、数据集汇总图、跨数据集汇总图
```

三个数据集采用不同入口但汇合到同一个规范化 `PredictionSeries`：

- `matpes_test`：从现有 `evaluation/test_predictions.pt` 适配；
- `matpes_train`：从每个 run 声明的原 cache `train` split 运行 head；
- `mad-test`：从新建的单数据集 cache 运行 head。

## 6. 配置与命令设计

新增一份正式远端配置和一份 smoke 配置：

```text
configs/predict_external_gpu.yaml
configs/predict_external_smoke.yaml
```

正式配置只声明：

- 基础 UPET checkpoint 路径与 SHA；
- 三个具名数据集及其来源模式；
- MAD test extxyz 路径与 SHA；
- MATPES train 要复用的 cache split 和数据 SHA；
- MATPES test 要复用的 evaluation；
- 九个 run 的根目录或显式 run 列表；
- batch size、device、输出根目录与绘图根目录。

数据源模式使用严格互斥联合类型：

```yaml
datasets:
  matpes_test:
    source: existing_evaluation
    expected_sha256: 1ffcdcad...
  matpes_train:
    source: cache_split
    split: train
    expected_sha256: 12ff9403...
  mad_test:
    source: extxyz
    path: data/dataset/mad-test.xyz
    expected_sha256: d9a12802...
```

正式入口拆为两个薄脚本：

```bash
python scripts/predict_external_datasets.py \
  --config configs/predict_external_gpu.yaml

python scripts/plot_prediction_datasets.py \
  --config configs/predict_external_gpu.yaml
```

第一条命令只负责 preflight、cache、head prediction 和 UQ；第二条命令只消费已发布 artifact 绘图。命令支持显式 `--dataset` 和 `--run-dir` 缩小范围，但默认严格要求九个目标 run 完整且唯一。

## 7. 公共特征与基础预测复用

### 7.1 MATPES test

不读取 extxyz、不运行基础 UPET、不运行 head。只验证并加载已有九份 evaluation。

### 7.2 MATPES train

不上传本地 380 MB 文件，不重建约 47 GB 的现有 cache。对每个 run：

1. 从 run manifest 取得其声明的训练 `cache_id`；
2. 验证 cache complete 状态和声明文件 SHA；
3. 验证 `train` split SHA 等于 `12ff9403...08cec`；
4. 验证基础 checkpoint、readouts、feature dimensions、dtype 和 execution schema；
5. 以只读方式打开 `CachedSplitDataset(..., "train", cache_id)`。

如果两个 run 使用不同的合法 cache identity，则分别读取其原 cache，不擅自把一个 cache 宣称为另一个 identity。实现可在验证张量级兼容后共享只读映射，但不得改写 manifest。

### 7.3 MAD test

将 `mad-test.xyz` 上传到远端 staging 路径，校验 SHA 后再原子发布。新增单数据集 cache builder，但底层复用现有：

- extxyz label loader；
- UPET checkpoint loader；
- energy/force 分离 readout；
- feature extraction；
- memmap writer；
- artifact hashing 与原子发布。

单数据集 cache 具有独立 schema 和 manifest，不伪造 train/validation/test 三分割。重复执行时，身份完全一致则复用；目标已存在但不一致则停止，不覆盖。

## 8. ConfidenceHead prediction 与 UQ 语义

### 8.1 Head 加载

每个 run 必须通过完整校验：

- run 状态为 complete；
- `resolved_config.yaml`、`binning.json`、`best.pt` 都是 manifest 声明的 artifact；
- artifact SHA 正确；
- checkpoint schema、run/config/cache/binning/model-loss identity 与原训练 run 一致；
- 一个且仅一个目标 active；
- energy orders 恰好覆盖 1–8；
- force 目标为 `atom_mean`。

模型输入严格分离：energy head 只读取 energy readout features，force head 只读取 force readout features。

### 8.2 能量

每个结构产生一个能量样本。observed error 定义为：

```text
abs(predicted_total_energy - reference_total_energy) / atom_count
```

单位为 `eV/atom`。绘图和统计阶段不得再次除以原子数。

### 8.3 力

每个原子产生一个力样本。observed error 定义为：

```text
mean(abs(predicted_force_xyz - reference_force_xyz), axis=xyz)
```

单位为 `eV/Å`。不得展开成三个独立 component 样本。

### 8.4 UQ

继续使用训练时固定线性分箱和 bin representatives。对 logits 计算分类概率，再得到：

```text
expected_error = sum(softmax(logits) * bin_representatives)
```

argmax bin 用于箱线图分组；labels 由 observed error 与原训练 thresholds 得到。不得根据新数据重新拟合边界或 representatives。

### 8.5 指标

沿用现有 classification metrics，并对 expected error 与 observed error 计算 Pearson 和 Spearman。所有指标只从已发布 prediction artifact 重算，不从图形或日志反推。

## 9. Artifact 与目录组织

### 9.1 Cache

```text
Uncertainty_Quantification/ConfidenceHead/outputs/
└── prediction_cache/
    └── mad-test-<dataset-identity>/
        ├── manifest.json
        └── shards-or-memmaps...
```

MATPES train 继续读取现有 `outputs/cache/<cache-id>`，不物化副本。

### 9.2 每个 run 的外部预测

```text
outputs/runs/<run>/predictions/<dataset-name>/
├── predictions.pt
├── metrics.json
├── energy_bin_summary.csv 或 force_bin_summary.csv
└── manifest.json
```

`predictions.pt` 保持现有绘图字段兼容，并保存当前 active target 所需的基础预测和参考值：

- `structure_ids`、`atom_offsets`；
- active target 的基础 UPET prediction/reference；
- `<target>_logits`、`<target>_labels`；
- `<target>_observed_errors`、`<target>_expected_errors`；
- `<target>_representatives`；
- force target mode 与 error definition（force-only）。

manifest 至少绑定：

- schema version、dataset name/source/SHA/counts；
- base checkpoint SHA；
- source run ID、原训练 cache ID、head checkpoint SHA；
- binning ID、target、order/force mode；
- 输入 cache identity 与 feature compatibility signature；
- 所有输出文件 SHA、开始和完成时间。

外部 prediction manifest 独立完整，不更新原 run manifest，也不把外部数据 cache ID 写成训练 cache ID。

### 9.3 绘图

```text
Uncertainty_Quantification/Plots/ConfidenceHead/
├── matpes_test/
│   ├── <run-name>/...
│   └── comparisons/...
├── mad-test/
│   ├── <run-name>/...
│   └── comparisons/...
├── matpes_train/
│   ├── <run-name>/...
│   └── comparisons/...
└── comparisons/
    ├── cross_dataset_energy_correlations.csv
    ├── cross_dataset_energy_correlations.png
    ├── cross_dataset_energy_correlations.pdf
    ├── cross_dataset_force_correlations.csv
    ├── cross_dataset_force_correlations.png
    ├── cross_dataset_force_correlations.pdf
    └── manifest.json
```

所有发布先进入同级 staging，验证成功后原子替换。已存在且身份一致时复用；不一致时拒绝覆盖。

## 10. 绘图设计

### 10.1 单任务箱线图

每个数据集生成九套 PNG、PDF 和统计 CSV：

- energy order 1–8 各一套；
- force atom-mean 一套。

图形沿用 Carnet/现有 UPET 规范：

- 横轴为预测 argmax bin，固定显示全部 50 个 bin；
- tick 标签包含 bin 编号和 `n=<sample_count>`；
- 空 bin 保留 `n=0`，不绘制虚假箱体；
- 纵轴为 observed error，使用非负 symlog；
- energy 标注 `eV/atom`；
- force 明确标注 per-atom Cartesian-component mean，单位 `eV/Å`；
- PNG 为 300 DPI，同时输出矢量 PDF；
- 统计 CSV 包含 count、mean、median、std、quartiles、min/max 和 mean expected error。

### 10.2 数据集内部汇总

每个数据集生成：

1. energy order 1–8 的 `4 × 2` 合并箱线图；
2. 独立 force 箱线图；
3. energy Pearson/Spearman 随 order 1–8 变化图；
4. 对应 PNG、PDF 和 CSV。

同一合并箱线图共享纵轴范围，不使用置信区间。

### 10.3 跨数据集汇总

energy 汇总采用两个并列面板：

- 左侧 Pearson，右侧 Spearman；
- 横轴为 order 1–8；
- 每条线代表 `matpes_test`、`mad-test` 或 `matpes_train`；
- 纵轴统一限制为 `[-1, 1]`。

force 只有一个 head，因此使用 Pearson/Spearman 分组点图或柱状图比较三个数据集，不构造不存在的 order 维度。

统一 CSV 包含：

```text
dataset,target,order,sample_count,pearson,spearman
```

## 11. 远端调度与恢复

### 11.1 Preflight

登录节点只执行 CPU/文件系统检查：

- 九个 run 和 `best.pt` 完整性；
- MATPES cache split SHA 与数据计数；
- MAD 文件 SHA；
- Conda/Python/Torch/UPET/metatrain 版本；
- 输出目录和磁盘空间；
- 目标目录不存在不一致的旧结果。

### 11.2 Slurm 依赖图

```text
mad-cache (GPU)
    └── mad-predict-uq (GPU)

matpes-train-predict-uq (GPU, reuse cache)

mad-predict-uq + matpes-train-predict-uq
    └── plot-all-datasets (CPU)
```

`matpes_test` 不需要计算任务，由最终 CPU 绘图任务读取现有 evaluation。

每个 prediction 任务顺序运行九个 head，避免九个作业同时读取大型 memmap。每个 head 独立原子发布，所以失败后只重跑未完成或身份不一致的 head。

### 11.3 W&B

本次不初始化 W&B。推理进度、耗时、吞吐和错误写入本地结构化日志与 Slurm 日志。

## 12. 错误处理与安全边界

以下任一情况必须停止，不降级猜测：

- 数据 SHA 或结构/原子计数不一致；
- 同类 run 缺失、重复或 order 不完整；
- run/checkpoint/cache/binning identity 不一致；
- 新 cache 与 head 的 feature compatibility signature 不一致；
- energy/force feature 被交叉使用；
- tensor 形状、dtype、offset 或 finite 校验失败；
- 目标目录存在不同 identity 的结果；
- Pearson/Spearman 无法在至少两个非常量有限样本上计算；
- 绘图输入与 manifest SHA 不一致。

任何失败都不得修改原九个 run 的 checkpoint、evaluation、manifest、配置、日志或 W&B 文件。

## 13. 测试策略

### 13.1 本地测试

本地不运行真实大数据推理，只使用合成小数据和 mock 基础模型验证：

- strict config 与三种 dataset source；
- run discovery 恰好识别 8 energy + 1 force；
- 训练 checkpoint identity 与外部 cache compatibility 分离；
- MATPES train cache split 只读复用且禁止调用 feature extractor；
- MATPES test existing evaluation 只读适配且禁止调用预测；
- MAD 单数据集 cache 构建、复用、冲突拒绝与中断恢复；
- energy 逐原子误差只归一化一次；
- force atom-mean 不展开为 component；
- expected error 使用原训练 representatives；
- prediction artifact round trip、SHA 与 shape 校验；
- 九套单图、数据集汇总和跨数据集图的 PNG/PDF/CSV 生成；
- 空 bin、常量输入、NaN、重复 order 和不完整 artifact 明确失败；
- 原子发布与 no-clobber 行为。

执行 ConfidenceHead 定向 pytest，并根据仓库要求运行 lint；不修改用户未跟踪的 `.idea/`。

### 13.2 远端 smoke

先使用 `matpes_n20.extxyz`：

- 构建单数据集 cache；
- 分别运行一个 energy head 和 force head；
- 验证 prediction/UQ artifact；
- 生成一套小数据图；
- 证明重复执行复用已发布结果。

smoke 完成后才能提交正式 Slurm 任务。

### 13.3 正式验收

- 三个数据集均恰好对应八个 energy order 和一个 force head；
- MATPES test 没有产生新的基础推理；
- MATPES train 没有产生新的基础 feature cache；
- MAD cache 的数据 SHA、结构数、原子数与 prediction manifest 一致；
- 所有 logits 最后一维为 50，所有值有限；
- energy 样本数等于结构数，force 样本数等于原子数；
- energy 语义为逐原子，force 语义为 atom-mean；
- 每个数据集九份 prediction/UQ artifact 完整；
- 每个单任务统计 CSV 恰好包含 50 个 bin；
- correlation CSV 的 energy order 恰好为 1–8；
- 从 prediction artifact 独立重算的指标与 CSV 在明确容差内一致；
- PNG、PDF 可打开且非空，图中样本数总和与 artifact 一致；
- 原九个 run 的 `best.pt`、原 evaluation 和 manifest SHA 均未改变。

## 14. 实施提交与远端发布

实现按职责拆成独立提交，至少覆盖：

1. 外部预测配置、单数据集 cache 与兼容性校验；
2. dataset-scoped ConfidenceHead prediction/UQ 与 verifier；
3. 三数据集绘图与跨数据集汇总；
4. CLI、正式/smoke 配置、Slurm 模板与中文 README；
5. 测试和必要修复。

本地提交并推送后，远端 `ConfidenceHead` 只 cherry-pick 本次提交。每次部署记录并核对 commit SHA；tracked 代码必须处于预期提交，远端 dirty 用户文件保持不动。

## 15. 成功标准

本次工作只有同时满足以下条件才算完成：

- 不训练、不覆盖原结果、不伪造 identity；
- 最大化复用现有 MATPES evaluation/cache 和 ConfidenceHead 公共模块；
- 只对 MAD 执行一次必要的基础 UPET 特征提取；
- 两个新数据集完成九个 head 的 prediction 与 UQ；
- 三个数据集完成约定的单图、数据集汇总和跨数据集图；
- 数据、checkpoint、cache、prediction 和 plots 均具有可核验 manifest；
- 本地测试、远端 smoke、正式远端验收全部通过；
- 原九个训练 run 的正式产物保持逐字节不变。
