# UPET ConfidenceHead 重新计算设计

日期：2026-07-29
状态：设计已获用户逐节确认，等待书面规格复核
目标目录：`Uncertainty_Quantification/ConfidenceHead/`
目标分支：`ConfidenceHead`

## 1. 背景

旧实现位于：

```text
/home/lilong/code/UQ/upet/UQ_orb_post_train_force
```

旧结果经审计后被判定只与旧代码自身的定义一致，并不严格等价于
`carnet_new/Uncertainty_Quantification/ConfidenceHead` 的方法。主要问题包括：

- force confidence 使用每原子 xyz 平均误差，而不是逐笛卡尔分量误差；
- energy 溢出样本使用 `ignore_index` 排除；
- energy 与 force 的训练、分箱、缓存和模型选择语义未完全对齐参考实现；
- 旧结果的 checkpoint 选择和部分测试数据来源不统一。

因此，本项目不迁移旧代码、旧配置、旧日志、旧 checkpoint 或旧数值结果。新代码将使用
当前 `upet_new`、当前数据和 checkpoint 重新执行完整 ConfidenceHead 计算。

参考实现为：

```text
/home/lilong/code/carnet_new/Uncertainty_Quantification/ConfidenceHead
```

新实现采用其科学方法、模块边界和产物身份设计，但必须完全自包含，不在运行时导入
`carnet_new` 或旧 `UQ_orb_post_train_force`。

## 2. 目标

在 `Uncertainty_Quantification/ConfidenceHead/` 中建立一套可独立运行、可验证和可精确
续训的 UPET ConfidenceHead 工作流：

1. 使用冻结的 UPET backbone 构建不可变 raw cache；
2. force confidence 使用逐原子、逐笛卡尔分量误差；
3. energy confidence 使用逐结构的每原子能量误差；
4. energy 和 force 严格读取各自独立 readout 的末层特征；
5. 使用固定线性分箱；
6. 训练 component-wise force head 与 structure-level energy head；
7. 使用加权 validation total loss 的跨 epoch EMA 控制 scheduler 和 early stopping；
8. 支持可验证的精确断点续训；
9. 保存完整数值预测、指标、checkpoint 和身份 manifest；
10. 在远程登录节点上完成真实 n20 CPU 全链路测试。

## 3. 已确认的设计决策

### 3.1 方法基线

- 采用 `carnet_new` 主体科学逻辑；
- 只把训练控制修改为 `/1` 实验风格的 EMA、plateau scheduler、best/last 和
  early-stopping 状态机；
- early-stopping 的监控对象由旧版 energy loss 改为用户确认的加权 validation total
  loss；
- 不复制旧实现中的任何源文件、配置、日志或结果。

### 3.2 标签

- force：逐笛卡尔分量绝对误差，shape 为 `[N_atom, 3]`；
- energy：结构总能量绝对误差除以结构原子数，shape 为 `[N_structure]`。

### 3.3 分箱

- 只支持固定线性分箱；
- 默认 50 bins；
- force 默认最大误差为 `0.5`；
- energy 默认最大每原子误差为 `0.3`；
- 溢出误差进入最后一个 bin，不使用 `ignore_index` 排除；
- 精确等于内部阈值的误差进入较高 bin。

### 3.4 模型与 loss

- force 和 energy head 默认各有三层 256 维隐藏层；
- activation 为 Shifted Softplus；
- dropout 默认为 0；
- 不使用旧版 LazyLinear、LayerNorm 或额外可训练 projection；
- force loss coefficient 为 `1.0`；
- energy loss coefficient 为 `1.5`；
- 不使用 label smoothing；
- 不使用 class weights。

### 3.5 执行边界

- 本地只进行静态验证；
- 不在本地加载真实 checkpoint、构建 cache、训练、评估或运行 pytest；
- 远程登录节点执行单元测试和真实 n20 CPU 全链路；
- 当前阶段不进行 GPU 或完整数据训练；
- 当前阶段不实现绘图，也不生成或迁移图片。

## 4. 非目标

本项目当前不：

- 迁移旧 ConfidenceHead 结果；
- 迁移旧代码、配置、日志、checkpoint 或图片；
- 兼容旧的每原子 xyz 平均 force confidence 语义；
- 支持 quantile-log 分箱；
- 在本地运行真实模型或动态测试；
- 在远程登录节点运行完整 train/validation/test；
- 提交 data、cache、checkpoint 或 outputs 到 Git；
- 将本功能首版并入 `src/upet` 公共 API；
- 在输出缺失或身份不匹配时进行模糊路径搜索或静默回退；
- 在国内镜像缺少指定 PyTorch build 时静默安装其他版本。

## 5. 总体架构

```text
Uncertainty_Quantification/ConfidenceHead/
├── README.md
├── __init__.py
├── configs/
│   ├── n20_cpu.yaml
│   └── full_linear.yaml
├── confidence_head/
│   ├── __init__.py
│   ├── config.py
│   ├── identity.py
│   ├── data.py
│   ├── checkpoint.py
│   ├── features.py
│   ├── cache.py
│   ├── errors.py
│   ├── binning.py
│   ├── adapters.py
│   ├── heads.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── trainer.py
│   ├── artifacts.py
│   └── workflows/
│       ├── __init__.py
│       ├── build_cache.py
│       ├── train.py
│       ├── evaluate.py
│       └── verify.py
├── scripts/
│   ├── build_cache.py
│   ├── train.py
│   ├── evaluate.py
│   └── verify.py
├── tests/
└── outputs/
```

### 5.1 科学计算核心

`errors.py`、`binning.py`、`adapters.py`、`heads.py`、`losses.py` 和
`metrics.py` 只处理 tensor。它们不读取配置文件、不加载 checkpoint、不访问文件系统，
也不包含 CLI 编排。

### 5.2 UPET 与产物集成层

`checkpoint.py`、`data.py`、`features.py`、`cache.py`、`identity.py` 和
`artifacts.py` 负责：

- 当前 UPET API；
- extxyz 数据；
- 两个独立 readout；
- raw cache；
- hash 与 identity；
- manifest；
- 原子性写入和完整性验证。

### 5.3 工作流层

`workflows/` 只编排阶段，不重复实现数学公式。`scripts/` 是薄入口，只负责参数解析、
日志初始化、调用 workflow 和设置退出码。

## 6. UPET 独立 readout 不变量

UPET 的 energy 与 non-conservative force 是分别读出的：

```text
共享表示网络
├── energy readout
│   ├── total energy prediction
│   └── energy last-layer atom features
└── non-conservative-force readout
    ├── force-component prediction
    └── force last-layer atom features
```

新实现必须把以下规则作为架构不变量，而不是可选配置：

1. force confidence head 只能读取 non-conservative-force readout 的逐原子末层特征；
2. energy confidence head 只能读取 energy readout 的逐原子末层特征；
3. energy 特征必须先按结构聚合为 cumulants；
4. force 特征不得进行 structure-level 聚合；
5. 两类特征分别缓存、分别记录 output key、shape、dtype 和 identity；
6. 即使两个 feature tensor 的维度相同，也禁止共享 tensor、adapter 或 head 参数；
7. 任一专用 readout 或 feature 缺失时立即失败；
8. 禁止拿另一 readout 的 feature 作为回退。

若当前 UPET API 支持在一次 forward 中同时请求四个输出，可以共享同一次 backbone
forward：

- total energy；
- non-conservative force；
- energy last-layer atom features；
- non-conservative-force last-layer atom features。

这只是计算优化，不能改变两个 readout 的独立语义。若 API 不支持联合请求，可以执行两个
明确的 readout 请求，但输出必须通过相同结构顺序和 identity 校验。

## 7. 数据与 raw cache

### 7.1 数据顺序

- train、validation 和 test 按 extxyz 中的稳定顺序读取；
- cache 构建阶段禁止 shuffle；
- 保存原始 structure index、原子数和 atom offsets；
- 每个 split 独立计算文件 SHA256、结构数和原子数；
- production profile 禁止 split 内容相同；
- smoke profile 只有显式设置 `allow_identical_splits: true` 才允许相同输入。

### 7.2 Cache 字段

每个 split 分 shard 保存：

```text
structure_ids
num_atoms
atom_offsets
atomic_numbers
force_prediction[N_atom, 3]
force_reference[N_atom, 3]
energy_prediction[N_structure]
energy_reference[N_structure]
force_features[N_atom, D_force]
energy_features[N_atom, D_energy]
```

默认浮点 dtype 为 float32。structure IDs、原子数、atomic numbers 和 offsets 使用整数
类型。

cache 不保存：

- force/energy labels；
- bin indices；
- logits；
- expected uncertainty；
- 与某个 cumulant order 或 head 结构绑定的中间结果。

因此，修改固定线性分箱最大值、cumulant order 或 head hidden dimensions 时，不需要
重新运行 UPET。

### 7.3 Cache identity

cache manifest 至少包含：

- cache schema version；
- checkpoint SHA256；
- train/validation/test 文件 SHA256；
- UPET、metatrain、metatomic、PyTorch 和 Python 版本；
- resolved prediction 与 feature output keys；
- energy/force feature dimensions 与 dtype；
- split structure/atom counts；
- shard 列表、每个 shard hash 和状态；
- git commit；
- device 和 AMP 设置；
- cache formula/version ID。

按完整结构写入 shard。临时文件完成校验后原子替换；中断不能留下可被误认为 complete
的 shard。

## 8. 误差定义

### 8.1 Force

对原子 `i` 和笛卡尔分量 `c`：

```text
e_F[i,c] = abs(F_pred[i,c] - F_ref[i,c])
```

输出 shape：

```text
force_error:  [N_atom, 3]
force_logits: [N_atom, 3, B_force]
```

三个分量各使用一个独立 ConfidenceHead。每个分量 head 接收相同原子的 force-readout
feature，但三个 head 的参数不共享。

### 8.2 Energy

对结构 `s`：

```text
e_E[s] = abs(E_pred_total[s] - E_ref_total[s]) / N_atoms[s]
```

输出 shape：

```text
energy_error:  [N_structure]
energy_logits: [N_structure, B_energy]
```

禁止把总能量误差直接作为标签，也禁止对 batch 原子总数归一化。

## 9. 固定线性分箱

默认参数：

```yaml
binning:
  algorithm: fixed_linear_v1
  force_num_bins: 50
  force_max_error: 0.5
  energy_num_bins: 50
  energy_max_error: 0.3
  overflow: saturate_last
  threshold_side: upper
```

默认 bin 宽度：

```text
force:  0.5 / 50 = 0.01
energy: 0.3 / 50 = 0.006
```

内部阈值数量为 `num_bins - 1`。精确等于阈值的误差进入较高 bin。所有高于最大误差的
样本进入最后一个 bin，不产生 `ignore_index`。

bin 代表值为 bin 中心。预测连续不确定性：

```text
p_b = softmax(logits)_b
expected_error = sum_b p_b * bin_center_b
```

evaluation 必须单独保存每个分支的 overflow count 和 overflow fraction，不能因为溢出
被分到最后一个 bin 而隐藏其数量。

## 10. Feature adapter 与 ConfidenceHead

### 10.1 Force

- 输入为 force readout 的逐原子末层 feature；
- 默认不做 projection；
- 三个独立 component heads；
- 输出 `[N_atom, 3, 50]`。

### 10.2 Energy

energy readout 的逐原子末层 feature 按结构聚合为 cumulants。`cumulant_order`：

- 配置范围为 1–8；
- 默认值为 3；
- 不限制后续正式实验使用的阶数；
- 高于一阶的 cumulant 默认使用 signed root；
- 输出维度为 `D_energy × cumulant_order`。

不添加旧版的 LazyLinear、LayerNorm、额外 projection 或 dropout adapter。

### 10.3 Head 默认结构

```yaml
model:
  force:
    hidden_dims: [256, 256, 256]
    dropout: 0.0
    num_bins: 50
  energy:
    hidden_dims: [256, 256, 256]
    dropout: 0.0
    num_bins: 50
    cumulant_order: 3
    signed_root: true
```

每个隐藏层使用：

```text
Linear -> ShiftedSoftplus -> optional Dropout
```

输出层为 Linear logits，不在模型内部执行 softmax。

## 11. Loss 与 optimizer

branch loss 分别按自己的有效样本数求均值：

```text
force_loss  = mean CE over all [atom, component]
energy_loss = mean CE over all structures
total_loss  = 1.0 * force_loss + 1.5 * energy_loss
```

不能把 `3 × N_atom` force 样本与 `N_structure` energy 样本直接拼接后统一求均值，否则
force 会因为样本数更多而隐式改变分支权重。

默认 optimizer：

```yaml
optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0
```

默认：

- label smoothing 为 0；
- class weights 为 null；
- seed 为 1234。

## 12. Validation 汇总

epoch 指标必须按真实样本数汇总，不能对 batch mean 等权平均。

对每个 branch 累加：

```text
cross_entropy_sum
sample_count
```

然后计算：

```text
force_epoch_loss  = force_cross_entropy_sum / force_sample_count
energy_epoch_loss = energy_cross_entropy_sum / energy_sample_count
total_epoch_loss  = 1.0 * force_epoch_loss + 1.5 * energy_epoch_loss
```

early-stopping 与 scheduler 使用 validation 的 `total_epoch_loss`，而不是 train loss、
energy-only loss 或 batch EMA。

## 13. Early-stopping 状态机

### 13.1 EMA

设当前 validation total loss 为 `v_t`：

```text
EMA_0 = v_0
EMA_t = 0.95 * EMA_(t-1) + 0.05 * v_t
```

每个完整 validation epoch 只更新一次 EMA。禁止对 batch loss 先做 EMA 后再次平滑。

### 13.2 改善条件

```text
current_ema < best_ema - 1e-4
```

改善时：

- `bad_epochs = 0`；
- 更新 best value、epoch 和 global step；
- 原子性保存 `best.pt`。

无改善时：

- `bad_epochs += 1`。

至少完成 3 epochs 后，连续 15 个 epoch 无改善时停止。最大训练 epoch 数为 200。

### 13.3 ReduceLROnPlateau

沿用 `/1` 的参数，但监控对象改为用户确认的 `val/total_loss_ema`：

```yaml
scheduler:
  name: plateau
  monitor: val/total_loss_ema
  mode: min
  factor: 0.5
  patience: 5
  threshold: 0.0001
  threshold_mode: abs
  cooldown: 0
  min_lr: 1.0e-6
  interval: epoch
  frequency: 1
```

scheduler 和 early stopping 必须读取同一个 epoch EMA 值。每次 scheduler 更新后的状态
必须写入 `last.pt`，从而保证 resume 后学习率历史连续。

## 14. Best、last 与精确断点续训

### 14.1 Checkpoint 语义

- `best.pt`：截至当前最优 `val/total_loss_ema` 对应的完整训练状态；
- `last.pt`：最近一个完整提交 epoch 的真实状态；
- evaluation 默认读取 `best.pt`；
- 训练结束时可以在内存中恢复 best model，但不得用 best 覆盖 `last.pt`；
- `last.pt` 必须保留真实 stop reason 和最终 bad-epoch 状态。

### 14.2 保存内容

`last.pt` 和 `best.pt` 保存：

- force/energy head 与 adapter state；
- optimizer state；
- scheduler state；
- epoch 和 global step；
- 当前学习率；
- EMA；
- best value、best epoch、best step；
- bad epochs；
- early-stopping stop reason；
- Python RNG；
- NumPy RNG；
- Torch CPU RNG；
- Torch CUDA RNG（若运行设备存在 CUDA）；
- DataLoader/sampler generator state；
- resolved config ID；
- cache ID；
- binning ID；
- model/loss ID；
- checkpoint schema version。

### 14.3 Resume

resume 前必须核对所有上游 identity。任何不匹配都直接失败。恢复后从下一个未完成 epoch
开始，不能重复已经提交的 epoch，也不能重置 scheduler、EMA、best value 或 sampler。

checkpoint 使用：

```text
write temporary -> flush/close -> validate -> atomic replace
```

## 15. 工作流

### 15.1 `build_cache.py`

- 加载并冻结 UPET；
- 验证 checkpoint SHA 和 capabilities；
- 解析两个独立 readout；
- 按稳定顺序读取三个 split；
- 生成 raw cache shards；
- 验证 shape、count、offset、有限性和 hash；
- 完成后提交 cache manifest。

### 15.2 `train.py`

- 只读取 complete raw cache；
- 构建固定线性 bins；
- 根据配置构建两个独立 adapter/head；
- 训练、验证、scheduler、early stopping；
- 保存 best/last 与 metrics JSONL；
- 支持精确 resume。

### 15.3 `evaluate.py`

- 默认加载 `best.pt`；
- 只读取 complete test cache；
- 计算完整 logits、labels、continuous errors 和 expected errors；
- 计算 force-component 和 energy-structure 指标；
- 保存预测、聚合 CSV 和 evaluation manifest；
- 不生成图。

### 15.4 `verify.py`

提供 metadata 与 full 两级验证：

- metadata：schema、identity、状态、shape 和 count；
- full：额外重新计算所有文件 hash，并检查完整预测字段。

正式消费方拒绝读取 `status != complete` 的产物。

## 16. 配置

两个初始配置使用相同 schema。

### 16.1 `n20_cpu.yaml`

```yaml
profile: smoke
allow_identical_splits: true
device: cpu
amp: false
cache:
  batch_size: 2
  num_workers: 0
trainer:
  max_epochs: 3
```

同一 n20 文件用于 train、validation 和 test，只验证流程，不产生正式科学结论。

### 16.2 `full_linear.yaml`

```yaml
profile: production
allow_identical_splits: false
binning:
  force_num_bins: 50
  force_max_error: 0.5
  energy_num_bins: 50
  energy_max_error: 0.3
trainer:
  max_epochs: 200
```

`cumulant_order`、batch size、device 和 AMP 保持显式可配置。当前阶段只提供并静态验证
full 配置，不执行它。

## 17. 结果协议

```text
outputs/
├── caches/
│   └── <cache_id>/
│       ├── manifest.json
│       ├── train/
│       ├── validation/
│       └── test/
└── runs/
    └── <run_name>/
        ├── resolved_config.yaml
        ├── manifest.json
        ├── binning.json
        ├── checkpoints/
        │   ├── best.pt
        │   └── last.pt
        ├── logs/
        │   └── metrics.jsonl
        └── evaluation/
            ├── test_predictions.pt
            ├── metrics.json
            ├── force_bin_summary.csv
            └── energy_bin_summary.csv
```

`test_predictions.pt` 至少包含：

- force/energy logits；
- force/energy labels；
- force/energy continuous observed errors；
- force/energy expected errors；
- structure IDs；
- atom offsets；
- bin representatives。

`metrics.json` 分别报告：

- sample count；
- overflow count 和 fraction；
- classification accuracy；
- NLL；
- Brier score；
- mean observed error；
- mean expected error；
- `MAE(expected_error, observed_error)`；
- Pearson；
- Spearman。

`outputs/` 写入 `.gitignore`，不提交 Git。

## 18. Manifest 与运行身份

run manifest 至少记录：

- run schema version；
- cache/binning/model/loss/config IDs；
- checkpoint/data SHA256；
- energy/force 独立 readout keys；
- git commit；
- Python 和所有关键依赖版本；
- seed、device 和 AMP；
- best/last checkpoint SHA256；
- best epoch、best metric 和 stop reason；
- 各阶段开始/完成时间；
- `smoke` 或 `production` profile；
- 状态 `incomplete` 或 `complete`。

run name 只用于人类阅读，不能替代 identity。

## 19. 错误处理

以下情况必须立即失败：

- checkpoint SHA 不符；
- 数据 SHA 不符；
- energy 或 force prediction readout 缺失；
- energy 或 force last-layer feature readout 缺失；
- 两个 ConfidenceHead 接错 readout；
- 任一 prediction、reference、feature 或 loss 出现 NaN/Inf；
- structure count、atom count、offset 或 feature shape 不一致；
- production 配置的 split 内容相同；
- resume identity 不匹配；
- 读取 incomplete artifact；
- cache shard 重复或缺失 structure ID；
- 国内镜像无法提供要求的精确 PyTorch build；
- 运行时 `torch.__version__` 不是 `2.11.0+cu128`。

数据错误必须报告 split、shard 和 structure ID；不能只输出通用的 shape mismatch。

## 20. 测试策略

### 20.1 单元测试

在远程 `upet_new` 环境运行，覆盖：

- force component error；
- per-atom energy error；
- 固定线性边界与 overflow；
- 三个独立 force heads；
- energy/force readout 特征隔离；
- cumulant order 和 signed root；
- branch-wise mean 与 loss coefficients；
- epoch sample-weighted aggregation；
- EMA 初值和递推；
- absolute min-delta；
- early-stopping patience/min-epochs；
- scheduler 降学习率 epoch；
- best/last 语义；
- exact resume；
- cache identity；
- artifact atomicity；
- production split guard；
- config path 和 stable hash。

readout 隔离测试必须注入刻意不同的 energy/force feature 值。错误连接任一 feature 时测试必须
失败。

### 20.2 本地验证

本地只运行静态检查：

- Ruff format check；
- Ruff lint；
- mypy；
- Sphinx lint（若 README 纳入文档检查）；
- YAML/JSON 静态格式检查；
- `git diff --check`；
- 文件清单与导入边界审查。

本地不：

- 加载真实 checkpoint；
- 读取真实 n20/full 数据执行模型；
- 构建 cache；
- 训练或评估；
- 运行动态 pytest。

### 20.3 远程 n20 全链路

远程连接：

```text
ssh -p 55801 bywang@121.48.164.204
```

代码目录：

```text
/home/bywang/code/UQ/upet_new
```

远程依次执行：

```text
创建/验证 conda 环境
-> 安装项目
-> 运行全部 ConfidenceHead 单元测试
-> build-cache
-> train
-> evaluate(best.pt)
-> verify(full)
```

另执行精确续训对照：

```text
连续训练到 epoch 3
vs.
epoch 1 后中断，再 resume 到 epoch 3
```

比较：

- model state；
- optimizer state；
- scheduler state；
- EMA；
- best/last metadata；
- metric sequence；
- evaluation 数值。

在确定性 CPU 设置下要求逐字段相同；若底层库存在已证明的非确定性，只允许使用设计中
明确记录的数值容差，不能临时放宽。

## 21. 远程环境与数据

### 21.1 Conda

创建独立环境：

```text
name: upet_new
python: 3.11
torch runtime version: 2.11.0+cu128
```

不得复用远程已有 `upet` 环境，因为该环境绑定旧仓库 editable source。

conda 和 PyPI 依赖优先使用国内镜像。安装前先检查国内镜像是否提供精确
`2.11.0+cu128` build。如果没有，停止并报告；未经用户确认不能改用其他 torch build。

登录节点即使安装 CUDA build，本次也固定：

```text
device=cpu
amp=false
```

### 21.2 Git

本地代码提交到 `ConfidenceHead` 分支并推送到：

```text
https://github.com/BoyuWang2024/upet_new.git
```

远程通过 Git clone/pull 获取代码，不通过复制工作树部署。

### 21.3 数据

本地 n20：

```text
/home/lilong/code/UQ/upet_new/data/dataset/matpes_n20.extxyz
```

包含：

- 20 structures；
- 143 atoms；
- 每个结构 2–19 atoms。

已审计 SHA256：

```text
c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d
```

远程已有 checkpoint：

```text
/home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt
```

已审计 SHA256：

```text
879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
```

实施阶段将 checkpoint 复制到新仓库 `data/checkpoint/`，将 n20 传输到
`data/dataset/`，随后重新计算 SHA256。当前阶段不复制或运行完整数据。

## 22. 验收标准

设计对应的实现只有同时满足以下条件才可进入完整数据计算：

1. 本地静态检查全部通过；
2. 远程运行时精确报告 `torch 2.11.0+cu128`；
3. 远程全部 ConfidenceHead 单元测试通过；
4. n20 cache 包含 20 structures、143 atoms；
5. force prediction/error/logits 分量维度为 3；
6. energy 使用 total-energy absolute error / structure atom count；
7. energy 与 force feature identity 明确不同并通过隔离测试；
8. 固定线性分箱边界和 overflow 测试通过；
9. loss 为 branch-wise mean 后按 `1.0/1.5` 加权；
10. scheduler 与 early stopping 同时监控 `val/total_loss_ema`；
11. 连续训练与中断续训结果一致；
12. evaluation 默认加载 best，而 last 保留真实终止状态；
13. full artifact verification 通过；
14. 输出中不存在图片；
15. smoke manifest 明确标记不得用于科学结论；
16. 没有运行 GPU 或完整数据任务。

## 23. 后续边界

n20 全链路通过后，是否进行 GPU full run 是一个独立决策。届时需要另外确认：

- GPU 执行节点与调度方式；
- full cache batch size、shard size 和存储位置；
- cumulant order 实验集合；
- AMP/float32 策略；
- W&B 模式；
- 正式运行预算和监控方式。

本设计不预先授权任何 GPU 或完整数据计算。
