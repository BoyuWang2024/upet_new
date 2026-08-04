# UPET ConfidenceHead：仿 metatrain 的 Memmap 训练重构设计

## 1. 背景与目标

当前 ConfidenceHead 将冻结 UPET 产生的预测、参考值以及力/能量最后一层特征保存为多个大型 `.pt` shard。训练集使用全局随机采样，而 `CachedSplitDataset` 默认只保留一个 shard。每次随机访问切换 shard 时，数据集会重新计算整个 shard 的 SHA256、加载 mmap、执行全张量有限值检查并驱逐上一 shard。正式缓存包含约 28 个 shard、约 22 GB 数据，这一组合造成严重读放大、CPU 与并行文件系统拥塞，并使 GPU 长时间等待数据。

本设计废弃旧 `.pt` shard 缓存，参考 metatrain `MemmapDataset` 的核心方式，将每个字段保存为一个连续 memory-mapped 数组。训练使用全局随机采样、可配置 DataLoader worker 和可选按原子数装批。力与能量仍使用不同 UPET readout，模型只构造和计算 loss 系数大于零的分支。

本次重构必须保持以下既定语义：

- 力默认采用 `atom_mean` 监督，同时保留逐分量模式；
- 能量误差始终是逐原子能量误差；
- 固定线性分箱定义保持不变；
- 优化目标是力/能量监督 loss 的加权总和；
- scheduler 与 early-stopping 继续监控验证总 loss 的 EMA；
- 本地只做静态与 CPU 小数据验证，远端进行 GPU 全链路和性能验证；
- 不迁移旧缓存、旧训练结果、旧日志或旧图。

## 2. 方案选择

采用自有 `ConfidenceMemmapDataset`，复刻 metatrain 的连续数组、按索引切片和 worker 内惰性打开方式，但不直接复用 metatrain 的 `MemmapDataset` 类。后者面向 `System` 和 `TensorMap` target，不能自然表达两套独立的 1024 维原子 readout、UPET 预测及参考值；强行包装会增加转换开销并耦合 metatrain 内部版本。

未采用的方案：

- 保留 `.pt` shard 并改成 shard-aware sampler：能快速缓解问题，但不能直接达到用户要求的 metatrain 风格，也保留复杂的 LRU 与 shard 校验生命周期。
- 将所有字段放入单个大型 Torch mmap 文件：布局不透明，多 worker 重开与字段级验证不如逐数组 memmap 清晰。

## 3. 缓存 schema v2

### 3.1 文件布局

新缓存使用 `CACHE_SCHEMA_VERSION = 2`：

```text
cache-<identity>/
├── manifest.json
├── train/
│   ├── structure_offsets.npy
│   ├── structure_ids.bin
│   ├── force_features.bin
│   ├── energy_features.bin
│   ├── force_prediction.bin
│   ├── force_reference.bin
│   ├── energy_prediction.bin
│   └── energy_reference.bin
├── validation/
│   └── ...
└── test/
    └── ...
```

对一个包含 `S` 个结构、`A` 个原子的 split，各字段定义如下：

| 字段 | dtype | shape | 语义 |
|---|---:|---:|---|
| `structure_offsets.npy` | int64 | `[S + 1]` | 各结构在原子字段中的半开区间 |
| `structure_ids.bin` | int64 | `[S]` | 原始结构标识 |
| `force_features.bin` | float | `[A, F_force]` | 力 readout 的原子特征 |
| `energy_features.bin` | float | `[A, F_energy]` | 能量 readout 的原子特征 |
| `force_prediction.bin` | float | `[A, 3]` | UPET 力预测 |
| `force_reference.bin` | float | `[A, 3]` | 参考力 |
| `energy_prediction.bin` | float | `[S]` | UPET 结构能量预测 |
| `energy_reference.bin` | float | `[S]` | 参考结构能量 |

力和能量特征必须来自配置声明的不同 UPET 输出，并保存在不同文件中。构建时拒绝两者共享存储或维度/样本布局不一致。

### 3.2 Manifest 与 identity

Manifest 保存：

- schema 版本和 `status`；
- checkpoint 路径身份与 SHA256；
- train/validation/test 输入文件 SHA256；
- 力/能量 prediction 与 feature readout 名称；
- 执行 dtype；
- 每个 split 的结构数与原子数；
- 每个数组的相对路径、dtype、shape、字节数和 SHA256；
- 力与能量特征维度；
- 完整 cache identity。

cache identity 由会改变缓存内容或解释方式的字段确定。训练超参数、ConfidenceHead 隐层和 loss coefficient 不进入 cache identity，因此九组实验可安全共享同一缓存。

### 3.3 流式构建与原子发布

缓存按 split 进行两阶段构建：

1. 第一遍扫描 extxyz，计算结构数、总原子数、结构 offsets 和输入 SHA256。
2. 第二遍流式读取结构，按 `cache.batch_size` 调用冻结 UPET。首个成功 batch 确认两套特征维度后预分配字段数组，之后将每个 batch 直接写入目标区间。

写入期间检查结构顺序、原子数、dtype、shape 和有限值。第二遍结束后重新计算输入 SHA256；若输入在构建期间变化，构建失败。

所有文件先写入同一输出根目录下的唯一临时目录。完成数组 flush、文件长度检查与发布前完整 SHA 后，写入 `status: complete` 的 manifest，并通过原子 rename 发布。缓存 identity 使用独占锁，禁止并发发布同一缓存。失败目录不能被训练读取。

### 3.4 加载与验证

`ConfidenceMemmapDataset` 在主进程只读取 manifest、offsets 和数组描述。每个 DataLoader worker 首次访问时惰性打开自己的 memmap 句柄，后续随机索引只根据 offsets 返回对应视图。worker 序列化后不得携带已打开的底层 mmap 句柄，必须在目标 worker 中重新打开。

普通训练启动执行快速验证：

- manifest schema、status 和 identity；
- 所有路径均限制在缓存根目录；
- 文件存在且字节数与 dtype/shape 一致；
- offsets 从零开始、严格递增并以总原子数结束；
- 各字段第一维与结构数/原子数一致。

普通训练和 `__getitem__` 不重新计算完整 SHA，也不扫描整个数组的有限值。完整 SHA 仅在缓存发布前和显式 `verify --full` 时执行。

schema v1 缓存一律拒绝，并提示重新构建；不提供转换器、自动兼容或回退路径。

## 4. DataLoader 与 batch

### 4.1 固定结构数模式

默认训练加载器采用 metatrain 的全局随机语义，但使用显式 epoch sampler 代替 DataLoader 内部不可寻址的随机状态：

```python
DataLoader(
    dataset,
    batch_size=trainer.batch_size,
    sampler=epoch_random_sampler,
    shuffle=False,
    drop_last=True,
    num_workers=resolved_num_workers,
)
```

验证和测试使用 `shuffle=False`、`drop_last=False`。连续 memmap 保持打开，全局随机访问只读取目标结构的页面，不存在 shard 驱逐或每样本全文件校验。

### 4.2 确定性与恢复

训练 sampler 的 epoch 顺序由 `run.seed` 和 epoch 唯一确定，并提供 `set_epoch(epoch)`。worker 的 Python、NumPy 与 Torch RNG 在 worker 启动时由 run seed 和 worker id 确定；Dataset 与 Collate 不执行随机变换，因此 worker RNG 不参与样本顺序。样本顺序不依赖 worker 调度时机。

Checkpoint 保存 `next_epoch` 和 sampler 基础 seed。从 epoch 边界恢复时重新生成该 epoch 的采样顺序，不依赖 DataLoader generator 的隐式状态。

### 4.3 可选按原子数装批

新增配置：

```yaml
trainer:
  batch_size: 128
  num_workers: null
  max_atoms_per_batch: null
  min_atoms_per_batch: 0
  pin_memory: true
  persistent_workers: true
```

旧 `cache.shard_max_atoms` 被删除，因为 v2 不再分 shard。旧 `cache.num_workers` 也被删除，训练加载 worker 统一由 `trainer.num_workers` 控制；`cache.batch_size` 继续表示冻结 UPET 构建缓存时的推理 batch。正式配置、CPU 测试配置和配置 schema 同步更新，残留旧字段时由严格配置解析直接报错。

`max_atoms_per_batch: null` 是正式默认值，继续使用固定结构数，避免首次性能修复同时改变 optimizer step 数。

设置 `max_atoms_per_batch` 后，batch sampler 先全局打乱结构索引，再通过 offsets 获得原子数并贪心装批，直到加入下一结构会超过上限。单个结构超过上限时明确报错；不静默丢弃该结构。`min_atoms_per_batch` 只控制 epoch 末尾不足下限的残余 batch，训练可丢弃，验证与测试不得丢弃。该模式下 `batch_size` 不参与训练/验证 batch 构造。

### 4.4 worker 与设备传输

`num_workers: null` 时自动选择 worker 数，但不超过当前 CPU affinity 和 `SLURM_CPUS_PER_TASK`。显式整数严格按配置执行。`persistent_workers` 仅在 worker 数大于零时生效。

Collate 一次性分配连续 batch tensor，并复制各结构的 memmap 视图。CUDA 下按配置使用 pinned memory，设备传输使用 `non_blocking=True`；CPU 模式自动关闭 pinned memory。batch-local `atom_counts` 在 CPU 上完成结构边界验证后随 batch 传输。

## 5. 按需模型分支与监督定义

### 5.1 Active target

运行开始时定义：

```python
force_active = model.force.enabled and loss.force_coefficient > 0
energy_active = model.energy.enabled and loss.energy_coefficient > 0
```

至少一个 target 必须 active。`enabled: false` 且 coefficient 大于零是配置错误；`enabled: true` 且 coefficient 等于零合法，但该分支不构造、不前向、不生成标签，也不进入 optimizer。

缓存始终保存两套 readout，因此不同 active target 的实验共享缓存。Run identity 必须包含 active target、模型、分箱与 loss 配置，避免单分支 checkpoint 互相混用。

### 5.2 力分支

力 ConfidenceHead 只读取 `force_features`。保留：

- `atom_mean`：默认，每个原子的三个分量误差先聚合，再生成一个标签；
- `component`：每个原子三个分量分别生成标签。

正式默认固定线性分箱最大力误差为 `0.5`。

### 5.3 能量分支

能量 ConfidenceHead 只读取 `energy_features`。监督误差固定为：

\[
\epsilon_E = \frac{|E_{\mathrm{pred}} - E_{\mathrm{ref}}|}{N_{\mathrm{atoms}}}.
\]

正式默认固定线性分箱最大逐原子能量误差为 `0.3`。

### 5.4 向量化 cumulant

Collate 提供每个结构的 `atom_counts`。适配器在设备上通过 `repeat_interleave` 构造每个原子的结构索引。对阶数 `1..K`，分别计算 raw moment，用 `index_add` 按结构求和并除以原子数，再使用现有递推关系计算 cumulant。只保留最多八次的阶数循环，不循环结构，也不在 CUDA tensor 上调用 `.item()`。

实现不得一次性构造 `[A, F_energy, K]` 中间 tensor；每次只保留当前 raw moment 和结构级聚合结果，以约束显存。

### 5.5 Loss、优化器与梯度裁剪

模型输出的两个 logits 均为可选值。只为 active 分支计算观测误差、分箱标签和交叉熵。监督总 loss 为：

\[
L_{\mathrm{total}} = w_f L_f + w_e L_e,
\]

其中不 active 的项不存在，而不是伪造为零样本 loss。epoch 聚合先按各分支真实样本数求均值，再使用 coefficient 组合总 loss。

保留 AdamW 和现有学习率配置。新增 `trainer.grad_clip_norm: 1.0`，在 backward 后、optimizer step 前裁剪；设为 `null` 时关闭。

## 6. Scheduler、early-stopping 与 checkpoint

本次不复制 metatrain 的 warmup + cosine scheduler，因为既定要求是由验证总 loss EMA 同时驱动 scheduler 与 early-stopping。继续使用 `ReduceLROnPlateau`：

1. 每个 epoch 完整执行 train 和 validation；
2. 计算 validation 加权总 loss；
3. 更新 EMA；
4. scheduler 使用 EMA 更新；
5. EMA 比历史 best 至少改善 `min_delta` 时保存 best；
6. 未改善 epoch 达到 patience 且已达到 `min_epochs` 时停止。

Checkpoint 保存 active targets、模型、optimizer、scheduler、epoch/global step、best 状态、early-stopping 控制状态、sampler seed、cache identity、run identity 和代码身份。恢复要求 cache、active targets、特征维度、分箱与 cumulant 配置完全匹配。

## 7. W&B 与日志

每个 epoch 上传 active 分支的 train/validation loss、总 loss、EMA 和学习率。未启用分支不上传伪零值。

新增 `logging.log_interval_steps: 100`。训练中每隔指定 step 上传：

- step total loss 与 active 分支 loss；
- samples/s、atoms/s；
- data wait 时间；
- 当前学习率。

Step 日志只用于可见性和性能诊断，不影响 scheduler、best checkpoint 或 early-stopping。恢复时复用已有 W&B run id。正式运行保持 online 模式；本地测试使用禁用或 offline 模式。

## 8. 错误处理与清理边界

- 缓存锁和运行锁继续使用独占、事务式语义。
- NaN/Inf、shape/dtype 变化、结构/原子数量不一致、输入文件变化均使构建失败。
- 训练异常写入失败状态和错误摘要；已提交的完整 epoch checkpoint 可恢复。
- 旧缓存、旧输出、失败日志在新实现验证前不自动删除。
- 新缓存和 GPU 验证成功后，旧 schema v1 缓存与被用户点名的九个失败运行才可作为单独清理步骤删除。
- 本设计不迁移旧图，也不新增绘图功能。

## 9. 测试驱动与验收

### 9.1 单元测试

- 所有 memmap 字段 round-trip；
- 随机结构索引的切片与 offsets 对齐；
- 多 worker 序列化后惰性重开；
- `__getitem__` 不调用 SHA 或全数组有限值扫描；
- schema v1、未完成 manifest、路径逃逸、文件尺寸和 shape 不匹配均拒绝；
- `verify --full` 可检测任一字段内容篡改。

### 9.2 数学与梯度测试

- cumulant order 1–8 与原参考公式输出一致；
- signed root 开关、不同结构原子数均一致；
- 特征输入梯度与参考公式一致；
- 系数为零的分支不构造、不前向、不进入 optimizer；
- 力 `atom_mean`/`component` 标签和逐原子能量标签定义不变；
- 单分支与双分支总 loss 聚合正确。

### 9.3 本地 CPU 验证

使用小数据完成配置解析、cache build、train、checkpoint resume、evaluate 和 full verify。不得在本地执行 GPU 全数据训练。

### 9.4 远端 GPU 验证

1. 小数据完成在线 W&B 全链路；
2. 用正式 v2 memmap 缓存执行限定 step 性能测试；
3. 证明训练阶段 SHA 调用次数为零，且不存在全文件有限值扫描；
4. 前 100 step 内出现 W&B step 指标；
5. 记录 samples/s、atoms/s、data wait、GPU 活跃情况和磁盘读取；
6. 估算完整 epoch 与完整训练耗时；
7. 验证通过后再生成并提交九个正式任务。

量化成功标准：

- 训练阶段 `sha256_file` 调用次数为零；
- 单任务完整训练 epoch 的实际磁盘读取不超过该 split 有效缓存字节数的两倍；
- 在与正式任务相同的 CPU/GPU 资源下，正式训练集首个 epoch 不超过 6 小时；若未达标，不提交九个长期任务；
- W&B 在前 100 step 内出现 step 指标，并持续按配置间隔更新；
- 单分支不构造或执行无关分支；
- 数学、恢复、缓存完整性和 early-stopping 测试全部通过。

CUDA 利用率作为诊断指标记录，但不单独设置固定阈值，因为 ConfidenceHead 的模型规模、batch 原子数和 GPU 型号都会影响该数值；是否通过以吞吐、data wait、磁盘读量和 epoch 时长共同判断。

## 10. 明确不在本次范围内的事项

- 不重新训练或微调基础 UPET；
- 不改变力/能量误差、分箱或监督总 loss 的数学定义；
- 不改成 metatrain 的 cosine scheduler；
- 不直接采用 metatrain `System`/`TensorMap` 缓存抽象；
- 不提供 schema v1 到 v2 的转换器；
- 不迁移、生成或绘制结果图。
