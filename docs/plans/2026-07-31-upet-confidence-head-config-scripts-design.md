# UPET ConfidenceHead 配置化训练入口设计

日期：2026-07-31  
状态：用户已确认

## 1. 背景与目标

当前 `Uncertainty_Quantification/ConfidenceHead` 已具有缓存构建、训练、评估和产物验证工作流，但缺少可直接运行的配置文件和薄脚本。训练参数仍有一部分固化在代码中，运行者难以像 `carnet_new/Uncertainty_Quantification/ConfidenceHead` 一样通过一份 YAML 完整描述实验。

本次工作的目标是参考 `carnet_new` 的组织格式，为 UPET ConfidenceHead 增加完整、严格且可复现的 YAML 配置与四阶段脚本，同时保持已经确认的 UPET 计算语义：

- 力误差按笛卡尔分量监督；
- 能量误差按原子归一化后监督；
- 力 confidence head 与能量 confidence head 使用 UPET 各自独立读出的特征；
- early stopping 和学习率调度共同监督验证集监督总 loss 的 EMA；
- 固定线性分箱，力误差上限默认 `0.5`，逐原子能量误差上限默认 `0.3`；
- 不迁移旧代码、旧配置、旧日志或旧结果，不加入绘图。

## 2. 目录结构

新增以下入口和配置：

```text
Uncertainty_Quantification/ConfidenceHead/
├── configs/
│   ├── n20_local_cpu.yaml
│   ├── n20_cpu.yaml
│   └── full_gpu.yaml
├── scripts/
│   ├── build_cache.py
│   ├── train.py
│   ├── evaluate.py
│   └── verify.py
└── confidence_head/
    └── ...
```

三个 YAML 都是可以独立阅读和运行的完整配置，不采用基础配置继承，也不依靠命令行覆盖来保存关键实验参数。四个脚本只接收 `--config`，业务逻辑继续位于 `confidence_head` 包中。

## 3. 配置结构

完整配置包含以下顶层字段：

```yaml
profile: smoke
allow_identical_splits: true

checkpoint:
  path: ...
  expected_sha256: ...

data:
  train:
    path: ...
    expected_sha256: ...
  validation:
    path: ...
    expected_sha256: ...
  test:
    path: ...
    expected_sha256: ...

readouts:
  energy_prediction: energy
  force_prediction: non_conservative_forces
  energy_features: mtt::aux::energy_last_layer_features
  force_features: mtt::aux::non_conservative_forces_last_layer_features

cache:
  batch_size: 2
  num_workers: 0
  pin_memory: false
  shard_max_atoms: 100000

binning:
  algorithm: fixed_linear_v1
  force_max_error: 0.5
  energy_max_error: 0.3

model:
  force:
    enabled: true
    hidden_dims: [16]
    dropout: 0.0
    num_bins: 50
  energy:
    enabled: true
    hidden_dims: [16]
    dropout: 0.0
    num_bins: 50
    cumulant_order: 3
    signed_root: true

loss:
  force_coefficient: 1.0
  energy_coefficient: 1.5
  label_smoothing: 0.0
  class_weights: null

optimizer:
  name: adamw
  learning_rate: 0.001
  weight_decay: 0.0

scheduler:
  name: reduce_lr_on_plateau
  monitor: val/total_loss_ema
  factor: 0.5
  patience: 5
  threshold: 0.0001
  threshold_mode: abs
  cooldown: 0
  min_lr: 0.000001

trainer:
  batch_size: 2
  max_epochs: 1
  ema_beta: 0.95
  early_stopping_patience: 1
  min_delta: 0.0001
  min_epochs: 1
  monitor: val/total_loss_ema
  resume_from: null

run:
  name_prefix: upet_n20_cpu
  output_root: ...
  seed: 1234
  device: cpu
  amp: false

logging:
  jsonl: true
  wandb: true
  wandb_mode: offline
  wandb_project: upet-confidence-head
```

配置继续使用 Pydantic 严格校验：禁止未知字段、禁止重复 YAML 键，并对数值范围、profile 与数据 split 关系、AMP/device 组合以及固定监控指标进行验证。scheduler 的数值参数由 YAML 控制，不再在模型校验器中强制为一组常数。

### 3.1 独立双头配置

力头和能量头具有独立的 `hidden_dims`、`dropout`、`num_bins` 和 `enabled` 字段。能量头另外具有 `cumulant_order` 与 `signed_root`。模型构造时分别传入缓存记录的 `force_feature_dim` 与 `energy_feature_dim`，不得拼接、互换或共享两种输入。

### 3.2 分离两种 batch size

`cache.batch_size` 只控制 UPET 基础模型前向与特征提取；`trainer.batch_size` 只控制已经缓存的 confidence head 训练。这样可以对大 checkpoint 使用保守的特征提取 batch，同时提高轻量读出头的训练吞吐量。

## 4. 三套配置

### 4.1 本地 n20 配置

`n20_local_cpu.yaml` 使用仓库相对路径：

- `data/checkpoint/pet-omatpes-l-v0.1.0.ckpt`
- `data/dataset/matpes_n20.extxyz`

训练、验证、测试均指向同一个 n20 文件，配置为 `profile: smoke`、`allow_identical_splits: true`。W&B 使用 `offline`。

### 4.2 远端 n20 配置

`n20_cpu.yaml` 使用用户指定的远端绝对路径：

- checkpoint：`/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/pet-omatpes-l-v0.1.0.ckpt`
- 三个 split：`/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_n20.extxyz`

该配置仅用于 CPU 全链路冒烟测试，不用于评价泛化能力。W&B 使用 `offline`。

### 4.3 完整 GPU 配置

`full_gpu.yaml` 使用同一个 checkpoint，以及用户指定的三个独立数据集：

- train：`/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_train.extxyz`
- validation：`/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_val.extxyz`
- test：`/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_test.extxyz`

该配置为 `profile: production`，禁止重复 split，默认 `device: cuda`，W&B 使用 `online`。本次只提供并静态验证该配置，不启动 GPU 全数据训练。

### 4.4 文件身份

已在目标服务器只读计算并确认：

| 文件 | SHA256 |
|---|---|
| checkpoint | `879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004` |
| local n20 | `c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d` |
| remote n20 | `f6516b4066f5c05b0b74061ef8f332c58c397098d575ec6b99fcf177faa15d8b` |
| train | `42bc5b908fbd70da740175f824fd87169dcc4bf62258e5096c3eda7e3372eae1` |
| validation | `5f267bbe8a6690196470ee86c04cc7a3ee2839ccbda542a7bff4bea88b9e9078` |
| test | `76d0c38064af8a904eac1b736b07e2ed952838d4311574d83664a4967ba6ec98` |

## 5. 默认超参数

| 参数 | n20 local/remote | full GPU |
|---|---:|---:|
| cache batch size | 2 | 2 |
| trainer batch size | 2 | 128 |
| force hidden dims | `[16]` | `[256, 256, 256]` |
| energy hidden dims | `[16]` | `[256, 256, 256]` |
| force bins | 50 | 50 |
| energy bins | 50 | 50 |
| force max error | 0.5 | 0.5 |
| energy max error | 0.3 | 0.3 |
| force loss coefficient | 1.0 | 1.0 |
| energy loss coefficient | 1.5 | 1.5 |
| learning rate | 0.001 | 0.001 |
| max epochs | 1 | 100 |
| EMA beta | 0.95 | 0.95 |
| early-stopping patience | 1 | 20 |
| device | cpu | cuda |
| AMP | false | false |
| W&B mode | offline | online |

`energy_max_error` 是逐原子能量误差的分箱上限；`energy_coefficient` 是监督总 loss 中的能量分支权重，两者语义不同。

## 6. 运行命名与产物发现

运行名参考 `carnet_new`，由 `name_prefix` 与关键模型语义组成，例如：

```text
upet_full_fixed-linear_f50-fmax0.5-fw1-fmlp256x256x256_e50-emax0.3-ew1.5-emlp256x256x256-order3
```

调整分箱、误差上限、loss 权重或网络宽度将得到不同运行目录。已存在的非恢复运行目录禁止覆盖。

缓存身份只包含 checkpoint、数据、UPET readout、特征维度和提取执行策略，不包含 confidence head 隐藏层或优化器参数。训练脚本扫描完整的内容寻址 cache manifest，并通过严格身份验证选择唯一匹配项：零个或多个匹配项都报错，不隐式重建缓存。

## 7. 四阶段脚本

标准使用方式：

```bash
python scripts/build_cache.py --config configs/n20_cpu.yaml
python scripts/train.py --config configs/n20_cpu.yaml
python scripts/evaluate.py --config configs/n20_cpu.yaml
python scripts/verify.py --config configs/n20_cpu.yaml
```

- `build_cache.py`：验证输入、分别提取两类 UPET readout、发布缓存并打印 manifest 路径。
- `train.py`：自动发现唯一匹配缓存，派生运行名，训练双头并发布训练 manifest。
- `evaluate.py`：自动定位对应运行和缓存，默认评估 `best.pt`，只生成数值产物。
- `verify.py`：验证 manifest、身份、哈希、checkpoint 和评估产物，不训练、不绘图。

`trainer.resume_from` 由 YAML 控制。恢复训练要求 config、cache、binning、模型和 loss 身份完全一致，并继续写入同一运行目录。

## 8. 训练控制逻辑

监督总 loss 定义为：

```text
total_loss = force_coefficient * force_loss
           + energy_coefficient * energy_loss
```

每轮在完整验证集上计算 `val/total_loss`，再以 `ema_beta` 更新 `val/total_loss_ema`。以下三项均使用该 EMA：

- best checkpoint 判定；
- ReduceLROnPlateau 学习率调度；
- early stopping。

原始分支 loss、总 loss、EMA、学习率、epoch 和 global step 都写入 JSONL 和 W&B。`min_epochs` 之前不得 early stop。

## 9. W&B 集成

W&B 由 `logging.wandb` 控制，并从 YAML 读取 mode 与 project。每轮记录：

- `train/force_loss`、`train/energy_loss`、`train/total_loss`；
- `val/force_loss`、`val/energy_loss`、`val/total_loss`；
- `val/total_loss_ema`、`learning_rate`、`epoch`、`global_step`。

运行完成后在 summary 中记录 best epoch、best EMA 和 stop reason。恢复训练复用 manifest 中的 W&B run ID，防止生成两条不连续记录。JSONL 和本地 manifest 是权威记录；W&B 初始化或上传异常产生明确警告，但不得破坏 checkpoint 或使已经完成的训练事务失效。W&B 生命周期必须在成功、early stop 和异常退出路径中正确结束。

## 10. 错误处理与安全约束

- checkpoint、数据文件缺失或 SHA256 不一致时立即停止；
- production 配置禁止相同 split；
- cache 缺失或不唯一时立即停止；
- UPET 缺少任一所需 readout 时立即停止；
- 缓存中的力/能量特征维度与模型输入不一致时立即停止；
- 运行目录禁止路径逃逸、符号链接替换和非恢复覆盖；
- 恢复训练继续使用事务式写入和失败回滚；
- 不读取或迁移任何旧 ConfidenceHead 结果。

## 11. 验证方案

1. 本地单元与静态验证：配置解析、重复键和未知键拒绝、三份配置、运行命名、缓存发现、双头独立结构、scheduler 参数传递、监督总 loss EMA early stopping、W&B mock 生命周期与恢复语义。
2. 本地 n20 动态验证：使用 `n20_local_cpu.yaml` 运行四阶段小数据流程，确认 W&B offline 目录、逐轮指标、checkpoint、evaluation 和 verify。
3. 提交当前 `ConfidenceHead` 分支并推送 GitHub。
4. 目标服务器从 GitHub 拉取同一提交。
5. 目标服务器使用 `n20_cpu.yaml` 做 CPU 四阶段全链路测试。
6. 核验 cache manifest、best/last checkpoint、JSONL、W&B offline、evaluation 和 verify 结果。
7. 不进行 GPU 全数据训练，也不生成图。

