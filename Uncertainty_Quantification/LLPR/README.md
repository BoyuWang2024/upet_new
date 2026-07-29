# UPET LLPR 不确定性量化

本目录提供一套完整的 LLPR（Last-Layer Parameter Regression）流程，用于计算能量和力的不确定性。

当前正式结果位于：

```text
Uncertainty_Quantification/LLPR/outputs/matpes_r2
```

该结果使用直接指定的 η：

```text
energy η = 1.0e-6
force  η = 1.0e-6
```

正式结果已经计算完成，日常使用时不需要重新计算。

## 环境与数据

在 Ubuntu 中进入仓库并激活环境：

```bash
cd /home/lilong/code/UQ/upet_new
conda activate upet_new
```

代码默认使用以下文件：

```text
data/checkpoint/pet-omatpes-l-v0.1.0.ckpt
data/dataset/matpes_train.extxyz
data/dataset/matpes_val.extxyz
data/dataset/matpes_test.extxyz
data/dataset/matpes_n20.extxyz
```

## 常用命令

### 验证当前正式结果

```bash
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2
```

### 绘制当前正式结果

```bash
python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/plot_matpes_r2.yaml
```

图片和绘图统计会写入：

```text
Uncertainty_Quantification/LLPR/plots/matpes_r2
```

绘图不会修改 `outputs/matpes_r2`。

### 小规模全链路

直接指定 η：

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml
```

拟合 η：

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fit.yaml
```

### 完整数据重新计算

直接指定 η：

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/gpu_full_fixed.yaml
```

拟合 η：

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/gpu_full_fit.yaml
```

完整数据计算量很大。`gpu_full_fixed.yaml` 对应当前正式结果；只要正式文件完整，代码会直接验证并复用它们。

## 计算流程

```text
checkpoint + 训练集
        ↓
曲率矩阵
        ↓
验证集 + η（直接指定或拟合）
        ↓
校准参数
        ↓
测试集评估
        ↓
按需绘图
```

## 新代码文件

| 文件 | 作用 |
|---|---|
| `__init__.py` | 将 `LLPR` 声明为 Python 包。 |
| `llpr/__init__.py` | 定义 LLPR 子包。 |
| `llpr/__main__.py` | 支持通过 `python -m Uncertainty_Quantification.LLPR.llpr` 运行命令。 |
| `llpr/cli.py` | 提供构建、校准、评估、完整运行、验证和绘图命令。 |
| `llpr/config.py` | 读取并严格检查 YAML 配置。 |
| `llpr/artifacts.py` | 管理结果目录、清单、自动编号、原子写入和完整校验。 |
| `llpr/checkpoint.py` | 检查 checkpoint，并加载模型和训练损失参数。 |
| `llpr/data.py` | 读取数据集、标签和数据集统计信息。 |
| `llpr/readout.py` | 定位模型最后一层的能量与力参数。 |
| `llpr/observables.py` | 计算参数 Jacobian 和 Huber 曲率权重。 |
| `llpr/curvature.py` | 构建能量和力的曲率矩阵。 |
| `llpr/curvature_progress.py` | 保存并检查曲率构建的断点。 |
| `llpr/ridge.py` | 处理 ridge 矩阵、候选 η 和二次型。 |
| `llpr/calibration.py` | 使用验证集计算 Alpha，并直接使用或拟合 η。 |
| `llpr/calibration_progress.py` | 保存并检查校准阶段的断点。 |
| `llpr/inference.py` | 在测试集上计算预测、残差和不确定性。 |
| `llpr/evaluation_shards.py` | 保存并检查分片评估结果。 |
| `llpr/plotting.py` | 从正式评估结果生成散点图、可靠性图和统计表。 |

## 配置文件

| 文件 | 作用 |
|---|---|
| `configs/cpu_n20_fixed.yaml` | 使用 20 个结构在 CPU 上测试直接指定 η 的全链路。 |
| `configs/cpu_n20_fit.yaml` | 使用 20 个结构在 CPU 上测试拟合 η 的全链路。 |
| `configs/gpu_full_fixed.yaml` | 使用完整数据和 GPU，直接指定 η；对应当前正式结果。 |
| `configs/gpu_full_fit.yaml` | 使用完整数据和 GPU，根据验证集拟合 η。 |
| `configs/plot_matpes_r2.yaml` | 读取当前正式结果，并将图片写到独立绘图目录。 |

## 正式结果文件

```text
outputs/matpes_r2/
├── manifest.json
├── curvature/981cc8bdf7f8b820/
│   ├── manifest.json
│   ├── curvature.npz
│   └── diagnostics.json
├── calibration/169d1d75c0dc1190/
│   ├── manifest.json
│   ├── candidates.json
│   └── summary.json
└── evaluation/169d1d75c0dc1190/519237da91c72b73/
    ├── manifest.json
    ├── details.npz
    ├── summary.json
    └── preview.json
```

| 文件 | 作用 |
|---|---|
| `outputs/matpes_r2/manifest.json` | 连接当前正式结果的曲率、校准和评估阶段。 |
| `curvature/981cc8bdf7f8b820/manifest.json` | 记录曲率阶段使用的模型、数据、参数和结果文件。 |
| `curvature/981cc8bdf7f8b820/curvature.npz` | 保存能量和力的正式曲率矩阵。 |
| `curvature/981cc8bdf7f8b820/diagnostics.json` | 保存曲率维度、训练集数量和矩阵诊断信息。 |
| `calibration/169d1d75c0dc1190/manifest.json` | 记录校准阶段依赖的曲率、验证集和参数。 |
| `calibration/169d1d75c0dc1190/candidates.json` | 保存能量和力的 η 校准候选结果。 |
| `calibration/169d1d75c0dc1190/summary.json` | 保存最终使用的 η、Alpha 和校准统计。 |
| `evaluation/169d1d75c0dc1190/519237da91c72b73/manifest.json` | 记录测试评估使用的曲率、校准和测试集。 |
| `evaluation/169d1d75c0dc1190/519237da91c72b73/details.npz` | 保存逐结构、逐原子和逐力分量的完整结果。 |
| `evaluation/169d1d75c0dc1190/519237da91c72b73/summary.json` | 保存测试集上的汇总指标。 |
| `evaluation/169d1d75c0dc1190/519237da91c72b73/preview.json` | 保存便于快速查看的少量结果预览。 |

目录中的自动编号由代码生成，用于避免不同模型、数据或参数的结果混用，不需要手动修改。

## 测试文件

| 文件 | 作用 |
|---|---|
| `tests/conftest.py` | 提供测试共用设置。 |
| `tests/test_artifacts.py` | 测试清单、编号、写入和文件校验。 |
| `tests/test_release_contract.py` | 测试正式根清单不包含来源字段并可独立验证。 |
| `tests/test_config.py` | 测试配置读取和错误输入。 |
| `tests/test_cli.py` | 测试命令行入口和执行顺序。 |
| `tests/test_checkpoint_readout_data.py` | 测试 checkpoint、最后一层和数据读取。 |
| `tests/test_curvature.py` | 测试曲率计算。 |
| `tests/test_curvature_progress_validation.py` | 测试曲率断点文件的完整性检查。 |
| `tests/test_n20_curvature_resume.py` | 测试小规模曲率计算中断后恢复。 |
| `tests/test_ridge_calibration.py` | 测试 ridge 和 η 校准。 |
| `tests/test_calibration_progress.py` | 测试校准断点的保存与恢复。 |
| `tests/test_applied_calibration.py` | 测试校准参数能被评估阶段正确读取。 |
| `tests/test_inference.py` | 测试评估结果的合并和统计。 |
| `tests/test_plotting.py` | 测试绘图计算、独立输出和只读行为。 |
| `tests/test_n20.py` | 使用真实 checkpoint 和 20 个结构测试两种 η 模式的完整链路。 |

`__pycache__/`、`.pyc`、运行日志和临时测试结果不属于发布内容。
