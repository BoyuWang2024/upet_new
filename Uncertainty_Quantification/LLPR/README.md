# UPET LLPR 不确定性量化

本目录提供能量和力的 LLPR（Last-Layer Parameter Regression）不确定性量化流程。当前可直接使用已有 MATPES-test 结果，并在不重建曲率矩阵的情况下计算 MAD-test 和 MATPES-train。

## 环境和数据

在 Ubuntu 中进入仓库并激活环境：

```bash
cd /home/lilong/code/UQ/upet_new
conda activate upet_new
```

正式流程使用：

- `data/checkpoint/pet-omatpes-l-v0.1.0.ckpt`：UPET 模型。
- `data/dataset/matpes_train.extxyz`：MATPES 训练集；远端计算可以复用语义一致的现有 extxyz。
- `data/dataset/matpes_val.extxyz`：MATPES 验证集。
- `data/dataset/matpes_test.extxyz`：MATPES 测试集。
- `data/dataset/mad-val.xyz`：MAD 验证集，用于重新计算 MAD 的 Alpha。
- `data/dataset/mad-test.xyz`：MAD 测试集。
- `data/dataset/matpes_n20.extxyz`：本地小规模全链路测试数据。

`mad-val.xyz` 和 `mad-test.xyz` 分别与兼容版 MAD 文件字节一致。

## 三个正式结果

- `outputs/matpes_r2`：已有 MATPES-test 完整结果，保留不变。
- `outputs/mad_test`：复用已有曲率，在 MAD 验证集重新计算 Alpha，再评估 MAD-test。
- `outputs/matpes_train`：复用已有曲率和 Alpha，只评估 MATPES-train。

当前 η 直接指定为：

```text
energy η = 1.0e-6
force  η = 1.0e-6
```

代码也保留 η 拟合模式，见 `cpu_n20_fit.yaml` 和 `gpu_full_fit.yaml`。

## 常用命令

先验证已有 MATPES-test 结果：

```bash
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2
```

运行 20 个结构的小规模完整链路：

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml
```

计算 MAD 的 Alpha 和测试集 UQ：

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/gpu_mad_test_fixed.yaml
```

复用已有 Alpha，计算 MATPES-train UQ：

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/gpu_matpes_train_fixed.yaml
```

生成三个数据集的六张图：

```bash
python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/plot_three_datasets.yaml
```

绘图目录为 `Uncertainty_Quantification/Plots/LLPR`，包含六张 PNG、六张 PDF、`plotting_statistics.csv` 和 `plotting_manifest.json`。

## 计算流程

```text
已有 checkpoint + 已有曲率
                │
        ┌───────┴────────┐
        │                │
MAD 验证集重新算 Alpha   MATPES 直接复用 Alpha
        │                │
    MAD-test UQ      MATPES-train UQ
        └───────┬────────┘
                │
       与 MATPES-test 一起绘图
```

曲率和校准复用前都会检查清单、身份和文件完整性。复用结果会放入新实验目录，已有 `outputs/matpes_r2` 不会被修改。

## 代码文件

| 文件 | 作用 |
|---|---|
| `llpr/__init__.py` | 定义 LLPR Python 子包。 |
| `llpr/__main__.py` | 支持 `python -m Uncertainty_Quantification.LLPR.llpr`。 |
| `llpr/cli.py` | 提供 build、calibrate、evaluate、run、verify 和 plot 命令。 |
| `llpr/config.py` | 严格读取 LLPR YAML，并检查数据、η 和复用设置。 |
| `llpr/artifacts.py` | 生成结果身份，原子写入文件，并校验清单和文件。 |
| `llpr/checkpoint.py` | 校验并加载 checkpoint。 |
| `llpr/data.py` | 按顺序读取 extxyz、能量标签和力标签。 |
| `llpr/dataset_fingerprint.py` | 比较两个 extxyz 的结构、坐标、能量和力是否语义一致。 |
| `llpr/readout.py` | 找到模型最后一层中用于能量和力的参数。 |
| `llpr/observables.py` | 计算 Jacobian 和 Huber 曲率权重。 |
| `llpr/curvature.py` | 构建或复用能量、力曲率矩阵。 |
| `llpr/curvature_progress.py` | 保存并校验曲率阶段的断点。 |
| `llpr/reuse.py` | 将已校验的曲率或校准产物安全放入新实验目录。 |
| `llpr/ridge.py` | 构造 ridge 矩阵并处理 η。 |
| `llpr/calibration.py` | 在验证集上计算 Alpha，并支持固定或拟合 η。 |
| `llpr/calibration_progress.py` | 保存并校验校准阶段的断点。 |
| `llpr/inference.py` | 计算预测、残差和校准后的 UQ。 |
| `llpr/evaluation_shards.py` | 保存、恢复和合并分片推理结果。 |
| `llpr/plotting.py` | 提供原有单结果统计和绘图辅助函数。 |
| `llpr/plot_multi.py` | 读取一个或多个现有评估结果并发布统一风格图片。 |
| `llpr/export.py` | 导出不含大型 `details.npz` 的可验证摘要。 |

## 配置文件

| 文件 | 作用 |
|---|---|
| `configs/cpu_n20_fixed.yaml` | CPU 小数据全链路，直接指定 η。 |
| `configs/cpu_n20_fit.yaml` | CPU 小数据全链路，拟合 η。 |
| `configs/gpu_full_fixed.yaml` | 完整 MATPES 数据重新计算，直接指定 η。 |
| `configs/gpu_full_fit.yaml` | 完整 MATPES 数据重新计算，拟合 η。 |
| `configs/gpu_mad_test_fixed.yaml` | 复用曲率，重新计算 MAD Alpha 和 MAD-test UQ。 |
| `configs/gpu_matpes_train_fixed.yaml` | 复用曲率与 Alpha，只计算 MATPES-train UQ。 |
| `configs/plot_matpes_r2.yaml` | 只绘制已有 MATPES-test 结果。 |
| `configs/plot_three_datasets.yaml` | 一次绘制 MATPES-test、MAD-test 和 MATPES-train。 |

## 结果文件

每个完整实验都按以下结构保存：

```text
outputs/<实验名>/
├── manifest.json
├── curvature/<身份>/
│   ├── manifest.json
│   ├── curvature.npz
│   └── diagnostics.json
├── calibration/<身份>/
│   ├── manifest.json
│   ├── candidates.json
│   └── summary.json
└── evaluation/<校准身份>/<评估身份>/
    ├── manifest.json
    ├── details.npz
    ├── summary.json
    └── preview.json
```

| 文件 | 作用 |
|---|---|
| 根目录 `manifest.json` | 连接一次实验使用的曲率、校准和评估。 |
| 曲率 `manifest.json` | 记录模型、训练集和曲率设置。 |
| `curvature.npz` | 保存能量和力曲率矩阵。 |
| `diagnostics.json` | 保存矩阵维度、样本数和数值诊断。 |
| 校准 `manifest.json` | 记录验证集、η 和对应曲率。 |
| `candidates.json` | 保存固定或拟合 η 的候选计算。 |
| 校准 `summary.json` | 保存最终 η、Alpha 和校准统计。 |
| 评估 `manifest.json` | 记录测试集、曲率、校准和结果文件。 |
| `details.npz` | 保存逐结构、逐原子和逐力分量的完整结果。 |
| 评估 `summary.json` | 保存整体误差和 UQ 指标。 |
| `preview.json` | 保存少量便于查看的结果。 |

本地保留完整 MAD-test 结果。MATPES-train 的完整 `details.npz` 保留在远端计算目录，本地只保留 `summary.json`、`preview.json` 和摘要 `manifest.json`。摘要清单会记录远端完整结果身份和 `details.npz` 的校验值。

## 测试文件

| 文件 | 作用 |
|---|---|
| `tests/conftest.py` | 测试共用配置和临时数据。 |
| `tests/test_artifacts.py` | 测试身份、清单、写入和校验。 |
| `tests/test_config.py` | 测试 YAML 和复用配置。 |
| `tests/test_reuse.py` | 测试复用产物的复制、链接和回滚。 |
| `tests/test_stage_resolution.py` | 测试各命令是否正确选择复用或计算。 |
| `tests/test_dataset_fingerprint.py` | 测试 extxyz 语义一致性。 |
| `tests/test_export.py` | 测试小型摘要导出。 |
| `tests/test_formal_configs.py` | 测试正式推理和绘图配置。 |
| `tests/test_plotting.py` | 测试多数据集绘图和 14 文件发布。 |
| `tests/test_checkpoint_readout_data.py` | 测试 checkpoint、最后一层和数据标签。 |
| `tests/test_curvature.py` | 测试曲率计算。 |
| `tests/test_curvature_progress_validation.py` | 测试曲率断点校验。 |
| `tests/test_n20_curvature_resume.py` | 测试小数据曲率中断恢复。 |
| `tests/test_ridge_calibration.py` | 测试 ridge、η 和 Alpha。 |
| `tests/test_calibration_progress.py` | 测试校准中断恢复。 |
| `tests/test_applied_calibration.py` | 测试评估是否正确应用 Alpha。 |
| `tests/test_inference.py` | 测试推理合并和汇总。 |
| `tests/test_n20.py` | 使用真实 checkpoint 做小数据全链路测试。 |
| `tests/test_cli.py` | 测试命令分发。 |
| `tests/test_release_contract.py` | 测试发布结果不依赖来源说明。 |

`__pycache__/`、`.pyc`、日志、断点临时目录和小规模测试输出不属于正式发布内容。
