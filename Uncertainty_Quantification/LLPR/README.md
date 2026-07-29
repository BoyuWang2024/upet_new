# UPET LLPR 不确定性量化

本目录用于构建、校准和评估 UPET 的 LLPR 不确定性。代码支持两种 `eta` 使用方式：

- `fixed`：直接指定 energy 和 force 的 `eta`；
- `fit`：只使用验证集拟合 `eta`。

当前正式迁移结果使用 `fixed` 模式。

## 使用方式

```bash
conda activate upet_new
cd /home/lilong/code/UQ/upet_new
```

验证当前正式迁移结果：

```bash
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
```

如果以后需要重新进行完整计算，可以运行：

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/gpu_full_fixed.yaml
```

将配置换成 `gpu_full_fit.yaml` 即可使用拟合 `eta`。旧成果迁移不会运行这两个配置。

## 正式迁移结果

正式结果位于：

```text
Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
```

目录作用如下：

```text
curvature/    规范化曲率结果
calibration/  规范化 eta 和 Alpha 校准结果
evaluation/   规范化正式测试结果
plots/        当前为空
legacy_raw/   原样保留的旧文件
```

下面逐项说明当前目录中的 64 个文件，其中包括 12 个规范化文件和 52 个旧文件。

旧文件原来的逻辑根目录为：

```text
/home/lilong/code/UQ/upet/UQ_LLPR/matpes_r2/Hef
```

## 规范化文件

<!-- BEGIN NORMALIZED FILE CATALOG -->
| 文件 | 作用 | 与旧文件的关系 |
| --- | --- | --- |
| `manifest.json` | 正式迁移结果的总目录。 | 迁移后新增，没有一一对应的旧文件。 |
| `inventory.json` | 记录已保留的全部旧文件。 | 根据旧目录文件生成。 |
| `curvature/406edc88d16fdcc7/manifest.json` | 记录规范化曲率结果包含哪些文件。 | 迁移后新增，没有一一对应的旧文件。 |
| `curvature/406edc88d16fdcc7/curvature.npz` | 保存 energy 和 force 曲率矩阵。 | 从旧 `results/H_E_full_run.npz`、`results/H_F_full_run.npz` 提取，并用 `results/H_EF_full_run.npz` 核对。 |
| `curvature/406edc88d16fdcc7/diagnostics.json` | 保存曲率构建规模和矩阵检查结果。 | 根据旧 `results/Hef_full_run_summary.json` 和旧曲率矩阵整理。 |
| `calibration/50217238109b0418/manifest.json` | 记录规范化校准结果包含哪些文件。 | 迁移后新增，没有一一对应的旧文件。 |
| `calibration/50217238109b0418/candidates.json` | 保存 energy 和 force 的校准候选。 | 根据旧 `results/alpha_val_full_joint_summary.json` 整理。 |
| `calibration/50217238109b0418/summary.json` | 保存当前选用的固定 `eta` 和 Alpha。 | 根据旧 `results/alpha_val_full_joint_summary.json` 整理。 |
| `evaluation/50217238109b0418/28d0d911b3060988/manifest.json` | 记录规范化评估结果包含哪些文件。 | 迁移后新增，没有一一对应的旧文件。 |
| `evaluation/50217238109b0418/28d0d911b3060988/details.npz` | 保存正式测试的完整逐样本结果。 | 由旧 `results/LLPR/llpr_test_full_gpu_details.npz` 中的数组重打包。 |
| `evaluation/50217238109b0418/28d0d911b3060988/summary.json` | 保存正式测试结果摘要。 | 根据旧正式测试 details 和 summary 重新整理。 |
| `evaluation/50217238109b0418/28d0d911b3060988/preview.json` | 提供正式测试结果的轻量预览。 | 根据规范化 details 和 summary 生成。 |
<!-- END NORMALIZED FILE CATALOG -->

## 原样保留的旧文件

以下文件均来自旧目录中的相同相对路径，并原样保留在 `legacy_raw/`。

<!-- BEGIN LEGACY RAW FILE CATALOG -->

### 配置

| 文件 | 作用 | 与旧文件的关系 |
| --- | --- | --- |
| `legacy_raw/configs/compute_Alpha.yaml` | 旧验证集校准配置。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/configs/compute_Hef.yaml` | 旧曲率构建配置。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/configs/compute_LLPR.yaml` | 旧正式测试配置。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/configs/plot_LLPR_reference.yaml` | 旧参考绘图配置。 | 旧目录同路径文件，原样保留。 |

### 曲率、校准和模型检查结果

| 文件 | 作用 | 与旧文件的关系 |
| --- | --- | --- |
| `legacy_raw/results/H_EF_full_run.npz` | 旧 energy 与 force 联合曲率矩阵。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/H_E_full_run.npz` | 旧 energy 曲率矩阵。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/H_F_full_run.npz` | 旧 force 曲率矩阵。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/Hef_full_run_summary.json` | 旧曲率构建摘要。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/alpha_val_full_energy_summary.json` | 旧 energy 校准摘要。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/alpha_val_full_force_summary.json` | 旧 force 校准摘要。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/alpha_val_full_joint_summary.json` | 旧 energy 与 force 联合校准摘要。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/alpha_val_full_run.log` | 旧完整校准运行日志。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/model/check_model.log` | 旧模型检查日志。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/model/check_model_summary.json` | 旧模型结构检查摘要。 | 旧目录同路径文件，原样保留。 |

### 正式评估、测试和绘图结果

| 文件 | 作用 | 与旧文件的关系 |
| --- | --- | --- |
| `legacy_raw/results/llpr_test_details.npz` | 旧小规模测试明细，不是正式结果。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/llpr_test_dry_run.log` | 旧小规模测试运行日志。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/llpr_test_small_preview.json` | 旧小规模测试预览。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/llpr_test_summary.json` | 旧小规模测试摘要。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_test_full_gpu_details.npz` | 旧正式测试完整明细。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_test_full_gpu_run.log` | 旧正式测试运行日志。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_test_full_gpu_small_preview.json` | 旧正式测试预览。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_test_full_gpu_summary.json` | 旧正式测试摘要。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/plot_LLPR.log` | 旧 reliability 绘图日志。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/reliability_matpes_energy_llpr.png` | 旧 energy reliability 图。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/reliability_matpes_force_component_llpr.png` | 旧 force reliability 图。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/reliability_matpes_llpr.png` | 旧额外合并图，未列入正式绘图摘要。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/reliability_matpes_plot_summary.json` | 旧 reliability 绘图摘要。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_energy_uncertainty_vs_residual.pdf` | 旧 energy 不确定性—残差参考图 PDF。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_energy_uncertainty_vs_residual.png` | 旧 energy 不确定性—残差参考图 PNG。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_force_uncertainty_vs_residual.pdf` | 旧 force 不确定性—残差参考图 PDF。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_force_uncertainty_vs_residual.png` | 旧 force 不确定性—残差参考图 PNG。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_reference_plotting_manifest.json` | 旧参考绘图发布清单。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/llpr_reference_plotting_statistics.csv` | 旧参考绘图统计数据。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/fit/fit_LLPR.log` | 旧拟合运行日志。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/fit/reliability_matpes_energy_log_fit.png` | 旧 energy 对数拟合图。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/fit/reliability_matpes_force_component_log_fit.png` | 旧 force 对数拟合图。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/fit/reliability_matpes_linear_fit_summary.json` | 旧线性拟合摘要，配套图片未完整保留。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/results/LLPR/fit/reliability_matpes_log_fit_summary.json` | 旧对数拟合摘要。 | 旧目录同路径文件，原样保留。 |

### 旧脚本、缓存和测试

| 文件 | 作用 | 与旧文件的关系 |
| --- | --- | --- |
| `legacy_raw/scripts/check_model.py` | 旧模型结构检查脚本。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/compute_Alpha.py` | 旧验证集校准脚本。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/compute_Hef.py` | 旧曲率构建脚本。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/compute_LLPR.py` | 旧正式测试计算脚本。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/fit_LLPR.py` | 旧结果拟合脚本。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/plot_LLPR.py` | 旧 reliability 绘图脚本。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/plot_LLPR_reference.py` | 旧参考图生成脚本。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/__pycache__/check_model.cpython-310.pyc` | `check_model.py` 的旧 Python 3.10 缓存。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/__pycache__/compute_Alpha.cpython-310.pyc` | `compute_Alpha.py` 的旧 Python 3.10 缓存。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/__pycache__/compute_Hef.cpython-310.pyc` | `compute_Hef.py` 的旧 Python 3.10 缓存。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/__pycache__/compute_LLPR.cpython-310.pyc` | `compute_LLPR.py` 的旧 Python 3.10 缓存。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/__pycache__/fit_LLPR.cpython-310.pyc` | `fit_LLPR.py` 的旧 Python 3.10 缓存。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/scripts/__pycache__/plot_LLPR.cpython-310.pyc` | `plot_LLPR.py` 的旧 Python 3.10 缓存。 | 旧目录同路径文件，原样保留。 |
| `legacy_raw/tests/test_plot_LLPR_reference.py` | 旧参考绘图测试。 | 旧目录同路径文件，原样保留。 |

<!-- END LEGACY RAW FILE CATALOG -->

## 结果边界

- 正式迁移没有重新加载 checkpoint，也没有重新执行完整模型或 Jacobian 计算。
- 小规模全链路测试结果已在验证完成后删除。
- 当前正式 `plots/` 目录为空；旧图只保留在 `legacy_raw/`。
- 更详细的哈希、数值和迁移审计证据见 [MIGRATION_REPORT.md](MIGRATION_REPORT.md)。
