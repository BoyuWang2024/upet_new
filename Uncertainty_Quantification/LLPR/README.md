# UPET LLPR 不确定性量化

本目录提供 UPET 最后一层拉普拉斯近似（LLPR）的构建、校准、评估、旧成果迁移、
完整性验证和绘图工具。当前正式成果采用固定 `eta`，同时代码支持：

- `fixed`：直接指定 energy/force 的 `eta`；
- `fit`：在验证集上分别拟合 energy/force 的 `eta`，测试集不参与选择。

## 环境与命令

```bash
conda activate upet_new
cd /home/lilong/code/UQ/upet_new
```

统一入口包含 `build`、`calibrate`、`evaluate`、`run`、`import-legacy`、
`verify` 和 `plot`：

```bash
python -m Uncertainty_Quantification.LLPR.llpr import-legacy \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml

python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy

python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml
```

`verify` 会验证所有 complete manifest、声明文件哈希、NPZ 可读性，以及
`legacy_raw` inventory 中每个文件的安全相对路径、大小和 SHA-256。

## 正式旧成果迁移

正式成果位于：

```text
Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
```

正式迁移不会加载 checkpoint，也不会重新执行完整 train/validation/test 的模型前向或
Jacobian 计算。它只审计旧文件并生成规范化视图：

```text
outputs/<experiment>/
├── curvature/<identity>/
├── calibration/<identity>/
├── evaluation/<calibration-id>/<identity>/
├── plots/<identity>/
└── legacy_raw/
```

三类数据来源严格分离：

- curvature 的结构数来自 `Hef_full_run_summary.json`；
- Alpha 与验证计数来自 `alpha_val_full_joint_summary.json`；
- 测试计数和预测明细来自正式 test summary/details。

旧验证摘要没有保存逐样本验证残差和方差，因此无法从旧成果恢复 Gaussian NLL 和
1/2/3 sigma coverage。这些字段明确保存为 `null`，并标记
`diagnostics_status: unavailable_from_legacy_validation_summary`；不会使用测试集结果替代。
推理阶段只消费经过验证的 `eta`、`Alpha` 和 `Alpha^2`。

所有正式数值阶段标记 `origin: legacy_import`。新生成的图标记
`origin: derived` 和 `source_origin: legacy_import`，不会冒充旧图。

## 小规模全链路

`cpu_n20_fixed.yaml` 与 `cpu_n20_fit.yaml` 只用于真实小规模链路测试，不是正式科学结果。
测试覆盖固定/拟合两种模式，以及曲率计算中断后的严格恢复和与不间断运行的逐数组比较：

```bash
UPET_RUN_LLPR_N20=1 tox -e llpr-tests -- -m llpr_n20 -v
```

`gpu_full_fixed.yaml` 和 `gpu_full_fit.yaml` 是未来正式重算配置。本次迁移禁止运行它们。

## 数学定义

对目标 `t`：

```text
A_t        = H_t + eta_t I
q_t        = g_t^T solve(A_t, g_t)
Alpha_t^2  = mean_validation(residual_t^2 / q_t)
variance_t = Alpha_t^2 q_t
std_t      = sqrt(variance_t)
rigidity_t = 1 / variance_t
```

energy 使用每原子残差，force 使用逐笛卡尔分量残差。矩阵、分解、二次型和校准累积均使用
float64。固定 `eta` 的条件数告警只记录风险，不会静默改变 `eta` 或添加隐藏 jitter；
Cholesky 失败会直接终止。

完整迁移证据和正式 identity 见 [MIGRATION_REPORT.md](MIGRATION_REPORT.md)。
