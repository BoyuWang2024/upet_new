# UPET LLPR

这里是 UPET 的最后一层参数刚度回归（LLPR）工作流。它同时支持：

- 将已经审核的旧固定 η 正式成果原样迁移到新目录；
- 在小数据上重新执行 `build → calibrate → evaluate` 全链路；
- 未来按 fixed 或 fit 两种 η 模式重新计算。

迁移本身不会调用模型。正式旧成果标记为 `legacy_fixed_ridge`，其
`η_energy = η_force = 1e-6` 和 Alpha 均保持旧结果，不会伪造成拟合 η 结果。

## 环境与入口

```bash
source /home/lilong/miniforge3/etc/profile.d/conda.sh
conda activate upet_new
cd /home/lilong/code/UQ/upet_new
python -m Uncertainty_Quantification.LLPR.llpr --help
```

七个命令为 `build`、`calibrate`、`evaluate`、`run`、`import-legacy`、
`verify` 和 `plot`。例如：

```bash
# 唯一允许在本次迁移中重新执行模型的 n20 功能测试
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/cpu_n20_fixed.yaml

# 迁移、完整校验和绘图；三者都不运行模型
python -m Uncertainty_Quantification.LLPR.llpr import-legacy \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml
```

`cpu_n20_fixed.yaml` 与 `cpu_n20_fit.yaml` 只用于链路通畅性验证，不能当成正式
科学结果。两个配置使用相同的 checkpoint、n20 数据和曲率设置，因此复用同一
curvature；它们分别产生独立的 calibration 和 evaluation。

`gpu_full_fixed.yaml` 与 `gpu_full_fit.yaml` 是未来重算配置。本次迁移禁止运行
它们：前者的正式结果来自旧成果导入，后者当前没有正式迁移结果。

## 数学定义

对结构 `s`（原子数 `N_s`），energy 使用每原子预测与残差：

```text
g_E,s = grad(E_pred,s / N_s)
r_E,s = E_pred,s / N_s - E_ref,s / N_s
c_E,s = 1 if |r_E,s| <= delta_E else 0
H_E   = sum_s w_E c_E,s outer(g_E,s, g_E,s)
```

其中 `w_E=1.0`、`delta_E=0.015 eV/atom`。对每个 force 分量：

```text
g_F,s,i = grad(F_pred,s,i)
r_F,s,i = F_pred,s,i - F_ref,s,i
c_F,s,i = 1 if |r_F,s,i| <= delta_F else 0
H_F = sum_s [w_F/(3 N_s)] sum_i c_F,s,i outer(g_F,s,i, g_F,s,i)
```

其中 `w_F=0.1`、`delta_F=0.01 eV/angstrom`。energy 与 force 使用互不重叠
的 readout 参数块。对目标 `t`：

```text
A_t        = H_t + eta_t I
q_t        = g_t^T solve(A_t, g_t)
alpha_t^2  = mean_validation(residual_t^2 / q_t)
variance_t = alpha_t^2 q_t
std_t      = sqrt(variance_t)
rigidity_t = 1 / variance_t
```

全部矩阵、分解、q 和校准累计使用 float64。旧诊断曾显示 `H + eta^2 I`，
但旧正式 Alpha/LLPR 实际使用的是 `H + eta I`；迁移后的规范诊断也以
`H + eta I` 为准。

## fixed 与 fit

fixed 模式直接分别指定 energy/force η。条件数超过阈值时只记录风险，不会
偷偷改变 η，也不会添加隐藏 jitter；Cholesky 失败会直接终止。旧正式结果虽能
完成求解，但固定 `1e-6` 对应的条件数很高，因此迁移成功不等于数值稳健。

fit 模式分别计算 energy 与 force 的谱下界，再按配置倍数生成候选。每个候选只
在完整 validation 上闭式校准 Alpha，并以 Gaussian NLL 选择；test 只消费最终
η/Alpha，绝不参与选择。若无有效候选则失败。

## 输出、身份与恢复

```text
outputs/<experiment>/
├── curvature/<identity>/
├── calibration/<curvature-id>/<identity>/
├── evaluation/<calibration-id>/<identity>/
├── plots/<identity>/
└── legacy_raw/
```

每阶段 identity 绑定其真实输入、公式版本和相关配置。fixed 与 fit 不影响
curvature identity，因此可安全复用曲率。曲率和评估会保存进度，重启时严格
核对 identity 与分片哈希；完成标志只在最终文件完整后发布。重复执行已完成且
身份一致的操作是幂等的。

`verify` 的 `--config` 参数在该命令下表示输出目录，并执行 full 校验。导入会
保留旧文件清单和原始文件副本，同时生成规范 curvature/calibration/evaluation
视图。`plot` 只读规范 `details.npz`，不会重新拟合 η、Alpha，也不会改写评估。
