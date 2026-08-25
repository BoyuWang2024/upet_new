# BootStrapping 全电子能量 E0 后处理设计

日期：2026-08-25
状态：设计已确认，等待用户审阅文档
范围：仅对已有 BootStrapping prediction 做能量后处理；不训练、不修改模型推理逻辑。

## 1. 背景与目标

MAD r2SCAN EXTXYZ 同时提供 `atomization_energy`（原子化能量）和 `energy`（FHI-aims 全电子绝对总能量）。当前三组 BootStrapping raw prediction 使用 UPET/MATPES 能量零点；本设计在不重新训练、不改变模型输入和推理代码的前提下，针对 MAD r2SCAN 执行两种后处理实验。MatPES 结果以及 raw prediction、raw UQ、raw targets 永远不覆盖。

三组 run 为 `full_remote_b8_e8`、`lr_1e-4`、`lr_1e-6`，每组使用完整 8-member raw ensemble。

## 2. 科学定义

### 2.1 方法一：test 逐结构全电子基线替换

对 MAD test 每个结构读取同一构型的 `energy` 与 `atomization_energy`，定义：

\[
b_i=E_i^{MAD,absolute}-E_i^{MAD,atomization}.
\]

对每个 run 的每个成员：

\[
E^{(1)}_{i,m}=E^{raw}_{i,m}+b_i.
\]

同一结构的 8 个成员使用同一个 offset。该方法不使用 val，使用 test reference，属于 oracle 后处理；不修改 force/stress，不改变 energy std、GMD 或成员排序，不用于力的重新求导或 MD。

### 2.2 方法二：val ensemble-mean model-aware E0 校准

三个 run 分别校准，各自产生一组逐元素修正量。对某个 run：

\[
\bar E_i^{val}=\frac{1}{8}\sum_{m=1}^{8}E^{raw,val}_{i,m},\qquad X_{iZ}=n_{iZ},\qquad y_i=E_i^{MAD,absolute,val}.
\]

按照论文公式求：

\[
\Delta E_0=\arg\min_{\Delta E_0}\|\bar E^{val}-X\Delta E_0-y\|_2^2.
\]

实现使用 float64 SVD 最小范数解：

```python
delta_e0 = np.linalg.lstsq(
    X_val,
    mean_energy_val - absolute_energy_val,
    rcond=None,
)[0]
```

将该 run 的同一组修正量用于 test 全部 8 个成员：

\[
E^{(2)}_{i,m}=E^{raw,test}_{i,m}-\sum_Z n_{iZ}\Delta E_{0,Z}.
\]

test absolute `energy` 不参与拟合。由于同一结构的成员都加同一组成偏移，energy std/GMD、force/stress UQ 理论上不变。

### 2.3 可辨识性与字段语义

元素组成矩阵秩不足时使用 SVD 最小范数解，不把单元素修正量解释为唯一物理孤立原子能量。记录行数、列数、rank、奇异值、条件数和校准残差；test 出现 val 未覆盖元素或不可识别组合时 fail closed，不默认填零。

EXTXYZ 的 `energy` 与 `atomization_energy` 必须显式分开读取，不能用未区分语义的 `get_potential_energy()` 同时代表两者。

## 3. 架构与数据流

建议新增职责模块：

```text
Uncertainty_Quantification/BootStrapping/bootstrap/
  energy_correction.py
  correction_campaign.py
  correction_publication.py
```

`energy_correction.py` 负责字段提取、逐结构 offset、逐元素 ΔE0 和 energy 后处理；`correction_campaign.py` 负责 run、MAD val/test 路径、raw artifact、member 顺序和方法配置；`correction_publication.py` 负责派生 prediction/UQ/plot manifest、hash 审计、staging、manifest-last 和原子发布。若 val prediction 缺失，可用相同 checkpoint、member state、raw mode 和 prediction loader 补做，不重新训练。

## 4. 派生结果与存储

raw target 不覆盖。两种 correction 各自产生全电子 target：

```text
targets.energy = MAD absolute energy
targets.forces = 原始 MAD forces
targets.stress = 仅 reference 完整时保留
```

corrected member 保持 raw 的结构顺序、member 编号、shape、dtype、单位和 structure IDs；只替换 energy，force/stress 原样保留。建议输出：

```text
outputs/energy_corrections/
  structural_reference_swap__<identity>/
    full_remote_b8_e8/  lr_1e-4/  lr_1e-6/
    correction_manifest.json  uq_manifest.json  plot_manifest.json
  mean_model_aware_e0__<identity>/
    full_remote_b8_e8/  lr_1e-4/  lr_1e-6/
    calibration/<run>_delta_e0.json
    correction_manifest.json  uq_manifest.json  plot_manifest.json
```

## 5. UQ 与绘图

每个 run/method 重新计算 energy mean、sample std（`ddof=1`）、GMD，以及原有 force/stress UQ（reference stress 完整时）。必须验证 corrected energy std/GMD、force UQ 和 stress UQ 与 raw 仅有浮点舍入差异。

绘制 `raw_atomization_baseline`、`structural_reference_swap`、`mean_model_aware_e0`。raw 使用 atomization target，两种 correction 使用 absolute target；energy 以 eV/atom，force 使用原始 reference；MAD 不生成 stress residual 图。相同物理量的所有 run/method 使用统一坐标尺度、随机种子、等高线参数和图尺寸。方法一的 energy residual 必须与 raw atomization baseline 数值相同。

## 6. Provenance

manifest 记录 correction method、run、raw prediction/UQ manifest hash、val/test hash、reference 字段、raw/corrected energy definition、member 顺序、offset/ΔE0 hash、SVD 诊断、输出 size/hash 和 PASS/FAIL。

方法一记录 `uses_test_reference=true`、`calibration_split=none`、`oracle_postprocessing=true`；方法二记录 `uses_test_reference_for_calibration=false`、`calibration_split=val`、`calibration_prediction=ensemble_mean_of_8_members`。

## 7. 远端执行与一致性检查

所有后处理和测试在远端执行，本地不复制大体积 prediction/UQ/checkpoint/EXTXYZ。配置显式提供 val/test 路径，执行前检查构型数、原子总数、每构型原子数、structure/configuration IDs、元素顺序、坐标和周期性与 raw prediction 对齐。不得按数据集名称自动替换 full/filtered 数据；不匹配则停止并报告。记录输入 hash 和布局摘要，但不把 semantic fingerprint 作为执行锁。

执行顺序：preflight；方法一 test correction/UQ/plot；三个 run 分别进行方法二 val calibration；方法二 test correction/UQ/plot；全部通过后发布 manifest；仅同步 PNG、PDF、CSV、JSON 和 manifest。

## 8. 验证与失败恢复

单元测试覆盖字段分离、offset 符号、元素矩阵、SVD 解、rank 诊断、未覆盖元素 fail closed、UQ 不变性和 raw 不覆盖。远端 CPU 小数据验证必须确认方法一 residual 与 raw baseline 相等、方法二不读取 test absolute energy。全量验收要求 raw hash 不变、六组 corrected publication 完整、方法一无 val calibration、方法二每 run 恰好一组 ΔE0、UQ 不变、坐标尺度一致。

reference 缺失、结构错位、元素未覆盖、非有限数组、UQ 不变性失败或 identity/hash 冲突均 fail closed；不删除或覆盖 raw，只重跑对应 method。

## 9. 非目标

- 不训练/微调模型，不修改 checkpoint；
- 不覆盖 raw prediction、raw UQ、raw targets；
- 不把方法一解释为独立可部署校准；
- 不对 8 个成员分别拟合论文 E0；
- 不使用 test absolute energy 拟合方法二；
- 不生成 MAD stress residual 图；
- 不在本地复制大规模数据或模型；
- 不把修正能量用于力的重新求导或 MD。
