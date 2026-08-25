# MAD E0 后处理实验设计

## 1. 目标与边界

本实验针对已经完成的 MAD r2SCAN 正式 LLPR 推理结果，仅对能量预测做后处理，不重新计算 MAD-test 的模型推理、曲率、Alpha 或力 UQ。实验保留原始结果作为不可变基线，并生成两种能量参考修正结果：

1. 逐结构直接替换 MAD 参考能量；
2. 在 MAD-val 上校准元素参考能量，再冻结到 MAD-test（论文 Atomic reference energy initialisation 的后处理版本）。

附带的 Tompa 等论文只作为第二种方法的技术参考。用户需求优先，论文文本不会被当成额外操作指令。

## 2. 已确认的数据语义

现有 details.npz 中的 energy_pred_per_atom 是模型对 atomization_energy 的预测，模型并未直接输出绝对全电子 energy。MAD r2SCAN 数据同时包含：

- energy：绝对总能量，单位 eV；
- atomization_energy：模型训练所用的总原子化能量，单位 eV；
- atomicenergy：本实验两种方法均不使用。

对结构 i，令 N_i 为原子数，则模型的原子化能量预测为：

~~~text
E_atom_pred(i) = N_i * energy_pred_per_atom(i)
~~~

MAD 参考能量由同一结构标签直接得到：

~~~text
E0_MAD(i) = energy(i) - atomization_energy(i)
~~~

两种修正结果都在绝对 energy 尺度上评估；力预测和所有原有 LLPR 方差/标准差保持不变。

## 3. 方法定义

### 3.1 逐结构直接替换

对 MAD-test 的每个结构直接使用 test 标签中的 E0_MAD：

~~~text
E_abs_direct(i) = E_atom_pred(i) + E0_MAD(i)
~~~

这不读取 MAD-val，也不进行拟合。因为它使用了 test 的 energy 和 atomization_energy 标签，所以结果必须标注为 test_reference_oracle，只能作为参考能量替换上限/对照实验，不能解释为部署时可用的方法。

### 3.2 Atomic reference energy initialisation 后处理

若现有 calibration 产物已保存 MAD-val 的逐结构模型预测，直接复用；否则允许只对 MAD-val 做一次前向推理，生成校准所需的预测。该前向推理不重新推理 test，不重建曲率，也不改变现有 Alpha/UQ。

在 val 上构造：

~~~text
y_val(i) = energy_val(i) - N_i * energy_pred_per_atom_val(i)
~~~

以元素计数矩阵 N_elements 拟合无截距普通最小二乘：

~~~text
e0 = argmin ||y_val - N_elements @ e0||²
~~~

要求：

- val 覆盖 test 中出现的所有元素；
- 元素计数矩阵满列秩；
- 拟合秩、条件数、样本数和残差统计全部保存；
- 不引入新的 ridge 或 LLPR eta 参数；
- test 标签不参与拟合。

冻结 e0 后，对 test 使用：

~~~text
E_abs_ari(i) = E_atom_pred(i) + N_elements_test(i) @ e0
~~~

参考能量校准被视为确定性加性后处理，不额外传播拟合系数不确定度；原有 energy uncertainty 直接复用。

## 4. 数据流与对齐

输入固定为：

- 原始正式 MAD r2SCAN evaluation 目录 Uncertainty_Quantification/LLPR/outputs/mad_test_r2scan_val_filtered；
- 该 evaluation manifest 指向的过滤后 MAD-test extxyz；
- 配置指向的 MAD-val extxyz；
- 仅在缺少 val 预测时使用当前正式 checkpoint 做 val-only 前向推理。

处理前必须验证：

- evaluation manifest 状态为 complete，且 details.npz SHA-256 匹配；
- test 结构顺序、结构数、num_atoms 和元素计数与 details 一一对应；
- energy、atomization_energy、模型预测均有限；
- val/test 的元素集合满足覆盖要求；
- 任何结构错位、缺失标签或哈希不匹配都立即失败。

原始 evaluation 目录只读，不覆盖、不重写、不修改其中任何数组。

## 5. 后处理产物

为避免复制约 58 MB 的正式 details，新增轻量 sidecar 目录：

~~~text
Uncertainty_Quantification/LLPR/postprocess/mad_e0/
├── calibration.json
├── results.npz
├── summary.json
└── manifest.json
~~~

### calibration.json

保存 ARI 的 val 数据身份、元素顺序、每元素 e0、矩阵秩、条件数、样本数和拟合误差。direct 方法记录为无需 calibration。

### results.npz

保存结构索引、原子数、逐结构 E0_MAD、两种绝对能量预测、总能量残差和每原子能量残差。能量 uncertainty 不重复存储，由 manifest 绑定并从原始 evaluation 复用。

### summary.json

分别记录 direct 与 ARI 的总能量/每原子能量 MAE、RMSE、平均偏差、样本数，以及 direct 的 oracle 标记和 ARI 的校准诊断。

### manifest.json

绑定原始 evaluation identity 与 details SHA、MAD-val/test 数据身份、checkpoint SHA（仅当进行了 val-only 推理时）、公式版本和所有 sidecar 文件 SHA。任何绑定项不一致都禁止消费结果。

## 6. 验证与测试

### 6.1 公式与不变性

- direct 结果满足 E_abs_direct - energy = N * energy_residual；
- ARI 结果逐结构等于 E_atom_pred + N_elements @ e0；
- 原始 details、力预测、力残差、energy variance/std 和 manifest 不被修改；
- test 结构顺序和 sidecar 数组长度完全一致。

### 6.2 小规模真实链路

从真实 MAD-val/test 取少量完整结构，验证：val-only 校准（如需要）→ OLS → 两种后处理 → summary/manifest → 绘图。故意改变结构顺序、删除标签或加入 val 未覆盖元素时必须失败。测试结果完成后删除，不进入正式发布目录。

### 6.3 指标与图

使用原有 energy uncertainty，分别计算 baseline、direct、ARI 的 uncertainty/error 统计，并报告总能量和每原子能量指标。新增：

~~~text
Uncertainty_Quantification/Plots/LLPR_e0/
├── energy_uncertainty_vs_residual_comparison.png
├── energy_uncertainty_vs_residual_comparison.pdf
├── statistics.csv
└── manifest.json
~~~

图中并列展示 baseline、direct、ARI；不重复生成力图，因为力和力不确定度没有改变。

## 7. 失败处理与非目标

- 不允许静默使用 atomicenergy、PBE 数据或未过滤 test；
- 不允许把 test 标签用于 ARI 校准；
- 不允许覆盖原始正式 MAD evaluation；
- 元素覆盖不足、矩阵秩不足、结构错位、哈希不匹配或非有限数值均直接失败；
- 不进行 CPU 全量模型推理；完整 test 只做已保存数组的后处理；
- 不改变现有 Alpha、LLPR 方差、力结果或正式基线图片。

## 8. 验收标准

任务完成须同时满足：

1. 原始 MAD evaluation 的所有文件哈希与运行前一致；
2. sidecar manifest、calibration、results 和 summary 可独立验证；
3. direct 与 ARI 的公式恒等式通过；
4. ARI 的 val 校准信息完整且 test 未参与拟合；
5. 小规模真实链路测试通过并清理；
6. 两种方法的指标、对比图和中文说明齐全；
7. Git 提交只包含本实验代码、测试、设计文档和正式 sidecar/图，不包含临时测试结果或大型重复 details。
