# MAD R2SCAN 的 FGE E0 后处理实验设计

**日期：** 2026-08-25
**状态：** 已完成设计评审，待 implementation plan
**范围：** 仅对既有 FGE 推理能量进行后处理；不重新训练、不改变模型推理、不修改原始 prediction

## 1. 背景与目标

当前 PET-OMATPES FGE 模型使用的 MATPES 原子参考能量（E0）与 MAD R2SCAN
数据中的能量零点不同。因此，模型输出的总能量不能直接和 MAD R2SCAN 的原始总能量比较，
但模型推理本身仍然有效。本实验保持 K=8 成员的推理结果不变，在推理完成后比较两种 E0
修正方法：

1. **严格 E0 替换（direct E0 replacement）**：移除 MATPES E0，再加入由 MAD R2SCAN
   test 数据直接恢复的 MAD E0；
2. **模型感知原子参考能量重估（model-aware reestimation）**：按照 Tompa 等（2026）
   Section 2.6 的思路，在 MAD R2SCAN val 上拟合模型预测与目标参考能量之间的逐元素偏移，
   再原样应用到 test。

两种方法均只修正 Energy。Force、Energy ensemble spread 和原始推理产物保持不变。

## 2. 已确认的设计决策

- 采用轻量 corrected-energy sidecar 架构，不复制、重写或覆盖 prediction chunk。
- baseline、direct E0 和 model-aware E0 都纳入同一次 test 评估。
- direct E0 只使用与现有推理对应的 MAD R2SCAN test 中的 raw energy 与
  atomization energy 恢复目标 E0；val 不参与方法 A。
- model-aware E0 只使用 MAD R2SCAN val 的标签以及 K=8 ensemble mean 拟合。
- 同一套 model-aware 修正应用到全部 8 个成员；禁止逐成员独立拟合。
- 方法 B 的校准产物必须在读取 test 标签前封存；test 标签不参与方法 B 的拟合、参数选择
  或返工。方法 A 按定义使用 test 的 raw/atomization energy 提取该 test 的 E0。
- 能量计算使用 float64；prediction 中既有 float32 预测只在读取后转换为 float64，
  prediction chunk 内的 float32 reference 不作为本实验的能量真值。
- 三种 Energy 共用原始 K=8 population standard deviation 作为不确定性；Force 只输出一份。
- 不生成 checkpoint sweep、相关性曲线或方法选择曲线，也不以 test 表现决定是否接受某种方法。

## 3. 非目标

本实验不训练、不微调、不执行 backward 或 optimizer，不修改 checkpoint，不改变 CompositionModel，
也不重新运行已经完成或正在运行的 test prediction。它不尝试把 MATPES 与 MAD 的物理能量定义
解释为同一数据生成流程，只比较两种明确的逐元素能量零点后处理。

## 4. 输入与身份边界

### 4.1 模型与成员

实验沿用当前已经完成的 PET-OMATPES K=8 FGE 成员：

- checkpoint SHA-256：
  `879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004`
- result manifest SHA-256：
  `b1f4a3c5c7713b4be369b467e98e62d59bc23a13aefceffbd49ac51773b6dc86`
- 支持元素：89 种，即 Z=1..83 与 Z=89..94。

后处理开始前必须从 checkpoint 的 `CompositionModel` 提取按固定元素顺序排列的
`E0_MATPES`，并逐成员验证其数值和元素映射完全一致。如任一成员的 CompositionModel
不同，实验立即失败，不得对各成员分别补偿。

### 4.2 MAD R2SCAN test

- 原始数据 SHA-256：
  `a2cfc12d3a7066f114a788621c55d79a0b6726b29e7d64947a2c40515dfba74b`
- 模型支持元素过滤后的数据 SHA-256：
  `499b479499eb56d0792360cb8bcb3397b566ac99c4e290ce0e866382c7f4d2ed`
- 过滤后规模：16,072 structures、311,657 atoms、129 prediction chunks。
- 方法 A 的 E0 提取范围严格限定为这 16,072 条过滤后 test 结构；不使用被元素过滤排除的结构。
- 每条结构必须同时具有 float64 `energy` 与 `atomization_energy`，并与 prediction 结构 ID
  和顺序一一对应。
- 正式 prediction 根：`outputs/inference_mad_r2scan_test_k8/`。

若该 prediction 正在运行，只允许只读监控；不得重启或并发启动。只有完整
`prediction/manifest.json` 通过既有验证后，才允许进入 test 后处理。

### 4.3 MAD R2SCAN val

- 数据：`data/dataset/val/co/co_0.extxyz`
- 原始数据 SHA-256：
  `44de0b84e5427a87b86c495ec523e7b4232018b78c4330bae524ffdcc5c6fabe`
- metadata SHA-256：
  `39c3cfabb7d8dcdcb5a45a611c6f9c50020e8e529a8414b81269e5cc3afe6faa`
- 预期过滤结果：16,098 structures、310,432 atoms，排除 2,207 structures。
- 覆盖全部 89 个模型支持元素；组成矩阵预审计 rank=89、condition number 约 130.7。

val 必须复用 test 的元素支持过滤、结构 ID 和排序规则，正式根为
`outputs/inference_mad_r2scan_val_k8/`。若完整且身份匹配的 val prediction 已存在则只读复用；
否则只运行一次 K=8 predict，不计算任何新的模型或 checkpoint。

## 5. 数学定义

令 `X` 为组成矩阵；第 i 行、第 z 列是结构 i 中元素 z 的原子数。令
`E_original^(k)` 为成员 k 的既有模型总能量，`E_raw_MAD` 为 extxyz 中以 float64
重新读取的 MAD R2SCAN raw energy，`E_atomization_MAD` 为同一结构的 atomization energy。

### 5.1 方法 A：严格 E0 替换

仅在过滤后的 test 上求 MAD 原子参考能量：

\[
X_{test} E0_{MAD,test} = E_{raw,test}^{MAD} - E_{atomization,test}^{MAD}.
\]

求解使用 float64、固定的 89 元素列顺序、无截距、无正则的 SVD 最小范数解。虽然该数据
中的 raw/atomization 差应由逐元素参考能构成，仍必须保存 singular values、rank、
condition number、RMSE 和最大绝对重构误差，禁止静默接受欠秩或非有限结果。方法 A
不读取 val，不在 val 上求解、验证或调整 `E0_MAD_test`。

只对 test 的每个成员应用：

\[
E_{direct,test}^{(k)} = E_{original,test}^{(k)}
- X_{test} E0_{MATPES} + X_{test} E0_{MAD,test}.
\]

除常规数值检查外，必须逐结构验证以下恒等式：

\[
E_{direct} - E_{raw}^{MAD}
= (E_{original} - X E0_{MATPES}) - E_{atomization}^{MAD}.
\]

该方法回答“如果保持模型学到的非组成能量不变，只把 MATPES 零点替换为 MAD R2SCAN
零点，会得到什么结果”。

### 5.2 方法 B：model-aware E0 重估

先在 val 上计算 8 个未修正预测的 ensemble mean：

\[
\bar E_{original,val} = \frac{1}{8}\sum_{k=1}^{8} E_{original,val}^{(k)}.
\]

随后拟合一套公共逐元素修正：

\[
\delta = \mathop{argmin}_{\delta}
\left\|X_{val}\delta -
\left(E_{raw,val}^{MAD} - \bar E_{original,val}\right)\right\|_2^2.
\]

求解同样使用 float64、总能量、无截距、无正则的 SVD 最小范数解。有效原子参考能量可记为
`E0_model-aware = E0_MATPES + delta`，但实现只需对既有总能量加组成修正：

\[
E_{model-aware}^{(k)} = E_{original}^{(k)} + X\delta.
\]

同一个 `delta` 必须应用到全部成员。禁止为每个成员独立拟合，因为那会改变并人为缩小
ensemble spread，使 UQ 与能量修正耦合。

### 5.3 baseline、残差与 UQ

baseline 是不做任何零点修正的 `E_original`。对方法 m 的结构 i，Energy 图使用：

\[
r_{i,m} = 1000\,\frac{\left|\bar E_{i,m} - E_{raw,i}^{MAD}\right|}{N_i}
\quad [\mathrm{meV/atom}],
\]

\[
u_i = 1000\,\frac{\operatorname{std}_{population}
\left(E_{i,original}^{(1)},\ldots,E_{i,original}^{(8)}\right)}{N_i}
\quad [\mathrm{meV/atom}].
\]

由于两种修正都是对全部成员施加同一个结构相关常数，baseline、direct E0 和 model-aware E0
理论上具有完全相同的 `u_i`。实现必须数值验证这一不变量，而不是重新定义 UQ。

Force 不受任何 E0 修正影响，继续复用现有 FGE 的逐原子残差、不确定性、统计与绘图语义。

### 5.4 固定数值容差

所有容差在读取方法 A 所需的 test 标签和计算任何 test 指标前固定：

- SVD rank threshold 为 `max(m, n) * eps_float64 * largest_singular_value`；
- 89 元素 E0 在不同成员间要求元素键完全相同，数值 `rtol=0, atol=1e-12 eV`；
- MAD E0 的 test 重构要求 RMSE 不超过 `1e-6 eV/structure`、最大绝对误差不超过
  `5e-6 eV/structure`；
- direct 恒等式的最大绝对误差不超过 `1e-6 eV/structure`；
- 公共 shift 沿 8 个成员的最大差异不超过 `1e-9 eV/structure`；
- 从修正后成员能量重新核验的 population STD 与原始 STD 要求
  `rtol=1e-10, atol=1e-9 eV`。

model-aware 的 val/test 误差不设置性能阈值；只要求满秩、有限、可复现和身份正确。

## 6. 组件与复用边界

新增逻辑应保持小而独立，不改变通用 prediction schema：

- `fge/energy_correction.py`：纯 float64 E0 提取、model-aware 校准、应用、恒等式验证和
  诊断数据结构；
- `scripts/correct_energy.py`：分成 `calibrate-model-aware`、`extract-direct-e0` 与 `apply`
  三个显式阶段的薄 CLI；
- 一份 MAD R2SCAN val inference 配置、一份 energy correction 配置和一份对应 plot 配置；
- 针对数值语义、身份绑定、无泄漏和只读保证的单元/集成测试。

优先复用现有模块：

- `fge/inference_data.py` 的元素过滤、结构身份与数据 SHA 规则；
- `fge/prediction.py`、`fge/inference_validation.py` 的 prediction/manifest 读取与完整性验证；
- `fge/manifests.py` 与 `fge/artifacts.py` 的 manifest 和 artifact 身份记录方式；
- `fge/uncertainty.py` 的 K=8 population STD 语义；
- `fge/plot_analysis.py`、`fge/plot_rendering.py` 与 `scripts/plot_dataset.py` 的统计和绘图基础设施。

若既有 reader 将 extxyz reference 转成 float32，本功能不得修改其全局语义；应增加一个仅供 E0
后处理使用的 float64 reference 读取边界。这样不会影响已完成结果或旧实验的数值契约。

## 7. Sidecar 与 manifest 设计

正式根目录：

```text
Uncertainty_Quantification/FGE/outputs/
├── inference_mad_r2scan_test_k8/       # 既有、只读
├── inference_mad_r2scan_val_k8/        # 复用或只运行一次 predict
└── mad_r2scan_energy_correction_k8/
    ├── model_aware_calibration/
    │   ├── model_aware_e0.json
    │   └── calibration_manifest.json
    ├── direct_test_reference/
    │   ├── direct_e0_from_test.json
    │   └── direct_reference_manifest.json
    ├── reference_float64/
    │   ├── val/
    │   │   ├── chunk_*.npz
    │   │   └── manifest.json
    │   └── test/
    │       ├── chunk_*.npz
    │       └── manifest.json
    ├── baseline/
    │   ├── chunk_*.npz
    │   └── manifest.json
    ├── direct_e0/
    │   ├── chunk_*.npz
    │   └── manifest.json
    ├── model_aware_e0/
    │   ├── chunk_*.npz
    │   └── manifest.json
    ├── shared_force/
    │   └── manifest.json
    └── result_manifest.json
```

每个 Energy chunk 至少绑定 `structure_id`、原 prediction chunk identity、`natoms`、组成向量、
8 个 float64 member energies、ensemble mean、共享 energy UQ 和 absolute residual per atom。
`reference_float64/val` 绑定从 immutable extxyz 重读的 raw energy、组成、结构顺序和数据
SHA；`reference_float64/test` 额外绑定方法 A 所需的 atomization energy。
`shared_force/manifest.json` 只指向并验证原 prediction 中唯一一套 Force 数据，
不为三种 Energy 方法复制三遍 Force。

`calibration_manifest.json` 只描述方法 B，必须记录：

- val 原始/过滤数据 SHA、metadata SHA、结构数、原子数和结构 ID 摘要；
- val prediction manifest 与全部 chunk identities；
- checkpoint、result manifest 和 8 个成员身份；
- 固定元素顺序、MATPES E0、model-aware delta；
- SVD 实现、dtype、rank、singular values、condition number、残差诊断；
- 校准代码版本、配置 SHA、产物 SHA、大小和 mtime。

`direct_reference_manifest.json` 只描述方法 A，必须记录 test 原始/过滤数据 SHA、结构数、
原子数、结构 ID/chunk 摘要、固定元素顺序、`E0_MAD_test`、test 组成矩阵的 SVD 诊断、
raw/atomization 重构误差以及全部产物身份。它不得声明为 val 校准或 held-out calibration。

`result_manifest.json` 还必须绑定 test 原始/过滤数据 SHA、test prediction manifest、129 个
chunk identities、封存后的 model-aware calibration manifest SHA、direct test reference
manifest SHA，以及三种 Energy 和共享 Force 的所有产物身份。任何身份不匹配都应硬失败，
不能按数组长度或当前位置猜测对应关系。

## 8. 方法隔离与执行顺序

1. 验证 K=8 成员、checkpoint 与 CompositionModel 一致性。
2. 复现 val 支持元素过滤并重读 float64 raw reference 与组成。
3. 验证或完成一次 val predict；不触碰 test 标签。
4. 只在 val 上求公共 `delta`，生成并封存方法 B 的 calibration manifest。
5. 重新读取 calibration manifest 并验证其 SHA；此后方法 B 的校准参数不可修改。
6. 只读验证完整 test prediction，按结构 ID/chunk identity 关联 test 的 float64 reference。
7. 只在过滤后的 test 上由 `energy - atomization_energy` 求 `E0_MAD_test`，封存 direct test
   reference manifest；不读取 val，也不利用模型残差拟合方法 A。
8. 同时生成 baseline、direct E0、model-aware E0 sidecar 和一份共享 Force manifest。
9. 计算 test 指标并绘图；无论结果优劣，都不得返回第 4 步重调方法 B。
10. 对原 prediction 树再次做 SHA/size/mtime 审计，证明前后未改变。

`calibrate-model-aware` 不得接受 test 路径；`extract-direct-e0` 必须绑定 test split 与本设计的
test SHA，且不得读取 val 或模型预测；`apply` 只接受 SHA 明确的两个封存产物。这样既满足
方法 A 直接使用 test 的要求，又保证方法 B 仍是严格的 val-to-test held-out 实验。

## 9. 指标与绘图

Energy 对每种方法分别报告 test 的 meV/atom：MAE、RMSE、median absolute error、P95
absolute error 和 max absolute error；同时报告相对 baseline 的改善比例、每种修正相对
baseline 的逐结构胜率，以及 direct 与 model-aware 的逐结构 head-to-head 胜率。

所有表格、图标题和统计 manifest 必须把 direct 标为 `test-derived direct E0`，把 model-aware
标为 `val-calibrated model-aware E0`。二者可以并列展示，但 direct 使用了同一 test 的
raw/atomization 标签提取零点，是 test 内的能量口径对齐；model-aware 才是 val-to-test
held-out 结果。不得把 direct 的指标描述为独立测试集泛化误差。

UQ 诊断只计算数值统计，不新增相关性曲线：

- uncertainty 与 absolute residual 的 Spearman correlation；
- 仅在 uncertainty 和 residual 都为正的点上计算 log10 Pearson correlation；
- 有效点数、被排除的零值/非有限点数。

最终图目录：

```text
Uncertainty_Quantification/Plots/FGE/mad_r2scan_test_energy_correction/
├── energy_baseline_uncertainty_vs_absolute_residual.png
├── energy_baseline_uncertainty_vs_absolute_residual.pdf
├── energy_direct_e0_uncertainty_vs_absolute_residual.png
├── energy_direct_e0_uncertainty_vs_absolute_residual.pdf
├── energy_model_aware_e0_uncertainty_vs_absolute_residual.png
├── energy_model_aware_e0_uncertainty_vs_absolute_residual.pdf
├── force_uncertainty_vs_absolute_residual.png
├── force_uncertainty_vs_absolute_residual.pdf
├── calibration_statistics.json
├── plot_statistics.json
└── plot_manifest.json
```

direct E0 与 model-aware E0 使用完全相同的 x/y 范围，便于公平比较。baseline 因零点不匹配
允许使用独立 y 范围，但标题和统计文件必须明确标记 `unmatched energy-zero baseline`，不可让其
视觉尺度暗示修正方法的细微差异。三张 Energy 图共享同一组原始 UQ；Force 图只生成一次。

`plot_manifest.json` 必须绑定 result manifest、各 sidecar manifest、绘图配置和每个图/统计文件
的 SHA-256。绘图后重新核对 source SHA，保证绘图过程未修改上游产物。

## 10. 失败策略

以下任一情况均为硬失败，保留现场且不删除任何文件：

- val 组成矩阵 rank 小于 89，或 model-aware 出现非有限 SVD/校准结果；
- test 组成矩阵 rank 小于 89，或 direct E0 出现非有限 SVD/重构结果；
- test 出现 val 未覆盖的模型支持元素，或任一方法的 E0/元素映射不完整；
- 任一 FGE 成员修改了 CompositionModel，或成员/manifest 身份不一致；
- val/test prediction manifest 不完整，结构数、顺序、ID 或 chunk identity 不匹配；
- 原始/过滤数据 SHA、metadata SHA、封存后的 model-aware calibration SHA 或 direct test
  reference SHA 漂移；
- float64 reference 无法按结构一一关联，或所需 raw/atomization energy 缺失、重复、非有限；
- 公共修正沿 member 轴不恒定，Energy UQ 不变量不成立，或 Force identity 发生变化；
- 后处理路径触发 predict、training、backward 或 optimizer；
- 后处理前后原 prediction 的 SHA、size 或 mtime 发生变化。

失败后只写诊断或报告，不自动删除、覆盖、重跑 prediction，也不退化到 float32 reference。

## 11. 测试与验证策略

### 11.1 单元测试

- 用已知逐元素 E0 的满秩 test 合成数据验证 direct E0 精确恢复和元素顺序；
- 用已知公共 delta 的 K=8 合成预测验证 model-aware 最小二乘恢复；
- 验证欠秩、缺元素、重复结构 ID、非有限值和 member CompositionModel 漂移均硬失败；
- 验证公共组成 shift 对全部成员相同，原始 population STD 被复用且数值不变；
- 用约百万 eV 的 raw energy 构造精度回归测试，证明 float32 reference 会失败而 float64 通过；
- 验证 direct 恒等式和 test/val split 身份，并单独验证方法 B 的 test 元素被 val 覆盖；
- 验证 `calibrate-model-aware` 不接受 test 输入，`extract-direct-e0` 不接受 val 或 prediction
  输入，`apply` 不接受任一未封存产物。

### 11.2 集成测试

- 在小型 val/test fixture 上执行过滤、校准、应用、统计、绘图和 manifest 链；
- 验证重新排序、缺 chunk、错误 SHA 和相同长度但不同结构 ID 都会被拒绝；
- 运行两次只读验证，比较上游 prediction 的 SHA/size/mtime 快照完全不变；
- 为后处理安装 runtime guards，使模型加载推理、训练和梯度路径一旦调用就立即失败；
- 验证三张 Energy 图的 source UQ 相同，Force 只有一个 source identity。

### 11.3 仓库级验证

实现阶段完成后运行完整 FGE tests，并按仓库要求运行 Ruff format/check 与 mypy。测试和提交
只纳入本任务文件，不修改或暂存 LLPR、BootStrapping、ConfidenceHead、`.idea`、`.orig/.rej`
及其他无关工作区内容。

## 12. 验收标准

实验被视为完成需同时满足：

1. val/test 数据、K=8 模型和 prediction 身份全部与本设计绑定值一致；
2. val 过滤结果为 16,098 structures / 310,432 atoms，test 过滤结果为 16,072 structures /
   311,657 atoms；两个组成矩阵均 rank=89；
3. test-derived MAD E0 重构、direct 恒等式和 val model-aware SVD 诊断均在预先固定的
   float64 容差内；
4. 三种 Energy sidecar、共享 Force manifest、result manifest 与 4 组 PNG/PDF 全部完整；
5. 三种 Energy 使用同一原始 population STD，Force 数据未改变；
6. test 标签未进入 model-aware calibration；方法 A 仅按定义使用 test 的
   `energy - atomization_energy` 提取 E0，未使用模型残差拟合，也没有依据 test 结果重调
   方法 B 或任何超参数；
7. 原始 prediction 在后处理与绘图前后 SHA/size/mtime 不变；
8. 完整 FGE tests、Ruff 和 mypy 通过，验收报告记录命令与结果；
9. 最终提交只包含本任务文件，保留所有无关本地修改。

验收不要求 model-aware 必须优于 direct，也不要求两种修正达到预设误差阈值。它们的 test
表现是实验结果，而不是通过条件。

## 13. 结果解释边界

direct E0 尽量保持 checkpoint 的非组成能量，只更换数据集零点；它最接近“严格替换 E0”。
由于其 `E0_MAD_test` 直接从当前 test 标签提取，它是 test 内部的能量约定对齐结果，不是
held-out 校准结果，也不能用于证明对未知 MAD 数据的零点迁移能力。
model-aware E0 会把 val 上与组成线性相关的模型系统误差一并吸收到 delta，因此通常更贴合
目标数据，但其有效 E0 不应被解释为独立原子的物理能量。baseline 仅用于展示未匹配零点时
直接比较 raw energy 的失真程度。

本实验能比较两种后处理对 MAD R2SCAN test 能量残差的影响，但不能据此声称模型经过
R2SCAN 微调，也不能把 Energy 修正解释为 Force 或 UQ 得到了重新校准。

## 14. 参考文献

Tompa 等，*Fine-tuning MLIP foundation models: strategies for accuracy and transferability*，
2026，Section 2.6（Atomic reference energy initialisation）。本设计采用其中 model-aware
逐元素残差拟合思想，但保持当前 FGE K=8 推理和所有 Force 结果不变。
