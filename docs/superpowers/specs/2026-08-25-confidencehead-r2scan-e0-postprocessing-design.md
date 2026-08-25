# ConfidenceHead MAD r2SCAN E0 后处理实验设计

## 1. 背景与目标

当前 UPET checkpoint 使用 MATPES/OMat 对应的逐元素 atomic reference energies（E0），新的 MAD-1.5 validation/test 数据则由 r2SCAN + FHI-aims all-electron 工作流生成。两套 E0 的绝对能量基准不同，直接比较模型总能量与 MAD r2SCAN 总能量会把组成相关的 E0 偏移计入模型误差。

本实验只在模型推理完成后修正 UPET 基础能量，不重新训练或修改 UPET，也不重新训练 ConfidenceHead。ConfidenceHead 的输入特征、logits、分箱 representatives 和 expected error 保持不变；E0 后处理后只重新计算作为 ConfidenceHead 评估 reference 的 observed energy error 及其分箱标签。

实验同时比较三种 test 视图：

1. `uncorrected`：不做 E0 修正的原始基线。
2. `direct_mad_e0_test_informed`：从 test 标签恢复 MAD r2SCAN E0，并直接替换模型 E0。
3. `model_aware_reestimated_val_calibrated`：严格按 Tompa 等人论文第 2.6 节的 model-aware reestimation，在 validation 上拟合逐元素 E0 correction，再固定应用到 test。

方法一使用 test 的 `energy` 和 `atomization_energy`，属于 test-informed/oracle 实验；方法二是 val-only calibration 和 held-out test evaluation。二者必须明确标注，不能解释为完全公平的盲测比较。

## 2. 已验证的数据条件

基础 checkpoint 支持原子序数 1–83 和 89–94，共 89 种元素。val 和 test 都按照这一集合整条过滤不支持结构，不保留结构中的部分原子。

- r2SCAN val 原始结构 18,305 条，保留 16,098 条，剔除 2,207 条，保留 310,432 个原子。
- r2SCAN test 原始结构 18,314 条，保留 16,072 条，剔除 2,242 条，保留 311,657 个原子。
- val 与 test 均覆盖全部 89 种受支持元素，test 没有 val 未覆盖元素。
- val 组成矩阵秩为 89，条件数约为 130.7。
- 使用 val 的 `energy - atomization_energy` 恢复 E0 后，在 test 重构该组成基线的 RMSE 约为 `5.5e-9 eV`，说明该属性由一套跨 split 一致的逐元素 E0 生成。

## 3. 能量语义

为避免“reference”同时指 DFT 标签和 ConfidenceHead 监督目标，所有新 artifact 使用以下字段名：

- `model_energy_raw`：UPET 原始总能量推理结果。
- `target_energy_r2scan`：MAD r2SCAN DFT 总能量，始终原样保留。
- `model_energy_corrected`：E0 后处理后的 UPET 总能量。
- `energy_observed_errors`：修正后逐结构、逐原子的绝对能量误差。
- `energy_labels`：使用原 ConfidenceHead energy thresholds 对修正后 observed error 重新分箱的标签。

对结构 i，令 `A[i, Z] = n_iZ` 为元素计数，`N_i` 为总原子数。所有视图的 observed error 均为：

```text
abs(model_energy_variant[i] - target_energy_r2scan[i]) / N_i
```

单位为 eV/atom；汇总指标以 meV/atom 报告。

## 4. 方法定义

### 4.1 原始基线

```text
E_uncorrected = E_raw
```

该视图用于量化未对齐 MATPES 与 MAD r2SCAN E0 时的结果。

### 4.2 方法一：test-informed direct MAD E0 replacement

MAD 文件同时提供总能量和 atomization energy。使用全部过滤后的 test 结构，以 float64 SVD 最小二乘恢复一套全局 89 元素 MAD E0：

```text
b_test = E_r2scan_test - E_atomization_test
E0_mad_test = pinv(A_test) @ b_test
```

从 checkpoint 的 PET composition additive model 读取原始 `E0_model`，然后计算：

```text
E_direct = E_raw - A_test @ E0_model + A_test @ E0_mad_test
```

执行时必须验证组成矩阵满秩，并记录重构残差、奇异值和条件数。该方法使用 test 标签恢复 E0，因此所有目录、manifest、图标题和汇总表必须带 `test_informed` 标识。

### 4.3 方法二：val-calibrated model-aware reestimation

严格复现论文 Eq. 6 的实验含义：使用全部过滤后的 val 结构，对总能量残差做不加权最小二乘，并采用 SVD 最小范数解。定义：

```text
r_val = E_r2scan_val - E_raw_val
delta_E0 = pinv(A_val) @ r_val
E0_reestimated = E0_model + delta_E0
```

固定该 correction 并应用到 test：

```text
E_reestimated_test = E_raw_test + A_test @ delta_E0
```

不使用逐原子加权，不加入 ridge 超参数，不使用 test 选择或调整 correction。manifest 同时绑定 val 与 test 数据 SHA256。

## 5. 与 ConfidenceHead 的关系

八个 energy ConfidenceHead 使用同一个 UPET checkpoint 和基础 energy prediction，但使用不同的 energy cumulant order。E0 correction 只拟合一次并由 order 1–8 共用。

后处理前必须验证八个 energy run 的结构 ID、顺序、原子数、逐结构组成、`model_energy_raw` 和 `target_energy_r2scan` 一致。

每个 order 保留自己的 logits、representatives 和 expected error。后处理只替换 observed error 和 labels。因此同一 E0 variant 的 observed error 在八个 order 之间相同，而 expected error 和 UQ 指标可因 order 不同而不同。

force-only run、力预测、force logits、force expected/observed error 和力图均不修改，也不在每个 E0 variant 中重复发布。

## 6. 模块边界与执行流程

新增独立、配置驱动的 E0 后处理模块，复用现有 external prediction、artifact 校验、原子发布和 density plotting 模块。训练、cache 特征提取和模型 forward 代码不修改。

建议组件边界：

- E0 数值模块：构造组成矩阵、读取 checkpoint E0、执行两类 SVD 求解、应用 correction、计算诊断。
- 派生结果模块：验证 raw predictions，生成三个 immutable variant，并绑定源 artifact 和 calibration artifact。
- workflow/CLI：加载严格 YAML 配置，先完成全量预检，再原子发布 calibration 与 variants。
- plotting adapter：把 raw ConfidenceHead expected error 与 variant observed error 配对，调用现有连续密度图模块。

数据流：

```text
filtered val/test extxyz
        |
        +--> unchanged UPET + ConfidenceHead inference
        |          |
        |          +--> immutable raw prediction artifacts
        |
        +--> composition matrices and target energies
                   |
                   +--> direct test-informed MAD E0 calibration
                   +--> model-aware val calibration
                                  |
                                  v
                      three derived energy variants
                                  |
                                  v
                    metrics + density plots + manifests
```

若当前 r2SCAN test raw prediction 已完整且 identity/SHA256 一致，则直接复用；否则只补齐缺失的 raw inference。model-aware 方法需要新增 val raw inference，但不重新训练任何模型。

## 7. 产物组织

后处理 artifact 根目录：

```text
Uncertainty_Quantification/ConfidenceHead/outputs/e0_postprocessing/mad_r2scan/
  calibration/
    direct_mad_e0_test_informed.json
    model_aware_reestimated_val_calibrated.json
    element_e0_comparison.csv
    manifest.json
  variants/
    uncorrected/
    direct_mad_e0_test_informed/
    model_aware_reestimated_val_calibrated/
```

`element_e0_comparison.csv` 每个元素一行，至少包含原子序数、元素符号、checkpoint E0、test-informed MAD E0、model-aware delta E0、reestimated E0，以及 val/test 覆盖统计。

每个 variant 仅发布发生变化或审计所需的字段：

- raw/corrected/target energy。
- corrected observed error 和 labels。
- 结构 ID、原子数与组成。
- 原始 prediction artifact 的相对路径与 SHA256。
- calibration artifact 的相对路径与 SHA256。

ConfidenceHead logits 和 expected error 不复制，由派生 manifest 引用已验证的原始 artifact。

绘图根目录：

```text
Uncertainty_Quantification/Plots/ConfidenceHead/density_scatter_r2scan_e0/
  uncorrected/
  direct_mad_e0_test_informed/
  model_aware_reestimated_val_calibrated/
  comparisons/
```

每个 variant 生成 order 1–8 的连续散点密度 PNG/PDF、绘图 CSV/JSON 和 manifest。`comparisons/` 使用统一坐标范围发布三方法对照图和汇总表。不生成 boxplot，不复制力图，不覆盖现有 PBE 或 r2SCAN 绘图目录。

## 8. 指标与报告

基础能量指标按 variant 报告，与 energy order 无关：

- test MAE、RMSE、mean signed error 和 P95 absolute error，单位 meV/atom。

UQ 指标按 variant 和 order 报告：

- expected-vs-observed Pearson correlation。
- expected-vs-observed Spearman correlation。
- 现有 density plotting/calibration 统计。

校准诊断包括：

- direct 方法的 test E0 重构残差。
- model-aware 方法的 val 修正前后残差，以及固定参数后的 test 泛化结果。
- 组成矩阵秩、奇异值、条件数、元素覆盖和结构过滤计数。

报告必须同时包含数值表和连续密度图，不能只凭图判断方法优劣。方法一与方法二必须分栏注明 test-informed 和 val-calibrated，禁止给出忽略数据使用差异的单一“胜负”结论。

## 9. 失败处理与不可变性

工作流采用 fail-closed 语义：

- raw prediction manifest 不完整、artifact 缺失或 SHA256 不匹配时停止。
- checkpoint、过滤数据、结构顺序或组成身份不一致时停止。
- test 出现 val 未覆盖元素时，model-aware 方法停止。
- 组成矩阵 rank deficient 时停止并报告不可辨识元素；不静默接受退化解。
- order 1–8 的 raw energy/reference/structure identity 不一致时停止。
- E0、corrected energy 或 observed error 出现 NaN/Inf 时停止。
- 已有完整输出 identity 不同则拒绝覆盖；相同 identity 才允许验证并复用。

原始 r2SCAN DFT 数据、raw predictions、ConfidenceHead logits/expected error、力结果、训练配置、checkpoint、日志和历史图均不修改或删除。后处理前后必须验证 raw logits、representatives 和 expected error 的 SHA256 不变。

## 10. 测试与验收

测试覆盖：

1. 合成多元素数据上的 direct E0 精确恢复、model-aware correction、符号和单位。
2. SVD 最小范数行为、满秩诊断和 rank-deficient 拒绝。
3. 逐原子误差与原 thresholds 下 labels 的重算。
4. raw prediction、logits、expected error 和 force artifacts 不变的回归测试。
5. 配置、manifest identity、SHA256、冲突拒绝和原子发布测试。
6. 小数据 CPU 端到端测试，覆盖 calibration、variants、指标和绘图。
7. 远端全量 val/test 执行和 artifact 审计。

全量验收要求：

- val/test 过滤计数分别为 16,098/16,072，且均覆盖 89 种元素。
- direct 和 model-aware calibration manifest 完整并通过 SHA256 校验。
- 三个 variant 与八个 energy order 全部发布成功。
- 每个 order 的 PNG/PDF/CSV/JSON 数量与 manifest 一致。
- 所有图可解码、PDF 非空、无 staging 目录残留。
- uncorrected 派生视图与原始 observed error 数值一致。
- ConfidenceHead expected error 和全部力结果在三个 variant 间保持不变。

## 11. 非目标

- 不修改或重新训练 UPET。
- 不修改或重新训练 ConfidenceHead。
- 不重新拟合 ConfidenceHead expected error、温度参数或 bin representatives。
- 不引入逐原子加权 model-aware variant、ridge regularisation 或 E0 数据量 sweep。
- 不修改 force UQ。
- 不删除或覆盖现有 PBE、r2SCAN、boxplot 或 density plot 结果。
