# UPET BootStrapping 三组结果、三数据集推理与绘图设计

日期：2026-08-16
状态：设计已确认
范围：仅执行既有 BootStrapping checkpoint 的 prediction、UQ 与绘图；不训练、不迁移 checkpoint、不重新生成 bootstrap 成员。

## 1. 目标

复用远端已经完成的三组 8-member BootStrapping 结果：

- `full_remote_b8_e8`；
- `lr_1e-4`；
- `lr_1e-6`。

三组结果均处理以下数据集：

- `matpes_test`：复用远端已有 prediction 与 raw UQ，只绘图；
- `mad_test`：新增 raw prediction、UQ 与绘图；
- `matpes_train`：新增 raw prediction、UQ 与绘图。

正式数值工作和测试全部在远端执行。大体积 checkpoint、prediction、UQ 与数据集不复制到本地，也不进入 Git。本地只开发可发布代码，并在远端绘图完成且审核通过后同步体积较小的 PNG、PDF、CSV、JSON 和 manifest。

## 2. 已确认的科学口径

- 三组 run 全部处理，不选择单一主 run。
- 只使用 checkpoint 的 `raw` 参数分支，不执行 EMA 推理。
- 每组使用完整 8-member ensemble，样本标准差采用 `ddof=1`。
- 不生成成员数扫描图，也不实现 `K=2,4,8,...` 前缀分析。
- `matpes_test` 和 `matpes_train` 绘制 energy、force、stress。
- `mad_test` 仅绘制 energy、force；其数据没有 reference stress，不允许用零值、预测值或其他占位数据伪造标签。
- 图形采用 Carnet BootStrapping 的不确定度—绝对残差定义、对数密度等高线和相关性统计。
- 同一物理量的所有 panel 使用完全相同的 x/y 坐标范围：9 个 energy panel 共用一组范围，9 个 force panel 共用一组范围，6 个 stress panel 共用一组范围。
- 原始数据集由 campaign 配置中的路径决定，不使用语义数据集指纹控制执行或复用。

## 3. 选定方案

采用独立的多 run、多数据集 campaign 配置，并在 BootStrapping 内扩展通用 prediction/UQ 数据集接口和专用绘图模块。

未采用以下方案：

- 不把每个新数据集伪装成独立 `test` run，因为这会制造含义错误的目录并削弱 provenance。
- 不把 BootStrapping 结果转换为 LLPR evaluation 后调用 LLPR plotter，因为两种方法的 schema 不同，且 LLPR 当前没有 stress panel。
- 正式代码不导入或依赖 `carnet_new`；Carnet 只作为科学定义、渲染规则和发布方式的参考。

## 4. 总体架构

一个严格校验的 campaign 配置同时描述：

- 三组既有 run 的正式配置和远端 run root；
- 三个数据集的显示标签、artifact 存储键、输入路径和 reference target 集合；
- prediction 的 device、batch size 和参数模式；
- UQ 的成员数、`ddof` 与输出模式；
- 绘图的密度、抽样、字体、尺寸、共享尺度和输出格式。

处理阶段保持独立：

1. `predict_campaign` 审核可复用 prediction，只补齐缺失组合；
2. `compute_campaign_uq` 审核或计算 UQ，并严格重算验证；
3. `plot_campaign` 在九组 UQ 全部通过后统一分析和发布图形。

阶段拆分允许 GPU prediction 与 CPU UQ/plot 分别调度，也允许失败后只重跑未完成阶段。

## 5. 模块边界

### 5.1 Campaign 配置

新增 `bootstrap/campaign.py`，负责：

- 严格读取 YAML；
- 校验 run label、dataset label 和 storage key；
- 解析仓库相对路径和远端绝对路径；
- 确保 run 与 dataset 标签唯一；
- 确保只请求 `raw`、完整 8-member 和已声明的 reference targets；
- 拒绝输出目录落在任何 prediction/UQ 输入目录内。

配置中的数据集映射为：

| label | storage key | 配置路径 | reference targets |
|---|---|---|---|
| `matpes_test` | `test` | `data/dataset/matpes_test.extxyz` | energy, forces, stress |
| `mad_test` | `mad_test` | `data/dataset/mad-test.xyz` | energy, forces |
| `matpes_train` | `matpes_train` | `data/dataset/matpes_train.extxyz` | energy, forces, stress |

`matpes_test -> test` 是显式兼容映射，使远端已有 artifact 原位复用而不复制、不改名。

### 5.2 Prediction

从 `bootstrap/native_prediction.py` 提取通用的 `predict_dataset(...)`：

- 继续调用现有 PET base checkpoint loader 和 member checkpoint branch loader；
- 继续调用现有 batch inference 和 `PredictionStore`；
- dataset path、storage key 与 reference targets 改由 campaign 提供；
- 原有 `predict_run(...)` 保持兼容，并委托给通用函数处理 `val/test`。

`PredictionStore` 的目录键从固定 `val/test` 推广为安全 dataset key。安全键只允许小写字母、数字和下划线，禁止路径分隔符、`.`、`..`、绝对路径和符号链接逃逸。

每个新 prediction publication 必须包含：

- dataset label、storage key 和配置路径；
- `reference_targets`；
- 8 个有序 member 记录；
- `raw` prediction 文件的路径、shape、dtype、size 和 SHA256；
- targets artifact 的路径、shape、dtype、size 和 SHA256；
- 完成状态和 schema version。

完整 publication 在唯一 staging 目录生成，所有成员成功后才原子提升。已存在且完整、身份一致的 publication 零写入跳过；不完整、损坏或身份冲突的目录 fail closed。

### 5.3 可选 reference stress

现有 targets v1 要求 energy、forces、stress 全部存在，无法正确表达 MAD。新增 targets/manifest v2：

- structure id、原子数、offset、energy 和 forces 仍为 MAD 必需字段；
- stress 可以缺失；
- manifest 通过 `reference_targets` 精确声明可用于残差计算的字段；
- member prediction 仍可保存模型输出的 energy、forces、stress；
- v1 reader 和现有 `test` artifact 保持兼容；
- validator 依据 schema version 和 `reference_targets` 校验实际 NPZ keys，禁止隐式填充。

### 5.4 UQ

保留现有 canonical UQ 实现：

- Welford streaming mean/STD；
- `float64` reduction；
- sample STD，`ddof=1`；
- distinct unordered-pair GMD；
- energy per atom、force vector RMS 和 stress tensor RMS 派生量。

只将固定 `val/test` 限制推广为安全 storage key。UQ 计算只依赖 prediction 和 targets 中的结构计数，因此 MAD 缺少 reference stress 不影响预测应力 UQ 的保存；MAD 的 stress UQ 不用于残差图。

每个 UQ publication 完成后必须通过逐数组严格重算验证，未通过的结果不能进入 plotting。

### 5.5 Plotting

BootStrapping 内新增职责分离的模块：

- `plot_analysis.py`：加载并审核 source、构建科学 panel、过滤 log-invalid 数据、计算统计和共享尺度；
- `plot_rendering.py`：只负责确定性 Matplotlib 渲染；
- `plot_store.py`：计算 plot identity、事务发布并验证最终文件集合；
- `scripts/plot_campaign.py`：CLI orchestration。

这些模块可复用现有 BootStrapping artifact/hash/atomic-write 辅助函数，但不依赖 LLPR evaluation schema。

## 6. 科学数据流

每个 `run x dataset` 组合先审核 8 个 member 的 `best.pt`、member manifest 和 raw branch。

Prediction 产生的 raw member arrays 进入 canonical UQ。Plot source 同时审核 prediction manifest 与 UQ manifest，并要求 UQ 与完整 8-member raw prediction 严格一致，然后按 Carnet 定义构建 panel：

- Energy：先将每个成员的总能量除以结构原子数，再计算 ensemble mean 和 sample STD；残差为每原子 mean 与每原子 reference energy 的绝对差。
- Force：对每个原子、每个笛卡尔分量计算 ensemble mean 和 sample STD；残差为 mean 与 reference force 的逐分量绝对差。
- Stress：先将每个成员和 reference 的 `3 x 3` 应力对称化并转换为 `[xx, yy, zz, yz, xz, xy]`，再计算 ensemble mean、sample STD 和逐分量绝对残差。

Stress panel 必须从有序 member prediction 的对称 Voigt 值计算，不能简单对已经形成的 `3 x 3` STD 矩阵做后处理，因为两者在非对称输入下不等价。

## 7. 绘图定义

每个 panel：

- x 轴为 sample STD uncertainty；
- y 轴为 ensemble mean 的 absolute residual；
- x/y 均为 log scale，并使用同一数值范围形成正方形坐标；
- 包含 `y=x` 参考线；
- 使用固定随机种子的确定性散点子采样；
- 使用二维 histogram、Gaussian smoothing 和累计质量等高线；
- 默认 contour masses 为 `0.50, 0.70, 0.85, 0.95, 0.99`；
- 图中标注 log-space Spearman 与 `log10` Pearson；
- PNG 与 PDF 均由同一 figure 产生。

对每个 target，将其所有 panel 中正值且有限的 uncertainty/residual 对合并求全局 log 最小值和最大值，再增加配置化 margin。由此得到 energy、force、stress 三组共享尺度。任何 target 没有至少两个有效点时，整个绘图 publication 失败。

过滤时每一对数据只归入一个类别：`nan`、`inf`、`zero`、`negative` 或 valid。原始数量、有效数量和各排除数量必须闭合。

## 8. 输出结构

正式绘图目录为：

```text
Uncertainty_Quantification/Plots/BootStrapping/
└── raw_std_three_runs_three_datasets__<identity>/
```

每组 run 有 8 个图项：MATPES-test 三个、MAD-test 两个、MATPES-train 三个。三组共 24 个图项，每个图项输出 PNG 和 PDF，共 48 张图。

文件名采用：

```text
<run_label>__<dataset_label>__raw_<target>_uncertainty_vs_residual.<format>
```

另有：

- `raw_std_statistics.csv`：24 行扁平统计；
- `raw_std_statistics.json`：分层统计、过滤计数、共享尺度和输入摘要；
- `plot_manifest.json`：plot identity、输入 artifact SHA256、渲染配置、输出 size/SHA256 和 `PASS` 状态。

成功 publication 恰好包含 51 个文件。不存在成员扫描图或成员扫描表。

Plot identity 绑定三组 run、三个 dataset/storage key、prediction/UQ/targets artifact SHA256、raw mode、8-member 顺序、分析定义和渲染配置。它不绑定原始数据集语义指纹。

发布采用唯一 staging、publisher lock、原子提升和 manifest-last。相同 identity 的完整目录可以零写入复用；损坏、非空未提交、符号链接或 identity 冲突全部 fail closed，不静默覆盖。

## 9. 远端执行设计

本地只开发并提交代码。远端执行顺序为：

1. 同步已提交代码，不同步任何 outputs；
2. 准备远端 `data/dataset` 路径；MATPES 优先复用服务器已有文件，避免复制 398 MB train set；MAD 若远端不存在才上传必要文件；
3. preflight 检查输入可读性、reference targets、8 个 checkpoint、raw branch、CUDA、依赖和存储空间；
4. 记录三组现有 `predictions/test` 与 `uncertainty/test/raw` 的文件清单和 SHA256；
5. 提交 6 个互不写同一目录的 GPU prediction 工作单元，即 `3 runs x {mad_test, matpes_train}`；
6. 在 CPU 上计算并验证 6 组新增 UQ，验证并复用 3 组既有 MATPES-test UQ；
7. 九组 source 全部通过后，运行一个 CPU plotting job；
8. 严格验证 plot manifest、51 个文件、size 和 SHA256；
9. 将整个 plot identity 目录同步到本地同构路径，并逐文件核对远端/本地 SHA256。

SSH 始终启用 strict host-key checking。端口负载均衡命中未登记主机时只重试，不接受新 key、不关闭校验。

## 10. 失败与恢复

- GPU OOM：当前 prediction publication 失败；调整配置中的 batch size 后显式重跑，不自动改变 dtype 或科学参数。
- 单个 member 失败：不发布该 run/dataset 的 prediction manifest，也不以少于 8 个成员继续。
- 数据标签不匹配：preflight 或 targets extraction 失败，不填充缺失标签。
- prediction/UQ 非有限：立即失败并报告字段、成员和数据集。
- 已有 artifact 完整且身份一致：只读复用。
- 已有 artifact 损坏、缺文件、hash 不符或身份冲突：fail closed；不自动删除或覆盖。
- 任一 UQ 缺失或未通过严格重算：不启动 plotting。
- 绘图失败：删除或保留唯一 staging 作为诊断，但不影响任何既有 PASS publication。
- 本地同步失败：远端 PASS 结果保持不变，可单独重试同步与 hash 核对。

## 11. 测试设计

所有测试在远端运行。

### 11.1 单元测试

- dataset/storage key 的安全校验和路径逃逸拒绝；
- targets v1 与 v2 reader 兼容；
- MAD 缺少 stress 时可 prediction/UQ，但不会生成 stress residual panel；
- 不允许 reference target 声明与 NPZ keys 不一致；
- 通用 dataset prediction 保持原 `val/test` 入口兼容；
- UQ 的 8-member、`ddof=1`、float64、GMD 和严格重算；
- energy per atom、force component、symmetric Voigt stress 的数值定义；
- invalid-pair 分类闭合；
- 同 target 全局共享尺度；
- identity、staging、manifest-last、零写入复用和 fail-closed 行为。

### 11.2 合成发布测试

构造三 run、三 dataset 的小型 artifacts，断言：

- 恰好 24 个 panel；
- MAD 没有 stress panel；
- 恰好 48 张图和 51 个总文件；
- 同 target 的所有 manifest/CSV 坐标范围逐值一致；
- 不存在任何 member sweep 文件；
- manifest 中每个输出 SHA256 与实际文件一致。

### 11.3 真实 checkpoint smoke

在远端 CPU 使用 `matpes_n20.extxyz` 与至少两个真实 member checkpoint 运行 prediction -> UQ -> plot，验证正式 loader、模型输出字段、targets schema 和绘图 publication。小数据与正式全量结果比较的是 schema、array layout、manifest 结构和图形文件集合，不比较数值相等。

### 11.4 正式验收

- 6 组新 prediction publication 完整；
- 6 组新 UQ publication 严格验证通过；
- 3 组既有 MATPES-test prediction/UQ 在执行前后 SHA256 不变；
- 24 个 panel 均来自完整 8-member raw ensemble；
- 共享尺度按 target 完全一致；
- 远端与本地 51 个绘图文件逐文件 SHA256 一致；
- 本地 Git 不包含 checkpoint、prediction、UQ、大数据或临时 staging；
- 只提交可发布代码、配置、测试和文档，运行产物由 `.gitignore` 排除。

## 12. 非目标

- 不训练或微调任何模型；
- 不迁移或重写 checkpoint；
- 不重新推理现有 MATPES-test；
- 不使用 EMA；
- 不生成成员数扫描图；
- 不为 MAD 生成 stress residual 图；
- 不引入 dataset semantic fingerprint；
- 不把大规模远端产物复制到本地；
- 不把迁移工具或历史迁移信息暴露为正式发布工作流；
- 不修改 BootStrapping 之外的 UQ 方法。
