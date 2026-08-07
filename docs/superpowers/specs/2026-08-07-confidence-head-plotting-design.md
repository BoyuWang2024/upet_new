# UPET ConfidenceHead 已完成任务绘图设计

## 1. 背景

远端 `bywang@121.48.164.204` 的 UPET ConfidenceHead 已完成九个目标任务：

- 一个仅训练力的任务：`force_coefficient=1`、`energy_coefficient=0`；
- 八个仅训练能量的任务：`force_coefficient=0`、`energy_coefficient=1`，多项式阶数为 1–8。

本设计参考 `carnet_new/Uncertainty_Quantification/ConfidenceHead/outputs` 及其当前绘图代码组织方式，为 UPET 增加可重复使用的结果分析和绘图能力。绘图只消费已经生成的预测结果，不修改 checkpoint、训练配置、训练日志或预测文件。

## 2. 目标与非目标

### 2.1 目标

1. 为九个已完成任务分别生成测试集 argmax-bin 箱线图和逐 bin 统计 CSV。
2. 为能量 order 1–8 生成 Pearson/Spearman 相关性比较图和 CSV。
3. 生成能量任务的组合箱线图以及力任务的组合图。
4. 保持图形样式、文件命名和目录结构尽量接近 Carnet，便于横向比较。
5. 对输入文件、目标语义、样本维度、bin 数和相关性结果做严格校验，避免对错误或不完整结果静默绘图。
6. 本地仅进行静态检查和合成小数据测试；真实九任务绘图在远端 CPU 环境完成。

### 2.2 非目标

- 不重新训练或重新评估 ConfidenceHead。
- 不迁移、修改或复制旧 UPET 结果。
- 不修改已有九个任务的超参数。
- 不绘制训练曲线、校准曲线、置信区间或当前范围之外的其他分析图。
- 不把能量 ConfidenceHead 的输入或预测用于力 ConfidenceHead，反之亦然。

## 3. 方案选择

采用仓库内可复用实现，而不是远端一次性脚本或先导出全部逐样本 CSV 的两阶段方案。

原因：训练重新运行后可直接复用；UPET 自身的数据语义和校验逻辑可以集中维护；输出目录与 Carnet 保持一致；不需要额外复制大规模逐样本数据。

## 4. 代码组织

新增以下模块和入口：

```text
Uncertainty_Quantification/ConfidenceHead/
├── confidence_head/
│   └── plot_analysis.py
└── scripts/
    ├── plot_argmax_bin_boxplots.py
    ├── plot_energy_order_correlations.py
    └── plot_completed_runs.py
```

职责如下：

- `confidence_head/plot_analysis.py`
  - 加载并校验 `test_predictions.pt`；
  - 提取能量或力的 logits、observed error 和 expected error；
  - 计算 argmax-bin 分组、逐 bin 描述性统计和相关性；
  - 提供单任务和组合图的公共绘图函数。
- `scripts/plot_argmax_bin_boxplots.py`
  - 面向一个任务生成箱线图和统计 CSV。
- `scripts/plot_energy_order_correlations.py`
  - 面向 order 1–8 的能量任务生成相关性比较图和 CSV。
- `scripts/plot_completed_runs.py`
  - 远端批量入口；发现、分类和验证目标九任务后一次生成全部结果。

批量入口接受 `--runs-root` 和 `--output-root`。默认自动发现时，只接受状态完整且配置语义符合要求的任务：一个 force-only 任务和 order 1–8 各一个 energy-only 任务。若同一类型存在多个完整候选，程序拒绝猜测，并要求用可重复的 `--run-dir` 明确指定目录。

## 5. 数据语义

### 5.1 能量

使用预测文件中的：

- `energy_logits`；
- `energy_observed_errors`；
- `energy_expected_errors`。

能量误差沿用训练和评估阶段已经定义好的逐原子能量误差。绘图阶段不得再次除以原子数。每个结构对应一个能量样本。

### 5.2 力

使用预测文件中的：

- `force_logits`；
- `force_observed_errors`；
- `force_expected_errors`。

每个原子对应一个力样本。观测误差沿用当前默认 `atom_mean` 定义，即每个原子的三个笛卡尔分量绝对误差的平均值。绘图阶段不重新解释为逐分量样本，也不对三个分量再次展开。

### 5.3 Argmax-bin 分组

对每个样本的 logits 沿最后一维取 `argmax` 作为预测 bin，再按预测 bin 对 observed error 分组。固定显示全部 50 个 bin：

- 有样本的 bin 绘制箱体；
- 空 bin 保留横轴位置并标注 `n=0`；
- 统计 CSV 始终包含 50 行。

统计 CSV 至少包含：

```text
bin, sample_count, mean_observed_error, median_observed_error,
std_observed_error, q25_observed_error, q75_observed_error,
min_observed_error, max_observed_error, mean_expected_error
```

空 bin 的 `sample_count` 为 0，其余统计值为空值。

## 6. 相关性计算

对每个能量 order，使用同一批逐结构样本的 `energy_expected_errors` 和 `energy_observed_errors` 计算：

- Pearson 相关系数；
- Spearman 相关系数。

只保留两者均为有限值的样本。有效样本少于 2 个、任一数组为常量或计算结果不是有限值时，程序报错并停止汇总。当前不计算置信区间。

相关性 CSV 包含：

```text
order, sample_count, pearson, spearman
```

重新计算的结果需与每个任务现有 `evaluation_metrics.json` 中对应指标交叉核对。允许小的浮点误差；超过实现中明确记录的容差则停止，并报告任务、指标、两侧数值和差异。

## 7. 图形规范

### 7.1 单任务箱线图

- PNG 300 DPI 和矢量 PDF 各一份；
- 横轴固定为 50 个 bin，标签包含 bin 编号和样本数；
- 纵轴使用与 Carnet 当前实现一致的 `symlog` 设置；
- 英文标题、坐标和图例；
- 配色、字体、网格、异常值和箱体样式尽量复用 Carnet 当前规范；
- 能量标题明确为 per-atom energy error；
- 力标题明确为 per-atom mean absolute Cartesian-component force error。

### 7.2 组合图

- 八个能量任务按 order 1–8 排列为 `4 × 2` 子图；
- 力任务单独成图，不与能量混合；
- 组合图至少输出 PDF，若生成 PNG 则同样使用 300 DPI。

### 7.3 相关性图

- 横轴为 order 1–8；
- 两条曲线分别表示 Pearson 和 Spearman；
- 不绘制置信区间；
- 输出 PNG 300 DPI、PDF 和底层 CSV。

## 8. 输出目录

每个任务自身的结果写入：

```text
outputs/runs/<run>/plots/argmax_bin_boxplots/
├── test_energy_argmax_bin_boxplot.png
├── test_energy_argmax_bin_boxplot.pdf
└── test_energy_argmax_bin_statistics.csv
```

力任务将文件名中的 `energy` 替换为 `force`。

跨任务比较写入：

```text
outputs/comparisons/
├── energy_correlations/
│   ├── linear_order_correlations_no_ci.png
│   ├── linear_order_correlations_no_ci.pdf
│   └── linear_order_correlations_no_ci.csv
└── argmax_bin_boxplots/
    ├── combined_energy_argmax_bin_boxplots.pdf
    └── combined_force_argmax_bin_boxplots.pdf
```

输出采用原子写入或临时文件后替换，避免中途中断留下看似完整的文件。

## 9. 校验与错误处理

批量绘图前必须一次性完成以下校验：

1. 九个任务的完成标记、预测文件和评估指标文件存在。
2. 任务分类准确：一个 force-only，八个 energy-only。
3. 能量任务的 order 恰好覆盖 1–8 且无重复。
4. 目标所需的三个张量键均存在。
5. logits 为二维且第二维为 50；误差数组是一维。
6. logits 行数与两个误差数组长度一致。
7. 数据均可转换为 CPU 浮点数组，且用于分组和相关性的值满足有限性要求。
8. `binning.json` 与目标类型、50-bin 固定线性分箱和训练时误差定义一致。
9. 相关性复算与 `evaluation_metrics.json` 在容差内一致。

任何一项失败时不生成跨任务汇总，错误信息应包含具体任务目录和失败字段。单任务入口也使用同一套校验函数，避免两套逻辑分叉。

## 10. 测试策略

本地测试使用小型合成预测数据，不读取真实 checkpoint 或远端数据：

- 能量逐结构样本和力逐原子样本均可正确加载；
- argmax 分组正确，50 个 bin 全部出现在 CSV 中；
- 空 bin 计数为 0、统计值为空；
- 能量不被再次按原子数归一化；
- 力不被展开为三个分量；
- Pearson/Spearman 与已知结果一致；
- 缺键、形状不匹配、错误 bin 数、重复 order、目标配置错误时明确失败；
- PNG/PDF/CSV 可生成且文件非空。

执行与仓库约定相适配的定向测试和静态检查。真实数据验收在远端 `upet_new` 环境、CPU 模式完成。

## 11. 远端执行与验收

1. 将当前分支的实现提交并同步到 `/home/bywang/code/UQ/upet_new`。
2. 使用 `/home/bywang/.conda/envs/upet_new/bin/python` 运行批量入口。
3. 明确指定或严格发现九个已完成任务，禁止把失败任务或历史重复任务混入汇总。
4. 检查九份单任务统计 CSV 均为 50 行。
5. 检查九套单任务图、两类组合图和能量相关性图均可读取且文件非空。
6. 检查相关性 CSV 的 order 恰好为 1–8。
7. 抽查图中样本数总和与预测数组长度一致。
8. 将代表性 PNG 取回展示，供人工确认布局和可读性。

## 12. 成功标准

- 不改变现有训练结果及训练配置；
- 九个指定任务均有正确目标类型的箱线图和 50-bin 统计 CSV；
- 能量 order 1–8 有完整且经复核的相关性比较结果；
- 能量逐原子与力 `atom_mean` 语义在代码、标签和统计中一致；
- 图形可正常打开，风格与 Carnet 当前输出相近；
- 本地定向测试和远端真实结果验收均通过。
