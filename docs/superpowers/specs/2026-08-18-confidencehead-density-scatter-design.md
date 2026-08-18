# ConfidenceHead 连续散点密度图设计

## 1. 背景与目标

ConfidenceHead 当前的发布图以固定 50 个 argmax bin 的箱线图为主。箱线图适合查看离散 bin 内的误差分布，但不能连续展示模型输出的 expected error 与真实误差之间的关系。

本设计新增一套参考 `carnet_new/Uncertainty_Quantification/Plots/FGE` 的连续散点密度图，用于现有已完成结果的后处理和绘图。绘图不重新训练、不重新推理、不修改历史结果。

目标范围：

- 数据集：`matpes_train`、`matpes_test`、`mad_test`。
- 能量：`order=1..8`，误差为逐原子绝对能量误差。
- 力：单独的 `atom_mean` 分支，误差为逐原子 Cartesian 分量绝对误差均值。
- 横轴使用 ConfidenceHead 的连续 `expected error`，不使用 argmax bin representative。
- 纵轴使用 `observed absolute error`。
- 双坐标轴均使用 log scale。
- 旧 `argmax_bin_boxplots/` 目录和旧 manifest 保留不变，新图另存为独立目录。

## 2. 方案选择

评估过三种实现路径：

1. 直接改造现有 `plot_analysis.py`，将箱线图函数替换为密度图。
2. 新增独立的 `density_plotting.py`，复用已验证的 `PlotSeries` 和数据加载逻辑。
3. 直接复用 `carnet_new` FGE 绘图模块。

采用方案 2。它保留 FGE 的视觉和密度算法思想，同时保持 ConfidenceHead 自己的输入契约和能量/力语义，避免跨仓库业务模型依赖；旧箱线图 API 和历史 manifest 也不需要改变。

## 3. 模块边界与数据流

新增模块：

```text
Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plotting.py
```

该模块只接收已经通过验证的 `PlotSeries`，不读取 checkpoint，不执行模型推理，也不修改训练或预测结果。

数据流：

```text
completed run directory
        |
        v
load_plot_series()
        |
        v
PlotSeries
  - expected error      -> x
  - observed error      -> y
  - target/order        -> 语义和标题
  - structure_ids       -> 结果审计
        |
        v
density_plotting.py
  - positive/finite filtering
  - log10 transformation
  - all-sample 2D density
  - deterministic scatter sampling
  - FGE-style rendering
  - CSV/JSON statistics
        |
        v
new density publication directory
```

模块划分：

- `DensitySettings`：保存散点上限、随机种子、网格、平滑、等高线、DPI 等绘图参数。
- `analyze_density_panel`：完成形状检查、正值/有限值筛选、log10 变换、相关系数、全量密度和固定种子散点抽样。
- `render_density_panel`：完成单图的双 log 坐标、散点、密度等高线、覆盖区域、参考线和统计框。
- `publish_density_suite`：生成单 run 图、能量 8 阶汇总图、统计文件和独立 manifest。

现有 `plot_analysis.py`、`plot_single_boxplot` 和旧 boxplot manifest 不改变。`external_plotting.py` 仅增加并行的 density publication 入口。

## 4. 图形设计

### 4.1 单图

每个单图包含：

- 横轴：`Expected absolute error`。
- 纵轴：`Observed absolute error`。
- 双 log 坐标和方形坐标区域。
- 黑色虚线 `y=x`。
- `y <= x` 区域的浅灰色填充，表示预测不确定性覆盖真实误差。
- FGE 风格的低透明度橙色原始散点。
- 深橙色 Gaussian-smoothed density 等高线。
- 统计框：`Spearman rho`、`Pearson r (log10)` 和 `valid / total`。

物理量标签严格区分：

- 能量：`Absolute energy error per atom (eV/atom)`。
- 力：`Mean absolute force-component error per atom (eV/A)`。

### 4.2 密度算法

所有正且有限的样本参与密度计算；显示散点可以抽样，但不能影响密度统计。流程为：

1. 过滤零值、负值、NaN、Inf，并记录每类排除数量。
2. 将 expected 和 observed 映射到 log10 空间。
3. 在共享 log 范围内生成 `160 x 160` 二维直方图。
4. 使用 Gaussian filter，默认 `sigma=1.2`。
5. 按累计密度质量生成 `50%`、`70%`、`85%`、`95%`、`99%` 等高线。
6. 散点层最多显示 `20,000` 个样本，使用固定随机种子、无放回抽样并按索引排序。
7. log10 范围两侧增加至少 `0.05` 的留白。

相关系数在 log10 数据上计算并写入图内和统计文件；详细排除分类不放入图内。

### 4.3 汇总图

- 每个数据集生成 8 张能量单图，分别对应 order 1 到 8。
- 每个数据集生成 1 张力单图。
- 每个数据集生成 1 张 4x2 能量汇总图。
- 同一数据集的 8 个能量面板共用由全部 order 联合确定的坐标范围和图例规范。
- 力单独确定坐标范围，不与能量混排。
- 三个数据集之间不强制共用 log 范围，避免 MAD 分布压缩 MATPES 的细节。

## 5. 输出与配置

新发布根目录：

```text
Uncertainty_Quantification/Plots/ConfidenceHead/density_scatter/
```

目录结构：

```text
density_scatter/
  matpes_train/
    runs/<run-name>/
    comparisons/
    manifest.json
  matpes_test/
    runs/<run-name>/
    comparisons/
    manifest.json
  mad_test/
    runs/<run-name>/
    comparisons/
    manifest.json
  comparisons/
    cross_dataset_energy_correlations.csv
    cross_dataset_force_correlations.csv
    manifest.json
```

每个 run 输出：

```text
test_<target>_expected_vs_observed_density.png
test_<target>_expected_vs_observed_density.pdf
density_statistics.csv
density_analysis.json
```

数据集 comparisons 输出：

```text
combined_energy_expected_vs_observed_density.png
combined_energy_expected_vs_observed_density.pdf
energy_density_summary.csv
energy_density_summary.json
```

新增配置文件：

```text
Uncertainty_Quantification/ConfidenceHead/configs/density_plots.yaml
```

默认参数：

```yaml
output_root: ...
scatter_max_points: 20000
scatter_seed: 20260714
grid_size: 160
gaussian_sigma: 1.2
contour_masses: [0.50, 0.70, 0.85, 0.95, 0.99]
log_margin: 0.05
dpi: 300
```

数据集和 run 路径继续由现有 external plotting 配置提供，不复制预测文件。

## 6. Manifest、可复现性与原子发布

新 manifest 使用独立 schema，例如：

```text
upet_confidence_density_plots_v1
```

identity 绑定以下内容：

- 输入 run manifest SHA256。
- 数据集名称。
- 目标分支和能量 order。
- 完整 density 配置。
- 绘图算法版本。

发布采用 staging 目录和原子替换。中途失败时不产生正式的“完成但不完整” manifest。相同输入和配置重复执行时验证并复用已有结果；identity 不同则拒绝覆盖，要求更换输出目录或显式清理。

旧 `argmax_bin_boxplots/` 输出、旧 manifest 和旧统计 CSV 不删除、不覆盖。

## 7. 执行入口与失败处理

新增入口：

```text
Uncertainty_Quantification/ConfidenceHead/scripts/plot_density_scatter.py
```

执行前预扫描三个数据集的九个已完成 run：

1. 调用 `verify_run(..., allow_plots=True)`。
2. 调用 `load_plot_series()`。
3. 确认能量 order 覆盖 1 到 8。
4. 确认力分支唯一且为 `atom_mean`。
5. 确认所有输入使用 50 个 `fixed_linear_v1` bins。
6. 确认 prediction/evaluation manifest 完整且 SHA256 正确。

只有预扫描全部通过后才创建 staging 输出。任意 run 缺失、语义不一致或 manifest 校验失败时，对应数据集不发布，且不影响旧 boxplot 结果。

## 8. 测试与验收

新增单元测试和端到端测试覆盖：

- 正值、零值、负值、NaN、Inf 的过滤和分类计数。
- 全量密度计算独立于散点显示上限。
- 固定 seed 得到稳定且可重复的散点索引。
- 密度网格形状、等高线质量和 log 范围稳定。
- 能量 8 个 order 共享数据集坐标范围。
- 力和能量标签、单位、target mode 校验正确。
- PNG 可解码、PDF 非空。
- 新 manifest 的完整性、SHA256 和 identity 冲突校验。
- 旧 boxplot 测试继续通过，旧目录和旧 manifest 字节级不变。
- 小型合成 `PlotSeries` 的 CPU 端到端发布测试。

最终验收要求：三个数据集均生成完整 density manifest；每个数据集拥有 8 张能量单图、1 张力单图和 1 张 4x2 能量汇总图；每张图同时有 PNG/PDF；所有统计文件可追溯到输入 manifest；远端生成的 density 目录可以完整拉取到本地。
