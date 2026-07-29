# UPET LLPR 不确定性量化

本目录提供 UPET 最后一层拉普拉斯近似（LLPR）的曲率构建、校准、评估、旧成果迁移、完整性验证和绘图工具。当前正式成果使用直接指定的 energy/force `eta`；代码同时支持两种模式：

- `fixed`：直接指定 energy/force 的 `eta`；
- `fit`：只在验证集上分别拟合 energy/force 的 `eta`，测试集不参与选择。

本文的逐文件清单范围是当前正式迁移目录：

```text
Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
```

清单依据当前磁盘文件、四级 manifest、`inventory.json`、保留的旧脚本/配置以及 NPZ/JSON 实际内容编写。它描述的是现存成果，不代表重新计算。

## 环境与命令

```bash
conda activate upet_new
cd /home/lilong/code/UQ/upet_new
```

统一入口包含 `build`、`calibrate`、`evaluate`、`run`、`import-legacy`、`verify` 和 `plot`：

```bash
python -m Uncertainty_Quantification.LLPR.llpr import-legacy \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml

python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy

python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml
```

`verify` 会检查所有 complete manifest、manifest 声明文件的 SHA-256、NPZ 可读性，以及 `legacy_raw` inventory 中每个文件的安全相对路径、大小和 SHA-256。

## 当前正式目录状态

| 项目 | 当前真实值 |
| --- | --- |
| 根 identity | `463ae307034cf391` |
| curvature identity | `406edc88d16fdcc7` |
| calibration identity | `50217238109b0418` |
| evaluation identity | `28d0d911b3060988` |
| 文件总数 | 64 |
| 规范化文件 | 12 |
| `legacy_raw` 原样保留文件 | 52 |
| complete manifest | 4 |
| full verify 报告的受检文件 | 60（64 个文件减去 4 个 manifest 本身） |
| `plots/` | 目录存在，但当前为空 |

因此，当前正式目录中没有规范化绘图产物。旧 PNG、PDF、CSV 和绘图日志仍作为历史证据保存在 `legacy_raw/`；本次文档整理不会重新生成图，也不会修改任何正式成果。

## 迁移边界和对应关系

旧目录已经删除，但其经审计的逻辑根路径是：

```text
OLD=/home/lilong/code/UQ/upet/UQ_LLPR/matpes_r2/Hef
```

后文 `$OLD/<relative-path>` 表示迁移前的旧文件；当前原样保留位置是 `legacy_raw/<relative-path>`。`inventory.json` 中的 `destination_relative` 是这项映射的依据。由于旧目录已清理，当前 inventory 的 `source_absolute` 指向新的 `legacy_raw` 保留副本，不能再把它误读为旧目录仍然存在。

关系列使用四种标签：

- **原样复制**：文件字节不变地进入 `legacy_raw`，并由大小和 SHA-256 复核。
- **提取并重打包**：从旧 NPZ 中读取实际数组，选择有效块或按同名数组写入规范化 NPZ；数组值不变，但压缩容器的文件哈希可以不同。
- **基于旧数据生成**：根据已审计的旧数组、旧摘要或迁移配置重新汇总元数据/统计量，不运行模型、checkpoint、前向传播或 Jacobian。
- **无旧文件一一对应**：迁移层新增的 identity、manifest 或 inventory，用于可追溯性和完整性验证。

`inventory.json` 的分类含义如下：

- `authoritative`：文件被完整保留并通过审计；它不等价于“规范化结果一定直接消费了该文件”。
- `legacy_smoke`：旧的小规模试运行产物，不是正式科学结果。
- `incomplete`：文件引用的配套输出不完整。
- `orphan`：文件存在，但不在对应正式绘图摘要声明的输出列表中。

`formal_source: true` 是当前 inventory 规则固定标记的五文件核心 payload allowlist，用于突出三份 H 矩阵和正式 test details/summary；它不是迁移器的完整依赖图。`Hef_full_run_summary.json` 和 `alpha_val_full_joint_summary.json` 虽标记为 false，迁移器仍直接读取它们以取得 build count 和 Alpha/validation count。其他 false 文件保留作配置、运行记录、交叉检查或历史追溯。

## 规范数据流

正式迁移不加载 checkpoint，也不重新执行 train/validation/test 的模型前向或 Jacobian。真实转换链路是：

```text
$OLD/results/H_E_full_run.npz + $OLD/results/H_F_full_run.npz
  -> 选择 energy/force 有效块 -> curvature.npz

$OLD/results/H_EF_full_run.npz
  -> 验证 H_EF == H_E + H_F

$OLD/results/Hef_full_run_summary.json + 有效曲率块 + 固定 eta
  -> diagnostics.json

$OLD/results/alpha_val_full_joint_summary.json + 固定 eta + 条件数
  -> candidates.json + calibration/summary.json

$OLD/results/LLPR/llpr_test_full_gpu_details.npz
  -> 同名数组重打包为 details.npz
  -> 重新汇总 evaluation/summary.json
  -> summary + 前 10 个 structure_index 生成 preview.json
```

旧验证摘要没有保存逐样本验证残差和方差，所以无法恢复 Gaussian NLL 和 1/2/3 sigma coverage。这些字段保持 `null`，并标记 `diagnostics_status: unavailable_from_legacy_validation_summary`；迁移器不会用测试集结果替代验证集诊断。

## 12 个规范化文件逐项说明

<!-- BEGIN NORMALIZED FILE CATALOG -->
| 当前相对路径 | 作用与真实内容 | 对应旧文件/输入 | 关系 |
| --- | --- | --- | --- |
| `manifest.json` | 迁移根清单：payload 记录 checkpoint/train/validation/test 数据集 SHA-256、三段计数、维度、固定 eta/Alpha 和 inventory SHA-256；顶层记录 schema/formula 版本、固定 ridge 标记及三个阶段 identity，并只在 `files` 中声明 `inventory.json` 的哈希。 | 汇总已审计输入与迁移配置，没有单个旧文件与它等价。 | 无旧文件一一对应 |
| `inventory.json` | 52 个旧文件的审计目录，逐项保存 `destination_relative`、字节数、SHA-256、分类和 `formal_source`；也是 `legacy_raw` 全量验证的依据。 | 由 `$OLD` 全树的实际文件清单生成。 | 无旧文件一一对应 |
| `curvature/406edc88d16fdcc7/manifest.json` | curvature 阶段清单：payload 直接保存 checkpoint SHA-256、build 数据集 SHA-256、root identity 和 origin；顶层保存 schema/formula、固定 ridge、状态，并声明 `curvature.npz` 与 `diagnostics.json` 的哈希。它不直接保存旧矩阵哈希、维度或 eta；旧矩阵由根 inventory 追溯，维度在实际 NPZ/diagnostics 中体现。 | 迁移层清单，没有单个旧文件与它等价。 | 无旧文件一一对应 |
| `curvature/406edc88d16fdcc7/curvature.npz` | 规范化曲率容器：`energy` 为 `(1026, 1026)` float64 有效块，`force` 为 `(3078, 3078)` float64 有效块。两块分别从旧 `H_E` 和 `H_F` 的联合 `(4104, 4104)` 画布提取；旧 `H_EF` 用于核对两者之和。 | `$OLD/results/H_E_full_run.npz`、`$OLD/results/H_F_full_run.npz`；`$OLD/results/H_EF_full_run.npz` 用于和不变量检查。 | 提取并重打包 |
| `curvature/406edc88d16fdcc7/diagnostics.json` | 保存 348780 个构建结构、energy/force/joint 维度、有效块条件数 `1.0623512652391861e15`/`1.3809081464650994e12`，以及 `legacy_eta_squared_diagnostic: raw provenance only`。该文件本身没有 eta 或 `condition_warning` 字段；条件数告警出现在 calibration 候选/摘要中。 | 构建计数来自 `$OLD/results/Hef_full_run_summary.json`，条件数由规范化有效块计算。 | 基于旧数据生成 |
| `calibration/50217238109b0418/manifest.json` | calibration 阶段清单：payload 直接保存 curvature identity、validation 数据集 SHA-256、`fixed` 模式、两类 `eta` 和 origin，并声明候选表与选中摘要的哈希。`validation_sha256` 对应 `data/dataset/matpes_val.extxyz`，不是旧联合 Alpha 摘要的文件哈希；旧摘要由根 inventory 间接保护。 | 迁移层清单，没有单个旧文件与它等价。 | 无旧文件一一对应 |
| `calibration/50217238109b0418/candidates.json` | energy/force 各保存一个固定候选。energy：`eta=1e-6`、`alpha=1.1467388818005693`、验证数 19370；force：`eta=1e-6`、`alpha=0.2095766082027508`、验证分量数 458877。NLL/coverage 为 `null`。 | `$OLD/results/alpha_val_full_joint_summary.json`，并加入规范化曲率条件数。 | 基于旧数据生成 |
| `calibration/50217238109b0418/summary.json` | `mode: fixed` 的最终选择，分别引用上述唯一 energy/force 候选；保存 `alpha_sq=1.31501006303322` 和 `0.04392235470576932` 以及验证计数、条件数告警和不可恢复诊断状态。 | `$OLD/results/alpha_val_full_joint_summary.json` 与固定迁移参数。 | 基于旧数据生成 |
| `evaluation/50217238109b0418/28d0d911b3060988/manifest.json` | evaluation 阶段清单：payload 直接保存 curvature/calibration identity、test 数据集 SHA-256 和 origin，并声明 `details.npz`、`summary.json`、`preview.json` 的哈希。`test_sha256` 对应 `data/dataset/matpes_test.extxyz`；旧正式 test details/summary 通过根 inventory 追溯，而不是由本阶段 manifest 直接列出哈希。 | 迁移层清单，没有单个旧文件与它等价。 | 无旧文件一一对应 |
| `evaluation/50217238109b0418/28d0d911b3060988/details.npz` | 正式测试逐结构、逐力分量、逐原子及结构聚合的 44 个数组。与旧正式 details 的同名数组逐数组完全相等；因规范化写回压缩，NPZ 文件 SHA-256 不要求相同。 | `$OLD/results/LLPR/llpr_test_full_gpu_details.npz`。 | 提取并重打包 |
| `evaluation/50217238109b0418/28d0d911b3060988/summary.json` | 从规范化 details 重新汇总：19374 个结构、149321 个原子、447963 个力分量；energy RMSE/atom `0.05493578732829762`，force RMSE/component `0.1506777498968727`，并附带经审计的 eta/Alpha。 | 数值聚合来自 `$OLD/results/LLPR/llpr_test_full_gpu_details.npz`；旧正式 summary 用于计数和来源一致性检查。 | 基于旧数据生成 |
| `evaluation/50217238109b0418/28d0d911b3060988/preview.json` | 包含完整规范化 summary 和前 10 个结构索引（0–9），用于轻量查看而不加载大型 details。 | 由规范化 `details.npz` 和 `summary.json` 生成。 | 基于旧数据生成 |
<!-- END NORMALIZED FILE CATALOG -->

## 52 个 `legacy_raw` 文件逐项说明

下表中每一行的旧文件列都与当前 `legacy_raw` 相对路径严格同构；所有文件均为**原样复制**。分类和 `formal_source` 值来自当前 `inventory.json`，不是人工推测。

<!-- BEGIN LEGACY RAW FILE CATALOG -->
| 当前相对路径 | 真实作用/内容 | 对应迁移前旧文件 | inventory 分类 | formal_source |
| --- | --- | --- | --- | --- |
| `legacy_raw/configs/compute_Alpha.yaml` | 旧验证集 Alpha 计算配置：指定 checkpoint、验证数据、energy/force 目标、固定 `eta`、矩阵输入、批次/设备和输出位置。 | `$OLD/configs/compute_Alpha.yaml` | `authoritative` | false |
| `legacy_raw/configs/compute_Hef.yaml` | 旧训练集曲率构建配置：指定 Huber 损失、energy/force readout、CUDA/float32 前向、float64 累积、对角 `eta` 和三份 H 矩阵输出。 | `$OLD/configs/compute_Hef.yaml` | `authoritative` | false |
| `legacy_raw/configs/compute_LLPR.yaml` | 旧正式测试评估配置：指定测试数据、H 矩阵、固定 energy/force Alpha 与 eta、逐分量/逐原子/逐结构明细保存。 | `$OLD/configs/compute_LLPR.yaml` | `authoritative` | false |
| `legacy_raw/configs/plot_LLPR_reference.yaml` | 旧参考绘图配置：定义正式 details 输入、energy/force 不确定性—残差图、PNG/PDF、统计 CSV 和原子发布参数。 | `$OLD/configs/plot_LLPR_reference.yaml` | `authoritative` | false |
| `legacy_raw/results/H_EF_full_run.npz` | 联合 `(4104, 4104)` float64 曲率矩阵，数组名 `H`；迁移时用于验证 `H_EF == H_E + H_F`。 | `$OLD/results/H_EF_full_run.npz` | `authoritative` | true |
| `legacy_raw/results/H_E_full_run.npz` | energy 曲率的联合 `(4104, 4104)` float64 画布，数组名 `H`；左上 `(1026, 1026)` 有效块进入规范化 `energy`。 | `$OLD/results/H_E_full_run.npz` | `authoritative` | true |
| `legacy_raw/results/H_F_full_run.npz` | force 曲率的联合 `(4104, 4104)` float64 画布，数组名 `H`；右下 `(3078, 3078)` 有效块进入规范化 `force`。 | `$OLD/results/H_F_full_run.npz` | `authoritative` | true |
| `legacy_raw/results/Hef_full_run_summary.json` | 旧曲率全量构建摘要：记录 348780 个已使用结构、参数/readout 维度、损失和矩阵输出等运行元数据；迁移提取构建计数。 | `$OLD/results/Hef_full_run_summary.json` | `authoritative` | false |
| `legacy_raw/results/alpha_val_full_energy_summary.json` | 旧 energy 专项验证视图，记录固定 `eta=1e-6` 下的 energy Alpha 和 19370 个有效验证样本；保留作交叉检查。 | `$OLD/results/alpha_val_full_energy_summary.json` | `authoritative` | false |
| `legacy_raw/results/alpha_val_full_force_summary.json` | 旧 force 专项验证视图，记录固定 `eta=1e-6` 下的 force Alpha 和 458877 个有效笛卡尔分量；保留作交叉检查。 | `$OLD/results/alpha_val_full_force_summary.json` | `authoritative` | false |
| `legacy_raw/results/alpha_val_full_joint_summary.json` | energy/force 联合验证摘要，是规范化 calibration 的直接统计来源：Alpha 分别为 `1.1467388818005693`、`0.2095766082027508`。 | `$OLD/results/alpha_val_full_joint_summary.json` | `authoritative` | false |
| `legacy_raw/results/alpha_val_full_run.log` | 旧联合验证运行日志，记录模型加载、数据遍历、求解进度、耗时和输出写入过程；用于运行级追溯，不直接生成规范化数值。 | `$OLD/results/alpha_val_full_run.log` | `authoritative` | false |
| `legacy_raw/results/llpr_test_details.npz` | 旧 smoke/dry-run 明细，仅含 2 个结构、23 个原子、69 个力分量；明确不是正式测试结果。 | `$OLD/results/llpr_test_details.npz` | `legacy_smoke` | false |
| `legacy_raw/results/llpr_test_dry_run.log` | 上述 2 结构 smoke/dry-run 的运行日志，用于验证旧链路可启动。 | `$OLD/results/llpr_test_dry_run.log` | `legacy_smoke` | false |
| `legacy_raw/results/llpr_test_small_preview.json` | 旧 dry-run 的小预览记录。inventory 规则将其保留为 authoritative，但其内容和配套 summary 明确表明它不是正式结果。 | `$OLD/results/llpr_test_small_preview.json` | `authoritative` | false |
| `legacy_raw/results/llpr_test_summary.json` | 旧 dry-run 摘要，包含 `dry_run: true`、2 个结构和 69 个力分量；用于历史追溯，不参与规范化 evaluation。 | `$OLD/results/llpr_test_summary.json` | `authoritative` | false |
| `legacy_raw/results/LLPR/fit/fit_LLPR.log` | 旧后处理拟合运行日志，记录线性尺度和 log-space 拟合、筛选和绘图过程。 | `$OLD/results/LLPR/fit/fit_LLPR.log` | `authoritative` | false |
| `legacy_raw/results/LLPR/fit/reliability_matpes_energy_log_fit.png` | 旧 energy log-space 不确定性—误差拟合图。 | `$OLD/results/LLPR/fit/reliability_matpes_energy_log_fit.png` | `authoritative` | false |
| `legacy_raw/results/LLPR/fit/reliability_matpes_force_component_log_fit.png` | 旧 force 分量 log-space 不确定性—误差拟合图。 | `$OLD/results/LLPR/fit/reliability_matpes_force_component_log_fit.png` | `authoritative` | false |
| `legacy_raw/results/LLPR/fit/reliability_matpes_linear_fit_summary.json` | 旧线性拟合摘要；它引用的线性拟合 PNG 未随旧目录完整保留，所以分类为 incomplete，不能当作完整绘图发布包。 | `$OLD/results/LLPR/fit/reliability_matpes_linear_fit_summary.json` | `incomplete` | false |
| `legacy_raw/results/LLPR/fit/reliability_matpes_log_fit_summary.json` | 旧 log-space 拟合摘要，与两张现存 log-fit PNG 配套，记录 energy/force 拟合统计和输出路径。 | `$OLD/results/LLPR/fit/reliability_matpes_log_fit_summary.json` | `authoritative` | false |
| `legacy_raw/results/LLPR/llpr_energy_uncertainty_vs_residual.pdf` | 旧批准版 energy 不确定性—残差参考图的 PDF。它只作为旧图原样保留；当前规范化 `plots/` 为空。 | `$OLD/results/LLPR/llpr_energy_uncertainty_vs_residual.pdf` | `authoritative` | false |
| `legacy_raw/results/LLPR/llpr_energy_uncertainty_vs_residual.png` | 与上一 PDF 对应的旧 energy 参考图 PNG。 | `$OLD/results/LLPR/llpr_energy_uncertainty_vs_residual.png` | `authoritative` | false |
| `legacy_raw/results/LLPR/llpr_force_uncertainty_vs_residual.pdf` | 旧批准版 force 分量不确定性—残差参考图的 PDF。 | `$OLD/results/LLPR/llpr_force_uncertainty_vs_residual.pdf` | `authoritative` | false |
| `legacy_raw/results/LLPR/llpr_force_uncertainty_vs_residual.png` | 与上一 PDF 对应的旧 force 分量参考图 PNG。 | `$OLD/results/LLPR/llpr_force_uncertainty_vs_residual.png` | `authoritative` | false |
| `legacy_raw/results/LLPR/llpr_reference_plotting_manifest.json` | 旧参考绘图原子发布清单，记录输入、配置、统计 CSV 和四张 PNG/PDF 的哈希/输出身份。 | `$OLD/results/LLPR/llpr_reference_plotting_manifest.json` | `authoritative` | false |
| `legacy_raw/results/LLPR/llpr_reference_plotting_statistics.csv` | 旧参考图使用的分组统计表，保存 uncertainty 与绝对 residual 的分箱/汇总数据。 | `$OLD/results/LLPR/llpr_reference_plotting_statistics.csv` | `authoritative` | false |
| `legacy_raw/results/LLPR/llpr_test_full_gpu_details.npz` | 旧正式 GPU 全量测试明细，含与规范化 details 对应的 44 个数组、19374 个结构、149321 个原子、447963 个力分量。 | `$OLD/results/LLPR/llpr_test_full_gpu_details.npz` | `authoritative` | true |
| `legacy_raw/results/LLPR/llpr_test_full_gpu_run.log` | 旧正式 GPU 测试的全运行日志，记录逐结构 energy 和逐 force 分量梯度/方差求解进度及输出写入。 | `$OLD/results/LLPR/llpr_test_full_gpu_run.log` | `authoritative` | false |
| `legacy_raw/results/LLPR/llpr_test_full_gpu_small_preview.json` | 旧正式 GPU 测试的小预览，供无需加载大型 NPZ 时抽查最前面的结构记录。 | `$OLD/results/LLPR/llpr_test_full_gpu_small_preview.json` | `authoritative` | false |
| `legacy_raw/results/LLPR/llpr_test_full_gpu_summary.json` | 旧正式测试摘要：声明处理/跳过计数、总力分量、误差和不确定性统计、公式及运行元数据；迁移用它验证 details 的正式计数和来源。 | `$OLD/results/LLPR/llpr_test_full_gpu_summary.json` | `authoritative` | true |
| `legacy_raw/results/LLPR/plot_LLPR.log` | 旧 reliability 绘图运行日志，记录 details 读取、分组统计和输出写入。 | `$OLD/results/LLPR/plot_LLPR.log` | `authoritative` | false |
| `legacy_raw/results/LLPR/reliability_matpes_energy_llpr.png` | 旧 reliability energy 图；由旧 `plot_LLPR.py` 生成，并在旧绘图摘要中声明。 | `$OLD/results/LLPR/reliability_matpes_energy_llpr.png` | `authoritative` | false |
| `legacy_raw/results/LLPR/reliability_matpes_force_component_llpr.png` | 旧 reliability force-component 图；由旧 `plot_LLPR.py` 生成，并在旧绘图摘要中声明。 | `$OLD/results/LLPR/reliability_matpes_force_component_llpr.png` | `authoritative` | false |
| `legacy_raw/results/LLPR/reliability_matpes_llpr.png` | 旧目录中的额外合并图，但不在 `reliability_matpes_plot_summary.json` 声明的正式输出列表中，因此标记为 orphan。 | `$OLD/results/LLPR/reliability_matpes_llpr.png` | `orphan` | false |
| `legacy_raw/results/LLPR/reliability_matpes_plot_summary.json` | 旧 reliability 绘图摘要，只声明 energy 与 force-component 两张正式 PNG，并记录其统计/路径。 | `$OLD/results/LLPR/reliability_matpes_plot_summary.json` | `authoritative` | false |
| `legacy_raw/results/model/check_model.log` | 旧 checkpoint/model 检查运行日志，记录加载环境、模型模块遍历、readout 定位和 probe 执行过程。 | `$OLD/results/model/check_model.log` | `authoritative` | false |
| `legacy_raw/results/model/check_model_summary.json` | 旧模型检查摘要，保存 checkpoint/runtime、energy/force readout、目标字段、参数维度和 probe 元数据，用于解释 1026/3078 维曲率来源。 | `$OLD/results/model/check_model_summary.json` | `authoritative` | false |
| `legacy_raw/scripts/check_model.py` | 旧模型检查脚本：加载 checkpoint，定位 energy/force 最后一层 readout，检查参数/目标并执行小 probe，写出 log 和 summary。 | `$OLD/scripts/check_model.py` | `authoritative` | false |
| `legacy_raw/scripts/compute_Alpha.py` | 旧验证校准脚本：加载曲率矩阵，逐验证样本计算二次型与残差，按 `mean(residual²/q)` 累积 energy/force Alpha 并写摘要。 | `$OLD/scripts/compute_Alpha.py` | `authoritative` | false |
| `legacy_raw/scripts/compute_Hef.py` | 旧曲率构建脚本：在训练集上对最后一层 readout 累积 Huber/Gauss–Newton 型 energy、force 和联合曲率矩阵。 | `$OLD/scripts/compute_Hef.py` | `authoritative` | false |
| `legacy_raw/scripts/compute_LLPR.py` | 旧正式/试运行评估脚本：用固定 eta/Alpha 求解最后一层二次型，保存 44 个测试明细数组、预览、摘要和日志。 | `$OLD/scripts/compute_LLPR.py` | `authoritative` | false |
| `legacy_raw/scripts/fit_LLPR.py` | 旧结果后处理脚本：从测试 details 做线性尺度与 log-space uncertainty/error 拟合并输出摘要和图；不属于当前规范化 calibration。 | `$OLD/scripts/fit_LLPR.py` | `authoritative` | false |
| `legacy_raw/scripts/plot_LLPR.py` | 旧 reliability 绘图脚本：读取测试 details，生成 energy 与 force-component 分组/总体图及绘图摘要。 | `$OLD/scripts/plot_LLPR.py` | `authoritative` | false |
| `legacy_raw/scripts/plot_LLPR_reference.py` | 旧批准版参考绘图脚本：生成确定性 PNG/PDF、统计 CSV 和发布 manifest，并实现临时目录到最终目录的原子发布/回滚。 | `$OLD/scripts/plot_LLPR_reference.py` | `authoritative` | false |
| `legacy_raw/scripts/__pycache__/check_model.cpython-310.pyc` | `check_model.py` 的 CPython 3.10 字节码缓存；不是可维护的源代码真值。 | `$OLD/scripts/__pycache__/check_model.cpython-310.pyc` | `authoritative` | false |
| `legacy_raw/scripts/__pycache__/compute_Alpha.cpython-310.pyc` | `compute_Alpha.py` 的 CPython 3.10 字节码缓存；仅作旧目录完整保留。 | `$OLD/scripts/__pycache__/compute_Alpha.cpython-310.pyc` | `authoritative` | false |
| `legacy_raw/scripts/__pycache__/compute_Hef.cpython-310.pyc` | `compute_Hef.py` 的 CPython 3.10 字节码缓存；仅作旧目录完整保留。 | `$OLD/scripts/__pycache__/compute_Hef.cpython-310.pyc` | `authoritative` | false |
| `legacy_raw/scripts/__pycache__/compute_LLPR.cpython-310.pyc` | `compute_LLPR.py` 的 CPython 3.10 字节码缓存；仅作旧目录完整保留。 | `$OLD/scripts/__pycache__/compute_LLPR.cpython-310.pyc` | `authoritative` | false |
| `legacy_raw/scripts/__pycache__/fit_LLPR.cpython-310.pyc` | `fit_LLPR.py` 的 CPython 3.10 字节码缓存；仅作旧目录完整保留。 | `$OLD/scripts/__pycache__/fit_LLPR.cpython-310.pyc` | `authoritative` | false |
| `legacy_raw/scripts/__pycache__/plot_LLPR.cpython-310.pyc` | `plot_LLPR.py` 的 CPython 3.10 字节码缓存；仅作旧目录完整保留。 | `$OLD/scripts/__pycache__/plot_LLPR.cpython-310.pyc` | `authoritative` | false |
| `legacy_raw/tests/test_plot_LLPR_reference.py` | 旧参考绘图测试：覆盖统计/绘图输出和原子发布失败回滚，说明批准版绘图产物如何被验证。 | `$OLD/tests/test_plot_LLPR_reference.py` | `authoritative` | false |
<!-- END LEGACY RAW FILE CATALOG -->

## 规范化 NPZ 内部字段

### `curvature.npz`：2 个数组

<!-- BEGIN CURVATURE FIELD CATALOG -->
| 字段 | shape | dtype | 含义与旧数据对应 |
| --- | --- | --- | --- |
| `energy` | `(1026, 1026)` | `float64` | energy readout 的有效曲率块，等于旧 `H_E_full_run.npz["H"][0:1026, 0:1026]`。 |
| `force` | `(3078, 3078)` | `float64` | force readout 的有效曲率块，等于旧 `H_F_full_run.npz["H"][1026:4104, 1026:4104]`。 |
<!-- END CURVATURE FIELD CATALOG -->

三份旧 H 文件各自都只含一个 `(4104, 4104)` float64 数组 `H`。迁移器还检查旧联合矩阵与 energy/force 矩阵之和一致，并检查非活动块满足旧计算布局。

### `details.npz`：44 个数组

所有 44 个数组都与旧正式 `llpr_test_full_gpu_details.npz` 的同名数组在 shape、dtype 和数值上相等。记结构数 `S=19374`、原子总数 `A=149321`、力分量总数 `C=447963`，energy 的正式口径为每原子。

<!-- BEGIN EVALUATION FIELD CATALOG -->

#### 结构身份

| 字段 | shape | dtype | 层级 | 含义/公式 |
| --- | --- | --- | --- | --- |
| `structure_index` | `(19374,)` | `int64` | 结构 | 每条正式测试结构在旧数据集中的索引；当前为 0–19373。 |
| `num_atoms` | `(19374,)` | `int64` | 结构 | 每个结构的原子数；总和为 149321。 |

#### Energy：逐结构

| 字段 | shape | dtype | 层级 | 含义/公式 |
| --- | --- | --- | --- | --- |
| `energy_pred_total` | `(19374,)` | `float64` | 结构 | 模型预测总能量。 |
| `energy_true_total` | `(19374,)` | `float64` | 结构 | 参考总能量。 |
| `energy_pred_per_atom` | `(19374,)` | `float64` | 结构 | `energy_pred_total / num_atoms`。 |
| `energy_true_per_atom` | `(19374,)` | `float64` | 结构 | `energy_true_total / num_atoms`。 |
| `energy_residual` | `(19374,)` | `float64` | 结构 | 正式有效 energy 残差；由于配置使用每原子口径，它等于 `energy_pred_per_atom - energy_true_per_atom`。 |
| `energy_residual_total` | `(19374,)` | `float64` | 结构 | `energy_pred_total - energy_true_total`。 |
| `energy_residual_per_atom` | `(19374,)` | `float64` | 结构 | `energy_pred_per_atom - energy_true_per_atom`，与当前 `energy_residual` 相同。 |
| `energy_raw_var` | `(19374,)` | `float64` | 结构 | 每原子 energy 的未校准 LLPR 方差 `q_E = g_E^T (H_E + eta_E I)^(-1) g_E`。 |
| `energy_raw_std` | `(19374,)` | `float64` | 结构 | `sqrt(energy_raw_var)`。 |
| `energy_calibrated_var` | `(19374,)` | `float64` | 结构 | `alpha_E^2 * energy_raw_var`。 |
| `energy_calibrated_std` | `(19374,)` | `float64` | 结构 | `sqrt(energy_calibrated_var)`。 |
| `energy_inverse_variance` | `(19374,)` | `float64` | 结构 | `1 / energy_calibrated_var`。 |
| `energy_rigidity` | `(19374,)` | `float64` | 结构 | 旧脚本保存的 rigidity 别名，数值与 `energy_inverse_variance` 相同。 |
| `energy_raw_var_total_derived` | `(19374,)` | `float64` | 结构 | 从每原子口径换算的总能量未校准方差：`energy_raw_var * num_atoms^2`。 |
| `energy_calibrated_var_total_derived` | `(19374,)` | `float64` | 结构 | `energy_calibrated_var * num_atoms^2`。 |
| `energy_calibrated_std_total_derived` | `(19374,)` | `float64` | 结构 | `sqrt(energy_calibrated_var_total_derived)`，等价于每原子校准标准差乘 `num_atoms`。 |

#### Force：逐笛卡尔分量

| 字段 | shape | dtype | 层级 | 含义/公式 |
| --- | --- | --- | --- | --- |
| `force_offsets` | `(19375,)` | `int64` | 结构边界 | 累积偏移；第 `i` 个结构使用切片 `[offsets[i]:offsets[i+1]]`，切片长度严格为 `3*num_atoms[i]`，末值 447963。 |
| `force_structure_index` | `(447963,)` | `int64` | 力分量 | 每个扁平力分量所属结构的数据集索引。 |
| `force_component_index_within_structure` | `(447963,)` | `int64` | 力分量 | 结构内部的扁平分量编号 `0..3N-1`，原子优先、每原子 x/y/z。 |
| `force_atom_index` | `(447963,)` | `int64` | 力分量 | 结构内原子编号，等于 `component_index // 3`。 |
| `force_cartesian_index` | `(447963,)` | `int64` | 力分量 | 笛卡尔轴编号，等于 `component_index % 3`；0/1/2 对应 x/y/z。 |
| `force_pred` | `(447963,)` | `float64` | 力分量 | 模型预测的扁平 force 分量。 |
| `force_true` | `(447963,)` | `float64` | 力分量 | 参考扁平 force 分量。 |
| `force_residual` | `(447963,)` | `float64` | 力分量 | `force_pred - force_true`。 |
| `force_raw_var_component` | `(447963,)` | `float64` | 力分量 | 每个分量的未校准 LLPR 方差 `q_F = g_F^T (H_F + eta_F I)^(-1) g_F`。 |
| `force_raw_std_component` | `(447963,)` | `float64` | 力分量 | `sqrt(force_raw_var_component)`。 |
| `force_calibrated_var_component` | `(447963,)` | `float64` | 力分量 | `alpha_F^2 * force_raw_var_component`。 |
| `force_calibrated_std_component` | `(447963,)` | `float64` | 力分量 | `sqrt(force_calibrated_var_component)`。 |
| `force_inverse_variance_component` | `(447963,)` | `float64` | 力分量 | `1 / force_calibrated_var_component`。 |
| `force_rigidity_component` | `(447963,)` | `float64` | 力分量 | 旧脚本保存的 rigidity 别名，数值与 `force_inverse_variance_component` 相同。 |
| `structure_force_component_count` | `(19374,)` | `int64` | 结构 | 每个结构保存的力分量数，严格等于 `3*num_atoms`。 |

#### Force：逐原子聚合

| 字段 | shape | dtype | 层级 | 含义/公式 |
| --- | --- | --- | --- | --- |
| `atom_structure_index` | `(149321,)` | `int64` | 原子 | 每个原子所属结构的数据集索引。 |
| `atom_index_within_structure` | `(149321,)` | `int64` | 原子 | 原子在所属结构内部的编号。 |
| `force_raw_var_atom_mean` | `(149321,)` | `float64` | 原子 | 该原子 x/y/z 三个未校准分量方差的均值。 |
| `force_calibrated_var_atom_mean` | `(149321,)` | `float64` | 原子 | 该原子 x/y/z 三个校准分量方差的均值。 |
| `force_calibrated_std_atom_rms` | `(149321,)` | `float64` | 原子 | `sqrt(mean(calibrated_var_x,y,z))`，即三个校准分量标准差的 RMS。 |
| `force_calibrated_std_atom_max` | `(149321,)` | `float64` | 原子 | 该原子 x/y/z 三个校准分量标准差的最大值。 |

#### Force：逐结构聚合

| 字段 | shape | dtype | 层级 | 含义/公式 |
| --- | --- | --- | --- | --- |
| `force_raw_var_component_mean_structure` | `(19374,)` | `float64` | 结构 | 结构内全部 `3N` 个未校准 force 分量方差的均值。 |
| `force_calibrated_var_component_mean_structure` | `(19374,)` | `float64` | 结构 | 结构内全部 `3N` 个校准 force 分量方差的均值。 |
| `force_calibrated_std_component_rms_structure` | `(19374,)` | `float64` | 结构 | `sqrt(mean(calibrated_var_component))`，即结构内校准分量标准差的 RMS。 |
| `force_calibrated_std_component_max_structure` | `(19374,)` | `float64` | 结构 | 结构内全部校准 force 分量标准差的最大值。 |
| `force_inverse_variance_component_mean_structure` | `(19374,)` | `float64` | 结构 | 结构内全部 `1/calibrated_var_component` 的均值。 |

<!-- END EVALUATION FIELD CATALOG -->

## 正式数值摘要

| 指标 | 当前规范化结果 |
| --- | --- |
| 测试结构/原子/力分量 | 19374 / 149321 / 447963 |
| energy RMSE（每原子） | `0.05493578732829762` |
| force RMSE（每分量） | `0.1506777498968727` |
| 平均校准 energy 标准差 | `0.058678765166641764` |
| 平均校准 force 分量标准差 | `0.16204910666077416` |
| energy `eta` / `Alpha` / `Alpha^2` | `1e-6` / `1.1467388818005693` / `1.31501006303322` |
| force `eta` / `Alpha` / `Alpha^2` | `1e-6` / `0.2095766082027508` / `0.04392235470576932` |

## 小规模全链路与未来重算边界

`cpu_n20_fixed.yaml` 与 `cpu_n20_fit.yaml` 只用于真实小规模全链路测试，不是正式科学结果。测试覆盖固定/拟合两种模式，以及曲率计算中断后的严格恢复和与不中断运行的逐数组比较：

```bash
UPET_RUN_LLPR_N20=1 tox -e llpr-tests -- -m llpr_n20 -v
```

`gpu_full_fixed.yaml` 和 `gpu_full_fit.yaml` 是未来正式重算配置。旧成果迁移不运行它们。此前生成的小规模测试结果目录已经删除，当前 `outputs/matpes_r2_legacy` 只保留正式迁移结果。

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

energy 使用每原子残差，force 使用逐笛卡尔分量残差。矩阵、分解、二次型和校准累积均使用 float64。固定 `eta` 的条件数告警只记录风险，不会静默改变 `eta` 或添加隐藏 jitter；Cholesky 失败会直接终止。

完整迁移证据、旧结果核对结论和正式 identity 见 [MIGRATION_REPORT.md](MIGRATION_REPORT.md)。
