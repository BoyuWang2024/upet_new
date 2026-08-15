# UPET LLPR 三数据集推理与绘图设计

日期：2026-08-15

状态：已完成逐节确认，等待书面审阅

## 1. 目标

在不重新计算 LLPR 曲率的前提下，复用当前已经完成的 MATPES LLPR 结果，完成以下工作：

1. 直接使用现有 `matpes_test` 评估结果绘图；
2. 在远端使用 `mad-val-compatible.xyz` 重新计算 MAD 的 energy/force Alpha，再对 `mad-test-compatible.xyz` 计算 UQ；
3. 在远端复用现有 MATPES energy/force Alpha，对已有 `matpes_train.extxyz` 只计算 UQ；
4. 为三个数据集生成统一坐标范围的 carnet 风格能量图和力图；
5. 完整回传 MAD 结果和全部绘图，但不把体积很大的 MATPES train `details.npz` 回传本地。

本地开发固定使用当前已有的 `Plots` 分支。用户自己的未跟踪 `.idea/` 不修改、不暂存、不提交。

## 2. 已确认的基础事实

### 2.1 现有 LLPR 结果

现有正式结果位于：

```text
Uncertainty_Quantification/LLPR/outputs/matpes_r2
```

关键身份为：

| 阶段 | 身份 |
|---|---|
| 曲率 | `981cc8bdf7f8b820` |
| MATPES 校准 | `169d1d75c0dc1190` |
| MATPES test 评估 | `519237da91c72b73` |

曲率绑定的 checkpoint SHA-256 为：

```text
879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
```

当前固定参数与 Alpha 为：

| 目标 | eta | Alpha |
|---|---:|---:|
| energy | `1.0e-6` | `1.1467388818005693` |
| force | `1.0e-6` | `0.2095766082027508` |

`matpes_r2` 保持只读，不改变已有文件、身份或哈希。

### 2.2 数据文件

本地确认使用：

```text
/home/lilong/code/UQ/upet_new/data/dataset/matpes_test.extxyz
/home/lilong/code/UQ/mace_new/data/dataset/mad-val-compatible.xyz
/home/lilong/code/UQ/mace_new/data/dataset/mad-test-compatible.xyz
/home/lilong/code/UQ/mace_new/data/dataset/matpes_train.extxyz
```

用户最初提到的 `mace_new/data/dataset/matpes_train.json` 不存在，已确认改用实际存在的 `matpes_train.extxyz`。该本地文件与 UPET 构建曲率所用训练文件逐字节一致，SHA-256 为：

```text
12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec
```

远端已有：

```text
/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_train.extxyz
```

远端文件包含同样的 `348780` 个结构，但采用不同字段名和文本精度：远端使用 `id`，本地使用 `structure_id`。两者字节身份不同：

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| 本地 | `397915595` | `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec` |
| 远端 | `278976222` | `42bc5b908fbd70da740175f824fd87169dcc4bf62258e5096c3eda7e3372eae1` |

正式推理前必须进行全数据语义指纹比较。只有比较通过才复用远端文件；否则停止并改用本地确认版本。

### 2.3 远端环境

远端登录目标为 `yt_hku_psmanyam_3@121.46.19.6:6688`，隔离工作目录为：

```text
/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_llpr_codex
```

现有远端 `upet_new` 正在 `ConfidenceHead` 分支上使用，并包含用户配置与 W&B 文件。该目录不得切换分支、覆盖、提交或清理。

远端 Conda 环境为：

```text
/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new
```

Slurm 可用分区为 `ai`。登录节点无法访问 NVIDIA 驱动，因此所有模型计算必须提交到计算节点。

该 SSH 入口可能分流到多个后端。自动化连接只接受已经由 MobaXterm 确认的 ED25519 指纹：

```text
SHA256:SPxE4NOiUvxqOCvLRGKSMb1URKFcLmVJoKiJSOVvH2g
```

命中其他指纹时只重试，不关闭主机校验，也不自动信任新的指纹。

## 3. 范围与非目标

### 3.1 本次范围

- 增加可验证、可物化的曲率/校准阶段复用配置；
- 新增 MAD test 与 MATPES train 正式配置；
- 复用现有 LLPR 校准、评估、分片恢复和清单验证代码；
- 将现有绘图模块改造成支持一个或多个评估结果的 carnet 风格绘图；
- 本地单元测试、真实小规模远端测试、正式远端计算、绘图、验证和按边界回传；
- 更新 LLPR 中文 README。

### 3.2 非目标

- 不重新计算曲率；
- 不为 MAD 拟合 eta；
- 不为 MATPES train 重新计算 Alpha；
- 不修改 `outputs/matpes_r2`；
- 不修改 carnet 代码或结果；
- 不修改远端 ConfidenceHead 工作区；
- 不回传 MATPES train 的完整 `details.npz`；
- 不生成 HE、HF、可靠性图或标准化残差 CDF。

## 4. 总体架构

现有 `matpes_r2` 是唯一可信的基础结果。远端隔离工作区包含代码、必要输入、基础结果和新结果：

```text
upet_new_llpr_codex/
├── data/
│   ├── checkpoint/
│   └── dataset/
└── Uncertainty_Quantification/
    ├── LLPR/
    │   ├── configs/
    │   ├── outputs/
    │   │   ├── matpes_r2/
    │   │   ├── mad_test/
    │   │   └── matpes_train/
    │   └── run/
    └── Plots/
        └── LLPR/
```

三个结果的依赖关系为：

```text
现有曲率 981cc8bdf7f8b820
├── 现有 MATPES Alpha 169d1d75c0dc1190
│   ├── 现有 matpes_test UQ
│   └── 新计算 matpes_train UQ
└── mad-val-compatible
    └── 固定 eta=1e-6，重新计算 MAD Alpha
        └── 新计算 mad-test-compatible UQ
```

正式计算继续使用现有 `calibrate`、`evaluate`、`verify` 和分片恢复实现，不建立第二套推理程序。

## 5. 可复用阶段设计

### 5.1 配置形式

`LLPRConfig` 增加可选的 `reuse`：

```yaml
reuse:
  curvature:
    path: Uncertainty_Quantification/LLPR/outputs/matpes_r2/curvature/981cc8bdf7f8b820
    identity: 981cc8bdf7f8b820
  calibration:
    path: Uncertainty_Quantification/LLPR/outputs/matpes_r2/calibration/169d1d75c0dc1190
    identity: 169d1d75c0dc1190
```

`reuse` 中的相对路径统一以仓库根目录解析；运行时可以接受绝对路径，但写入正式结果清单时不保存该绝对来源路径。

约束如下：

- `reuse.curvature` 存在时，`build` 数据路径可以省略，且不得调用 `run_build`；
- `reuse.calibration` 存在时，必须同时存在 `reuse.curvature`，`calibration` 数据路径可以省略，且不得调用 `run_calibrate`；
- `test` 数据路径始终必填；
- MAD 配置只设置 `reuse.curvature`；
- MATPES train 配置同时设置 `reuse.curvature` 和 `reuse.calibration`；
- 现有配置不设置 `reuse` 时保持原行为。

### 5.2 验证与物化

复用阶段不能只信任路径。运行时依次检查：

1. 来源路径为普通目录，且不位于目标目录内部；
2. `manifest.json` 阶段类型与配置相符；
3. 身份与配置中的 `identity` 完全一致；
4. 所有声明文件存在且 SHA-256 正确；
5. 曲率清单中的 checkpoint SHA 与当前 checkpoint 一致；
6. 校准清单绑定的曲率身份与当前选择的曲率一致；
7. 公式版本、schema 版本和 readout layout 与当前代码可接受版本一致。

验证通过后，将阶段物化到新结果目录：

- 同一文件系统优先创建硬链接，避免重复占用磁盘；
- 跨文件系统退化为逐文件复制；
- 只复制清单声明的正式文件；
- 先写唯一 staging 目录，重新验证后原子改名；
- 目标已存在且完整一致时直接复用；
- 目标已存在但不一致时停止，绝不覆盖；
- 正式结果清单只记录阶段身份，不记录本机绝对来源路径。

## 6. 两条正式计算链路

### 6.1 MAD

MAD 配置使用：

- 复用曲率 `981cc8bdf7f8b820`；
- calibration 数据为 `mad-val-compatible.xyz`；
- test 数据为 `mad-test-compatible.xyz`；
- energy eta 与 force eta 均固定为 `1.0e-6`；
- 输出 experiment 为 `mad_test`。

`evaluate` 先调用校准阶段。校准阶段直接读取复用曲率，在 MAD validation 上计算新的 energy/force Alpha，再由评估阶段对 MAD test 计算预测、残差和校准后 UQ。

### 6.2 MATPES train

MATPES train 配置使用：

- 复用曲率 `981cc8bdf7f8b820`；
- 复用校准 `169d1d75c0dc1190`；
- test 数据为通过语义审核的远端 `matpes_train.extxyz`；
- 输出 experiment 为 `matpes_train`。

`evaluate` 直接读取两个复用阶段，只计算 MATPES train 预测、残差和 UQ。不得进入构建或校准代码路径。

## 7. MATPES train 语义审核

字节 SHA 不同不能作为同一文件处理，但字段名和打印精度不同也不应自动判为不同数据。因此对本地和远端文件分别流式计算规范化语义指纹。

指纹按文件顺序包含：

- 结构序号和结构数；
- `id` 或 `structure_id` 规范化后的结构标识；
- 原子数、原子序数及原子顺序；
- PBC；
- cell 与 positions，按 `1e-6 Å` 量化；
- 总能量，按 `1e-6 eV` 量化；
- 力，按 `1e-6 eV/Å` 量化。

stress 和不参与 LLPR 的文本元数据不进入指纹。双方规范化 SHA-256 必须完全相同。任何字段缺失、顺序变化或指纹不一致都使审核失败；失败后停止正式任务并改为上传本地确认版本，不降低容差继续尝试。

## 8. 绘图设计

### 8.1 输入

绘图配置包含三个已完成评估：

| 标签 | 输入 |
|---|---|
| `matpes_test` | 现有 `matpes_r2` 评估 |
| `mad_test` | 新 MAD 评估 |
| `matpes_train` | 新 MATPES train 评估 |

每个输入必须唯一解析到一个完成且通过哈希验证的 `details.npz`。

绘图字段为：

- 能量：`energy_calibrated_std` 与 `abs(energy_residual)`，单位 `eV/atom`；
- 力：`force_calibrated_std_component` 与 `abs(force_residual)`，单位 `eV/Å`。

### 8.2 分析与视觉参数

沿用 carnet 正式 LLPR 绘图逻辑：

| 参数 | 值 |
|---|---|
| grid size | `160` |
| Gaussian sigma | `1.2` |
| contour masses | `[0.5, 0.7, 0.85, 0.95, 0.99]` |
| scatter maximum | `20000` |
| random seed | `20260714` |
| log margin | `0.05` |
| figure size | `[7.0, 7.0]` |
| color | `#f28e2b` |
| scatter size/alpha | `12.0 / 0.04` |
| DPI | `300` |
| formats | `png`, `pdf` |

三张能量图使用三套能量数据联合得到的方形对数范围；三张力图同理。每图包含：

- 确定性抽样散点；
- 平滑概率质量等高线；
- 黑色 `y=x` 虚线；
- carnet 同款灰色区域；
- Spearman 与 log10 Pearson 标注；
- 明确的数据集、LLPR、Energy/Force 标题和物理单位。

非有限值、零值和负值按 carnet 规则分类排除，并将排除数量、相关系数、分位数和 uncertainty/error 比率写入统计结果。

### 8.3 输出

最终目录严格包含 14 个文件：

```text
Uncertainty_Quantification/Plots/LLPR/
├── llpr_matpes_test_energy_uncertainty_vs_residual.pdf
├── llpr_matpes_test_energy_uncertainty_vs_residual.png
├── llpr_matpes_test_force_uncertainty_vs_residual.pdf
├── llpr_matpes_test_force_uncertainty_vs_residual.png
├── llpr_mad_test_energy_uncertainty_vs_residual.pdf
├── llpr_mad_test_energy_uncertainty_vs_residual.png
├── llpr_mad_test_force_uncertainty_vs_residual.pdf
├── llpr_mad_test_force_uncertainty_vs_residual.png
├── llpr_matpes_train_energy_uncertainty_vs_residual.pdf
├── llpr_matpes_train_energy_uncertainty_vs_residual.png
├── llpr_matpes_train_force_uncertainty_vs_residual.pdf
├── llpr_matpes_train_force_uncertainty_vs_residual.png
├── plotting_statistics.csv
└── plotting_manifest.json
```

绘图先发布到 staging。清单记录三套评估身份、输入文件哈希、共享坐标范围、统计结果和全部输出文件哈希，验证成功后才原子替换最终目录。

现有单结果可靠性图/CDF 模式改为通用的一个或多个数据集 carnet 风格主图模式。单数据集配置仍受支持，但本次正式配置固定包含三个数据集。

## 9. 远端执行设计

### 9.1 部署隔离

本地 `Plots` 分支的已提交代码部署到新的 `upet_new_llpr_codex`。基础 `matpes_r2`、两份 MAD compatible 数据和必要 checkpoint 链接单独传输或建立。远端原有 `upet_new` 不作为运行目录。

每次部署记录并核对本地提交 SHA。运行目录中的 tracked 文件必须对应同一提交；运行数据和结果保持 Git ignored。

### 9.2 Slurm

登录节点只提交任务。正式脚本使用：

- partition `ai`；
- 1 node、1 task、8 CPUs；
- 64 GiB 内存；
- `CUDA_VISIBLE_DEVICES=0`；
- 绝对路径调用远端 `upet_new` 环境；
- `set -euo pipefail` 和 `PYTHONUNBUFFERED=1`；
- 每项任务独立日志目录。

当前 Slurm 未登记 GPU GRES，因此脚本不填写 `--gres`。小规模任务开始时必须在分配节点执行 `nvidia-smi` 和 CUDA 可用性检查；失败则停止，不提交正式任务。

作业依赖顺序为：

```text
MAD Alpha + MAD UQ
        ↓ afterok
MATPES train UQ
        ↓ afterok
统一绘图与完整验证
```

## 10. 失败恢复与写入安全

- 所有来源在读取前验证，所有目标通过 staging 原子发布；
- 不允许覆盖不一致的同名结果；
- MAD 校准继续使用 `progress.npz`；
- 评估继续使用完整结构分片和 `progress.json`；
- 失败时保留进度、分片和日志；
- 恢复时重新验证所有已记录分片及 next index；
- 阶段完成并通过正式验证后才删除进度和中间分片；
- 绘图必须等待三个评估均完整验证；
- 同步回本地后再次核对文件哈希。

## 11. 测试设计

### 11.1 本地自动测试

- `reuse` 合法与非法配置组合；
- 复用曲率时禁止调用 `run_build`；
- 复用校准时禁止调用 `run_calibrate`；
- checkpoint、身份、公式、schema、layout 或清单哈希不匹配时拒绝；
- 硬链接物化与跨文件系统复制回退；
- 已存在一致目标复用、不一致目标拒绝；
- 多数据集输入解析与唯一评估选择；
- energy/force 字段映射；
- 共享坐标范围；
- 确定性抽样、等高线、过滤和统计；
- staging 故障不污染正式目录；
- 正式输出严格为 14 个非空文件；
- 现有 LLPR 测试、Ruff、mypy 和文档检查。

### 11.2 远端小规模测试

正式任务前创建临时完整结构子集：

- MAD val/test 少量结构：验证复用曲率、固定 eta、新 Alpha、UQ 和恢复；
- 远端 `matpes_n20.extxyz`：验证复用曲率与 MATPES Alpha后仅执行 UQ；
- 验证 CUDA、checkpoint、结果清单、数值和分片恢复。

小规模验证完成后删除测试结果和临时子集，只保留日志到正式验收结束；正式结果不受影响。

## 12. 正式验收标准

### 12.1 计算结果

- `matpes_r2` 的 11 个现有文件和哈希保持不变；
- MAD 曲率身份为 `981cc8bdf7f8b820`；
- MAD energy/force eta 均为 `1.0e-6`；
- MAD 新 Alpha 有限且严格大于零；
- MATPES train 曲率身份为 `981cc8bdf7f8b820`；
- MATPES train 校准身份为 `169d1d75c0dc1190`；
- MATPES energy/force Alpha 与现有值完全一致；
- 结构数、原子数和力分量数与目标数据一致；
- 预测、残差、方差和标准差均有限；
- 方差和标准差严格大于零；
- 结构顺序、force offsets 和数组长度一致；
- 每套远端完整结果通过 full verify。

### 12.2 绘图结果

- 正式目录严格为 14 个文件；
- PNG、PDF、CSV 和清单均非空；
- 三张能量图共享完全相同的坐标范围；
- 三张力图共享完全相同的坐标范围；
- CSV 行数、样本计数、排除计数和清单统计一致；
- 所有输入和输出哈希验证通过。

### 12.3 回传与保留边界

本地最终保留：

```text
LLPR/outputs/matpes_r2/           # 现有完整结果
LLPR/outputs/mad_test/            # 新增完整结果
LLPR/summaries/matpes_train/      # 汇总、预览和身份清单
Plots/LLPR/                       # 14 个正式绘图文件
```

远端最终保留完整：

```text
LLPR/outputs/matpes_r2/
LLPR/outputs/mad_test/
LLPR/outputs/matpes_train/
Plots/LLPR/
```

本地不出现 MATPES train `details.npz`。远端完整文件必须仍存在并通过验证。本地 README 明确说明完整逐项结果的远端保留位置。

不保留本地或远端小规模测试结果、临时子集、staging、完成后的分片和无用缓存。Slurm 正式日志保留在远端隔离工作区，不作为发布内容回传或提交。

## 13. 代码与文件变更边界

预计修改或新增：

- `LLPR/llpr/config.py`：阶段复用配置与条件校验；
- `LLPR/llpr/artifacts.py`：验证和原子物化正式阶段；
- `LLPR/llpr/calibration.py`：解析复用曲率；
- `LLPR/llpr/inference.py`：解析复用曲率与校准；
- `LLPR/llpr/plotting.py`：多数据集 carnet 风格分析和渲染；
- `LLPR/configs/gpu_mad_test_fixed.yaml`；
- `LLPR/configs/gpu_matpes_train_fixed.yaml`；
- `LLPR/configs/plot_three_datasets.yaml`；
- 对应单元测试和远端小规模测试；
- `LLPR/README.md`；
- `Uncertainty_Quantification/Plots/LLPR/` 正式生成结果；
- `LLPR/summaries/matpes_train/` 本地摘要结果。

远端 Slurm 脚本属于隔离工作区运行资产，不提交到发布分支，避免把服务器专用路径写入公开代码。
