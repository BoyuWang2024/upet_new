# UPET LLPR 旧成果迁移与新计算链设计

日期：2026-07-26
状态：设计已获用户确认，等待书面规格复核
目标目录：`Uncertainty_Quantification/LLPR/`

## 1. 背景

旧目录：

```text
/home/lilong/code/UQ/upet/UQ_LLPR
```

包含针对旧 UPET 运行环境编写的 LLPR 代码，以及已经完成的训练集曲率、
验证集 Alpha 校准、测试集 LLPR 和绘图结果。新仓库：

```text
/home/lilong/code/UQ/upet_new
```

使用当前 UPET API，数据和 checkpoint 位于仓库的 `data/` 下。迁移目标不是只做
旧结果查看器，而是同时完成两件事：

1. 在 `Uncertainty_Quantification/LLPR/` 中实现可重新执行
   `build -> calibrate -> evaluate -> plot` 的完整新代码；
2. 本次不重新执行完整 train/validation/test 计算，而是把经过审计的旧成果导入
   新代码使用的统一产物协议。

本次允许使用 `matpes_n20.extxyz` 执行小规模全链路测试，确保新计算链能够运行。

## 2. 已确认的输入身份

旧 checkpoint 与新仓库 checkpoint 的 SHA-256 相同：

```text
879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
```

旧固定 `eta=1e-6` 求解在算术上成功，全部正式 q 和方差为正，但活动块条件数约为：

```text
energy: 1.135e15
force:  1.381e12
```

因此旧结果必须标记为 `legacy_fixed_ridge`。旧 H 阶段的 Cholesky 诊断曾使用
`H + eta^2 I`，而 Alpha/LLPR 正式求解使用 `H + eta I`。该差异只影响旧诊断字段，
不影响保存的原始 H 或正式 Alpha/LLPR；统一产物不会把旧 Cholesky 布尔值作为正式
求解证明，而会基于 `H + eta I` 重新生成只读诊断。

数据身份如下：

| Split | 结构数 | SHA-256 |
| --- | ---: | --- |
| train | 348,780 | `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec` |
| validation | 19,370 | `5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef` |
| test | 19,374 | `1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d` |

旧正式测试结果包含：

- 19,374 个结构；
- 149,321 个原子；
- 447,963 个力分量；
- 0 个跳过结构。

旧核心代码和结果由旧仓库提交
`4e0614c8309ed707c9cf93d04dab75274221119a` 引入。后续提交主要补充绘图产物。

## 3. 已确认的设计决策

1. 采用“统一新产物协议 + 隔离的旧结果导入器”。
2. 新计算代码完整支持 build、calibrate、evaluate、run 和 plot。
3. 本次只对 n20 执行模型计算，不对完整数据重新计算。
4. 旧正式成果按固定 `eta=1e-6` 原样迁移。
5. 新代码支持 fixed 和 fit 两种 eta 模式。
6. fit 模式只使用 validation，测试集不参与 eta 或 Alpha 的选择。
7. energy 和 force 分别拟合 `eta_energy` 与 `eta_force`。
8. fit 模式为每个 eta 候选闭式校准 Alpha，并以 validation Gaussian NLL
   最小为选择标准。
9. 旧原始代码、配置、日志和结果保留为不可变审计副本。
10. 正式消费代码只读取统一产物，不直接读取旧格式。

## 4. 非目标

本次不：

- 对完整 train/validation/test 运行模型；
- 改变旧 H、Alpha、eta 或 LLPR 数值；
- 用旧结果推造 fitted-eta 正式结果；
- 把 n20 结果解释为正式科学统计；
- 修改旧仓库；
- 将 LLPR 首版并入 `src/upet` 的公共 API；
- 自动选择隐藏 jitter 或在失败后静默更换 eta；
- 维护两套长期并行的新旧消费接口。

## 5. 总体架构

```text
Uncertainty_Quantification/LLPR/
├── README.md
├── configs/
│   ├── cpu_n20_fixed.yaml
│   ├── cpu_n20_fit.yaml
│   ├── gpu_full_fixed.yaml
│   ├── gpu_full_fit.yaml
│   ├── import_legacy.yaml
│   └── plot_legacy.yaml
├── llpr/
│   ├── __init__.py
│   ├── __main__.py
│   ├── config.py
│   ├── checkpoint.py
│   ├── readout.py
│   ├── data.py
│   ├── observables.py
│   ├── curvature.py
│   ├── ridge.py
│   ├── calibration.py
│   ├── inference.py
│   ├── artifacts.py
│   ├── legacy.py
│   ├── plotting.py
│   └── cli.py
├── tests/
└── outputs/
```

两条数据路径汇合到同一正式产物协议：

```text
当前 UPET + 数据 -- build/calibrate/evaluate --+
                                                +-- 统一产物 -- verify/plot
旧代码和结果 ----------- import-legacy ---------+
```

旧字段、旧绝对路径和特殊兼容逻辑只能出现在 `legacy.py`。计算核心、绘图和正式
产物读取器不包含旧格式分支。

## 6. 模块职责

### `config.py`

- 解析并验证 YAML；
- 将相对路径固定解析为仓库根路径；
- 禁止 basename 搜索和依赖当前工作目录的路径语义；
- 为 resolved config 生成稳定哈希；
- 验证 fixed/fit 配置互斥且完整。

### `checkpoint.py`

- 使用当前 UPET API 加载 checkpoint；
- 验证 checkpoint SHA 和训练损失元数据；
- 规范化当前 API 的 `non_conservative_force` 名称；
- 不依赖已弃用的旧输出名完成正式计算。

### `readout.py`

- 从模型结构发现 energy 与 non-conservative-force 末层参数；
- 把 weight 和 bias 全部纳入布局；
- 固定参数名、shape、顺序、offset 和 layout SHA；
- 当前预期活动维度为 energy 1026、force 3078。

### `data.py`

- 确定性读取 extxyz；
- 保留 structure index、原子数、cell、PBC、参考总能量和参考力；
- 计算数据 SHA 和结构计数；
- build/calibration/test 均禁止 shuffle。

### `observables.py`

- 计算总能量、每原子能量、力以及对应的末层 Jacobian；
- 按 readout manifest flatten；
- 支持可配置的 force-component chunk；
- 检查输出和梯度的有限性。

### `curvature.py`

- 严格实现已确认的 Huber 曲率和归一化；
- 以 float64 流式累计 energy 与 force 活动块；
- 每个完整结构作为最小提交单位；
- 不把 eta 写入原始曲率；
- 输出谱、条件数、范数、计数和对称性诊断。

### `ridge.py`

- 处理 fixed eta；
- 为 fit 模式生成确定性、尺度感知的 eta 候选；
- 构造 Cholesky；
- 计算 q 和条件数诊断；
- 不执行隐藏 jitter 或未配置的回退。

### `calibration.py`

- 在 validation 上校准 Alpha；
- fixed 模式保留显式 eta；
- fit 模式分别选择 energy/force eta；
- 保存全部候选和排除原因；
- 不读取 test。

### `inference.py`

- 在 test 上计算 residual、q、variance、std 和 rigidity；
- energy 使用每原子层级；
- force 使用逐笛卡尔分量层级；
- 生成原子级和结构级 force 汇总。

### `artifacts.py`

- 定义 schema 和公式版本；
- 负责 hash、identity、状态、临时写入、原子替换和恢复；
- 拒绝消费 `status != complete` 或身份不匹配的上游产物；
- 提供 metadata/full 两种 verify。

### `legacy.py`

- 复制并清点旧原始文件；
- 验证旧 H、Alpha、LLPR、日志和绘图 manifest；
- 把旧正式结果转换成统一产物；
- 不导入 checkpoint loader，不调用模型，不计算 Jacobian；
- 对相同导入保持幂等，对不同身份拒绝覆盖。

### `plotting.py`

- 只读取完成的统一 evaluation 产物；
- 不加载 checkpoint；
- 不重新拟合 eta 或 Alpha；
- 原子性发布图片、统计 CSV 和 plotting manifest。

### `cli.py` 与 `__main__.py`

提供：

```text
build
calibrate
evaluate
run
import-legacy
verify
plot
```

CLI 只负责编排、日志和退出码，不包含数学实现。

## 7. 数学定义

### 7.1 Energy 曲率

对结构 `s`，原子数为 `N_s`：

```text
g_E,s = grad(E_pred,s / N_s)
r_E,s = E_pred,s / N_s - E_ref,s / N_s
c_E,s = 1 if abs(r_E,s) <= delta_E else 0
H_E   = sum_s w_E c_E,s outer(g_E,s, g_E,s)
```

当前正式配置：

```text
w_E     = 1.0
delta_E = 0.015 eV/atom
```

新代码显式乘入 `w_E`。当前 `w_E=1.0`，因此这项修正不改变旧结果。

### 7.2 Force 曲率

对结构 `s` 的第 `i` 个力分量：

```text
g_F,s,i = grad(F_pred,s,i)
r_F,s,i = F_pred,s,i - F_ref,s,i
c_F,s,i = 1 if abs(r_F,s,i) <= delta_F else 0

H_F = sum_s [w_F / (3 N_s)] *
      sum_i c_F,s,i outer(g_F,s,i, g_F,s,i)
```

当前正式配置：

```text
w_F     = 0.1
delta_F = 0.01 eV/angstrom
```

Energy 和 force 使用不重叠的活动参数块。联合矩阵只作为
`block_diag(H_E, H_F)` 的派生视图，不是必需持久化产物。

### 7.3 LLPR

目标 `t` 为 energy 或 force：

```text
A_t        = H_t + eta_t I
q_t        = g_t^T solve(A_t, g_t)
alpha_t^2  = mean_validation(residual_t^2 / q_t)
variance_t = alpha_t^2 q_t
std_t      = sqrt(variance_t)
rigidity_t = 1 / variance_t
```

矩阵累计、谱分解、Cholesky、q 和校准统计使用 float64。

## 8. Eta 模式

### 8.1 Fixed

配置允许一个标量同时应用于 energy 和 force，也允许分别指定：

```yaml
calibration:
  ridge:
    mode: fixed
    max_condition_number: 1.0e10
    eta:
      energy: 1.0e-6
      force: 1.0e-6
```

固定模式：

- 不因条件数超限改变 eta；
- 条件数超限时记录 warning；
- Cholesky 失败时直接失败；
- 不增加隐藏 jitter；
- 当前旧正式结果使用 `eta_energy=eta_force=1e-6`。

### 8.2 Fit

```yaml
calibration:
  ridge:
    mode: fit
    max_condition_number: 1.0e10
    fit:
      candidate_multipliers: [1, 3, 10, 30, 100, 300, 1000]
      score: gaussian_nll
```

对 energy 和 force 分别执行候选生成与选择。对对称活动矩阵的最小、最大
特征值 `mu_min`、`mu_max`，设：

```text
spectral_eps = eps_float64 * max(1, abs(mu_max))
eta_pd       = max(0, -mu_min) + spectral_eps
eta_cond     = max(0, (mu_max - kappa * mu_min) / (kappa - 1))
eta_floor    = nextafter(max(eta_pd, eta_cond), +infinity)
```

候选为去重、升序排列的：

```text
eta_floor * candidate_multiplier
```

每个候选在完整 validation 上独立计算 q 和闭式 Alpha。评分为：

```text
NLL = 0.5 * mean[
    log(2 pi alpha^2 q) + residual^2 / (alpha^2 q)
]
```

选择规则：

1. 排除非有限、非正 q、Cholesky 失败或超过配置条件数的候选；
2. 选择 validation Gaussian NLL 最小的候选；
3. NLL 数值相同时选择较小 eta；
4. 保存所有候选的 eta、Alpha、NLL、coverage、条件数和排除原因；
5. 没有有效候选时失败；
6. test 只消费已选择的 eta/Alpha，不参与选择。

固定 calibration ID 显式包含两个 eta。拟合 calibration ID 包含候选配置哈希，
summary 中保存最终两个 eta 和 Alpha。

## 9. 正式产物协议

目录：

```text
outputs/<experiment>/<checkpoint-sha12>/
├── manifest.json
├── resolved_configs/
│   └── <config-hash>.yaml
├── legacy_raw/
│   ├── source_inventory.json
│   ├── scripts/
│   ├── configs/
│   ├── logs/
│   ├── results/
│   └── tests/
├── curvature/
│   ├── manifest.json
│   ├── energy_block.npz
│   ├── force_block.npz
│   └── diagnostics.json
├── calibration/
│   └── <calibration-id>/
│       ├── manifest.json
│       ├── summary.json
│       └── candidates.json
├── evaluation/
│   └── <calibration-id>/
│       ├── manifest.json
│       ├── details.npz
│       ├── summary.json
│       └── preview.json
└── plots/
    └── <calibration-id>/
        ├── *.png
        ├── *.pdf
        ├── plotting_statistics.csv
        └── plotting_manifest.json
```

`candidates.json` 在 fixed 模式保存单个显式候选，在 fit 模式保存完整候选集合。
大型矩阵和逐样本数据使用压缩 NPZ，摘要和身份使用 JSON，绘图统计使用 CSV。

根 `manifest.json` 绑定运行根身份和可用阶段，不保存一个含糊的“当前 calibration”。
它至少绑定：

- artifact schema 和公式版本；
- `origin: legacy_import` 或 `origin: recomputed`；
- checkpoint、train、validation、test SHA；
- 参数名称、shape、顺序、offset 和 layout SHA；
- Huber 阈值、loss weight 和归一化定义；
- 结构、原子和力分量计数；
- dtype、device、Jacobian backend 和最终 chunk；
- 原始文件哈希和旧 Git commit；
- 新代码 Git commit 和已使用的 resolved config hash 列表；
- `legacy_fixed_ridge` 风险；
- `status: complete`。

各阶段 `manifest.json` 只绑定该阶段及其上游依赖：

- curvature manifest：checkpoint、train、公式、readout、dtype/backend 和曲率配置；
- calibration manifest：curvature identity、validation、eta 模式、候选配置、最终 eta
  和 Alpha；
- evaluation manifest：curvature/calibration identity、test 和逐样本输出 hash；
- plotting manifest：evaluation identity、绘图配置、统计和图片 hash。

每个阶段使用独立 identity hash。curvature identity 明确排除 calibration、evaluation
和 plotting 配置，因此 fixed 与 fit 可以安全复用同一 curvature。任何影响某阶段
数值的配置必须进入该阶段 identity。

`outputs/` 是本地计算和迁移产物，不把大型数值结果提交到 Git。代码、配置、测试、
README 和设计文档进入版本控制。

## 10. Legacy 导入

### 10.1 原始副本

`legacy_raw` 保存旧代码、配置、日志、结果和测试的字节级副本。inventory 为每个
文件记录：

- 原始绝对路径；
- 迁移后的相对路径；
- 大小；
- SHA-256；
- 分类；
- 是否属于正式统一产物的来源。

checkpoint 与数据不重复复制，只记录新仓库路径和 SHA。

### 10.2 文件分类

正式来源：

- 全量 `H_E/H_F/H_EF`；
- 全量 validation Alpha；
- 全量 test LLPR details/summary；
- 通过原 manifest hash 验证的 reference plotting 产物。

归档但不作为正式结果：

- 两结构 CPU dry-run，分类为 `legacy_smoke`；
- 引用缺失 PNG 的 linear-fit summary，分类为 `incomplete`；
- 没有当前生成来源的组合图，分类为 `orphan`。

### 10.3 转换验证

导入必须验证：

1. checkpoint/data SHA；
2. 4104 维旧布局及 1026/3078 块边界；
3. 矩阵有限性、对称性、块隔离和 `H_EF=H_E+H_F`；
4. Alpha、eta、样本计数和校准公式；
5. test structure/atom/component 索引和 `3N` offsets；
6. residual、raw variance、calibrated variance、std 和 rigidity；
7. summary 与 details 的统计一致性；
8. plotting manifest 输入和输出哈希。

通过后，从旧全矩阵提取 energy/force 活动块，写入统一 curvature。旧全矩阵仍保留
在 `legacy_raw`。转换不加载模型。

旧结果缺少全部 validation force q/residual 明细，因此只能导入 fixed 模式，不能
生成 fitted-eta 正式结果。

### 10.4 原子性和幂等性

- 所有内容先写同文件系统临时目录；
- 全部验证成功后原子替换正式目录；
- 相同 identity 再导入时重新验证后无操作；
- 已有不同 identity 或 hash 时拒绝覆盖；
- 失败时不留下 `complete` 产物。

## 11. 阶段流与恢复

| 命令 | 输入 | 输出 | 运行模型 |
| --- | --- | --- | --- |
| `build` | checkpoint + train | curvature | 是 |
| `calibrate` | curvature + validation | calibration | 是 |
| `evaluate` | curvature + calibration + test | evaluation | 是 |
| `run` | 完整配置 | 顺序执行前三阶段 | 是 |
| `import-legacy` | 旧文件 | 统一产物 | 否 |
| `verify` | 统一产物 | 校验报告 | 否 |
| `plot` | evaluation | 图片和统计 | 否 |

恢复规则：

- build 周期性保存完整结构边界上的 H 和计数；
- calibration 保存结构位置和所有 eta 候选累计统计；
- evaluate 先写结构级 NPZ 分片，完成后合并；
- energy 与同结构全部 `3N` force 分量构成一个提交单元；
- 已提交结构不重复累计；
- 恢复前严格核对对应阶段的完整 identity；
- identity 不一致时拒绝恢复。

相同曲率身份下，fixed 与 fit calibration 复用同一 curvature。

## 12. 错误处理

默认策略为严格失败，不跳过样本。错误信息包含阶段、split、structure ID，并在适用
时包含原子索引和笛卡尔方向。

- 缺失标签、NaN/Inf、布局变化：失败；
- 矩阵不对称或块关系错误：失败；
- fixed eta 条件数超限：警告并记录，不改变 eta；
- fixed eta Cholesky 失败：失败；
- 单个 fit 候选无效：记录排除原因；
- 所有 fit 候选无效：失败；
- GPU OOM：允许缩小配置的 force chunk 后重试，最终 chunk 写入 manifest；
- 目标 identity 不同：拒绝覆盖；
- plotting 输入或统计不一致：不发布任何新图。

`verify metadata` 检查 schema、hash、状态和依赖关系。`verify full` 还加载全部矩阵
和 details，重新检查公式、索引和摘要。`import-legacy` 强制 full 验证。

## 13. 绘图

绘图使用已校准的原始 std，不在 test 上重新拟合或缩放。至少生成：

- energy 每原子 uncertainty 与绝对残差图；
- force component uncertainty 与绝对残差图；
- reliability/coverage 诊断；
- 标准化残差诊断；
- plotting statistics 和 manifest。

导入后的旧结果重新绘图时，要求输入统计和关键数值与旧 reference plotting 一致；
由于 Matplotlib 等库版本可能不同，不要求 PNG 字节完全一致。

## 14. 测试

### 14.1 单元测试

- readout weight/bias 发现和稳定布局；
- energy 每原子 Jacobian；
- force `1/(3N)` 与 `0.1` 只应用一次；
- Huber 阈值内外曲率；
- energy/force 块和联合关系；
- fixed eta 的 Cholesky q、Alpha、variance、rigidity；
- energy/force 分别 fit eta；
- Gaussian NLL、tie-break 和候选排除；
- 通过测试标记证明 fit 不读取 test；
- artifact identity、原子写入、恢复和幂等；
- 篡改旧矩阵、summary 或 hash 时导入失败。

新增 `tox -e llpr-tests`，并把 LLPR 源码和测试加入 lint、format 和 mypy 范围。
命令从 `conda activate upet_new` 后执行。

### 14.2 n20 全链路

`data/dataset/matpes_n20.extxyz` 同时作为 build/calibration/test，只验证功能。

执行：

```text
build
calibrate fixed
evaluate fixed
plot fixed
calibrate fit
evaluate fit
plot fit
verify full
```

验收：

- 当前 checkpoint/API 成功加载；
- readout 布局正确；
- 矩阵、q、Alpha、variance、std 全部有限；
- 正式方差严格为正；
- 不跳过样本；
- fixed 保持两个 eta 都为 `1e-6`；
- fit 保存分别选择的 eta 和全部候选；
- fixed/fit 复用 curvature；
- 索引完整；
- plotting 不加载模型；
- 恢复结果与连续运行一致；
- 重复执行不重复累计。

### 14.3 旧正式结果

验收：

- 旧源目录未修改；
- 原始文件 inventory 完整；
- checkpoint/data SHA 一致；
- 导入不调用模型；
- H 矩阵不变量通过；
- 精确保留：

```text
alpha_energy = 1.1467388818005693
alpha_force  = 0.2095766082027508
```

- 保留 19,374 个结构、149,321 个原子、447,963 个力分量；
- 导入后的逐样本数值与旧 NPZ 一致；
- `verify full` 通过；
- 新图统计与旧 reference 统计一致；
- 第二次导入为无操作；
- incomplete/orphan 不进入正式结果；
- manifest 明确记录固定 eta 的条件数风险。

## 15. 本次实施顺序

1. 编写测试和模块化计算核心；
2. 完成 fixed eta；
3. 完成 fit eta；
4. 完成统一 artifact 层；
5. 完成 legacy importer；
6. 运行快速单元测试；
7. 在 `upet_new` 环境运行 n20 fixed/fit 全链路；
8. 导入旧正式结果；
9. 对导入结果执行 full verify；
10. 从导入结果重新绘图；
11. 确认未执行完整数据模型计算；
12. 更新 README 和交付说明。

具体文件级步骤和测试命令将在本设计再次获得用户确认后，由 writing-plans 流程
生成实施计划。
