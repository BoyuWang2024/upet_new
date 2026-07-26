# UPET LLPR 旧成果迁移验收报告

日期：2026-07-26  
环境：`conda activate upet_new`  
验收提交主题：`test: verify UPET LLPR migration end to end`

## 1. 范围与结论

旧目录：

```text
/home/lilong/code/UQ/upet/UQ_LLPR
```

旧正式成果源：

```text
/home/lilong/code/UQ/upet/UQ_LLPR/matpes_r2/Hef
```

新代码与成果目录：

```text
/home/lilong/code/UQ/upet_new/Uncertainty_Quantification/LLPR
```

正式迁移输出：

```text
/home/lilong/code/UQ/upet_new/Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
```

结论：

- 已按 fixed ridge 原样迁移旧正式 H、η、Alpha、LLPR details/summary；
- 正式迁移未加载模型、未执行 train/validation/test 全量前向或 Jacobian；
- 仅对 `matpes_n20.extxyz` 执行了 fixed 和 fit 两条真实全链路功能测试；
- 正式输出已通过两次 full 校验和一次导入幂等性校验；
- 正式绘图统计与旧 reference manifest 在
  `rtol=1e-12, atol=1e-14` 下相符。

## 2. 来源身份与文件哈希

旧代码 Git commit：

```text
3dab1f5e16fb37edeccf137f10a4b7e91ef21d44
```

迁移 inventory：

```text
inventory_sha256 = 6f5e3b2b9308e807bcb8ce6b4fca43eea4c54f36f76d9aa84843ab073708d587
file_count       = 52
```

输入哈希：

```text
checkpoint = 879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
train      = 12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec
validation = 5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef
test       = 1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d
n20        = c92161329aab539064a2c2438a395cb01e38bfc91211c558aebbc1ff94702e3d
```

## 3. 正式迁移身份与数值

规范身份：

```text
legacy root = cb141ef8c0b24c3c
curvature   = 25721b77f57d3659
calibration = bc6a0db34ca88ca1
evaluation  = dd72e9f459fc1ba3
plot        = 73dcfcfa4a9bb7b8
```

所有正式数值阶段 manifest 均为：

```text
origin = legacy_import
legacy_fixed_ridge = true
```

固定参数：

```text
eta_energy   = 1.0e-6
eta_force    = 1.0e-6
Alpha_energy = 1.1467388818005693
Alpha_force  = 0.2095766082027508
```

维度与正式计数：

```text
energy dimension       = 1026
force dimension        = 3078
total dimension        = 4104
test structures        = 19374
test atoms             = 149321
test force components  = 447963
```

规范 `H + eta I` 条件数：

```text
energy = 1.0623512652391861e15
force  = 1.3809081464650994e12
```

两者均超过 `1e10`，因此 manifest 保留 `condition_warning=true`。固定
`eta=1e-6` 的旧结果虽然 Cholesky、q 和方差均成功且为正，但不应据此声称数值
稳健。旧 `H + eta^2 I` 条件诊断只作为 raw provenance 保存，未用于规范求解。

正式评估摘要：

```text
energy RMSE/atom                 = 0.05493578732829762
force RMSE/component             = 0.1506777498968727
mean calibrated energy std       = 0.058678765166641764
mean calibrated force comp. std  = 0.16204910666077416
```

正式绘图相关性：

```text
energy log10 Pearson = 0.14123766264998672
energy Spearman      = 0.1365768894301838
force log10 Pearson  = 0.4116680400150901
force Spearman       = 0.45650505211915415
```

生成 9 个绘图/统计文件；加入 plot manifest 后，full verify 检查 5 个
manifest 和 69 个验证文件（17 个规范文件及 52 个 raw 审计文件）。

## 4. 旧文件分类

```text
authoritative = 48
incomplete    = 1
orphan        = 1
legacy_smoke  = 2
```

非正式文件：

```text
incomplete:
  results/LLPR/fit/reliability_matpes_linear_fit_summary.json
orphan:
  results/LLPR/reliability_matpes_llpr.png
legacy_smoke:
  results/llpr_test_details.npz
  results/llpr_test_dry_run.log
```

这些文件保留在 `legacy_raw/` 中，但不会进入规范正式 curvature、
calibration 或 evaluation。

## 5. n20 全链路

n20 共享 curvature：

```text
curvature = 19782c0944bdbf86
structures / atoms / force components = 20 / 143 / 429
```

fixed：

```text
calibration = f7c784e86761226a
evaluation  = 2b0dbdd481ffc0cb
eta_energy  = 1.0e-6
eta_force   = 1.0e-6
Alpha_energy = 0.004065729676684197
Alpha_force  = 0.00015999033492658578
```

fit：

```text
calibration = 8247bfc5b9068ead
evaluation  = 8250ddea1d3e0aa2
energy candidates = 7
force candidates  = 7
selected eta_energy = 0.003961338496504261
selected eta_force  = 3.211714185382302e-05
Alpha_energy = 0.004699296649766541
Alpha_force  = 0.0001788234944392615
```

energy 与 force 分别在 validation 上以 Gaussian NLL 选 η；test 未参与
η/Alpha 选择。两套 details 的 q、variance、std 均为有限正值，结构和分量无
遗漏，并各自生成完整绘图。

真实门控验收结果：

```text
2 passed, 53 deselected in 70.09s
```

## 6. 执行与验证证据

主要命令（均在 `upet_new` 环境运行）：
复核修复后又在独立的 `n20_progress_acceptance` 实验目录从零执行 fixed/fit，
用于真实覆盖 calibration checkpoint 写入和成功清理：

```text
full verify = 5 manifests / 12 declared files; residual progress files = 0
```

```bash
tox -e llpr-tests -- -m "not llpr_n20 and not llpr_legacy" -q

UPET_RUN_LLPR_N20=1 tox -e llpr-tests -- \
  Uncertainty_Quantification/LLPR/tests/test_n20.py \
  Uncertainty_Quantification/LLPR/tests/test_checkpoint_readout_data.py \
  -m llpr_n20 -v

python -m Uncertainty_Quantification.LLPR.llpr import-legacy \
  --config Uncertainty_Quantification/LLPR/configs/import_legacy.yaml

python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy

MPLBACKEND=Agg python -m Uncertainty_Quantification.LLPR.llpr plot \
  --config Uncertainty_Quantification/LLPR/configs/plot_legacy.yaml

tox -e lint
```

结果：

```text
fast LLPR tests = 61 passed, 2 deselected
n20 tests       = 2 passed, 53 deselected
lint            = ruff + mypy (56 files) + sphinx-lint passed
formal verify   = full, 5 manifests, 69 verified files
```

第一次 full verify、第二次 identity-matched import、第二次 full verify 均退出
0。第二次导入前后：

```text
formal details SHA/mtime   unchanged = true/true
canonical curvature SHA/mtime unchanged = true/true
```

正式树中不存在 `origin: recomputed`。搜索到的 `build_system` /
`Processing structure` 字样只位于原样复制的 `legacy_raw/scripts/` 源代码，
不是本次迁移执行日志。`origin: recomputed` 仅存在于获准运行的
`outputs/n20_smoke`。

## 7. 兼容性修复

真实 n20 测试发现并修复了两处当前依赖兼容问题：

- 使用 vesin 公共 `NeighborList` API 替换已删除的
  `metatomic.torch.ase_calculator._compute_ase_neighbors` 私有函数；
- 使用 readout layout 已发现的 checkpoint 源键
  `non_conservative_forces` 调用模型，同时保持规范制品字段命名不变。

两处修复均先由真实验收失败定位，再重跑快速测试和 n20 全链路确认。

独立代码复核后增加的完整性与恢复修复：

- import 配置中的 checkpoint/train/validation/test SHA 必须与实际文件重新计算
  的 SHA 一致，否则在 staging 前失败；
- full verify 逐条校验 inventory 中 52 个 `legacy_raw` 文件的安全相对路径、
  大小和 SHA；
- calibration 保存 identity-bound、结构原子的进度并可等价恢复；
