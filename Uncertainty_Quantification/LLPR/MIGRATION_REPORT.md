# UPET LLPR 旧成果迁移验收报告

日期：2026-07-28
环境：`conda activate upet_new`

## 结论

旧 UPET LLPR 正式成果已经按固定 ridge 原样迁移到：

```text
/home/lilong/code/UQ/upet_new/Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
```

正式迁移没有加载模型，没有重新执行完整数据集的前向、Jacobian、曲率或校准计算。
迁移程序只复制并审计旧原始文件，再从已保存的矩阵、Alpha 和测试明细生成规范视图。

代码同时支持直接指定 `eta` 和在验证集拟合 `eta`；当前正式成果仍为直接指定：

```text
eta_energy = 1.0e-6
eta_force  = 1.0e-6
```

## 修正后的正式身份

```text
legacy root = 463ae307034cf391
curvature   = 406edc88d16fdcc7
calibration = 50217238109b0418
evaluation  = 28d0d911b3060988
plot        = 761260bfde30937d
```

来源标记：

```text
root / curvature / calibration / evaluation:
  origin = legacy_import
  legacy_fixed_ridge = true

plot:
  origin = derived
  source_origin = legacy_import
```

正式数值阶段中不存在 `origin: recomputed`。

## 输入身份

```text
checkpoint = 879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
train      = 12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec
validation = 5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef
test       = 1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d
```

迁移 inventory：

```text
inventory_sha256 = 0cf590fa39055b689d829aa80725426135fdacbf7bc6782fcf3e341532fc4517
file_count       = 52
```

这 52 个 `legacy_raw` 文件与替换前正式树中的原始副本逐文件 SHA-256 完全一致。

## 三类数据来源与计数

曲率构建摘要：

```text
build structures = 348780
```

验证摘要：

```text
validation energy samples    = 19370
validation force components  = 458877
```

正式测试摘要与明细：

```text
test structures        = 19374
test atoms             = 149321
test force components  = 447963
```

曲率 diagnostics 只记录可恢复的 build structure count，不再把测试 atom/force count
错误写入曲率阶段。

## 固定参数与诊断

```text
Alpha_energy = 1.1467388818005693
Alpha_force  = 0.2095766082027508

energy dimension = 1026
force dimension  = 3078
total dimension  = 4104
```

旧验证摘要没有逐样本验证残差和方差，不能重建验证 NLL 与 coverage。因此规范校准记录使用：

```text
gaussian_nll    = null
coverage_1sigma = null
coverage_2sigma = null
coverage_3sigma = null
diagnostics_status = unavailable_from_legacy_validation_summary
```

不会用测试集明细伪装成验证诊断。

`H + eta I` 条件数：

```text
energy = 1.0623512652391861e15
force  = 1.3809081464650994e12
```

两者均保留 `condition_warning=true`。迁移成功不代表该固定 `eta` 数值稳健。

## 数值等价审计

候选树发布前执行了以下只读比较：

```text
52 source/raw hashes exact       = true
energy curvature array exact     = true
force curvature array exact      = true
44 evaluation detail arrays exact = true
```

数组比较使用 `numpy.array_equal`；含 NaN 的明细使用 `equal_nan=True`。因此正式旧数值没有
被重新计算或近似转换。

完整验证：

```text
status              = complete
level               = full
manifest_count      = 5
verified_file_count = 69
```

## 绘图统计

```text
energy log10 Pearson = 0.14123766264998672
energy Spearman      = 0.1365768894301838
force log10 Pearson  = 0.4116680400150901
force Spearman       = 0.45650505211915415
```

图由规范 evaluation 明细重新生成，因此标记为 derived，并显式绑定
`source_origin: legacy_import`。

## 代码与小规模全链路验收

常规测试：

```text
71 passed, 3 deselected
```

真实 n20 全链路：

```text
3 passed, 71 deselected
```

覆盖：

- fixed `eta` 全链路；
- fit `eta` 全链路；
- checkpoint/readout 真实维度；
- 曲率在第 6 次 Jacobian 前中断、从前 5 个结构恢复；
- 恢复运行与不间断运行的曲率数组和 diagnostics 完全一致。

静态检查：

```text
ruff format = passed
ruff check  = passed
mypy        = passed (63 source files)
sphinx-lint = passed
```

独立代码复核从设计基线 `d8c6349` 检查到候选发布前 HEAD。代码层面无 Critical，
也无未解决的 Important 问题；唯一发布阻断是替换旧正式目录并更新本报告，已纳入最终
事务发布步骤。

## 发布策略

候选树先在独立目录生成并完成 full verify 与逐数组审计。正式发布采用同一文件系统内的
目录重命名：

1. 旧正式目录重命名为临时备份；
2. 已验证候选目录重命名为正式目录；
3. 在正式路径再次运行 full verify 和逐数组/来源审计；
4. 仅在全部通过后删除临时备份。

这样可避免部分覆盖，并保证最终删除旧版本前仍可回滚。
