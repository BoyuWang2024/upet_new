# LLPR README Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rewrite the LLPR README as a short user-oriented guide while still documenting all 64 formal migrated files individually.

**Architecture:** Replace the current audit-style README with a minimal introduction, commands, directory overview, grouped three-column file tables, and a short boundary note. Preserve invisible Markdown markers around the normalized and legacy catalogs so a mechanical check can prove that all current files are still documented.

**Tech Stack:** Markdown, Python 3.11, JSON, pathlib, regular expressions, existing LLPR verification CLI, tox.

## Global Constraints

- Modify only `Uncertainty_Quantification/LLPR/README.md`.
- Do not modify any file below `Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy`.
- Document all 64 current files individually: 12 normalized files and 52 `legacy_raw` files.
- Each file row has exactly three user-facing columns: `文件`, `作用`, `与旧文件的关系`.
- Each purpose is one short sentence.
- Remove SHA-256, identity, condition numbers, manifest field internals, `classification`, `formal_source`, detailed numerical audit, mathematical derivations, and all NPZ field-level tables.
- State that `plots/` currently exists and is empty.
- State that formal migration reused old results and did not rerun the full model computation.
- Keep the old logical root `/home/lilong/code/UQ/upet/UQ_LLPR/matpes_r2/Hef` for path correspondence.
- Keep `MIGRATION_REPORT.md` as the destination for detailed audit evidence.

---

### Task 1: Rewrite and verify the simplified README

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/README.md`
- Read: `Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy/inventory.json`
- Read: `Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy/**`
- Read: `Uncertainty_Quantification/LLPR/MIGRATION_REPORT.md`

**Interfaces:**
- Consumes: the current formal output tree, inventory paths, the approved simplification design, and the existing README's verified file-purpose mapping.
- Produces: one simplified README containing exactly 12 normalized rows and 52 legacy rows between machine-checkable comment markers.

- [ ] **Step 1: Reconfirm the formal result tree before editing**

Run:

```bash
cd /home/lilong/code/UQ/upet_new
find Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy \
  -type f -printf '%P\n' | sort
find Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy \
  -type f | wc -l
find Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy/legacy_raw \
  -type f | wc -l
find Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy/plots \
  -mindepth 1 | wc -l
```

Expected:

```text
64 total files
52 legacy_raw files
0 entries below plots/
```

- [ ] **Step 2: Replace the README structure**

Use these headings and no additional audit sections:

```markdown
# UPET LLPR 不确定性量化
## 使用方式
## 正式迁移结果
## 规范化文件
## 原样保留的旧文件
### 配置
### 曲率、校准和模型检查结果
### 正式评估、测试和绘图结果
### 旧脚本、缓存和测试
## 结果边界
```

The introduction must state:

```text
LLPR supports fixed eta and validation-fitted eta.
The current formal migrated result uses fixed eta.
Detailed audit evidence is in MIGRATION_REPORT.md.
```

The command section must include:

```bash
conda activate upet_new
cd /home/lilong/code/UQ/upet_new
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
```

Also show the future full recomputation entry without running it:

```bash
python -m Uncertainty_Quantification.LLPR.llpr run \
  --config Uncertainty_Quantification/LLPR/configs/gpu_full_fixed.yaml
```

State that `gpu_full_fit.yaml` selects fitted eta instead, and neither configuration is used during legacy migration.

The directory overview must explain only:

```text
curvature/   normalized curvature
calibration/ normalized eta/Alpha calibration
evaluation/  normalized formal test results
plots/       currently empty
legacy_raw/  byte-preserved old files
```

- [ ] **Step 3: Add the 12 normalized file rows**

Keep these markers:

```markdown
<!-- BEGIN NORMALIZED FILE CATALOG -->
<!-- END NORMALIZED FILE CATALOG -->
```

Use these exact paths and meanings:

| File | Short purpose | Legacy relationship |
| --- | --- | --- |
| `manifest.json` | 正式迁移结果的总目录。 | 迁移后新增，没有一一对应的旧文件。 |
| `inventory.json` | 记录已保留的全部旧文件。 | 根据旧目录文件生成。 |
| `curvature/406edc88d16fdcc7/manifest.json` | 记录规范化曲率结果包含哪些文件。 | 迁移后新增，没有一一对应的旧文件。 |
| `curvature/406edc88d16fdcc7/curvature.npz` | 保存 energy 和 force 曲率矩阵。 | 从旧 `H_E_full_run.npz`、`H_F_full_run.npz` 提取，并用 `H_EF_full_run.npz` 核对。 |
| `curvature/406edc88d16fdcc7/diagnostics.json` | 保存曲率构建规模和矩阵检查结果。 | 根据旧 `Hef_full_run_summary.json` 和旧曲率矩阵整理。 |
| `calibration/50217238109b0418/manifest.json` | 记录规范化校准结果包含哪些文件。 | 迁移后新增，没有一一对应的旧文件。 |
| `calibration/50217238109b0418/candidates.json` | 保存 energy 和 force 的校准候选。 | 根据旧 `alpha_val_full_joint_summary.json` 整理。 |
| `calibration/50217238109b0418/summary.json` | 保存当前选用的固定 eta 和 Alpha。 | 根据旧 `alpha_val_full_joint_summary.json` 整理。 |
| `evaluation/50217238109b0418/28d0d911b3060988/manifest.json` | 记录规范化评估结果包含哪些文件。 | 迁移后新增，没有一一对应的旧文件。 |
| `evaluation/50217238109b0418/28d0d911b3060988/details.npz` | 保存正式测试的完整逐样本结果。 | 由旧 `llpr_test_full_gpu_details.npz` 原数组重打包。 |
| `evaluation/50217238109b0418/28d0d911b3060988/summary.json` | 保存正式测试结果摘要。 | 根据旧正式测试 details 和 summary 重新整理。 |
| `evaluation/50217238109b0418/28d0d911b3060988/preview.json` | 提供正式测试结果的轻量预览。 | 根据规范化 details 和 summary 生成。 |

- [ ] **Step 4: Add all 52 legacy rows in directory groups**

Keep these markers around all four legacy subsections:

```markdown
<!-- BEGIN LEGACY RAW FILE CATALOG -->
<!-- END LEGACY RAW FILE CATALOG -->
```

Every row's relationship column must be:

```text
旧目录同路径文件，原样保留。
```

Use these exact paths and short purposes:

**配置**

```text
legacy_raw/configs/compute_Alpha.yaml | 旧验证集校准配置。
legacy_raw/configs/compute_Hef.yaml | 旧曲率构建配置。
legacy_raw/configs/compute_LLPR.yaml | 旧正式测试配置。
legacy_raw/configs/plot_LLPR_reference.yaml | 旧参考绘图配置。
```

**曲率、校准和模型检查结果**

```text
legacy_raw/results/H_EF_full_run.npz | 旧 energy 与 force 联合曲率矩阵。
legacy_raw/results/H_E_full_run.npz | 旧 energy 曲率矩阵。
legacy_raw/results/H_F_full_run.npz | 旧 force 曲率矩阵。
legacy_raw/results/Hef_full_run_summary.json | 旧曲率构建摘要。
legacy_raw/results/alpha_val_full_energy_summary.json | 旧 energy 校准摘要。
legacy_raw/results/alpha_val_full_force_summary.json | 旧 force 校准摘要。
legacy_raw/results/alpha_val_full_joint_summary.json | 旧 energy 与 force 联合校准摘要。
legacy_raw/results/alpha_val_full_run.log | 旧完整校准运行日志。
legacy_raw/results/model/check_model.log | 旧模型检查日志。
legacy_raw/results/model/check_model_summary.json | 旧模型结构检查摘要。
```

**正式评估、测试和绘图结果**

```text
legacy_raw/results/llpr_test_details.npz | 旧小规模测试明细，不是正式结果。
legacy_raw/results/llpr_test_dry_run.log | 旧小规模测试运行日志。
legacy_raw/results/llpr_test_small_preview.json | 旧小规模测试预览。
legacy_raw/results/llpr_test_summary.json | 旧小规模测试摘要。
legacy_raw/results/LLPR/llpr_test_full_gpu_details.npz | 旧正式测试完整明细。
legacy_raw/results/LLPR/llpr_test_full_gpu_run.log | 旧正式测试运行日志。
legacy_raw/results/LLPR/llpr_test_full_gpu_small_preview.json | 旧正式测试预览。
legacy_raw/results/LLPR/llpr_test_full_gpu_summary.json | 旧正式测试摘要。
legacy_raw/results/LLPR/plot_LLPR.log | 旧 reliability 绘图日志。
legacy_raw/results/LLPR/reliability_matpes_energy_llpr.png | 旧 energy reliability 图。
legacy_raw/results/LLPR/reliability_matpes_force_component_llpr.png | 旧 force reliability 图。
legacy_raw/results/LLPR/reliability_matpes_llpr.png | 旧额外合并图，未列入正式绘图摘要。
legacy_raw/results/LLPR/reliability_matpes_plot_summary.json | 旧 reliability 绘图摘要。
legacy_raw/results/LLPR/llpr_energy_uncertainty_vs_residual.pdf | 旧 energy 不确定性—残差参考图 PDF。
legacy_raw/results/LLPR/llpr_energy_uncertainty_vs_residual.png | 旧 energy 不确定性—残差参考图 PNG。
legacy_raw/results/LLPR/llpr_force_uncertainty_vs_residual.pdf | 旧 force 不确定性—残差参考图 PDF。
legacy_raw/results/LLPR/llpr_force_uncertainty_vs_residual.png | 旧 force 不确定性—残差参考图 PNG。
legacy_raw/results/LLPR/llpr_reference_plotting_manifest.json | 旧参考绘图发布清单。
legacy_raw/results/LLPR/llpr_reference_plotting_statistics.csv | 旧参考绘图统计数据。
legacy_raw/results/LLPR/fit/fit_LLPR.log | 旧拟合运行日志。
legacy_raw/results/LLPR/fit/reliability_matpes_energy_log_fit.png | 旧 energy 对数拟合图。
legacy_raw/results/LLPR/fit/reliability_matpes_force_component_log_fit.png | 旧 force 对数拟合图。
legacy_raw/results/LLPR/fit/reliability_matpes_linear_fit_summary.json | 旧线性拟合摘要，配套图片未完整保留。
legacy_raw/results/LLPR/fit/reliability_matpes_log_fit_summary.json | 旧对数拟合摘要。
```

**旧脚本、缓存和测试**

```text
legacy_raw/scripts/check_model.py | 旧模型结构检查脚本。
legacy_raw/scripts/compute_Alpha.py | 旧验证集校准脚本。
legacy_raw/scripts/compute_Hef.py | 旧曲率构建脚本。
legacy_raw/scripts/compute_LLPR.py | 旧正式测试计算脚本。
legacy_raw/scripts/fit_LLPR.py | 旧结果拟合脚本。
legacy_raw/scripts/plot_LLPR.py | 旧 reliability 绘图脚本。
legacy_raw/scripts/plot_LLPR_reference.py | 旧参考图生成脚本。
legacy_raw/scripts/__pycache__/check_model.cpython-310.pyc | `check_model.py` 的旧 Python 3.10 缓存。
legacy_raw/scripts/__pycache__/compute_Alpha.cpython-310.pyc | `compute_Alpha.py` 的旧 Python 3.10 缓存。
legacy_raw/scripts/__pycache__/compute_Hef.cpython-310.pyc | `compute_Hef.py` 的旧 Python 3.10 缓存。
legacy_raw/scripts/__pycache__/compute_LLPR.cpython-310.pyc | `compute_LLPR.py` 的旧 Python 3.10 缓存。
legacy_raw/scripts/__pycache__/fit_LLPR.cpython-310.pyc | `fit_LLPR.py` 的旧 Python 3.10 缓存。
legacy_raw/scripts/__pycache__/plot_LLPR.cpython-310.pyc | `plot_LLPR.py` 的旧 Python 3.10 缓存。
legacy_raw/tests/test_plot_LLPR_reference.py | 旧参考绘图测试。
```

- [ ] **Step 5: Add the short result boundary**

The final section must state:

```text
The formal migration did not reload the checkpoint or rerun the complete model/Jacobian calculation.
The small-scale chain-test output directories were deleted after validation.
The current formal plots/ directory is empty; old plots remain only in legacy_raw.
Detailed hashes, identities, values, and audit evidence are in MIGRATION_REPORT.md.
```

- [ ] **Step 6: Verify that the simplified README still covers all files**

Run this check from the repository root:

```python
import json
import re
from pathlib import Path

readme_path = Path("Uncertainty_Quantification/LLPR/README.md")
root = Path("Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy")
text = readme_path.read_text(encoding="utf-8")

def documented(marker):
    match = re.search(
        rf"<!-- BEGIN {marker} -->(.*?)<!-- END {marker} -->",
        text,
        re.S,
    )
    assert match is not None
    rows = re.findall(r"^\| `([^`]+)` \|", match.group(1), re.M)
    assert len(rows) == len(set(rows))
    return set(rows)

normalized = documented("NORMALIZED FILE CATALOG")
legacy = documented("LEGACY RAW FILE CATALOG")
actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
}
inventory = json.loads((root / "inventory.json").read_text(encoding="utf-8"))
expected_legacy = {
    "legacy_raw/" + item["destination_relative"]
    for item in inventory["files"]
}

assert len(normalized) == 12
assert len(legacy) == 52
assert legacy == expected_legacy
assert normalized | legacy == actual
assert len(actual) == 64
assert not any((root / "plots").iterdir())

for forbidden in (
    "CURVATURE FIELD CATALOG",
    "EVALUATION FIELD CATALOG",
    "condition_number",
    "formal_source",
    "SHA-256",
    "## 数学定义",
):
    assert forbidden not in text

assert len(text.splitlines()) <= 220
print("README catalog verified: 12 normalized + 52 legacy files")
```

Expected:

```text
README catalog verified: 12 normalized + 52 legacy files
```

- [ ] **Step 7: Run formal artifact and repository checks**

Run:

```bash
source /home/lilong/miniforge3/etc/profile.d/conda.sh
conda activate upet_new
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
tox -e lint
git diff --check
git diff --name-only
```

Expected:

```text
full verify completes with 4 manifests and 60 verified files
tox lint passes
git diff --check prints nothing
git diff --name-only prints only Uncertainty_Quantification/LLPR/README.md
```

- [ ] **Step 8: Commit**

```bash
git add Uncertainty_Quantification/LLPR/README.md
git commit -m "docs: simplify LLPR file guide"
```
