# LLPR Migrated File Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expand the LLPR README into an exhaustive, mechanically verified catalog of all 64 files currently present in the formal migrated result tree.

**Architecture:** Keep the existing operational and mathematical introduction, then add a current-state snapshot, a relationship legend, a normalized-file catalog, a complete legacy-raw catalog, and an NPZ field reference. Markdown comment markers delimit machine-checkable catalog sections so validation can compare documented paths and fields directly against the filesystem, inventory, manifests, and NPZ archives.

**Tech Stack:** Markdown, Python 3.11, `pathlib`, `json`, NumPy, existing LLPR `verify` CLI, tox/ruff/mypy/sphinx-lint.

## Global Constraints

- Modify only `Uncertainty_Quantification/LLPR/README.md`; do not alter any file below `Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy`.
- Document the current tree: 64 files, 52 `legacy_raw` files, 12 normalized files, 4 manifests, 60 full-verified files, and an empty `plots/` directory.
- Do not regenerate plots.
- Derive every statement from the current filesystem, manifests, `inventory.json`, retained legacy code/configuration, or actual NPZ/JSON contents.
- Use the deleted old logical root `/home/lilong/code/UQ/upet/UQ_LLPR/matpes_r2/Hef` for legacy-path correspondence.
- Treat `inventory.json.destination_relative` as the path mapping source; current `source_absolute` values point to the retained new `legacy_raw` tree.
- Explain all 64 files individually.
- Explain both fields in normalized `curvature.npz` and all 44 fields in normalized `details.npz`.
- Keep the distinction between exact copies, extracted/repacked data, derived summaries, and migration-only metadata.

---

### Task 1: Document current state and every normalized file

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/README.md`

**Interfaces:**
- Consumes: current root and stage manifests, `llpr/legacy.py:393-644`, and the actual normalized files.
- Produces: README sections Current formal directory state, Relationship legend, Normalized data flow, and a normalized catalog delimited by <!-- BEGIN NORMALIZED FILE CATALOG --> / <!-- END NORMALIZED FILE CATALOG -->.

- [ ] **Step 1: Reconfirm the normalized filesystem before editing**

Run:

```bash
find Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy \
  -path '*/legacy_raw/*' -prune -o -type f -printf '%P\n' | sort
```

Expected normalized paths:

```text
calibration/50217238109b0418/candidates.json
calibration/50217238109b0418/manifest.json
calibration/50217238109b0418/summary.json
curvature/406edc88d16fdcc7/curvature.npz
curvature/406edc88d16fdcc7/diagnostics.json
curvature/406edc88d16fdcc7/manifest.json
evaluation/50217238109b0418/28d0d911b3060988/details.npz
evaluation/50217238109b0418/28d0d911b3060988/manifest.json
evaluation/50217238109b0418/28d0d911b3060988/preview.json
evaluation/50217238109b0418/28d0d911b3060988/summary.json
inventory.json
manifest.json
```

- [ ] **Step 2: Add the current-state snapshot and relationship legend**

State the exact counts and explain these four labels:

```text
exact copy
extracted and repacked
derived from legacy data
no one-to-one legacy counterpart
```

Also state:

```text
plots/ currently exists as an empty directory.
Current full verification: 4 manifests / 60 verified files.
```

- [ ] **Step 3: Add the normalized data-flow explanation**

Document this exact flow:

```text
H_E_full_run.npz + H_F_full_run.npz
  -> active energy/force blocks -> curvature.npz
H_EF_full_run.npz
  -> validates H_EF == H_E + H_F
Hef_full_run_summary.json + active blocks + eta
  -> diagnostics.json
alpha_val_full_joint_summary.json + eta + condition numbers
  -> candidates.json + calibration summary.json
llpr_test_full_gpu_details.npz
  -> repacked details.npz -> derived summary.json + preview.json
```

- [ ] **Step 4: Add one catalog row for each of the 12 normalized files**

Use columns:

```markdown
| Current relative path | Purpose and actual content | Legacy counterpart/input | Relationship |
```

The rows must encode these mappings:

- `manifest.json`: migration root identity, input hashes, split counts, and stage identities; no old counterpart.
- `inventory.json`: 52 legacy paths, sizes, SHA-256 values, classifications, and formal-source flags; no old counterpart.
- curvature manifest: declares identity and hashes for the two curvature files; no old counterpart.
- `curvature.npz`: energy and force active blocks from old `H_E_full_run.npz` and `H_F_full_run.npz`, with old `H_EF_full_run.npz` used for the sum invariant.
- `diagnostics.json`: dimensions/build count from old build summary and condition numbers from normalized blocks plus fixed eta.
- calibration manifest: binds curvature identity, validation hash, fixed mode, eta, and declared files; no old counterpart.
- `candidates.json`: one fixed candidate per target, from old joint Alpha summary plus normalized conditions.
- calibration `summary.json`: selected fixed candidate per target, same source as candidates.
- evaluation manifest: binds curvature, calibration, and test identities; no old counterpart.
- `details.npz`: all 44 old formal detail arrays, array-exact but recompressed.
- evaluation `summary.json`: re-aggregated metrics from details plus audited eta/Alpha.
- `preview.json`: canonical summary plus first ten structure indices.

- [ ] **Step 5: Verify the normalized catalog paths**

Run a Python check that extracts first-column paths between the normalized
markers and compares them with the filesystem paths outside `legacy_raw`:

```python
import re
from pathlib import Path

readme = Path("Uncertainty_Quantification/LLPR/README.md").read_text()
root = Path("Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy")
section = re.search(
    r"<!-- BEGIN NORMALIZED FILE CATALOG -->(.*?)"
    r"<!-- END NORMALIZED FILE CATALOG -->",
    readme,
    re.S,
)
assert section is not None
documented = set(re.findall(r"^\| `([^`]+)` \|", section.group(1), re.M))
actual = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file() and "legacy_raw" not in path.relative_to(root).parts
}
assert documented == actual, (sorted(actual - documented), sorted(documented - actual))
assert len(actual) == 12
```

Expected: exit 0.

---

### Task 2: Document all 52 retained legacy files

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/README.md`

**Interfaces:**
- Consumes: `inventory.json`, all files below `legacy_raw`, retained YAML/scripts/results, and the classification rules in `llpr/legacy.py:92-110`.
- Produces: a legacy catalog delimited by `<!-- BEGIN LEGACY RAW FILE CATALOG -->` / `<!-- END LEGACY RAW FILE CATALOG -->`.

- [ ] **Step 1: Add the legacy-root and inventory semantics**

Explain:

```text
old logical path =
  /home/lilong/code/UQ/upet/UQ_LLPR/matpes_r2/Hef/<destination_relative>
current retained path =
  legacy_raw/<destination_relative>
```

State that every retained file is byte-preserved and hash-checked, and that
`authoritative` means intact audited retention rather than direct numerical
consumption.

- [ ] **Step 2: Add four configuration rows**

Document individually:

```text
legacy_raw/configs/compute_Alpha.yaml
legacy_raw/configs/compute_Hef.yaml
legacy_raw/configs/compute_LLPR.yaml
legacy_raw/configs/plot_LLPR_reference.yaml
```

Use actual YAML keys to explain validation calibration, curvature construction,
formal test evaluation, and approved reference plotting respectively.

- [ ] **Step 3: Add all legacy matrix/build/model rows**

Document individually:

```text
legacy_raw/results/H_EF_full_run.npz
legacy_raw/results/H_E_full_run.npz
legacy_raw/results/H_F_full_run.npz
legacy_raw/results/Hef_full_run_summary.json
legacy_raw/results/model/check_model.log
legacy_raw/results/model/check_model_summary.json
```

Include the actual matrix shape `(4104, 4104)`, float64 dtype, build count
348780, and model/checkpoint inspection purpose.

- [ ] **Step 4: Add all validation and formal-test rows**

Document individually:

```text
legacy_raw/results/alpha_val_full_energy_summary.json
legacy_raw/results/alpha_val_full_force_summary.json
legacy_raw/results/alpha_val_full_joint_summary.json
legacy_raw/results/alpha_val_full_run.log
legacy_raw/results/LLPR/llpr_test_full_gpu_details.npz
legacy_raw/results/LLPR/llpr_test_full_gpu_run.log
legacy_raw/results/LLPR/llpr_test_full_gpu_small_preview.json
legacy_raw/results/LLPR/llpr_test_full_gpu_summary.json
```

Include validation counts 19370/458877, test counts
19374/149321/447963, and distinguish the target-specific Alpha views from the
joint Alpha source used by normalized calibration.

- [ ] **Step 5: Add smoke, plotting, and fit-result rows**

Document individually:

```text
legacy_raw/results/llpr_test_details.npz
legacy_raw/results/llpr_test_dry_run.log
legacy_raw/results/llpr_test_small_preview.json
legacy_raw/results/llpr_test_summary.json
legacy_raw/results/LLPR/plot_LLPR.log
legacy_raw/results/LLPR/reliability_matpes_energy_llpr.png
legacy_raw/results/LLPR/reliability_matpes_force_component_llpr.png
legacy_raw/results/LLPR/reliability_matpes_llpr.png
legacy_raw/results/LLPR/reliability_matpes_plot_summary.json
legacy_raw/results/LLPR/llpr_energy_uncertainty_vs_residual.pdf
legacy_raw/results/LLPR/llpr_energy_uncertainty_vs_residual.png
legacy_raw/results/LLPR/llpr_force_uncertainty_vs_residual.pdf
legacy_raw/results/LLPR/llpr_force_uncertainty_vs_residual.png
legacy_raw/results/LLPR/llpr_reference_plotting_manifest.json
legacy_raw/results/LLPR/llpr_reference_plotting_statistics.csv
legacy_raw/results/LLPR/fit/fit_LLPR.log
legacy_raw/results/LLPR/fit/reliability_matpes_energy_log_fit.png
legacy_raw/results/LLPR/fit/reliability_matpes_force_component_log_fit.png
legacy_raw/results/LLPR/fit/reliability_matpes_linear_fit_summary.json
legacy_raw/results/LLPR/fit/reliability_matpes_log_fit_summary.json
```

Mark the two dry-run artifacts as `legacy_smoke`, the linear-fit summary as
`incomplete`, and `reliability_matpes_llpr.png` as `orphan`. Explain that the
reference PNG/PDF set is old preserved output even though normalized `plots/`
is currently empty.

- [ ] **Step 6: Add all retained source, bytecode, and test rows**

Document individually:

```text
legacy_raw/scripts/check_model.py
legacy_raw/scripts/compute_Alpha.py
legacy_raw/scripts/compute_Hef.py
legacy_raw/scripts/compute_LLPR.py
legacy_raw/scripts/fit_LLPR.py
legacy_raw/scripts/plot_LLPR.py
legacy_raw/scripts/plot_LLPR_reference.py
legacy_raw/scripts/__pycache__/check_model.cpython-310.pyc
legacy_raw/scripts/__pycache__/compute_Alpha.cpython-310.pyc
legacy_raw/scripts/__pycache__/compute_Hef.cpython-310.pyc
legacy_raw/scripts/__pycache__/compute_LLPR.cpython-310.pyc
legacy_raw/scripts/__pycache__/fit_LLPR.cpython-310.pyc
legacy_raw/scripts/__pycache__/plot_LLPR.cpython-310.pyc
legacy_raw/tests/test_plot_LLPR_reference.py
```

Explain each source script from its real functions and outputs. State that each
`.pyc` is a Python 3.10 cache corresponding to the same-named `.py` source and
is not executable source of truth.

- [ ] **Step 7: Verify legacy paths and classifications**

Run:

```python
import json
import re
from pathlib import Path

readme = Path("Uncertainty_Quantification/LLPR/README.md").read_text()
root = Path("Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy")
inventory = json.loads((root / "inventory.json").read_text())
section = re.search(
    r"<!-- BEGIN LEGACY RAW FILE CATALOG -->(.*?)"
    r"<!-- END LEGACY RAW FILE CATALOG -->",
    readme,
    re.S,
)
assert section is not None
documented = set(re.findall(r"^\| `([^`]+)` \|", section.group(1), re.M))
actual = {
    "legacy_raw/" + row["destination_relative"]
    for row in inventory["files"]
}
assert documented == actual, (sorted(actual - documented), sorted(documented - actual))
assert len(actual) == 52
for row in inventory["files"]:
    path = "legacy_raw/" + row["destination_relative"]
    matching = [
        line for line in section.group(1).splitlines()
        if line.startswith(f"| `{path}` |")
    ]
    assert len(matching) == 1
    assert f"`{row['classification']}`" in matching[0]
```

Expected: exit 0 and all 52 rows match inventory classifications.

---

### Task 3: Add NPZ field reference and run final documentation verification

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/README.md`

**Interfaces:**
- Consumes: normalized `curvature.npz`, normalized `details.npz`, the completed 64-file catalog, and current full verifier.
- Produces: complete NPZ field reference and verified final README.

- [ ] **Step 1: Add the curvature field table**

Between `<!-- BEGIN CURVATURE FIELD CATALOG -->` and
`<!-- END CURVATURE FIELD CATALOG -->`, document:

```text
energy | (1026, 1026) | float64 | active energy curvature block
force  | (3078, 3078) | float64 | active force curvature block
```

- [ ] **Step 2: Add all 44 evaluation fields**

Between `<!-- BEGIN EVALUATION FIELD CATALOG -->` and
`<!-- END EVALUATION FIELD CATALOG -->`, add one row for each actual field:

```text
structure_index
num_atoms
energy_pred_total
energy_true_total
energy_pred_per_atom
energy_true_per_atom
energy_residual
energy_residual_total
energy_residual_per_atom
energy_raw_var
energy_raw_std
energy_calibrated_var
energy_calibrated_std
energy_inverse_variance
energy_rigidity
energy_raw_var_total_derived
energy_calibrated_var_total_derived
energy_calibrated_std_total_derived
force_offsets
force_structure_index
force_component_index_within_structure
force_atom_index
force_cartesian_index
force_pred
force_true
force_residual
force_raw_var_component
force_raw_std_component
force_calibrated_var_component
force_calibrated_std_component
force_inverse_variance_component
force_rigidity_component
structure_force_component_count
atom_structure_index
atom_index_within_structure
force_raw_var_atom_mean
force_calibrated_var_atom_mean
force_calibrated_std_atom_rms
force_calibrated_std_atom_max
force_raw_var_component_mean_structure
force_calibrated_var_component_mean_structure
force_calibrated_std_component_rms_structure
force_calibrated_std_component_max_structure
force_inverse_variance_component_mean_structure
```

For every row, include the actual shape, dtype, semantic level, and formula or
index meaning. Group the table visually into structure identity, energy,
component-force, atom aggregation, and structure aggregation subsections while
keeping every field explicit.

- [ ] **Step 3: Verify all 64 documented files and all NPZ fields**

Run:

```python
import json
import re
from pathlib import Path
import numpy as np

readme = Path("Uncertainty_Quantification/LLPR/README.md").read_text()
root = Path("Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy")

def paths_between(begin, end):
    section = re.search(
        rf"<!-- BEGIN {begin} -->(.*?)<!-- END {end} -->",
        readme,
        re.S,
    )
    assert section is not None
    return set(re.findall(r"^\| `([^`]+)` \|", section.group(1), re.M))

documented_files = (
    paths_between("NORMALIZED FILE CATALOG", "NORMALIZED FILE CATALOG")
    | paths_between("LEGACY RAW FILE CATALOG", "LEGACY RAW FILE CATALOG")
)
actual_files = {
    path.relative_to(root).as_posix()
    for path in root.rglob("*")
    if path.is_file()
}
assert documented_files == actual_files
assert len(actual_files) == 64

inventory = json.loads((root / "inventory.json").read_text())
assert len(inventory["files"]) == 52

curvature_path = next(root.glob("curvature/*/curvature.npz"))
details_path = next(root.glob("evaluation/*/*/details.npz"))
with np.load(curvature_path, allow_pickle=False) as archive:
    curvature_fields = set(archive.files)
with np.load(details_path, allow_pickle=False) as archive:
    evaluation_fields = set(archive.files)

documented_curvature = paths_between(
    "CURVATURE FIELD CATALOG", "CURVATURE FIELD CATALOG"
)
documented_evaluation = paths_between(
    "EVALUATION FIELD CATALOG", "EVALUATION FIELD CATALOG"
)
assert documented_curvature == curvature_fields == {"energy", "force"}
assert documented_evaluation == evaluation_fields
assert len(evaluation_fields) == 44
assert not any((root / "plots").iterdir())
```

Expected: exit 0.

- [ ] **Step 4: Run artifact and repository verification**

Run:

```bash
source /home/lilong/miniforge3/etc/profile.d/conda.sh
conda activate upet_new
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
tox -e lint
git diff --check
```

Expected:

```text
full verify: 4 manifests / 60 verified files
ruff format/check: passed
mypy: passed
sphinx-lint: passed
git diff --check: no output
```

- [ ] **Step 5: Confirm formal artifacts did not change**

Run:

```bash
git status --short
find Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy \
  -type f -printf '%P\n' | sort
```

Expected: only README and the committed plan/spec are source changes; the
formal output tree still has exactly 64 files and `plots/` remains empty.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/LLPR/README.md
git commit -m "docs: catalog migrated LLPR files"
```
