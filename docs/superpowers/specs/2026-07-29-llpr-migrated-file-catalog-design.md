# LLPR Migrated File Catalog Design

Date: 2026-07-29

## Goal

Expand `Uncertainty_Quantification/LLPR/README.md` into an evidence-backed
catalog of every file currently stored below:

```text
Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
```

The catalog must explain each file's purpose and identify its corresponding
legacy file or legacy inputs when a correspondence exists.

## Source of truth

The documentation must be derived from the current filesystem and these
authoritative sources:

- the root and stage manifests;
- `inventory.json`;
- the arrays and metadata actually stored in every NPZ/JSON/YAML/CSV file;
- the legacy scripts retained under `legacy_raw/scripts`;
- the import implementation in `llpr/legacy.py`;
- the verification implementation in `llpr/artifacts.py`.

The deleted legacy logical root was:

```text
/home/lilong/code/UQ/upet/UQ_LLPR/matpes_r2/Hef
```

`legacy_raw/<relative-path>` preserves the layout and bytes of
`<old-root>/<relative-path>`. The current `source_absolute` fields point back
to the retained `legacy_raw` tree after cleanup, so legacy correspondence must
be explained using `destination_relative` plus the audited old logical root.

## Current state

The README must describe the current state, not the earlier acceptance state:

- 64 files total;
- 52 files under `legacy_raw`;
- 12 normalized files;
- 4 manifests;
- 60 files verified by current full verification;
- `plots/` exists but is empty.

No plots will be regenerated as part of this documentation task.

## Document structure

1. Current-state snapshot and root-path conventions.
2. Relationship legend.
3. Normalized-tree data flow.
4. Exhaustive normalized-file catalog.
5. Exhaustive `legacy_raw` file catalog, grouped by directory.
6. NPZ field reference.
7. Formal-source and classification boundaries.
8. Verification commands and mechanically checked completeness.

The relationship legend uses four terms:

- **Exact copy**: byte-preserved legacy file under `legacy_raw`.
- **Extracted/repacked**: values copied or sliced from legacy arrays and saved
  in a normalized container.
- **Derived from legacy data**: new summaries or diagnostics computed only
  from saved legacy artifacts and audited constants.
- **No direct legacy counterpart**: migration metadata introduced by the new
  framework.

## Normalized-file mappings

The README must document these proven mappings:

- `curvature.npz` contains the active energy block
  `H_E[0:1026, 0:1026]` and active force block
  `H_F[1026:4104, 1026:4104]`. `H_EF_full_run.npz` validates that the joint
  matrix equals `H_E + H_F`.
- `diagnostics.json` derives dimensions and the build count from
  `Hef_full_run_summary.json`, and condition numbers from the extracted blocks
  with fixed `eta=1e-6`.
- calibration `candidates.json` and `summary.json` use Alpha and validation
  counts from `alpha_val_full_joint_summary.json`, fixed eta, and normalized
  condition numbers. Missing legacy validation NLL/coverage remain `null`.
- evaluation `details.npz` repacks all 44 arrays from
  `llpr_test_full_gpu_details.npz` without changing array values.
- evaluation `summary.json` re-aggregates the normalized detail arrays and adds
  audited eta/Alpha.
- evaluation `preview.json` contains the canonical summary and the first ten
  structure indices.
- all normalized manifests and the root `inventory.json` have no direct legacy
  file counterpart.

For each normalized file, the catalog will list every direct legacy input
rather than selecting only one approximate filename.

## Exhaustive legacy catalog

All 52 `legacy_raw` files must appear individually, including:

- four YAML configuration files;
- legacy matrices, summaries, logs, previews, plots, and fitting outputs;
- seven Python source files;
- six Python 3.10 bytecode cache files;
- one legacy plotting test.

Each row records:

- current relative path;
- purpose based on actual contents or retained code;
- old logical path;
- inventory classification;
- whether it is a core `formal_source`.

Special cases must be explicit:

- `results/llpr_test_details.npz` and
  `results/llpr_test_dry_run.log` are `legacy_smoke`;
- `reliability_matpes_linear_fit_summary.json` is `incomplete` because its
  referenced linear-fit PNGs are absent;
- `reliability_matpes_llpr.png` is `orphan` because the legacy plotting summary
  does not list it as an official output;
- `.pyc` files are historical Python 3.10 caches, not source of truth.

`authoritative` means the file was retained as an intact audited artifact. It
does not mean every such file directly feeds the normalized numerical stages.

## NPZ field reference

The README will document:

- both arrays, dimensions, and dtypes in normalized `curvature.npz`;
- all 44 field names in normalized `details.npz`, grouped by structure, energy,
  force component, atom aggregation, and structure aggregation;
- row counts and index/offset semantics from the real arrays;
- the difference between file-level recompression and array-level equality.

The legacy formal details file and normalized details file have different
container SHA values but all 44 arrays are exactly equal, including NaN-aware
comparison where applicable.

## Empty plots directory

The README must state that normalized `plots/` is currently empty. It must not
document the previously generated derived plot files as present. The old
PNG/PDF files retained under `legacy_raw/results/LLPR` remain documented as
exact legacy copies.

## Validation

After editing, validation must:

1. enumerate the current formal tree and confirm 64 documented file paths;
2. compare documented `legacy_raw` paths with all 52 inventory rows;
3. compare documented normalized paths with all files outside `legacy_raw`;
4. open both normalized NPZ files and verify the documented 2 and 44 fields;
5. run full artifact verification, expecting 4 manifests and 60 verified files;
6. run `tox -e lint`;
7. run `git diff --check`.

No formal data files are modified by this task.
