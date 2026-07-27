# UPET LLPR Correctness Remediation Design

Date: 2026-07-27

## Context

The migrated legacy LLPR numerical artifacts are intact:

- the canonical energy and force curvature blocks exactly equal the authoritative
  legacy blocks;
- all 44 arrays in canonical evaluation details exactly equal the authoritative
  legacy test details;
- fixed eta and Alpha match the legacy validation result;
- all 52 raw legacy files match their source hashes.

The second audit nevertheless found four correctness gaps:

1. corrupted curvature progress can silently skip build structures;
2. imported curvature diagnostics contain test counts instead of build provenance;
3. imported calibration diagnostics are recomputed from test details while the
   stage claims validation provenance;
4. completed upstream artifacts can be consumed without checking their declared
   file hashes.

This remediation must not recompute the formal full dataset. It may read and
convert legacy artifacts and summaries. Real model computation remains limited to
the n20 acceptance tests.

## Selected approach

Generate a corrected formal tree in staging, validate it, preserve the current
formal tree as a temporary backup, atomically promote the corrected tree to the
existing `matpes_r2_legacy` path, validate the published path, then delete the
backup. If validation after promotion fails, restore the backup.

The old formal tree is deleted only after the corrected formal tree passes all
checks at its final path.

## Curvature progress validation

`load_curvature_progress` will validate:

- the progress identity;
- exact energy and force matrix dimensions expected by the discovered readout;
- finite and symmetric matrices;
- non-negative structure, atom, force-component, and next-index counts;
- `next_structure_index == structure_count`;
- `next_structure_index <= build.structure_count`;
- `force_component_count == 3 * atom_count`.

`run_build` will pass the expected dimensions and build structure count into this
validator. A corrupt progress file must fail before dataset iteration and must
never publish a completed curvature artifact.

An n20 acceptance test will interrupt curvature after an atomic checkpoint,
resume it, and compare the final matrices and diagnostics with an uninterrupted
run.

## Complete artifact consumption

A unified complete-stage loader will:

- require a complete manifest;
- validate requested identity fields;
- reject absolute declared file paths and paths containing `..`;
- ensure every declared file remains within the manifest directory;
- verify every declared file SHA before returning the manifest;
- optionally validate declared NPZ arrays as finite.

Build, calibration, evaluation, and plotting will use this loader both when
reusing their own completed stage and when consuming upstream stages. The
standalone full verifier will use the same containment and hash rules.

## Split-specific legacy provenance

The legacy import contract will represent three independent stages:

- build provenance: 348,780 structures, sourced from
  `Hef_full_run_summary.json`;
- validation provenance: 19,370 energy samples and 458,877 force components,
  sourced from `alpha_val_full_joint_summary.json`;
- test provenance: 19,374 structures, 149,321 atoms, and 447,963 force
  components, sourced from `llpr_test_full_gpu_summary.json` and details.

The importer will validate these source summaries rather than reuse one common
test-count object for all stages. Build atom and force-component counts are not
recoverable from the saved build summary and will be omitted instead of guessed.

These split-specific expectations become part of the root import identity, so the
corrected artifact tree receives a new identity.

## Legacy calibration representation

The selected fixed calibration retains:

- target;
- eta;
- Alpha and Alpha squared;
- condition number and warning;
- validation count.

Validation Gaussian NLL and coverage cannot be recovered from the saved aggregate
summary because per-sample validation residuals and quadratic forms were not
saved. These fields will therefore be `null`, with:

```text
diagnostics_status = unavailable_from_legacy_summary
```

Test diagnostics will not be substituted into calibration. Evaluation runtime
loading will use a small applied-calibration representation containing only the
fields required for inference, chiefly eta and Alpha squared. Recomputed fixed
and fitted calibrations retain their complete NLL and coverage records.

## Plot provenance

Plot manifests generated from a migrated evaluation will record that they are
derived from a legacy-import evaluation. This distinguishes regenerated display
artifacts from recomputed model outputs without claiming that the image files
were copied from the old tree.

## Publication and rollback

The corrected importer output is first generated at a temporary sibling path.
Before promotion:

1. run full manifest and file verification;
2. compare canonical curvature blocks to the legacy matrices;
3. compare every canonical evaluation details array to legacy details;
4. validate split-specific counts and nullable legacy calibration diagnostics.

Promotion then:

1. renames the current formal directory to a temporary backup;
2. renames the corrected staging directory to `matpes_r2_legacy`;
3. repeats full verification and source-to-canonical comparisons;
4. restores the backup on any failure;
5. deletes the backup only after final-path validation succeeds.

No full checkpoint loading, forward pass, Jacobian, curvature, calibration, or
test evaluation is permitted during formal publication.

## Test strategy

Test-driven changes will cover:

- corrupted curvature next index, counts, dimensions, symmetry, and finite values;
- real n20 interrupted curvature recovery equivalence;
- rejected consumption of a tampered complete upstream artifact;
- safe containment of manifest-declared paths;
- synthetic legacy import with deliberately different build, validation, and
  test counts;
- nullable validation-only diagnostics for imported calibration;
- plot provenance inherited from legacy evaluation;
- exact formal old-to-new matrix and details comparison;
- idempotent import and complete verification of all raw and canonical files.

Final verification will run:

- fast LLPR tests;
- real n20 fixed and fitted paths, calibration recovery, and curvature recovery;
- formatting, lint, mypy, and sphinx-lint;
- formal staging verification;
- final-path verification after promotion;
- repeated import with unchanged canonical SHA and modification times;
- an independent full code review.

## Success criteria

The remediation is complete only when:

- no corrupted progress can silently skip a build, calibration, or evaluation
  structure;
- all completed artifacts are hash-verified before reuse or consumption;
- canonical build, validation, and test provenance is split-correct;
- no test-derived metric is represented as validation calibration evidence;
- formal H, fixed eta, Alpha, and all evaluation arrays remain unchanged from the
  authoritative legacy sources;
- the corrected tree is published at `matpes_r2_legacy`;
- the old tree backup is deleted only after final verification succeeds.
