# UPET LLPR Correctness Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct LLPR recovery integrity and split provenance while republishing the authoritative legacy numerical results without full-dataset model recomputation.

**Architecture:** Add one hash-verifying completed-stage loader and one strict curvature-progress validator, then make every workflow stage consume those interfaces. Separate imported build, validation, and test provenance, represent unavailable legacy validation diagnostics explicitly, and publish a fully verified corrected tree transactionally at the existing formal path.

**Tech Stack:** Python 3.11, PyTorch, NumPy, Pydantic, ASE, pytest, tox, YAML, Git.

## Global Constraints

- Activate the runtime with `conda activate upet_new`.
- Do not run checkpoint inference, Jacobians, curvature construction, calibration, or evaluation over the formal full datasets.
- Formal H, fixed eta, Alpha, and all evaluation detail arrays must remain identical to the authoritative legacy sources.
- Real model computation is permitted only for the n20 acceptance suite.
- Use tests first and observe the expected failure before changing production code.
- Generate the corrected formal tree outside `matpes_r2_legacy`, verify it, promote it, verify the final path, and delete the old backup only after success.
- Preserve the current formal tree and restore it if final-path verification fails.

---

## File map

- `Uncertainty_Quantification/LLPR/llpr/artifacts.py`: completed-stage containment, SHA, and NPZ verification.
- `Uncertainty_Quantification/LLPR/llpr/curvature.py`: strict curvature progress validation and build-stage consumption.
- `Uncertainty_Quantification/LLPR/llpr/calibration.py`: verified upstream loading.
- `Uncertainty_Quantification/LLPR/llpr/inference.py`: applied-calibration parsing and verified upstream loading.
- `Uncertainty_Quantification/LLPR/llpr/legacy.py`: split-specific legacy provenance and nullable unavailable diagnostics.
- `Uncertainty_Quantification/LLPR/llpr/plotting.py`: verified evaluation loading and derived legacy provenance.
- `Uncertainty_Quantification/LLPR/configs/import_legacy.yaml`: audited build, validation, and test expectations.
- `Uncertainty_Quantification/LLPR/tests/`: regression and real n20 acceptance tests.
- `Uncertainty_Quantification/LLPR/MIGRATION_REPORT.md`: corrected identities, counts, and verification evidence.

---

### Task 1: Verify completed artifacts before reuse or consumption

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/artifacts.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/curvature.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/calibration.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/inference.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/plotting.py`
- Test: `Uncertainty_Quantification/LLPR/tests/test_artifacts.py`

**Interfaces:**
- Produces: `load_verified_manifest(path: Path, expected_identity: Mapping[str, object] | None = None, *, verify_npz: bool = False) -> dict[str, object]`
- Consumes: existing `load_complete_manifest`, `sha256_file`, and `_verify_npz`.

- [ ] **Step 1: Add failing containment and tamper tests**

Add tests that create a complete manifest declaring `../outside.bin` and a
manifest whose declared file content is changed after publication:

```python
def test_verified_manifest_rejects_escaping_declared_path(tmp_path: Path) -> None:
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"data")
    stage = tmp_path / "stage"
    stage.mkdir()
    atomic_json_dump(
        stage / "manifest.json",
        {"status": "complete", "identity": "x", "files": {"../outside.bin": sha256_file(outside)}},
    )
    with pytest.raises(ValueError, match="escapes"):
        load_verified_manifest(stage / "manifest.json")


def test_verified_manifest_rejects_tampered_declared_file(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    stage.mkdir()
    artifact = stage / "artifact.bin"
    artifact.write_bytes(b"original")
    atomic_json_dump(
        stage / "manifest.json",
        {"status": "complete", "identity": "x", "files": {"artifact.bin": sha256_file(artifact)}},
    )
    artifact.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA mismatch"):
        load_verified_manifest(stage / "manifest.json")
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
tox -e llpr-tests -- -k "verified_manifest" -q
```

Expected: collection/import failure because `load_verified_manifest` does not exist.

- [ ] **Step 3: Implement the verified loader**

The loader must call `load_complete_manifest`, validate every declared relative
path, resolve it under `manifest_path.parent`, verify SHA, and optionally run
`_verify_npz` for NPZ files. Update `_verify_declared_files` to reuse the same
safe path resolution.

- [ ] **Step 4: Route all stage reuse and upstream consumption through it**

Replace completed-stage calls in build, calibration, evaluation, and plotting.
Use `verify_npz=True` when the consumer will load a declared NPZ.

- [ ] **Step 5: Run focused and fast tests**

```bash
tox -e llpr-tests -- -k "artifact or cli or plotting" -q
tox -e llpr-tests -- -m "not llpr_n20 and not llpr_legacy" -q
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr Uncertainty_Quantification/LLPR/tests/test_artifacts.py
git commit -m "fix: verify LLPR artifacts before consumption"
```

---

### Task 2: Make curvature recovery reject inconsistent progress

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/curvature.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_curvature.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_n20.py`

**Interfaces:**
- Produces: extended `load_curvature_progress(path, *, expected_identity, expected_energy_dimension, expected_force_dimension, expected_structure_count)`.
- Consumes: `DatasetIdentity.structure_count` and discovered readout dimensions.

- [ ] **Step 1: Add failing corrupt-progress unit tests**

Parameterize mutations for:

- `next_structure_index != structure_count`;
- next index larger than expected build structures;
- force count different from `3 * atom_count`;
- wrong matrix shape;
- non-finite matrix value;
- non-symmetric matrix.

Each mutation must raise `ValueError` with a specific progress-validation message.

- [ ] **Step 2: Run unit tests and verify RED**

```bash
tox -e llpr-tests -- -k "curvature and progress" -q
```

Expected: the corrupt cases do not raise.

- [ ] **Step 3: Implement strict validation**

Load scalar counts as integers, reject booleans/non-integral arrays, validate
dimensions and matrix properties, and enforce:

```python
next_index == accumulator.structure_count
next_index <= expected_structure_count
accumulator.force_component_count == 3 * accumulator.atom_count
```

- [ ] **Step 4: Add the real n20 interruption test**

Use a temporary output root. Interrupt `compute_structure_jacobians` immediately
after a completed checkpoint, assert progress exists, resume, then compare both
curvature arrays and diagnostics against an uninterrupted temporary run.

- [ ] **Step 5: Verify GREEN**

```bash
tox -e llpr-tests -- -k "curvature and progress" -q
UPET_RUN_LLPR_N20=1 tox -e llpr-tests -- -m llpr_n20 -v
```

Expected: unit tests and both n20 tests pass.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/curvature.py Uncertainty_Quantification/LLPR/tests/test_curvature.py Uncertainty_Quantification/LLPR/tests/test_n20.py
git commit -m "fix: validate LLPR curvature recovery state"
```

---

### Task 3: Separate imported build, validation, and test provenance

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/legacy.py`
- Modify: `Uncertainty_Quantification/LLPR/configs/import_legacy.yaml`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_legacy.py`

**Interfaces:**
- Produces Pydantic contracts:
  - `LegacyBuildCounts(structures: int)`
  - `LegacyValidationCounts(energy: int, force_components: int)`
  - existing `LegacyCounts` renamed or used as test counts.
- Consumes legacy `Hef_full_run_summary.json`,
  `alpha_val_full_joint_summary.json`, and formal test summary/details.

- [ ] **Step 1: Extend the synthetic legacy fixture with distinct split summaries**

Write a build summary with 7 structures, a validation summary with 2 energy
samples and 9 force components, and a test result with 1 structure and 3 force
components. Add assertions that canonical diagnostics preserve 7, 2/9, and 1/3
in their respective stages.

- [ ] **Step 2: Run and verify RED**

```bash
tox -e llpr-tests -- -k "legacy and split" -q
```

Expected: configuration validation or count assertions fail because one test
count object is reused.

- [ ] **Step 3: Implement split contracts and source-summary validation**

The importer must validate:

```text
build summary num_structures_used == expected_build_counts.structures
validation summary energy_calibration.valid_count == expected_validation_counts.energy
validation summary force_calibration.valid_count == expected_validation_counts.force_components
test details and summary == expected_test_counts
```

Curvature diagnostics record only the recoverable build structure count.

- [ ] **Step 4: Update the audited formal config**

Set:

```yaml
expected_build_counts:
  structures: 348780
expected_validation_counts:
  energy: 19370
  force_components: 458877
expected_test_counts:
  structures: 19374
  atoms: 149321
  force_components: 447963
```

- [ ] **Step 5: Verify GREEN**

```bash
tox -e llpr-tests -- -k legacy -q
```

Expected: all legacy import tests pass.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/legacy.py Uncertainty_Quantification/LLPR/configs/import_legacy.yaml Uncertainty_Quantification/LLPR/tests/test_legacy.py
git commit -m "fix: preserve LLPR split provenance"
```

---

### Task 4: Represent unavailable legacy validation diagnostics honestly

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/inference.py`
- Modify: `Uncertainty_Quantification/LLPR/llpr/legacy.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_legacy.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_inference.py`

**Interfaces:**
- Produces: `AppliedCalibration(target: str, eta: float, alpha: float, alpha_sq: float)`.
- Changes `_load_calibration(path: Path) -> dict[str, AppliedCalibration]`.
- Imported selected records contain nullable `gaussian_nll` and coverage plus `diagnostics_status`.

- [ ] **Step 1: Add failing imported-diagnostics assertions**

Assert imported calibration selected records use validation counts, contain:

```python
assert selected["gaussian_nll"] is None
assert selected["coverage_1sigma"] is None
assert selected["coverage_2sigma"] is None
assert selected["coverage_3sigma"] is None
assert selected["diagnostics_status"] == "unavailable_from_legacy_summary"
```

Also verify `_load_calibration` accepts the imported summary and returns eta and
Alpha squared.

- [ ] **Step 2: Run and verify RED**

```bash
tox -e llpr-tests -- -k "legacy or load_calibration" -q
```

Expected: imported diagnostics contain test-derived floats and counts.

- [ ] **Step 3: Implement applied-calibration parsing and nullable diagnostics**

Keep `CalibrationRecord` unchanged for recomputed selection. The legacy importer
must construct a separate JSON record rather than call `_calibration_record` with
test details. `_load_calibration` validates finite positive eta/Alpha fields and
does not require diagnostic fields.

- [ ] **Step 4: Verify GREEN**

```bash
tox -e llpr-tests -- -k "legacy or inference" -q
```

Expected: imported and recomputed calibration loading tests pass.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/LLPR/llpr/legacy.py Uncertainty_Quantification/LLPR/llpr/inference.py Uncertainty_Quantification/LLPR/tests/test_legacy.py Uncertainty_Quantification/LLPR/tests/test_inference.py
git commit -m "fix: distinguish legacy calibration diagnostics"
```

---

### Task 5: Bind plot provenance to its evaluation source

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/llpr/plotting.py`
- Modify: `Uncertainty_Quantification/LLPR/tests/test_plotting.py`

**Interfaces:**
- Plot manifest adds `origin` and `source_origin`.
- Consumes verified evaluation manifest `origin`.

- [ ] **Step 1: Add a failing provenance test**

Create an evaluation manifest with `origin: legacy_import`, run plotting, and
assert:

```python
assert plot_manifest["origin"] == "derived"
assert plot_manifest["source_origin"] == "legacy_import"
```

- [ ] **Step 2: Run and verify RED**

```bash
tox -e llpr-tests -- -k "plot and provenance" -q
```

Expected: the fields are absent.

- [ ] **Step 3: Add the two manifest fields and verified source loading**

Do not label generated image files as copied legacy artifacts.

- [ ] **Step 4: Verify GREEN and commit**

```bash
tox -e llpr-tests -- -k plotting -q
git add Uncertainty_Quantification/LLPR/llpr/plotting.py Uncertainty_Quantification/LLPR/tests/test_plotting.py
git commit -m "fix: record LLPR plot provenance"
```

---

### Task 6: Full verification before formal publication

**Files:**
- Modify: `Uncertainty_Quantification/LLPR/README.md`
- Modify after publication: `Uncertainty_Quantification/LLPR/MIGRATION_REPORT.md`

**Interfaces:**
- Consumes all earlier task interfaces.
- Produces a verified corrected candidate formal tree.

- [ ] **Step 1: Run all code verification**

```bash
tox -e llpr-tests -- -m "not llpr_n20 and not llpr_legacy" -q
UPET_RUN_LLPR_N20=1 tox -e llpr-tests -- -m llpr_n20 -v
tox -e lint
git diff --check
```

- [ ] **Step 2: Request independent full code review**

Review from `d8c6349` to current HEAD. Fix all Critical and Important findings
with new failing tests, then rerun Step 1.

- [ ] **Step 3: Generate a corrected candidate from legacy files**

Use a temporary experiment name and adjusted import config so the current
`matpes_r2_legacy` directory remains untouched. Do not load the checkpoint.
Generate plots from the candidate evaluation.

- [ ] **Step 4: Verify the candidate**

Run full verify and a read-only audit asserting:

- 52 source/raw hashes match;
- energy and force curvature blocks are exactly equal to old blocks;
- all 44 details arrays are exactly equal to old details;
- build structure count is 348,780;
- validation counts are 19,370 and 458,877;
- test counts are 19,374, 149,321, and 447,963;
- legacy NLL and coverage are null;
- no formal numerical manifest has `origin: recomputed`;
- plot source origin is `legacy_import`.

- [ ] **Step 5: Update documentation and commit**

Record the corrected candidate identities and fresh verification results.

```bash
git add Uncertainty_Quantification/LLPR/README.md Uncertainty_Quantification/LLPR/MIGRATION_REPORT.md
git commit -m "docs: record corrected LLPR migration"
```

---

### Task 7: Promote the corrected tree and remove the old version

**Files:**
- Replace generated directory:
  `Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy`

**Interfaces:**
- Consumes the verified candidate from Task 6.
- Produces the corrected formal tree at the original path.

- [ ] **Step 1: Resolve and validate exact paths**

Confirm candidate, current formal path, and backup are sibling paths under:

```text
/home/lilong/code/UQ/upet_new/Uncertainty_Quantification/LLPR/outputs
```

Abort if the candidate lacks a complete root manifest or if a pre-existing backup
path is present.

- [ ] **Step 2: Create the recoverable backup and promote**

Rename current `matpes_r2_legacy` to a unique sibling backup, then rename the
candidate to `matpes_r2_legacy`. Do not copy across filesystems.

- [ ] **Step 3: Verify the final path**

Repeat Task 6 Step 4 at the final path, rerun idempotent import, and assert
canonical SHA and mtimes do not change.

- [ ] **Step 4: Roll back on failure**

If any final-path check fails, move the failed new directory aside and rename the
backup back to `matpes_r2_legacy`. Report the failure and retain the failed
candidate for diagnosis.

- [ ] **Step 5: Delete the old backup after success**

Resolve the absolute backup path, confirm it is the exact sibling created in Step
2, and remove only that backup. Report that deletion is irreversible.

- [ ] **Step 6: Final verification and status**

```bash
python -m Uncertainty_Quantification.LLPR.llpr verify \
  --config Uncertainty_Quantification/LLPR/outputs/matpes_r2_legacy
git status --short
git log -1 --oneline
```

Expected: full verification passes, no source changes remain uncommitted, and no
backup/candidate/progress/shard files remain.
