# UPET FGE Three-Dataset Inference and Plotting Acceptance Report

## Scope and final plotting contract

This run reused the completed K=8 FGE ensemble and performed no training,
checkpoint sweep, checkpoint-count curve, or cross-dataset correlation plot.
MATPES test was plot-only, while MAD test and MATPES train used the existing
inference-only prediction and population-STD UQ stages. Each dataset has its own
plot directory.

The final filenames follow the subsequently confirmed user-facing contract,
which supersedes the earlier design draft's `raw_*` names:

- Energy, Force, and, where referenced, Stress use
  `*_uncertainty_vs_absolute_residual.{png,pdf}`.
- Every directory contains `plot_statistics.json` and `plot_manifest.json`.
- MAD has no stress plot because its source has no stress reference.

## Immutable inputs

- Remote checkout: `/home/bywang/code/UQ/upet_new_fge_test`
- Ensemble root:
  `Uncertainty_Quantification/FGE/outputs/upet-FGE-CKPT-UQ-v1.0-full-v3`
- Ensemble result-manifest SHA-256:
  `b1f4a3c5c7713b4be369b467e98e62d59bc23a13aefceffbd49ac51773b6dc86`
- Base checkpoint SHA-256:
  `879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004`
- Ordered members: `member_001` through `member_008`
- MAD dataset SHA-256:
  `d9a1280246a7a678f699e7654aebd29e4273ab6dcd1dfb4f74334a9b15edb66b`
- MATPES train dataset SHA-256:
  `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec`

## Real-data execution

### MATPES test

The existing completed prediction/UQ result was reopened read-only and plotted
with PET loading, inference, training, backward, and optimizer-step guards.
No inference guard was tripped and the completed-result metadata/hash snapshots
were unchanged.

- Domains: Energy, Force, Stress
- Final files: 8
- Plot manifest SHA-256:
  `262a7fd41700dd018aa3dfbe8f4a8a7692d211eca8d6d2f41bb842ad0a2cfddb`

### MAD test

- Formal output:
  `Uncertainty_Quantification/FGE/outputs/inference_mad_test_k8`
- Prediction: 78 chunks, 9,486 structures, 258,586 atoms, K=8
- Prediction elapsed time: 2:10:29
- Prediction peak RSS: approximately 204.6 GiB; swap count 0
- UQ elapsed time: 41 seconds
- References: Energy and Force present; Stress absent
- Formula: population standard deviation (`unbiased=False`, denominator K)
- Read-only validation: two `PASS/read_only` runs
- Validator artifact count: 159 declared artifacts; 160 actual files
- Exact before/after snapshot digest:
  `11634ea4ddbcdc883d87a3bf7ba4c4b98735e805af12096dc0a5c1a8d0050499`
- Final files: 6
- Plot manifest SHA-256:
  `b09acbc88d01f944366e3c29f74727abada48b14083b3cdee7401de4b01e1e35`

### MATPES train

- Formal output:
  `Uncertainty_Quantification/FGE/outputs/inference_matpes_train_k8`
- Prediction manifest SHA-256:
  `ddb18036bacebe52b7edfe96f0e3b60fa16dab7508a3829f9d443c5830151c8a`
- Prediction: 682 chunks, 348,780 structures, 2,753,112 atoms, K=8
- Prediction elapsed time: 28:55:10
- Prediction peak RSS: 160,235,684 KiB (approximately 152.8 GiB)
- Prediction swap count: 0
- UQ: 682 chunks, Energy/Force/Stress references present
- UQ manifest SHA-256:
  `d0def1c76c4529b78ab430bd5da5daf11c5ab14adfa11e29ad77f929904ce25e`
- UQ wall interval from guarded script publication to final manifest:
  approximately 14:50
- Formula version: `legacy_upet_fge_v1`, population standard deviation
- UQ guards: member inference, training, backward, and optimizer-step guards
  were active and were not tripped
- Read-only validation: two complete runs, followed by a fresh release run
- Fresh release validation: PASS/read_only, 1,367 declared artifacts, 5:26.34 elapsed, 2,342,776 KiB peak RSS, 0 filesystem outputs, exit 0
- Exact pre-validation/after-run-1/after-run-2 snapshot file SHA-256:
  `1e130f66faa103fb8559f308d5ffc4d1c0a57728ea3de97a351ca4e905d9b929`
- Actual source-file inventory: 1,368 files
- Post-plot source snapshot: 1,368 files, exact match; canonical snapshot
  digest `e0b97731fff3af6a75e2582900b59c254dfa66023194564ada160d670a46db5d`
- Final files: 8
- Plot manifest SHA-256:
  `32c4678f0d1262ee2339b0cc0bf8ab66a67afe45fd77bf9b74b2a0a9c4989cd5`

The formal prediction was not repeated. The earlier failed inline UQ invocation
failed at shell/Python command transport before application code ran and did not
create formal UQ artifacts. The successful UQ and plot stages used transferred
temporary guard scripts, avoiding inline quoting.

## Plot acceptance

The three directories were synced locally under
`Uncertainty_Quantification/Plots/FGE/` and checked independently:

- Exact file inventories are 8 (`matpes_test`), 6 (`mad_test`), and 8
  (`matpes_train`).
- Every PNG is 2100 x 2100 and decodes successfully.
- Every PDF begins with a valid `%PDF-` header.
- Every declared output byte count and SHA-256 matches its plot manifest.
- No filename contains `sweep` or `correlation`.
- Source identities are respectively
  `b1f4a3c5...b6dc86`, `daafcfa1...ee941`, and `75d52fa5...e1188d`.

## Runtime guards and retained evidence

Prediction was never relaunched after the formal MATPES train manifest became
complete. UQ and plotting ran with guards against training, backward, optimizer
steps, and prediction-member inference as appropriate. All guard scripts printed
their active/not-tripped success markers before their stages were accepted.

No remote smoke, failed-command, or intermediate evidence was deleted. This
includes the MATPES train smoke output
`outputs/smoke_matpes_train_k8_s512_a8192`, the full prediction log/PID/wrapper,
the MAD smoke/formal outputs, validation snapshots in `/tmp`, and the failed
inline-command evidence. Large prediction and UQ artifacts remain remote and
were not copied into Git.

## Verification

- Full FGE suite: `483 passed, 2 skipped` in 41.78 seconds
- Ruff format: 95 files already formatted
- Ruff lint: all checks passed
- mypy: no issues in 126 source files
- Local plot manifest/inventory/image/PDF verification: passed for all three
  datasets
- `git diff --check`: passed before final staging and repeated before commit
- Read-only acceptance audit: no implementation defects found; final artifact checks above were executed by the primary agent

Only the three FGE plot directories and this report are staged for the final
commit. Unrelated `.idea`, ConfidenceHead/LLPR plot directories, BootStrapping
tests, and temporary local verification scripts are excluded.
