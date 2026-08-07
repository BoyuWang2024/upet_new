# UPET ConfidenceHead Result Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reusable, Carnet-style plotting for the nine completed UPET ConfidenceHead runs and generate the validated plots on the remote server.

**Architecture:** A core `plot_analysis.py` module owns verified artifact loading, target-specific semantics, 50-bin statistics, correlation checks, and rendering. Three thin scripts expose single-run, energy-correlation, and complete-nine-run workflows; all real-run discovery is strict and refuses ambiguous duplicates. Existing `verify_run(full=True)` remains the integrity gate before any prediction is loaded.

**Tech Stack:** Python 3.11, PyTorch, NumPy, SciPy, Matplotlib with the non-interactive `Agg` backend, CSV/JSON/YAML from the standard library and PyYAML, pytest, tox, ruff, mypy.

## Global Constraints

- Plot only existing results; never modify checkpoints, resolved configs, logs, manifests, binning artifacts, or prediction artifacts.
- Energy observed errors are already per atom and must not be divided by atom count again.
- Force observed errors use one `atom_mean` value per atom and must not be expanded into Cartesian components.
- Require exactly 50 fixed-linear bins and display all 50 positions, including empty bins labeled `n=0`.
- Require one completed force-only run and one completed energy-only run for every order from 1 through 8.
- Generate PNG at 300 DPI, vector PDF, and CSV outputs with English labels and Carnet-style symlog boxplots.
- Recompute Pearson and Spearman from expected versus observed energy errors and cross-check `evaluation/metrics.json` before comparison plotting.
- Local execution is limited to static checks and synthetic test data; real nine-run plotting runs remotely on CPU.

---

## File Structure

- Create `Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py`: validated data model, discovery, statistics, correlation checks, atomic writers, and rendering.
- Create `Uncertainty_Quantification/ConfidenceHead/scripts/plot_argmax_bin_boxplots.py`: one-run command.
- Create `Uncertainty_Quantification/ConfidenceHead/scripts/plot_energy_order_correlations.py`: explicit eight-energy-run command.
- Create `Uncertainty_Quantification/ConfidenceHead/scripts/plot_completed_runs.py`: strict automatic/explicit nine-run command.
- Create `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py`: data semantics, validation, statistics, rendering, and discovery tests.
- Create `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_scripts.py`: thin-entrypoint argument forwarding tests.
- Modify `Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt`: add Matplotlib as an explicit plotting dependency.
- Modify `tox.ini`: add Matplotlib to the ConfidenceHead test environment.
- Modify `Uncertainty_Quantification/ConfidenceHead/README.md`: document commands and output locations.

### Task 1: Verified plot-series loading and numerical analysis

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py`

**Interfaces:**
- Consumes: `workflows.verify.verify_run(run_dir, full=True)`, `artifacts.load_verified_torch`, run `resolved_config.yaml`, `binning.json`, `evaluation/metrics.json`, and `evaluation/test_predictions.pt`.
- Produces: `PlotSeries`, `BinRow`, `CorrelationRow`, `load_plot_series(Path)`, `bin_rows(PlotSeries)`, and `energy_correlation(PlotSeries)`.

- [ ] **Step 1: Add a synthetic completed-run fixture and failing energy/force loading tests**

Create a helper in `test_plot_analysis.py` that writes a minimal run tree and monkeypatches `plot_analysis.verify_run` so these unit tests isolate plotting semantics:

```python
def _write_plot_run(
    tmp_path: Path,
    *,
    target: Literal["energy", "force"],
    order: int = 1,
) -> Path:
    run_dir = tmp_path / f"run-{target}-order{order}"
    evaluation = run_dir / "evaluation"
    evaluation.mkdir(parents=True)
    config = {
        "binning": {"algorithm": "fixed_linear_v1"},
        "model": {
            "force": {"enabled": True, "num_bins": 50, "target_mode": "atom_mean"},
            "energy": {"enabled": True, "num_bins": 50, "cumulant_order": order},
        },
        "loss": {
            "force_coefficient": 1.0 if target == "force" else 0.0,
            "energy_coefficient": 1.0 if target == "energy" else 0.0,
        },
    }
    (run_dir / "resolved_config.yaml").write_text(
        yaml.safe_dump(config), encoding="utf-8"
    )
    representatives = torch.linspace(0.003, 0.297, 50)
    (run_dir / "binning.json").write_text(
        json.dumps(
            {
                "force": {
                    "algorithm": "fixed_linear_v1",
                    "num_bins": 50,
                    "max_error": 0.5,
                    "target_mode": "atom_mean",
                    "error_definition": "abs_cartesian_component_mean_v1",
                    "thresholds": torch.linspace(0.01, 0.49, 49).tolist(),
                    "representatives": torch.linspace(0.005, 0.495, 50).tolist(),
                },
                "energy": {
                    "algorithm": "fixed_linear_v1",
                    "num_bins": 50,
                    "max_error": 0.3,
                    "thresholds": torch.linspace(0.006, 0.294, 49).tolist(),
                    "representatives": representatives.tolist(),
                },
            }
        ),
        encoding="utf-8",
    )
    prefix = target
    logits = torch.full((4, 50), -10.0)
    logits[torch.arange(4), torch.tensor([0, 1, 1, 49])] = 10.0
    observed = torch.tensor([0.01, 0.02, 0.04, 0.5])
    expected = torch.tensor([0.01, 0.03, 0.05, 0.4])
    payload = {
        "structure_ids": torch.arange(4),
        f"{prefix}_logits": logits,
        f"{prefix}_labels": torch.tensor([0, 1, 1, 49]),
        f"{prefix}_observed_errors": observed,
        f"{prefix}_expected_errors": expected,
        f"{prefix}_representatives": representatives,
        "atom_offsets": torch.arange(5),
    }
    if target == "force":
        payload["force_target_mode"] = "atom_mean"
        payload["force_error_definition"] = "abs_cartesian_component_mean_v1"
    torch.save(payload, evaluation / "test_predictions.pt")
    metrics = {target: {"sample_count": 4, "pearson": 0.99, "spearman": 0.8}}
    (evaluation / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    return run_dir
```

Add tests asserting that energy has order 1 and four structure samples, while force has `order is None`, four atomic samples, and `target_mode == "atom_mean"`. Add negative tests for a second normalization of energy, force component expansion, a non-50 logits width, non-finite errors, wrong coefficients, and force mode `component`.

- [ ] **Step 2: Run the focused tests and verify the missing module failure**

Run:

```bash
python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py -q
```

Expected: collection fails because `confidence_head.plot_analysis` does not exist.

- [ ] **Step 3: Implement immutable data contracts and verified loading**

Implement these public types and loader signatures:

```python
Target = Literal["energy", "force"]

@dataclass(frozen=True)
class PlotSeries:
    run_dir: Path
    structure_ids: torch.Tensor
    target: Target
    order: int | None
    logits: torch.Tensor
    observed: torch.Tensor
    expected: torch.Tensor
    representatives: torch.Tensor
    stored_metrics: Mapping[str, int | float]
    force_target_mode: str | None

@dataclass(frozen=True)
class BinRow:
    bin: int
    sample_count: int
    mean_observed_error: float | None
    median_observed_error: float | None
    std_observed_error: float | None
    q25_observed_error: float | None
    q75_observed_error: float | None
    min_observed_error: float | None
    max_observed_error: float | None
    mean_expected_error: float | None

@dataclass(frozen=True)
class CorrelationRow:
    order: int
    sample_count: int
    pearson: float
    spearman: float


```

The loader must validate exactly one positive coefficient, fixed-linear binning, 50 representatives, `[N, 50]` logits, matching one-dimensional observed/expected arrays, non-negative finite errors, and target-specific semantics. Load predictions with `load_verified_torch(root / evaluation / test_predictions.pt, expected_sha256=None, weights_only=False)` after `verify_run(full=True)` succeeds. Preserve stored observed values unchanged.

Compute Pearson/Spearman with the same double-precision centering and average-tie ranks used by `confidence_head.metrics`; reject fewer than two samples or constant inputs before comparing to stored metrics with `math.isclose`.

- [ ] **Step 4: Add and run exact bin-statistics and correlation tests**

Assert all of the following:

```python
rows = bin_rows(series)
assert len(rows) == 50
assert rows[0].sample_count == 1
assert rows[1].sample_count == 2
assert rows[2].sample_count == 0
assert rows[2].mean_observed_error is None
assert sum(row.sample_count for row in rows) == len(series.observed)
```

Build expected/observed pairs with analytically known perfect positive and negative correlations. Add failures for stored metric disagreement and constant data.

Run:

```bash
python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py -q
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit the verified analysis layer**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py
git commit -m "feat: validate confidence head plot data"
```

### Task 2: Atomic CSV and Carnet-style plot rendering

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt`
- Modify: `tox.ini`

**Interfaces:**
- Consumes: `PlotSeries`, `BinRow`, and `CorrelationRow` from Task 1.
- Produces: `write_bin_csv`, `write_correlation_csv`, `plot_single_boxplot`, `plot_combined_energy_boxplots`, `plot_combined_force_boxplot`, and `plot_energy_correlations`.

- [ ] **Step 1: Declare Matplotlib and add failing output tests**

Add `matplotlib` to `requirements-remote.txt` and the dependency list of `[testenv:confidence-head-tests]` in `tox.ini`.

Add tests that call each writer under `matplotlib.use("Agg")` and assert:

```python
assert png_path.is_file() and png_path.stat().st_size > 0
assert pdf_path.is_file() and pdf_path.stat().st_size > 0
with csv_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 50
assert rows[2]["sample_count"] == "0"
assert rows[2]["mean_observed_error"] == ""
```

For the correlation CSV, assert the order column is exactly `1,2,3,4,5,6,7,8`.

- [ ] **Step 2: Run the focused output tests and verify missing functions**

Run:

```bash
python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py -k "csv or plot" -q
```

Expected: failures name the not-yet-defined writer and plotting functions.

- [ ] **Step 3: Implement atomic writers and rendering functions**

Add the exact public signatures named write_bin_csv, write_correlation_csv, plot_single_boxplot, plot_combined_energy_boxplots, plot_combined_force_boxplot, and plot_energy_correlations, with the parameter and return types listed in the Interfaces block above.

Use a temporary file in the destination directory, flush/fsync CSV data, save each figure to its temporary path, close the figure in `finally`, then `os.replace` the completed temporary file. Configure boxplots with all positions `0..49`, labels `"{index}\nn={count}"`, `symlog` on y, English target-specific labels, grid lines, and 300-DPI PNG. The energy combined plot uses `plt.subplots(4, 2)` in order 1–8. The force combined plot is separate. Correlation plotting uses two order-sorted lines and y limits clipped to `[-1, 1]` with small padding.

- [ ] **Step 4: Run rendering tests and inspect generated image dimensions**

Run:

```bash
python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py -q
python - <<'PY'
from pathlib import Path
from PIL import Image
for path in Path('/tmp').glob('pytest-*/**/*.png'):
    with Image.open(path) as image:
        image.verify()
PY
```

Expected: tests pass and every discovered test PNG verifies without decoding errors. If Pillow is unavailable, rely on Matplotlib `imread` in the pytest assertion rather than adding Pillow as a project dependency.

- [ ] **Step 5: Commit rendering support**

```bash
git add tox.ini Uncertainty_Quantification/ConfidenceHead/requirements-remote.txt Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py
git commit -m "feat: render confidence head result plots"
```

### Task 3: Strict nine-run discovery and batch orchestration

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py`

**Interfaces:**
- Consumes: Task 1 loaders and Task 2 writers/renderers.
- Produces: `CompletedRuns`, `discover_completed_runs`, and `plot_completed_runs`.

- [ ] **Step 1: Add failing discovery tests for exact coverage and duplicates**

Define tests around this result type:

```python
@dataclass(frozen=True)
class CompletedRuns:
    force: PlotSeries
    energy_by_order: Mapping[int, PlotSeries]
```

Create nine synthetic directories and assert orders are exactly `set(range(1, 9))`. Add explicit failures for missing order 8, duplicate order 4, two force runs, an energy run with a force coefficient, an incomplete verification result, and an unrelated directory that has no completed evaluation (the unrelated directory is ignored).

- [ ] **Step 2: Run discovery tests and verify they fail**

Run:

```bash
python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py -k "discover or completed" -q
```

Expected: failures identify missing `discover_completed_runs` and `plot_completed_runs`.

- [ ] **Step 3: Implement discovery and end-to-end orchestration**

Add the exact public signatures discover_completed_runs(runs_root: Path, explicit_run_dirs: Sequence[Path]) returning CompletedRuns, and plot_completed_runs(runs: CompletedRuns, comparisons_dir: Path) returning a tuple of artifact paths.

Automatic discovery inspects child directories containing both `manifest.json` and `evaluation/manifest.json`; explicit mode considers only the supplied paths. Load every candidate through `load_plot_series`. Require one force series, energy orders exactly 1–8, identical energy `structure_ids` ordering where available, and elementwise-equal energy observed arrays. Generate every single-run output first in its own `plots/argmax_bin_boxplots` directory. Only after all single-run data and correlations validate, generate comparison outputs under `comparisons_dir`.

Return all created paths so CLI scripts can print a deterministic artifact list.

- [ ] **Step 4: Run batch tests and verify no partial comparison output on failure**

Add a test with one deliberately mismatched order-8 observed array, assert a `ValueError`, and assert `comparisons_dir` does not exist. Then run:

```bash
python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py -q
```

Expected: all analysis and orchestration tests pass.

- [ ] **Step 5: Commit strict batch orchestration**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_analysis.py
git commit -m "feat: orchestrate completed confidence head plots"
```

### Task 4: Command-line entrypoints and user documentation

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_argmax_bin_boxplots.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_energy_order_correlations.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_completed_runs.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_plot_scripts.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/README.md`

**Interfaces:**
- Consumes: all public plotting interfaces from Tasks 1–3.
- Produces: executable `main(argv: Sequence[str] | None = None) -> int` in every script.

- [ ] **Step 1: Add failing thin-script tests**

Use `importlib.util.spec_from_file_location` as in `test_commands_scripts.py`, monkeypatch the imported plotting function, and assert the following invocations forward resolved paths:

```python
assert single.main(["--run-dir", str(run_dir)]) == 0
assert correlations.main(["--run-dir", *map(str, energy_dirs), "--output-dir", str(out)]) == 0
assert completed.main(["--runs-root", str(runs_root), "--output-root", str(out)]) == 0
```

Also test nine repeated `--run-dir` arguments for explicit selection in the completed command.

- [ ] **Step 2: Run script tests and verify entrypoints are absent**

Run:

```bash
python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_scripts.py -q
```

Expected: collection or loading fails because the three scripts do not exist.

- [ ] **Step 3: Implement the three thin scripts**

Each script must follow the existing ConfidenceHead entrypoint pattern:

```python
CONFIDENCE_HEAD_ROOT = Path(__file__).resolve().parents[1]
if str(CONFIDENCE_HEAD_ROOT) not in sys.path:
    sys.path.insert(0, str(CONFIDENCE_HEAD_ROOT))

def main(argv: Sequence[str] | None = None) -> int:
    parser = ArgumentParser()
    parser.add_argument(--runs-root, required=True, type=Path)
    parser.add_argument(--output-root, required=True, type=Path)
    parser.add_argument(--run-dir, action=append, default=[], type=Path)
    args = parser.parse_args(argv)
    from confidence_head.plot_analysis import discover_completed_runs, plot_completed_runs
    runs = discover_completed_runs(args.runs_root.resolve(), explicit_run_dirs=tuple(path.resolve() for path in args.run_dir))
    artifacts = plot_completed_runs(runs, comparisons_dir=args.output_root.resolve() / comparisons)
    for artifact in artifacts:
        print(artifact)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

The single-run command accepts `--run-dir`. The correlation command accepts exactly eight repeated `--run-dir` values plus `--output-dir`. The completed command accepts `--runs-root`, `--output-root`, and optional repeated `--run-dir`; its comparison directory is `<output-root>/comparisons`.

- [ ] **Step 4: Document reproducible local and remote commands**

Add a README section showing:

```bash
python Uncertainty_Quantification/ConfidenceHead/scripts/plot_completed_runs.py \
  --runs-root Uncertainty_Quantification/ConfidenceHead/outputs/runs \
  --output-root Uncertainty_Quantification/ConfidenceHead/outputs
```

Document the single-run command, explicit `--run-dir` selection for duplicate histories, all output directories, per-atom energy semantics, force `atom_mean` semantics, 50-bin display rule, and the fact that plotting never uploads to WandB or changes run artifacts.

- [ ] **Step 5: Run command and documentation checks**

Run:

```bash
python -m pytest Uncertainty_Quantification/ConfidenceHead/tests/test_plot_scripts.py -q
python Uncertainty_Quantification/ConfidenceHead/scripts/plot_completed_runs.py --help
python Uncertainty_Quantification/ConfidenceHead/scripts/plot_argmax_bin_boxplots.py --help
python Uncertainty_Quantification/ConfidenceHead/scripts/plot_energy_order_correlations.py --help
```

Expected: all tests pass and each help command exits 0 with its required arguments shown.

- [ ] **Step 6: Commit entrypoints and documentation**

```bash
git add Uncertainty_Quantification/ConfidenceHead/scripts Uncertainty_Quantification/ConfidenceHead/tests/test_plot_scripts.py Uncertainty_Quantification/ConfidenceHead/README.md
git commit -m "docs: add confidence head plotting commands"
```

### Task 5: Full local verification, remote deployment, and real-result generation

**Files:**
- Verify all files from Tasks 1–4.
- Generate remotely under `/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/outputs`; do not commit generated outputs.

**Interfaces:**
- Consumes: completed plotting implementation and the nine existing remote run directories.
- Produces: verified remote PNG, PDF, and CSV artifacts plus a representative local preview copy.

- [ ] **Step 1: Run the complete local ConfidenceHead test suite**

```bash
tox -e confidence-head-tests
```

Expected: every ConfidenceHead test passes with warnings treated according to repository policy.

- [ ] **Step 2: Run formatting, lint, typing, and diff checks**

```bash
tox -e lint
git diff --check
git status --short
```

Expected: lint and diff checks pass; only intentional implementation changes are present before their final commit.

- [ ] **Step 3: Commit any mechanical lint-only correction and verify a clean tree**

If formatting changes are required, apply `tox -e format`, rerun Task 5 Steps 1–2, then commit only those mechanical changes:

```bash
git add Uncertainty_Quantification/ConfidenceHead tox.ini
git commit -m "style: format confidence head plotting"
git status --short
```

Expected: clean working tree.

- [ ] **Step 4: Push the current branch and update the remote checkout**

```bash
git push origin ConfidenceHead
ssh -p 55801 bywang@121.48.164.204 \
  'cd /home/bywang/code/UQ/upet_new && git fetch origin ConfidenceHead && git checkout ConfidenceHead && git pull --ff-only origin ConfidenceHead'
```

Expected: local and remote `git rev-parse HEAD` match.

- [ ] **Step 5: Confirm remote plotting dependencies without changing training state**

```bash
ssh -p 55801 bywang@121.48.164.204 \
  '/home/bywang/.conda/envs/upet_new/bin/python -c "import matplotlib, scipy, torch; print(matplotlib.__version__, scipy.__version__, torch.__version__)"'
```

If Matplotlib is missing, install only the declared remote requirements into the existing `upet_new` environment, then repeat the import check.

- [ ] **Step 6: Resolve the exact nine completed runs and execute plotting on CPU**

First list eligible directories and their run/evaluation status. If automatic discovery reports duplicate completed histories, invoke the same command with exactly nine explicit `--run-dir` values chosen from the previously verified jobs. Then run:

```bash
ssh -p 55801 bywang@121.48.164.204 \
  'cd /home/bywang/code/UQ/upet_new && MPLBACKEND=Agg /home/bywang/.conda/envs/upet_new/bin/python Uncertainty_Quantification/ConfidenceHead/scripts/plot_completed_runs.py --runs-root Uncertainty_Quantification/ConfidenceHead/outputs/runs --output-root Uncertainty_Quantification/ConfidenceHead/outputs'
```

Expected: command exits 0 and prints every generated artifact path.

- [ ] **Step 7: Validate remote artifact counts and numerical contents**

Use a read-only Python check to assert:

```python
assert len(single_statistics_csvs) == 9
assert all(sum(1 for _ in csv.DictReader(path.open())) == 50 for path in single_statistics_csvs)
assert [int(row["order"]) for row in correlation_rows] == list(range(1, 9))
assert all(Path(path).stat().st_size > 0 for path in png_pdf_outputs)
```

Also load every PNG with `matplotlib.image.imread`, check every PDF begins with `%PDF`, and verify the sum of bin `sample_count` equals 19,374 for each energy run and 149,321 for the force run.

- [ ] **Step 8: Retrieve representative previews and report exact outputs**

Copy the correlation PNG, one energy order boxplot PNG, and the force boxplot PNG into a local review directory outside the repository's tracked outputs. Open all three for visual inspection and report:

- remote output root;
- exact run directories used;
- correlation table for order 1–8;
- generated file counts;
- local commit hash and matching remote commit hash;
- any visual differences from Carnet that remain intentional because UPET uses energy per atom and force `atom_mean`.
