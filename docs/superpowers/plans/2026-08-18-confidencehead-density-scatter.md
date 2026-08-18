# ConfidenceHead Density Scatter Plots Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a reproducible FGE-style expected-error versus observed-error density plot publication for `matpes_train`, `matpes_test`, and `mad_test` without changing existing boxplot outputs.

**Architecture:** Reuse the verified `PlotSeries`/`load_plot_series` contract, add an isolated `density_plotting.py` module for filtering, log-space analysis, rendering, and density manifests, and expose a separate CLI/config path that writes under `Plots/ConfidenceHead/density_scatter`. Existing `plot_analysis.py`, boxplot functions, and historical manifests remain unchanged.

**Tech Stack:** Python 3.11+, PyTorch tensors, NumPy, SciPy `gaussian_filter`, Matplotlib Agg, PyYAML/Pydantic, pytest.

---

### Task 1: Lock the density-analysis contract with unit tests

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_density_plotting.py`
- Reference: `Uncertainty_Quantification/ConfidenceHead/confidence_head/plot_analysis.py:PlotSeries`

- [ ] **Step 1: Add synthetic `PlotSeries` fixtures and failing tests.**

Create a fixture with positive values plus zero, negative, NaN, and Inf pairs. Add tests that require:

```python
def test_analyze_density_filters_pairs_and_reports_counts() -> None:
    panel = analyze_density_panel(_series_with_invalid_pairs(), DensitySettings())
    assert panel.filtered.original_count == 8
    assert panel.filtered.valid_count == 4
    assert panel.filtered.excluded == {
        "nan": 1,
        "inf": 1,
        "negative": 1,
        "zero": 1,
    }
    assert panel.density.grid.shape == (160, 160)
```

Add tests for deterministic sampling and all-sample density independence:

```python
def test_sampling_is_deterministic_and_density_uses_all_valid_pairs() -> None:
    first = analyze_density_panel(
        _large_series(500), DensitySettings(scatter_max_points=20, scatter_seed=17)
    )
    second = analyze_density_panel(
        _large_series(500), DensitySettings(scatter_max_points=20, scatter_seed=17)
    )
    assert torch.equal(first.scatter_indices, second.scatter_indices)
    assert first.scatter_indices.numel() == 20
    assert first.density.histogram_count == first.filtered.valid_count
```

Add tests requiring shared energy limits to contain all eight orders and rejecting fewer than two valid positive pairs.

- [ ] **Step 2: Run the focused tests and verify they fail for the missing module.**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_plotting.py -q
```

Expected: collection fails with `ModuleNotFoundError` for `confidence_head.density_plotting`.

- [ ] **Step 3: Commit the test-only contract.**

```bash
git add Uncertainty_Quantification/ConfidenceHead/tests/test_density_plotting.py
git commit -m "test: define confidencehead density analysis contract"
```

### Task 2: Implement log-space filtering and density analysis

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plotting.py`
- Test: `Uncertainty_Quantification/ConfidenceHead/tests/test_density_plotting.py`

- [ ] **Step 1: Add immutable settings and analysis dataclasses.**

Define `DensitySettings`, `FilteredDensityPairs`, `DensityContours`, and `DensityPanel`. Defaults must be:

```python
scatter_max_points=20_000
scatter_seed=20260714
grid_size=160
gaussian_sigma=1.2
contour_masses=(0.50, 0.70, 0.85, 0.95, 0.99)
log_margin=0.05
dpi=300
```

Validate positive settings, strictly increasing masses inside `(0, 1)`, and `grid_size >= 8`.

- [ ] **Step 2: Implement paired filtering and deterministic sampling.**

Implement `filter_density_pairs(expected, observed)` to flatten matching tensors, classify `nan`, `inf`, `zero`, and `negative`, retain only finite positive pairs, and expose both raw and `log10` tensors. Implement `_sample_indices(count, maximum, seed)` using `np.random.default_rng(seed)` and sorted indices.

- [ ] **Step 3: Implement shared log limits and Gaussian contours.**

Implement `shared_log_limits(series, margin)` over all expected/observed values supplied for one dataset target. Implement `_density_contours` with `np.histogram2d` over the shared log range, `scipy.ndimage.gaussian_filter(..., mode="nearest")`, normalization to total mass one, and thresholds selected by cumulative descending cell mass for the configured contour masses.

- [ ] **Step 4: Implement `analyze_density_panel`.**

The function must compute log-space Spearman and Pearson correlations, sample indices, density contours, exclusion counts, and limits. It must reject fewer than two valid pairs and never use the sampled subset for density or statistics.

- [ ] **Step 5: Run the focused tests and commit.**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_plotting.py -q
```

Expected: all analysis tests pass. Commit:

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plotting.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_plotting.py
git commit -m "feat: add confidencehead log density analysis"
```

### Task 3: Implement FGE-style rendering and artifact statistics

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plotting.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_density_rendering.py`

- [ ] **Step 1: Add rendering tests before implementation.**

Test `render_density_panel` with a synthetic energy and force `PlotSeries`; require non-empty PNG/PDF output, a square log-scaled axis, and a statistics JSON/CSV containing `valid_count`, `original_count`, `spearman_log10`, `pearson_log10`, `excluded`, and `actual_contour_masses`.

Test `render_energy_comparison` with orders 1 through 8 and require exactly eight axes sharing identical x/y limits.

- [ ] **Step 2: Implement the single-panel renderer.**

Use Matplotlib Agg and FGE palette conventions: low-alpha orange scatter, darker orange contours, light-gray `y <= x` fill, dark dashed `y=x`, equal aspect, and a white statistics box. Use target-specific labels with `eV/atom` for energy and `eV/A` for force. Save both PNG (configured DPI) and PDF through the existing atomic output pattern.

- [ ] **Step 3: Implement per-run CSV/JSON output.**

Write one row/JSON object containing the filtering counts, log limits, correlation values, density settings, histogram count, contour levels, and actual contour masses. Do not write or modify any `argmax_bin_*` file.

- [ ] **Step 4: Implement energy 4x2 and force comparison renderers.**

For a dataset, compute one shared energy limit from all eight orders, render the eight energy panels in a 4x2 figure, and render force separately. Keep force out of the energy grid.

- [ ] **Step 5: Run rendering tests and commit.**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_rendering.py -q
```

Expected: all PNG/PDF, axis, and statistics tests pass. Commit:

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plotting.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_rendering.py
git commit -m "feat: render confidencehead density scatter plots"
```

### Task 4: Add atomic density publication and manifest verification

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plotting.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_density_publication.py`

- [ ] **Step 1: Add publication tests.**

Use synthetic `DatasetSeries` values to require three dataset-local publications with `manifest.json`, per-run PNG/PDF/CSV/JSON, comparison PNG/PDF, and SHA256 descriptors. Require a second identical call to reuse the publication and a changed input identity to raise a conflict. Add a test that creates an old `argmax_bin_boxplots` file and asserts its bytes are unchanged after density publication.

- [ ] **Step 2: Implement `DensityPublication` and identity helpers.**

Use schema `upet_confidence_density_plots_v1`. Bind dataset name, input run manifest digests, target/order set, and normalized settings in the identity payload. Implement confined artifact descriptors and full SHA256 verification separately from the old boxplot schema.

- [ ] **Step 3: Implement `publish_density_dataset`.**

Preflight all eight energy series and one force series before creating a staging directory. Generate `runs/<run-name>/`, dataset comparisons, and manifest. On any exception remove staging and leave existing output untouched. Reuse a complete matching identity; reject a complete conflicting identity.

- [ ] **Step 4: Implement cross-dataset correlation publication.**

Reuse verified `DatasetSeries` values to write energy order and force correlation CSV/PNG/PDF under `density_scatter/comparisons`, with the density schema and input identities. Do not create cross-dataset raw scatter panels with incompatible per-dataset axes.

- [ ] **Step 5: Run publication tests and commit.**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_publication.py -q
```

Expected: publication reuse, conflict, SHA256, and legacy-preservation tests pass. Commit:

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plotting.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_publication.py
git commit -m "feat: publish audited confidencehead density suites"
```

### Task 5: Add configuration-driven command and CLI

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/density_plots.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/scripts/plot_density_scatter.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/external_commands.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_density_commands.py`

- [ ] **Step 1: Add command tests.**

Mock `discover_external_runs`, `load_dataset_series`, and `publish_density_dataset`; assert selected dataset names are validated exactly like the existing external plotting command, the output root is the configured density root, and no prediction function is called. Add a script test requiring `--config` and optional repeated `--dataset` forwarding.

- [ ] **Step 2: Add a strict density config loader.**

Load `external_config` plus `output_root` and the exact `DensitySettings` fields from YAML. Resolve relative paths against the density config location; reject duplicate keys, unknown fields, non-safe dataset names, invalid contour masses, and non-positive numeric settings.

- [ ] **Step 3: Add `plot_density_from_config`.**

Use the existing external config only to discover the nine completed run sources. Select requested datasets, call `load_dataset_series`, and call the new density publication functions under the configured density root. Never call `predict_external_datasets`.

- [ ] **Step 4: Add the CLI and formal config.**

The CLI accepts `--config` and repeatable `--dataset`, prints each manifest path, and exits nonzero on preflight or manifest failure. The formal YAML points at the existing external prediction config and writes to `../Plots/ConfidenceHead/density_scatter` relative to the repository configuration directory.

- [ ] **Step 5: Run command tests and commit.**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_commands.py -q
```

Expected: command and CLI tests pass. Commit:

```bash
git add Uncertainty_Quantification/ConfidenceHead/configs/density_plots.yaml Uncertainty_Quantification/ConfidenceHead/scripts/plot_density_scatter.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/external_commands.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_commands.py
git commit -m "feat: add density plotting command"
```

### Task 6: Run static/local verification and remote plotting

**Files:**
- No source changes expected unless a focused test exposes a defect.

- [ ] **Step 1: Run the complete ConfidenceHead plotting test subset.**

Run:

```bash
pytest Uncertainty_Quantification/ConfidenceHead/tests/test_density_plotting.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_rendering.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_publication.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_commands.py Uncertainty_Quantification/ConfidenceHead/tests/test_plot_rendering.py Uncertainty_Quantification/ConfidenceHead/tests/test_external_plotting.py -q
```

Expected: new tests pass and legacy boxplot tests remain green.

- [ ] **Step 2: Run formatting/static checks on changed source.**

Run:

```bash
ruff format --check Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plotting.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_plotting.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_rendering.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_publication.py Uncertainty_Quantification/ConfidenceHead/tests/test_density_commands.py
ruff check Uncertainty_Quantification/ConfidenceHead/confidence_head/density_plotting.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/external_commands.py Uncertainty_Quantification/ConfidenceHead/scripts/plot_density_scatter.py
```

Expected: no formatting or lint errors.

- [ ] **Step 3: Run the remote CPU/GPU-available plotting command against existing completed runs.**

On the configured remote server, activate the existing `upet_new` environment and run:

```bash
python Uncertainty_Quantification/ConfidenceHead/scripts/plot_density_scatter.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/density_plots.yaml
```

Expected: three dataset manifests plus the comparison manifest complete without inference jobs.

- [ ] **Step 4: Verify remote artifacts and pull only the new directory.**

Verify every density manifest with the new verifier, count 3 x (8 energy + 1 force) single plots plus 3 energy comparison plots, and compare PNG/PDF/CSV/JSON counts. Pull only `Plots/ConfidenceHead/density_scatter/` to the local matching directory; leave existing boxplot results untouched.

- [ ] **Step 5: Commit any final test-only fix and report evidence.**

Run `git status --short` and ensure unrelated existing modifications remain unstaged. If a focused fix was needed, commit only its files with a targeted message. Report commit IDs, test commands, manifest counts, and the local pull path.
