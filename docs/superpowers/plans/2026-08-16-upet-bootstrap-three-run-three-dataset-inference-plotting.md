# UPET BootStrapping Three-Run Three-Dataset Inference and Plotting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reuse three completed 8-member BootStrapping runs, predict only MAD-test and MATPES-train, compute canonical raw UQ, and publish Carnet-style shared-scale plots for all three runs and three datasets.

**Architecture:** Add a strict campaign layer above the existing run configuration, generalize prediction/UQ storage from fixed `val/test` names to safe dataset keys, and represent missing MAD reference stress explicitly. Keep numerical prediction, UQ, plot-source auditing, analysis, rendering, and transactional publication as separate units; execute prediction on the remote GPU and UQ/plotting on the remote CPU.

**Tech Stack:** Python 3.11, PyTorch, NumPy, ASE, metatrain/metatomic, SciPy, Matplotlib, PyYAML, pytest, Ruff, mypy, Conda `upet_new`, SSH, optional Slurm.

## Global Constraints

- Work directly on the current `Plots` branch; do not create a worktree.
- Preserve unrelated `.idea` and ConfidenceHead working-tree files and never include them in a BootStrapping commit.
- Local work is limited to editing and committing code, configuration, tests, and documentation. Run every test, smoke workflow, prediction, UQ computation, and plot render on the remote server.
- Use strict SSH host-key checking. Port `6688` is load-balanced; retry exit 255 without changing `known_hosts`, accepting a new key, or disabling verification.
- Remote login is `yt_hku_psmanyam_3@121.46.19.6` with port `6688` and key `C:\Users\52657\.ssh\yt_hku_psmanyam_3.id`.
- Remote production repository is `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new`.
- Remote isolated unit-test root is `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_bootstrap_plots_codex`.
- Remote Python is `/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python`.
- Remote base checkpoint is `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/pet-omatpes-l-v0.1.0.ckpt`.
- Reuse remote MATPES files under `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_test.extxyz` and `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_train.extxyz`; only upload local `data/dataset/mad-test.xyz` if no remote equivalent exists.
- Formal runs are `full_remote_b8_e8`, `lr_1e-4`, and `lr_1e-6`; each uses all eight ordered members and only the raw checkpoint branch.
- Reuse existing `predictions/test` and `uncertainty/test/raw` byte-for-byte. Do not rerun MATPES-test inference.
- New prediction/UQ keys are `mad_test` and `matpes_train`.
- MAD reference targets are energy and forces only. Never synthesize reference stress.
- UQ uses float64 Welford sample STD with `ddof=1` and distinct unordered-pair GMD.
- Do not implement or publish a member-count sweep.
- Shared plot limits are global per physical target: nine energy panels, nine force panels, and six stress panels.
- Plot stress in the units declared by the prediction artifact (`eV/Angstrom^3`); do not label unconverted values as GPa.
- Generated checkpoint, prediction, UQ, plot, staging, scheduler, and audit artifacts remain ignored by Git.
- Each implementation task follows TDD: add a failing test locally, sync the BootStrapping tree to the isolated remote root, confirm the expected failure remotely, implement locally, resync, confirm passing remotely, then commit only that task's files.

## Remote Test Protocol

For each test step, create an archive locally without outputs:

```powershell
wsl -d Ubuntu-22.04 -- tar -C /home/lilong/code/UQ/upet_new -czf /tmp/upet-bootstrap-plots-code.tgz --exclude='Uncertainty_Quantification/BootStrapping/outputs' --exclude='__pycache__' Uncertainty_Quantification/BootStrapping
scp.exe -P 6688 -i "C:\Users\52657\.ssh\yt_hku_psmanyam_3.id" "\\wsl$\Ubuntu-22.04\tmp\upet-bootstrap-plots-code.tgz" yt_hku_psmanyam_3@121.46.19.6:/tmp/upet-bootstrap-plots-code.tgz
ssh.exe -p 6688 -i "C:\Users\52657\.ssh\yt_hku_psmanyam_3.id" -o StrictHostKeyChecking=yes yt_hku_psmanyam_3@121.46.19.6 "mkdir -p /XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_bootstrap_plots_codex && tar -xzf /tmp/upet-bootstrap-plots-code.tgz -C /XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_bootstrap_plots_codex"
```

Run tests with:

```bash
cd /XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_bootstrap_plots_codex
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests -q
```

If strict SSH verification reaches the other load-balanced host, repeat the same command; do not alter security options.

---

## File Map

- Create `bootstrap/identifiers.py`: validate safe dataset/run/storage keys.
- Create `bootstrap/campaign.py`: strict campaign dataclasses and YAML loader.
- Modify `bootstrap/prediction.py`: optional reference stress, safe dataset keys, staging-root store support.
- Modify `bootstrap/schema.py`: v1/v2 prediction schema compatibility.
- Modify `bootstrap/native_prediction.py`: generic one-dataset inference while preserving `predict_run`.
- Create `bootstrap/prediction_publication.py`: complete prediction publication audit and campaign orchestration.
- Refactor `bootstrap/uq_publication.py`: separate UQ reduction from publication and accept dataset keys.
- Modify `bootstrap/validation.py`: generic dataset-key UQ validation.
- Create `bootstrap/uq_campaign.py`: reuse-or-compute campaign UQ orchestration.
- Create `bootstrap/plot_source.py`: audit campaign inputs and expose bounded-memory domain arrays.
- Create `bootstrap/plot_analysis.py`: Carnet-style filtering, density, correlation, and shared limits.
- Create `bootstrap/plot_rendering.py`: deterministic PNG/PDF and statistics payload rendering.
- Create `bootstrap/plot_store.py`: content identity, exact 51-file formal publication, and validation.
- Create `scripts/predict_campaign.py`, `scripts/compute_campaign_uq.py`, and `scripts/plot_campaign.py`: thin CLIs.
- Create `configs/three_run_three_dataset_raw.yaml`: formal remote campaign.
- Modify `README.md`, `.gitignore`, and release tests: public workflow and generated-result boundary.
- Add focused tests under `Uncertainty_Quantification/BootStrapping/tests/`.

---

### Task 1: Strict campaign configuration and safe identifiers

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/identifiers.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/campaign.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_campaign.py`

**Interfaces:**
- Consumes: `BootstrapConfig`, `load_config`, `HardFailure`.
- Produces: `validate_artifact_key(value, location) -> str`, `CampaignRun`, `CampaignDataset`, `CampaignPrediction`, `CampaignPlotStyle`, `CampaignConfig`, `load_campaign(path) -> CampaignConfig`, `select_campaign_items(campaign, run_labels, dataset_labels)`.

- [ ] **Step 1: Write failing campaign tests**

```python
def test_campaign_resolves_runs_datasets_and_shared_output(tmp_path: Path) -> None:
    campaign = load_campaign(_write_campaign(tmp_path))
    assert [run.label for run in campaign.runs] == [
        "full_remote_b8_e8", "lr_1e-4", "lr_1e-6"
    ]
    assert [(item.label, item.storage_key, item.reference_targets) for item in campaign.datasets] == [
        ("matpes_test", "test", ("energy", "forces", "stress")),
        ("mad_test", "mad_test", ("energy", "forces")),
        ("matpes_train", "matpes_train", ("energy", "forces", "stress")),
    ]
    assert campaign.prediction.mode == "raw"
    assert campaign.prediction.member_count == 8
    assert campaign.output_root.is_absolute()

@pytest.mark.parametrize("value", ["../escape", "Mad-Test", "a/b", ".", ""])
def test_campaign_rejects_unsafe_artifact_keys(tmp_path: Path, value: str) -> None:
    source = _write_campaign(tmp_path, dataset_storage_key=value)
    with pytest.raises(HardFailure, match="storage_key"):
        load_campaign(source)

def test_campaign_rejects_duplicate_labels_and_ema(tmp_path: Path) -> None:
    with pytest.raises(HardFailure, match="duplicate run label"):
        load_campaign(_write_campaign(tmp_path, duplicate_run=True))
    with pytest.raises(HardFailure, match="mode must be raw"):
        load_campaign(_write_campaign(tmp_path, mode="ema"))
```

The helper writes three minimal run YAML files with matching run IDs and one campaign YAML. The campaign loader resolves paths relative to the campaign file but does not require remote-only checkpoint/data files to exist during local parsing.

- [ ] **Step 2: Sync and confirm the tests fail remotely**

Run:

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests/test_campaign.py -q
```

Expected: import failure for `bootstrap.campaign`.

- [ ] **Step 3: Implement identifiers and strict dataclasses**

```python
_ARTIFACT_KEY = re.compile(r"[a-z0-9][a-z0-9_]*\Z")

def validate_artifact_key(value: object, location: str) -> str:
    if not isinstance(value, str) or _ARTIFACT_KEY.fullmatch(value) is None:
        raise HardFailure(f"{location} must use lowercase letters, digits, or underscores")
    return value

@dataclass(frozen=True)
class CampaignRun:
    label: str
    config_path: Path
    run_root: Path
    config: BootstrapConfig

@dataclass(frozen=True)
class CampaignDataset:
    label: str
    storage_key: str
    path: Path
    reference_targets: tuple[str, ...]

@dataclass(frozen=True)
class CampaignPrediction:
    mode: str
    member_count: int
    device: str
    batch_size: int

@dataclass(frozen=True)
class CampaignPlotStyle:
    grid_size: int
    gaussian_sigma: float
    contour_masses: tuple[float, ...]
    scatter_max_points: int
    scatter_seed: int
    scatter_size: float
    scatter_alpha: float
    log_margin: float
    figure_size: tuple[float, float]
    dpi: int
    formats: tuple[str, ...]

@dataclass(frozen=True)
class CampaignConfig:
    schema_version: int
    runs: tuple[CampaignRun, ...]
    datasets: tuple[CampaignDataset, ...]
    prediction: CampaignPrediction
    plot: CampaignPlotStyle
    output_root: Path
    source_path: Path
```

Require `schema_version=1`, nonempty unique run/dataset labels, raw mode, `member_count >= 2`, positive batch/render values, `formats == ("png", "pdf")`, and strictly increasing contour masses inside `(0, 1)`. Load each run config and require its `run_id` and ensemble size to match the campaign record.

- [ ] **Step 4: Resync and confirm campaign tests pass remotely**

Run the Task 1 pytest command. Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/identifiers.py Uncertainty_Quantification/BootStrapping/bootstrap/campaign.py Uncertainty_Quantification/BootStrapping/tests/test_campaign.py
git commit -m "feat(bootstrap): define multi-dataset campaigns"
```

---

### Task 2: Prediction targets v2 with optional reference stress

**Files:**
- Modify: `Uncertainty_Quantification/BootStrapping/bootstrap/prediction.py:16-190`
- Modify: `Uncertainty_Quantification/BootStrapping/bootstrap/schema.py:1-85`
- Modify: `Uncertainty_Quantification/BootStrapping/tests/test_prediction.py`
- Modify: `Uncertainty_Quantification/BootStrapping/tests/test_native_prediction.py`

**Interfaces:**
- Consumes: `validate_artifact_key`.
- Produces: `TargetArrays(stress: NDArray | None)`, `reference_targets(targets)`, v1/v2 `load_target_arrays`, generic `PredictionStore`, and `PredictionStore.at_split_root(split_root, split, units)`.

- [ ] **Step 1: Add failing optional-stress and key-safety tests**

```python
def test_target_store_round_trips_mad_without_stress(tmp_path: Path) -> None:
    targets = replace(_targets(), stress=None)
    store = PredictionStore(tmp_path, split="mad_test", units=UNITS)
    path = store.write_targets(targets)
    loaded = load_target_arrays(path)
    assert loaded.stress is None
    assert reference_targets(loaded) == ("energy", "forces")
    with np.load(path, allow_pickle=False) as archive:
        assert "stress" not in archive.files
    store.write_member(0, "raw", _predictions())

def test_prediction_store_rejects_escaping_dataset_key(tmp_path: Path) -> None:
    with pytest.raises(HardFailure, match="split"):
        PredictionStore(tmp_path, split="../outside", units=UNITS)

def test_existing_v1_targets_remain_readable(tmp_path: Path) -> None:
    path = PredictionStore(tmp_path, split="test", units=UNITS).write_targets(_targets())
    assert load_target_arrays(path).stress is not None
```

Also extend the schema-signature test so a stress-free dataset has a stable v2 signature and existing full targets retain their current v1 signature.

- [ ] **Step 2: Sync and confirm focused tests fail remotely**

Run:

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests/test_prediction.py Uncertainty_Quantification/BootStrapping/tests/test_native_prediction.py -q
```

Expected: `TargetArrays` rejects `None` stress or validation calls `_array(None)`.

- [ ] **Step 3: Implement the compatible target contract**

```python
@dataclass(frozen=True)
class TargetArrays:
    structure_ids: NDArray
    num_atoms: NDArray
    atom_offsets: NDArray
    energy: NDArray
    forces: NDArray
    stress: NDArray | None

def reference_targets(targets: TargetArrays) -> tuple[str, ...]:
    fields = ["energy", "forces"]
    if targets.stress is not None:
        fields.append("stress")
    return tuple(fields)
```

Validate energy/forces exactly as before. Validate stress only when present. `write_targets` omits the `stress` NPZ key when it is `None`; `load_target_arrays` accepts exactly the six-key MAD set or the seven-key existing set. Predictions always retain energy, forces, and stress and are validated against structure/atom counts rather than reference-stress presence.

Add an optional `split_root` override through:

```python
@classmethod
def at_split_root(cls, split_root: str | Path, *, split: str, units: Mapping[str, str]) -> "PredictionStore":
    return cls(split_root, split=split, units=units, direct_split_root=True)
```

The ordinary constructor still resolves `root / split`; the classmethod treats its path as the final dataset directory for sibling staging.

- [ ] **Step 4: Resync and run prediction/schema tests remotely**

Run the Task 2 pytest command. Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/prediction.py Uncertainty_Quantification/BootStrapping/bootstrap/schema.py Uncertainty_Quantification/BootStrapping/tests/test_prediction.py Uncertainty_Quantification/BootStrapping/tests/test_native_prediction.py
git commit -m "feat(bootstrap): support datasets without reference stress"
```

---

### Task 3: Transactional generic dataset prediction and reuse audit

**Files:**
- Modify: `Uncertainty_Quantification/BootStrapping/bootstrap/native_prediction.py:75-270`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/prediction_publication.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_prediction_publication.py`
- Modify: `Uncertainty_Quantification/BootStrapping/tests/test_native_prediction.py`

**Interfaces:**
- Consumes: `CampaignConfig`, `CampaignRun`, `CampaignDataset`, `PredictionStore.at_split_root`, `sibling_staging`, `load_pet_member_model`.
- Produces: `DatasetPredictionRequest`, `extract_targets`, `predict_dataset`, `validate_prediction_publication`, `predict_campaign`.

- [ ] **Step 1: Write failing transaction, MAD, and reuse tests**

```python
def test_extract_targets_does_not_request_missing_mad_stress(mad_atoms) -> None:
    targets = extract_targets(mad_atoms, "mad_test", ("energy", "forces"))
    assert targets.stress is None

def test_predict_dataset_publishes_manifest_last(tmp_path: Path, fake_runtime) -> None:
    manifest = predict_dataset(fake_runtime.request(tmp_path))
    document = json.loads(manifest.read_text())
    assert document["schema"] == "upet.bootstrap.predictions/v2"
    assert document["dataset_label"] == "mad_test"
    assert document["reference_targets"] == ["energy", "forces"]
    assert document["member_count"] == 2
    assert len(document["members"]) == 2

def test_predict_campaign_reuses_complete_test_without_loader_calls(campaign, monkeypatch) -> None:
    _write_existing_v1_test_publication(campaign.runs[0].run_root)
    monkeypatch.setattr(native_prediction, "load_pet_member_model", fail_if_called)
    publications = predict_campaign(campaign, ("full_remote_b8_e8",), ("matpes_test",))
    assert publications[0].skipped is True

def test_member_failure_does_not_publish_destination(tmp_path: Path, fake_runtime) -> None:
    fake_runtime.fail_member = 1
    with pytest.raises(HardFailure, match="member 1"):
        predict_dataset(fake_runtime.request(tmp_path))
    assert not (tmp_path / "predictions/mad_test").exists()
```

- [ ] **Step 2: Sync and confirm the new tests fail remotely**

Run:

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests/test_native_prediction.py Uncertainty_Quantification/BootStrapping/tests/test_prediction_publication.py -q
```

Expected: missing `prediction_publication` and `predict_dataset` APIs.

- [ ] **Step 3: Refactor target extraction and generic prediction**

```python
@dataclass(frozen=True)
class DatasetPredictionRequest:
    run: CampaignRun
    dataset: CampaignDataset
    mode: str
    member_count: int
    device: torch.device
    dtype: torch.dtype
    batch_size: int
    structure_limit: int | None = None

def extract_targets(atoms: list[Any], label: str, targets: tuple[str, ...]) -> TargetArrays:
    stress = None
    if "stress" in targets:
        stress = np.stack([item.get_stress(voigt=False) for item in atoms]).astype(np.float64)
    return TargetArrays(
        structure_ids=_structure_ids(atoms, label),
        num_atoms=_num_atoms(atoms),
        atom_offsets=_atom_offsets(atoms),
        energy=np.asarray([item.get_potential_energy() for item in atoms], dtype=np.float64),
        forces=np.concatenate([item.get_forces() for item in atoms]).astype(np.float64),
        stress=stress,
    )
```

Build all files under `sibling_staging(run_root / "predictions" / storage_key)`, use `PredictionStore.at_split_root`, and write `manifest.json` only after all eight member files are durable. Preserve `predict_run` by translating configured `val/test` into `DatasetPredictionRequest` objects.

- [ ] **Step 4: Implement strict publication validation and campaign selection**

`validate_prediction_publication` must support the existing v1 test manifest and new v2 manifests. It verifies exact dataset key, expected mode/member count, target keys/layout, each declared member path, SHA256, shape and dtype, and rejects extra/missing member records. `predict_campaign` first validates an existing destination and returns `skipped=True`; it never calls model loading for a valid publication.

- [ ] **Step 5: Resync and run focused tests remotely**

Run the Task 3 pytest command. Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/native_prediction.py Uncertainty_Quantification/BootStrapping/bootstrap/prediction_publication.py Uncertainty_Quantification/BootStrapping/tests/test_native_prediction.py Uncertainty_Quantification/BootStrapping/tests/test_prediction_publication.py
git commit -m "feat(bootstrap): publish campaign predictions transactionally"
```

---

### Task 4: Generic transactional UQ and campaign reuse

**Files:**
- Modify: `Uncertainty_Quantification/BootStrapping/bootstrap/uq_publication.py:1-130`
- Modify: `Uncertainty_Quantification/BootStrapping/bootstrap/validation.py:1-160`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/uq_campaign.py`
- Modify: `Uncertainty_Quantification/BootStrapping/tests/test_uncertainty.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_uq_campaign.py`

**Interfaces:**
- Consumes: generic prediction publications and campaign selectors.
- Produces: `compute_uncertainty_results`, `publish_uncertainty_results`, backward-compatible `compute_store_uncertainty`, generic `validate_uq_publication`, and `compute_campaign_uq`.

- [ ] **Step 1: Write failing generic-key, rollback, and reuse tests**

```python
def test_uq_accepts_mad_dataset_without_reference_stress(tmp_path: Path) -> None:
    prediction_root = _write_predictions(tmp_path, "mad_test", stress_target=False)
    result = compute_store_uncertainty(
        prediction_root, tmp_path / "uncertainty", split="mad_test",
        mode="raw", member_count=2, units=UNITS,
    )
    with np.load(result.results_path, allow_pickle=False) as archive:
        assert archive["stress_std"].shape == (2, 3, 3)

def test_campaign_uq_reuses_existing_verified_test(campaign, monkeypatch) -> None:
    _write_valid_uq(campaign.runs[0].run_root, "test")
    monkeypatch.setattr(uq_campaign, "compute_uncertainty_results", fail_if_called)
    result = compute_campaign_uq(campaign, ("full_remote_b8_e8",), ("matpes_test",))
    assert result[0].skipped is True

def test_uq_publication_failure_leaves_no_destination(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(uq_publication, "atomic_write_json", raise_disk_error)
    with pytest.raises(HardFailure, match="disk"):
        compute_dataset_uncertainty(_request(tmp_path))
    assert not (tmp_path / "uncertainty/mad_test/raw").exists()
```

- [ ] **Step 2: Sync and confirm UQ campaign tests fail remotely**

Run:

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests/test_uncertainty.py Uncertainty_Quantification/BootStrapping/tests/test_uq_campaign.py -q
```

Expected: the fixed `val/test` guard rejects `mad_test` and `uq_campaign` is missing.

- [ ] **Step 3: Separate numerical reduction from publication**

```python
def compute_uncertainty_results(
    prediction_split_root: str | Path, *, mode: str, member_count: int
) -> dict[str, NDArray[np.float64]]:
    targets = load_target_arrays(Path(prediction_split_root) / "targets.npz")
    members = _ordered_member_paths(prediction_split_root, mode, member_count)
    results = _component_statistics(members)
    results.update(_scalar_reductions(results, targets.num_atoms, targets.atom_offsets))
    return results

def publish_uncertainty_results(
    publication_root: str | Path, results: Mapping[str, NDArray[np.float64]],
    *, dataset_key: str, mode: str, member_count: int, units: Mapping[str, str]
) -> UncertaintyPublication:
    root = Path(publication_root).expanduser().resolve()
    results_path = atomic_write_npz(root / "results.npz", **results)
    manifest_path = atomic_write_json(
        root / "manifest.json",
        _manifest_document(results_path, results, dataset_key, mode, member_count, units),
    )
    return UncertaintyPublication(
        results_path, manifest_path, dataset_key, mode, member_count
    )
```

Define `_manifest_document(results_path, results, dataset_key, mode, member_count, units) -> dict[str, object]` in the same module. It returns the current `upet.bootstrap.uncertainty/v1` document with `sample_ddof_1`, `distinct_unordered_pairs`, `float64`, dataset key in `split`, raw mode, member count, units, `results.npz` SHA256, and sorted array shape/dtype declarations.

- [ ] **Step 4: Implement transactional campaign orchestration**

Use `sibling_staging(run_root / "uncertainty" / key / "raw")`. If the destination exists, run the generic `validate_uq_publication`; return `skipped=True` only on complete equality. Generalize validator guards with `validate_artifact_key` while retaining existing v1 manifest acceptance.

- [ ] **Step 5: Resync and run UQ tests remotely**

Run the Task 4 pytest command. Expected: PASS.

- [ ] **Step 6: Commit Task 4**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/uq_publication.py Uncertainty_Quantification/BootStrapping/bootstrap/validation.py Uncertainty_Quantification/BootStrapping/bootstrap/uq_campaign.py Uncertainty_Quantification/BootStrapping/tests/test_uncertainty.py Uncertainty_Quantification/BootStrapping/tests/test_uq_campaign.py
git commit -m "feat(bootstrap): compute campaign uncertainty transactionally"
```

---

### Task 5: Audited bounded-memory plot sources

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/plot_source.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_plot_source.py`

**Interfaces:**
- Consumes: campaign config, prediction validator, UQ validator, `load_target_arrays`, `load_prediction_arrays`.
- Produces: `PanelKey`, `PlotSource`, `discover_plot_sources`, `load_panel_arrays`, `symmetric_voigt`.

- [ ] **Step 1: Write failing source-count and scientific-definition tests**

```python
def test_discover_sources_builds_24_panels_and_omits_mad_stress(campaign_artifacts) -> None:
    sources = discover_plot_sources(campaign_artifacts.campaign)
    assert len(sources) == 24
    assert sum(source.key.target == "energy" for source in sources) == 9
    assert sum(source.key.target == "force" for source in sources) == 9
    assert sum(source.key.target == "stress" for source in sources) == 6
    assert not any(
        source.key.dataset_label == "mad_test" and source.key.target == "stress"
        for source in sources
    )

def test_panel_arrays_match_carnet_energy_force_and_symmetric_stress(source_fixture) -> None:
    energy_u, energy_r = load_panel_arrays(source_fixture.energy)
    assert energy_u == pytest.approx(np.array([1.0, 0.5]))
    assert energy_r == pytest.approx(np.array([0.25, 0.75]))
    force_u, force_r = load_panel_arrays(source_fixture.force)
    assert force_u.shape == force_r.shape == (3, 3)
    stress_u, stress_r = load_panel_arrays(source_fixture.nonsymmetric_stress)
    assert stress_u.shape == stress_r.shape == (2, 6)
    assert not np.array_equal(stress_u, symmetric_voigt(source_fixture.matrix_std))
```

- [ ] **Step 2: Sync and confirm plot-source tests fail remotely**

Run:

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests/test_plot_source.py -q
```

Expected: import failure for `bootstrap.plot_source`.

- [ ] **Step 3: Implement source discovery and audit**

```python
@dataclass(frozen=True, order=True)
class PanelKey:
    run_label: str
    dataset_label: str
    storage_key: str
    target: Literal["energy", "force", "stress"]

@dataclass(frozen=True)
class PlotSource:
    key: PanelKey
    targets_path: Path
    uq_results_path: Path
    member_paths: tuple[Path, ...]
    prediction_manifest_sha256: str
    uq_manifest_sha256: str
    targets_sha256: str
```

Discovery follows campaign order, validates each prediction and UQ publication exactly once per run/dataset, and emits domains in `energy, force, stress` order, omitting stress when not declared by `reference_targets`.

- [ ] **Step 4: Implement bounded-memory domain loading**

Energy and force use the already validated UQ `energy_mean/energy_std` and `forces_mean/forces_std` arrays plus references. Stress streams the eight ordered member stress arrays through Welford after `0.5 * (S + S.T)` and Voigt conversion, preserving the Carnet definition. At most one panel's mean, M2, current member, reference, uncertainty, and residual arrays are resident.

- [ ] **Step 5: Resync and run plot-source tests remotely**

Run the Task 5 pytest command. Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/plot_source.py Uncertainty_Quantification/BootStrapping/tests/test_plot_source.py
git commit -m "feat(bootstrap): audit plotting sources"
```

---

### Task 6: Carnet-style analysis and global target scales

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/plot_analysis.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_plot_analysis.py`

**Interfaces:**
- Consumes: ordered `PlotSource` instances and campaign plot settings.
- Produces: `FilteredPairs`, `DensityContours`, `PanelAnalysis`, `filter_log_pairs`, `scan_shared_log_limits`, `analyze_panel_source`.

- [ ] **Step 1: Write failing filter, scale, and deterministic-density tests**

```python
def test_filter_log_pairs_classifies_every_input_once() -> None:
    result = filter_log_pairs(
        np.array([1.0, 0.0, -1.0, np.nan, np.inf, 2.0]),
        np.array([2.0, 1.0, 1.0, 1.0, 1.0, 0.0]),
    )
    assert result.original_count == 6
    assert result.valid_count == 1
    assert result.excluded == {"nan": 1, "inf": 1, "negative": 1, "zero": 2}
    assert result.valid_count + sum(result.excluded.values()) == result.original_count

def test_shared_limits_are_identical_for_every_panel_of_target(sources) -> None:
    limits = scan_shared_log_limits(sources, margin=0.05)
    analyses = [analyze_panel_source(source, limits[source.key.target], SETTINGS) for source in sources]
    for target in ("energy", "force", "stress"):
        assert {item.log_limits for item in analyses if item.key.target == target} == {limits[target]}

def test_analysis_is_deterministic(sources) -> None:
    limits = scan_shared_log_limits(sources, margin=0.05)
    left = analyze_panel_source(sources[0], limits["energy"], SETTINGS)
    right = analyze_panel_source(sources[0], limits["energy"], SETTINGS)
    assert np.array_equal(left.scatter_indices, right.scatter_indices)
    assert np.array_equal(left.density.grid, right.density.grid)
    assert left.spearman_log == right.spearman_log
```

- [ ] **Step 2: Sync and confirm analysis tests fail remotely**

Run:

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests/test_plot_analysis.py -q
```

Expected: import failure for `bootstrap.plot_analysis`.

- [ ] **Step 3: Implement filtering, two-pass limits, statistics, and density**

Use exclusive masks in order `nan`, `inf`, `zero`, `negative`, valid. `scan_shared_log_limits` performs a first bounded-memory pass, collecting only each panel's finite positive log minima/maxima, and applies `max((high-low)*margin, margin)` padding. `analyze_panel_source` performs the second pass, computes SciPy Spearman/Pearson on log values, deterministic scatter indices, a `grid_size x grid_size` histogram, Gaussian smoothing, and positive contour levels for configured cumulative masses.

```python
@dataclass(frozen=True)
class PanelAnalysis:
    key: PanelKey
    filtered: FilteredPairs
    density: DensityContours
    scatter_indices: NDArray[np.int64]
    log_limits: tuple[float, float]
    spearman_log: float
    pearson_log10: float
```

Reject panels with fewer than two valid pairs and reject a target with no analyzable panel.

- [ ] **Step 4: Resync and run analysis tests remotely**

Run the Task 6 pytest command. Expected: PASS.

- [ ] **Step 5: Commit Task 6**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/plot_analysis.py Uncertainty_Quantification/BootStrapping/tests/test_plot_analysis.py
git commit -m "feat(bootstrap): analyze shared-scale uncertainty plots"
```

---

### Task 7: Deterministic rendering and transactional plot publication

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/plot_rendering.py`
- Create: `Uncertainty_Quantification/BootStrapping/bootstrap/plot_store.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_plot_rendering.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_plot_store.py`

**Interfaces:**
- Consumes: campaign, `PlotSource`, shared limits, `PanelAnalysis`, atomic artifact helpers.
- Produces: `render_panel`, `render_statistics`, `compute_plot_identity`, `publish_campaign_plots`, `validate_plot_publication`.

- [ ] **Step 1: Write failing visual-contract and 51-file tests**

```python
def test_render_panel_has_log_square_axes_and_declared_units(tmp_path: Path, analysis) -> None:
    figure = build_panel_figure(analysis, STYLE)
    try:
        axis = figure.axes[0]
        assert axis.get_xscale() == axis.get_yscale() == "log"
        assert axis.get_aspect() == pytest.approx(1.0)
        assert "eV/atom" in axis.get_xlabel()
        assert "Absolute residual" in axis.get_ylabel()
        assert {line.get_label() for line in axis.lines} >= {"Ideal calibration"}
    finally:
        plt.close(figure)

def test_formal_publication_has_exactly_51_files(campaign_artifacts) -> None:
    publication = publish_campaign_plots(campaign_artifacts.campaign)
    names = {path.name for path in publication.plot_dir.iterdir()}
    assert len(names) == 51
    assert sum(name.endswith(".png") for name in names) == 24
    assert sum(name.endswith(".pdf") for name in names) == 24
    assert {"raw_std_statistics.csv", "raw_std_statistics.json", "plot_manifest.json"} <= names
    assert not any("member_sweep" in name for name in names)

def test_existing_complete_identity_is_zero_write(campaign_artifacts) -> None:
    first = publish_campaign_plots(campaign_artifacts.campaign)
    mtimes = {path.name: path.stat().st_mtime_ns for path in first.plot_dir.iterdir()}
    second = publish_campaign_plots(campaign_artifacts.campaign)
    assert second.skipped is True
    assert mtimes == {path.name: path.stat().st_mtime_ns for path in second.plot_dir.iterdir()}
```

- [ ] **Step 2: Sync and confirm rendering/store tests fail remotely**

Run:

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests/test_plot_rendering.py Uncertainty_Quantification/BootStrapping/tests/test_plot_store.py -q
```

Expected: missing rendering/store modules.

- [ ] **Step 3: Implement deterministic Carnet-style rendering**

Render gray `residual <= uncertainty` region, deterministic scatter, orange density contours, black `y=x` line, exact shared square limits, correlations, and valid/original counts. Use Agg, configured fonts/size/DPI, and deterministic PDF metadata with creation/modification timestamps set to `None`. File stems are:

```python
f"{run_label}__{dataset_label}__raw_{target}_uncertainty_vs_residual"
```

Energy units are `eV/atom`, forces `eV/Angstrom`, and stress `eV/Angstrom^3`.

- [ ] **Step 4: Implement statistics and content-addressed publication**

`raw_std_statistics.csv` has one row per panel with key, shared limits, shape, original/valid/exclusion counts, Spearman, and Pearson. JSON contains the same rows plus nested source identities and style. Compute SHA256 identity over ordered run/dataset/source manifest hashes, targets hashes, member order, raw mode, member count, analysis contract, and rendering settings.

Destination name is:

```python
f"raw_std_three_runs_three_datasets__{identity[:16]}"
```

For a generic smoke campaign use `raw_std_campaign__{identity[:16]}`. Publish into `sibling_staging(destination)`, render all 48 formal figures, write CSV and JSON, then write `plot_manifest.json` last. The manifest records every other artifact's relative path, size, and SHA256. `validate_plot_publication` requires the exact declared set and no extras.

- [ ] **Step 5: Resync and run rendering/store tests remotely**

Run the Task 7 pytest command. Expected: PASS.

- [ ] **Step 6: Commit Task 7**

```bash
git add Uncertainty_Quantification/BootStrapping/bootstrap/plot_rendering.py Uncertainty_Quantification/BootStrapping/bootstrap/plot_store.py Uncertainty_Quantification/BootStrapping/tests/test_plot_rendering.py Uncertainty_Quantification/BootStrapping/tests/test_plot_store.py
git commit -m "feat(bootstrap): publish deterministic campaign plots"
```

---

### Task 8: CLIs, formal configuration, documentation, ignores, and release boundary

**Files:**
- Create: `Uncertainty_Quantification/BootStrapping/scripts/predict_campaign.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/compute_campaign_uq.py`
- Create: `Uncertainty_Quantification/BootStrapping/scripts/plot_campaign.py`
- Create: `Uncertainty_Quantification/BootStrapping/configs/three_run_three_dataset_raw.yaml`
- Modify: `Uncertainty_Quantification/BootStrapping/README.md`
- Modify: `.gitignore`
- Modify: `Uncertainty_Quantification/BootStrapping/tests/test_scripts.py`
- Modify: `Uncertainty_Quantification/BootStrapping/tests/test_release.py`
- Create: `Uncertainty_Quantification/BootStrapping/tests/test_formal_campaign.py`

**Interfaces:**
- Consumes: campaign prediction/UQ/plot orchestration.
- Produces: three thin module CLIs, formal YAML, publishable documentation, ignored generated plot directory.

- [ ] **Step 1: Write failing CLI/config/release tests**

```python
def test_formal_campaign_has_three_runs_and_expected_targets() -> None:
    campaign = load_campaign(FORMAL_CAMPAIGN)
    assert len(campaign.runs) == 3
    assert campaign.prediction.member_count == 8
    assert campaign.prediction.mode == "raw"
    assert campaign.prediction.device == "cuda"
    assert [item.reference_targets for item in campaign.datasets] == [
        ("energy", "forces", "stress"),
        ("energy", "forces"),
        ("energy", "forces", "stress"),
    ]

@pytest.mark.parametrize("module", ["predict_campaign", "compute_campaign_uq", "plot_campaign"])
def test_campaign_cli_returns_two_on_domain_failure(module, monkeypatch, capsys) -> None:
    entry = importlib.import_module(f"Uncertainty_Quantification.BootStrapping.scripts.{module}")
    monkeypatch.setattr(entry, entry.STAGE_NAME, raise_hard_failure)
    assert entry.main(["--campaign", str(FORMAL_CAMPAIGN)]) == 2
    assert "campaign failed" in capsys.readouterr().err

def test_release_contains_campaign_and_plot_modules(release_archive) -> None:
    names = release_archive.names
    assert "BootStrapping/bootstrap/campaign.py" in names
    assert "BootStrapping/bootstrap/plot_store.py" in names
    assert "BootStrapping/scripts/plot_campaign.py" in names
    assert "BootStrapping/configs/three_run_three_dataset_raw.yaml" in names
    assert all("Plots/" not in name for name in names)
```

- [ ] **Step 2: Sync and confirm CLI/config tests fail remotely**

Run:

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests/test_scripts.py Uncertainty_Quantification/BootStrapping/tests/test_release.py Uncertainty_Quantification/BootStrapping/tests/test_formal_campaign.py -q
```

Expected: formal campaign and CLI modules are missing.

- [ ] **Step 3: Add thin CLIs with selectors**

Each prediction/UQ CLI accepts repeatable `--run` and `--dataset`; plot CLI always consumes the complete campaign so shared limits cannot be computed from a partial selection.

```python
parser.add_argument("--campaign", required=True, type=Path)
parser.add_argument("--run", action="append")
parser.add_argument("--dataset", action="append")
```

Catch only `HardFailure`, print one stderr line, and return 2. Successful commands print each publication path and return 0.

- [ ] **Step 4: Write formal YAML and README workflow**

The formal YAML uses the three existing run configs, run roots under `outputs/upet-bootstrap-head-posttrain-v1`, repository-relative dataset paths, raw mode, member count 8, CUDA, prediction batch size 8, 300 DPI, 7x7 figures, grid size 160, sigma 1.2, contour masses `0.5,0.7,0.85,0.95,0.99`, scatter cap 20,000, seed 20260714, log margin 0.05, and PNG/PDF formats.

README documents exactly:

```bash
python -m Uncertainty_Quantification.BootStrapping.scripts.predict_campaign --campaign Uncertainty_Quantification/BootStrapping/configs/three_run_three_dataset_raw.yaml
python -m Uncertainty_Quantification.BootStrapping.scripts.compute_campaign_uq --campaign Uncertainty_Quantification/BootStrapping/configs/three_run_three_dataset_raw.yaml
python -m Uncertainty_Quantification.BootStrapping.scripts.plot_campaign --campaign Uncertainty_Quantification/BootStrapping/configs/three_run_three_dataset_raw.yaml
```

Add `Uncertainty_Quantification/Plots/BootStrapping/` to root `.gitignore`. Do not ignore source modules, configs, tests, or docs.

- [ ] **Step 5: Resync and run CLI/config/release tests remotely**

Run the Task 8 pytest command. Expected: PASS.

- [ ] **Step 6: Commit Task 8**

```bash
git add .gitignore Uncertainty_Quantification/BootStrapping/README.md Uncertainty_Quantification/BootStrapping/scripts/predict_campaign.py Uncertainty_Quantification/BootStrapping/scripts/compute_campaign_uq.py Uncertainty_Quantification/BootStrapping/scripts/plot_campaign.py Uncertainty_Quantification/BootStrapping/configs/three_run_three_dataset_raw.yaml Uncertainty_Quantification/BootStrapping/tests/test_scripts.py Uncertainty_Quantification/BootStrapping/tests/test_release.py Uncertainty_Quantification/BootStrapping/tests/test_formal_campaign.py
git commit -m "docs(bootstrap): publish campaign inference workflow"
```

---

### Task 9: Full remote verification and real-checkpoint CPU smoke

**Files:**
- Modify only if failures reveal a scoped defect: files from Tasks 1-8 and their tests.
- Generated remote only: `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_bootstrap_plots_codex/Uncertainty_Quantification/BootStrapping/outputs/smoke`.

**Interfaces:**
- Consumes: complete implementation and eight existing member checkpoints.
- Produces: remote lint/type/test evidence and one real PET `matpes_n20` prediction -> UQ -> plot publication.

- [ ] **Step 1: Run all BootStrapping tests remotely with warnings as errors**

```bash
cd /XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new_bootstrap_plots_codex
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m pytest -W error Uncertainty_Quantification/BootStrapping/tests -q
```

Expected: all tests PASS.

- [ ] **Step 2: Run remote formatting, lint, and types**

```bash
/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/ruff format --check Uncertainty_Quantification/BootStrapping
/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/ruff check Uncertainty_Quantification/BootStrapping
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/mypy Uncertainty_Quantification/BootStrapping/bootstrap
```

Expected: all commands exit 0.

- [ ] **Step 3: Build and inspect the formal source release remotely**

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m Uncertainty_Quantification.BootStrapping.scripts.build_release --source-root Uncertainty_Quantification/BootStrapping --output /tmp/upet-bootstrap-plots-release.tar.gz
tar -tzf /tmp/upet-bootstrap-plots-release.tar.gz | sort
```

Expected: campaign/plot source, scripts, config and README are present; tests, internal migration, outputs, plots, checkpoints and caches are absent.

- [ ] **Step 4: Prepare an isolated eight-member CPU smoke run**

Create `Uncertainty_Quantification/BootStrapping/outputs/smoke/n20_cpu/members`. Copy production `full_remote_b8_e8/members/member_000` through `member_007` manifests and `checkpoints/best.pt` into the matching smoke paths. Do not modify the production run.

Copy `configs/full_remote_b8_e8.yaml` to `outputs/smoke/run.yaml` and change exactly: `experiment.run_id=n20_cpu`, `experiment.output_root=.`, `checkpoint.base_path` to the absolute real base checkpoint, all three data paths to the isolated `data/dataset/matpes_n20.extxyz`, `training.batch_size=2`, `training.max_epochs=1`, `training.device=cpu`, `prediction.splits=[test]`, `prediction.batch_size=2`, and `prediction.device=cpu`. Keep ensemble size 8, float32, raw mode, STD and GMD unchanged.

Write `outputs/smoke/campaign.yaml` with one run label `n20_cpu`, config `run.yaml`, run root `n20_cpu`; one dataset label/storage key `matpes_n20`, path to isolated `data/dataset/matpes_n20.extxyz`, references `[energy, forces, stress]`; prediction `{mode: raw, member_count: 8, device: cpu, batch_size: 2}`; the formal plot settings; and output root `plots`. Copy the 21,213-byte local n20 file into the isolated data path and link the isolated base-checkpoint path to the real checkpoint.

- [ ] **Step 5: Run the real PET smoke pipeline**

```bash
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m Uncertainty_Quantification.BootStrapping.scripts.predict_campaign --campaign Uncertainty_Quantification/BootStrapping/outputs/smoke/campaign.yaml
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m Uncertainty_Quantification.BootStrapping.scripts.compute_campaign_uq --campaign Uncertainty_Quantification/BootStrapping/outputs/smoke/campaign.yaml
PYTHONPATH=$PWD /HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/.conda/envs/upet_new/bin/python -m Uncertainty_Quantification.BootStrapping.scripts.plot_campaign --campaign Uncertainty_Quantification/BootStrapping/outputs/smoke/campaign.yaml
```

Expected: prediction and UQ manifests validate; the three-target smoke plot publication contains six figures plus CSV, JSON and manifest.

- [ ] **Step 6: Fix only verified defects, rerun Steps 1-5, and commit fixes**

Use one commit per independently diagnosed defect with a focused regression test. Do not weaken validation to make the smoke pass.

---

### Task 10: Production remote GPU prediction, UQ, plotting, and local result sync

**Files:**
- Generated remote only: six new `predictions/{mad_test,matpes_train}` publications.
- Generated remote only: six new `uncertainty/{mad_test,matpes_train}/raw` publications.
- Generated remote and local ignored: one `Uncertainty_Quantification/Plots/BootStrapping/raw_std_three_runs_three_datasets__{identity_prefix}` publication, where `identity_prefix` is the first 16 hexadecimal characters of the full manifest identity.
- Generated remote only: audit and scheduler logs outside the release tree.

**Interfaces:**
- Consumes: committed formal code/config, three existing run roots, base checkpoint and three datasets.
- Produces: verified nine-combination numerical source set, 51-file plot publication, and matching local copy.

- [ ] **Step 1: Sync committed formal code to the production repository without outputs**

Archive and transfer `Uncertainty_Quantification/BootStrapping` excluding outputs, then extract it under the production repository. Verify the exact resolved target is `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet_new/Uncertainty_Quantification/BootStrapping` before extraction. Do not sync `.git`, ConfidenceHead, LLPR, FGE, local data, or local plots.

- [ ] **Step 2: Materialize remote data paths without copying MATPES-train**

Create production `data/dataset` and `data` directories. Link the configured base checkpoint to the existing remote checkpoint, and link MATPES-test/train to the existing remote files. Search the user's remote code tree for `mad-test.xyz` or `mad-test-compatible.xyz`; if found, link it as `data/dataset/mad-test.xyz`. If absent, upload the local 30,505,970-byte `data/dataset/mad-test.xyz` and verify its byte size after transfer. Do not create a dataset semantic fingerprint.

- [ ] **Step 3: Run production preflight and preserve MATPES-test hashes**

Validate campaign parsing, three run manifests, all 24 `best.pt` files with expected parameter count 13,338, three dataset reference-target declarations, CUDA availability, and free output space. Save a sorted SHA256 inventory of every file below each run's `predictions/test` and `uncertainty/test/raw` into an external audit directory.

- [ ] **Step 4: Execute six GPU prediction units**

For each run in `full_remote_b8_e8`, `lr_1e-4`, `lr_1e-6`, execute:

```bash
python -m Uncertainty_Quantification.BootStrapping.scripts.predict_campaign --campaign Uncertainty_Quantification/BootStrapping/configs/three_run_three_dataset_raw.yaml --run "$RUN" --dataset mad_test
python -m Uncertainty_Quantification.BootStrapping.scripts.predict_campaign --campaign Uncertainty_Quantification/BootStrapping/configs/three_run_three_dataset_raw.yaml --run "$RUN" --dataset matpes_train
```

Use one allocated GPU per command. If Slurm is available, submit six jobs with one GPU, eight CPUs, 64 GB memory and 48-hour limits; otherwise start no direct GPU process until `nvidia-smi` confirms an available device, then run one command at a time with an explicit `CUDA_VISIBLE_DEVICES`. Monitor logs and manifest appearance. A failed unit is rerun only after diagnosing the exact failure; never reduce member count or change precision.

- [ ] **Step 5: Compute and strictly validate all campaign UQ on CPU**

Run the campaign UQ CLI for all runs/datasets. It must validate and skip the three existing MATPES-test UQ sets, and compute only the six new sets. Then call `validate_uq_publication` for every one of the nine combinations and require exact array equality with recomputation.

- [ ] **Step 6: Publish the shared-scale plots on CPU**

Run:

```bash
python -m Uncertainty_Quantification.BootStrapping.scripts.plot_campaign --campaign Uncertainty_Quantification/BootStrapping/configs/three_run_three_dataset_raw.yaml
```

Expected: 24 panels, 24 PNG, 24 PDF, CSV, JSON and manifest; energy/force/stress panel counts are 9/9/6; no member-sweep file exists.

- [ ] **Step 7: Run final read-only audits**

Validate the plot publication exact file set, artifact sizes and SHA256. Assert all nine energy panels share identical limits, all nine force panels share identical limits, and all six stress panels share identical limits. Recompute the MATPES-test hash inventories and require byte-for-byte equality with Step 3.

- [ ] **Step 8: Sync only the verified plot identity directory to local storage**

Copy the complete remote plot identity directory to local `Uncertainty_Quantification/Plots/BootStrapping/`. Compare sorted remote and local SHA256 inventories and require equality. Do not copy prediction/UQ NPZ files, checkpoints, datasets, scheduler logs or audit directories.

- [ ] **Step 9: Verify Git cleanliness and report evidence**

Run local `git status --short`, confirm plot results are ignored, and confirm only pre-existing unrelated `.idea`/ConfidenceHead files remain. Report code commits, remote test counts, six prediction and six new UQ manifests, reused MATPES-test hashes, plot identity, 51-file count, and remote/local SHA256 equality.
