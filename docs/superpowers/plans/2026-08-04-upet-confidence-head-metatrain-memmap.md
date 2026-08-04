# UPET ConfidenceHead Metatrain-Style Memmap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the pathological `.pt` shard cache with a metatrain-style continuous memmap pipeline, preserve the confirmed confidence-label and early-stopping semantics, and make single-branch GPU training observable and performant.

**Architecture:** Cache schema v2 stores every split field in a dedicated continuous binary array plus structure offsets and an immutable manifest. A lazy per-worker dataset supports deterministic global shuffling and optional atom-count batching; force and energy heads are constructed and evaluated only when their weighted target is active. The existing transactional run, weighted validation-total EMA, scheduler, and early-stopping behavior remain authoritative.

**Tech Stack:** Python 3.11, PyTorch 2.11, NumPy memmap, Pydantic, pytest, W&B, YAML, SLURM.

## Global Constraints

- Execute on the existing `ConfidenceHead` branch; do not create a worktree or new branch.
- Use `conda activate upet_new` (or `conda run -n upet_new`) for local checks.
- Cache schema v1 is rejected; do not migrate old caches, results, logs, configs, or plots.
- Force and energy must retain distinct UPET readout inputs.
- Force defaults to `atom_mean`; component mode remains supported.
- Energy labels use `abs(prediction-reference) / num_atoms` with default fixed-linear maximum `0.3`.
- Force fixed-linear maximum remains `0.5`.
- Backpropagate the weighted supervised total loss.
- Preserve validation-total EMA, `min_delta`, patience, `min_epochs`, best checkpoint, and epoch-boundary resume behavior.
- Local work is static/CPU-small-data only; GPU and full-path validation occurs on `bywang@121.48.164.204`.
- Implement production behavior only after its focused test has failed for the expected reason.
- Commit each independently reviewable task on the current branch.

---

### Task 1: Configuration and identity for cache schema v2

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/commands.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/identity.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py`

**Interfaces:**
- Produces: `TrainerConfig.num_workers: int | None`, `max_atoms_per_batch: int | None`, `min_atoms_per_batch: int`, `pin_memory: bool`, `persistent_workers: bool`, `grad_clip_norm: float | None`.
- Produces: `LoggingConfig.log_interval_steps: int`.
- Changes: force/energy `enabled` fields from `Literal[True]` to strict booleans while rejecting a positive coefficient for a disabled target.
- Removes: `CacheConfig.num_workers` and `CacheConfig.shard_max_atoms`; retains `CacheConfig.batch_size`.

- [ ] **Step 1: Write failing strict-schema and identity tests**

```python
def test_v2_runtime_options_live_under_trainer(base_config):
    config = ConfidenceConfig.model_validate(base_config)
    assert config.trainer.num_workers is None
    assert config.trainer.max_atoms_per_batch is None
    assert config.trainer.min_atoms_per_batch == 0
    assert config.trainer.pin_memory is True
    assert config.trainer.persistent_workers is True
    assert config.trainer.grad_clip_norm == 1.0
    assert config.logging.log_interval_steps == 100

def test_v1_shard_options_are_rejected(base_config):
    base_config["cache"]["shard_max_atoms"] = 100_000
    with pytest.raises(ValidationError):
        ConfidenceConfig.model_validate(base_config)
```

- [ ] **Step 2: Run focused tests and verify failure**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py`

Expected: FAIL because new trainer/logging fields do not exist and legacy cache fields remain accepted.

- [ ] **Step 3: Implement strict configuration and cache identity changes**

Define the exact defaults above, validate `min_atoms_per_batch <= max_atoms_per_batch` when the maximum is set, require non-negative worker counts, and remove shard fields from cache identity. Include all batching, active-target, and logging behavior that affects a run in run identity, but not cache identity.

- [ ] **Step 4: Run focused tests and verify pass**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/commands.py Uncertainty_Quantification/ConfidenceHead/confidence_head/identity.py Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py
git commit -m "refactor(confidence-head): define memmap runtime configuration"
```

### Task 2: Continuous memmap cache schema, writer, and lazy reader

**Files:**
- Rewrite focused sections: `Uncertainty_Quantification/ConfidenceHead/confidence_head/cache.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py`

**Interfaces:**
- Produces: `CACHE_SCHEMA_VERSION = 2`.
- Produces: `write_memmap_cache(root, identity, split_structures, metadata) -> Path` or equivalent existing public writer replacement.
- Produces: `ConfidenceMemmapDataset(manifest_path, split, expected_identity)` with `__len__`, `__getitem__`, `num_atoms(index)`, and pickle-safe lazy maps.
- Produces: `collate_cached_structures(items) -> dict[str, Any]` with `atom_counts`.

- [ ] **Step 1: Replace shard-oriented tests with failing v2 layout tests**

```python
def test_memmap_cache_round_trip(tmp_path, cached_structures):
    manifest = write_cache(tmp_path, "identity", cached_structures)
    dataset = ConfidenceMemmapDataset(manifest, "train", "identity")
    assert len(dataset) == len(cached_structures["train"])
    assert dataset[1]["structure_id"] == cached_structures["train"][1].structure_id
    assert torch.equal(dataset[1]["force_features"], cached_structures["train"][1].force_features)
    assert torch.equal(dataset[1]["energy_features"], cached_structures["train"][1].energy_features)

def test_getitem_does_not_hash_or_validate_full_arrays(monkeypatch, complete_cache):
    monkeypatch.setattr(cache_module, "sha256_file", Mock(side_effect=AssertionError))
    dataset = ConfidenceMemmapDataset(*complete_cache)
    dataset[0]
    dataset[-1]
```

Add tests for schema-v1 rejection, path confinement, mismatched byte length, non-monotonic offsets, independent force/energy storage, negative indexing, and pickle/reopen behavior.

- [ ] **Step 2: Run cache tests and verify expected failures**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py`

Expected: FAIL because the implementation still emits `.pt` shards and hashes on every shard miss.

- [ ] **Step 3: Implement v2 file descriptors and atomic writer**

Use exact field descriptors containing `path`, `dtype`, `shape`, `bytes`, and `sha256`. Store offsets as int64 and prediction/reference/features in their resolved floating dtype. Build under a transaction directory, flush arrays, hash once, then publish a complete manifest atomically.

- [ ] **Step 4: Implement lazy dataset and contiguous collate**

Keep mmap handles out of pickle state, reopen each array lazily in the worker, slice via `structure_offsets`, and return distinct force/energy views. Collate must allocate contiguous tensors once and return `atom_counts`, `structure_ids`, predictions, references, and both feature tensors.

- [ ] **Step 5: Run cache tests and verify pass**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/cache.py Uncertainty_Quantification/ConfidenceHead/tests/test_cache.py
git commit -m "feat(confidence-head): replace shard cache with memmap arrays"
```

### Task 3: Stream UPET extraction into v2 cache and full verification

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_cache.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/verify.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_cache_workflow.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_data_checkpoint_features.py`

**Interfaces:**
- Consumes: schema-v2 writer descriptors from Task 2.
- Produces: two-pass split extraction that precomputes offsets and streams frozen-UPET results into preallocated arrays.
- Produces: full verification that hashes every declared array only when explicitly requested.

- [ ] **Step 1: Write failing streaming and mutation tests**

```python
def test_build_cache_publishes_v2_memmap_fields(fake_upet, n20_config):
    manifest_path = build_cache(n20_config)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 2
    assert manifest["status"] == "complete"
    assert manifest["splits"]["train"]["arrays"]["force_features"]["path"].endswith("force_features.bin")

def test_full_verify_detects_mutated_memmap_byte(complete_cache):
    mutate_one_declared_array_byte(complete_cache)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_cache(complete_cache.manifest, full=True)
```

Also assert that a source SHA change between pass one and pass two prevents publication and that NaN/Inf readouts never produce `status: complete`.

- [ ] **Step 2: Run focused workflow tests and verify failure**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_cache_workflow.py Uncertainty_Quantification/ConfidenceHead/tests/test_data_checkpoint_features.py`

Expected: FAIL on schema/layout and streaming assertions.

- [ ] **Step 3: Implement two-pass extraction and explicit full verify**

First pass records structure ids/counts/offsets and source digest. Second pass performs inference under `torch.inference_mode()`, verifies distinct readouts, writes exact slices, and rechecks the source digest before atomic publication. Normal training uses fast metadata validation; `verify --full` performs the declared SHA scan.

- [ ] **Step 4: Run focused tests and verify pass**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/build_cache.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/verify.py Uncertainty_Quantification/ConfidenceHead/tests/test_cache_workflow.py Uncertainty_Quantification/ConfidenceHead/tests/test_data_checkpoint_features.py
git commit -m "feat(confidence-head): stream UPET readouts into memmap cache"
```

### Task 4: Deterministic global sampling and optional atom-count batching

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/confidence_head/sampler.py`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_sampler.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

**Interfaces:**
- Produces: `EpochRandomSampler(dataset, seed)` with `set_epoch(epoch)`.
- Produces: `MaxAtomBatchSampler(dataset, max_atoms, min_atoms, seed, shuffle, drop_last)` with `set_epoch(epoch)`.
- Produces: loader construction using trainer workers, persistence, pinning, and non-blocking transfer.

- [ ] **Step 1: Write failing sampler coverage and determinism tests**

```python
def test_epoch_sampler_is_complete_deterministic_and_epoch_specific(dataset):
    sampler = EpochRandomSampler(dataset, seed=1234)
    sampler.set_epoch(3)
    first = list(sampler)
    sampler.set_epoch(3)
    assert list(sampler) == first
    assert sorted(first) == list(range(len(dataset)))
    sampler.set_epoch(4)
    assert list(sampler) != first

def test_max_atom_batches_respect_limit(dataset):
    batches = list(MaxAtomBatchSampler(dataset, max_atoms=12, min_atoms=0, seed=4, shuffle=True, drop_last=False))
    assert sorted(i for batch in batches for i in batch) == list(range(len(dataset)))
    assert all(sum(dataset.num_atoms(i) for i in batch) <= 12 for batch in batches)
```

- [ ] **Step 2: Run sampler/workflow tests and verify failure**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_sampler.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

Expected: FAIL because samplers and new loader options do not exist.

- [ ] **Step 3: Implement samplers and loader integration**

Use `torch.randperm` with a CPU generator seeded from base seed and epoch. For fixed batches, pass the sampler with `shuffle=False`; for atom-count mode, pass only `batch_sampler`. Validation/test atom-count batching is ordered and never drops residual samples. Resolve automatic workers from CPU affinity and SLURM allocation.

- [ ] **Step 4: Implement contiguous pinned transfer**

Enable persistence only for positive workers, enable pinning only on CUDA, and move tensors with `non_blocking=True`. Ensure CPU tests do not request pinned CUDA allocation.

- [ ] **Step 5: Run focused tests and verify pass**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/sampler.py Uncertainty_Quantification/ConfidenceHead/tests/test_sampler.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py
git commit -m "feat(confidence-head): add deterministic memmap data loading"
```

### Task 5: Optional target branches and weighted loss

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/losses.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

**Interfaces:**
- Produces: `ConfidenceOutput(force_logits: Tensor | None, energy_logits: Tensor | None)`.
- Produces: model constructor flags `force_active: bool`, `energy_active: bool`.
- Consumes: `force_active = enabled and coefficient > 0`, `energy_active = enabled and coefficient > 0`.

- [ ] **Step 1: Write failing branch-elision tests**

```python
def test_force_only_model_has_no_energy_modules(model_kwargs):
    model = ConfidenceModel(**model_kwargs, force_active=True, energy_active=False)
    assert model.force_head is not None
    assert model.energy_adapter is None
    assert model.energy_head is None
    output = model(force_features=torch.randn(5, 4), energy_features=None, atom_counts=None)
    assert output.force_logits is not None
    assert output.energy_logits is None
```

Mirror for energy-only, assert inactive parameters are absent from optimizer, and assert a positive coefficient on `enabled: false` is rejected.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

Expected: FAIL because both heads are always built and both labels/losses are always calculated.

- [ ] **Step 3: Implement optional model and loss paths**

Construct only active modules, validate only active inputs, generate labels only for active targets, and aggregate only real sample counts. Preserve default force `atom_mean`, component mode, energy-per-atom labels, and weighted total loss.

- [ ] **Step 4: Run focused tests and verify pass**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py Uncertainty_Quantification/ConfidenceHead/confidence_head/losses.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py
git commit -m "feat(confidence-head): skip inactive confidence targets"
```

### Task 6: Vectorized structure cumulants with gradient parity

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/adapters.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py`

**Interfaces:**
- Changes: `LocalToGlobalCumulantAdapter.forward(features, atom_counts) -> Tensor`.
- Requires: no structure-level Python loop and no CUDA `.item()` in forward.

- [ ] **Step 1: Add failing output and gradient parity tests for orders 1–8**

```python
@pytest.mark.parametrize("order", range(1, 9))
@pytest.mark.parametrize("signed_root", [False, True])
def test_vectorized_cumulants_match_reference_and_gradients(order, signed_root):
    features = torch.randn(9, 3, dtype=torch.float64, requires_grad=True)
    counts = torch.tensor([2, 4, 3])
    actual = LocalToGlobalCumulantAdapter(3, order, signed_root)(features, counts)
    expected = reference_cumulants(features, counts, order, signed_root)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), features)[0], torch.autograd.grad(expected.sum(), features)[0])
```

- [ ] **Step 2: Run the parity tests and verify failure against the new interface**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py -k cumulant`

Expected: FAIL because the current adapter accepts offsets and performs `.item()` per structure.

- [ ] **Step 3: Implement degree-only vectorization**

Create structure ids with `repeat_interleave`, aggregate each raw moment with `index_add_`, divide by counts, and apply the existing cumulant recurrence and signed root. Never allocate `[atoms, feature_dim, order]`.

- [ ] **Step 4: Run parity tests and full model math tests**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/adapters.py Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py
git commit -m "perf(confidence-head): vectorize energy cumulants"
```

### Task 7: Step telemetry, gradient clipping, checkpoint identity, and resume

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/tracking.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_tracking.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

**Interfaces:**
- Produces step metrics: active loss fields, `performance/samples_per_second`, `performance/atoms_per_second`, `performance/data_wait_seconds`, and learning rate.
- Extends snapshot identity with active targets and epoch sampler seed.
- Preserves validation-total EMA control state exactly.

- [ ] **Step 1: Write failing telemetry, clipping, and resume tests**

```python
def test_step_logging_occurs_at_configured_interval(fake_tracker, training_fixture):
    train_run(..., tracker_factory=fake_tracker, log_interval_steps=2)
    step_records = [r for r in fake_tracker.records if "performance/atoms_per_second" in r]
    assert [r["global_step"] for r in step_records] == [2, 4]
    assert all("train/energy_loss" not in r for r in step_records)  # force-only fixture
```

Assert `clip_grad_norm_` occurs between backward and optimizer step when configured, is absent for `null`, resume reproduces the uninterrupted epoch order, and mismatched active targets/cache identity are rejected.

- [ ] **Step 2: Run focused tests and verify failure**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_tracking.py Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

Expected: FAIL because step telemetry and new snapshot identity are absent.

- [ ] **Step 3: Implement telemetry, clipping, and resume state**

Measure data wait around iterator fetches, use active sample/atom counts for rates, log without affecting control logic, clip after backward, and save the explicit sampler identity. Keep epoch records and W&B run-id resume behavior intact.

- [ ] **Step 4: Run focused tests and verify pass**

Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py Uncertainty_Quantification/ConfidenceHead/confidence_head/tracking.py Uncertainty_Quantification/ConfidenceHead/tests/test_tracking.py Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py
git commit -m "feat(confidence-head): add observable resumable training"
```

### Task 8: Shipped configs, integration verification, and remote-ready handoff

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu.yaml`
- Modify: `Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml`
- Modify: `Uncertainty_Quantification/ConfidenceHead/configs/n20_local_cpu.yaml`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py`
- Modify if required by behavior: `Uncertainty_Quantification/ConfidenceHead/README.md`

**Interfaces:**
- Produces strict released configs with v2 fields only.
- Produces a clean repository state ready to push and deploy.

- [ ] **Step 1: Add failing shipped-config assertions**

```python
@pytest.mark.parametrize("name", ["full_gpu.yaml", "n20_cpu.yaml", "n20_local_cpu.yaml"])
def test_shipped_config_uses_memmap_runtime_schema(name):
    raw = yaml.safe_load((CONFIG_ROOT / name).read_text())
    assert "shard_max_atoms" not in raw["cache"]
    assert "num_workers" not in raw["cache"]
    assert "num_workers" in raw["trainer"]
    load_config(CONFIG_ROOT / name)
```

- [ ] **Step 2: Run config tests and verify failure**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py`

Expected: FAIL while released YAML still contains v1 fields.

- [ ] **Step 3: Update released/test configs and concise user documentation**

Set full GPU defaults to fixed `batch_size: 128`, `num_workers: null`, `max_atoms_per_batch: null`, pinning/persistence enabled, clipping `1.0`, and step logging `100`. Set CPU test configs to `num_workers: 0`, pinning/persistence disabled, and small deterministic limits.

- [ ] **Step 4: Run complete local ConfidenceHead suite**

Run: `conda run -n upet_new python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests`

Expected: PASS with no warnings.

- [ ] **Step 5: Run repository formatting/static checks scoped to changed code**

Run: `conda run -n upet_new python -m ruff format --check Uncertainty_Quantification/ConfidenceHead`

Run: `conda run -n upet_new python -m ruff check Uncertainty_Quantification/ConfidenceHead`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Uncertainty_Quantification/ConfidenceHead/configs Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py Uncertainty_Quantification/ConfidenceHead/README.md
git commit -m "chore(confidence-head): release memmap training configuration"
```

- [ ] **Step 7: Remote validation gate before nine jobs**

Push the current branch, pull it into `/home/bywang/code/UQ/upet_new`, rebuild a v2 cache from the configured checkpoint and datasets, complete the small online-W&B GPU chain, then run a formal-cache benchmark. Do not submit nine long jobs unless training SHA calls are zero, per-epoch reads are at most twice effective split bytes, W&B reports within 100 steps, and projected/observed first epoch is at most six hours.
