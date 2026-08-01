# UPET ConfidenceHead Default Atom-Mean Force Target Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add configuration-driven `atom_mean` and `component` force targets to UPET ConfidenceHead, make per-atom arithmetic-mean absolute force error the default, and preserve explicit component-mode compatibility with historical runs.

**Architecture:** Keep the raw UPET cache mode-independent with force predictions/references shaped `[N, 3]` and distinct force/energy readout features. Select the error reduction, force head, loss shape, identities, artifacts, evaluation counts, and verification rules from one explicit force target mode; normalize only component identities to their historical representation.

**Tech Stack:** Python 3.11+, PyTorch, Pydantic v2, PyYAML, pytest, Ruff, mypy, W&B, Git/GitHub, remote CPU smoke workflow.

## Global Constraints

- `model.force.target_mode` accepts only `atom_mean` and `component`; its code default is `atom_mean`.
- `atom_mean` is `abs(prediction - reference).mean(dim=-1)`, not vector norm, RMS, probability averaging, or maximum component.
- Force uses `force_features`; energy uses `energy_features`; the two readout tensors remain distinct.
- Raw cache tensors, cache schema, cache identity, and `force_component_count = 3 * atom_count` remain unchanged and shared across modes.
- Atom-mean force logits/labels/errors have shapes `[N, B]`/`[N]`/`[N]`; component shapes remain `[N, 3, B]`/`[N, 3]`/`[N, 3]`.
- Fixed linear bins remain configurable; current defaults remain force `0.5`, per-atom energy `0.3`, and 50 bins in shipped configurations.
- Supervised total loss remains `1.0 * force_loss + 1.5 * energy_loss` unless those existing configuration coefficients are explicitly changed.
- Scheduler, best checkpoint, EMA, and early stopping continue to monitor only `val/total_loss_ema`.
- Atom-mean run names contain `-ftarget-atommean`; component run names retain the historical name without a target tag.
- Historical artifacts missing force target metadata are interpreted as component only in compatibility readers; atom-mean artifacts must declare their mode.
- Cross-mode checkpoint, resume, binning, prediction, evaluation, and verification combinations fail before publishing new artifacts.
- No old result migration, no plotting, and no full-data GPU training are included.
- Work directly on the current `ConfidenceHead` branch and make focused commits after each independently passing task.

---

## File Structure

- `confidence_head/config.py`: define `ForceTargetMode` and `ForceModelConfig`; make atom-mean the runtime default.
- `confidence_head/errors.py`: validate raw force tensors and construct mode-specific errors plus stable semantic names.
- `confidence_head/model.py`: choose a scalar or three-component force head from the explicit mode.
- `confidence_head/losses.py`: validate mode-specific logits/labels and return the correct target count.
- `confidence_head/identity.py`: normalize explicit component configuration/model-loss payloads to historical identity bytes and read legacy mode metadata safely.
- `confidence_head/run_naming.py`: add the atom-mean target tag while preserving component names byte-for-byte.
- `confidence_head/trainer.py` and `confidence_head/checkpoint.py`: persist force semantics in new checkpoints and allow missing metadata only for component compatibility.
- `confidence_head/workflows/train.py`: build mode-aware bins, identities, model, targets, losses, manifests, and resume preflight.
- `confidence_head/workflows/evaluate.py`: emit mode-aware predictions, metrics, counts, and evaluation metadata.
- `confidence_head/workflows/verify.py`: verify current atom-mean artifacts and compatible historical component artifacts without weakening identity/hash checks.
- `configs/*.yaml`: explicitly select `atom_mean` in all shipped configurations.
- `README.md`: document mode selection, shapes, counts, naming, compatibility, and unchanged early-stopping behavior.
- `tests/test_errors_binning.py`: force reduction and bin payload tests.
- `tests/test_model_math.py`: head and loss shape/count tests.
- `tests/test_config_identity_artifacts.py`: defaults, strict values, identity normalization, and artifact metadata tests.
- `tests/test_run_naming.py`: atom-mean and legacy component naming tests.
- `tests/test_trainer.py`: checkpoint metadata and compatibility tests.
- `tests/test_workflows.py`: training, resume, evaluation, and cross-mode rejection tests.
- `tests/test_commands_scripts.py`: shipped YAML and command-level defaults.

---

### Task 1: Force Target Contract, Error Reduction, Head, and Loss

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/errors.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/losses.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_errors_binning.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py`

**Interfaces:**
- Produces: `ForceTargetMode = Literal["atom_mean", "component"]`.
- Produces: `ForceModelConfig(BranchModelConfig)` with `target_mode: ForceTargetMode = "atom_mean"`.
- Produces: `force_error(prediction: Tensor, reference: Tensor, target_mode: ForceTargetMode) -> Tensor`.
- Produces: `force_error_definition(target_mode: ForceTargetMode) -> str` returning `abs_cartesian_component_mean_v1` or `abs_cartesian_component_v1`.
- Preserves: `force_component_error(prediction, reference)` as a component-mode compatibility wrapper.
- Changes: `ConfidenceModel(..., force_target_mode: ForceTargetMode, ...)`.
- Changes: `confidence_loss(..., *, force_target_mode: ForceTargetMode, force_weight=1.0, energy_weight=1.5) -> LossOutput`.

- [ ] **Step 1: Write failing configuration and force-error tests**

Add tests equivalent to:

```python
def test_force_target_mode_defaults_to_atom_mean(tmp_path: Path) -> None:
    config = ConfidenceConfig.model_validate(_valid_config(tmp_path))
    assert config.model.force.target_mode == "atom_mean"


@pytest.mark.parametrize("mode", ["norm", "rms", "mean"])
def test_force_target_mode_rejects_unknown_values(tmp_path: Path, mode: str) -> None:
    raw = _valid_config(tmp_path)
    raw["model"] = {"force": {"target_mode": mode}}
    with pytest.raises(ValidationError, match="target_mode"):
        ConfidenceConfig.model_validate(raw)


def test_atom_mean_force_error_is_arithmetic_mean_of_absolute_components() -> None:
    prediction = torch.tensor([[3.0, -4.0, 12.0], [1.0, 2.0, 3.0]])
    reference = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 6.0]])
    result = force_error(prediction, reference, "atom_mean")
    torch.testing.assert_close(result, torch.tensor([19.0 / 3.0, 2.0]))
    assert result.shape == (2,)


def test_component_force_error_preserves_three_targets_per_atom() -> None:
    prediction = torch.tensor([[1.0, -2.0, 3.0]])
    reference = torch.zeros_like(prediction)
    torch.testing.assert_close(
        force_error(prediction, reference, "component"),
        torch.tensor([[1.0, 2.0, 3.0]]),
    )
```

- [ ] **Step 2: Run the new configuration/error tests and verify they fail for missing symbols/default behavior**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_errors_binning.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py \
  -k 'force_target_mode or atom_mean_force_error or component_force_error' -v
```

Expected: failures because `ForceModelConfig`, `force_error`, and the atom-mean default do not exist.

- [ ] **Step 3: Implement strict mode configuration and error construction**

Implement the configuration shape:

```python
ForceTargetMode = Literal["atom_mean", "component"]


class ForceModelConfig(BranchModelConfig):
    target_mode: ForceTargetMode = "atom_mean"


class ModelConfig(StrictModel):
    force: ForceModelConfig = Field(default_factory=ForceModelConfig)
    energy: EnergyModelConfig = Field(default_factory=EnergyModelConfig)
```

Implement one validated error path:

```python
def force_error(
    prediction: Tensor,
    reference: Tensor,
    target_mode: ForceTargetMode,
) -> Tensor:
    _validate_force_tensors(prediction, reference)
    component_errors = torch.abs(prediction - reference)
    if target_mode == "component":
        return component_errors
    if target_mode == "atom_mean":
        return component_errors.mean(dim=-1)
    raise ValueError(f"unsupported force target mode: {target_mode!r}")


def force_error_definition(target_mode: ForceTargetMode) -> str:
    return {
        "component": "abs_cartesian_component_v1",
        "atom_mean": "abs_cartesian_component_mean_v1",
    }[target_mode]
```

Keep all existing matching-shape, `[N, 3]`, and finite-value checks in `_validate_force_tensors`; make `force_component_error` call `force_error(..., "component")`.

- [ ] **Step 4: Write failing head/loss tests for both modes and explicit mismatch rejection**

Add parametrized tests equivalent to:

```python
@pytest.mark.parametrize(
    ("mode", "expected_shape", "head_type"),
    [
        ("atom_mean", (3, 5), ConfidenceHead),
        ("component", (3, 3, 5), ComponentConfidenceHead),
    ],
)
def test_model_selects_force_head_from_explicit_mode(mode, expected_shape, head_type):
    model = ConfidenceModel(
        force_input_dim=2,
        energy_input_dim=2,
        force_hidden_dims=(4,),
        energy_hidden_dims=(4,),
        force_dropout=0.0,
        energy_dropout=0.0,
        force_num_bins=5,
        energy_num_bins=7,
        cumulant_order=1,
        signed_root=True,
        force_target_mode=mode,
    )
    assert isinstance(model.force_head, head_type)
    output = model(torch.zeros(3, 2), torch.ones(3, 2), torch.tensor([0, 3]))
    assert output.force_logits.shape == expected_shape


def test_atom_mean_loss_counts_atoms_and_keeps_supervised_total() -> None:
    force_logits = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    force_labels = torch.tensor([0, 1])
    energy_logits = torch.tensor([[2.0, 0.0]])
    energy_labels = torch.tensor([1])
    result = confidence_loss(
        force_logits,
        force_labels,
        energy_logits,
        energy_labels,
        force_target_mode="atom_mean",
    )
    assert result.force_count == 2
    torch.testing.assert_close(
        result.total,
        result.force + 1.5 * result.energy,
    )
```

Also test that atom-mean rejects `[N, 3, B]`, component rejects `[N, B]`, and labels must exactly match their selected mode.

- [ ] **Step 5: Run the new model/loss tests and verify they fail on the old fixed component implementation**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py \
  -k 'force_head_from_explicit_mode or atom_mean_loss or force_target_mode' -v
```

Expected: failures because the model and loss do not accept `force_target_mode`.

- [ ] **Step 6: Implement mode-selected head construction and explicit loss validation**

Construct the force head without inferring semantics from output rank:

```python
self.force_target_mode = force_target_mode
force_head_type = (
    ConfidenceHead if force_target_mode == "atom_mean" else ComponentConfidenceHead
)
self.force_head = force_head_type(
    force_input_dim,
    force_hidden_dims,
    force_dropout,
    force_num_bins,
)
```

In `confidence_loss`, branch on `force_target_mode`, validate exact shapes, flatten only component logits/labels for cross entropy, and set `force_count` to `N` or `3N`. Preserve independent energy-bin support and the configured weighted total.

- [ ] **Step 7: Run focused mathematical tests and the existing readout-separation regression tests**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_errors_binning.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py -v
```

Expected: all selected files pass, including distinct force/energy readout checks.

- [ ] **Step 8: Commit the mathematical contract**

```bash
git add \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/errors.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/model.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/losses.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_errors_binning.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_model_math.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py
git commit -m "feat(confidence-head): add atom-mean force target math"
```

---

### Task 2: Mode-Aware Binning, Identities, and Run Names

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/identity.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/run_naming.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_run_naming.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

**Interfaces:**
- Produces: `artifact_force_target_mode(payload: Mapping[str, Any]) -> ForceTargetMode`, where missing metadata means component only for artifact compatibility.
- Produces: `assert_force_semantics(payload, expected_mode, context) -> None` for explicit mismatch errors.
- Changes: `config_id` and `model_loss_id` canonicalization omit `model.force.target_mode` only when it equals `component`.
- Changes: `_bin_payload(force, energy, force_target_mode)`; component returns the historical force branch, atom-mean adds `target_mode` and `error_definition`.
- Changes: `build_run_name`; only atom-mean inserts `-ftarget-atommean` after `fmax...`.

- [ ] **Step 1: Write failing identity and naming compatibility tests**

Cover these exact invariants:

```python
def test_explicit_component_config_id_matches_historical_missing_mode() -> None:
    legacy = {"model": {"force": {"num_bins": 50}}, "loss": {"force_coefficient": 1.0}}
    explicit = copy.deepcopy(legacy)
    explicit["model"]["force"]["target_mode"] = "component"
    assert config_id(explicit) == config_id(legacy)


def test_atom_mean_config_and_model_loss_ids_differ_from_component() -> None:
    component = {"model": {"force": {"num_bins": 50, "target_mode": "component"}}}
    atom_mean = {"model": {"force": {"num_bins": 50, "target_mode": "atom_mean"}}}
    assert config_id(component) != config_id(atom_mean)
    assert model_loss_id(component) != model_loss_id(atom_mean)


def test_atom_mean_run_name_adds_target_tag_but_component_is_legacy_name(tmp_path):
    atom_mean = _config(tmp_path)
    component = _with_force_mode(atom_mean, "component")
    assert "-ftarget-atommean" in build_run_name(atom_mean)
    assert build_run_name(component) == (
        "demo_fixed-linear_f3-fmax0.5-fw1-fmlp4_"
        "e3-emax0.3-ew1.5-emlp4-order2"
    )
```

Also assert the same raw cache manifest gives the same cache ID for both modes, while `_bin_payload` and `binning_id` differ.

- [ ] **Step 2: Run the new identity/name tests and verify failure**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_run_naming.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py \
  -k 'component_config_id or atom_mean_config or target_tag or shared_cache or binning_identity' -v
```

Expected: failures because mode-aware normalization, bin payloads, and naming are absent.

- [ ] **Step 3: Implement non-mutating component identity normalization**

Deep-copy mappings before removing `model.force.target_mode`; never mutate resolved configuration or loaded artifacts. Apply the normalization inside `config_id` and `model_loss_id`, not `stable_id` or `cache_id`:

```python
def _legacy_component_identity_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(payload))
    model = normalized.get("model")
    force = model.get("force") if isinstance(model, dict) else None
    if isinstance(force, dict) and force.get("target_mode") == "component":
        force.pop("target_mode")
    return normalized
```

Add artifact readers that return component when the field is absent, validate declared mode/error-definition pairs, and reject malformed or contradictory metadata.

- [ ] **Step 4: Implement bin payload and run-name mode isolation**

For atom-mean only, extend the force branch:

```python
force_payload.update(
    {
        "target_mode": "atom_mean",
        "error_definition": "abs_cartesian_component_mean_v1",
    }
)
```

For component, return the original force branch exactly. In `build_run_name`, insert `-ftarget-atommean` only when the configured mode is atom-mean.

- [ ] **Step 5: Run identity/name/binning tests**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_run_naming.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_errors_binning.py -v
```

Expected: all pass; legacy component expected strings and IDs remain unchanged.

- [ ] **Step 6: Commit identity and naming behavior**

```bash
git add \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/identity.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/run_naming.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_run_naming.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py
git commit -m "feat(confidence-head): isolate force target identities"
```

---

### Task 3: Training, Checkpoint, Resume, and Early-Stopping Integration

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/checkpoint.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`

**Interfaces:**
- Extends: `TrainingIdentity` with `force_target_mode` and `force_error_definition` semantic fields.
- New checkpoints persist those fields; compatibility validation accepts both fields missing only when expected mode is component.
- `_batch_loss` passes `config.model.force.target_mode` into `force_error` and `confidence_loss`.
- `ConfidenceModel` construction passes `force_target_mode=config.model.force.target_mode`.
- Run manifests and resolved config explicitly persist atom-mean semantics; new component manifests may persist semantics while retaining historical hashed IDs.

- [ ] **Step 1: Write failing end-to-end training-shape and checkpoint metadata tests**

Use the synthetic cached workflow fixture to train both modes and assert:

```python
atom_run = train_from_config(_with_force_mode(config, "atom_mean"))
component_run = train_from_config(_with_force_mode(config, "component"))

atom_snapshot = torch.load(atom_run / "checkpoints" / "best.pt", weights_only=False)
component_snapshot = torch.load(component_run / "checkpoints" / "best.pt", weights_only=False)

assert atom_snapshot["force_target_mode"] == "atom_mean"
assert atom_snapshot["force_error_definition"] == "abs_cartesian_component_mean_v1"
assert component_snapshot["force_target_mode"] == "component"
assert component_snapshot["force_error_definition"] == "abs_cartesian_component_v1"
```

Add a loss-accumulator assertion that atom-mean contributes `N` force samples and component contributes `3N`, while the JSONL keys and `val/total_loss_ema` monitor remain identical.

- [ ] **Step 2: Run the focused workflow/checkpoint tests and verify failure**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py \
  -k 'force_target_metadata or force_target_count or total_loss_ema' -v
```

Expected: failures because snapshots and training paths are component-only.

- [ ] **Step 3: Thread the explicit mode through training without altering early stopping**

Replace training error construction with:

```python
force_observed = force_error(
    batch["force_prediction"],
    batch["force_reference"],
    config.model.force.target_mode,
)
```

Pass the same mode to `ConfidenceModel` and `confidence_loss`. Do not modify `advance_validation_epoch`, scheduler creation, EMA formulas, checkpoint selection, stop-reason computation, or JSONL metric names.

- [ ] **Step 4: Persist and validate force semantics in checkpoints and run manifests**

Write both semantic fields into new snapshots and the run manifest. Add a single compatibility check:

```python
def _validate_force_semantics(
    payload: Mapping[str, Any],
    expected_mode: ForceTargetMode,
    *,
    context: str,
) -> None:
    declared = payload.get("force_target_mode")
    definition = payload.get("force_error_definition")
    if declared is None and definition is None and expected_mode == "component":
        return
    if declared != expected_mode or definition != force_error_definition(expected_mode):
        raise ValueError(f"{context} force target semantics mismatch")
```

Call it before loading model/optimizer state and before creating or rewriting run artifacts.

- [ ] **Step 5: Write failing cross-mode resume tests**

Train one epoch in each mode, then attempt both invalid resumes:

```python
with pytest.raises(ValueError, match="force target semantics mismatch"):
    _resume_with_mode(atom_mean_checkpoint, "component")

with pytest.raises(ValueError, match="force target semantics mismatch"):
    _resume_with_mode(component_checkpoint, "atom_mean")
```

Also create a historical component snapshot by removing both semantic fields and assert explicit component resume succeeds, while atom-mean rejects it.

- [ ] **Step 6: Run resume tests and verify the new tests fail before adding compatibility logic**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py \
  -k 'cross_mode_resume or legacy_component_checkpoint' -v
```

Expected: cross-mode or legacy-compatibility assertions fail.

- [ ] **Step 7: Complete resume preflight compatibility and rerun training tests**

Ensure resolved-config comparison uses semantic component compatibility rather than raw dictionary equality, while atom-mean requires explicit equality. Keep all existing run confinement, lock, hash, immutable-artifact, and W&B resume preflight checks.

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py -v
```

Expected: both files pass, including all existing restart and early-stopping tests.

- [ ] **Step 8: Commit training integration**

```bash
git add \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/trainer.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/checkpoint.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/train.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_trainer.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py
git commit -m "feat(confidence-head): train mode-aware force confidence"
```

---

### Task 4: Evaluation Artifacts, Counts, and Verification

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/verify.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py`

**Interfaces:**
- `test_predictions.pt` adds string metadata `force_target_mode` and `force_error_definition`; tensor payload shapes follow the selected mode.
- Evaluation `test_counts` contains `structures`, `atoms`, raw `force_components`, actual `force_targets`, and `force_target_mode`.
- Historical component prediction files may omit the two semantic fields and historical evaluation manifests may omit `force_targets`; verification derives component semantics and `3 * atoms` only in that legacy case.
- `_verify_cache` compares only raw cache counts (`structures`, `atoms`, `force_components`) and separately validates `force_targets` against the selected mode.

- [ ] **Step 1: Write failing atom-mean evaluation artifact tests**

For a synthetic test split containing `N` atoms and `B` force bins, assert:

```python
predictions = torch.load(run_dir / "evaluation/test_predictions.pt", weights_only=True)
assert predictions["force_logits"].shape == (N, B)
assert predictions["force_labels"].shape == (N,)
assert predictions["force_observed_errors"].shape == (N,)
assert predictions["force_expected_errors"].shape == (N,)
assert predictions["force_target_mode"] == "atom_mean"

evaluation = json.loads((run_dir / "evaluation/manifest.json").read_text())
assert evaluation["test_counts"] == {
    "structures": S,
    "atoms": N,
    "force_components": 3 * N,
    "force_targets": N,
    "force_target_mode": "atom_mean",
}
```

Store metadata in a form accepted by `weights_only=True` and update the verifier's required-field/type partition accordingly.

- [ ] **Step 2: Run atom-mean evaluation tests and verify failure**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py \
  -k 'atom_mean_evaluation or force_target_counts or prediction_semantics' -v
```

Expected: failures because evaluation still emits component tensors and overloads `force_components`.

- [ ] **Step 3: Implement mode-aware evaluation and raw/target count separation**

Pass mode into model construction and `force_error`; keep metric and CSV code flattening the current target tensor only after preserving mode-specific tensors in `test_predictions.pt`. Compute counts as:

```python
atom_count = int(predictions["atom_offsets"][-1])
force_target_count = int(predictions["force_labels"].numel())
test_counts = {
    "structures": len(predictions["structure_ids"]),
    "atoms": atom_count,
    "force_components": 3 * atom_count,
    "force_targets": force_target_count,
    "force_target_mode": config.model.force.target_mode,
}
```

Write both semantic fields at prediction and evaluation-manifest top level and verify the checkpoint semantics before inference.

- [ ] **Step 4: Write failing verifier compatibility and tamper tests**

Cover:

- valid atom-mean `[N, B]` predictions and `force_targets=N` pass;
- valid component `[N, 3, B]` predictions and `force_targets=3N` pass;
- historical component predictions/manifests missing semantic fields and `force_targets` pass;
- atom-mean artifacts missing semantic fields fail;
- mismatched mode, definition, logits rank, label shape, force target count, or raw component count fail;
- evaluation with a checkpoint from the other mode fails before writing evaluation artifacts.

- [ ] **Step 5: Run verifier compatibility tests and verify failure before implementation**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py \
  -k 'verify_force_target or legacy_component_evaluation or cross_mode_evaluation' -v
```

Expected: failures on mode-aware shapes/counts and historical compatibility.

- [ ] **Step 6: Implement strict mode-aware verification with narrow legacy fallback**

Resolve mode once from resolved configuration and artifact metadata. Validate tensor fields separately from string metadata. For historical component manifests only, derive:

```python
force_target_mode = "component"
force_targets = 3 * atom_count
```

Do not infer atom-mean from tensor rank and do not reinterpret raw `force_components`. Preserve all existing artifact-set, digest, identity, offset, label-range, representative, expected-error, cache, and no-image checks.

- [ ] **Step 7: Run full workflow/command test files**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py -v
```

Expected: all pass.

- [ ] **Step 8: Commit evaluation and verification behavior**

```bash
git add \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/evaluate.py \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/workflows/verify.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_workflows.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py
git commit -m "feat(confidence-head): verify force target artifacts"
```

---

### Task 5: Shipped Configurations and User Documentation

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/configs/n20_local_cpu.yaml`
- Modify: `Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml`
- Modify: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu.yaml`
- Modify: `Uncertainty_Quantification/ConfidenceHead/README.md`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py`

**Interfaces:**
- Every shipped YAML explicitly contains `model.force.target_mode: atom_mean`.
- README documents the exact reduction, output shapes, count meanings, compatibility rule, run tag, and unchanged monitor.

- [ ] **Step 1: Write a failing shipped-configuration contract test**

Add:

```python
@pytest.mark.parametrize(
    "name",
    ["n20_local_cpu.yaml", "n20_cpu.yaml", "full_gpu.yaml"],
)
def test_shipped_configs_explicitly_select_atom_mean(name: str) -> None:
    raw = yaml.safe_load((CONFIG_ROOT / name).read_text(encoding="utf-8"))
    assert raw["model"]["force"]["target_mode"] == "atom_mean"
    config = load_config(CONFIG_ROOT / name)
    assert config.model.force.target_mode == "atom_mean"
    assert config.scheduler.monitor == "val/total_loss_ema"
    assert config.trainer.monitor == "val/total_loss_ema"
```

- [ ] **Step 2: Run the shipped-config test and verify failure**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py \
  -k shipped_configs_explicitly_select_atom_mean -v
```

Expected: all three cases fail because the field is absent.

- [ ] **Step 3: Add explicit mode to all YAML files and document both modes**

Insert directly under each `model.force.enabled`:

```yaml
target_mode: atom_mean
```

Document:

- arithmetic mean of absolute Cartesian-component errors;
- atom-mean and component tensor shapes;
- `force_components` versus `force_targets`;
- `-ftarget-atommean` naming;
- explicit `component` access to historical runs;
- shared raw cache and distinct UPET energy/force readouts;
- force `0.5`, per-atom energy `0.3`, supervised total-loss weights;
- `val/total_loss_ema` monitoring;
- no plotting and no old-result migration.

- [ ] **Step 4: Run config and command tests**

Run:

```bash
conda run -n upet_new pytest \
  Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py -v
```

Expected: all pass and production config remains online W&B while n20 CPU configs remain offline as currently configured.

- [ ] **Step 5: Commit configuration and documentation**

```bash
git add \
  Uncertainty_Quantification/ConfidenceHead/configs/n20_local_cpu.yaml \
  Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml \
  Uncertainty_Quantification/ConfidenceHead/configs/full_gpu.yaml \
  Uncertainty_Quantification/ConfidenceHead/README.md \
  Uncertainty_Quantification/ConfidenceHead/tests/test_commands_scripts.py
git commit -m "docs(confidence-head): default force targets to atom mean"
```

---

### Task 6: Full Local Verification, GitHub Transfer, and Remote CPU n20 Chain

**Files:**
- Verify: `Uncertainty_Quantification/ConfidenceHead/**`
- Create only on remote runtime: a temporary explicit-component config outside tracked source if needed for legacy evaluation
- Do not create plots or modify historical output directories

**Interfaces:**
- Local validation is static/unit-only in `upet_new`.
- Remote validation checks out the exact pushed GitHub commit below `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/`.
- Remote n20 uses `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/upet/matpes_n20.extxyz` and the configured checkpoint path.

- [ ] **Step 1: Run the complete ConfidenceHead test suite**

Run:

```bash
conda run -n upet_new pytest Uncertainty_Quantification/ConfidenceHead/tests -q
```

Expected: zero failures and zero errors.

- [ ] **Step 2: Run formatting, lint, and type checks without changing unrelated files**

Run:

```bash
conda run -n upet_new ruff format --check \
  Uncertainty_Quantification/ConfidenceHead/confidence_head \
  Uncertainty_Quantification/ConfidenceHead/tests
conda run -n upet_new ruff check \
  Uncertainty_Quantification/ConfidenceHead/confidence_head \
  Uncertainty_Quantification/ConfidenceHead/tests
conda run -n upet_new mypy \
  Uncertainty_Quantification/ConfidenceHead/confidence_head \
  Uncertainty_Quantification/ConfidenceHead/tests
```

Expected: all commands exit zero. If formatting is required, apply Ruff only to the listed ConfidenceHead files, inspect the diff, and rerun all three checks.

- [ ] **Step 3: Parse all shipped YAML and check the final diff**

Run:

```bash
conda run -n upet_new python -c "from pathlib import Path; from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import load_config; root=Path('Uncertainty_Quantification/ConfidenceHead/configs'); [load_config(path) for path in sorted(root.glob('*.yaml'))]"
git diff --check
git status --short
```

Expected: all YAML parse; diff check is clean; only intended files are modified or committed.

- [ ] **Step 4: Commit any final focused fixes and push the current branch**

```bash
git status --short
git push origin ConfidenceHead
git rev-parse HEAD
```

Expected: push succeeds and returns the SHA used for remote checkout.

- [ ] **Step 5: Pull the exact GitHub SHA on the remote server**

Connect with the user-authorized SSH identity and port, enter `/XYFS01/HOME/yt_hku_psmanyam/yt_hku_psmanyam_3/code/`, clone or fetch the repository, check out the pushed `ConfidenceHead` SHA, and verify:

```bash
git rev-parse HEAD
```

Expected: exact equality with the local pushed SHA.

- [ ] **Step 6: Reuse/build one raw cache and run the default atom-mean CPU chain**

Activate the previously provisioned remote environment containing PyTorch `2.11.0+cu128`, then run the repository's existing script sequence with `configs/n20_cpu.yaml`:

```bash
python Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/train.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml
python Uncertainty_Quantification/ConfidenceHead/scripts/verify.py \
  --config Uncertainty_Quantification/ConfidenceHead/configs/n20_cpu.yaml
```

Expected: CPU-only completion, no figures, W&B offline run for n20, and no second feature extraction when the complete cache already exists.

- [ ] **Step 7: Inspect atom-mean n20 acceptance values**

Load the completed evaluation artifacts and assert:

```python
assert predictions["force_logits"].shape == (143, 50)
assert evaluation["test_counts"]["atoms"] == 143
assert evaluation["test_counts"]["force_components"] == 429
assert evaluation["test_counts"]["force_targets"] == 143
assert evaluation["test_counts"]["force_target_mode"] == "atom_mean"
assert "ftarget-atommean" in run_dir.name
```

Confirm checkpoints, JSONL, run manifest, evaluation manifest, metrics, CSV files, and W&B offline directory exist and are declared with matching hashes.

- [ ] **Step 8: Validate explicit component compatibility against the shared cache**

Create a temporary copy of `n20_cpu.yaml` outside the tracked checkout, set only `model.force.target_mode: component`, and use it to evaluate/verify an existing historical component run. Confirm the old run name has no target tag, the checkpoint is not rewritten, predictions remain `[143, 3, 50]`, `force_components=429`, `force_targets=429` or is correctly derived when absent, and the cache ID/path is identical to atom-mean.

- [ ] **Step 9: Report exact evidence and stop before GPU/full-data work**

Record the pushed SHA, remote SHA, local test counts, lint/type results, atom-mean run path, cache ID, prediction shape, count fields, W&B mode/path, component compatibility result, and absence of image artifacts. Do not launch `full_gpu.yaml`.

