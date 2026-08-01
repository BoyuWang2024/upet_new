# ConfidenceHead Nine Single-Branch Ablations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add nine independently editable UPET ConfidenceHead GPU configurations and matching submission scripts for one force-only run and eight energy-only cumulant-order runs.

**Architecture:** Keep the existing two-head training pipeline and derived run naming intact. Relax only the loss-coefficient configuration contract so either branch may have zero weight while rejecting an all-zero objective, then encode every experiment as a complete YAML file and every launch as a complete Slurm script. Deploy through the current Git branch and GitHub so the bywang checkout can fast-forward without overwriting its modified `full_gpu.yaml` or untracked `run/submit.sh`.

**Tech Stack:** Python 3.11, Pydantic v2, PyYAML, pytest, Ruff, mypy, Bash/Slurm, Git.

## Global Constraints

- Do not start full-data GPU training and do not add plotting commands.
- Keep force supervision at `model.force.target_mode: atom_mean`.
- Keep energy errors per atom and preserve independent force/energy readout inputs.
- Keep scheduler, best-checkpoint selection, EMA, and early stopping on `val/total_loss_ema`.
- Use force bin maximum `0.5`, energy bin maximum `0.3`, 50 bins, and MLP dimensions `[256, 256, 256]`.
- Use `trainer.batch_size: 32`, `max_epochs: 100`, and online W&B project `upet-confidence-head`.
- Use `run.name_prefix: upet_full_rtx5090` and preserve the `ftarget-atommean` derived-name tag.
- Use checkpoint `/home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt` with SHA-256 `879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004`.
- Use train/validation/test files below `/home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/` with verified SHA-256 values `12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec`, `5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef`, and `1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d` respectively.
- Use output root `/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/outputs`; actual run directories remain under its `runs/` child.
- Preserve the remote user's modified `configs/full_gpu.yaml` and untracked `run/submit.sh`.

---

## File Structure

- `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py`: permit one zero loss coefficient and reject the all-zero objective.
- `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py`: unit-test the revised loss contract.
- `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_energy0.yaml`: complete force-only production configuration.
- `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order{1..8}.yaml`: complete energy-only production configurations.
- `Uncertainty_Quantification/ConfidenceHead/run/submit_energy0.sh`: four-stage force-only Slurm launch.
- `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order{1..8}.sh`: four-stage energy-only Slurm launches.
- `Uncertainty_Quantification/ConfidenceHead/tests/test_ablation_configs.py`: validate the nine YAML contracts, derived names, paths, and one-to-one script mapping.

### Task 1: Permit Exactly One Zero Loss Coefficient

**Files:**
- Modify: `Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py:128-133`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py:310-350`

**Interfaces:**
- Consumes: `ConfidenceConfig.model_validate(raw: dict[str, Any]) -> ConfidenceConfig`.
- Produces: `LossConfig` accepting `(force_coefficient, energy_coefficient)` values `(1, 0)` and `(0, 1)`, while rejecting `(0, 0)` and all negative values.

- [ ] **Step 1: Write failing tests for one-zero and all-zero coefficients**

Add these tests after `test_config_exposes_independent_heads_scheduler_batch_and_logging`:

```python
@pytest.mark.parametrize(
    ("force_coefficient", "energy_coefficient"),
    [(1.0, 0.0), (0.0, 1.0)],
)
def test_config_accepts_single_branch_loss(
    tmp_path: Path,
    force_coefficient: float,
    energy_coefficient: float,
) -> None:
    raw = _valid_config(tmp_path)
    raw["loss"] = {
        "force_coefficient": force_coefficient,
        "energy_coefficient": energy_coefficient,
    }

    config = ConfidenceConfig.model_validate(raw)

    assert config.loss.force_coefficient == force_coefficient
    assert config.loss.energy_coefficient == energy_coefficient


def test_config_rejects_all_zero_loss(tmp_path: Path) -> None:
    raw = _valid_config(tmp_path)
    raw["loss"] = {"force_coefficient": 0.0, "energy_coefficient": 0.0}

    with pytest.raises(ValidationError, match="at least one loss coefficient"):
        ConfidenceConfig.model_validate(raw)
```

- [ ] **Step 2: Run the focused tests and confirm the contract is currently unsupported**

Run from the repository root:

```bash
conda run -n upet_new pytest -q \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py \
  -k 'single_branch_loss or all_zero_loss'
```

Expected: both single-branch cases fail because zero violates `gt=0`; the all-zero case does not yet produce the required combination-specific message.

- [ ] **Step 3: Implement the minimal Pydantic validation change**

Replace `LossConfig` with:

```python
class LossConfig(StrictModel):
    force_coefficient: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    energy_coefficient: float = Field(default=1.5, ge=0, allow_inf_nan=False)
    label_smoothing: Literal[0] = 0
    class_weights: None = None

    @model_validator(mode="after")
    def validate_nonzero_objective(self) -> "LossConfig":
        if self.force_coefficient == 0 and self.energy_coefficient == 0:
            raise ValueError("at least one loss coefficient must be positive")
        return self
```

- [ ] **Step 4: Add negative-coefficient regression coverage**

Add:

```python
@pytest.mark.parametrize("field", ["force_coefficient", "energy_coefficient"])
def test_config_rejects_negative_loss_coefficient(
    tmp_path: Path, field: st
) -> None:
    raw = _valid_config(tmp_path)
    raw["loss"] = {"force_coefficient": 1.0, "energy_coefficient": 1.0}
    raw["loss"][field] = -0.1

    with pytest.raises(ValidationError, match=field):
        ConfidenceConfig.model_validate(raw)
```

- [ ] **Step 5: Run the loss-contract tests**

Run the Task 1 command again. Expected: all selected tests pass.

- [ ] **Step 6: Commit the loss-contract change**

```bash
git add \
  Uncertainty_Quantification/ConfidenceHead/confidence_head/config.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py
git commit -m "feat(confidence-head): allow single-branch loss"
```

### Task 2: Add Nine Complete Production Configurations

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_energy0.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order1.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order2.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order3.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order4.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order5.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order6.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order7.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order8.yaml`
- Create: `Uncertainty_Quantification/ConfidenceHead/tests/test_ablation_configs.py`

**Interfaces:**
- Consumes: `load_config(path: Path) -> ConfidenceConfig` and `build_run_name(config: ConfidenceConfig) -> str`.
- Produces: nine strict production configs whose names and loss/order triples are described by `EXPECTED_EXPERIMENTS` below.

- [ ] **Step 1: Write the failing experiment-matrix test**

Create `test_ablation_configs.py` with imports and matrix:

```python
from __future__ import annotations

from pathlib import Path

import yaml

from Uncertainty_Quantification.ConfidenceHead.confidence_head.config import load_config
from Uncertainty_Quantification.ConfidenceHead.confidence_head.run_naming import (
    build_run_name,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RUN = ROOT / "run"
EXPECTED_EXPERIMENTS = {
    "full_gpu_energy0.yaml": (1.0, 0.0, 3),
    **{
        f"full_gpu_force0_order{order}.yaml": (0.0, 1.0, order)
        for order in range(1, 9)
    },
}


def test_ablation_config_matrix() -> None:
    run_names: set[str] = set()
    for name, expected in EXPECTED_EXPERIMENTS.items():
        config = load_config(CONFIGS / name)
        assert (
            config.loss.force_coefficient,
            config.loss.energy_coefficient,
            config.model.energy.cumulant_order,
        ) == expected
        assert config.model.force.target_mode == "atom_mean"
        assert config.trainer.batch_size == 32
        assert config.trainer.monitor == "val/total_loss_ema"
        assert config.run.name_prefix == "upet_full_rtx5090"
        assert config.run.output_root == Path(
            "/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/"
            "ConfidenceHead/outputs"
        )
        run_names.add(build_run_name(config))

    assert len(run_names) == 9
```

- [ ] **Step 2: Run the matrix test and confirm files are absent**

```bash
conda run -n upet_new pytest -q \
  Uncertainty_Quantification/ConfidenceHead/tests/test_ablation_configs.py \
  -k ablation_config_matrix
```

Expected: FAIL with `FileNotFoundError` for `full_gpu_energy0.yaml`.

- [ ] **Step 3: Create the force-only YAML**

Copy the complete field structure of `configs/full_gpu.yaml`, then set the following exact values in `full_gpu_energy0.yaml`:

```yaml
checkpoint:
  path: /home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt
  expected_sha256: 879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004
data:
  train:
    path: /home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_train.extxyz
    expected_sha256: 12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec
  validation:
    path: /home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_val.extxyz
    expected_sha256: 5b2ce7f0835f0f69d27840116608ee264536d2cc0ac253a33625ece29f985eef
  test:
    path: /home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_test.extxyz
    expected_sha256: 1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d
loss:
  force_coefficient: 1.0
  energy_coefficient: 0.0
trainer:
  batch_size: 32
run:
  output_root: /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/outputs
  name_prefix: upet_full_rtx5090
```

Keep `model.energy.cumulant_order: 3`, `model.force.target_mode: atom_mean`, all fixed-linear binning fields, optimizer, scheduler, early-stopping fields, and W&B online fields identical to the approved base configuration.

- [ ] **Step 4: Create the eight energy-only YAML files**

For each integer `N` from 1 through 8, create `full_gpu_force0_orderN.yaml` with the same complete content as the force-only YAML except for these exact fields:

```yaml
model:
  energy:
    cumulant_order: N
loss:
  force_coefficient: 0.0
  energy_coefficient: 1.0
```

Replace `N` with the literal integer in each file; do not add YAML anchors, includes, or runtime templating.

- [ ] **Step 5: Extend the test with verified input identities and name fragments**

Inside the matrix loop add:

```python
        assert config.checkpoint.expected_sha256 == (
            "879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004"
        )
        assert config.data.train.expected_sha256 == (
            "12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec"
        )
        run_name = build_run_name(config)
        assert "-ftarget-atommean-" in run_name
        assert f"-fw{expected[0]:g}-" in run_name
        assert f"-ew{expected[1]:g}-" in run_name
        assert run_name.endswith(f"-order{expected[2]}")
```

Use `run_names.add(run_name)` instead of rebuilding the name.

- [ ] **Step 6: Run the configuration tests**

```bash
conda run -n upet_new pytest -q \
  Uncertainty_Quantification/ConfidenceHead/tests/test_config_identity_artifacts.py \
  Uncertainty_Quantification/ConfidenceHead/tests/test_ablation_configs.py
```

Expected: PASS.

- [ ] **Step 7: Commit the production configurations**

```bash
git add \
  Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_energy0.yaml \
  Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order*.yaml \
  Uncertainty_Quantification/ConfidenceHead/tests/test_ablation_configs.py
git commit -m "feat(confidence-head): add single-branch GPU configs"
```

### Task 3: Add Nine One-to-One Slurm Submission Scripts

**Files:**
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_energy0.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order1.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order2.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order3.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order4.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order5.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order6.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order7.sh`
- Create: `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order8.sh`
- Modify: `Uncertainty_Quantification/ConfidenceHead/tests/test_ablation_configs.py`

**Interfaces:**
- Consumes: the nine YAML filenames from `EXPECTED_EXPERIMENTS`.
- Produces: one launch script per YAML, each running cache, train, evaluate, and verify in that order.

- [ ] **Step 1: Write the failing script-mapping test**

Append:

```python
def test_ablation_submit_scripts_are_one_to_one() -> None:
    for config_name in EXPECTED_EXPERIMENTS:
        suffix = config_name.removeprefix("full_gpu_").removesuffix(".yaml")
        script = RUN / f"submit_{suffix}.sh"
        text = script.read_text(encoding="utf-8")
        config_path = (
            "/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/"
            f"ConfidenceHead/configs/{config_name}"
        )
        assert "conda activate upet_new" in text
        assert text.count(config_path) == 4
        commands = [
            "scripts/build_cache.py",
            "scripts/train.py",
            "scripts/evaluate.py",
            "scripts/verify.py",
        ]
        positions = [text.index(command) for command in commands]
        assert positions == sorted(positions)
        assert "plot" not in text.lower()
```

- [ ] **Step 2: Run the mapping test and confirm scripts are absent**

Run the Task 2 test command with `-k submit_scripts`. Expected: FAIL with `FileNotFoundError` for `submit_energy0.sh`.

- [ ] **Step 3: Create all nine complete scripts**

Use this exact body for every file, substituting only `CONFIG_BASENAME` with its matching YAML filename:

```bash
#!/bin/bash -l
#SBATCH --job-name=carnet_SE
#SBATCH --output=./logs/slurm-%j.out
#SBATCH --error=./logs/slurm-%j.e
## 如需指定分区，去掉下一行开头的“##”并把 gpu 改成你的分区名
## #SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10-24:00:00
#SBATCH --gres=gpu:1

set -euo pipefail

conda activate upet_new
#ulimit -n 65535
export WANDB_BASE_URL="https://api.bandw.top"

python /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py --config /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/configs/CONFIG_BASENAME
python /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/scripts/train.py --config /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/configs/CONFIG_BASENAME
python /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py --config /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/configs/CONFIG_BASENAME
python /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/scripts/verify.py --config /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/configs/CONFIG_BASENAME
```

The nine substitutions are `full_gpu_energy0.yaml` and `full_gpu_force0_order1.yaml` through `full_gpu_force0_order8.yaml`.

- [ ] **Step 4: Mark the scripts executable and run static shell parsing**

```bash
chmod +x Uncertainty_Quantification/ConfidenceHead/run/submit_*.sh
bash -n Uncertainty_Quantification/ConfidenceHead/run/submit_energy0.sh
for order in 1 2 3 4 5 6 7 8; do
  bash -n "Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order${order}.sh"
done
```

Expected: exit code 0 with no output.

- [ ] **Step 5: Run the one-to-one mapping test**

Run the Task 2 test command. Expected: PASS.

- [ ] **Step 6: Commit the submission scripts**

```bash
git add \
  Uncertainty_Quantification/ConfidenceHead/run/submit_*.sh \
  Uncertainty_Quantification/ConfidenceHead/tests/test_ablation_configs.py
git commit -m "feat(confidence-head): add single-branch submit scripts"
```

### Task 4: Verify, Push, and Deploy Without Starting Training

**Files:**
- Verify all files from Tasks 1–3.
- Do not modify remote `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu.yaml`.
- Do not add remote `Uncertainty_Quantification/ConfidenceHead/run/submit.sh`.

**Interfaces:**
- Consumes: the completed local branch and the existing bywang SSH session on port `55801`.
- Produces: the same commit on GitHub and `/home/bywang/code/UQ/upet_new`, with nine parseable configs and nine shell-parseable scripts.

- [ ] **Step 1: Run focused ConfidenceHead tests**

```bash
conda run -n upet_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests
```

Expected: all tests pass.

- [ ] **Step 2: Run formatting, lint, and type checks**

```bash
conda run -n upet_new ruff format --check Uncertainty_Quantification/ConfidenceHead
conda run -n upet_new ruff check Uncertainty_Quantification/ConfidenceHead
conda run -n upet_new mypy Uncertainty_Quantification/ConfidenceHead/confidence_head
```

Expected: all commands exit 0.

- [ ] **Step 3: Confirm the local worktree and commit history**

```bash
git status --short
git log -4 --oneline
```

Expected: clean worktree; design, loss-contract, configuration, and submission-script commits are visible.

- [ ] **Step 4: Push the current `ConfidenceHead` branch to GitHub**

```bash
git push origin ConfidenceHead
```

Expected: GitHub advances from commit `9253a14`; no force push.

- [ ] **Step 5: Recheck remote user-owned changes before deployment**

```bash
ssh -p 55801 bywang@121.48.164.204 \
  'cd /home/bywang/code/UQ/upet_new && git status --short'
```

Expected: only `M .../configs/full_gpu.yaml` and `?? .../run/submit.sh` appear. If any other modified or untracked target file appears, stop deployment and report it.

- [ ] **Step 6: Fast-forward the remote branch through GitHub**

```bash
ssh -p 55801 bywang@121.48.164.204 \
  'cd /home/bywang/code/UQ/upet_new && git pull --ff-only origin ConfidenceHead'
```

Expected: fast-forward succeeds; the user's base YAML and untracked base submit script remain present.

- [ ] **Step 7: Parse all nine configs in the remote `upet_new` environment**

```bash
ssh -p 55801 bywang@121.48.164.204 \
  'cd /home/bywang/code/UQ/upet_new && /home/bywang/.conda/envs/upet_new/bin/python -m pytest -q Uncertainty_Quantification/ConfidenceHead/tests/test_ablation_configs.py'
```

Expected: both configuration and script-mapping tests pass.

- [ ] **Step 8: Verify remote input files and script syntax without launching jobs**

```bash
ssh -p 55801 bywang@121.48.164.204 \
  'cd /home/bywang/code/UQ/upet_new && sha256sum /home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt /home/bywang/code/UQ/mace/UQ_orb_post_train_force/data/matpes_{train,val,test}.extxyz && bash -n Uncertainty_Quantification/ConfidenceHead/run/submit_energy0.sh && for order in 1 2 3 4 5 6 7 8; do bash -n Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order${order}.sh; done'
```

Expected: hashes match Global Constraints and every script exits shell parsing with code 0. Do not call `sbatch` and do not run any of the four Python workflow stages.

- [ ] **Step 9: Report deployed commit, retained remote changes, filenames, and verification evidence**

Report the final commit hash, GitHub push result, remote HEAD, the two intentionally retained dirty paths, nine YAML names, nine script names, and all passing test/static-check counts.

## Self-Review

- Spec coverage: the plan covers the nine-run matrix, single-branch validation, server paths and hashes, naming/W&B behavior, output location, four-stage scripts, early stopping, no plotting, and no GPU execution.
- Placeholder scan: `CONFIG_BASENAME` is defined as a literal per-file substitution with all nine values enumerated; `N` is defined as each literal integer 1–8. There are no deferred implementation decisions.
- Type consistency: every test uses the existing `ConfidenceConfig`, `load_config`, and `build_run_name` interfaces; all loss coefficient fields remain floats and `cumulant_order` remains an integer 1–8.

## Final Execution Amendment: Keep Server Assets Out of the Release Branch

This amendment supersedes Tasks 2, 3, and 4 above. Execute Task 1 unchanged, then execute Tasks A and B below. Do not stage or commit any of the nine YAML files or nine Slurm scripts, and do not create a release test that requires those files to exist.

### Task A: Build and Validate Temporary Deployment Assets

**Files:**
- Temporarily create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_energy0.yaml`
- Temporarily create: `Uncertainty_Quantification/ConfidenceHead/configs/full_gpu_force0_order{1..8}.yaml`
- Temporarily create: `Uncertainty_Quantification/ConfidenceHead/run/submit_energy0.sh`
- Temporarily create: `Uncertainty_Quantification/ConfidenceHead/run/submit_force0_order{1..8}.sh`

**Interfaces:**
- Consumes: the server's approved `full_gpu.yaml` values, the exact paths and SHA-256 identities in Global Constraints, `load_config`, and `build_run_name`.
- Produces: eighteen locally validated but untracked deployment files.

- [ ] **Step A1: Create the nine complete YAML files without staging them**

Use complete copies of `full_gpu.yaml`. Set checkpoint/data/output paths and hashes to the Global Constraints values, `trainer.batch_size: 32`, and `run.name_prefix: upet_full_rtx5090`. Use this exact experiment matrix:

```text
full_gpu_energy0.yaml         force_coefficient=1.0 energy_coefficient=0.0 cumulant_order=3
full_gpu_force0_order1.yaml  force_coefficient=0.0 energy_coefficient=1.0 cumulant_order=1
full_gpu_force0_order2.yaml  force_coefficient=0.0 energy_coefficient=1.0 cumulant_order=2
full_gpu_force0_order3.yaml  force_coefficient=0.0 energy_coefficient=1.0 cumulant_order=3
full_gpu_force0_order4.yaml  force_coefficient=0.0 energy_coefficient=1.0 cumulant_order=4
full_gpu_force0_order5.yaml  force_coefficient=0.0 energy_coefficient=1.0 cumulant_order=5
full_gpu_force0_order6.yaml  force_coefficient=0.0 energy_coefficient=1.0 cumulant_order=6
full_gpu_force0_order7.yaml  force_coefficient=0.0 energy_coefficient=1.0 cumulant_order=7
full_gpu_force0_order8.yaml  force_coefficient=0.0 energy_coefficient=1.0 cumulant_order=8
```

- [ ] **Step A2: Create nine executable submission scripts without staging them**

Preserve the base Slurm resource directives. Every script activates `upet_new`, exports `WANDB_BASE_URL=https://api.bandw.top`, and invokes these four absolute script paths in order with its matching absolute config path:

```text
/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/scripts/build_cache.py
/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/scripts/train.py
/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/scripts/evaluate.py
/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/scripts/verify.py
```

The filename mapping is `submit_energy0.sh -> full_gpu_energy0.yaml` and `submit_force0_orderN.sh -> full_gpu_force0_orderN.yaml` for every literal integer N from 1 through 8. Do not add plotting or `sbatch` commands.

- [ ] **Step A3: Validate all temporary assets**

Run a Python check in `upet_new` that loads all nine YAMLs, asserts the matrix above, asserts `atom_mean`, `batch_size=32`, `val/total_loss_ema`, unique derived names, server paths, and W&B online mode. Run `bash -n` on all nine scripts and assert each contains its matching config path exactly four times in build-cache/train/evaluate/verify order.

Expected: nine unique valid configs and nine syntax-valid one-to-one scripts; `git status --short` shows only the eighteen untracked deployment files plus the intended release-code/document changes.

### Task B: Release Reusable Code, Deploy Assets, and Restore a Clean Local Branch

**Files:**
- Commit: `confidence_head/config.py`, its unit tests, and the revised design/plan documents.
- Deploy but do not commit: the eighteen temporary assets from Task A.

**Interfaces:**
- Consumes: Task 1's tested code and Task A's verified files.
- Produces: a clean local `ConfidenceHead` release branch and a bywang `ConfidenceHead` checkout containing the eighteen untracked server assets.

- [ ] **Step B1: Run release verification before commit**

```bash
conda run -n upet_new pytest -q Uncertainty_Quantification/ConfidenceHead/tests
conda run -n upet_new ruff format --check Uncertainty_Quantification/ConfidenceHead
conda run -n upet_new ruff check Uncertainty_Quantification/ConfidenceHead
conda run -n upet_new mypy Uncertainty_Quantification/ConfidenceHead/confidence_head
```

Expected: all commands exit 0. Confirm none of the eighteen deployment files is staged.

- [ ] **Step B2: Commit and push only reusable release changes**

Stage the revised design/plan documents if not already committed, commit them separately from Task 1 code, and push `ConfidenceHead` without force. Verify with `git diff --cached --name-only` before each commit that no `full_gpu_energy0`, `full_gpu_force0_order`, `submit_energy0`, or `submit_force0_order` path is present.

- [ ] **Step B3: Pull reusable code on bywang without touching user-owned files**

Before pulling, require remote status to contain no unexpected target paths. Fast-forward `/home/bywang/code/UQ/upet_new` from GitHub `ConfidenceHead`; preserve modified `configs/full_gpu.yaml` and untracked `run/submit.sh`.

- [ ] **Step B4: Copy the eighteen untracked assets to the bywang checkout**

Transfer the nine YAML files to `/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/configs/` and the nine executable scripts to `/home/bywang/code/UQ/upet_new/Uncertainty_Quantification/ConfidenceHead/run/`. Do not stage or commit them remotely.

- [ ] **Step B5: Verify the deployed files without launching GPU work**

Using `/home/bywang/.conda/envs/upet_new/bin/python`, load all nine remote configs, assert the exact experiment matrix and nine unique run names, and verify all checkpoint/data hashes. Run `bash -n` on every remote submit script. Do not run any workflow Python command and do not invoke `sbatch`.

- [ ] **Step B6: Remove only the eighteen local temporary assets**
