# UPET FGE Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 在当前 FGE 分支实现可发布、可测试的 UPET-native FGE 五阶段流程，将旧 8-member 结果只读迁移为 canonical schema，并在远端完成 n20、A3 和 full migration 验收。

**Architecture:** 正式 fge 包只识别 strict config、A3 member、canonical prediction/evaluation 和 immutable result contracts；五个 CLI 是薄适配层。internal_migration 单向依赖正式领域层，只负责旧格式读取、A3 打包、chunk 合并、审计和原子发布，正式代码永不依赖旧目录。

**Tech Stack:** Python 3.11、PyTorch、metatrain >=2026.3.1,<2026.4、metatomic-ase、PyYAML、pytest、tox、ruff、mypy。

## Global Constraints

- 直接在当前 FGE 分支实施；不得创建 worktree、不得修改 ConfidenceHead、LLPR、src/upet 或 BootStrapping。
- 本地和远端统一执行 conda activate upet_new；允许为本任务调整同名环境。
- 本地只开发和提交代码；所有可执行测试、n20、A3 等价和 full migration 均在远端 /home/bywang/code/UQ/upet_new 的 clean checkout/worktree 执行。
- 本计划中所有 python/pytest/tox 形式的 Run 命令都通过 SSH 在远端执行；只有 git status、git add、git commit、只读源码检查和 apply_patch 在当前本地分支执行。
- Python >=3.11；metatrain>=2026.3.1,<2026.4；不得按过期 AGENTS 描述降级。
- 正式 base checkpoint 为 /home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt，SHA256 879b1045391d88869522605a8b8b3cedeed74668e7062fdd7487548ab7b08004。
- 训练数据 SHA256 12ff9403254c955537827ba96c140ee1753a7410ada7910f13c42be0aa308cec；独立 test SHA256 1ffcdcad2fc6f0b0907b91cd29bfee340eb02cddf6b525268290c6329f56182d。
- 正式训练使用 restart model_state_dict、五项 checkpoint Huber loss、Adam weight_decay=0、float32、seed=2026、EMA=0.999 仅 validation。
- readout scope 必须精确为 node_last_layers.* 与 edge_last_layers.* 的 12 tensors、13,338 scalars；所有其他 parameters/buffers 与 base 精确相等。
- full FGE 固定 K=8、8 epochs/cycle、batch_size=16、drop_last=true、lr_min=1e-8、lr_max=1e-7、rise_fraction=0.2。
- A3 schema 固定 upet.fge.member-delta.v1；保存完整 endpoint replacement values，不保存算术差值；必须 weights_only=True、CPU 可加载。
- UQ 公式固定 legacy_upet_fge_v1：population STD、包含 i=j 的 ordered K^2 GMD、equal-weight only、无 stress UQ。
- 派生浮点等价使用 rtol=2e-6, atol=2e-7；chunk 重组后的 prediction/reference 元素必须精确相等。
- result_manifest.json 是唯一且最后写入的完成标记；completed result immutable，重复 validation 完全只读。
- full migration 禁止 train、predict、evaluate、forward、backward 和 optimizer；external audit 必须在 canonical rename 前成功写入。
- publication_files.txt 只允许 README.md、__init__.py、publication_files.txt、configs/、fge/、scripts/、tests/。
- W&B 历史、日志、plots、run_state、失败结果、train-derived prediction、旧 resume state、outputs、大 checkpoint 和大数据不得进入 Git。
- 每个生产行为先写失败测试并确认因缺少行为而 RED，再写最小实现至 GREEN；每个任务由 fresh implementer 提交并接受独立 spec/quality review。

---

### Task 1: Package foundation, strict configuration, and atomic artifacts

**Files:**
- Create: Uncertainty_Quantification/FGE/__init__.py
- Create: Uncertainty_Quantification/FGE/fge/__init__.py
- Create: Uncertainty_Quantification/FGE/fge/errors.py
- Create: Uncertainty_Quantification/FGE/fge/config.py
- Create: Uncertainty_Quantification/FGE/fge/artifacts.py
- Create: Uncertainty_Quantification/FGE/configs/upet_fge_full.yaml
- Create: Uncertainty_Quantification/FGE/configs/upet_fge_n20_cpu.yaml
- Create: Uncertainty_Quantification/FGE/outputs/.gitignore
- Create: Uncertainty_Quantification/FGE/tests/__init__.py
- Create: Uncertainty_Quantification/FGE/tests/conftest.py
- Create: Uncertainty_Quantification/FGE/tests/test_config.py
- Create: Uncertainty_Quantification/FGE/tests/test_artifacts.py
- Modify: .gitignore
- Modify: tox.ini
- Modify: pyproject.toml

**Interfaces:**
- Produces: HardFailure(message: str).
- Produces: load_config(path: str | Path) -> FGEConfig.
- Produces: FGEConfig.sanitized() -> dict[str, object] and FGEConfig.assert_stage(stage: str) -> None.
- Produces: sha256_file(path), atomic_write_json(path, payload), atomic_write_yaml(path, payload), atomic_torch_save(path, payload), normalize_artifact_path(root, path).
- Produces: ExperimentLayout(root: Path) with fixed preflight/training/prediction/evaluation/validation/result paths.
- Produces: sibling_staging(destination: Path) context used by migration publication.

- [ ] **Step 1: Add failing strict-config tests**

Write table-driven tests that load a literal valid config and assert unknown top-level/nested keys fail, relative paths resolve from the YAML directory, named UPET_FGE runtime environment variables resolve explicitly, fixed literals reject wrong types, full and n20 scientific flags match the design, and sanitized output contains SHA/logical roles but no runtime absolute paths.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_config.py -q
Expected: FAIL because Uncertainty_Quantification.FGE.fge.config does not exist.

- [ ] **Step 2: Implement strict dataclass configuration**

Implement frozen nested dataclasses for paths, targets, training, FGE schedule, prediction, evaluation, scientific state and runtime. Parse YAML with a recursive allowed-key check; expand only UPET_FGE_BASE_CHECKPOINT, UPET_FGE_TRAIN_DATA, UPET_FGE_VAL_DATA, UPET_FGE_TEST_DATA and UPET_FGE_OUTPUT_ROOT. Enforce exact global constants while permitting n20 overrides K=2, cycles=2, epochs_per_cycle=2, batch_size=4 and CPU.

Run the focused config test until GREEN.

- [ ] **Step 3: Add failing artifact and layout tests**

Test temp-file fsync/replace behavior, JSON rejection of NaN/Inf, torch round-trip, path escape rejection, fixed ExperimentLayout paths, sibling staging on the same filesystem, destination-exists refusal and preservation of an existing target when replace fails.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_artifacts.py -q
Expected: FAIL because artifact functions do not exist.

- [ ] **Step 4: Implement artifacts and package exports**

Implement same-directory temporary files with flush, os.fsync and os.replace; use torch.save only through atomic_torch_save; never overwrite a completed destination. Keep errors.py limited to HardFailure and make package exports explicit.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_config.py Uncertainty_Quantification/FGE/tests/test_artifacts.py -q
Expected: PASS with no warnings.

- [ ] **Step 5: Add test and ignore integration**

Add Uncertainty_Quantification/FGE/ to lint_folders; add testenv:fge-tests with pytest and pyyaml, running formal and internal migration test directories. Add fge_n20 and fge_legacy markers. Place precise negations after the root outputs/ rule so only Uncertainty_Quantification/FGE/outputs/.gitignore is trackable; that file ignores everything except itself.

Run: conda activate upet_new && tox -e fge-tests -- Uncertainty_Quantification/FGE/tests/test_config.py Uncertainty_Quantification/FGE/tests/test_artifacts.py
Expected: PASS.

- [ ] **Step 6: Commit Task 1**

Run: git add .gitignore tox.ini pyproject.toml Uncertainty_Quantification/FGE && git commit -m "feat: add FGE configuration and artifact foundations"

### Task 2: Legacy math, schedule, and evaluation contracts

**Files:**
- Create: Uncertainty_Quantification/FGE/fge/schedule.py
- Create: Uncertainty_Quantification/FGE/fge/uncertainty.py
- Create: Uncertainty_Quantification/FGE/fge/evaluation.py
- Create: Uncertainty_Quantification/FGE/tests/test_schedule.py
- Create: Uncertainty_Quantification/FGE/tests/test_uncertainty.py
- Create: Uncertainty_Quantification/FGE/tests/test_evaluation.py

**Interfaces:**
- Produces: asymmetric_triangular_lr(step: int, updates_per_cycle: int, lr_min: float, lr_max: float, rise_fraction: float) -> float.
- Produces: population_std(members, dim=0), scalar_gmd(members), vector_gmd(members), reduce_force_by_structure(values, offsets, quantile=0.95).
- Produces: evaluate_prediction(payload: Mapping[str, object], coverages: Sequence[float], constant_tolerance: float) -> EvaluationArtifacts.
- Produces: global_mae(prediction, reference) and stable risk/correlation diagnostics.

- [ ] **Step 1: Write failing schedule tests**

Use literal expected values for cycle start, the 0.2 peak boundary, descending midpoint and final update; test invalid update counts and LR bounds.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_schedule.py -q
Expected: FAIL because schedule.py does not exist.

- [ ] **Step 2: Implement the minimal asymmetric schedule**

Match the old update-index semantics exactly and ensure every cycle endpoint returns approximately lr_min without introducing a scheduler state object.

Run the schedule test until GREEN.

- [ ] **Step 3: Write failing legacy uncertainty tests**

Use hand-derived K=2 and K=3 tensors to prove unbiased=False population STD, ordered K^2 scalar/vector GMD including self pairs, force component/vector and structure mean/max/q95 reductions, and rejection of K mismatch/nonfinite/misaligned offsets. Include a regression whose expected value differs from sample STD and unordered-pair GMD.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_uncertainty.py -q
Expected: FAIL because uncertainty.py does not exist.

- [ ] **Step 4: Implement legacy_upet_fge_v1 math**

Use torch operations on CPU tensors, preserve dtype, do not add weights, and expose FORMULA_VERSION = "legacy_upet_fge_v1".

Run the uncertainty test until GREEN.

- [ ] **Step 5: Write failing evaluation tests and implement**

Tests independently derive total/per-atom energy MAE, force component MAE, stress 3x3 component MAE, constant-input warning, stable tied ranks and the exact coverage list. Implement evaluation without loading a model or dataset and return structured ensemble, uncertainty, metrics and report inputs.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_evaluation.py Uncertainty_Quantification/FGE/tests/test_schedule.py Uncertainty_Quantification/FGE/tests/test_uncertainty.py -q
Expected: PASS with no warnings.

- [ ] **Step 6: Commit Task 2**

Run: git add Uncertainty_Quantification/FGE/fge Uncertainty_Quantification/FGE/tests && git commit -m "feat: implement legacy FGE uncertainty semantics"

### Task 3: UPET checkpoint, readout scope, and A3 members

**Files:**
- Create: Uncertainty_Quantification/FGE/fge/checkpoint.py
- Create: Uncertainty_Quantification/FGE/fge/members.py
- Create: Uncertainty_Quantification/FGE/tests/test_checkpoint.py
- Create: Uncertainty_Quantification/FGE/tests/test_members.py

**Interfaces:**
- Produces: load_checkpoint_bundle(path: Path) -> CheckpointBundle using restart model_state_dict.
- Produces: recover_loss_contract(raw_checkpoint) -> five-term immutable LossContract.
- Produces: readout_tensor_names(model) -> tuple[str, ...], assert_readout_contract(model) -> ReadoutAudit.
- Produces: frozen_fingerprint(model) -> tuple[TensorFingerprint, ...] and assert_frozen_unchanged.
- Produces: pack_member(model, member_id, cycle, global_step, base_sha256) -> dict.
- Produces: load_member(path, base_sha256, expected_names) -> MemberPayload and apply_member(model, base_state, member) -> None.

- [ ] **Step 1: Add failing checkpoint-selection and loss tests**

Build a tiny literal checkpoint containing different model_state_dict and best_model_state_dict values; assert restart wins and missing restart fails. Build five literal Huber entries for energy, forces, virial, non_conservative_forces and non_conservative_stress and assert sliding_factor=None and gradient clipping fields survive.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_checkpoint.py -q
Expected: FAIL because checkpoint.py does not exist.

- [ ] **Step 2: Implement checkpoint parsing**

Adapt behavior, not artifact layout, from /home/lilong/code/UQ/upet/Ensemble/src/FGE/checkpoint.py and loss_config.py to metatrain 2026.3.1. Keep raw checkpoint loading isolated and fail with HardFailure on missing/malformed contract.

Run the checkpoint test until GREEN.

- [ ] **Step 3: Add failing readout and A3 tests**

Use a TinyPET module with node_last_layers and edge_last_layers plus frozen parameters/buffers. Assert deterministic 12-name ordering, count enforcement, frozen drift detection, nonfinite rejection, replacement-not-delta semantics, CPU tensors, weights_only torch.load, strict base/name/dtype/shape checks and restoration of base before switching members.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_members.py -q
Expected: FAIL because members.py does not exist.

- [ ] **Step 4: Implement readout and A3 contracts**

Use schema_version "upet.fge.member-delta.v1"; serialize only primitive metadata and ordered tensor mappings. Verify base SHA before loading a member, clone the base state once on CPU, restore it before every replacement, and require exact equality for all non-readout parameters and buffers.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_checkpoint.py Uncertainty_Quantification/FGE/tests/test_members.py -q
Expected: PASS with no warnings.

- [ ] **Step 5: Commit Task 3**

Run: git add Uncertainty_Quantification/FGE/fge/checkpoint.py Uncertainty_Quantification/FGE/fge/members.py Uncertainty_Quantification/FGE/tests/test_checkpoint.py Uncertainty_Quantification/FGE/tests/test_members.py && git commit -m "feat: add UPET A3 member contract"

### Task 4: Canonical data, prediction, and manifest schemas

**Files:**
- Create: Uncertainty_Quantification/FGE/fge/data.py
- Create: Uncertainty_Quantification/FGE/fge/prediction.py
- Create: Uncertainty_Quantification/FGE/fge/manifests.py
- Create: Uncertainty_Quantification/FGE/tests/test_data.py
- Create: Uncertainty_Quantification/FGE/tests/test_prediction.py
- Create: Uncertainty_Quantification/FGE/tests/test_manifests.py

**Interfaces:**
- Produces: DatasetIdentity, PredictionShape(K, S, A), canonical_prediction(payload) and validate_prediction_payload(payload).
- Produces: predict_members(config: FGEConfig) -> Path, with a dependency-injected batch inference seam for unit tests.
- Produces: build_training_manifest, build_prediction_manifest and build_result_manifest; all reject nonfinite JSON and absolute formal paths.
- Consumes: ExperimentLayout, load_checkpoint_bundle, load_member/apply_member, atomic writers.

- [ ] **Step 1: Write failing data and payload tests**

Construct two literal structures with three total atoms. Assert energy [K,S], force [K,A,3], stress [K,S,3,3], one copy of references, exact member/structure order, atomic numbers, n_atoms, offsets and target/unit metadata. Mutate each key, dtype, shape, order and mapping to prove hard failure.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_data.py Uncertainty_Quantification/FGE/tests/test_prediction.py -q
Expected: FAIL because data.py and prediction.py do not exist.

- [ ] **Step 2: Implement canonical payload validation**

Keep extxyz/metatrain loading behind data.py; prediction.py only stacks validated batch outputs and atom mappings. Never skip a member or shrink K. Expose a fake runtime seam that exercises real stacking and validation without a real checkpoint.

Run the data and prediction tests until GREEN.

- [ ] **Step 3: Write failing manifest tests**

Assert native and migrated manifests have identical key trees; training_code_identity is present for native and status=unavailable for migrated; artifact_writer_code_identity and validator_code_identity are mandatory; member hashes/order/steps are stable; result manifest hashes every formal artifact and excludes itself until final write.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_manifests.py -q
Expected: FAIL until manifests.py implements the contract.

- [ ] **Step 4: Implement manifests and prediction stage**

Build JSON-safe identities from git commit/dirty digest and dependency versions. Ensure config_resolved contains logical roles and content hashes only. Implement base-once, restore-base, apply-A3, infer E/F/stress member order and atomic prediction publication.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_data.py Uncertainty_Quantification/FGE/tests/test_prediction.py Uncertainty_Quantification/FGE/tests/test_manifests.py -q
Expected: PASS with no warnings.

- [ ] **Step 5: Commit Task 4**

Run: git add Uncertainty_Quantification/FGE/fge Uncertainty_Quantification/FGE/tests && git commit -m "feat: define canonical FGE prediction schema"

### Task 5: Preflight, validation, and immutable completion

**Files:**
- Create: Uncertainty_Quantification/FGE/fge/preflight.py
- Create: Uncertainty_Quantification/FGE/fge/validation.py
- Create: Uncertainty_Quantification/FGE/tests/test_preflight.py
- Create: Uncertainty_Quantification/FGE/tests/test_validation.py
- Create: Uncertainty_Quantification/FGE/tests/test_failure_policy.py
- Create: Uncertainty_Quantification/FGE/tests/test_completed_read_only.py
- Create: Uncertainty_Quantification/FGE/tests/test_schema_signature.py

**Interfaces:**
- Produces: run_preflight(config, stage, basis="runtime_inputs") -> dict.
- Produces: validate_result(config, root, *, publish_completion=True) -> ValidationReport.
- Produces: schema_signature(root) -> dict with symbolic K/S/A dimensions.
- Consumes: all artifacts/manifests/prediction/evaluation contracts from Tasks 1-4.

- [ ] **Step 1: Write failing preflight tests**

Assert stages are exactly train/predict/evaluate; preflight performs no model forward/backward/optimizer; runtime basis checks hashes, targets, units, 12/13,338 scope, K/device/dtype/disk/output path and scientific flags; canonical basis uses on-disk canonical artifacts and contains no migration/source markers.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_preflight.py -q
Expected: FAIL because preflight.py does not exist.

- [ ] **Step 2: Implement preflight**

Write sanitized config atomically only for initial native train preflight; write three same-schema reports with stage/status/basis/identity/scientific flags. Predict/evaluate preflights validate prior artifact identities without rewriting completed roots.

Run the preflight test until GREEN.

- [ ] **Step 3: Write failing validator and failure-policy tests**

Create a complete tiny canonical tree and independently corrupt hashes, member order, tensor dtype/shape, mapping, references, formulas, forbidden paths and nonfinite values. Assert first validation writes validation.json then result_manifest.json last; second validation leaves every mtime and byte unchanged. Assert normalized n20 K=2 and migrated K=8 signatures are identical.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_validation.py Uncertainty_Quantification/FGE/tests/test_failure_policy.py Uncertainty_Quantification/FGE/tests/test_completed_read_only.py Uncertainty_Quantification/FGE/tests/test_schema_signature.py -q
Expected: FAIL because validation.py does not exist.

- [ ] **Step 4: Implement disk-reopening validation**

Reload every artifact from disk, recompute means/UQ/MAE without replacing stored migrated values, compare with exact or rtol=2e-6,atol=2e-7 rules, write validation first and result manifest last, and enter pure read-only mode when a valid result_manifest already exists.

Run the four validation tests until GREEN.

- [ ] **Step 5: Commit Task 5**

Run: git add Uncertainty_Quantification/FGE/fge/preflight.py Uncertainty_Quantification/FGE/fge/validation.py Uncertainty_Quantification/FGE/tests && git commit -m "feat: validate immutable FGE results"

### Task 6: Native UPET training runtime and five independent CLIs

**Files:**
- Create: Uncertainty_Quantification/FGE/fge/training.py
- Modify: Uncertainty_Quantification/FGE/fge/prediction.py
- Modify: Uncertainty_Quantification/FGE/fge/evaluation.py
- Create: Uncertainty_Quantification/FGE/scripts/__init__.py
- Create: Uncertainty_Quantification/FGE/scripts/preflight.py
- Create: Uncertainty_Quantification/FGE/scripts/train.py
- Create: Uncertainty_Quantification/FGE/scripts/predict.py
- Create: Uncertainty_Quantification/FGE/scripts/evaluate.py
- Create: Uncertainty_Quantification/FGE/scripts/validate.py
- Create: Uncertainty_Quantification/FGE/tests/test_training.py
- Create: Uncertainty_Quantification/FGE/tests/test_scripts.py

**Interfaces:**
- Produces: train_fge(config: FGEConfig) -> Path to training/manifest.json.
- Produces: evaluate_fge(config: FGEConfig) -> Path to evaluation/legacy_equal_weight.
- Each CLI exposes build_parser() and main(argv: Sequence[str] | None = None) -> int.
- Consumes old UPET official_pet_trainer.py, trainer.py, data.py and ema.py only as behavioral oracles; emits only new canonical artifacts.

- [ ] **Step 1: Write failing training orchestration tests**

Use a fake PET runtime with real torch parameters to assert one Adam optimizer and readout-only EMA span all cycles; LR updates per optimizer step; drop_last update counts are exact; five loss terms participate; frozen checks occur per epoch/endpoint; only finite, reloadable endpoint A3 members enter manifest; resume identity mismatch fails.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_training.py -q
Expected: FAIL because training.py does not exist.

- [ ] **Step 2: Implement minimal orchestration core**

Separate TrainingRuntime protocol from orchestration. Implement cycle/epoch/update loops, gradient clipping, EMA validation-only behavior, ignored _work resume state, atomic member acceptance and training manifest publication. W&B is optional native-only warning behavior and defaults disabled.

Run the training test until GREEN.

- [ ] **Step 3: Integrate metatrain 2026.3.1 PET runtime**

Load restart state, restore five-term loss, build train loader with random rotation/neighbor list/additive removal/scale removal and validation loader without random rotation. Freeze exact readout scope and preserve composition/scaler. Do not copy old artifact/run/logging layout.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_training.py Uncertainty_Quantification/FGE/tests/test_checkpoint.py Uncertainty_Quantification/FGE/tests/test_members.py -q
Expected: PASS with no warnings.

- [ ] **Step 4: Write failing CLI tests and implement thin scripts**

Assert every script requires --config, preflight additionally requires --stage train|predict|evaluate, and no run-all/uq/plot CLI exists. Scripts load one YAML, call one formal stage and return 0; HardFailure becomes a concise nonzero exit.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_scripts.py -q
Expected: PASS after implementing all five CLIs.

- [ ] **Step 5: Run formal unit suite and commit**

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests -q -m "not fge_n20 and not fge_legacy"
Expected: PASS with no warnings.

Run: git add Uncertainty_Quantification/FGE/fge/training.py Uncertainty_Quantification/FGE/fge/prediction.py Uncertainty_Quantification/FGE/fge/evaluation.py Uncertainty_Quantification/FGE/scripts Uncertainty_Quantification/FGE/tests && git commit -m "feat: add native UPET FGE stages"

### Task 7: Read-only legacy reader and atomic converter

**Files:**
- Create: Uncertainty_Quantification/FGE/internal_migration/README.md
- Create: Uncertainty_Quantification/FGE/internal_migration/__init__.py
- Create: Uncertainty_Quantification/FGE/internal_migration/migration/__init__.py
- Create: Uncertainty_Quantification/FGE/internal_migration/migration/legacy_reader.py
- Create: Uncertainty_Quantification/FGE/internal_migration/migration/converter.py
- Create: Uncertainty_Quantification/FGE/internal_migration/scripts/inspect_legacy.py
- Create: Uncertainty_Quantification/FGE/internal_migration/scripts/audit_results.py
- Create: Uncertainty_Quantification/FGE/internal_migration/scripts/migrate_results.py
- Create: Uncertainty_Quantification/FGE/internal_migration/scripts/validate_migration.py
- Create: Uncertainty_Quantification/FGE/internal_migration/tests/__init__.py
- Create: Uncertainty_Quantification/FGE/internal_migration/tests/conftest.py
- Create: Uncertainty_Quantification/FGE/internal_migration/tests/test_legacy_reader.py
- Create: Uncertainty_Quantification/FGE/internal_migration/tests/test_converter.py
- Create: Uncertainty_Quantification/FGE/internal_migration/tests/test_audit.py
- Create: Uncertainty_Quantification/FGE/internal_migration/tests/test_no_compute_guard.py

**Interfaces:**
- Produces: read_legacy_run(root: Path, expected: LegacyExpectations) -> LegacyRun.
- Produces: convert_legacy_run(source, destination, audit_root, config, base_checkpoint) -> Path.
- Produces: verify_source_unchanged(snapshot) and write_external_audit(...).
- Consumes formal artifact writers, A3 packer and validator; formal fge never imports internal_migration.

- [ ] **Step 1: Write failing legacy-reader tests**

Build synthetic 2-member/3-chunk old tree with literal hashes and references. Assert numeric member/chunk ordering, accepted K, continuous chunk coverage, exact prediction/reference/mapping merge, old metrics/UQ preservation and rejection of crash, hash, missing chunk, reference or source path escape defects.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/internal_migration/tests/test_legacy_reader.py -q
Expected: FAIL because legacy_reader.py does not exist.

- [ ] **Step 2: Implement streaming legacy reader**

Read only successful old layout identified by explicit expectations; do not scan for alternatives. Hash before reading, stream chunks in numeric order, retain source paths/hashes only in LegacyRun audit fields and never put them in canonical payloads.

Run the reader test until GREEN.

- [ ] **Step 3: Write failing converter, audit and no-compute tests**

Use synthetic full checkpoints whose only differences are 12 readout tensors. Assert non-readout exact equality, A3 extraction, canonical staging, independent validation, source hashes before/after, audit-before-rename, publication_authorized=true, destination refusal and fault injection at each write/rename boundary. Monkeypatch Module.forward, Tensor.backward, torch.optim.Optimizer.step, train_fge, predict_members and evaluate_fge to raise if called.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/internal_migration/tests/test_converter.py Uncertainty_Quantification/FGE/internal_migration/tests/test_audit.py Uncertainty_Quantification/FGE/internal_migration/tests/test_no_compute_guard.py -q
Expected: FAIL because converter.py does not exist.

- [ ] **Step 4: Implement converter and audit CLIs**

Write canonical artifacts into same-filesystem sibling staging, invoke formal validation on disk, snapshot source hashes again, atomically write external audit under outputs/_internal_migration, then rename staging to destination. On failure leave no completed formal root. Scripts accept explicit source/destination/audit/config/base arguments.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/internal_migration/tests -q
Expected: PASS with no warnings.

- [ ] **Step 5: Commit Task 7**

Run: git add Uncertainty_Quantification/FGE/internal_migration && git commit -m "feat: migrate legacy FGE artifacts atomically"

### Task 8: Publication contract, documentation, and repository integration

**Files:**
- Create: Uncertainty_Quantification/FGE/README.md
- Create: Uncertainty_Quantification/FGE/publication_files.txt
- Create: Uncertainty_Quantification/FGE/tests/test_publication_boundary.py
- Modify: Uncertainty_Quantification/FGE/configs/upet_fge_full.yaml
- Modify: Uncertainty_Quantification/FGE/configs/upet_fge_n20_cpu.yaml
- Modify: Uncertainty_Quantification/FGE/fge/__init__.py
- Modify: Uncertainty_Quantification/FGE/__init__.py

**Interfaces:**
- Produces documented commands for five stages and isolated migration tools.
- Produces executable publication allowlist and formal-to-internal dependency checks.
- Produces full config driven by the five UPET_FGE runtime variables and n20 config with K=2, 2 cycles, 2 epochs/cycle, batch_size=4, drop_last=true, W&B disabled.

- [ ] **Step 1: Write failing publication-boundary test**

Walk publication_files.txt and assert the allowlist is exact, every formal Python file lacks internal_migration imports, formal YAML/JSON/text contains no old absolute paths/source member SHAs/migration markers/W&B/log/plot/BootStrapping/run_state, and Git tracks no runtime content below outputs except .gitignore.

Run: conda activate upet_new && python -m pytest Uncertainty_Quantification/FGE/tests/test_publication_boundary.py -q
Expected: FAIL because README/publication allowlist are incomplete.

- [ ] **Step 2: Finalize formal configs and README**

Document environment activation, runtime variables, scientific limitations, five independent commands, result schema, immutable validation, warning semantics and internal migration separation. State n20 is path-only and full historical train=validation metrics are not generalization evidence.

Run the publication test until GREEN.

- [ ] **Step 3: Run repository test gates**

Run: conda activate upet_new && tox -e fge-tests
Expected: PASS.

Run: conda activate upet_new && tox -e lint
Expected: PASS.

Run: conda activate upet_new && tox -e upet-tests
Expected: PASS.

- [ ] **Step 4: Inspect tracked files and commit**

Run: git status --short && git ls-files Uncertainty_Quantification/FGE
Expected: no .pt/.ckpt, runtime output, W&B, log, plot, failure or BootStrapping artifact.

Run: git add Uncertainty_Quantification/FGE tox.ini pyproject.toml .gitignore && git commit -m "docs: publish UPET FGE workflow"

### Task 9: Remote n20, eight-member A3, and full migration acceptance

**Files:**
- Runtime only, ignored: /home/bywang/code/UQ/upet_new/Uncertainty_Quantification/FGE/outputs/
- Runtime only, ignored: /home/bywang/code/UQ/upet_new/.superpowers/sdd/
- No tracked file is created solely to record large-result evidence.

**Interfaces:**
- Consumes committed branch and formal/internal CLIs.
- Produces completed native n20 canonical root, 8/8 A3 equivalence report, completed full migrated canonical root and external audit, all ignored.
- Produces fresh command evidence in the task report and SDD ledger.

- [ ] **Step 1: Prepare remote clean checkout and environment**

Connect with ssh bywang@121.48.164.204. In /home/bywang/code/UQ/upet_new verify FGE branch/commit and clean tracked worktree, then run:

~~~bash
conda activate upet_new
python -m pip install --no-build-isolation -e .
tox -e fge-tests
tox -e lint
tox -e upet-tests
~~~

All commands exit 0 before scientific acceptance.

- [ ] **Step 2: Resolve and hash runtime inputs**

Hash base checkpoint, train data, independent test and n20 file. Require three formal hashes from Global Constraints and exactly 20 n20 structures. Export five UPET_FGE runtime variables without writing absolute paths into tracked config.

- [ ] **Step 3: Execute native n20 five-stage flow**

Run exactly:

~~~bash
python Uncertainty_Quantification/FGE/scripts/preflight.py --config Uncertainty_Quantification/FGE/configs/upet_fge_n20_cpu.yaml --stage train
python Uncertainty_Quantification/FGE/scripts/train.py --config Uncertainty_Quantification/FGE/configs/upet_fge_n20_cpu.yaml
python Uncertainty_Quantification/FGE/scripts/preflight.py --config Uncertainty_Quantification/FGE/configs/upet_fge_n20_cpu.yaml --stage predict
python Uncertainty_Quantification/FGE/scripts/predict.py --config Uncertainty_Quantification/FGE/configs/upet_fge_n20_cpu.yaml
python Uncertainty_Quantification/FGE/scripts/preflight.py --config Uncertainty_Quantification/FGE/configs/upet_fge_n20_cpu.yaml --stage evaluate
python Uncertainty_Quantification/FGE/scripts/evaluate.py --config Uncertainty_Quantification/FGE/configs/upet_fge_n20_cpu.yaml
python Uncertainty_Quantification/FGE/scripts/validate.py --config Uncertainty_Quantification/FGE/configs/upet_fge_n20_cpu.yaml
~~~

Verify K=2, 20 updates, complete manifest, W&B disabled and non-scientific flags.

- [ ] **Step 4: Run eight-member A3 equivalence on CPU**

Against /home/bywang/code/UQ/upet/Ensemble/results/FGE/upet-FGE-CKPT-UQ-v1.0, verify all eight source member SHAs from design, exact non-readout state, then compare old full versus base+A3 E/F/stress forward on same n20 input at rtol=2e-6,atol=2e-7. Report max absolute/relative difference for each member and target; require 8/8 PASS.

- [ ] **Step 5: Execute and validate full migration**

Run these explicit commands after exporting SOURCE=/home/bywang/code/UQ/upet/Ensemble/results/FGE/upet-FGE-CKPT-UQ-v1.0, DESTINATION under ignored FGE/outputs, AUDIT_ROOT under ignored FGE/outputs/_internal_migration, FULL_CONFIG=Uncertainty_Quantification/FGE/configs/upet_fge_full.yaml and BASE_CHECKPOINT=/home/bywang/code/UQ/upet/pet-omatpes-l-v0.1.0.ckpt:

~~~bash
python Uncertainty_Quantification/FGE/internal_migration/scripts/inspect_legacy.py --source "$SOURCE" --config "$FULL_CONFIG"
python Uncertainty_Quantification/FGE/internal_migration/scripts/migrate_results.py --source "$SOURCE" --destination "$DESTINATION" --audit-root "$AUDIT_ROOT" --config "$FULL_CONFIG" --base-checkpoint "$BASE_CHECKPOINT"
python Uncertainty_Quantification/FGE/internal_migration/scripts/validate_migration.py --destination "$DESTINATION" --config "$FULL_CONFIG"
python Uncertainty_Quantification/FGE/internal_migration/scripts/audit_results.py --destination "$DESTINATION" --audit-root "$AUDIT_ROOT"
~~~

Require 969 chunks/member, 7,752 chunks total, K=8, independent test S=19,374/A=149,321, exact canonical prediction elements, preserved old UQ/metrics, exact MAEs 0.01294630383014235 / 0.08171965440375559 / 0.00215389879550793, equal normalized schema signature and unchanged source hashes.

- [ ] **Step 6: Verify publication cleanliness and commit fixes if required**

Run git status --short and git ls-files checks locally and remotely. Runtime outputs remain ignored and tracked worktrees clean. If acceptance exposes a defect, return to RED with a focused regression test, fix through task review loop and commit; rerun every affected acceptance command. If no tracked fix is needed, record clean acceptance in SDD report/ledger without creating an evidence-only Git commit.

