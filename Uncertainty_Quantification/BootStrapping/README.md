# BootStrapping

This directory provides a configuration-driven bootstrap ensemble workflow for
PET last-layer uncertainty quantification. Public artifacts use one canonical
checkpoint/prediction layout regardless of how they were produced.

The three supplied configurations preserve the established experiment settings.
Batch sizes are ordinary YAML values and can be changed; `64/64` is the default
for `full_remote_b8_e8`, while the learning-rate runs default to `16/8`.

Train the configured replacement-bootstrap members, then produce canonical raw
predictions and recompute UQ:

```bash
CONFIG=Uncertainty_Quantification/BootStrapping/configs/full_remote_b8_e8.yaml
RUN=Uncertainty_Quantification/BootStrapping/outputs/upet-bootstrap-head-posttrain-v1/full_remote_b8_e8

python -m Uncertainty_Quantification.BootStrapping.scripts.train \
  --config "$CONFIG" --run-root "$RUN"
python -m Uncertainty_Quantification.BootStrapping.scripts.predict \
  --config "$CONFIG" --run-root "$RUN"
python -m Uncertainty_Quantification.BootStrapping.scripts.compute_store_uq \
  --config "$CONFIG" --run-root "$RUN"
```

Training reconstructs the PET restart model from the configured base checkpoint,
freezes every parameter except the audited node/edge last layers (13,338
parameters), and uses the checkpoint-defined five-term Huber loss contract.
Every member records its deterministic sampling/loader/Python/Torch seeds and
replacement-bootstrap indices. `best.pt` and `final.pt` are inference artifacts;
`latest.pt` additionally contains optimizer and Python/NumPy/Torch RNG states.

Formal source archives are produced with `scripts.build_release`. The release
allowlist contains only the public package, configuration, documentation, and
workflow commands; tests, generated outputs, caches, and operational migration
tooling are excluded.

Generated data belongs under `outputs/`, which is ignored by Git. The
`internal_migration/` directory is operational tooling and is excluded from
formal source releases.
