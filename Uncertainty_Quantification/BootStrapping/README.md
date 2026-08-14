# BootStrapping

This directory provides a configuration-driven bootstrap ensemble workflow for
PET last-layer uncertainty quantification. Public artifacts use one canonical
checkpoint/prediction layout regardless of how they were produced.

The three supplied configurations preserve the established experiment settings.
Batch sizes are ordinary YAML values and can be changed; `64/64` is the default
for `full_remote_b8_e8`, while the learning-rate runs default to `16/8`.

Run UQ after predictions are present:

```bash
python -m Uncertainty_Quantification.BootStrapping.scripts.compute_uq \
  --config Uncertainty_Quantification/BootStrapping/configs/full_remote_b8_e8.yaml
```

Generated data belongs under `outputs/`, which is ignored by Git. The
`internal_migration/` directory is operational tooling and must be excluded from
any source release of this workflow.
