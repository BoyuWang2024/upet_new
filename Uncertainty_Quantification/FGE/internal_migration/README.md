# Internal FGE migration tools

This directory is an isolated, one-time importer for the authenticated legacy
UPET FGE result. It is intentionally not part of the formal `fge` dependency
graph. The formal package and its five native stage commands must never import
this directory.

The reader accepts only the fixed successful run, eight authenticated member
checkpoints, 969 contiguous independent-test chunks per member, the preserved
legacy uncertainty values, metric JSON, and risk-coverage CSV files. It rejects
failed members, path escapes, symlinks, hash drift, missing chunks, reordered
members, and reference/mapping disagreement.

`migrate_results` writes a complete canonical result into same-filesystem sibling
staging. It converts only the 12 readout tensors to A3 members, verifies all
non-readout state exactly, invokes the disk-backed formal validator, rechecks
every critical source hash, writes the external audit, and only then renames the
staging directory to the final destination. The converter never trains or runs
model prediction/evaluation entry points.

All commands require explicit paths. Use `inspect_legacy` before migration,
`migrate_results` for the atomic conversion, `validate_migration` for canonical
read-only validation, and `audit_results` to recheck the durable external audit.
Large checkpoints, chunks, outputs, logs, W&B files, plots, and failure artifacts
remain remote and are never committed.
