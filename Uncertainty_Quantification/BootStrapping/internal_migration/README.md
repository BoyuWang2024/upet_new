# Internal BootStrapping result adapter

This directory contains one-time operational tooling. It is excluded from release
artifacts and is not imported by the public BootStrapping workflow. The adapter
copies existing checkpoints byte-for-byte and normalizes existing prediction
chunks; it must never train a model, run model inference, or compute uncertainty.
