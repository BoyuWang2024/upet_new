#!/usr/bin/env bash
#SBATCH --job-name=upet-e0-postprocess
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/upet-e0-%x-%j.out
#SBATCH --error=logs/upet-e0-%x-%j.err

set -euo pipefail

ROOT=/home/bywang/code/UQ/upet_new
CONFIDENCE_HEAD_ROOT="$ROOT/Uncertainty_Quantification/ConfidenceHead"
CONFIG="$CONFIDENCE_HEAD_ROOT/configs/e0_postprocessing_gpu.yaml"
PYTHON=/home/bywang/.conda/envs/upet_new/bin/python
STAGE="${STAGE:-all}"

case "$STAGE" in
  predict|postprocess|plot|all) ;;
  *)
    echo "STAGE must be predict, postprocess, plot, or all" >&2
    exit 2
    ;;
esac

cd "$ROOT"
mkdir -p "$CONFIDENCE_HEAD_ROOT/run/logs"
"$PYTHON" "$CONFIDENCE_HEAD_ROOT/scripts/postprocess_r2scan_e0.py" \
  --config "$CONFIG" --stage "$STAGE"
