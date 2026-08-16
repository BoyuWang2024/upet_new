#!/usr/bin/env bash
#SBATCH --job-name=upet-external-uq
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/upet-external-%x-%j.out
#SBATCH --error=logs/upet-external-%x-%j.err

set -euo pipefail

ROOT=/home/bywang/code/UQ/upet_new
CONFIDENCE_HEAD_ROOT="$ROOT/Uncertainty_Quantification/ConfidenceHead"
CONFIG="$CONFIDENCE_HEAD_ROOT/configs/predict_external_gpu.yaml"
PYTHON=/home/bywang/.conda/envs/upet_new/bin/python

cd "$ROOT"
mkdir -p "$CONFIDENCE_HEAD_ROOT/run/logs"

case "${STAGE:-}" in
  mad-cache|mad-predict)
    "$PYTHON" "$CONFIDENCE_HEAD_ROOT/scripts/predict_external_datasets.py" \
      --config "$CONFIG" --dataset mad_test
    ;;
  matpes-train-predict)
    "$PYTHON" "$CONFIDENCE_HEAD_ROOT/scripts/predict_external_datasets.py" \
      --config "$CONFIG" --dataset matpes_train
    ;;
  plot)
    "$PYTHON" "$CONFIDENCE_HEAD_ROOT/scripts/plot_prediction_datasets.py" \
      --config "$CONFIG"
    ;;
  *)
    echo "STAGE must be mad-cache, mad-predict, matpes-train-predict, or plot" >&2
    exit 2
    ;;
esac
