#!/usr/bin/env bash
set -euo pipefail

# Bounded π0.5 validation run for one full 80 GB H100.
# The current 40 GB MIG is too small: π0.5 needs about 50 GiB at batch size 1.

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENPI_DIR="${OPENPI_DIR:-$CODE_ROOT/openpi}"
NORM_FILE="$OPENPI_DIR/outputs/assets/pi05_b1k/behavior-1k/2026-challenge-demos/norm_stats.json"

if [[ ! -f "$NORM_FILE" ]]; then
    echo "Missing persisted normalization statistics: $NORM_FILE" >&2
    echo "Compute them before starting the timed run." >&2
    exit 1
fi

export NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_DEVICES="${CUDA_DEVICES:-0}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export TRAIN_STEPS="${TRAIN_STEPS:-1000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_STEPS}"
export NUM_WORKERS="${NUM_WORKERS:-4}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export EXP_NAME="${EXP_NAME:-pi05_b1k_90min_$(date +%Y%m%d_%H%M%S)}"

echo "Starting bounded π0.5 run: $EXP_NAME"
echo "Expected hardware: one full 80 GB H100 (not a 40 GB MIG instance)"
echo "steps=$TRAIN_STEPS batch_size=$BATCH_SIZE devices=$CUDA_DEVICES"

exec "$CODE_ROOT/scripts/train_pi05_b1k.sh" full
