#!/usr/bin/env bash
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-/mnt/afs/visitor34/code}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
EXP_NAME="${EXP_NAME:-gr00t_b1k_convergence_${RUN_ID}}"
RESULT_ROOT="${RESULT_ROOT:-$CODE_ROOT/experiments/groot_convergence/$EXP_NAME}"
TRAIN_OUTPUT_ROOT="${TRAIN_OUTPUT_ROOT:-$CODE_ROOT/Isaac-GR00T/outputs}"
MAX_STEPS="${MAX_STEPS:-200}"

mkdir -p "$RESULT_ROOT"

export EXP_NAME
export OUTPUT_DIR="$TRAIN_OUTPUT_ROOT"
export MAX_STEPS
export SAVE_STEPS="${SAVE_STEPS:-$MAX_STEPS}"
export SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-true}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-1}"
export NUM_GPUS="${NUM_GPUS:-1}"
export CUDA_DEVICES="${CUDA_DEVICES:-0}"
export NUM_WORKERS="${NUM_WORKERS:-0}"
export EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.01}"
export NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-1}"
export LEARNING_RATE="${LEARNING_RATE:-1e-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

cat >"$RESULT_ROOT/config.txt" <<EOF
experiment_name=$EXP_NAME
max_steps=$MAX_STEPS
global_batch_size=$GLOBAL_BATCH_SIZE
num_gpus=$NUM_GPUS
cuda_devices=$CUDA_DEVICES
episode_sampling_rate=$EPISODE_SAMPLING_RATE
num_shards_per_epoch=$NUM_SHARDS_PER_EPOCH
learning_rate=$LEARNING_RATE
gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS
save_only_model=$SAVE_ONLY_MODEL
EOF

echo "Convergence experiment: $EXP_NAME"
echo "Metrics and log directory: $RESULT_ROOT"
echo "Training output directory: $TRAIN_OUTPUT_ROOT/$EXP_NAME"

"$CODE_ROOT/scripts/train_gr00t_b1k.sh" smoke 2>&1 | tee "$RESULT_ROOT/train.log"

"$CODE_ROOT/Isaac-GR00T/.venv/bin/python" \
    "$CODE_ROOT/scripts/summarize_groot_convergence.py" \
    --run-dir "$TRAIN_OUTPUT_ROOT/$EXP_NAME" \
    --output-dir "$RESULT_ROOT"

echo "Convergence summary: $RESULT_ROOT/summary.txt"
