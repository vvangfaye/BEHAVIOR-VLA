#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENPI_DIR="${OPENPI_DIR:-$CODE_ROOT/openpi}"
DATASET_ROOT="${DATASET_ROOT:-/data/2026-challenge-demos}"
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
EXP_NAME="${EXP_NAME:-pi05_b1k_${MODE}_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
# Optional (default off = unchanged behavior):
#   SUBTASK_PROMPTS=<segment parquet>  -> condition on the current sub-task
#   FRAME_STRIDE=<int>                 -> subsample training frames by this stride
SUBTASK_PROMPTS="${SUBTASK_PROMPTS:-}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"

if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
    echo "Dataset not found or incomplete: $DATASET_ROOT/meta/info.json" >&2
    exit 1
fi
if [[ ! -x "$OPENPI_DIR/.venv/bin/python" ]]; then
    echo "OpenPI environment not found: $OPENPI_DIR/.venv" >&2
    exit 1
fi

case "$MODE" in
    smoke)
        REPO_ID="${REPO_ID:-behavior-1k/2026-challenge-demos-smoke}"
        EPISODES_INDEX=(0 1)
        TRAIN_STEPS="${TRAIN_STEPS:-2}"
        BATCH_SIZE="${BATCH_SIZE:-$NUM_GPUS}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-$TRAIN_STEPS}"
        NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-256}"
        ;;
    full)
        REPO_ID="${REPO_ID:-behavior-1k/2026-challenge-demos}"
        EPISODES_INDEX=()
        TRAIN_STEPS="${TRAIN_STEPS:-50000}"
        BATCH_SIZE="${BATCH_SIZE:-64}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
        NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-500000}"
        ;;
    *)
        echo "Usage: $0 [smoke|full]" >&2
        exit 2
        ;;
esac

if [[ "$NUM_GPUS" -ne "$(awk -F, '{print NF}' <<<"$CUDA_DEVICES")" ]]; then
    echo "NUM_GPUS=$NUM_GPUS does not match CUDA_DEVICES=$CUDA_DEVICES" >&2
    exit 2
fi
if (( BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "BATCH_SIZE=$BATCH_SIZE must be divisible by NUM_GPUS=$NUM_GPUS" >&2
    exit 2
fi

cd "$OPENPI_DIR"
source .venv/bin/activate

# openpi shards every batch across all devices jax can see, so the device set
# must be restricted before the norm-stats step too, not just for training.
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"

NORM_FILE="$OPENPI_DIR/outputs/assets/pi05_b1k/$REPO_ID/norm_stats.json"
if [[ ! -f "$NORM_FILE" ]]; then
    echo "Computing normalization statistics..."
    NORM_ARGS=(
        --config-name pi05_b1k
        --dataset-root "$DATASET_ROOT"
        --repo-id "$REPO_ID"
        --batch-size "$BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
    )
    if [[ -n "$NORM_MAX_FRAMES" ]]; then
        NORM_ARGS+=(--max-frames "$NORM_MAX_FRAMES")
    fi
    if (( ${#EPISODES_INDEX[@]} )); then
        NORM_ARGS+=(--episodes-index "${EPISODES_INDEX[@]}")
    fi
    python scripts/compute_norm_stats.py "${NORM_ARGS[@]}"
fi

echo "Starting π0.5 $MODE run: $EXP_NAME"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.9}"
# per-step loss goes through tqdm's pbar.write -> stdout, which block-buffers
# when redirected to a log file and would hide loss for ~12k steps at a time.
export PYTHONUNBUFFERED=1
TRAIN_DATA_ARGS=()
if (( ${#EPISODES_INDEX[@]} )); then
    TRAIN_DATA_ARGS+=(--data.episodes-index "${EPISODES_INDEX[@]}")
fi
if [[ -n "${FSDP_DEVICES:-}" ]]; then
    TRAIN_DATA_ARGS+=(--fsdp-devices "$FSDP_DEVICES")
fi
if [[ -n "$SUBTASK_PROMPTS" ]]; then
    TRAIN_DATA_ARGS+=(--data.subtask-prompts-path "$SUBTASK_PROMPTS")
    TRAIN_DATA_ARGS+=(--data.subtask-whole-task-prob "${WHOLE_TASK_PROB:-0.15}")
fi
if (( FRAME_STRIDE > 1 )); then
    TRAIN_DATA_ARGS+=(--data.frame-stride "$FRAME_STRIDE")
fi

# CosineDecaySchedule defaults to decay_steps=30_000 regardless of
# --num-train-steps, so a longer run silently spends everything past 30k at the
# 2.5e-6 floor. Set LR_DECAY_STEPS to the real run length to stretch the cosine.
if [[ -n "${LR_DECAY_STEPS:-}" ]]; then
    TRAIN_DATA_ARGS+=(--lr-schedule.decay-steps "$LR_DECAY_STEPS")
fi
if [[ -n "${LR_PEAK:-}" ]]; then
    TRAIN_DATA_ARGS+=(--lr-schedule.peak-lr "$LR_PEAK")
fi
if [[ -n "${LR_FLOOR:-}" ]]; then
    TRAIN_DATA_ARGS+=(--lr-schedule.decay-lr "$LR_FLOOR")
fi

# openpi refuses --resume together with --overwrite. RESUME=1 continues an
# interrupted run from the last checkpoint in the same EXP_NAME directory.
if [[ -n "${RESUME:-}" ]]; then
    TRAIN_DATA_ARGS+=(--resume)
else
    TRAIN_DATA_ARGS+=(--overwrite)
fi

python scripts/b1k/train_b1k.py pi05_b1k \
    --exp-name "$EXP_NAME" \
    --batch-size "$BATCH_SIZE" \
    --num-workers "$NUM_WORKERS" \
    --num-train-steps "$TRAIN_STEPS" \
    --save-interval "$SAVE_INTERVAL" \
    --no-wandb-enabled \
    --data.repo-id "$REPO_ID" \
    --data.dataset-root "$DATASET_ROOT" \
    "${TRAIN_DATA_ARGS[@]}"

echo "Checkpoint root: $OPENPI_DIR/outputs/checkpoints/pi05_b1k/$EXP_NAME"
