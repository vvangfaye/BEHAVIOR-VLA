#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
CODE_ROOT="${CODE_ROOT:-/mnt/afs/visitor34/code}"
GROOT_DIR="${GROOT_DIR:-$CODE_ROOT/Isaac-GR00T}"
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
BASE_MODEL="${BASE_MODEL:-nvidia/GR00T-N1.7-3B}"
OUTPUT_DIR="${OUTPUT_DIR:-$GROOT_DIR/outputs}"
EXP_NAME="${EXP_NAME:-gr00t_b1k_${MODE}_$(date +%Y%m%d_%H%M%S)}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_ONLY_MODEL="${SAVE_ONLY_MODEL:-false}"
PERSISTENT_CACHE_ROOT="${PRETRAINED_CACHE_ROOT:-/mnt/afs/visitor34/devhome/cache}"
HF_HOME="${HF_HOME:-$PERSISTENT_CACHE_ROOT/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7890}"

if [[ ! -x "$GROOT_DIR/.venv/bin/python" ]]; then
    echo "GR00T environment not found: $GROOT_DIR/.venv" >&2
    exit 1
fi

DATASET_PATH="${DATASET_PATH:-$("$CODE_ROOT/scripts/prepare_gr00t_b1k_view.sh" "$MODE")}"

case "$MODE" in
    smoke)
        MAX_STEPS="${MAX_STEPS:-2}"
        GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-$NUM_GPUS}"
        SAVE_STEPS="${SAVE_STEPS:-$MAX_STEPS}"
        EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.001}"
        NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-2}"
        ;;
    full)
        MAX_STEPS="${MAX_STEPS:-150000}"
        GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-2048}"
        SAVE_STEPS="${SAVE_STEPS:-1500}"
        EPISODE_SAMPLING_RATE="${EPISODE_SAMPLING_RATE:-0.1}"
        NUM_SHARDS_PER_EPOCH="${NUM_SHARDS_PER_EPOCH:-100000}"
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
if (( GLOBAL_BATCH_SIZE % NUM_GPUS != 0 )); then
    echo "GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE must be divisible by NUM_GPUS=$NUM_GPUS" >&2
    exit 2
fi
if [[ "$SAVE_ONLY_MODEL" != "true" && "$SAVE_ONLY_MODEL" != "false" ]]; then
    echo "SAVE_ONLY_MODEL must be true or false, got: $SAVE_ONLY_MODEL" >&2
    exit 2
fi

EXTRA_TRAIN_ARGS=()
if [[ "$SAVE_ONLY_MODEL" == "true" ]]; then
    EXTRA_TRAIN_ARGS+=(--save-only-model)
fi

cd "$GROOT_DIR"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HOME
export HF_HUB_CACHE
if [[ -n "$PROXY_URL" ]]; then
    export http_proxy="$PROXY_URL"
    export https_proxy="$PROXY_URL"
    export HTTP_PROXY="$PROXY_URL"
    export HTTPS_PROXY="$PROXY_URL"
    export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
fi

echo "Starting GR00T $MODE run: $EXP_NAME"
torchrun --nproc_per_node="$NUM_GPUS" --master_port="${MASTER_PORT:-29500}" \
    scripts/b1k/train_b1k.py \
    --experiment-name "$EXP_NAME" \
    --base-model-path "$BASE_MODEL" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/b1k/r1pro.py \
    --num-gpus "$NUM_GPUS" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    --output-dir "$OUTPUT_DIR" \
    --save-steps "$SAVE_STEPS" \
    --save-total-limit 2 \
    --max-steps "$MAX_STEPS" \
    --dataloader-num-workers "$NUM_WORKERS" \
    --episode-sampling-rate "$EPISODE_SAMPLING_RATE" \
    --num-shards-per-epoch "$NUM_SHARDS_PER_EPOCH" \
    --decode-only-used-frames \
    "${EXTRA_TRAIN_ARGS[@]}"

echo "Output root: $OUTPUT_DIR/$EXP_NAME"
