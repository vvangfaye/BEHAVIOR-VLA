#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
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
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
PERSISTENT_CACHE_ROOT="${PRETRAINED_CACHE_ROOT:-/mnt/workspace/wangfaye/cache}"
HF_HOME="${HF_HOME:-$PERSISTENT_CACHE_ROOT/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
PROXY_URL="${PROXY_URL:-}"

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
# RESUME=1 continues from the newest complete checkpoint under
# $OUTPUT_DIR/$EXP_NAME, so EXP_NAME must match the interrupted run.
if [[ -n "${RESUME:-}" ]]; then
    EXTRA_TRAIN_ARGS+=(--resume-from-checkpoint)
fi

cd "$GROOT_DIR"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export HF_HOME
export HF_HUB_CACHE
# transformers resolves the Cosmos-Reason2-2B tokenizer through
# _patch_mistral_regex -> is_base_mistral -> HfApi.model_info, a plain requests
# call that ignores HF_HUB_OFFLINE. huggingface.co is unreachable here, so point
# it at the mirror; otherwise every rank dies on a connect timeout.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
# The Cosmos-Reason2-2B backbone repo is gated and unreachable from this node, and
# setup.py rebuilds the processor from config.model.model_name rather than the value
# stored in the checkpoint. Point it at the local merged copy when present: being a
# local directory also stops transformers from probing the hub for the tokenizer.
GROOT_BACKBONE_PATH="${GROOT_BACKBONE_PATH:-/mnt/workspace/wangfaye/cache/local_models/nvidia/Cosmos-Reason2-2B}"
if [[ -d "$GROOT_BACKBONE_PATH" ]]; then
    export GROOT_BACKBONE_PATH
    echo "Using local VLM backbone: $GROOT_BACKBONE_PATH"
else
    echo "Local VLM backbone not found, falling back to $BASE_MODEL default backbone" >&2
    unset GROOT_BACKBONE_PATH
fi
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
    --save-total-limit "$SAVE_TOTAL_LIMIT" \
    --max-steps "$MAX_STEPS" \
    --dataloader-num-workers "$NUM_WORKERS" \
    --episode-sampling-rate "$EPISODE_SAMPLING_RATE" \
    --num-shards-per-epoch "$NUM_SHARDS_PER_EPOCH" \
    --decode-only-used-frames \
    "${EXTRA_TRAIN_ARGS[@]}"

echo "Output root: $OUTPUT_DIR/$EXP_NAME"
