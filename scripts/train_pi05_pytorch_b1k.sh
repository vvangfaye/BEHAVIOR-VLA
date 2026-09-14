#!/usr/bin/env bash
# BATCH_SIZE is the global microbatch; effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS.
set -euo pipefail
MODE="${1:-smoke}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$REPO_ROOT/openpi}"
PYTORCH_ENV="${PYTORCH_ENV:-$OPENPI_DIR/.venv-torch}"
TRAIN_PYTHON="$PYTORCH_ENV/bin/python"
DATASET_ROOT="${DATASET_ROOT:-/data/2026-challenge-demos}"
NUM_GPUS="${NUM_GPUS:-1}"
CUDA_DEVICES="${CUDA_DEVICES:-0}"
BATCH_SIZE="${BATCH_SIZE:-$NUM_GPUS}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
SUBTASK_PREDICTION="${SUBTASK_PREDICTION:-0}"
EXP_NAME="${EXP_NAME:-pi05_b1k_torch_${MODE}_$(date +%Y%m%d_%H%M%S)}"
ASSETS_BASE_DIR="${ASSETS_BASE_DIR:-$OPENPI_DIR/outputs/assets}"
CHECKPOINT_BASE_DIR="${CHECKPOINT_BASE_DIR:-$OPENPI_DIR/outputs/checkpoints}"
CONFIG_NAME=pi05_b1k_pytorch
case "$MODE" in
    check|full)
        REPO_ID="${REPO_ID:-behavior-1k/2026-challenge-demos}"
        TRAIN_STEPS="${TRAIN_STEPS:-50000}"
        NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-500000}"
        ;;
    smoke)
        REPO_ID="${REPO_ID:-behavior-1k/2026-challenge-demos-smoke}"
        TRAIN_STEPS="${TRAIN_STEPS:-2}"
        NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-256}"
        ;;
    *) echo "Usage: $0 [check|smoke|full]" >&2; exit 2 ;;
esac
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
for value in "$NUM_GPUS" "$BATCH_SIZE" "$GRAD_ACCUM_STEPS" "$TRAIN_STEPS" "$FRAME_STRIDE" "$SAVE_INTERVAL" "${LOG_INTERVAL:-100}" "$NORM_MAX_FRAMES"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "GPU, batch, accumulation, steps and stride must be positive integers" >&2; exit 2; }
done
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || { echo "NUM_WORKERS must be a nonnegative integer" >&2; exit 2; }
for flag in "${GRAD_CHECKPOINTING:-1}" "${RESUME:-0}" "${OVERWRITE:-0}"; do
    [[ "$flag" == 0 || "$flag" == 1 ]] || { echo "GRAD_CHECKPOINTING, RESUME and OVERWRITE must be 0 or 1" >&2; exit 2; }
done
[[ "$SUBTASK_PREDICTION" == 0 || "$SUBTASK_PREDICTION" == 1 ]] || { echo "SUBTASK_PREDICTION must be 0 or 1" >&2; exit 2; }
[[ "$NUM_GPUS" -eq "$(awk -F, '{print NF}' <<< "$CUDA_DEVICES")" ]] || { echo "NUM_GPUS does not match CUDA_DEVICES" >&2; exit 2; }
(( BATCH_SIZE % NUM_GPUS == 0 )) || { echo "Global BATCH_SIZE must be divisible by NUM_GPUS" >&2; exit 2; }
[[ "${RESUME:-0}" != 1 || "${OVERWRITE:-0}" != 1 ]] || { echo "RESUME and OVERWRITE cannot both be 1" >&2; exit 2; }
[[ -x "$TRAIN_PYTHON" ]] || { echo "Torch environment missing: $PYTORCH_ENV; run scripts/setup_pytorch_env.sh" >&2; exit 1; }
[[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "Dataset metadata missing: $DATASET_ROOT/meta/info.json" >&2; exit 1; }
# Interpret user-supplied relative paths from the caller's directory before cd.
for name in OPENPI_DIR DATASET_ROOT PYTORCH_ENV EPISODES_FILE PYTORCH_WEIGHT_PATH SUBTASK_PROMPTS ASSETS_BASE_DIR CHECKPOINT_BASE_DIR PREFLIGHT_OUTPUT_DIR; do
    value="${!name:-}"
    if [[ -n "$value" && "$value" != /* ]]; then printf -v "$name" '%s/%s' "$PWD" "$value"; fi
done
TRAIN_PYTHON="$PYTORCH_ENV/bin/python"
# Only Torch owns CUDA. Shared config/tokenization still import JAX on the CPU.
export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
cd "$OPENPI_DIR"
"$TRAIN_PYTHON" scripts/install_transformers_patch.py --check

EPISODES=()
if [[ -n "${EPISODES_FILE:-}" ]]; then
    [[ -f "$EPISODES_FILE" ]] || { echo "Episode file missing: $EPISODES_FILE" >&2; exit 1; }
    while IFS= read -r episode || [[ -n "$episode" ]]; do
        [[ "$episode" =~ ^[0-9]+$ ]] || { echo "Invalid episode id: $episode" >&2; exit 2; }
        EPISODES+=("$episode")
    done < "$EPISODES_FILE"
    (( ${#EPISODES[@]} )) || { echo "Episode file is empty" >&2; exit 2; }
elif [[ "$MODE" == smoke ]]; then
    EPISODES=(0 1)
fi

SUBTASK_PROMPTS="${SUBTASK_PROMPTS:-}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-512}"
MAX_SUBTASK_LEN="${MAX_SUBTASK_LEN:-64}"
if [[ "$SUBTASK_PREDICTION" == 1 ]]; then
    [[ "${WHOLE_TASK_PROB:-0}" == 0 || "${WHOLE_TASK_PROB:-0}" == 0.0 ]] || { echo "Joint subtask training requires WHOLE_TASK_PROB=0" >&2; exit 2; }
    SUBTASK_PROMPTS="${SUBTASK_PROMPTS:-$DATASET_ROOT/meta/subtask_segments.parquet}"
    PREFLIGHT_OUTPUT_DIR="${PREFLIGHT_OUTPUT_DIR:-$OPENPI_DIR/outputs/subtask_preflight}"
    mkdir -p "$PREFLIGHT_OUTPUT_DIR"
    VALIDATED_EPISODES="$PREFLIGHT_OUTPUT_DIR/$EXP_NAME.episodes.txt"
    CHECK_ARGS=(--dataset-root "$DATASET_ROOT" --segments-path "$SUBTASK_PROMPTS"
        --max-token-len "$MAX_TOKEN_LEN" --max-subtask-len "$MAX_SUBTASK_LEN"
        --empty-label-policy "${SUBTASK_EMPTY_LABEL_POLICY:-drop_episode}"
        --report-file "$PREFLIGHT_OUTPUT_DIR/$EXP_NAME.json" --write-episodes-file "$VALIDATED_EPISODES")
    if (( ${#EPISODES[@]} )); then CHECK_ARGS+=(--episodes-index "${EPISODES[@]}"); fi
    "$TRAIN_PYTHON" scripts/b1k/check_subtask_data.py "${CHECK_ARGS[@]}"
    EPISODES=()
    while IFS= read -r episode || [[ -n "$episode" ]]; do
        [[ "$episode" =~ ^[0-9]+$ ]] || { echo "Invalid validated episode: $episode" >&2; exit 2; }
        EPISODES+=("$episode")
    done < "$VALIDATED_EPISODES"
    (( ${#EPISODES[@]} )) || { echo "No fully annotated episodes remain" >&2; exit 1; }
fi
if [[ "$MODE" == check ]]; then
    [[ "$SUBTASK_PREDICTION" == 1 ]] || { echo "Use the subtask wrapper for annotation preflight" >&2; exit 2; }
    exit 0
fi

TRAIN_ARGS=("$CONFIG_NAME" --exp-name "$EXP_NAME" --batch-size "$BATCH_SIZE"
    --pytorch-gradient-accumulation-steps "$GRAD_ACCUM_STEPS"
    --pytorch-training-precision "${TRAIN_PRECISION:-bfloat16}"
    --num-workers "$NUM_WORKERS" --num-train-steps "$TRAIN_STEPS"
    --save-interval "$SAVE_INTERVAL" --log-interval "${LOG_INTERVAL:-100}"
    --assets-base-dir "$ASSETS_BASE_DIR" --checkpoint-base-dir "$CHECKPOINT_BASE_DIR"
    --data.repo-id "$REPO_ID" --data.dataset-root "$DATASET_ROOT"
    --data.frame-stride "$FRAME_STRIDE" --no-wandb-enabled)
if [[ "${GRAD_CHECKPOINTING:-1}" == 0 ]]; then TRAIN_ARGS+=(--no-pytorch-gradient-checkpointing); fi
if [[ "${RESUME:-0}" == 1 ]]; then
    TRAIN_ARGS+=(--resume)
else
    : "${PYTORCH_WEIGHT_PATH:?Set PYTORCH_WEIGHT_PATH to converted weights; see scripts/convert_pi05_to_pytorch.sh}"
    if [[ ! -f "$PYTORCH_WEIGHT_PATH" && ! -f "$PYTORCH_WEIGHT_PATH/model.safetensors" ]]; then
        echo "model.safetensors not found: $PYTORCH_WEIGHT_PATH" >&2; exit 1
    fi
    TRAIN_ARGS+=(--pytorch-weight-path "$PYTORCH_WEIGHT_PATH")
    if [[ "${OVERWRITE:-0}" == 1 ]]; then TRAIN_ARGS+=(--overwrite); fi
fi
if (( ${#EPISODES[@]} )); then TRAIN_ARGS+=(--data.episodes-index "${EPISODES[@]}"); fi
if [[ -n "$SUBTASK_PROMPTS" ]]; then
    TRAIN_ARGS+=(--data.subtask-prompts-path "$SUBTASK_PROMPTS" --data.subtask-whole-task-prob "${WHOLE_TASK_PROB:-0}")
fi
if [[ "$SUBTASK_PREDICTION" == 1 ]]; then
    TRAIN_ARGS+=(--model.train-subtask-prediction --model.max-token-len "$MAX_TOKEN_LEN"
        --model.max-subtask-len "$MAX_SUBTASK_LEN" --model.subtask-loss-weight "${SUBTASK_LOSS_WEIGHT:-1.0}"
        --model.subtask-loss-chunk-size "${SUBTASK_LOSS_CHUNK_SIZE:-8}")
fi
if [[ -n "${LR_DECAY_STEPS:-}" ]]; then TRAIN_ARGS+=(--lr-schedule.decay-steps "$LR_DECAY_STEPS"); fi
if [[ -n "${LR_PEAK:-}" ]]; then TRAIN_ARGS+=(--lr-schedule.peak-lr "$LR_PEAK"); fi
if [[ -n "${LR_FLOOR:-}" ]]; then TRAIN_ARGS+=(--lr-schedule.decay-lr "$LR_FLOOR"); fi

NORM_FILE="$ASSETS_BASE_DIR/$CONFIG_NAME/$REPO_ID/norm_stats.json"
if [[ "${RESUME:-0}" != 1 && ! -f "$NORM_FILE" ]]; then
    NORM_ARGS=(--config-name "$CONFIG_NAME" --dataset-root "$DATASET_ROOT" --repo-id "$REPO_ID"
        --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --max-frames "$NORM_MAX_FRAMES"
        --assets-base-dir "$ASSETS_BASE_DIR")
    if (( ${#EPISODES[@]} )); then NORM_ARGS+=(--episodes-index "${EPISODES[@]}"); fi
    "$TRAIN_PYTHON" scripts/compute_norm_stats.py "${NORM_ARGS[@]}"
fi
echo "Torch: $NUM_GPUS GPUs, global microbatch=$BATCH_SIZE, accumulation=$GRAD_ACCUM_STEPS, effective batch=$((BATCH_SIZE * GRAD_ACCUM_STEPS))"
if (( NUM_GPUS == 1 )); then
    exec "$TRAIN_PYTHON" scripts/b1k/train_b1k_pytorch.py "${TRAIN_ARGS[@]}"
else
    exec "$TRAIN_PYTHON" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node "$NUM_GPUS" \
        scripts/b1k/train_b1k_pytorch.py "${TRAIN_ARGS[@]}"
fi
