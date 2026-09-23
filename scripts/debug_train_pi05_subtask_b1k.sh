#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPU="${GPU:-0}"
DEBUGPY_PORT="${DEBUGPY_PORT:-5678}"
EPISODE_INDEX="${EPISODE_INDEX:-2800}"
STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-$REPO_ROOT/openpi/outputs/debug/pi05_subtask_$STAMP}"
EPISODES_FILE="$RUN_ROOT/episodes.txt"
REPO_ID="${REPO_ID:-b1k_two_tasks_factorized_b256_all400_e4_full_20260920_213800}"
NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_multihost/pi05_two_tasks_factorized_b256_all400_e4_full_20260920_213800/norm/$REPO_ID/norm_stats.json}"

GPU_PROCESSES="$(nvidia-smi --id="$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
if [[ -n "$GPU_PROCESSES" ]]; then
    printf 'GPU %s already has compute processes:\n%s\n' "$GPU" "$GPU_PROCESSES" >&2
    exit 2
fi

if [[ ! -s "$NORM_STATS_SOURCE" ]]; then
    printf 'Norm stats not found: %s\n' "$NORM_STATS_SOURCE" >&2
    exit 2
fi
mkdir -p "$RUN_ROOT/assets/pi05_b1k/$REPO_ID"
printf '%s\n' "$EPISODE_INDEX" > "$EPISODES_FILE"
cp -a "$NORM_STATS_SOURCE" "$RUN_ROOT/assets/pi05_b1k/$REPO_ID/norm_stats.json"

export DATASET_ROOT="${DATASET_ROOT:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542}"
export SUBTASK_PROMPTS="${SUBTASK_PROMPTS:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/annotations/subtasks/v2_first_skill_backfill/subtask_segments.parquet}"
export EPISODES_FILE
export CUDA_DEVICES="$GPU"
export NUM_GPUS=1
export OPENPI_EXPECTED_GLOBAL_DEVICES=1
export FSDP_DEVICES=1
export BATCH_SIZE="${BATCH_SIZE:-1}"
export TRAIN_STEPS="${TRAIN_STEPS:-1}"
export LR_DECAY_STEPS="$TRAIN_STEPS"
export LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-0}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1}"
export LOG_INTERVAL=1
export NUM_WORKERS="${NUM_WORKERS:-0}"
export NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-256}"
export DEBUGPY_PORT
export SUBTASK_ACTION_CONDITIONING=subtask_only
export WHOLE_TASK_PROB=0
export RESUME=0
export OVERWRITE=0
export EXP_NAME="${EXP_NAME:-pi05_b1k_subtask_debug_$STAMP}"
export REPO_ID
export ASSETS_BASE_DIR="$RUN_ROOT/assets"
export CHECKPOINT_BASE_DIR="$RUN_ROOT/checkpoints"
export PREFLIGHT_OUTPUT_DIR="$RUN_ROOT/preflight"
export OPENPI_REQUIRE_FINITE_METRICS=1
export OPENPI_VERIFY_FINAL_CHECKPOINT=1
export OPENPI_SUCCESS_PATH="$RUN_ROOT/_SUCCESS.json"

printf 'Debugger endpoint 127.0.0.1:%s will open after preflight and norm-stat preparation.\n' "$DEBUGPY_PORT"
printf 'GPU: %s, episode: %s, steps: %s, output: %s\n' "$GPU" "$EPISODE_INDEX" "$TRAIN_STEPS" "$RUN_ROOT"
exec bash "$REPO_ROOT/scripts/train_pi05_subtask_b1k.sh" full
