#!/usr/bin/env bash
# Fine-tune G0.5-base on BEHAVIOR-1K 2026-challenge-demos.
#
# Usage:
#   NUM_GPUS=8 ./scripts/run_g05_b1k_finetune.sh
#   MAX_STEPS=50000 NUM_GPUS=8 BATCH_SIZE=64 ./scripts/run_g05_b1k_finetune.sh
#
# Required environment:
#   G05_CKPT_DIR       directory containing model_state_dict.pt,
#                      action_tokenizer.pt, qwen3_5_2b_base_processor/
#                      (defaults to the standard HF layout under
#                      /mnt/workspace/wangfaye/cache/huggingface/hub/OpenGalaxea/G05)
#   DATASET_ROOT       BEHAVIOR LeRobot v3 dataset
#                      (defaults to /var/tmp/behavior-1k/2026-challenge-demos)
#   G05_OUTPUT_DIR     Where checkpoints / Hydra logs / trainer logs land
#                      (defaults to /mnt/workspace/wangfaye/cache/g05_finetune)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GALAXEA_REPO="${GALAXEA_REPO:-$REPO_ROOT/GalaxeaVLA}"
GALAXEA_VENV="${GALAXEA_VENV:-$GALAXEA_REPO/.venv}"

# uv-sync on this host ran out of JuiceFS inode quota; use /tmp instead.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export PYTHONPATH="${GALAXEA_REPO}:${PYTHONPATH:-}"

# G05 shipped checkpoints (already downloaded on this host).
G05_CKPT_DIR="${G05_CKPT_DIR:-/mnt/workspace/wangfaye/cache/huggingface/hub/OpenGalaxea/G05}"
export G05_CKPT_DIR

# BEHAVIOR-1K data.
DATASET_ROOT="${DATASET_ROOT:-/var/tmp/behavior-1k/2026-challenge-demos}"
export DATASET_ROOT

# Where checkpoints land. /mnt/workspace survives Pod restarts.
G05_OUTPUT_DIR="${G05_OUTPUT_DIR:-/mnt/workspace/wangfaye/cache/g05_finetune}"
export G05_OUTPUT_DIR
mkdir -p "$G05_OUTPUT_DIR"

# Knobs.
NUM_GPUS="${NUM_GPUS:-8}"
MAX_STEPS="${MAX_STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SAVE_INTERVAL="${SAVE_INTERVAL:-2000}"
NUM_WORKERS="${NUM_WORKERS:-8}"

TASK_CFG="${TASK_CFG:-b1k}"

LOG_FILE="${LOG_FILE:-/dev/shm/wangfaye/logs/g05_finetune.log}"

# shellcheck disable=SC1091
source "$GALAXEA_VENV/bin/activate"

# Avoid TF / CUDA noise.
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HYDRA_FULL_ERROR=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1

echo "[$(date '+%F %T')] launching G05 fine-tune on $TASK_CFG"
echo "  num_gpus=$NUM_GPUS batch_size=$BATCH_SIZE max_steps=$MAX_STEPS save_interval=$SAVE_INTERVAL"
echo "  output_dir=$G05_OUTPUT_DIR"
echo "  log_file=$LOG_FILE"

# Override the b1k config defaults for our full run: 50k steps, 8-GPU FSDP-friendly
# batch, 2k-step save interval (50 saves × ~6 GB ≈ 300 GB disk, fits in budget).
exec torchrun \
    --standalone --nnodes=1 --nproc_per_node="$NUM_GPUS" \
    scripts/finetune.py \
    "task=$TASK_CFG" \
    "model.max_steps=$MAX_STEPS" \
    "model.batch_size=$BATCH_SIZE" \
    "model.num_workers=$NUM_WORKERS" \
    "+trainer.save_interval=$SAVE_INTERVAL" \
    "logger.mode=offline" \
    "logger.workspace=g05_b1k" \
    "+use_fsdp=true" \
    "+fsdp_sharding_strategy=full_shard" \
    "+fsdp_reshard_after_forward=true" \
    >> "$LOG_FILE" 2>&1