#!/usr/bin/env bash
# Pre-compute BEHAVIOR-1K normalization statistics for G0.5 fine-tune.
#
# Without this, the trainer does an in-process online stats sweep over all
# 210 M frames, which exhausts the pod memcg before the first step. Sampling
# every 100th frame is more than enough for stable z-scores and finishes in
# minutes on a single CPU.
#
# Output goes to /mnt/workspace/wangfaye/cache/g05_finetune/b1k_stats.json
# and is consumed via the existing b1k.yaml `datastatics_path` override.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GALAXEA_REPO="${GALAXEA_REPO:-$REPO_ROOT/GalaxeaVLA}"
GALAXEA_VENV="${GALAXEA_VENV:-$GALAXEA_REPO/.venv}"
STATS_OUT="${STATS_OUT:-/mnt/workspace/wangfaye/cache/g05_finetune/b1k_stats.json}"
STATS_DOWNSAMPLE_RATE="${STATS_DOWNSAMPLE_RATE:-100}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv_cache}"
export PYTHONPATH="${GALAXEA_REPO}:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HYDRA_FULL_ERROR=1
export PYTHONNOUSERSITE=1

# shellcheck disable=SC1091
source "$GALAXEA_VENV/bin/activate"

mkdir -p "$(dirname "$STATS_OUT")"

# Match the runtime config:
#   - mixture config: configs/data/b1k.yaml
#   - task config:    configs/task/b1k.yaml
# The compute script ignores the trainer-side fields (max_steps, optimizer,
# etc.) and only reads data + datastatics_path.
echo "[$(date '+%F %T')] computing stats from b1k mixture, output -> $STATS_OUT"
cd "$GALAXEA_REPO"
python tests/test_dataloader_batch.py \
    --task-config "b1k" \
    --stats "$STATS_OUT" \
    --stats-downsample-rate "$STATS_DOWNSAMPLE_RATE" \
    --batch-size 4 \
    --num-workers 4
echo "[$(date '+%F %T')] stats written to $STATS_OUT"
ls -la "$STATS_OUT"