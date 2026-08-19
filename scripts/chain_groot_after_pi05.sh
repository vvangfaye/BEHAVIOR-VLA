#!/usr/bin/env bash
# Wait for the running π0.5 job to exit, then start GR00T N1.7 training on the
# freed GPUs. Keeps the 8 H20s busy without a human waiting for the handover.
#
# Usage:
#   ./scripts/chain_groot_after_pi05.sh          # waits for the running π0.5 job
#   FORCE=1 ./scripts/chain_groot_after_pi05.sh  # start GR00T now
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${LOG_DIR:-/dev/shm/wangfaye/logs}"
GROOT_LOG="$LOG_DIR/groot_full.log"

pi05_running() {
    pgrep -f "train_b1k\.py pi05_b1k" >/dev/null 2>&1
}

if ! pi05_running && [[ -z "${FORCE:-}" ]]; then
    echo "No running π0.5 trainer found (train_b1k.py pi05_b1k)." >&2
    echo "Refusing to start: pass FORCE=1 to launch GR00T immediately." >&2
    exit 1
fi

# GR00T reads the shipped meta/stats.json, so there is no norm-stats pass.
DATASET_PATH="${DATASET_PATH:-/var/tmp/behavior-1k/2026-challenge-demos}"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-50000}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
NUM_WORKERS="${NUM_WORKERS:-24}"

echo "[$(date '+%F %T')] waiting for the π0.5 trainer to finish"
while pi05_running; do
    sleep 60
done
echo "[$(date '+%F %T')] π0.5 trainer is gone"

# Only now is it safe to reap leftover dataloader workers; each holds tens of GB
# and would otherwise push the pod's memcg toward its limit.
for _ in 1 2 3; do
    pkill -KILL -x pt_data_worker 2>/dev/null || true
    sleep 5
done
echo "[$(date '+%F %T')] workers left: $(pgrep -x pt_data_worker | wc -l)"

sleep 30  # let GPU memory be released

cd "$REPO_ROOT"
echo "[$(date '+%F %T')] starting GR00T training -> $GROOT_LOG"
exec env \
    PYTHONUNBUFFERED=1 \
    DATASET_PATH="$DATASET_PATH" \
    PRETRAINED_CACHE_ROOT=/mnt/workspace/wangfaye/cache \
    HF_HUB_OFFLINE=1 \
    PROXY_URL= \
    NUM_GPUS="$NUM_GPUS" CUDA_DEVICES="$CUDA_DEVICES" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    MAX_STEPS="$MAX_STEPS" SAVE_STEPS="$SAVE_STEPS" \
    NUM_WORKERS="$NUM_WORKERS" \
    EXP_NAME="gr00t_b1k_full_$(date +%Y%m%d_%H%M%S)" \
    stdbuf -oL -eL ./scripts/train_gr00t_b1k.sh full >> "$GROOT_LOG" 2>&1
