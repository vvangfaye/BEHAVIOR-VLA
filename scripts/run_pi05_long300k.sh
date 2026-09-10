#!/usr/bin/env bash
# Resume π0.5 training from an existing checkpoint and continue up to
# 300,000 steps, matching the GR00T long300k run for an apples-to-apples
# comparison. Mirrors scripts/run_pi05_90min.sh but in resume/long mode.
#
# The first launch started 2026-08-14 21:48 and ran for ~11.5 h before a
# single bad video frame in chunk-023/file-004.mp4 tripped
# video_utils.FrameTimestampError (queried ts 8642.1328 vs loaded 8642.1338,
# beyond the 5e-4 tolerance). The trainer has been bumped to 5e-3 (5 ms ≈
# 15% of one 30 fps frame) in src/openpi/training/config.py.
#
# Required environment:
#   DATASET_ROOT  dataset location (defaults to /var/tmp/behavior-1k/2026-challenge-demos)
#   OPENPI_CACHE  OpenPI persistent cache holding pi05_base (defaults to
#                 /mnt/workspace/wangfaye/cache/openpi)
#   LOG_FILE      where to redirect trainer stdout/stderr (defaults to
#                 /dev/shm/wangfaye/logs/pi05_long300k.log)
#   RESUME_FROM   step to resume from under $EXP_NAME (defaults to 80000,
#                 the last full save before the timestamp crash)
set -euo pipefail

CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
OPENPI_DIR="${OPENPI_DIR:-$CODE_ROOT/openpi}"
NORM_FILE="$OPENPI_DIR/outputs/assets/pi05_b1k/behavior-1k/2026-challenge-demos/norm_stats.json"
RESUME_DIR="$OPENPI_DIR/outputs/checkpoints/pi05_b1k/pi05_b1k_full_20260803_162620"
RESUME_FROM="${RESUME_FROM:-80000}"

if [[ ! -f "$NORM_FILE" ]]; then
    echo "Missing persisted normalization statistics: $NORM_FILE" >&2
    echo "Compute them before starting the timed run." >&2
    exit 1
fi
if [[ ! -d "$RESUME_DIR/$RESUME_FROM" ]]; then
    echo "Missing resume checkpoint: $RESUME_DIR/$RESUME_FROM" >&2
    exit 1
fi

# openpi.shared.download reads OPENPI_DATA_HOME to resolve gs://openpi-assets
# URLs. Point it at the persistent cache so we still find pi05_base/params
# after a Pod restart wipes /mnt/afs but leaves /mnt/workspace/wangfaye alone.
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/mnt/workspace/wangfaye/cache/openpi}"

# Match the configuration used for the 50k run: 8 GPUs, global batch 64
# (per-rank 8), one worker per rank; XLA mem fraction mirrors train_pi05_b1k.sh.
# FSDP_DEVICES=8 is required to resume the 49999 checkpoint: Orbax stored the
# optimizer state with mesh ('batch':1, 'fsdp':8) and reloading it on a
# ('batch':8, 'fsdp':1) mesh trips a pjit sharding mismatch on the first step.
export NUM_GPUS="${NUM_GPUS:-8}"
export CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
export BATCH_SIZE="${BATCH_SIZE:-64}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export FSDP_DEVICES="${FSDP_DEVICES:-8}"
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}"
export DATASET_ROOT="${DATASET_ROOT:-/var/tmp/behavior-1k/2026-challenge-demos}"

# Resume from the existing 50k checkpoint and grow the run to 300k steps.
# RESUME=1 trains/train_pi05_b1k.sh passes --resume so Orbax restores the
# last complete step (49999) under the same EXP_NAME directory.
export EXP_NAME="pi05_b1k_full_20260803_162620"
export TRAIN_STEPS="${TRAIN_STEPS:-300000}"
export RESUME="${RESUME:-1}"

LOG_FILE="${LOG_FILE:-/dev/shm/wangfaye/logs/pi05_long300k.log}"
mkdir -p "$(dirname "$LOG_FILE")"

echo "Resuming π0.5 training up to step ${TRAIN_STEPS} from $RESUME_DIR/${RESUME_FROM}"
echo "GPUs=${CUDA_DEVICES} batch=${BATCH_SIZE} save_interval=${SAVE_INTERVAL}"
echo "Log file: $LOG_FILE"

# stdbuf keeps the trainer output line-buffered so `tail -f` is readable.
exec stdbuf -oL -eL \
    env PYTHONUNBUFFERED=1 \
    "$CODE_ROOT/scripts/train_pi05_b1k.sh" full \
    >> "$LOG_FILE" 2>&1
