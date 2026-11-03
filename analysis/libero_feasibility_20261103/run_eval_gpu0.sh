#!/bin/bash
# Full LIBERO-10 baseline on GPU 0 ONLY (configs/libero_10_pi05_sft_gpu0.yaml pins env,rollout to GPU 0).
# Ray's temp dir is moved off /tmp (95% full on this host) to /mnt/data1.
# Usage: bash run_eval_gpu0.sh <tag> <total_num_envs> <rollout_epoch>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="$1"; ENVS="$2"; EPOCHS="$3"
REPO=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf
cd "$REPO"
source .venv/bin/activate
export EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}" HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR=/mnt/data1/faye/ray_tmp_libero
LOG="$HERE/results/$TAG"; mkdir -p "$LOG"
start=$(date +%s)
python evaluations/eval_embodied_agent.py --config-path /mnt/data1/faye/code/libero_rlinf_20261103/configs/ \
    --config-name libero_10_pi05_sft_gpu0 "runner.logger.log_path=$LOG" \
    "env.eval.total_num_envs=$ENVS" "env.eval.rollout_epoch=$EPOCHS" 2>&1 | tee "$LOG/console.log"
echo "{\"tag\": \"$TAG\", \"envs\": $ENVS, \"rollout_epoch\": $EPOCHS, \"gpu\": \"0\", \"wall_seconds\": $(( $(date +%s) - start ))}" > "$LOG/wall.json"
