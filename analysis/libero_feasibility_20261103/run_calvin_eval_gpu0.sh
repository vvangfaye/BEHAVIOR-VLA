#!/bin/bash
# CALVIN ABC->D baseline on GPU 0 only, in the separate .venv-calvin. Ray temp dir off /tmp.
# Usage: bash run_calvin_eval_gpu0.sh <tag> <total_num_envs> <max_steps_per_rollout_epoch>
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="$1"; ENVS="$2"; STEPS="$3"
REPO=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf
cd "$REPO"
source .venv-calvin/bin/activate
export EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}" HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR=/mnt/data1/faye/ray_tmp_libero
LOG="$HERE/results/$TAG"; mkdir -p "$LOG"
start=$(date +%s)
python evaluations/eval_embodied_agent.py --config-path /mnt/data1/faye/code/libero_rlinf_20261103/configs/ \
    --config-name calvin_abc_d_pi05_sft_eval_gpu0 "runner.logger.log_path=$LOG" \
    "env.eval.total_num_envs=$ENVS" "env.eval.max_steps_per_rollout_epoch=$STEPS" 2>&1 | tee "$LOG/console.log"
echo "{\"tag\": \"$TAG\", \"envs\": $ENVS, \"max_steps_per_rollout_epoch\": $STEPS, \"gpu\": \"0\", \"wall_seconds\": $(( $(date +%s) - start ))}" > "$LOG/wall.json"
