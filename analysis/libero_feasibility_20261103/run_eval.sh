#!/bin/bash
# LIBERO-10 baseline evaluation of RLinf-Pi05-LIBERO-SFT on ONE local GPU (feasibility check, 2026-11-03).
# Usage: bash run_eval.sh <tag> <total_num_envs> <rollout_epoch> [gpu]
#   smoke: bash run_eval.sh smoke 10 1      -> 10 trajectories (one per task)
#   full : bash run_eval.sh full 50 10      -> 500 trajectories (all LIBERO-10 init states); ~100 GiB host RAM
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="$1"; ENVS="$2"; EPOCHS="$3"; GPU="${4:-0}"
# The RLinf clone lives at a space-free path (its install script breaks on "Behavior-1k Challenge");
# $HERE/RLinf is a symlink to it. cd to the physical path so every derived path is space-free.
cd "$(readlink -f "$HERE/RLinf")"
source .venv/bin/activate
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
LOG="$HERE/results/$TAG"
mkdir -p "$LOG"
start=$(date +%s)
bash evaluations/run_eval.sh libero_10_openpi_pi05_eval \
    "rollout.model.model_path=$HERE/models/RLinf-Pi05-LIBERO-SFT" \
    "env.eval.total_num_envs=$ENVS" \
    "env.eval.rollout_epoch=$EPOCHS" \
    "env.eval.video_cfg.save_video=False" \
    "runner.logger.log_path=$LOG" 2>&1 | tee "$LOG/console.log"
echo "{\"tag\": \"$TAG\", \"envs\": $ENVS, \"rollout_epoch\": $EPOCHS, \"gpu\": \"$GPU\", \"wall_seconds\": $(( $(date +%s) - start ))}" > "$LOG/wall.json"
