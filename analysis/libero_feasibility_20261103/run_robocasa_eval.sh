#!/bin/bash
# RoboCasa365 evaluation in .venv-robocasa365 (installed 2026-11-04 with user approval). Both GPUs.
# Usage: bash run_robocasa_eval.sh <tag> [config=robocasa365_composite4_eval_pi0_2gpu] [hydra overrides...]
# Never run two RLinf jobs at once (RLinf's ray.init(address='auto') attaches to any running Ray cluster).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="$1"; CFG="${2:-robocasa365_composite4_eval_pi0_2gpu}"; shift; shift || true
REPO=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf
cd "$REPO"
source .venv-robocasa365/bin/activate
export EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}" HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ROBOCASA_ASSETS_PATH=/mnt/data1/faye/code/libero_rlinf_20261103/robocasa_assets
export RAY_TMPDIR=/mnt/data1/faye/ray_tmp_robocasa
mkdir -p "$RAY_TMPDIR"
LOG="$HERE/results/$TAG"; mkdir -p "$LOG"
if ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; then echo "another RLinf job is running; refusing to start" >&2; exit 3; fi
start=$(date +%s)
python evaluations/eval_embodied_agent.py --config-path /mnt/data1/faye/code/libero_rlinf_20261103/configs/ \
    --config-name "$CFG" "runner.logger.log_path=$LOG" "$@" 2>&1 | tee "$LOG/console.log"
echo "{\"tag\": \"$TAG\", \"config\": \"$CFG\", \"wall_seconds\": $(( $(date +%s) - start ))}" > "$LOG/wall.json"
