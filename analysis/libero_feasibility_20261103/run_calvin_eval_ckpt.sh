#!/bin/bash
# CALVIN ABC->D evaluation (1000 chains: 50 envs x 9600 steps, 480 steps per chain) of an RL checkpoint or the SFT
# start (CKPT=none) on both local GPUs. Per-chain lines "[calvin eval] trial_id=K, completed=C" feed calvin_paired.py.
# Usage: bash run_calvin_eval_ckpt.sh <tag> <path/to/full_weights.pt|none> [envs=50] [steps=9600]
# Never run two RLinf jobs at once (RLinf's ray.init(address='auto') attaches to any running Ray cluster).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="$1"; CKPT="$2"; ENVS="${3:-50}"; STEPS="${4:-9600}"
REPO=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf
cd "$REPO"
source .venv-calvin/bin/activate
export EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}" HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false NCCL_P2P_DISABLE=1
export RAY_TMPDIR=/mnt/data1/faye/ray_tmp_eval_calvin
mkdir -p "$RAY_TMPDIR"
LOG="$HERE/results/$TAG"; mkdir -p "$LOG"
EXTRA=()
if [ "$CKPT" != "none" ]; then
    test -f "$CKPT"
    # The eval-mode policy has no value head and RLinf loads strictly: evaluate a copy without value_head.* tensors.
    EVALW="$LOG/eval_weights_novalue.pt"
    python "$HERE/strip_value_head.py" "$CKPT" "$EVALW"
    EXTRA+=("runner.ckpt_path=$EVALW")
fi
if ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; then echo "another RLinf job is running; refusing to start" >&2; exit 3; fi
start=$(date +%s)
python evaluations/eval_embodied_agent.py --config-path /mnt/data1/faye/code/libero_rlinf_20261103/configs/ \
    --config-name calvin_abc_d_pi05_eval_2gpu "runner.logger.log_path=$LOG" \
    "env.eval.total_num_envs=$ENVS" "env.eval.max_steps_per_rollout_epoch=$STEPS" "${EXTRA[@]}" 2>&1 | tee "$LOG/console.log"
[ -n "${EVALW:-}" ] && rm -f "$EVALW"
echo "{\"tag\": \"$TAG\", \"ckpt\": \"$CKPT\", \"envs\": $ENVS, \"max_steps_per_rollout_epoch\": $STEPS, \"gpu\": \"all\", \"wall_seconds\": $(( $(date +%s) - start ))}" > "$LOG/wall.json"
