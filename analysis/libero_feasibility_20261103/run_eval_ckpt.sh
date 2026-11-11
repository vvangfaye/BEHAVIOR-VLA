#!/bin/bash
# LIBERO-10 evaluation of an RL checkpoint (or the SFT start when CKPT=none) on ONE local GPU, same protocol as
# results/full500 (configs/libero_10_pi05_sft_gpu0.yaml: 50 envs x 10 epochs = all 500 init states, 520 steps).
# Usage: bash run_eval_ckpt.sh <tag> <path/to/full_weights.pt|none> [gpu=0|all] [envs=50] [epochs=10]
# EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10" evaluates with replan horizon 10 (piRL's LIBERO-Long H'=10).
# gpu=all uses configs/libero_10_pi05_eval_2gpu.yaml (both GPUs). Never run two RLinf jobs at once: RLinf calls
# ray.init(address='auto'), which attaches to ANY running Ray cluster on the machine.
# RL checkpoints contain the value head; a stripped copy is evaluated (the eval policy has none, load is strict).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="$1"; CKPT="$2"; GPU="${3:-0}"; ENVS="${4:-50}"; EPOCHS="${5:-10}"
REPO=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf
cd "$REPO"
source .venv/bin/activate
CFGNAME=libero_10_pi05_sft_gpu0
if [ "$GPU" = "all" ]; then CFGNAME=libero_10_pi05_eval_2gpu; else export CUDA_VISIBLE_DEVICES="$GPU"; fi
export EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}" HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR=/mnt/data1/faye/ray_tmp_eval_gpu$GPU
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
    --config-name "$CFGNAME" "runner.logger.log_path=$LOG" \
    "env.eval.total_num_envs=$ENVS" "env.eval.rollout_epoch=$EPOCHS" "${EXTRA[@]}" ${EVAL_EXTRA_OVERRIDES:-} 2>&1 | tee "$LOG/console.log"
[ -n "${EVALW:-}" ] && rm -f "$EVALW"
echo "{\"tag\": \"$TAG\", \"ckpt\": \"$CKPT\", \"envs\": $ENVS, \"rollout_epoch\": $EPOCHS, \"gpu\": \"$GPU\", \"wall_seconds\": $(( $(date +%s) - start ))}" > "$LOG/wall.json"
