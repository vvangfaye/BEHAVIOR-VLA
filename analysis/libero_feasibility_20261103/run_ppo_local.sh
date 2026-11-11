#!/bin/bash
# Local LIBERO-10 PPO feasibility run (user 2026-11-03 "两个都做吧"): 10 iterations x 128 episodes,
# rollout+env on GPU 0, actor on GPU 1, RLinf PPO hyperparameters unchanged (lr 5e-6, 4 update epochs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:-ppo_local}"
REPO=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf
cd "$REPO"
source .venv/bin/activate
export EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}" HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false ROBOT_PLATFORM=LIBERO LIBERO_TYPE=standard
export RAY_TMPDIR=/mnt/data1/faye/ray_tmp_libero
# GPU0 and GPU1 sit on different NUMA nodes (nvidia-smi topo: SYS); NCCL P2P across them hung the
# actor->rollout weight sync for 22 min on 2026-11-03. Use shared-memory transport instead.
export NCCL_P2P_DISABLE=1
LOG="$HERE/results/$TAG"; mkdir -p "$LOG"
start=$(date +%s)
python examples/embodiment/train_embodied_agent.py --config-path /mnt/data1/faye/code/libero_rlinf_20261103/configs/ \
    --config-name libero_10_ppo_pi05_local2gpu "runner.logger.log_path=$LOG" \
    "env.train.video_cfg.save_video=False" "env.eval.video_cfg.save_video=False" 2>&1 | tee "$LOG/console.log"
echo "{\"tag\": \"$TAG\", \"wall_seconds\": $(( $(date +%s) - start ))}" > "$LOG/wall.json"
