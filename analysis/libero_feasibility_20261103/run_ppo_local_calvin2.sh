#!/bin/bash
# Local CALVIN ABC->D PPO run (2026-11-03): 10 iterations x 128 chains, env+rollout on GPU 0, actor on GPU 1,
# RLinf PPO hyperparameters unchanged. Extra Hydra overrides (e.g. +algorithm.length_normalization=False) go after the tag.
# CONFIG=calvin_abc_d_ppo_pi05_local2gpu_all selects the colocated layout.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TAG="${1:?tag}"; shift
REPO=/mnt/data1/faye/code/libero_rlinf_20261103/RLinf
cd "$REPO"
source .venv-calvin/bin/activate
export EMBODIED_PATH="$REPO/examples/embodiment" PYTHONPATH="$REPO:${PYTHONPATH:-}" HYDRA_FULL_ERROR=1
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR=/mnt/data1/faye/ray_tmp_libero
# GPU0 and GPU1 sit on different NUMA nodes (nvidia-smi topo: SYS); NCCL P2P across them hung the
# actor->rollout weight sync for 22 min on 2026-11-03. Use shared-memory transport instead.
export NCCL_P2P_DISABLE=1
# 2026-11-04: ESPO training peaked at ~47/48 GB per GPU (allocator fragmentation); expandable segments reduce it.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG="$HERE/results/$TAG"; mkdir -p "$LOG"
CFG="${CONFIG:-calvin_abc_d_ppo_pi05_local2gpu}"
printf '{"tag": "%s", "config": "%s", "overrides": "%s", "start": "%s", "rlinf_head": "%s", "rlinf_diff_sha256_16": "%s"}\n' "$TAG" "$CFG" "$*" "$(date -Iseconds)" "$(git -C "$REPO" rev-parse --short HEAD)" "$(git -C "$REPO" diff | sha256sum | cut -c1-16)" > "$LOG/run.json"
start=$(date +%s)
python examples/embodiment/train_embodied_agent.py --config-path /mnt/data1/faye/code/libero_rlinf_20261103/configs/ \
    --config-name "$CFG" "runner.logger.log_path=$LOG" \
    "$@" 2>&1 | tee "$LOG/console.log"
echo "{\"tag\": \"$TAG\", \"wall_seconds\": $(( $(date +%s) - start ))}" > "$LOG/wall.json"
