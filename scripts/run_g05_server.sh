#!/usr/bin/env bash
# Standalone G05 (Galaxea OpenGalaxea G0.5) policy server for the BEHAVIOR-1K
# OmniGibson evaluator.
#
# Mirrors the pattern in scripts/eval_b1k.sh (pi05/gr00t) but starts the G05
# WebSocket server used by the GalaxeaVLA repo. The server speaks a different
# wire protocol than the openpi/Isaac-GR00T servers the OmniGibson evaluator
# expects (see docs/g05_eval.md), so until an adapter is written the evaluator
# will fail at the first /healthz handshake -- this script still launches the
# server cleanly for ad-hoc inspection (websocat, etc.).
#
# Defaults:
#   GALAXEA_REPO   /mnt/workspace/wangfaye/code/BEHAVIOR-Challenge/GalaxeaVLA
#   GALAXEA_VENV   $GALAXEA_REPO/.venv           (run `uv sync` there first)
#   G05_CKPT       /mnt/workspace/wangfaye/cache/huggingface/hub/OpenGalaxea/G05/...
#   PORT           8090                          (avoids pi05's 8000)
#   ACTION_STEPS   16                            (matches R1 Pro deployment)
#
# Usage:
#   ./scripts/run_g05_server.sh
#   G05_CKPT=/path/to/other.pt PORT=8091 ./scripts/run_g05_server.sh
#
# Stop with Ctrl-C; a SIGTERM is forwarded to the server pid.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GALAXEA_REPO="${GALAXEA_REPO:-$REPO_ROOT/GalaxeaVLA}"
GALAXEA_VENV="${GALAXEA_VENV:-$GALAXEA_REPO/.venv}"
G05_CKPT="${G05_CKPT:-/mnt/workspace/wangfaye/cache/huggingface/hub/OpenGalaxea/G05/checkpoints/model_state_dict.pt}"
PORT="${PORT:-8090}"
HOST="${HOST:-127.0.0.1}"
DEVICE="${DEVICE:-cuda}"
ACTION_STEPS="${ACTION_STEPS:-16}"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/eval_results/g05}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/server_$PORT.log}"

if [[ ! -d "$GALAXEA_VENV" ]]; then
    echo "venv not found: $GALAXEA_VENV" >&2
    echo "Run 'uv sync' inside $GALAXEA_REPO first." >&2
    exit 1
fi
if [[ ! -f "$G05_CKPT" ]]; then
    echo "checkpoint not found: $G05_CKPT" >&2
    echo "Wait for the OpenGalaxea/G05 download to complete, or set G05_CKPT=..." >&2
    exit 1
fi

mkdir -p "$LOG_DIR"

# shellcheck disable=SC1091
source "$GALAXEA_VENV/bin/activate"

# start.sh adds PYTHONPATH, mirrors, ROS2 etc. We only need PYTHONPATH for the
# server-side g05 import; skip ROS2 source (it is only required for the r1pro
# client, not the inference server).
export PYTHONPATH="$GALAXEA_REPO/src:$GALAXEA_REPO:${PYTHONPATH:-}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "G05 server starting"
echo "  repo        = $GALAXEA_REPO"
echo "  venv        = $GALAXEA_VENV"
echo "  ckpt        = $G05_CKPT"
echo "  url         = ws://$HOST:$PORT"
echo "  log         = $LOG_FILE"

cd "$GALAXEA_REPO"
python scripts/serve_policy_mem.py \
    --ckpt_path "$G05_CKPT" \
    --host "$HOST" --port "$PORT" --device "$DEVICE" \
    --action_steps "$ACTION_STEPS" \
    eval_embodiment=galaxea_r1pro \
    model.model_weights_to_bf16=true \
    model.use_torch_compile=true \
    model.model_arch.attn_implementation=sdpa \
    model.model_arch.discrete_action=true \
    model.model_arch.continuous_action=false \
    >> "$LOG_FILE" 2>&1