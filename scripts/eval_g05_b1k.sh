#!/usr/bin/env bash
# Closed-loop BEHAVIOR-1K evaluation of the G05 (Galaxea OpenGalaxea G0.5)
# policy in OmniGibson.
#
# Pattern follows scripts/eval_b1k.sh (pi05/gr00t):
#   1. start the G05 msgpack WebSocket server on a private upstream port,
#   2. start g05_policy_shim.py on PORT (8090) bridging OmniGibson ↔ G05,
#   3. wait for shim's /healthz to return 200,
#   4. run the OmniGibson evaluator against the shim,
#   5. tear both servers down on EXIT.
#
# The shim is required because:
#   - OmniGibson's WebsocketClientPolicy polls GET /healthz and expects a
#     flat `{"action": ndarray}` reply.
#   - serve_policy_mem.py speaks a pure msgpack WebSocket with per-camera /
#     per-part dict payloads, no /healthz route, and per-part action.
# scripts/g05_policy_shim.py bridges the two protocols.
#
# Usage:
#   ./scripts/eval_g05_b1k.sh turning_on_radio
#   NUM_ROLLOUTS=2 INSTANCE_INDICES="0 1" ./scripts/eval_g05_b1k.sh turning_on_radio
#
# Override everything that scripts/eval_b1k.sh exposes plus G05-specific knobs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GALAXEA_REPO="${GALAXEA_REPO:-$REPO_ROOT/GalaxeaVLA}"
GALAXEA_VENV="${GALAXEA_VENV:-$GALAXEA_REPO/.venv}"
G05_CKPT="${G05_CKPT:-/mnt/workspace/wangfaye/cache/huggingface/hub/OpenGalaxea/G05/checkpoints/model_state_dict.pt}"
EVAL_ENV="${EVAL_ENV:-/dev/shm/wangfaye/conda_envs/b1k-eval}"
PORT="${PORT:-8090}"                              # shim port (what OmniGibson connects to)
UPSTREAM_PORT="${UPSTREAM_PORT:-8765}"            # G05 server port (private)
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
MODE="${MODE:-public_test}"
INSTANCE_INDICES="${INSTANCE_INDICES:-}"        # empty = evaluator default
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/eval_results}"
OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-$REPO_ROOT/BEHAVIOR-1K/omnigibson-data}"
SERVER_GPU="${SERVER_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
ACTION_STEPS="${ACTION_STEPS:-16}"
EMBODIMENT="${EMBODIMENT:-galaxea_r1pro}"

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "Usage: $0 <task-name> [task-name...]" >&2
    exit 2
fi

if [[ ! -d "$GALAXEA_VENV" ]]; then
    echo "venv not found: $GALAXEA_VENV" >&2
    echo "Run 'uv sync' inside $GALAXEA_REPO first." >&2
    exit 1
fi
if [[ ! -f "$G05_CKPT" ]]; then
    echo "checkpoint not found: $G05_CKPT" >&2
    exit 1
fi
if [[ ! -x "$EVAL_ENV/bin/python" ]]; then
    echo "evaluator env not found: $EVAL_ENV" >&2
    echo "Set EVAL_ENV=... to the conda env that has the BEHAVIOR-1K evaluator." >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/g05"
SERVER_LOG="$OUTPUT_ROOT/g05/server_${UPSTREAM_PORT}.log"
SHIM_LOG="$OUTPUT_ROOT/g05/shim_${PORT}.log"

stop_servers() {
    [[ -n "${SHIM_PID:-}" ]] && kill -TERM "$SHIM_PID" 2>/dev/null || true
    [[ -n "${SERVER_PID:-}" ]] && kill -TERM "$SERVER_PID" 2>/dev/null || true
    for pid in "$SHIM_PID" "$SERVER_PID"; do
        [[ -n "$pid" ]] || continue
        for _ in $(seq 1 20); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -KILL "$pid" 2>/dev/null || true
    done
    SHIM_PID=""
    SERVER_PID=""
}
trap stop_servers EXIT

start_server() {
    # shellcheck disable=SC1091
    source "$GALAXEA_VENV/bin/activate"
    export PYTHONPATH="$GALAXEA_REPO/src:$GALAXEA_REPO:${PYTHONPATH:-}"
    export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
    export HYDRA_FULL_ERROR=1

    cd "$GALAXEA_REPO"
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
    python scripts/serve_policy_mem.py \
        --ckpt_path "$G05_CKPT" \
        --host 127.0.0.1 --port "$UPSTREAM_PORT" --device cuda \
        --action_steps "$ACTION_STEPS" \
        eval_embodiment="$EMBODIMENT" \
        model.model_weights_to_bf16=true \
        model.use_torch_compile=true \
        model.model_arch.attn_implementation=sdpa \
        model.model_arch.discrete_action=true \
        model.model_arch.continuous_action=false \
        >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "G05 server pid=$SERVER_PID port=$UPSTREAM_PORT log=$SERVER_LOG"
}

start_shim() {
    # Same venv as G05 (needs g05.utils.websocket for the numpy-aware msgpack
    # codec). The shim does NOT need GPU; runs on CPU and only does HTTP + WS.
    # shellcheck disable=SC1091
    source "$GALAXEA_VENV/bin/activate"
    export PYTHONPATH="$REPO_ROOT/scripts:$GALAXEA_REPO/src:$GALAXEA_REPO:${PYTHONPATH:-}"

    cd "$REPO_ROOT"
    python scripts/g05_policy_shim.py \
        --listen-host 127.0.0.1 --listen-port "$PORT" \
        --upstream-host 127.0.0.1 --upstream-port "$UPSTREAM_PORT" \
        --task "$TASKS[0]" \
        --embodiment "$EMBODIMENT" \
        >> "$SHIM_LOG" 2>&1 &
    SHIM_PID=$!
    echo "shim pid=$SHIM_PID port=$PORT log=$SHIM_LOG"
}

wait_for_shim() {
    echo "waiting for shim /healthz on port $PORT"
    for i in $(seq 1 600); do
        if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
            echo "shim healthy after ${i}s"; return 0
        fi
        kill -0 "$SHIM_PID" 2>/dev/null || {
            echo "shim died; see $SHIM_LOG" >&2; tail -25 "$SHIM_LOG" >&2; return 1; }
        kill -0 "$SERVER_PID" 2>/dev/null || {
            echo "G05 server died; see $SERVER_LOG" >&2; tail -25 "$SERVER_LOG" >&2; return 1; }
        sleep 1
    done
    echo "shim did not respond on port $PORT in time; see $SHIM_LOG" >&2
    tail -25 "$SHIM_LOG" >&2
    return 1
}

for task in "${TASKS[@]}"; do
    out="$OUTPUT_ROOT/g05/$task"
    mkdir -p "$out"
    echo "=== evaluating G05 on $task ==="
    start_server
    start_shim
    if ! wait_for_shim; then
        stop_servers
        echo "SKIP $task (shim failed to start)" | tee -a "$OUTPUT_ROOT/g05/failures.txt"
        continue
    fi

    cd "$REPO_ROOT/BEHAVIOR-1K"
    EVAL_ARGS=()
    if [[ -n "$INSTANCE_INDICES" ]]; then
        # shellcheck disable=SC2086
        EVAL_ARGS+=(--instance-indices $INSTANCE_INDICES)
    fi

    env OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH" \
        CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
        HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
        PYTHONNOUSERSITE=1 \
        OMNI_KIT_ACCEPT_EULA=YES \
        LD_LIBRARY_PATH="$EVAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        VK_ICD_FILENAMES="$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json" \
        __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_ENV/share/glvnd/egl_vendor.d/10_nvidia.json" \
        "$EVAL_ENV/bin/python" -m omnigibson.eval.eval \
            --task-name "$task" \
            --host 127.0.0.1 --port "$PORT" \
            --mode "$MODE" \
            --num-rollouts "$NUM_ROLLOUTS" \
            "${EVAL_ARGS[@]}" \
            --output-dir "$out" \
            --write-video 2>&1 | tee "$out/eval.log" || echo "eval failed for $task" | tee -a "$OUTPUT_ROOT/g05/failures.txt"
    stop_servers
done

echo "results under $OUTPUT_ROOT/g05"