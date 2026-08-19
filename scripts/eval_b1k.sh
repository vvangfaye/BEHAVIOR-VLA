#!/usr/bin/env bash
# Closed-loop evaluation of a trained policy in OmniGibson.
#
# Starts the model's websocket policy server, runs omnigibson.eval.eval against
# it for each task, then stops the server. Success rate per task is written under
# $OUTPUT_ROOT/<model>/<task>/.
#
# Usage:
#   MODEL=pi05  ./scripts/eval_b1k.sh turning_on_radio [more tasks...]
#   MODEL=groot ./scripts/eval_b1k.sh turning_on_radio
#   MODEL=pi05 NUM_ROLLOUTS=3 ./scripts/eval_b1k.sh <task>
#
# π0.5 and GR00T report different training losses (flow matching vs diffusion,
# different action normalisation), so only closed-loop success rate compares them.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:?set MODEL=pi05 or MODEL=groot}"
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
EVAL_ENV="${EVAL_ENV:-$ENV_ROOT/b1k-eval}"
PORT="${PORT:-8000}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
MODE="${MODE:-public_test}"
# Space-separated indices into the selected split's instance list. Empty means
# the evaluator's default (all 20 public_test instances), which is 20x the work.
INSTANCE_INDICES="${INSTANCE_INDICES:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/eval_results}"
OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-$REPO_ROOT/BEHAVIOR-1K/omnigibson-data}"
SERVER_GPU="${SERVER_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "Usage: MODEL=pi05|groot $0 <task-name> [task-name...]" >&2
    exit 2
fi

case "$MODEL" in
    pi05)
        PI05_CKPT="${PI05_CKPT:-$REPO_ROOT/openpi/outputs/checkpoints/pi05_b1k/pi05_b1k_full_20260803_162620/49999}"
        # Must match the assets dir written during training, otherwise the server
        # derives b1k/<task> and looks for norm stats in the wrong place.
        PI05_REPO_ID="${PI05_REPO_ID:-behavior-1k/2026-challenge-demos}"
        [[ -d "$PI05_CKPT" ]] || { echo "checkpoint not found: $PI05_CKPT" >&2; exit 1; }
        [[ -d "$PI05_CKPT/assets/$PI05_REPO_ID" ]] || {
            echo "norm stats not found: $PI05_CKPT/assets/$PI05_REPO_ID" >&2; exit 1; }
        ;;
    groot)
        GROOT_CKPT="${GROOT_CKPT:?set GROOT_CKPT to the checkpoint-N directory}"
        [[ -d "$GROOT_CKPT" ]] || { echo "checkpoint not found: $GROOT_CKPT" >&2; exit 1; }
        ;;
    *) echo "MODEL must be pi05 or groot" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_ROOT/$MODEL"
# Keyed by port so parallel workers sweeping disjoint task shards do not
# interleave their server output into one unreadable log.
SERVER_LOG="$OUTPUT_ROOT/$MODEL/server_$PORT.log"

activate_env() {
    # conda-pack's activate script tests $CONDA_PREFIX without a default, which
    # aborts under set -u when no environment is active yet.
    set +u
    # shellcheck disable=SC1091
    source "$1"
    set -u
}

start_server() {
    local task="$1"
    case "$MODEL" in
        pi05)
            cd "$REPO_ROOT/openpi"
            activate_env .venv/bin/activate
            # The prompt is bound at startup from TASK_REGISTRY, and --repo-id must
            # match the assets dir written during training.
            CUDA_VISIBLE_DEVICES="$SERVER_GPU" XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
            HF_HUB_OFFLINE=1 \
            python scripts/b1k/serve_b1k.py \
                --robot b1k/R1Pro \
                --task "b1k/$task" \
                --repo-id "$PI05_REPO_ID" \
                --policy.config pi05_b1k \
                --policy.dir "$PI05_CKPT" \
                --control_mode receding_horizon \
                --action_horizon 16 \
                --port "$PORT" >> "$SERVER_LOG" 2>&1 &
            ;;
        groot)
            cd "$REPO_ROOT/Isaac-GR00T"
            activate_env .venv/bin/activate
            CUDA_VISIBLE_DEVICES="$SERVER_GPU" \
            HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
            HF_HOME=/mnt/workspace/wangfaye/cache/huggingface \
            HF_HUB_OFFLINE=1 \
            python scripts/b1k/serve_b1k.py \
                --model-path "$GROOT_CKPT" \
                --modality-config-path examples/b1k/r1pro.py \
                --embodiment-tag NEW_EMBODIMENT \
                --control_mode receeding_horizon \
                --host 127.0.0.1 \
                --port "$PORT" >> "$SERVER_LOG" 2>&1 &
            ;;
    esac
    SERVER_PID=$!
    echo "policy server pid=$SERVER_PID task=$task log=$SERVER_LOG"
}

stop_server() {
    [[ -n "${SERVER_PID:-}" ]] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$SERVER_PID" 2>/dev/null || { SERVER_PID=""; return 0; }; sleep 1; done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap stop_server EXIT

wait_for_server() {
    echo "waiting for policy server on port $PORT"
    for i in $(seq 1 900); do
        # Both servers speak websocket, so probe the listening socket rather than HTTP.
        if "$EVAL_ENV/bin/python" - "$PORT" <<'PY' >/dev/null 2>&1
import socket, sys
s = socket.socket()
s.settimeout(2)
sys.exit(0 if s.connect_ex(("127.0.0.1", int(sys.argv[1]))) == 0 else 1)
PY
        then
            echo "server accepting connections after ${i}s"; return 0
        fi
        kill -0 "$SERVER_PID" 2>/dev/null || { echo "server died; see $SERVER_LOG" >&2; tail -25 "$SERVER_LOG" >&2; return 1; }
        sleep 1
    done
    echo "server did not open port $PORT in time; see $SERVER_LOG" >&2
    tail -25 "$SERVER_LOG" >&2
    return 1
}

for task in "${TASKS[@]}"; do
    out="$OUTPUT_ROOT/$MODEL/$task"
    mkdir -p "$out"
    echo "=== evaluating $MODEL on $task ==="
    start_server "$task"
    if ! wait_for_server; then
        stop_server
        echo "SKIP $task (server failed to start)" | tee -a "$OUTPUT_ROOT/$MODEL/failures.txt"
        continue
    fi
    cd "$REPO_ROOT/BEHAVIOR-1K"
    EVAL_ARGS=()
    if [[ -n "$INSTANCE_INDICES" ]]; then
        # shellcheck disable=SC2086
        EVAL_ARGS+=(--instance-indices $INSTANCE_INDICES)
    fi
    # This runs the eval interpreter directly instead of sourcing the env's
    # activation hook, so the hook's graphics environment has to be passed here:
    # Isaac Sim needs the Vulkan loader and libGLU from the env's lib directory
    # (prepended, since start_server already sourced the policy env into this
    # shell), the ICD manifests written by setup_eval_env.sh, and the EULA flag
    # that Kit re-checks on every launch.
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
            --write-video 2>&1 | tee "$out/eval.log" || echo "eval failed for $task" | tee -a "$OUTPUT_ROOT/$MODEL/failures.txt"
    stop_server
done

echo "results under $OUTPUT_ROOT/$MODEL"
