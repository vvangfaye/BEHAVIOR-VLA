#!/usr/bin/env bash
# Closed-loop evaluation of π0.5 in OmniGibson.
# Starts the policy server, evaluates each task, then stops the server.
# Usage: PI05_CKPT=/path/to/checkpoint bash scripts/eval_b1k.sh turning_on_radio
# For joint subtask checkpoints, set SUBTASK_PREDICTION=1; for action-only
# subtask checkpoints, set PROVIDED_SUBTASK_AS_PROMPT=1 with an external subtask source.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${MODEL:-pi05}"
[[ "$MODEL" == pi05 ]] || { echo "Only MODEL=pi05 is supported" >&2; exit 2; }
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
EVAL_ENV="${EVAL_ENV:-$ENV_ROOT/b1k-eval}"
PORT="${PORT:-8000}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
MODE="${MODE:-public_test}"
PROMPT_MODE="${PROMPT_MODE:-whole_task}"
EXTERNAL_SUBTASK_SET="${EXTERNAL_SUBTASK+x}"
EXTERNAL_SUBTASK="${EXTERNAL_SUBTASK:-}"
EXTERNAL_SUBTASK_SELECTOR="${EXTERNAL_SUBTASK_SELECTOR:-}"
SUBTASK_SAMPLE_PANEL="${PI05_SUBTASK_SAMPLE_PANEL:-}"
SUBTASK_RAW_DATA_ROOT="${PI05_SUBTASK_RAW_DATA_ROOT:-}"
SUBTASK_SAMPLE_MODE=0
if [[ -n "$SUBTASK_SAMPLE_PANEL" ]]; then
    SUBTASK_SAMPLE_MODE=1
    : "${SUBTASK_RAW_DATA_ROOT:?set PI05_SUBTASK_RAW_DATA_ROOT with PI05_SUBTASK_SAMPLE_PANEL}"
fi
case "$PROMPT_MODE" in
    whole_task|oracle_unsatisfied_goal) ;;
    *) echo "PROMPT_MODE must be whole_task or oracle_unsatisfied_goal" >&2; exit 2 ;;
esac
# Space-separated indices into the selected split's instance list. Empty means
# the evaluator's default (all 20 public_test instances), which is 20x the work.
INSTANCE_INDICES="${INSTANCE_INDICES:-}"
MAX_STEPS="${MAX_STEPS:-}"
if [[ -n "$MAX_STEPS" && ( ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ ) ]]; then
    echo "MAX_STEPS must be a positive integer" >&2
    exit 2
fi
WRITE_VIDEO="${WRITE_VIDEO:-1}"
case "$WRITE_VIDEO" in
    0) VIDEO_ARG="--no-write-video" ;;
    1) VIDEO_ARG="--write-video" ;;
    *) echo "WRITE_VIDEO must be 0 or 1" >&2; exit 2 ;;
esac
if [[ -z "${OUTPUT_ROOT:-}" ]]; then
    if [[ "$PROMPT_MODE" == "oracle_unsatisfied_goal" ]]; then
        OUTPUT_ROOT="$REPO_ROOT/eval_results/oracle_unsatisfied_goal"
    else
        OUTPUT_ROOT="$REPO_ROOT/eval_results"
    fi
fi
EVAL_START_EPOCH="$(date +%s)"
OMNIGIBSON_DATA_PATH="${OMNIGIBSON_DATA_PATH:-$REPO_ROOT/BEHAVIOR-1K/omnigibson-data}"
SERVER_GPU="${SERVER_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-1}"
PI05_MEM_FRACTION="${PI05_MEM_FRACTION:-0.85}"
SUBTASK_PREDICTION="${SUBTASK_PREDICTION:-0}"
PROVIDED_SUBTASK_AS_PROMPT="${PROVIDED_SUBTASK_AS_PROMPT:-0}"
if [[ "$SUBTASK_PREDICTION" != 0 && "$SUBTASK_PREDICTION" != 1 ]]; then
    echo "SUBTASK_PREDICTION must be 0 or 1" >&2; exit 2
fi
if [[ "$PROVIDED_SUBTASK_AS_PROMPT" != 0 && "$PROVIDED_SUBTASK_AS_PROMPT" != 1 ]]; then
    echo "PROVIDED_SUBTASK_AS_PROMPT must be 0 or 1" >&2; exit 2
fi
if [[ "$SUBTASK_PREDICTION" == 1 && "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    echo "SUBTASK_PREDICTION and PROVIDED_SUBTASK_AS_PROMPT are mutually exclusive" >&2; exit 2
fi
EXTERNAL_SUBTASK_ARGS=()
if (( SUBTASK_SAMPLE_MODE )) && [[ -n "$EXTERNAL_SUBTASK_SET" || -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    echo "PI05_SUBTASK_SAMPLE_PANEL supplies per-sample prompts and cannot be combined with EXTERNAL_SUBTASK or EXTERNAL_SUBTASK_SELECTOR" >&2
    exit 2
fi
if [[ -n "$EXTERNAL_SUBTASK_SET" ]]; then
    if [[ ! "$EXTERNAL_SUBTASK" =~ [^[:space:]] ]]; then
        echo "EXTERNAL_SUBTASK must be a non-empty, non-whitespace string" >&2; exit 2
    fi
    if [[ "$SUBTASK_PREDICTION" != 1 && "$PROVIDED_SUBTASK_AS_PROMPT" != 1 ]]; then
        echo "EXTERNAL_SUBTASK requires a subtask conditioning mode" >&2; exit 2
    fi
    if [[ "$PROMPT_MODE" != whole_task ]]; then
        echo "EXTERNAL_SUBTASK requires PROMPT_MODE=whole_task" >&2; exit 2
    fi
    EXTERNAL_SUBTASK_ARGS=(--external-subtask "$EXTERNAL_SUBTASK")
fi
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    if [[ -n "$EXTERNAL_SUBTASK_SET" ]]; then
        echo "EXTERNAL_SUBTASK and EXTERNAL_SUBTASK_SELECTOR are mutually exclusive" >&2; exit 2
    fi
    if [[ "$EXTERNAL_SUBTASK_SELECTOR" != state_machine_v1 ]]; then
        echo "EXTERNAL_SUBTASK_SELECTOR must be state_machine_v1" >&2; exit 2
    fi
    if [[ "$SUBTASK_PREDICTION" != 1 && "$PROVIDED_SUBTASK_AS_PROMPT" != 1 ]]; then
        echo "EXTERNAL_SUBTASK_SELECTOR requires a subtask conditioning mode" >&2; exit 2
    fi
    if [[ "$PROMPT_MODE" != whole_task ]]; then
        echo "EXTERNAL_SUBTASK_SELECTOR requires PROMPT_MODE=whole_task" >&2; exit 2
    fi
    EXTERNAL_SUBTASK_ARGS=(--external-subtask-selector "$EXTERNAL_SUBTASK_SELECTOR")
fi
SUBTASK_SERVER_ARGS=()
if [[ "$SUBTASK_PREDICTION" == 1 ]]; then
    : "${PI05_CKPT:?set PI05_CKPT to your joint subtask-trained checkpoint}"
    SUBTASK_SERVER_ARGS=(--subtask-prediction --max-token-len "${MAX_TOKEN_LEN:-512}"
        --max-subtask-len "${MAX_SUBTASK_LEN:-64}"
        --subtask-temperature "${SUBTASK_TEMPERATURE:-0.0}")
    if [[ -n "${SUBTASK_INTERVAL:-}" ]]; then
        SUBTASK_SERVER_ARGS+=(--subtask-interval "$SUBTASK_INTERVAL")
    fi
    if [[ -n "${SUBTASK_ACTION_CONDITIONING:-}" ]]; then
        SUBTASK_SERVER_ARGS+=(--subtask-action-conditioning "$SUBTASK_ACTION_CONDITIONING")
    fi
elif [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    if [[ -z "$EXTERNAL_SUBTASK_SET" && -z "$EXTERNAL_SUBTASK_SELECTOR" && "$SUBTASK_SAMPLE_MODE" == 0 ]]; then
        echo "PROVIDED_SUBTASK_AS_PROMPT requires EXTERNAL_SUBTASK, EXTERNAL_SUBTASK_SELECTOR, or PI05_SUBTASK_SAMPLE_PANEL" >&2; exit 2
    fi
    if [[ "${MAX_TOKEN_LEN:-200}" != 200 ]]; then
        echo "PROVIDED_SUBTASK_AS_PROMPT requires MAX_TOKEN_LEN=200" >&2; exit 2
    fi
    SUBTASK_SERVER_ARGS=(--provided-subtask-as-prompt --max-token-len 200)
fi
EVAL_RUNTIME_ROOT="${EVAL_RUNTIME_ROOT:-/dev/shm/wangfaye/omnigibson-runtime/$PORT}"
QUEUE_MODE=0
QUEUE_EVAL_ARGS=()
if [[ -n "${PI05_EVAL_QUEUE_DB:-}" ]]; then
    QUEUE_MODE=1
    : "${PI05_EVAL_QUEUE_WORKER:?set PI05_EVAL_QUEUE_WORKER with PI05_EVAL_QUEUE_DB}"
    : "${PI05_EVAL_QUEUE_RUN_ID:?set PI05_EVAL_QUEUE_RUN_ID with PI05_EVAL_QUEUE_DB}"
    : "${PI05_EVAL_QUEUE_GENERATION:?set PI05_EVAL_QUEUE_GENERATION with PI05_EVAL_QUEUE_DB}"
    : "${PI05_EVAL_QUEUE_ATTEMPT_ROOT:?set PI05_EVAL_QUEUE_ATTEMPT_ROOT with PI05_EVAL_QUEUE_DB}"
    [[ "$PI05_EVAL_QUEUE_GENERATION" =~ ^[0-9]+$ ]] || {
        echo "PI05_EVAL_QUEUE_GENERATION must be a non-negative integer" >&2; exit 2; }
    QUEUE_EVAL_ARGS=(
        --queue-db "$PI05_EVAL_QUEUE_DB"
        --queue-worker "$PI05_EVAL_QUEUE_WORKER"
        --queue-run-id "$PI05_EVAL_QUEUE_RUN_ID"
        --queue-generation "$PI05_EVAL_QUEUE_GENERATION"
        --queue-attempt-root "$PI05_EVAL_QUEUE_ATTEMPT_ROOT"
    )
    QUEUE_LOG_DIR="${PI05_EVAL_QUEUE_LOG_ROOT:-$PI05_EVAL_QUEUE_ATTEMPT_ROOT/../logs}"
    QUEUE_LOG_DIR="$QUEUE_LOG_DIR/$PI05_EVAL_QUEUE_WORKER/generation-$PI05_EVAL_QUEUE_GENERATION"
    mkdir -p "$QUEUE_LOG_DIR"
fi
if (( SUBTASK_SAMPLE_MODE )); then
    (( QUEUE_MODE )) || {
        echo "PI05_SUBTASK_SAMPLE_PANEL requires PI05_EVAL_QUEUE_DB" >&2; exit 2; }
    [[ "$NUM_ROLLOUTS" == 1 ]] || {
        echo "PI05_SUBTASK_SAMPLE_PANEL requires NUM_ROLLOUTS=1" >&2; exit 2; }
    [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]] || {
        echo "PI05_SUBTASK_SAMPLE_PANEL requires PROVIDED_SUBTASK_AS_PROMPT=1" >&2; exit 2; }
    [[ "${SUBTASK_SAMPLE_MAX_STEPS:-2048}" =~ ^[1-9][0-9]*$ ]] || {
        echo "SUBTASK_SAMPLE_MAX_STEPS must be a positive integer" >&2; exit 2; }
fi

TASKS=("$@")
if [[ ${#TASKS[@]} -eq 0 ]]; then
    echo "Usage: MODEL=pi05 $0 <task-name> [task-name...]" >&2
    exit 2
fi
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    for task in "${TASKS[@]}"; do
        case "$task" in
            carrying_in_groceries|make_rose_centerpieces) ;;
            *)
                echo "EXTERNAL_SUBTASK_SELECTOR only supports carrying_in_groceries and make_rose_centerpieces" >&2
                exit 2
                ;;
        esac
    done
fi

PI05_CKPT="${PI05_CKPT:-$REPO_ROOT/openpi/outputs/checkpoints/pi05_b1k/pi05_b1k_full_20260803_162620/49999}"
# Must match the assets dir written during training, otherwise the server
# derives b1k/<task> and looks for norm stats in the wrong place.
PI05_REPO_ID="${PI05_REPO_ID:-behavior-1k/2026-challenge-demos}"
[[ -d "$PI05_CKPT" ]] || { echo "checkpoint not found: $PI05_CKPT" >&2; exit 1; }
[[ -d "$PI05_CKPT/assets/$PI05_REPO_ID" ]] || {
    echo "norm stats not found: $PI05_CKPT/assets/$PI05_REPO_ID" >&2; exit 1; }
if [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 && -f "$PI05_CKPT/model.safetensors" ]]; then
    echo "PROVIDED_SUBTASK_AS_PROMPT currently supports JAX checkpoints only" >&2; exit 2
fi
if [[ -f "$PI05_CKPT/model.safetensors" ]]; then
    PI05_ENV="${PI05_ENV:-$REPO_ROOT/openpi/.venv-torch}"
    export JAX_PLATFORMS=cpu
else
    PI05_ENV="${PI05_ENV:-$REPO_ROOT/openpi/.venv}"
fi
for name in PI05_CKPT PI05_ENV; do
    value="${!name}"
    if [[ "$value" != /* ]]; then printf -v "$name" '%s/%s' "$PWD" "$value"; fi
done

mkdir -p "$OUTPUT_ROOT/$MODEL"
echo "prompt mode: $PROMPT_MODE"
if [[ "$PROMPT_MODE" == "oracle_unsatisfied_goal" ]]; then
    echo "WARNING: using privileged simulator BDDL goal state; results are not comparable to whole-task conditioning"
fi
# Keyed by port so parallel workers sweeping disjoint task shards do not
# interleave their server output into one unreadable log.
if (( QUEUE_MODE )); then
    SERVER_LOG="$QUEUE_LOG_DIR/server_$PORT.log"
else
    SERVER_LOG="$OUTPUT_ROOT/$MODEL/server_$PORT.log"
fi

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
    cd "$REPO_ROOT/openpi"
    activate_env "$PI05_ENV/bin/activate"
    if [[ -f "$PI05_CKPT/model.safetensors" ]]; then
        python scripts/install_transformers_patch.py --check
    fi
    # The startup prompt is the fallback when the evaluator does not send
    # a dynamic oracle prompt; --repo-id must match the training assets.
    CUDA_VISIBLE_DEVICES="$SERVER_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION="$PI05_MEM_FRACTION" HF_HUB_OFFLINE=1 \
    python scripts/b1k/serve_b1k.py \
        --robot b1k/R1Pro \
        --task "b1k/$task" \
        --repo-id "$PI05_REPO_ID" \
        --policy.config pi05_b1k \
        --policy.dir "$PI05_CKPT" \
        --control_mode receding_horizon \
        --action_horizon "${EXECUTE_HORIZON:-16}" \
        "${SUBTASK_SERVER_ARGS[@]}" \
        --port "$PORT" >> "$SERVER_LOG" 2>&1 &
    SERVER_PID=$!
    echo "policy server pid=$SERVER_PID task=$task log=$SERVER_LOG"
}

stop_server() {
    [[ -n "${SERVER_PID:-}" ]] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    for _ in $(seq 1 20); do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            return 0
        fi
        sleep 1
    done
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap stop_server EXIT

wait_for_server() {
    echo "waiting for policy server on port $PORT"
    for i in $(seq 1 900); do
        # The policy server speaks WebSocket; probe the listening socket.
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

status=0
if (( QUEUE_MODE )); then
    FAILURES_LOG="$QUEUE_LOG_DIR/failures.txt"
else
    FAILURES_LOG="$OUTPUT_ROOT/$MODEL/failures.txt"
fi
for task in "${TASKS[@]}"; do
    out="$OUTPUT_ROOT/$MODEL/$task"
    mkdir -p "$out"
    if (( QUEUE_MODE )); then EVAL_LOG="$QUEUE_LOG_DIR/eval_$task.log"; else EVAL_LOG="$out/eval.log"; fi
    echo "=== evaluating $MODEL on $task ==="
    start_server "$task"
    if ! wait_for_server; then
        stop_server
        echo "FAIL $task (server failed to start)" | tee -a "$FAILURES_LOG"
        status=1
        if (( QUEUE_MODE )); then break; else continue; fi
    fi
    cd "$REPO_ROOT/BEHAVIOR-1K"
    EVAL_ARGS=("${EXTERNAL_SUBTASK_ARGS[@]}")
    EVAL_COMMAND=("$EVAL_ENV/bin/python" -m omnigibson.eval.eval)
    if (( SUBTASK_SAMPLE_MODE )); then
        EVAL_ARGS+=(
            "${QUEUE_EVAL_ARGS[@]}"
            --subtask-panel "$SUBTASK_SAMPLE_PANEL"
            --raw-data-root "$SUBTASK_RAW_DATA_ROOT"
            --sample-max-steps "${SUBTASK_SAMPLE_MAX_STEPS:-2048}"
        )
        EVAL_COMMAND=("$EVAL_ENV/bin/python" "$REPO_ROOT/scripts/eval_b1k_subtask_samples.py")
    elif (( QUEUE_MODE )); then
        EVAL_ARGS+=("${QUEUE_EVAL_ARGS[@]}")
        EVAL_COMMAND=("$EVAL_ENV/bin/python" "$REPO_ROOT/scripts/eval_b1k_queue.py")
    fi
    if [[ -n "$INSTANCE_INDICES" ]]; then
        # shellcheck disable=SC2086
        EVAL_ARGS+=(--instance-indices $INSTANCE_INDICES)
    fi
    if [[ -n "$MAX_STEPS" ]]; then
        EVAL_ARGS+=(--max-steps "$MAX_STEPS")
    fi
    # This runs the eval interpreter directly instead of sourcing the env's
    # activation hook, so the hook's graphics environment has to be passed here:
    # Isaac Sim needs the Vulkan loader and libGLU from the env's lib directory
    # (prepended, since start_server already sourced the policy env into this
    # shell), the ICD manifests written by setup_eval_env.sh, and the EULA flag
    # that Kit re-checks on every launch.
    mkdir -p "$EVAL_RUNTIME_ROOT"/{appdata,cache,config,warp}
    run_evaluator() {
        env OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH" \
            OMNIGIBSON_APPDATA_PATH="$EVAL_RUNTIME_ROOT/appdata" \
            OMNIGIBSON_GPU_ID="$EVAL_GPU" \
            XDG_CACHE_HOME="$EVAL_RUNTIME_ROOT/cache" \
            XDG_CONFIG_HOME="$EVAL_RUNTIME_ROOT/config" \
            WARP_CACHE_PATH="$EVAL_RUNTIME_ROOT/warp" \
            HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
            PYTHONNOUSERSITE=1 \
            OMNI_KIT_ACCEPT_EULA=YES \
            LD_LIBRARY_PATH="$EVAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            VK_ICD_FILENAMES="$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json" \
            __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_ENV/share/glvnd/egl_vendor.d/10_nvidia.json" \
            "${EVAL_COMMAND[@]}" \
                --task-name "$task" \
                --host 127.0.0.1 --port "$PORT" \
                --mode "$MODE" \
                --prompt-mode "$PROMPT_MODE" \
                --num-rollouts "$NUM_ROLLOUTS" \
                "${EVAL_ARGS[@]}" \
                --output-dir "$out" \
                "$VIDEO_ARG"
    }
    set +e
    if (( QUEUE_MODE )); then
        run_evaluator > "$EVAL_LOG" 2>&1
        pipeline_status=("$?" 0)
    else
        run_evaluator 2>&1 | tee "$EVAL_LOG"
        pipeline_status=("${PIPESTATUS[@]}")
    fi
    set -e
    stop_server
    if (( pipeline_status[0] != 0 || pipeline_status[1] != 0 )); then
        echo "FAIL $task (evaluator=${pipeline_status[0]} tee=${pipeline_status[1]})" \
            | tee -a "$FAILURES_LOG"
        status=1
        if (( QUEUE_MODE )); then break; fi
    fi
done

EVAL_ELAPSED_SECONDS=$(( $(date +%s) - EVAL_START_EPOCH ))
(( EVAL_ELAPSED_SECONDS > 0 )) || EVAL_ELAPSED_SECONDS=1
if (( QUEUE_MODE )); then
    METRICS_PATH="$QUEUE_LOG_DIR/eval_metrics_$PORT.json"
else
    METRICS_PATH="$OUTPUT_ROOT/$MODEL/eval_metrics_$PORT.json"
fi
printf '{"port":%d,"server_gpu":%d,"eval_gpu":%d,"elapsed_seconds":%d,"exit_code":%d}\n' \
    "$PORT" "$SERVER_GPU" "$EVAL_GPU" "$EVAL_ELAPSED_SECONDS" "$status" > "$METRICS_PATH"
echo "results under $OUTPUT_ROOT/$MODEL"
exit "$status"
