#!/usr/bin/env bash
# One isolated teacher-assisted attempt. No evaluation queue or score reporting.
set -euo pipefail
set +x

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${PI05_ENV:?set policy runtime environment}"
: "${EVAL_ENV:?set simulator runtime environment}"
: "${PI05_CKPT:?set immutable checkpoint}"
: "${CODEX_COLLECTION_TASK:?set one task}"
: "${CODEX_COLLECTION_INSTANCE:?set collection instance}"
: "${CODEX_COLLECTION_ATTEMPT:?set independent attempt ID}"
: "${CODEX_COLLECTION_OUTPUT:?set fresh attempt output directory}"
: "${CODEX_COLLECTION_LOG_ROOT:?set isolated log root}"
: "${CODEX_DEMONSTRATION_MANIFEST:?set original demonstration manifest}"
: "${CODEX_ROBOT_CONFIG:?set collection robot controller config}"
: "${SCO_EPHEMERAL_RUNTIME_ROOT:?set worker-local cache root}"
: "${OMNIGIBSON_DATA_PATH:?set installed simulation assets}"
PI05_REPO_ID="${PI05_REPO_ID:-b1k_full100_joint_subtask_v1_drop_episode_s1}"
PORT="${PORT:-8920}"
MAX_STEPS="${MAX_STEPS:-4500}"
CODEX_MAX_DECISIONS="${CODEX_MAX_DECISIONS:-32}"
CODEX_MAX_SECONDS="${CODEX_MAX_SECONDS:-7200}"
CODEX_TEACHER_INTERVAL="${CODEX_TEACHER_INTERVAL:-256}"
CODEX_STEP_TIMEOUT="${CODEX_STEP_TIMEOUT:-120}"
CODEX_CHECKPOINT_ID="${CODEX_CHECKPOINT_ID:-1688977}"
CODEX_TEACHER_BACKEND="${CODEX_TEACHER_BACKEND:-mailbox}"
CODEX_COLLECTION_MODE="${CODEX_COLLECTION_MODE:-teacher_conditioned}"
case "$CODEX_COLLECTION_MODE" in
    teacher_conditioned|native_with_corrections) ;;
    *) echo "Unsupported collection mode" >&2; exit 2 ;;
esac
TEACHER_ARGS=()
case "$CODEX_TEACHER_BACKEND" in
    codex_local)
        : "${CODEX_TEACHER_BINARY:?set cluster Codex executable}"
        : "${CODEX_HOME:?set verified cluster Codex account home}"
        [[ -x "$CODEX_TEACHER_BINARY" ]] || { echo "Codex executable missing" >&2; exit 2; }
        TEACHER_ARGS=(--teacher-codex --codex-binary "$CODEX_TEACHER_BINARY" --teacher-timeout "${CODEX_TEACHER_TIMEOUT:-120}") ;;
    mailbox)
        : "${CODEX_TEACHER_MAILBOX_CONFIG:?set bounded teacher mailbox config}"
        [[ -f "$CODEX_TEACHER_MAILBOX_CONFIG" ]] || { echo "Teacher mailbox config missing" >&2; exit 2; }
        TEACHER_ARGS=(--teacher-mailbox-config "$CODEX_TEACHER_MAILBOX_CONFIG") ;;
    *) echo "Unsupported teacher backend" >&2; exit 2 ;;
esac
PI05_MEM_FRACTION="${PI05_MEM_FRACTION:-0.35}"
[[ "$CODEX_COLLECTION_INSTANCE" =~ ^[0-9]+$ ]] && (( CODEX_COLLECTION_INSTANCE >= 311 && CODEX_COLLECTION_INSTANCE <= 320 )) || {
    echo "Collection requires instance 311..320" >&2; exit 2; }
for setting in PORT MAX_STEPS CODEX_MAX_DECISIONS CODEX_MAX_SECONDS CODEX_TEACHER_INTERVAL CODEX_STEP_TIMEOUT; do
    [[ "${!setting}" =~ ^[1-9][0-9]*$ ]] || { echo "$setting must be positive" >&2; exit 2; }
done
[[ ! -e "$CODEX_COLLECTION_OUTPUT" ]] || { echo "Attempt output must be fresh" >&2; exit 2; }
[[ -f "$PI05_CKPT/assets/$PI05_REPO_ID/norm_stats.json" ]] || { echo "Checkpoint norm stats missing" >&2; exit 2; }
for path in "$CODEX_DEMONSTRATION_MANIFEST" "$CODEX_ROBOT_CONFIG"; do
    [[ -f "$path" ]] || { echo "Collection input missing" >&2; exit 2; }
done
case "$SCO_EPHEMERAL_RUNTIME_ROOT" in
    */../*|*/..|*/./*|*/.) echo "Invalid private runtime cache path" >&2; exit 2 ;;
    /tmp/behavior-1k-runtime/*|/var/tmp/behavior-1k-runtime/*) ;;
    *) echo "Runtime caches must be worker-local" >&2; exit 2 ;;
esac
mkdir -p "$CODEX_COLLECTION_LOG_ROOT"
SERVER_LOG="$CODEX_COLLECTION_LOG_ROOT/policy.log"
EVAL_LOG="$CODEX_COLLECTION_LOG_ROOT/collector.log"
EVAL_RUNTIME_ROOT="$SCO_EPHEMERAL_RUNTIME_ROOT/port-$PORT"
mkdir -p "$EVAL_RUNTIME_ROOT"/{appdata,cache,config,warp}

stop_server() {
    [[ -n "${SERVER_PID:-}" ]] || return 0
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    "$EVAL_ENV/bin/python" - "$SERVER_PID" <<'PY'
import os, sys, threading, time
from pathlib import Path
pid = int(sys.argv[1]); deadline = time.monotonic() + 20
while time.monotonic() < deadline:
    try:
        os.kill(pid, 0)
        status = Path(f"/proc/{pid}/stat")
        if status.is_file() and status.read_text().rsplit(")", 1)[1].split()[0] == "Z":
            break
    except ProcessLookupError:
        break
    threading.Event().wait(.25)
PY
    kill -KILL "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
    SERVER_PID=""
}
trap stop_server EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
set +u
source "$PI05_ENV/bin/activate"
set -u
cd "$REPO_ROOT/openpi"
SERVER_TOKEN="$(python -c 'import secrets; print(secrets.token_hex(32))')"
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=true \
XLA_PYTHON_CLIENT_MEM_FRACTION="$PI05_MEM_FRACTION" HF_HUB_OFFLINE=1 \
PYTHONPATH="$REPO_ROOT/openpi/src:$REPO_ROOT/openpi/packages/openpi-client/src" \
python scripts/b1k/serve_b1k.py \
    --robot b1k/R1Pro --task "b1k/$CODEX_COLLECTION_TASK" \
    --repo-id "$PI05_REPO_ID" --policy.config pi05_b1k --policy.dir "$PI05_CKPT" \
    --control_mode receding_horizon --action_horizon 16 \
    --subtask-prediction --max-token-len 512 --max-subtask-len 64 \
    --subtask-temperature 0.0 --subtask-action-conditioning task_and_subtask \
    --server-token "$SERVER_TOKEN" --port "$PORT" >> "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
if ! "$EVAL_ENV/bin/python" - "$PORT" "$SERVER_TOKEN" "$SERVER_PID" <<'PY'
import msgpack
import os
import sys
import threading
import time
import websockets.sync.client
deadline = time.monotonic() + 900
while time.monotonic() < deadline:
    try:
        os.kill(int(sys.argv[3]), 0)
    except ProcessLookupError:
        raise SystemExit("Policy server exited during startup; see policy.log")
    try:
        with websockets.sync.client.connect(f"ws://127.0.0.1:{int(sys.argv[1])}", compression=None,
                max_size=None, open_timeout=2, close_timeout=1) as connection:
            metadata = msgpack.unpackb(connection.recv(timeout=2), raw=False)
        if (isinstance(metadata, dict) and metadata.get("server_instance_token") == sys.argv[2]
                and metadata.get("subtask_generation") is True
                and metadata.get("subtask_action_conditioning") == "task_and_subtask"):
            raise SystemExit(0)
    except Exception:
        pass
    # Bounded backoff while a real policy process initializes, never an idle job.
    threading.Event().wait(min(1, max(0, deadline-time.monotonic())))
raise SystemExit("Policy server readiness deadline exceeded")
PY
then exit 1; fi
cd "$REPO_ROOT/BEHAVIOR-1K"
set +e
env OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH" \
    OMNIGIBSON_APPDATA_PATH="$EVAL_RUNTIME_ROOT/appdata" OMNIGIBSON_GPU_ID=0 \
    XDG_CACHE_HOME="$EVAL_RUNTIME_ROOT/cache" XDG_CONFIG_HOME="$EVAL_RUNTIME_ROOT/config" \
    WARP_CACHE_PATH="$EVAL_RUNTIME_ROOT/warp" \
    PYTHONPATH="$REPO_ROOT/BEHAVIOR-1K/OmniGibson:$REPO_ROOT/BEHAVIOR-1K/bddl3:$REPO_ROOT/BEHAVIOR-1K/joylo" \
    PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    LD_LIBRARY_PATH="$EVAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$EVAL_ENV/bin/python" "$REPO_ROOT/scripts/collect_b1k_codex.py" \
        --task-name "$CODEX_COLLECTION_TASK" --instance-id "$CODEX_COLLECTION_INSTANCE" \
        --attempt-id "$CODEX_COLLECTION_ATTEMPT" --checkpoint-id "$CODEX_CHECKPOINT_ID" \
        --output-dir "$CODEX_COLLECTION_OUTPUT" \
        --max-steps "$MAX_STEPS" --max-decisions "$CODEX_MAX_DECISIONS" \
        --max-seconds "$CODEX_MAX_SECONDS" --step-timeout "$CODEX_STEP_TIMEOUT" \
        --teacher-interval "$CODEX_TEACHER_INTERVAL" \
        --collection-mode "$CODEX_COLLECTION_MODE" \
        "${TEACHER_ARGS[@]}" \
        --demonstration-manifest "$CODEX_DEMONSTRATION_MANIFEST" \
        --robot-config "$CODEX_ROBOT_CONFIG" --host 127.0.0.1 --port "$PORT" \
        --expected-server-token "$SERVER_TOKEN" > "$EVAL_LOG" 2>&1
collector_status=$?
set -e
# Native simulator shutdown can exit the interpreter with status 0 while
# unwinding a failed rollout. Require its completion receipt independently.
if [[ ! -f "$CODEX_COLLECTION_OUTPUT/completed.json" || -f "$CODEX_COLLECTION_OUTPUT/failed.json" || -f "${CODEX_COLLECTION_OUTPUT%/*}/collector_failure.json" ]]; then
    echo "Collector completion missing or failure recorded; inspect collector_failure.json and collector.log" >&2
    (( collector_status != 0 )) || collector_status=1
fi
kill -0 "$SERVER_PID" 2>/dev/null || { echo "Policy server exited before collector completed" >&2; exit 1; }
exit "$collector_status"
