#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$REPO_ROOT/openpi"
source "$SCRIPT_DIR/multihost_job.env"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_file() { [[ -s "$1" ]] || fail "required file missing or empty: $1"; }

if [[ "$NORM_STATS_SOURCE" != /* ]]; then
    NORM_STATS_SOURCE="$REPO_ROOT/$NORM_STATS_SOURCE"
fi

RENDEZVOUS_DIR="$OUTPUT_ROOT/rendezvous"
mkdir -p "$RENDEZVOUS_DIR"

HOST_IP="$(hostname -i 2>/dev/null | awk '{print $1}')"
[[ -n "$HOST_IP" ]] || HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
[[ -n "$HOST_IP" ]] || fail "could not determine a routable host IP"
IDENTITY_TExT="$(hostname)|$HOST_IP|$$"
IDENTITY_FILE="$RENDEZVOUS_DIR/$(hostname)-$$.host"
printf '%s\n' "$IDENTITY_TExT" > "$IDENTITY_FILE"
log "wrote rendezvous identity $IDENTITY_FILE ($IDENTITY_TExT)"

log "waiting for $PROCESS_COUNT workers to register"
DEADLINE=$(( $(date +%s) + RENDEZVOUS_TIMEOUT ))
while :; do
    mapfile -t IDENTITIES < <(cat "$RENDEZVOUS_DIR"/*.host 2>/dev/null | sort -u)
    (( ${#IDENTITIES[@]} >= PROCESS_COUNT )) && break
    (( $(date +%s) < DEADLINE )) || fail "rendezvous timed out with ${#IDENTITIES[@]}/$PROCESS_COUNT workers"
    sleep 5
done
if (( ${#IDENTITIES[@]} != PROCESS_COUNT )); then
    fail "rendezvous found ${#IDENTITIES[@]} identities, expected $PROCESS_COUNT: ${IDENTITIES[*]}"
fi

PROCESS_ID=-1
for index in "${!IDENTITIES[@]}"; do
    if [[ "${IDENTITIES[index]}" == "$IDENTITY_TExT" ]]; then
        PROCESS_ID="$index"
    fi
done
(( PROCESS_ID >= 0 )) || fail "this worker is not present in the sorted rendezvous set"

COORDINATOR_IP="$(cut -d'|' -f2 <<< "${IDENTITIES[0]}")"
COORDINATOR_ADDRESS="$COORDINATOR_IP:$COORDINATOR_PORT"
log "assigned process_id=$PROCESS_ID/$PROCESS_COUNT coordinator=$COORDINATOR_ADDRESS"

if [[ "$MODE" == "discover" ]]; then
    REPORT="$RENDEZVOUS_DIR/rank-$PROCESS_ID.discovery.json"
    READY_FILE="$RENDEZVOUS_DIR/coordinator.ready"
    NETWORK_OK_FILE="$RENDEZVOUS_DIR/rank-$PROCESS_ID.network-ok"
    GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
    GPU_COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
    (( GPU_COUNT >= EXPECTED_LOCAL_DEVICES )) || fail "expected >= $EXPECTED_LOCAL_DEVICES GPUs, saw $GPU_COUNT"

    if (( PROCESS_ID == 0 )); then
        python3 - "$COORDINATOR_PORT" "$((PROCESS_COUNT - 1))" "$READY_FILE" <<'PY' &
import pathlib
import socket
import sys

port, peers, ready = int(sys.argv[1]), int(sys.argv[2]), pathlib.Path(sys.argv[3])
with socket.socket() as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", port))
    server.listen(peers)
    ready.write_text("ready\n")
    for _ in range(peers):
        connection, _ = server.accept()
        with connection:
            connection.sendall(b"ok")
PY
        TCP_SERVER_PID=$!
        wait "$TCP_SERVER_PID" || fail "coordinator TCP listener failed"
        printf 'ok\n' > "$NETWORK_OK_FILE"
    else
        while [[ ! -f "$READY_FILE" ]]; do
            (( $(date +%s) < DEADLINE )) || fail "coordinator readiness timed out"
            sleep 2
        done
        python3 - "$COORDINATOR_IP" "$COORDINATOR_PORT" <<'PY'
import socket
import sys
import time

host, port = sys.argv[1], int(sys.argv[2])
for _ in range(60):
    try:
        with socket.create_connection((host, port), timeout=5) as connection:
            if connection.recv(2) != b"ok":
                raise RuntimeError("unexpected coordinator response")
        break
    except OSError:
        time.sleep(1)
else:
    raise SystemExit(f"could not connect to coordinator {host}:{port}")
PY
        printf 'ok\n' > "$NETWORK_OK_FILE"
    fi

    while :; do
        network_count="$(find "$RENDEZVOUS_DIR" -maxdepth 1 -name 'rank-*.network-ok' -type f | wc -l)"
        (( network_count == PROCESS_COUNT )) && break
        (( $(date +%s) < DEADLINE )) || fail "network verification timed out with $network_count/$PROCESS_COUNT reports"
        sleep 2
    done

    printf '{"process_id":%d,"process_count":%d,"host":"%s","ip":"%s","coordinator":"%s","gpu_count":%d,"gpus":"%s","shared_oss_visible":true,"coordinator_reachable":true}\n' \
        "$PROCESS_ID" "$PROCESS_COUNT" "$(hostname)" "$HOST_IP" "$COORDINATOR_ADDRESS" "$GPU_COUNT" "$GPU_NAMES" \
        > "$REPORT"
    log "discovery report: $REPORT"
    while :; do
        report_count="$(find "$RENDEZVOUS_DIR" -maxdepth 1 -name 'rank-*.discovery.json' -type f | wc -l)"
        (( report_count == PROCESS_COUNT )) && break
        (( $(date +%s) < DEADLINE )) || fail "discovery report timeout with $report_count/$PROCESS_COUNT reports"
        sleep 2
    done
    if (( PROCESS_ID == 0 )); then
        printf '{"process_count":%d,"coordinator":"%s","coordinator_reachable":true}\n' \
            "$PROCESS_COUNT" "$COORDINATOR_ADDRESS" > "$OUTPUT_ROOT/_DISCOVERY_OK.json"
    fi
    log "discovery complete for rank $PROCESS_ID"
    exit 0
fi

RANK_ROOT="/var/tmp/wangfaye/aihub-b1k-multihost/$RUN_ID/rank-$PROCESS_ID"
ENV_PREFIX="/dev/shm/wangfaye/aihub-b1k-multihost/$RUN_ID/rank-$PROCESS_ID/b1k-pi05"
OPENPI_DATA_HOME="$RANK_ROOT/openpi-cache"
LOCAL_DATASET="$RANK_ROOT/two_tasks_joint_subtask_s1"
ASSETS_BASE_DIR="$RANK_ROOT/assets"
CHECKPOINT_BASE_DIR="$OUTPUT_ROOT/checkpoints"
LOG_DIR="$OUTPUT_ROOT/logs"
RANK_LOG="$LOG_DIR/rank-$PROCESS_ID.log"
mkdir -p "$RANK_ROOT" "$ENV_PREFIX" "$LOG_DIR"
exec > >(tee -a "$RANK_LOG") 2>&1

run_with_heartbeat() {
    local label="$1"
    shift
    "$@" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        log "$label in progress"
        sleep 30
    done
    wait "$pid"
}

require_file "$ENV_ARCHIVE"
log "restoring packed environment"
run_with_heartbeat "environment extraction" tar -xzf "$ENV_ARCHIVE" -C "$ENV_PREFIX"
require_file "$ENV_PREFIX/bin/python"
if [[ -x "$ENV_PREFIX/bin/conda-unpack" ]]; then
    "$ENV_PREFIX/bin/conda-unpack"
fi
SITE_PACKAGES="$("$ENV_PREFIX/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n%s\n' "$OPENPI_DIR/src" "$OPENPI_DIR/packages/openpi-client/src" \
    > "$SITE_PACKAGES/00-b1k-multihost-source.pth"

export OPENPI_PROCESS_COUNT="$PROCESS_COUNT"
export OPENPI_PROCESS_ID="$PROCESS_ID"
export OPENPI_COORDINATOR_ADDRESS="$COORDINATOR_ADDRESS"
export OPENPI_LOCAL_DEVICE_IDS="$LOCAL_DEVICE_IDS"
export OPENPI_EXPECTED_GLOBAL_DEVICES="$EXPECTED_GLOBAL_DEVICES"

if [[ "$MODE" == "preflight" ]]; then
    cd "$OPENPI_DIR"
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" FSDP_DEVICES="$FSDP_DEVICES" BATCH_SIZE="$BATCH_SIZE"
    export OPENPI_PREFLIGHT_DIR="$OUTPUT_ROOT/preflight"
    set +e
    "$ENV_PREFIX/bin/python" scripts/b1k/distributed_preflight.py
    status=$?
    set -e
    exit "$status"
fi

require_file "$OPENPI_CACHE_ARCHIVE"
require_file "$SOURCE_DATASET/meta/info.json"
require_file "$NORM_STATS_SOURCE"
require_file "$SUBTASK_SEGMENTS_SOURCE"
[[ "$(sha256sum "$SUBTASK_SEGMENTS_SOURCE" | cut -d' ' -f1)" == "$SUBTASK_SEGMENTS_SHA256" ]] \
    || fail "subtask annotation checksum mismatch"
mkdir -p "$OPENPI_DATA_HOME" "$LOCAL_DATASET/meta" "$ASSETS_BASE_DIR"
if [[ -e "$OPENPI_DIR/.venv" && ! -L "$OPENPI_DIR/.venv" ]]; then
    fail "refusing to replace non-symlink $OPENPI_DIR/.venv"
fi
ln -sfn "$ENV_PREFIX" "$OPENPI_DIR/.venv"

log "staging OpenPI cache"
run_with_heartbeat "OpenPI cache extraction" tar -xf "$OPENPI_CACHE_ARCHIVE" -C "$OPENPI_DATA_HOME"
require_file "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
require_file "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params/_METADATA"

log "staging dataset chunks"
run_with_heartbeat "dataset metadata copy" cp -r --no-preserve=ownership "$SOURCE_DATASET/meta/." "$LOCAL_DATASET/meta/"
for chunk in chunk-014 chunk-056; do
    mkdir -p "$LOCAL_DATASET/data/$chunk"
    run_with_heartbeat "data $chunk copy" cp -r --no-preserve=ownership \
        "$SOURCE_DATASET/data/$chunk/." "$LOCAL_DATASET/data/$chunk/"
    for camera in observation.rgb.zed_link_camera_0 observation.rgb.left_realsense_link_camera_0 observation.rgb.right_realsense_link_camera_0; do
        mkdir -p "$LOCAL_DATASET/videos/$camera/$chunk"
        run_with_heartbeat "$camera $chunk copy" cp -r --no-preserve=ownership \
            "$SOURCE_DATASET/videos/$camera/$chunk/." "$LOCAL_DATASET/videos/$camera/$chunk/"
    done
done
"$ENV_PREFIX/bin/python" "$REPO_ROOT/scripts/strip_depth_from_info.py" "$LOCAL_DATASET"
cp "$SUBTASK_SEGMENTS_SOURCE" "$LOCAL_DATASET/meta/subtask_segments.parquet"
mkdir -p "$ASSETS_BASE_DIR/pi05_b1k/$REPO_ID"
cp "$NORM_STATS_SOURCE" "$ASSETS_BASE_DIR/pi05_b1k/$REPO_ID/norm_stats.json"

EPISODES_FILE="$LOCAL_DATASET/meta/pi05_two_tasks_episodes.txt"
{ seq 2800 2999; seq 11200 11399; } > "$EPISODES_FILE"

export OPENPI_REQUIRE_FINITE_METRICS=1
export OPENPI_VERIFY_FINAL_CHECKPOINT=1
export OPENPI_SUCCESS_PATH="$OUTPUT_ROOT/_SUCCESS.rank0.json"
export PREFLIGHT_OUTPUT_DIR="$RANK_ROOT/subtask_preflight"

log "starting multi-host $MODE; log: $RANK_LOG"

run_with_heartbeat "multi-host training" env \
    OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
    DATASET_ROOT="$LOCAL_DATASET" \
    REPO_ID="$REPO_ID" \
    EPISODES_FILE="$EPISODES_FILE" \
    EXP_NAME="$EXP_NAME" \
    ASSETS_BASE_DIR="$ASSETS_BASE_DIR" \
    CHECKPOINT_BASE_DIR="$CHECKPOINT_BASE_DIR" \
    NUM_GPUS="$EXPECTED_LOCAL_DEVICES" \
    CUDA_DEVICES="$CUDA_DEVICES" \
    FSDP_DEVICES="$FSDP_DEVICES" \
    NUM_WORKERS="$NUM_WORKERS" \
    BATCH_SIZE="$BATCH_SIZE" \
    TRAIN_STEPS="$TRAIN_STEPS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    LOG_INTERVAL=1 \
    FRAME_STRIDE=1 \
    LR_DECAY_STEPS="$TRAIN_STEPS" \
    SUBTASK_PREDICTION=1 \
    WHOLE_TASK_PROB=0 \
    MAX_TOKEN_LEN=512 \
    MAX_SUBTASK_LEN=64 \
    SUBTASK_LOSS_WEIGHT=1.0 \
    SUBTASK_LOSS_CHUNK_SIZE=8 \
    SUBTASK_EMPTY_LABEL_POLICY=drop_episode \
    SUBTASK_PROMPTS="$LOCAL_DATASET/meta/subtask_segments.parquet" \
    NORM_MAX_FRAMES="$NORM_MAX_FRAMES" \
    bash "$REPO_ROOT/scripts/train_pi05_subtask_b1k.sh" full
