#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$REPO_ROOT/openpi"
source "$SCRIPT_DIR/multihost_job.env"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
fail() {
    printf 'ERROR: %s\n' "$*" >&2
    if [[ -n "${RENDEZVOUS_DIR:-}" && "${PROCESS_ID:--1}" =~ ^[0-9]+$ ]]; then
        printf '%s\n' "$*" > "$RENDEZVOUS_DIR/rank-$PROCESS_ID.failed" 2>/dev/null || true
    fi
    exit 1
}
require_file() { [[ -s "$1" ]] || fail "required file missing or empty: $1"; }

if [[ -n "$NORM_STATS_SOURCE" && "$NORM_STATS_SOURCE" != /* ]]; then
    NORM_STATS_SOURCE="$REPO_ROOT/$NORM_STATS_SOURCE"
fi

RENDEZVOUS_DIR="$OUTPUT_ROOT/rendezvous/$ATTEMPT_ID"
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
mapfile -t WORKER_HOST_IPS < <(printf '%s\n' "${IDENTITIES[@]}" | cut -d'|' -f2 | sort -u)
if (( ${#WORKER_HOST_IPS[@]} != PROCESS_COUNT )); then
    fail "multi-host run requires $PROCESS_COUNT distinct hosts, found ${#WORKER_HOST_IPS[@]}: ${WORKER_HOST_IPS[*]}"
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

case "$DATASET_PROFILE" in
    two_tasks|two_tasks_joint_h200|full100) ;;
    *) fail "unknown dataset profile: $DATASET_PROFILE" ;;
esac
CONTRACT_PROFILE=0
if [[ "$DATASET_PROFILE" == "full100" || "$DATASET_PROFILE" == "two_tasks_joint_h200" ]]; then
    CONTRACT_PROFILE=1
fi
if (( CONTRACT_PROFILE )); then
    mapfile -t GPU_NAME_LIST < <(nvidia-smi --query-gpu=name --format=csv,noheader)
    (( ${#GPU_NAME_LIST[@]} == EXPECTED_LOCAL_DEVICES )) \
        || fail "$DATASET_PROFILE requires exactly $EXPECTED_LOCAL_DEVICES GPUs, saw ${#GPU_NAME_LIST[@]}"
    for gpu_name in "${GPU_NAME_LIST[@]}"; do
        case "$gpu_name" in
            *H200*|"NVIDIA L20Z") ;;
            *) fail "$DATASET_PROFILE requires H200 GPUs or the NVIDIA L20Z platform alias, saw: $gpu_name" ;;
        esac
    done
fi

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

RANK_ROOT="/var/tmp/wangfaye/aihub-b1k-multihost/$RUN_ID/$ATTEMPT_ID/rank-$PROCESS_ID"
ENV_PREFIX="/dev/shm/wangfaye/aihub-b1k-multihost/$RUN_ID/$ATTEMPT_ID/rank-$PROCESS_ID/b1k-pi05"
export OPENPI_DATA_HOME="$RANK_ROOT/openpi-cache"
LOCAL_DATASET="$RANK_ROOT/$LOCAL_DATASET_NAME"
LOCAL_STAGING="$RANK_ROOT/staging"
OSSUTIL_CHECKPOINT_DIR="$RANK_ROOT/ossutil-checkpoints"
OSSUTIL_OUTPUT_DIR="$RANK_ROOT/ossutil-output"
OSSUTIL="$SCRIPT_DIR/ossutil"
ASSETS_BASE_DIR="$RANK_ROOT/assets"
CHECKPOINT_BASE_DIR="$OUTPUT_ROOT/checkpoints"
LOG_DIR="$OUTPUT_ROOT/logs/$ATTEMPT_ID"
CONTRACT_DIR="$OUTPUT_ROOT/contracts/$ATTEMPT_ID"
STAGING_DIR="$OUTPUT_ROOT/staging/$ATTEMPT_ID"
RANK_LOG="$LOG_DIR/rank-$PROCESS_ID.log"
mkdir -p "$RANK_ROOT" "$ENV_PREFIX" "$LOCAL_STAGING" "$OSSUTIL_CHECKPOINT_DIR" "$OSSUTIL_OUTPUT_DIR" \
    "$LOG_DIR" "$CONTRACT_DIR" "$STAGING_DIR"
exec > >(tee -a "$RANK_LOG") 2>&1
record_unhandled_failure() {
    local status=$?
    trap - ERR
    printf 'unhandled command failure (exit %d)\n' "$status" > "$RENDEZVOUS_DIR/rank-$PROCESS_ID.failed" 2>/dev/null || true
    exit "$status"
}
trap record_unhandled_failure ERR

AVAILABLE_BYTES="$(df -PB1 "$RANK_ROOT" | awk 'NR==2 {print $4}')"
[[ "$AVAILABLE_BYTES" =~ ^[0-9]+$ ]] || fail "could not determine local free space for $RANK_ROOT"
if [[ "$DATASET_PROFILE" == "full100" && "$MODE" != "smoke" ]]; then
    (( AVAILABLE_BYTES >= MIN_LOCAL_BYTES )) \
        || fail "full100 needs at least $MIN_LOCAL_BYTES local bytes, only $AVAILABLE_BYTES are available"
    log "full100 local storage check passed: $AVAILABLE_BYTES bytes available"
fi

run_with_heartbeat() {
    local label="$1"
    shift
    "$@" &
    local pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        log "$label in progress"
        for (( second = 0; second < 30; second++ )); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
    done
    wait "$pid"
}

[[ -f "$OSSUTIL" ]] || fail "bundled ossutil is missing"
chmod 0755 "$OSSUTIL" || fail "could not make bundled ossutil executable"
[[ "$(sha256sum "$OSSUTIL" | cut -d' ' -f1)" == "$OSSUTIL_SHA256" ]] \
    || fail "bundled ossutil checksum mismatch"
[[ -n "${OSS_ACCESS_KEY_ID:-}" ]] || fail "OSS access key ID is missing"
[[ -n "${OSS_ACCESS_KEY_SECRET:-}" ]] || fail "OSS access key secret is missing"
[[ -n "$OSS_DOWNLOAD_ENDPOINT" ]] || fail "OSS download endpoint is missing"
for value in "$OSSUTIL_JOBS" "$OSSUTIL_PARALLEL"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || fail "ossutil concurrency must be a positive integer"
done
for uri in "$ENV_ARCHIVE_OSS_URI" "$OPENPI_CACHE_ARCHIVE_OSS_URI" \
    "$SOURCE_DATASET_OSS_URI" "$SUBTASK_SEGMENTS_OSS_URI"; do
    [[ "$uri" == oss://* ]] || fail "invalid OSS source URI"
done

ossutil_copy_file() {
    "$OSSUTIL" cp "$1" "$2" \
        --endpoint "$OSS_DOWNLOAD_ENDPOINT" --sign-version v1 \
        --parallel "$OSSUTIL_PARALLEL" \
        --checkpoint-dir "$OSSUTIL_CHECKPOINT_DIR" \
        --output-dir "$OSSUTIL_OUTPUT_DIR" \
        --retry-times 10 --force --no-progress
}

ossutil_copy_tree() {
    "$OSSUTIL" cp "${1%/}/" "${2%/}/" --recursive \
        --endpoint "$OSS_DOWNLOAD_ENDPOINT" --sign-version v1 \
        --job "$OSSUTIL_JOBS" --parallel "$OSSUTIL_PARALLEL" \
        --checkpoint-dir "$OSSUTIL_CHECKPOINT_DIR" \
        --output-dir "$OSSUTIL_OUTPUT_DIR" \
        --retry-times 10 --force --no-progress
}

download_file() {
    local label="$1" source="$2" destination="$3" start elapsed bytes
    mkdir -p "$(dirname "$destination")"
    start="$(date +%s)"
    run_with_heartbeat "$label download" ossutil_copy_file "$source" "$destination"
    require_file "$destination"
    elapsed=$(( $(date +%s) - start ))
    (( elapsed > 0 )) || elapsed=1
    bytes="$(stat -c '%s' "$destination")"
    log "$label download complete: $bytes bytes in ${elapsed}s ($(( bytes / elapsed / 1048576 )) MiB/s)"
}

download_tree() {
    local label="$1" source="$2" destination="$3" start elapsed bytes
    mkdir -p "$destination"
    start="$(date +%s)"
    run_with_heartbeat "$label download" ossutil_copy_tree "$source" "$destination"
    [[ -n "$(find "$destination" -type f -print -quit)" ]] \
        || fail "$label download produced no files: $destination"
    elapsed=$(( $(date +%s) - start ))
    (( elapsed > 0 )) || elapsed=1
    read -r bytes _ < <(du -sb "$destination")
    log "$label download complete: $bytes bytes in ${elapsed}s ($(( bytes / elapsed / 1048576 )) MiB/s)"
}

validate_tree() {
    local label="$1" root="$2" expected_files="$3" expected_bytes="$4" expected_hash="$5"
    local actual_files actual_bytes actual_hash
    read -r actual_files actual_bytes actual_hash < <(
        "$ENV_PREFIX/bin/python" - "$root" <<'PY'
import hashlib
import os
import sys

root = os.path.abspath(sys.argv[1])
records = []
for directory, _, names in os.walk(root):
    for name in names:
        path = os.path.join(directory, name)
        records.append((os.path.relpath(path, root), os.path.getsize(path)))
records.sort()
digest = hashlib.sha256()
for relative_path, size in records:
    digest.update(f"{relative_path}\t{size}\n".encode())
print(len(records), sum(size for _, size in records), digest.hexdigest())
PY
    )
    [[ "$actual_files" == "$expected_files" && "$actual_bytes" == "$expected_bytes" ]] \
        || fail "$label staging mismatch: files=$actual_files/$expected_files bytes=$actual_bytes/$expected_bytes"
    [[ "$actual_hash" == "$expected_hash" ]] \
        || fail "$label manifest mismatch: $actual_hash != $expected_hash"
    log "$label staging verified: $actual_files files, $actual_bytes bytes, manifest=$actual_hash"
}

LOCAL_ENV_ARCHIVE="$LOCAL_STAGING/$(basename "$ENV_ARCHIVE_OSS_URI")"
download_file "packed environment" "$ENV_ARCHIVE_OSS_URI" "$LOCAL_ENV_ARCHIVE"
log "restoring packed environment"
run_with_heartbeat "environment extraction" tar -xzf "$LOCAL_ENV_ARCHIVE" -C "$ENV_PREFIX"
require_file "$ENV_PREFIX/bin/python"
if [[ -x "$ENV_PREFIX/bin/conda-unpack" ]]; then
    "$ENV_PREFIX/bin/conda-unpack"
fi
rm -f "$LOCAL_ENV_ARCHIVE"
SITE_PACKAGES="$("$ENV_PREFIX/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n%s\n' "$OPENPI_DIR/src" "$OPENPI_DIR/packages/openpi-client/src" \
    > "$SITE_PACKAGES/00-b1k-multihost-source.pth"

if (( PROCESS_COUNT > 1 )); then
    export OPENPI_PROCESS_COUNT="$PROCESS_COUNT"
    export OPENPI_PROCESS_ID="$PROCESS_ID"
    export OPENPI_COORDINATOR_ADDRESS="$COORDINATOR_ADDRESS"
    export OPENPI_LOCAL_DEVICE_IDS="$LOCAL_DEVICE_IDS"
    export OPENPI_EXPECTED_GLOBAL_DEVICES="$EXPECTED_GLOBAL_DEVICES"
else
    unset OPENPI_PROCESS_COUNT OPENPI_PROCESS_ID OPENPI_COORDINATOR_ADDRESS
    unset OPENPI_LOCAL_DEVICE_IDS OPENPI_EXPECTED_GLOBAL_DEVICES
fi

if [[ "$MODE" == "preflight" ]]; then
    cd "$OPENPI_DIR"
    export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" FSDP_DEVICES="$FSDP_DEVICES" BATCH_SIZE="$BATCH_SIZE"
    export OPENPI_PREFLIGHT_DIR="$OUTPUT_ROOT/preflight/$ATTEMPT_ID"
    export OPENPI_AVAILABLE_LOCAL_BYTES="$AVAILABLE_BYTES"
    export OPENPI_GPU_NAMES="$(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ';')"
    "$ENV_PREFIX/bin/python" scripts/b1k/distributed_preflight.py
    if [[ "$DATASET_PROFILE" == "two_tasks" ]]; then
        exit 0
    fi
fi

if [[ -n "$NORM_STATS_SOURCE" ]]; then
    require_file "$NORM_STATS_SOURCE"
fi
mkdir -p "$OPENPI_DATA_HOME" "$LOCAL_DATASET/meta" "$ASSETS_BASE_DIR"
if [[ -e "$OPENPI_DIR/.venv" && ! -L "$OPENPI_DIR/.venv" ]]; then
    fail "refusing to replace non-symlink $OPENPI_DIR/.venv"
fi
ln -sfn "$ENV_PREFIX" "$OPENPI_DIR/.venv"

LOCAL_CACHE_ARCHIVE="$LOCAL_STAGING/$(basename "$OPENPI_CACHE_ARCHIVE_OSS_URI")"
download_file "OpenPI cache archive" "$OPENPI_CACHE_ARCHIVE_OSS_URI" "$LOCAL_CACHE_ARCHIVE"
log "restoring OpenPI cache"
run_with_heartbeat "OpenPI cache extraction" tar -xf "$LOCAL_CACHE_ARCHIVE" -C "$OPENPI_DATA_HOME"
rm -f "$LOCAL_CACHE_ARCHIVE"
require_file "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
require_file "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params/_METADATA"

wait_for_files() {
    local directory="$1" pattern="$2" expected="$3" label="$4" timeout="${5:-$RENDEZVOUS_TIMEOUT}"
    local deadline count now last_log=0
    deadline=$(( $(date +%s) + timeout ))
    while :; do
        if [[ -n "$(find "$RENDEZVOUS_DIR" -maxdepth 1 -name 'rank-*.failed' -type f -print -quit)" ]]; then
            fail "$label aborted because another rank failed"
        fi
        count="$(find "$directory" -maxdepth 1 -name "$pattern" -type f | wc -l)"
        (( count == expected )) && return 0
        now="$(date +%s)"
        (( now < deadline )) || fail "$label timed out with $count/$expected files"
        if (( now - last_log >= 60 )); then
            log "waiting for $label: $count/$expected"
            last_log="$now"
        fi
        sleep 5
    done
}

publish_file() {
    local source="$1" destination="$2" temporary
    temporary="$destination.tmp.$(hostname).$$"
    cp "$source" "$temporary"
    mv -f "$temporary" "$destination"
}

STAGING_STARTED_AT="$(date +%s)"
log "staging dataset metadata with $OSSUTIL_JOBS object workers and $OSSUTIL_PARALLEL part workers"
download_tree "dataset metadata" "${SOURCE_DATASET_OSS_URI%/}/meta" "$LOCAL_DATASET/meta"
if [[ "$DATASET_PROFILE" == "full100" ]]; then
    validate_tree "metadata" "$LOCAL_DATASET/meta" \
        "$EXPECTED_META_FILES" "$EXPECTED_META_BYTES" "$EXPECTED_META_MANIFEST_SHA256"
fi
download_file "subtask annotations" "$SUBTASK_SEGMENTS_OSS_URI" "$LOCAL_DATASET/meta/subtask_segments.parquet"
[[ "$(sha256sum "$LOCAL_DATASET/meta/subtask_segments.parquet" | cut -d' ' -f1)" == "$SUBTASK_SEGMENTS_SHA256" ]] \
    || fail "subtask annotation checksum mismatch"

if (( CONTRACT_PROFILE )); then
    CONTRACT_REPORT="$RANK_ROOT/${DATASET_PROFILE}_contract.json"
    EPISODES_FILE="$RANK_ROOT/${DATASET_PROFILE}_episodes.txt"
    CONTRACT_ARGS=(
        --dataset-root "$LOCAL_DATASET"
        --segments-path "$LOCAL_DATASET/meta/subtask_segments.parquet"
        --max-subtask-len 64
        --empty-label-policy drop_episode
        --frame-stride "$FRAME_STRIDE"
        --global-batch-size "$BATCH_SIZE"
        --process-count "$PROCESS_COUNT"
        --report-file "$CONTRACT_REPORT"
        --write-episodes-file "$EPISODES_FILE"
    )
    if [[ "$DATASET_PROFILE" == "two_tasks_joint_h200" ]]; then
        REQUESTED_EPISODES=()
        while IFS= read -r episode; do REQUESTED_EPISODES+=("$episode"); done < <(
            { seq 2800 2999; seq 11200 11399; }
        )
        read -r -a EXPECTED_TASK_INDEX_ARRAY <<< "$EXPECTED_TASK_INDICES"
        CONTRACT_ARGS+=(
            --episodes-index "${REQUESTED_EPISODES[@]}"
            --expected-tasks "$EXPECTED_TASKS"
            --expected-task-indices "${EXPECTED_TASK_INDEX_ARRAY[@]}"
            --max-token-len 512
        )
        if [[ "$MODE" != "smoke" ]]; then
            CONTRACT_ARGS+=(--epochs "$TRAIN_EPOCHS")
        fi
    elif [[ "$MODE" == "smoke" ]]; then
        SMOKE_EPISODES=()
        while IFS= read -r episode; do SMOKE_EPISODES+=("$episode"); done < <(
            { seq 2800 2999; seq 11200 11399; }
        )
        CONTRACT_ARGS+=(--episodes-index "${SMOKE_EPISODES[@]}" --expected-tasks 2 --max-token-len 512)
    else
        CONTRACT_ARGS+=(--epochs "$TRAIN_EPOCHS" --expected-tasks "$EXPECTED_TASKS" --max-token-len 512)
    fi
    "$ENV_PREFIX/bin/python" "$OPENPI_DIR/scripts/b1k/check_subtask_data.py" "${CONTRACT_ARGS[@]}"
    require_file "$CONTRACT_REPORT"
    require_file "$EPISODES_FILE"

    IFS='|' read -r CONTRACT_SHA ANNOTATION_SHA EPISODE_LIST_SHA RETAINED_EPISODES RETAINED_FRAMES \
        RETAINED_TASKS TRAINING_SAMPLES BATCHES_PER_EPOCH CONTRACT_TRAIN_STEPS < <(
        "$ENV_PREFIX/bin/python" - "$CONTRACT_REPORT" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
contract = report["training_contract"]
values = [
    report["training_contract_sha256"],
    contract["annotation_sha256"],
    contract["episode_list_sha256"],
    contract["retained_episodes"],
    contract["retained_frames"],
    contract["retained_tasks"],
    contract["strided_samples"],
    contract.get("batches_per_epoch", ""),
    contract.get("train_steps", ""),
]
print("|".join(map(str, values)))
PY
    )
    [[ "$ANNOTATION_SHA" == "$SUBTASK_SEGMENTS_SHA256" ]] \
        || fail "contract annotation hash mismatch: $ANNOTATION_SHA"
    [[ "$(sha256sum "$EPISODES_FILE" | cut -d' ' -f1)" == "$EPISODE_LIST_SHA" ]] \
        || fail "contract episode file hash mismatch"
    if [[ "$DATASET_PROFILE" == "two_tasks_joint_h200" ]]; then
        [[ "$RETAINED_EPISODES" == "$EXPECTED_RETAINED_EPISODES" ]] \
            || fail "retained episodes mismatch: $RETAINED_EPISODES != $EXPECTED_RETAINED_EPISODES"
        [[ "$RETAINED_FRAMES" == "$EXPECTED_RETAINED_FRAMES" ]] \
            || fail "retained frames mismatch: $RETAINED_FRAMES != $EXPECTED_RETAINED_FRAMES"
        [[ "$RETAINED_TASKS" == "$EXPECTED_TASKS" ]] \
            || fail "retained tasks mismatch: $RETAINED_TASKS != $EXPECTED_TASKS"
        [[ "$TRAINING_SAMPLES" == "$EXPECTED_TRAINING_SAMPLES" ]] \
            || fail "training samples mismatch: $TRAINING_SAMPLES != $EXPECTED_TRAINING_SAMPLES"
        [[ "$BATCHES_PER_EPOCH" == "$EXPECTED_BATCHES_PER_EPOCH" ]] \
            || fail "batches per epoch mismatch: $BATCHES_PER_EPOCH != $EXPECTED_BATCHES_PER_EPOCH"
        [[ "$EPISODE_LIST_SHA" == "$EXPECTED_EPISODE_LIST_SHA256" ]] \
            || fail "episode-list hash mismatch: $EPISODE_LIST_SHA != $EXPECTED_EPISODE_LIST_SHA256"
        if [[ "$MODE" != "smoke" ]]; then
            [[ "$CONTRACT_TRAIN_STEPS" == "$EXPECTED_TRAIN_STEPS" ]] \
                || fail "contract train steps mismatch: $CONTRACT_TRAIN_STEPS != $EXPECTED_TRAIN_STEPS"
            [[ "$CONTRACT_SHA" == "$EXPECTED_CONTRACT_SHA256" ]] \
                || fail "contract hash mismatch: $CONTRACT_SHA != $EXPECTED_CONTRACT_SHA256"
        fi
    elif [[ "$MODE" != "smoke" ]]; then
        [[ "$RETAINED_EPISODES" == "$EXPECTED_RETAINED_EPISODES" ]] \
            || fail "retained episodes mismatch: $RETAINED_EPISODES != $EXPECTED_RETAINED_EPISODES"
        [[ "$RETAINED_FRAMES" == "$EXPECTED_RETAINED_FRAMES" ]] \
            || fail "retained frames mismatch: $RETAINED_FRAMES != $EXPECTED_RETAINED_FRAMES"
        [[ "$RETAINED_TASKS" == "$EXPECTED_TASKS" ]] \
            || fail "retained tasks mismatch: $RETAINED_TASKS != $EXPECTED_TASKS"
        [[ "$BATCHES_PER_EPOCH" == "$EXPECTED_BATCHES_PER_EPOCH" ]] \
            || fail "batches per epoch mismatch: $BATCHES_PER_EPOCH != $EXPECTED_BATCHES_PER_EPOCH"
        [[ "$CONTRACT_TRAIN_STEPS" == "$EXPECTED_TRAIN_STEPS" ]] \
            || fail "contract train steps mismatch: $CONTRACT_TRAIN_STEPS != $EXPECTED_TRAIN_STEPS"
        [[ "$EPISODE_LIST_SHA" == "$EXPECTED_EPISODE_LIST_SHA256" ]] \
            || fail "episode-list hash mismatch: $EPISODE_LIST_SHA != $EXPECTED_EPISODE_LIST_SHA256"
        [[ "$CONTRACT_SHA" == "$EXPECTED_CONTRACT_SHA256" ]] \
            || fail "contract hash mismatch: $CONTRACT_SHA != $EXPECTED_CONTRACT_SHA256"
    fi

    publish_file "$CONTRACT_REPORT" "$CONTRACT_DIR/rank-$PROCESS_ID.json"
    publish_file "$EPISODES_FILE" "$CONTRACT_DIR/rank-$PROCESS_ID.episodes.txt"
    wait_for_files "$CONTRACT_DIR" 'rank-*.json' "$PROCESS_COUNT" "contract report rendezvous"
    wait_for_files "$CONTRACT_DIR" 'rank-*.episodes.txt' "$PROCESS_COUNT" "episode-list rendezvous"
    "$ENV_PREFIX/bin/python" - "$CONTRACT_DIR" "$PROCESS_COUNT" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

root, process_count = Path(sys.argv[1]), int(sys.argv[2])
reports = [json.loads((root / f"rank-{rank}.json").read_text()) for rank in range(process_count)]
hashes = {report["training_contract_sha256"] for report in reports}
contracts = {json.dumps(report["training_contract"], sort_keys=True, separators=(",", ":")) for report in reports}
if len(hashes) != 1 or len(contracts) != 1:
    raise SystemExit("workers produced different training contracts")
for rank, report in enumerate(reports):
    episode_path = root / f"rank-{rank}.episodes.txt"
    digest = hashlib.sha256(episode_path.read_bytes()).hexdigest()
    if digest != report["training_contract"]["episode_list_sha256"]:
        raise SystemExit(f"rank {rank} episode-list file does not match its contract")
PY
    printf '%s\n' "$CONTRACT_SHA" > "$CONTRACT_DIR/rank-$PROCESS_ID.contract-ok"
    wait_for_files "$CONTRACT_DIR" 'rank-*.contract-ok' "$PROCESS_COUNT" "contract validation rendezvous"
    if (( PROCESS_ID == 0 )); then
        publish_file "$CONTRACT_REPORT" "$CONTRACT_DIR/report.json"
        publish_file "$EPISODES_FILE" "$CONTRACT_DIR/episodes.txt"
    fi
    log "$DATASET_PROFILE contract verified across $PROCESS_COUNT ranks: $CONTRACT_SHA"

    if [[ "$MODE" == "preflight" ]]; then
        printf 'ok\n' > "$CONTRACT_DIR/rank-$PROCESS_ID.preflight-ok"
        wait_for_files "$CONTRACT_DIR" 'rank-*.preflight-ok' "$PROCESS_COUNT" "$DATASET_PROFILE preflight rendezvous"
        if (( PROCESS_ID == 0 )); then
            "$ENV_PREFIX/bin/python" - "$OUTPUT_ROOT/_PREFLIGHT_OK.json" "$CONTRACT_SHA" "$AVAILABLE_BYTES" \
                "$PROCESS_COUNT" "$EXPECTED_LOCAL_DEVICES" "$OPENPI_GPU_NAMES" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "available_local_bytes_rank0": int(sys.argv[3]),
    "contract_sha256": sys.argv[2],
    "gpu_names": sys.argv[6].split(";"),
    "local_devices": int(sys.argv[5]),
    "processes": int(sys.argv[4]),
    "status": "passed",
}, sort_keys=True) + "\n")
PY
        fi
        log "$DATASET_PROFILE preflight complete for rank $PROCESS_ID"
        exit 0
    fi
else
    EPISODES_FILE="$RANK_ROOT/pi05_two_tasks_episodes.txt"
    { seq 2800 2999; seq 11200 11399; } > "$EPISODES_FILE"
fi

if [[ "$DATASET_PROFILE" == "full100" && "$MODE" == "full" ]]; then
    CHUNKS=()
    while IFS= read -r chunk; do CHUNKS+=("$chunk"); done < <(printf 'chunk-%03d\n' {0..99})
    if (( PROCESS_ID == 1 )); then
        REVERSED_CHUNKS=()
        for (( index = ${#CHUNKS[@]} - 1; index >= 0; index-- )); do
            REVERSED_CHUNKS+=("${CHUNKS[index]}")
        done
        CHUNKS=("${REVERSED_CHUNKS[@]}")
    fi
else
    CHUNKS=(chunk-014 chunk-056)
fi

log "staging ${#CHUNKS[@]} data and RGB chunks"
for chunk in "${CHUNKS[@]}"; do
    download_tree "data $chunk" "${SOURCE_DATASET_OSS_URI%/}/data/$chunk" "$LOCAL_DATASET/data/$chunk"
    for camera in observation.rgb.zed_link_camera_0 observation.rgb.left_realsense_link_camera_0 observation.rgb.right_realsense_link_camera_0; do
        download_tree "$camera $chunk" \
            "${SOURCE_DATASET_OSS_URI%/}/videos/$camera/$chunk" \
            "$LOCAL_DATASET/videos/$camera/$chunk"
    done
done

if [[ "$DATASET_PROFILE" == "full100" && "$MODE" == "full" ]]; then
    validate_tree "data" "$LOCAL_DATASET/data" \
        "$EXPECTED_DATA_FILES" "$EXPECTED_DATA_BYTES" "$EXPECTED_DATA_MANIFEST_SHA256"
    validate_tree "left RGB" "$LOCAL_DATASET/videos/observation.rgb.left_realsense_link_camera_0" \
        "$EXPECTED_LEFT_RGB_FILES" "$EXPECTED_LEFT_RGB_BYTES" "$EXPECTED_LEFT_RGB_MANIFEST_SHA256"
    validate_tree "right RGB" "$LOCAL_DATASET/videos/observation.rgb.right_realsense_link_camera_0" \
        "$EXPECTED_RIGHT_RGB_FILES" "$EXPECTED_RIGHT_RGB_BYTES" "$EXPECTED_RIGHT_RGB_MANIFEST_SHA256"
    validate_tree "base RGB" "$LOCAL_DATASET/videos/observation.rgb.zed_link_camera_0" \
        "$EXPECTED_BASE_RGB_FILES" "$EXPECTED_BASE_RGB_BYTES" "$EXPECTED_BASE_RGB_MANIFEST_SHA256"
fi
"$ENV_PREFIX/bin/python" "$REPO_ROOT/scripts/strip_depth_from_info.py" "$LOCAL_DATASET"

STAGING_FINISHED_AT="$(date +%s)"
read -r STAGED_BYTES _ < <(du -sb "$LOCAL_DATASET")
"$ENV_PREFIX/bin/python" - "$STAGING_DIR/rank-$PROCESS_ID.json" "$PROCESS_ID" "$HOST_IP" \
    "$STAGING_STARTED_AT" "$STAGING_FINISHED_AT" "$STAGED_BYTES" "$AVAILABLE_BYTES" "$DATASET_PROFILE" "$MODE" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
path.write_text(json.dumps({
    "available_bytes_before_staging": int(sys.argv[7]),
    "dataset_profile": sys.argv[8],
    "elapsed_seconds": int(sys.argv[5]) - int(sys.argv[4]),
    "finished_at_unix": int(sys.argv[5]),
    "host_ip": sys.argv[3],
    "mode": sys.argv[9],
    "process_id": int(sys.argv[2]),
    "staged_bytes": int(sys.argv[6]),
    "started_at_unix": int(sys.argv[4]),
}, sort_keys=True) + "\n")
PY

mkdir -p "$ASSETS_BASE_DIR/pi05_b1k/$REPO_ID"
if (( CONTRACT_PROFILE )); then
    SHARED_NORM_DIR="$OUTPUT_ROOT/norm/$REPO_ID"
    SHARED_NORM_FILE="$SHARED_NORM_DIR/norm_stats.json"
    SHARED_NORM_PROVENANCE="$SHARED_NORM_DIR/provenance.json"
    SHARED_NORM_READY="$SHARED_NORM_DIR/_READY"
    mkdir -p "$SHARED_NORM_DIR"
    if (( PROCESS_ID == 0 )) && [[ ! -f "$SHARED_NORM_READY" ]]; then
        if [[ -e "$SHARED_NORM_FILE" || -e "$SHARED_NORM_PROVENANCE" ]]; then
            fail "incomplete shared norm stats exist without $SHARED_NORM_READY"
        fi
        mapfile -t NORM_EPISODES < "$EPISODES_FILE"
        log "computing $DATASET_PROFILE normalization statistics from at most $NORM_MAX_FRAMES frames"
        cd "$OPENPI_DIR"
        NORM_ARGS=(
            --config-name pi05_b1k
            --dataset-root "$LOCAL_DATASET"
            --repo-id "$REPO_ID"
            --batch-size "$BATCH_SIZE"
            --num-workers "$NUM_WORKERS"
            --max-frames "$NORM_MAX_FRAMES"
            --episodes-index "${NORM_EPISODES[@]}"
            --assets-base-dir "$ASSETS_BASE_DIR"
        )
        run_with_heartbeat "normalization statistics" \
            "$ENV_PREFIX/bin/python" scripts/compute_norm_stats.py "${NORM_ARGS[@]}"
        LOCAL_NORM_FILE="$ASSETS_BASE_DIR/pi05_b1k/$REPO_ID/norm_stats.json"
        require_file "$LOCAL_NORM_FILE"
        NORM_SHA="$(sha256sum "$LOCAL_NORM_FILE" | cut -d' ' -f1)"
        NORM_TEMP="$SHARED_NORM_FILE.tmp.$(hostname).$$"
        PROVENANCE_TEMP="$SHARED_NORM_PROVENANCE.tmp.$(hostname).$$"
        cp "$LOCAL_NORM_FILE" "$NORM_TEMP"
        "$ENV_PREFIX/bin/python" - "$PROVENANCE_TEMP" "$CONTRACT_SHA" "$EPISODE_LIST_SHA" \
            "$ANNOTATION_SHA" "$NORM_SHA" "$NORM_MAX_FRAMES" "$REPO_ID" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "annotation_sha256": sys.argv[4],
    "contract_sha256": sys.argv[2],
    "episode_list_sha256": sys.argv[3],
    "max_frames": int(sys.argv[6]),
    "norm_stats_sha256": sys.argv[5],
    "repo_id": sys.argv[7],
}, sort_keys=True) + "\n")
PY
        mv -f "$NORM_TEMP" "$SHARED_NORM_FILE"
        mv -f "$PROVENANCE_TEMP" "$SHARED_NORM_PROVENANCE"
        printf '%s\n' "$NORM_SHA" > "$SHARED_NORM_READY.tmp.$(hostname).$$"
        mv -f "$SHARED_NORM_READY.tmp.$(hostname).$$" "$SHARED_NORM_READY"
    fi
    wait_for_files "$SHARED_NORM_DIR" '_READY' 1 "normalization statistics" "$NORM_STATS_TIMEOUT"
    require_file "$SHARED_NORM_FILE"
    require_file "$SHARED_NORM_PROVENANCE"
    IFS='|' read -r NORM_SHA PROVENANCE_CONTRACT PROVENANCE_EPISODES PROVENANCE_ANNOTATION < <(
        "$ENV_PREFIX/bin/python" - "$SHARED_NORM_PROVENANCE" <<'PY'
import json
import sys

provenance = json.load(open(sys.argv[1]))
print("|".join([
    provenance["norm_stats_sha256"],
    provenance["contract_sha256"],
    provenance["episode_list_sha256"],
    provenance["annotation_sha256"],
]))
PY
    )
    [[ "$(sha256sum "$SHARED_NORM_FILE" | cut -d' ' -f1)" == "$NORM_SHA" ]] \
        || fail "shared norm stats checksum mismatch"
    [[ "$PROVENANCE_CONTRACT" == "$CONTRACT_SHA" && "$PROVENANCE_EPISODES" == "$EPISODE_LIST_SHA" \
        && "$PROVENANCE_ANNOTATION" == "$ANNOTATION_SHA" ]] \
        || fail "shared norm stats provenance does not match this training contract"
    cp "$SHARED_NORM_FILE" "$ASSETS_BASE_DIR/pi05_b1k/$REPO_ID/norm_stats.json"
else
    cp "$NORM_STATS_SOURCE" "$ASSETS_BASE_DIR/pi05_b1k/$REPO_ID/norm_stats.json"
    CONTRACT_SHA=""
fi

export OPENPI_REQUIRE_FINITE_METRICS=1
export OPENPI_VERIFY_FINAL_CHECKPOINT=1
export OPENPI_SUCCESS_PATH="$OUTPUT_ROOT/_SUCCESS.rank0.json"
export PREFLIGHT_OUTPUT_DIR="$RANK_ROOT/subtask_preflight"
LR_DECAY_STEPS_VALUE="$TRAIN_STEPS"
(( LR_DECAY_STEPS_VALUE > 1000 )) || LR_DECAY_STEPS_VALUE=1001
TRAIN_EXPECTED_TASKS="$EXPECTED_TASKS"
if [[ "$DATASET_PROFILE" == "full100" && "$MODE" == "smoke" ]]; then
    TRAIN_EXPECTED_TASKS=2
fi
TRAIN_SCRIPT="$REPO_ROOT/scripts/train_pi05_subtask_b1k.sh"
TRAIN_SUBTASK_PREDICTION=1
TRAIN_MAX_TOKEN_LEN=512
if [[ "${FRAMESAMP_MODUL:-0}" == 1 ]]; then
    require_file "${MEMORY_CACHE_DIR:?Set MEMORY_CACHE_DIR}/manifest.json"
fi

log "starting multi-host $MODE; log: $RANK_LOG"

run_with_heartbeat "multi-host training" env \
    OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
    OPENPI_TRAINING_CONTRACT_SHA256="$CONTRACT_SHA" \
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
    TRAIN_EPOCHS="$TRAIN_EPOCHS" \
    TRAIN_STEPS="$TRAIN_STEPS" \
    EXPECTED_TASKS="$TRAIN_EXPECTED_TASKS" \
    EXPECTED_TASK_INDICES="$EXPECTED_TASK_INDICES" \
    EXPECTED_RETAINED_EPISODES="$EXPECTED_RETAINED_EPISODES" \
    EXPECTED_RETAINED_FRAMES="$EXPECTED_RETAINED_FRAMES" \
    EXPECTED_TRAINING_SAMPLES="$EXPECTED_TRAINING_SAMPLES" \
    EXPECTED_BATCHES_PER_EPOCH="$EXPECTED_BATCHES_PER_EPOCH" \
    EXPECTED_TRAIN_STEPS="$EXPECTED_TRAIN_STEPS" \
    SAVE_INTERVAL="$SAVE_INTERVAL" \
    KEEP_PERIOD="$KEEP_PERIOD" \
    LOG_INTERVAL="$LOG_INTERVAL" \
    FRAME_STRIDE="$FRAME_STRIDE" \
    LR_DECAY_STEPS="$LR_DECAY_STEPS_VALUE" \
    RESUME="$RESUME" \
    SUBTASK_PREDICTION="$TRAIN_SUBTASK_PREDICTION" \
    SUBTASK_ACTION_CONDITIONING="$SUBTASK_ACTION_CONDITIONING" \
    FRAMESAMP_MODUL="${FRAMESAMP_MODUL:-0}" \
    MEMORY_CACHE_DIR="${MEMORY_CACHE_DIR:-}" \
    MEMORY_NUM_FRAMES="${MEMORY_NUM_FRAMES:-32}" \
    MEMORY_GRID_SIZE="${MEMORY_GRID_SIZE:-4}" \
    MEMORY_FRAME_STRIDE="${MEMORY_FRAME_STRIDE:-16}" \
    MEMORY_CAMERA="${MEMORY_CAMERA:-base_0_rgb}" \
    INIT_PARAMS="${INIT_PARAMS:-}" \
    WHOLE_TASK_PROB=0 \
    MAX_TOKEN_LEN="$TRAIN_MAX_TOKEN_LEN" \
    MAX_SUBTASK_LEN=64 \
    SUBTASK_LOSS_WEIGHT=1.0 \
    SUBTASK_LOSS_CHUNK_SIZE=8 \
    SUBTASK_EMPTY_LABEL_POLICY=drop_episode \
    SUBTASK_PROMPTS="$LOCAL_DATASET/meta/subtask_segments.parquet" \
    NORM_MAX_FRAMES="$NORM_MAX_FRAMES" \
    bash "$TRAIN_SCRIPT" full
