#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$REPO_ROOT"

QUEUE="${QUEUE:-demand_general}"
GPU_CARD_TYPE="${GPU_CARD_TYPE:-}"
CLUSTER_FILE="${CLUSTER_FILE:-nebula/multihost_cluster.json}"
ENTRY_FILE="nebula/train_pi05_multihost.py"
ALGO_NAME="${ALGO_NAME:-}"
AIHUB_MODEL_NAME="${AIHUB_MODEL_NAME:-behavior}"
MODE="${MODE:-discover}"
case "$MODE" in
    discover|preflight|smoke|full) ;;
    *) echo "MODE must be discover, preflight, smoke, or full" >&2; exit 2 ;;
esac
DATASET_PROFILE="${DATASET_PROFILE:-two_tasks}"
case "$DATASET_PROFILE" in
    two_tasks|full100) ;;
    *) echo "DATASET_PROFILE must be two_tasks or full100" >&2; exit 2 ;;
esac

PROCESS_COUNT="${PROCESS_COUNT:-2}"
EXPECTED_LOCAL_DEVICES="${EXPECTED_LOCAL_DEVICES:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
EXPECTED_GLOBAL_DEVICES="${EXPECTED_GLOBAL_DEVICES:-$((PROCESS_COUNT * EXPECTED_LOCAL_DEVICES))}"
COORDINATOR_PORT="${COORDINATOR_PORT:-53791}"
RENDEZVOUS_TIMEOUT="${RENDEZVOUS_TIMEOUT:-1200}"
NORM_STATS_TIMEOUT="${NORM_STATS_TIMEOUT:-21600}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-500000}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
LOCAL_DEVICE_IDS="${LOCAL_DEVICE_IDS:-0,1,2,3,4,5,6,7}"

if [[ "$DATASET_PROFILE" == "full100" ]]; then
    ALGO_NAME="${ALGO_NAME:-pytorch2120}"
    BATCH_SIZE="${BATCH_SIZE:-192}"
    FRAME_STRIDE="${FRAME_STRIDE:-1}"
    TRAIN_EPOCHS="${TRAIN_EPOCHS:-2}"
    if [[ "$MODE" == "smoke" ]]; then
        TRAIN_EPOCHS=""
    fi
    EXPECTED_TASKS="${EXPECTED_TASKS:-100}"
    EXPECTED_RETAINED_EPISODES="${EXPECTED_RETAINED_EPISODES:-16051}"
    EXPECTED_RETAINED_FRAMES="${EXPECTED_RETAINED_FRAMES:-162141964}"
    EXPECTED_BATCHES_PER_EPOCH="${EXPECTED_BATCHES_PER_EPOCH:-844489}"
    EXPECTED_TRAIN_STEPS="${EXPECTED_TRAIN_STEPS:-1688978}"
    EXPECTED_EPISODE_LIST_SHA256="${EXPECTED_EPISODE_LIST_SHA256:-6020fdb366f39181787305aff5a3dd2e538c4d11eb1bdbb34c853c77a9973e44}"
    EXPECTED_CONTRACT_SHA256="${EXPECTED_CONTRACT_SHA256:-f0ef514971bc60f43748af954e1ed6930e3c719c620d1753a93072e61d9b57a9}"
    if [[ "$MODE" == "full" ]]; then
        TRAIN_STEPS="${TRAIN_STEPS:-$EXPECTED_TRAIN_STEPS}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-25000}"
        KEEP_PERIOD="${KEEP_PERIOD:-250000}"
    else
        TRAIN_STEPS="${TRAIN_STEPS:-3}"
        SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
        KEEP_PERIOD="${KEEP_PERIOD:-250000}"
    fi
    LOG_INTERVAL="${LOG_INTERVAL:-100}"
    if [[ "$MODE" == "smoke" ]]; then
        REPO_ID="${REPO_ID:-b1k_full100_joint_subtask_v1_s1_smoke}"
    else
        REPO_ID="${REPO_ID:-b1k_full100_joint_subtask_v1_drop_episode_s1}"
    fi
    LOCAL_DATASET_NAME="${LOCAL_DATASET_NAME:-full100_joint_subtask_v1_s1}"
    NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-}"
    MIN_LOCAL_BYTES="${MIN_LOCAL_BYTES:-1200000000000}"
    EXPECTED_META_FILES="${EXPECTED_META_FILES:-104}"
    EXPECTED_META_BYTES="${EXPECTED_META_BYTES:-5977514}"
    EXPECTED_META_MANIFEST_SHA256="${EXPECTED_META_MANIFEST_SHA256:-}"
    EXPECTED_DATA_FILES="${EXPECTED_DATA_FILES:-955}"
    EXPECTED_DATA_BYTES="${EXPECTED_DATA_BYTES:-74290115299}"
    EXPECTED_DATA_MANIFEST_SHA256="${EXPECTED_DATA_MANIFEST_SHA256:-}"
    EXPECTED_LEFT_RGB_FILES="${EXPECTED_LEFT_RGB_FILES:-1117}"
    EXPECTED_LEFT_RGB_BYTES="${EXPECTED_LEFT_RGB_BYTES:-215852639464}"
    EXPECTED_LEFT_RGB_MANIFEST_SHA256="${EXPECTED_LEFT_RGB_MANIFEST_SHA256:-}"
    EXPECTED_RIGHT_RGB_FILES="${EXPECTED_RIGHT_RGB_FILES:-1119}"
    EXPECTED_RIGHT_RGB_BYTES="${EXPECTED_RIGHT_RGB_BYTES:-216418530281}"
    EXPECTED_RIGHT_RGB_MANIFEST_SHA256="${EXPECTED_RIGHT_RGB_MANIFEST_SHA256:-}"
    EXPECTED_BASE_RGB_FILES="${EXPECTED_BASE_RGB_FILES:-3050}"
    EXPECTED_BASE_RGB_BYTES="${EXPECTED_BASE_RGB_BYTES:-569990749398}"
    EXPECTED_BASE_RGB_MANIFEST_SHA256="${EXPECTED_BASE_RGB_MANIFEST_SHA256:-}"
else
    ALGO_NAME="${ALGO_NAME:-pytorch280}"
    BATCH_SIZE="${BATCH_SIZE:-64}"
    FRAME_STRIDE="${FRAME_STRIDE:-1}"
    TRAIN_EPOCHS="${TRAIN_EPOCHS:-}"
    TRAIN_STEPS="${TRAIN_STEPS:-3}"
    SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
    KEEP_PERIOD="${KEEP_PERIOD:-5000}"
    LOG_INTERVAL="${LOG_INTERVAL:-1}"
    EXPECTED_TASKS="${EXPECTED_TASKS:-}"
    EXPECTED_RETAINED_EPISODES="${EXPECTED_RETAINED_EPISODES:-}"
    EXPECTED_RETAINED_FRAMES="${EXPECTED_RETAINED_FRAMES:-}"
    EXPECTED_BATCHES_PER_EPOCH="${EXPECTED_BATCHES_PER_EPOCH:-}"
    EXPECTED_TRAIN_STEPS="${EXPECTED_TRAIN_STEPS:-}"
    EXPECTED_EPISODE_LIST_SHA256="${EXPECTED_EPISODE_LIST_SHA256:-}"
    EXPECTED_CONTRACT_SHA256="${EXPECTED_CONTRACT_SHA256:-}"
    REPO_ID="${REPO_ID:-b1k_two_tasks_joint_subtask_s1}"
    LOCAL_DATASET_NAME="${LOCAL_DATASET_NAME:-two_tasks_joint_subtask_s1}"
    NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-$REPO_ROOT/nebula/pi05_two_tasks_norm_stats.json}"
    MIN_LOCAL_BYTES="${MIN_LOCAL_BYTES:-0}"
    EXPECTED_META_FILES="${EXPECTED_META_FILES:-}"
    EXPECTED_META_BYTES="${EXPECTED_META_BYTES:-}"
    EXPECTED_META_MANIFEST_SHA256="${EXPECTED_META_MANIFEST_SHA256:-}"
    EXPECTED_DATA_FILES="${EXPECTED_DATA_FILES:-}"
    EXPECTED_DATA_BYTES="${EXPECTED_DATA_BYTES:-}"
    EXPECTED_DATA_MANIFEST_SHA256="${EXPECTED_DATA_MANIFEST_SHA256:-}"
    EXPECTED_LEFT_RGB_FILES="${EXPECTED_LEFT_RGB_FILES:-}"
    EXPECTED_LEFT_RGB_BYTES="${EXPECTED_LEFT_RGB_BYTES:-}"
    EXPECTED_LEFT_RGB_MANIFEST_SHA256="${EXPECTED_LEFT_RGB_MANIFEST_SHA256:-}"
    EXPECTED_RIGHT_RGB_FILES="${EXPECTED_RIGHT_RGB_FILES:-}"
    EXPECTED_RIGHT_RGB_BYTES="${EXPECTED_RIGHT_RGB_BYTES:-}"
    EXPECTED_RIGHT_RGB_MANIFEST_SHA256="${EXPECTED_RIGHT_RGB_MANIFEST_SHA256:-}"
    EXPECTED_BASE_RGB_FILES="${EXPECTED_BASE_RGB_FILES:-}"
    EXPECTED_BASE_RGB_BYTES="${EXPECTED_BASE_RGB_BYTES:-}"
    EXPECTED_BASE_RGB_MANIFEST_SHA256="${EXPECTED_BASE_RGB_MANIFEST_SHA256:-}"
fi
RESUME="${RESUME:-0}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-pi05_${DATASET_PROFILE}_multihost_${MODE}_$STAMP}"
ATTEMPT_ID="${ATTEMPT_ID:-attempt_$STAMP}"
JOB_NAME="${JOB_NAME:-pi05-${DATASET_PROFILE}-multihost-${MODE}-$STAMP}"
EXP_NAME="${EXP_NAME:-pi05_b1k_${DATASET_PROFILE}_multihost_${MODE}_$STAMP}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_multihost}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$OUTPUT_BASE/$RUN_ID}"

SOURCE_DATASET="${SOURCE_DATASET:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542}"
ENV_ARCHIVE="${ENV_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05.tar.gz}"
OPENPI_CACHE_ARCHIVE="${OPENPI_CACHE_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05-openpi-cache.tar}"
SUBTASK_SEGMENTS_SOURCE="${SUBTASK_SEGMENTS_SOURCE:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/annotations/subtasks/v1/subtask_segments.parquet}"
SUBTASK_SEGMENTS_SHA256="${SUBTASK_SEGMENTS_SHA256:-870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d}"
OSS_MOUNT_PREFIX="${OSS_MOUNT_PREFIX:-/data/oss_bucket_0}"
OSSUTIL_SOURCE="${OSSUTIL_SOURCE:-/usr/local/bin/ossutil}"
OSSUTIL_SHA256="${OSSUTIL_SHA256:-c97d5da7fe472180645ae8a6c2e8e185474f00fd6035c047be0a0c91b57d4728}"
OSSUTIL_JOBS="${OSSUTIL_JOBS:-32}"
OSSUTIL_PARALLEL="${OSSUTIL_PARALLEL:-8}"
OSS_DOWNLOAD_ENDPOINT="${OSS_DOWNLOAD_ENDPOINT:-$OSS_ENDPOINT}"

oss_uri_for_mount_path() {
    local path="$1"
    [[ "$path" == "$OSS_MOUNT_PREFIX/"* ]] || {
        echo "Cannot map path outside $OSS_MOUNT_PREFIX to OSS: $path" >&2
        return 1
    }
    printf 'oss://%s/%s' "${OSS_BUCKET%/}" "${path#"$OSS_MOUNT_PREFIX/"}"
}

SOURCE_DATASET_OSS_URI="${SOURCE_DATASET_OSS_URI:-$(oss_uri_for_mount_path "$SOURCE_DATASET")}"
ENV_ARCHIVE_OSS_URI="${ENV_ARCHIVE_OSS_URI:-$(oss_uri_for_mount_path "$ENV_ARCHIVE")}"
OPENPI_CACHE_ARCHIVE_OSS_URI="${OPENPI_CACHE_ARCHIVE_OSS_URI:-$(oss_uri_for_mount_path "$OPENPI_CACHE_ARCHIVE")}"
SUBTASK_SEGMENTS_OSS_URI="${SUBTASK_SEGMENTS_OSS_URI:-$(oss_uri_for_mount_path "$SUBTASK_SEGMENTS_SOURCE")}"

REQUIRED_INPUTS=("$ENV_ARCHIVE" "$OPENPI_CACHE_ARCHIVE" "$SUBTASK_SEGMENTS_SOURCE" "$SOURCE_DATASET/meta/info.json")
if [[ -n "$NORM_STATS_SOURCE" ]]; then
    REQUIRED_INPUTS+=("$NORM_STATS_SOURCE")
fi
for path in "${REQUIRED_INPUTS[@]}"; do
    [[ -s "$path" ]] || { echo "Required input missing or empty: $path" >&2; exit 1; }
done

tree_manifest_summary() {
    python3 - "$1" <<'PY'
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
}

validate_source_tree() {
    local label="$1" root="$2" expected_files="$3" expected_bytes="$4" hash_variable="$5"
    local actual_files actual_bytes actual_hash expected_hash
    [[ -d "$root" ]] || { echo "Required source tree missing: $root" >&2; exit 1; }
    read -r actual_files actual_bytes actual_hash < <(tree_manifest_summary "$root")
    [[ "$actual_files" == "$expected_files" && "$actual_bytes" == "$expected_bytes" ]] || {
        echo "$label source mismatch: files=$actual_files/$expected_files bytes=$actual_bytes/$expected_bytes" >&2
        exit 1
    }
    expected_hash="${!hash_variable}"
    if [[ -n "$expected_hash" && "$actual_hash" != "$expected_hash" ]]; then
        echo "$label source manifest mismatch: $actual_hash != $expected_hash" >&2
        exit 1
    fi
    printf -v "$hash_variable" '%s' "$actual_hash"
}

if [[ "$DATASET_PROFILE" == "full100" ]]; then
    [[ -z "$NORM_STATS_SOURCE" ]] || { echo "full100 computes dedicated norm stats; NORM_STATS_SOURCE must be empty" >&2; exit 2; }
    validate_source_tree "metadata" "$SOURCE_DATASET/meta" \
        "$EXPECTED_META_FILES" "$EXPECTED_META_BYTES" EXPECTED_META_MANIFEST_SHA256
    validate_source_tree "data" "$SOURCE_DATASET/data" \
        "$EXPECTED_DATA_FILES" "$EXPECTED_DATA_BYTES" EXPECTED_DATA_MANIFEST_SHA256
    validate_source_tree "left RGB" "$SOURCE_DATASET/videos/observation.rgb.left_realsense_link_camera_0" \
        "$EXPECTED_LEFT_RGB_FILES" "$EXPECTED_LEFT_RGB_BYTES" EXPECTED_LEFT_RGB_MANIFEST_SHA256
    validate_source_tree "right RGB" "$SOURCE_DATASET/videos/observation.rgb.right_realsense_link_camera_0" \
        "$EXPECTED_RIGHT_RGB_FILES" "$EXPECTED_RIGHT_RGB_BYTES" EXPECTED_RIGHT_RGB_MANIFEST_SHA256
    validate_source_tree "base RGB" "$SOURCE_DATASET/videos/observation.rgb.zed_link_camera_0" \
        "$EXPECTED_BASE_RGB_FILES" "$EXPECTED_BASE_RGB_BYTES" EXPECTED_BASE_RGB_MANIFEST_SHA256
    [[ "$PROCESS_COUNT" == 2 && "$EXPECTED_LOCAL_DEVICES" == 8 && "$EXPECTED_GLOBAL_DEVICES" == 16 ]] \
        || { echo "full100 requires 2 workers x 8 GPUs" >&2; exit 2; }
    [[ "$FSDP_DEVICES" == 8 && "$BATCH_SIZE" == 192 && "$FRAME_STRIDE" == 1 ]] \
        || { echo "full100 requires FSDP=8, batch=192, stride=1" >&2; exit 2; }
    if [[ "$MODE" != "smoke" && "$TRAIN_EPOCHS" != 2 ]]; then
        echo "full100 $MODE requires exactly 2 epochs" >&2
        exit 2
    fi
    if [[ "$MODE" == "full" && "$TRAIN_STEPS" != "$EXPECTED_TRAIN_STEPS" ]]; then
        echo "full100 TRAIN_STEPS=$TRAIN_STEPS must equal $EXPECTED_TRAIN_STEPS" >&2
        exit 2
    fi
fi
case "$RESUME" in
    0) [[ ! -e "$OUTPUT_ROOT" ]] || { echo "Output already exists: $OUTPUT_ROOT" >&2; exit 1; } ;;
    1)
        [[ "$MODE" == "full" ]] || { echo "RESUME=1 is only valid for MODE=full" >&2; exit 2; }
        [[ -d "$OUTPUT_ROOT/checkpoints/pi05_b1k/$EXP_NAME" ]] \
            || { echo "Resume checkpoint directory is missing under $OUTPUT_ROOT" >&2; exit 1; }
        ;;
    *) echo "RESUME must be 0 or 1" >&2; exit 2 ;;
esac
[[ -x "$OSSUTIL_SOURCE" ]] || { echo "ossutil executable missing: $OSSUTIL_SOURCE" >&2; exit 1; }
[[ "$(sha256sum "$OSSUTIL_SOURCE" | cut -d' ' -f1)" == "$OSSUTIL_SHA256" ]] || {
    echo "ossutil checksum mismatch: $OSSUTIL_SOURCE" >&2
    exit 1
}
for value in "$OSSUTIL_JOBS" "$OSSUTIL_PARALLEL"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "ossutil concurrency must be a positive integer" >&2; exit 2; }
done
for value in "$RENDEZVOUS_TIMEOUT" "$NORM_STATS_TIMEOUT"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "timeouts must be positive integers" >&2; exit 2; }
done
for secret in "$OSS_ACCESS_ID" "$OSS_ACCESS_KEY"; do
    [[ "$secret" != *','* && "$secret" != *$'\n'* && "$secret" != *$'\r'* ]] || {
        echo "OSS credential contains a character unsupported by AIHub --env" >&2
        exit 1
    }
done
if [[ "$(sha256sum "$SUBTASK_SEGMENTS_SOURCE" | cut -d' ' -f1)" != "$SUBTASK_SEGMENTS_SHA256" ]]; then
    echo "Subtask annotation checksum mismatch: $SUBTASK_SEGMENTS_SOURCE" >&2
    exit 1
fi
if (( BATCH_SIZE % EXPECTED_GLOBAL_DEVICES != 0 )); then
    echo "BATCH_SIZE=$BATCH_SIZE must be divisible by global devices=$EXPECTED_GLOBAL_DEVICES" >&2
    exit 1
fi

BUNDLE_PARENT="${BUNDLE_PARENT:-${TMPDIR:-/tmp}}"
mkdir -p "$BUNDLE_PARENT"
BUNDLE_DIR="$(mktemp -d "$BUNDLE_PARENT/pi05-multihost-bundle.XXXXXX")"
cleanup_bundle() {
    if [[ "${KEEP_BUNDLE:-0}" == 1 ]]; then
        printf 'Kept submission bundle: %s\n' "$BUNDLE_DIR"
    else
        rm -rf -- "$BUNDLE_DIR"
    fi
}
trap cleanup_bundle EXIT

mkdir -p \
    "$BUNDLE_DIR/openpi/packages" \
    "$BUNDLE_DIR/openpi/scripts/b1k" \
    "$BUNDLE_DIR/scripts" \
    "$BUNDLE_DIR/nebula"
cp -a "$REPO_ROOT/openpi/src" "$BUNDLE_DIR/openpi/"
cp -a "$REPO_ROOT/openpi/packages/openpi-client" "$BUNDLE_DIR/openpi/packages/"
cp -a \
    "$REPO_ROOT/openpi/pyproject.toml" \
    "$REPO_ROOT/openpi/README.md" \
    "$REPO_ROOT/openpi/LICENSE" \
    "$BUNDLE_DIR/openpi/"
cp -a "$REPO_ROOT/openpi/scripts/compute_norm_stats.py" "$BUNDLE_DIR/openpi/scripts/"
cp -a \
    "$REPO_ROOT/openpi/scripts/b1k/train_b1k.py" \
    "$REPO_ROOT/openpi/scripts/b1k/check_subtask_data.py" \
    "$REPO_ROOT/openpi/scripts/b1k/distributed_preflight.py" \
    "$BUNDLE_DIR/openpi/scripts/b1k/"
cp -a \
    "$REPO_ROOT/scripts/train_pi05_b1k.sh" \
    "$REPO_ROOT/scripts/train_pi05_subtask_b1k.sh" \
    "$REPO_ROOT/scripts/strip_depth_from_info.py" \
    "$BUNDLE_DIR/scripts/"
cp -a \
    "$REPO_ROOT/nebula/train_pi05_multihost.py" \
    "$REPO_ROOT/nebula/train_pi05_multihost.sh" \
    "$BUNDLE_DIR/nebula/"
install -m 0755 "$OSSUTIL_SOURCE" "$BUNDLE_DIR/nebula/ossutil"
cp -a "$REPO_ROOT/$CLUSTER_FILE" "$BUNDLE_DIR/$CLUSTER_FILE"
BUNDLED_NORM_STATS=""
if [[ -n "$NORM_STATS_SOURCE" ]]; then
    BUNDLED_NORM_STATS="nebula/norm_stats.json"
    cp -a "$NORM_STATS_SOURCE" "$BUNDLE_DIR/$BUNDLED_NORM_STATS"
fi
find "$BUNDLE_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$BUNDLE_DIR" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

write_setting() { printf 'export %s=%q\n' "$1" "$2"; }
{
    write_setting MODE "$MODE"
    write_setting DATASET_PROFILE "$DATASET_PROFILE"
    write_setting RUN_ID "$RUN_ID"
    write_setting ATTEMPT_ID "$ATTEMPT_ID"
    write_setting EXP_NAME "$EXP_NAME"
    write_setting OUTPUT_ROOT "$OUTPUT_ROOT"
    write_setting RESUME "$RESUME"
    write_setting PROCESS_COUNT "$PROCESS_COUNT"
    write_setting EXPECTED_LOCAL_DEVICES "$EXPECTED_LOCAL_DEVICES"
    write_setting EXPECTED_GLOBAL_DEVICES "$EXPECTED_GLOBAL_DEVICES"
    write_setting COORDINATOR_PORT "$COORDINATOR_PORT"
    write_setting RENDEZVOUS_TIMEOUT "$RENDEZVOUS_TIMEOUT"
    write_setting NORM_STATS_TIMEOUT "$NORM_STATS_TIMEOUT"
    write_setting CUDA_DEVICES "$CUDA_DEVICES"
    write_setting LOCAL_DEVICE_IDS "$LOCAL_DEVICE_IDS"
    write_setting FSDP_DEVICES "$FSDP_DEVICES"
    write_setting NUM_WORKERS "$NUM_WORKERS"
    write_setting BATCH_SIZE "$BATCH_SIZE"
    write_setting FRAME_STRIDE "$FRAME_STRIDE"
    write_setting TRAIN_EPOCHS "$TRAIN_EPOCHS"
    write_setting TRAIN_STEPS "$TRAIN_STEPS"
    write_setting SAVE_INTERVAL "$SAVE_INTERVAL"
    write_setting KEEP_PERIOD "$KEEP_PERIOD"
    write_setting LOG_INTERVAL "$LOG_INTERVAL"
    write_setting NORM_MAX_FRAMES "$NORM_MAX_FRAMES"
    write_setting EXPECTED_TASKS "$EXPECTED_TASKS"
    write_setting EXPECTED_RETAINED_EPISODES "$EXPECTED_RETAINED_EPISODES"
    write_setting EXPECTED_RETAINED_FRAMES "$EXPECTED_RETAINED_FRAMES"
    write_setting EXPECTED_BATCHES_PER_EPOCH "$EXPECTED_BATCHES_PER_EPOCH"
    write_setting EXPECTED_TRAIN_STEPS "$EXPECTED_TRAIN_STEPS"
    write_setting EXPECTED_EPISODE_LIST_SHA256 "$EXPECTED_EPISODE_LIST_SHA256"
    write_setting EXPECTED_CONTRACT_SHA256 "$EXPECTED_CONTRACT_SHA256"
    write_setting MIN_LOCAL_BYTES "$MIN_LOCAL_BYTES"
    write_setting EXPECTED_META_FILES "$EXPECTED_META_FILES"
    write_setting EXPECTED_META_BYTES "$EXPECTED_META_BYTES"
    write_setting EXPECTED_META_MANIFEST_SHA256 "$EXPECTED_META_MANIFEST_SHA256"
    write_setting EXPECTED_DATA_FILES "$EXPECTED_DATA_FILES"
    write_setting EXPECTED_DATA_BYTES "$EXPECTED_DATA_BYTES"
    write_setting EXPECTED_DATA_MANIFEST_SHA256 "$EXPECTED_DATA_MANIFEST_SHA256"
    write_setting EXPECTED_LEFT_RGB_FILES "$EXPECTED_LEFT_RGB_FILES"
    write_setting EXPECTED_LEFT_RGB_BYTES "$EXPECTED_LEFT_RGB_BYTES"
    write_setting EXPECTED_LEFT_RGB_MANIFEST_SHA256 "$EXPECTED_LEFT_RGB_MANIFEST_SHA256"
    write_setting EXPECTED_RIGHT_RGB_FILES "$EXPECTED_RIGHT_RGB_FILES"
    write_setting EXPECTED_RIGHT_RGB_BYTES "$EXPECTED_RIGHT_RGB_BYTES"
    write_setting EXPECTED_RIGHT_RGB_MANIFEST_SHA256 "$EXPECTED_RIGHT_RGB_MANIFEST_SHA256"
    write_setting EXPECTED_BASE_RGB_FILES "$EXPECTED_BASE_RGB_FILES"
    write_setting EXPECTED_BASE_RGB_BYTES "$EXPECTED_BASE_RGB_BYTES"
    write_setting EXPECTED_BASE_RGB_MANIFEST_SHA256 "$EXPECTED_BASE_RGB_MANIFEST_SHA256"
    write_setting LOCAL_DATASET_NAME "$LOCAL_DATASET_NAME"
    write_setting SOURCE_DATASET_OSS_URI "$SOURCE_DATASET_OSS_URI"
    write_setting ENV_ARCHIVE_OSS_URI "$ENV_ARCHIVE_OSS_URI"
    write_setting OPENPI_CACHE_ARCHIVE_OSS_URI "$OPENPI_CACHE_ARCHIVE_OSS_URI"
    write_setting NORM_STATS_SOURCE "$BUNDLED_NORM_STATS"
    write_setting SUBTASK_SEGMENTS_OSS_URI "$SUBTASK_SEGMENTS_OSS_URI"
    write_setting SUBTASK_SEGMENTS_SHA256 "$SUBTASK_SEGMENTS_SHA256"
    write_setting REPO_ID "$REPO_ID"
    write_setting OSS_DOWNLOAD_ENDPOINT "$OSS_DOWNLOAD_ENDPOINT"
    write_setting OSSUTIL_SHA256 "$OSSUTIL_SHA256"
    write_setting OSSUTIL_JOBS "$OSSUTIL_JOBS"
    write_setting OSSUTIL_PARALLEL "$OSSUTIL_PARALLEL"
} > "$BUNDLE_DIR/nebula/multihost_job.env"

BUNDLE_SIZE="$(du -sh "$BUNDLE_DIR" | cut -f1)"
printf 'AIHUB_MODEL_NAME: %s\n' "$AIHUB_MODEL_NAME"
printf 'JOB_NAME: %s\n' "$JOB_NAME"
printf 'RUN_ID: %s\n' "$RUN_ID"
printf 'ATTEMPT_ID: %s\n' "$ATTEMPT_ID"
printf 'MODE: %s\n' "$MODE"
printf 'DATASET_PROFILE: %s\n' "$DATASET_PROFILE"
printf 'QUEUE: %s\n' "$QUEUE"
printf 'WORKERS: %d x %d GPUs (global %d, fsdp %d)\n' "$PROCESS_COUNT" "$EXPECTED_LOCAL_DEVICES" "$EXPECTED_GLOBAL_DEVICES" "$FSDP_DEVICES"
printf 'BATCH: global %d, local %d\n' "$BATCH_SIZE" "$((BATCH_SIZE / PROCESS_COUNT))"
printf 'TRAINING: stride=%s epochs=%s steps=%s resume=%s\n' "$FRAME_STRIDE" "${TRAIN_EPOCHS:-n/a}" "$TRAIN_STEPS" "$RESUME"
printf 'OUTPUT: %s\n' "$OUTPUT_ROOT"
printf 'BUNDLE: %s (%s)\n' "$BUNDLE_DIR" "$BUNDLE_SIZE"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: submission skipped."
    exit 0
fi

WORKER_ENV="OSS_ACCESS_KEY_ID=$OSS_ACCESS_ID,OSS_ACCESS_KEY_SECRET=$OSS_ACCESS_KEY"
mdl_args=(
    "--token=$AI_HUB_TOKEN"
    "--namespace=$AI_HUB_NAMESPACE"
    "--name=$AIHUB_MODEL_NAME"
    "--queue=$QUEUE"
    "--entry=$ENTRY_FILE"
    "--worker_count=$PROCESS_COUNT"
    "--file.cluster_file=$CLUSTER_FILE"
    "--oss_access_id=$OSS_ACCESS_ID"
    "--oss_access_key=$OSS_ACCESS_KEY"
    "--oss_bucket=$OSS_BUCKET"
    "--oss_endpoint=$OSS_ENDPOINT"
    "--env=$WORKER_ENV"
    "--job_name=$JOB_NAME"
    "--algo_name=$ALGO_NAME"
    "--oss_appendable=true"
    "--debug_mode"
    "--ignore=__pycache__,*.pyc,*.pyo"
)
if [[ -n "$GPU_CARD_TYPE" ]]; then
    mdl_args+=("--gpu_card_type=$GPU_CARD_TYPE")
fi

cd "$BUNDLE_DIR"
ai-hub-cli train mdl "${mdl_args[@]}"
