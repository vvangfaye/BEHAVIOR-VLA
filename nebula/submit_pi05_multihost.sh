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
ALGO_NAME="${ALGO_NAME:-pytorch280}"
AIHUB_MODEL_NAME="${AIHUB_MODEL_NAME:-behavior}"
MODE="${MODE:-discover}"
case "$MODE" in
    discover|preflight|smoke|full) ;;
    *) echo "MODE must be discover, preflight, smoke, or full" >&2; exit 2 ;;
esac

PROCESS_COUNT="${PROCESS_COUNT:-2}"
EXPECTED_LOCAL_DEVICES="${EXPECTED_LOCAL_DEVICES:-8}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EXPECTED_GLOBAL_DEVICES="${EXPECTED_GLOBAL_DEVICES:-$((PROCESS_COUNT * EXPECTED_LOCAL_DEVICES))}"
COORDINATOR_PORT="${COORDINATOR_PORT:-53791}"
RENDEZVOUS_TIMEOUT="${RENDEZVOUS_TIMEOUT:-1200}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TRAIN_STEPS="${TRAIN_STEPS:-3}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000000}"
NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-500000}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
LOCAL_DEVICE_IDS="${LOCAL_DEVICE_IDS:-0,1,2,3,4,5,6,7}"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-pi05_multihost_${MODE}_$STAMP}"
JOB_NAME="${JOB_NAME:-pi05-multihost-${MODE}-$STAMP}"
EXP_NAME="${EXP_NAME:-pi05_b1k_multihost_${MODE}_$STAMP}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_multihost}"
OUTPUT_ROOT="$OUTPUT_BASE/$RUN_ID"

SOURCE_DATASET="${SOURCE_DATASET:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542}"
ENV_ARCHIVE="${ENV_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05.tar.gz}"
OPENPI_CACHE_ARCHIVE="${OPENPI_CACHE_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05-openpi-cache.tar}"
NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-$REPO_ROOT/nebula/pi05_two_tasks_norm_stats.json}"
SUBTASK_SEGMENTS_SOURCE="${SUBTASK_SEGMENTS_SOURCE:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/annotations/subtasks/v1/subtask_segments.parquet}"
SUBTASK_SEGMENTS_SHA256="${SUBTASK_SEGMENTS_SHA256:-870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d}"
REPO_ID="${REPO_ID:-b1k_two_tasks_joint_subtask_s1}"

for path in "$ENV_ARCHIVE" "$OPENPI_CACHE_ARCHIVE" "$NORM_STATS_SOURCE" "$SUBTASK_SEGMENTS_SOURCE" \
    "$SOURCE_DATASET/meta/info.json"; do
    [[ -s "$path" ]] || { echo "Required input missing or empty: $path" >&2; exit 1; }
done
if [[ "$(sha256sum "$SUBTASK_SEGMENTS_SOURCE" | cut -d' ' -f1)" != "$SUBTASK_SEGMENTS_SHA256" ]]; then
    echo "Subtask annotation checksum mismatch: $SUBTASK_SEGMENTS_SOURCE" >&2
    exit 1
fi
if (( BATCH_SIZE % EXPECTED_GLOBAL_DEVICES != 0 )); then
    echo "BATCH_SIZE=$BATCH_SIZE must be divisible by global devices=$EXPECTED_GLOBAL_DEVICES" >&2
    exit 1
fi
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Output already exists: $OUTPUT_ROOT" >&2; exit 1; }

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
cp -a "$REPO_ROOT/$CLUSTER_FILE" "$BUNDLE_DIR/$CLUSTER_FILE"
find "$BUNDLE_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$BUNDLE_DIR" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

write_setting() { printf 'export %s=%q\n' "$1" "$2"; }
{
    write_setting MODE "$MODE"
    write_setting RUN_ID "$RUN_ID"
    write_setting EXP_NAME "$EXP_NAME"
    write_setting OUTPUT_ROOT "$OUTPUT_ROOT"
    write_setting PROCESS_COUNT "$PROCESS_COUNT"
    write_setting EXPECTED_LOCAL_DEVICES "$EXPECTED_LOCAL_DEVICES"
    write_setting EXPECTED_GLOBAL_DEVICES "$EXPECTED_GLOBAL_DEVICES"
    write_setting COORDINATOR_PORT "$COORDINATOR_PORT"
    write_setting RENDEZVOUS_TIMEOUT "$RENDEZVOUS_TIMEOUT"
    write_setting CUDA_DEVICES "$CUDA_DEVICES"
    write_setting LOCAL_DEVICE_IDS "$LOCAL_DEVICE_IDS"
    write_setting FSDP_DEVICES "$FSDP_DEVICES"
    write_setting NUM_WORKERS "$NUM_WORKERS"
    write_setting BATCH_SIZE "$BATCH_SIZE"
    write_setting TRAIN_STEPS "$TRAIN_STEPS"
    write_setting SAVE_INTERVAL "$SAVE_INTERVAL"
    write_setting NORM_MAX_FRAMES "$NORM_MAX_FRAMES"
    write_setting SOURCE_DATASET "$SOURCE_DATASET"
    write_setting ENV_ARCHIVE "$ENV_ARCHIVE"
    write_setting OPENPI_CACHE_ARCHIVE "$OPENPI_CACHE_ARCHIVE"
    write_setting NORM_STATS_SOURCE "nebula/pi05_two_tasks_norm_stats.json"
    write_setting SUBTASK_SEGMENTS_SOURCE "$SUBTASK_SEGMENTS_SOURCE"
    write_setting SUBTASK_SEGMENTS_SHA256 "$SUBTASK_SEGMENTS_SHA256"
    write_setting REPO_ID "$REPO_ID"
} > "$BUNDLE_DIR/nebula/multihost_job.env"
cp -a "$NORM_STATS_SOURCE" "$BUNDLE_DIR/nebula/pi05_two_tasks_norm_stats.json"

BUNDLE_SIZE="$(du -sh "$BUNDLE_DIR" | cut -f1)"
printf 'AIHUB_MODEL_NAME: %s\n' "$AIHUB_MODEL_NAME"
printf 'JOB_NAME: %s\n' "$JOB_NAME"
printf 'RUN_ID: %s\n' "$RUN_ID"
printf 'MODE: %s\n' "$MODE"
printf 'QUEUE: %s\n' "$QUEUE"
printf 'WORKERS: %d x %d GPUs (global %d, fsdp %d)\n' "$PROCESS_COUNT" "$EXPECTED_LOCAL_DEVICES" "$EXPECTED_GLOBAL_DEVICES" "$FSDP_DEVICES"
printf 'BATCH: global %d, local %d\n' "$BATCH_SIZE" "$((BATCH_SIZE / PROCESS_COUNT))"
printf 'OUTPUT: %s\n' "$OUTPUT_ROOT"
printf 'BUNDLE: %s (%s)\n' "$BUNDLE_DIR" "$BUNDLE_SIZE"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: submission skipped."
    exit 0
fi

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
