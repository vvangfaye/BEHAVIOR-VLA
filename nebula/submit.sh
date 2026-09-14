#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$REPO_ROOT"

QUEUE="${QUEUE:-demand_general}"

CLUSTER_FILE="${CLUSTER_FILE:-nebula/cluster.json}"
ENTRY_FILE="${ENTRY_FILE:-nebula/train_pi05_two_tasks.py}"
ALGO_NAME="${ALGO_NAME:-pytorch280}"
AIHUB_MODEL_NAME="${AIHUB_MODEL_NAME:-behavior}"
WORKER_COUNT="${WORKER_COUNT:-1}"
if [[ "$WORKER_COUNT" != "1" ]]; then
    echo "This OpenPI launcher supports exactly one 8-GPU worker; got WORKER_COUNT=$WORKER_COUNT" >&2
    exit 2
fi
STAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-pi05_b1k_2tasks_joint_subtask_s1_117542_$STAMP}"
JOB_NAME="${JOB_NAME:-pi05-joint-subtask-s1-$STAMP}"

SOURCE_DATASET="${SOURCE_DATASET:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542}"
LOCAL_DATASET="${LOCAL_DATASET:-/var/tmp/behavior-1k/two_tasks_joint_subtask_s1}"
ENV_ARCHIVE="${ENV_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05.tar.gz}"
ENV_PREFIX="${ENV_PREFIX:-/dev/shm/wangfaye/conda_envs/b1k-pi05}"
OPENPI_CACHE_ARCHIVE="${OPENPI_CACHE_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05-openpi-cache.tar}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/var/tmp/wangfaye/openpi-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_two_tasks_joint_subtask}"
SUBTASK_SEGMENTS_SOURCE="${SUBTASK_SEGMENTS_SOURCE:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/annotations/subtasks/v1/subtask_segments.parquet}"
SUBTASK_SEGMENTS_SHA256="${SUBTASK_SEGMENTS_SHA256:-870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d}"

if [[ ! -f "$OPENPI_CACHE_ARCHIVE" ]]; then
    echo "OpenPI cache archive not found: $OPENPI_CACHE_ARCHIVE" >&2
    exit 1
fi
if (( $(stat -c %s "$OPENPI_CACHE_ARCHIVE") < 10 * 1024 * 1024 * 1024 )); then
    echo "OpenPI cache archive is unexpectedly small: $OPENPI_CACHE_ARCHIVE" >&2
    exit 1
fi
if [[ ! -f "$SUBTASK_SEGMENTS_SOURCE" ]]; then
    echo "Subtask annotation file not found: $SUBTASK_SEGMENTS_SOURCE" >&2
    exit 1
fi
if [[ "$(sha256sum "$SUBTASK_SEGMENTS_SOURCE" | cut -d' ' -f1)" != "$SUBTASK_SEGMENTS_SHA256" ]]; then
    echo "Subtask annotation checksum mismatch: $SUBTASK_SEGMENTS_SOURCE" >&2
    exit 1
fi

BUNDLE_PARENT="${BUNDLE_PARENT:-${TMPDIR:-/tmp}}"
mkdir -p "$BUNDLE_PARENT"
BUNDLE_DIR="$(mktemp -d "$BUNDLE_PARENT/pi05-nebula-bundle.XXXXXX")"
cleanup_bundle() {
    if [[ "${KEEP_BUNDLE:-0}" == "1" ]]; then
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
    "$BUNDLE_DIR/openpi/scripts/b1k/"
cp -a \
    "$REPO_ROOT/scripts/train_pi05_b1k.sh" \
    "$REPO_ROOT/scripts/train_pi05_subtask_b1k.sh" \
    "$REPO_ROOT/scripts/strip_depth_from_info.py" \
    "$BUNDLE_DIR/scripts/"
cp -a \
    "$REPO_ROOT/nebula/cluster.json" \
    "$REPO_ROOT/nebula/train_pi05_two_tasks.py" \
    "$REPO_ROOT/nebula/train_pi05_two_tasks.sh" \
    "$REPO_ROOT/nebula/pi05_two_tasks_norm_stats.json" \
    "$BUNDLE_DIR/nebula/"
find "$BUNDLE_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$BUNDLE_DIR" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
BUNDLE_SIZE="$(du -sh "$BUNDLE_DIR" | cut -f1)"

mdl_args=(
    "--name=$AIHUB_MODEL_NAME"
    "--queue=$QUEUE"
    "--entry=$ENTRY_FILE"
    "--worker_count=$WORKER_COUNT"
    "--file.cluster_file=$CLUSTER_FILE"
    "--oss_access_id=$OSS_ACCESS_ID"
    "--oss_access_key=$OSS_ACCESS_KEY"
    "--oss_bucket=$OSS_BUCKET"
    "--oss_endpoint=$OSS_ENDPOINT"
    "--job_name=$JOB_NAME"
    "--algo_name=$ALGO_NAME"
    "--oss_appendable=true"
    "--debug_mode"
    "--ignore=__pycache__,*.pyc"
    "--env=SOURCE_DATASET=$SOURCE_DATASET"
    "--env=LOCAL_DATASET=$LOCAL_DATASET"
    "--env=ENV_ARCHIVE=$ENV_ARCHIVE"
    "--env=ENV_PREFIX=$ENV_PREFIX"
    "--env=OPENPI_CACHE_ARCHIVE=$OPENPI_CACHE_ARCHIVE"
    "--env=OPENPI_DATA_HOME=$OPENPI_DATA_HOME"
    "--env=OUTPUT_ROOT=$OUTPUT_ROOT"
    "--env=SUBTASK_SEGMENTS_SOURCE=$SUBTASK_SEGMENTS_SOURCE"
    "--env=SUBTASK_SEGMENTS_SHA256=$SUBTASK_SEGMENTS_SHA256"
    "--env=REPO_ID=b1k_two_tasks_joint_subtask_s1"
    "--env=EXP_NAME=$EXP_NAME"
    "--env=NUM_GPUS=8"
    "--env=CUDA_DEVICES=0,1,2,3,4,5,6,7"
    "--env=FSDP_DEVICES=8"
    "--env=NUM_WORKERS=8"
    "--env=BATCH_SIZE=64"
    "--env=TRAIN_STEPS=117542"
    "--env=SAVE_INTERVAL=5000"
    "--env=NORM_MAX_FRAMES=500000"
    "--env=FRAME_STRIDE=1"
    "--env=LR_DECAY_STEPS=117542"
    "--env=SUBTASK_EMPTY_LABEL_POLICY=drop_episode"
    "--env=MAX_TOKEN_LEN=512"
    "--env=MAX_SUBTASK_LEN=64"
    "--env=SUBTASK_LOSS_WEIGHT=1.0"
    "--env=SUBTASK_LOSS_CHUNK_SIZE=8"
    "--env=WHOLE_TASK_PROB=0"
)

if [[ -n "${GPU_CARD_TYPE:-}" ]]; then
    mdl_args+=("--gpu_card_type=$GPU_CARD_TYPE")
fi
if [[ "${RESUME:-0}" == "1" ]]; then
    mdl_args+=("--env=RESUME=1")
fi

printf 'AIHUB_MODEL_NAME: %s\n' "$AIHUB_MODEL_NAME"
printf 'JOB_NAME: %s\n' "$JOB_NAME"
printf 'EXP_NAME: %s\n' "$EXP_NAME"
printf 'QUEUE: %s\n' "$QUEUE"
printf 'ENTRY_FILE: %s\n' "$ENTRY_FILE"
printf 'SOURCE_DATASET: %s\n' "$SOURCE_DATASET"
printf 'LOCAL_DATASET: %s\n' "$LOCAL_DATASET"
printf 'SUBTASK_SEGMENTS: %s -> worker-local copy (sha256=%s)\n' "$SUBTASK_SEGMENTS_SOURCE" "$SUBTASK_SEGMENTS_SHA256"
printf 'OPENPI_CACHE: %s -> %s\n' "$OPENPI_CACHE_ARCHIVE" "$OPENPI_DATA_HOME"
printf 'OUTPUT_ROOT: %s\n' "$OUTPUT_ROOT"
printf 'BUNDLE_DIR: %s\n' "$BUNDLE_DIR"
printf 'BUNDLE_SIZE: %s\n' "$BUNDLE_SIZE"
printf 'TRAINING: 2 tasks, 400 requested episodes with incomplete-label episodes dropped, joint task-to-subtask-to-action, stride 1, 117542 steps, batch 64, 8 GPUs\n'

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    echo "DRY_RUN=1: submission skipped."
    exit 0
fi

cd "$BUNDLE_DIR"
ai-hub-cli train mdl \
    "--token=$AI_HUB_TOKEN" \
    "--namespace=$AI_HUB_NAMESPACE" \
    "${mdl_args[@]}"
