#!/usr/bin/env bash
set -euo pipefail

# Nebula activates its launcher environment and injects PYTHONHOME/PYTHONPATH.
# Both override the packed interpreter's own prefix before conda-unpack can run.
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OPENPI_DIR="$REPO_ROOT/openpi"

SOURCE_DATASET="${SOURCE_DATASET:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542}"
LOCAL_DATASET="${LOCAL_DATASET:-/var/tmp/behavior-1k/two_tasks_joint_subtask_s1}"
ENV_ARCHIVE="${ENV_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05.tar.gz}"
ENV_PREFIX="${ENV_PREFIX:-/dev/shm/wangfaye/conda_envs/b1k-pi05}"
OPENPI_CACHE_ARCHIVE="${OPENPI_CACHE_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05-openpi-cache.tar}"
OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-/var/tmp/wangfaye/openpi-cache}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_two_tasks_joint_subtask}"
NORM_STATS_SOURCE="${NORM_STATS_SOURCE:-$SCRIPT_DIR/pi05_two_tasks_norm_stats.json}"
SUBTASK_SEGMENTS_SOURCE="${SUBTASK_SEGMENTS_SOURCE:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/annotations/subtasks/v1/subtask_segments.parquet}"
SUBTASK_SEGMENTS_SHA256="${SUBTASK_SEGMENTS_SHA256:-870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d}"

REPO_ID="${REPO_ID:-b1k_two_tasks_joint_subtask_s1}"
EXP_NAME="${EXP_NAME:-pi05_b1k_2tasks_joint_subtask_s1_117542_$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-8}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
FSDP_DEVICES="${FSDP_DEVICES:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
TRAIN_STEPS="${TRAIN_STEPS:-117542}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
NORM_MAX_FRAMES="${NORM_MAX_FRAMES:-500000}"
FRAME_STRIDE="${FRAME_STRIDE:-1}"
LR_DECAY_STEPS="${LR_DECAY_STEPS:-117542}"
SUBTASK_EMPTY_LABEL_POLICY="${SUBTASK_EMPTY_LABEL_POLICY:-drop_episode}"
MAX_TOKEN_LEN="${MAX_TOKEN_LEN:-512}"
MAX_SUBTASK_LEN="${MAX_SUBTASK_LEN:-64}"
SUBTASK_LOSS_WEIGHT="${SUBTASK_LOSS_WEIGHT:-1.0}"
SUBTASK_LOSS_CHUNK_SIZE="${SUBTASK_LOSS_CHUNK_SIZE:-8}"
WHOLE_TASK_PROB="${WHOLE_TASK_PROB:-0}"

CHUNKS=(chunk-014 chunk-056)
CAMERAS=(
    observation.rgb.zed_link_camera_0
    observation.rgb.left_realsense_link_camera_0
    observation.rgb.right_realsense_link_camera_0
)

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
require_file() { [[ -f "$1" ]] || { echo "Required file not found: $1" >&2; exit 1; }; }
require_dir() { [[ -d "$1" ]] || { echo "Required directory not found: $1" >&2; exit 1; }; }

require_file "$ENV_ARCHIVE"
require_file "$OPENPI_CACHE_ARCHIVE"
require_file "$SOURCE_DATASET/meta/info.json"
require_file "$NORM_STATS_SOURCE"
require_file "$SUBTASK_SEGMENTS_SOURCE"
if [[ "$(sha256sum "$SUBTASK_SEGMENTS_SOURCE" | cut -d' ' -f1)" != "$SUBTASK_SEGMENTS_SHA256" ]]; then
    echo "Subtask annotation checksum mismatch: $SUBTASK_SEGMENTS_SOURCE" >&2
    exit 1
fi
for chunk in "${CHUNKS[@]}"; do
    require_dir "$SOURCE_DATASET/data/$chunk"
    for camera in "${CAMERAS[@]}"; do
        require_dir "$SOURCE_DATASET/videos/$camera/$chunk"
    done
done

log "π0.5 joint-subtask Nebula job: $EXP_NAME"
log "source dataset: $SOURCE_DATASET"
log "local dataset:  $LOCAL_DATASET"
log "subtasks:       $SUBTASK_SEGMENTS_SOURCE (sha256=$SUBTASK_SEGMENTS_SHA256)"
log "model cache:    $OPENPI_CACHE_ARCHIVE -> $OPENPI_DATA_HOME"
log "output root:    $OUTPUT_ROOT"
log "training: 400 requested episodes with incomplete-label episodes dropped, stride=$FRAME_STRIDE, steps=$TRAIN_STEPS, batch=$BATCH_SIZE, GPUs=$NUM_GPUS"

if [[ "${NEBULA_WORKER_DRY_RUN:-0}" == "1" ]]; then
    log "worker dry-run complete; no files were changed"
    exit 0
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    log "restoring packed environment to $ENV_PREFIX"
    mkdir -p "$ENV_PREFIX"
    tar -xzf "$ENV_ARCHIVE" -C "$ENV_PREFIX"
fi
if [[ ! -f "$ENV_PREFIX/.b1k-conda-unpacked" ]]; then
    require_file "$ENV_PREFIX/bin/conda-unpack"
    "$ENV_PREFIX/bin/conda-unpack"
    touch "$ENV_PREFIX/.b1k-conda-unpacked"
fi
"$ENV_PREFIX/bin/python" -c \
    'import encodings, sys; print(f"packed Python ready: {sys.executable} (prefix={sys.prefix})")'

log "binding OpenPI packages to this job's source tree"
SITE_PACKAGES="$("$ENV_PREFIX/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n%s\n' \
    "$OPENPI_DIR/src" \
    "$OPENPI_DIR/packages/openpi-client/src" \
    > "$SITE_PACKAGES/00-b1k-job-source.pth"
"$ENV_PREFIX/bin/python" -c \
    'import pathlib, sys, openpi, openpi_client; root = pathlib.Path(sys.argv[1]).resolve(); paths = [pathlib.Path(openpi.__file__).resolve(), pathlib.Path(openpi_client.__file__).resolve()]; assert all(root in path.parents for path in paths), paths; print(f"openpi source: {paths[0]}"); print(f"openpi-client source: {paths[1]}")' \
    "$OPENPI_DIR"
if [[ -e "$OPENPI_DIR/.venv" && ! -L "$OPENPI_DIR/.venv" ]]; then
    echo "Refusing to replace non-symlink OpenPI environment: $OPENPI_DIR/.venv" >&2
    exit 1
fi
ln -sfn "$ENV_PREFIX" "$OPENPI_DIR/.venv"

CACHE_MARKER="$OPENPI_DATA_HOME/.b1k-pi05-cache-complete"
if [[ ! -f "$CACHE_MARKER" ]]; then
    log "staging tokenizer and π0.5 base weights to local cache"
    mkdir -p "$OPENPI_DATA_HOME"
    tar -xf "$OPENPI_CACHE_ARCHIVE" -C "$OPENPI_DATA_HOME"
    require_file "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
    require_file "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params/_METADATA"
    touch "$CACHE_MARKER"
else
    log "reusing local OpenPI cache at $OPENPI_DATA_HOME"
fi
require_file "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
require_file "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params/_METADATA"

STAGE_MARKER="$LOCAL_DATASET/.pi05_two_tasks_rgb_complete"
if [[ ! -f "$STAGE_MARKER" ]]; then
    log "staging metadata and selected RGB chunks"
    mkdir -p "$LOCAL_DATASET/meta"
    cp -r --no-preserve=ownership "$SOURCE_DATASET/meta/." "$LOCAL_DATASET/meta/"
    for chunk in "${CHUNKS[@]}"; do
        mkdir -p "$LOCAL_DATASET/data/$chunk"
        cp -r --no-preserve=ownership "$SOURCE_DATASET/data/$chunk/." "$LOCAL_DATASET/data/$chunk/"
        for camera in "${CAMERAS[@]}"; do
            destination="$LOCAL_DATASET/videos/$camera/$chunk"
            mkdir -p "$destination"
            cp -r --no-preserve=ownership "$SOURCE_DATASET/videos/$camera/$chunk/." "$destination/"
        done
    done
    "$ENV_PREFIX/bin/python" "$REPO_ROOT/scripts/strip_depth_from_info.py" "$LOCAL_DATASET"
    touch "$STAGE_MARKER"
else
    log "reusing staged dataset at $LOCAL_DATASET"
    "$ENV_PREFIX/bin/python" "$REPO_ROOT/scripts/strip_depth_from_info.py" "$LOCAL_DATASET"
fi
log "copying versioned subtask annotations from OSS"
cp "$SUBTASK_SEGMENTS_SOURCE" "$LOCAL_DATASET/meta/subtask_segments.parquet"
if [[ "$(sha256sum "$LOCAL_DATASET/meta/subtask_segments.parquet" | cut -d' ' -f1)" != "$SUBTASK_SEGMENTS_SHA256" ]]; then
    echo "Worker-local subtask annotation checksum mismatch" >&2
    exit 1
fi

EPISODES_FILE="$LOCAL_DATASET/meta/pi05_two_tasks_episodes.txt"
{
    seq 2800 2999
    seq 11200 11399
} > "$EPISODES_FILE"
if [[ "$(wc -l < "$EPISODES_FILE")" -ne 400 ]]; then
    echo "Expected 400 selected episodes in $EPISODES_FILE" >&2
    exit 1
fi

mkdir -p "$OUTPUT_ROOT/assets" "$OUTPUT_ROOT/checkpoints" "$OPENPI_DATA_HOME" "$OPENPI_DIR/outputs"
for output_kind in assets checkpoints; do
    output_link="$OPENPI_DIR/outputs/$output_kind"
    output_target="$OUTPUT_ROOT/$output_kind"
    if [[ -e "$output_link" || -L "$output_link" ]]; then
        if [[ ! -L "$output_link" || "$(readlink -f "$output_link")" != "$(readlink -f "$output_target")" ]]; then
            echo "Refusing to replace existing OpenPI output path: $output_link" >&2
            exit 1
        fi
    else
        ln -s "$output_target" "$output_link"
    fi
done

NORM_STATS_TARGET="$OUTPUT_ROOT/assets/pi05_b1k/$REPO_ID/norm_stats.json"
if [[ ! -f "$NORM_STATS_TARGET" ]]; then
    log "installing precomputed two-task normalization statistics"
    mkdir -p "$(dirname "$NORM_STATS_TARGET")"
    cp "$NORM_STATS_SOURCE" "$NORM_STATS_TARGET"
fi

SUBTASK_PROMPTS="${SUBTASK_PROMPTS:-$LOCAL_DATASET/meta/subtask_segments.parquet}"
require_file "$SUBTASK_PROMPTS"
export OPENPI_DATA_HOME DATASET_ROOT="$LOCAL_DATASET" REPO_ID EPISODES_FILE EXP_NAME
export NUM_GPUS CUDA_DEVICES FSDP_DEVICES NUM_WORKERS BATCH_SIZE TRAIN_STEPS SAVE_INTERVAL
export NORM_MAX_FRAMES FRAME_STRIDE LR_DECAY_STEPS SUBTASK_PROMPTS SUBTASK_EMPTY_LABEL_POLICY
export MAX_TOKEN_LEN MAX_SUBTASK_LEN SUBTASK_LOSS_WEIGHT SUBTASK_LOSS_CHUNK_SIZE WHOLE_TASK_PROB
if [[ "${RESUME:-0}" == "1" ]]; then
    export RESUME=1
else
    unset RESUME
fi

LOG_DIR="$OUTPUT_ROOT/logs"
LOG_FILE="$LOG_DIR/$EXP_NAME.log"
mkdir -p "$LOG_DIR"
log "starting joint-subtask training; log: $LOG_FILE"
bash "$REPO_ROOT/scripts/train_pi05_subtask_b1k.sh" full 2>&1 | tee -a "$LOG_FILE"
log "joint-subtask training complete; checkpoints: $OUTPUT_ROOT/checkpoints/pi05_b1k/$EXP_NAME"
