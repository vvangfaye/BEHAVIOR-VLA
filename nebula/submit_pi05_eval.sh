#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$REPO_ROOT"

QUEUE="${QUEUE:-aios_specific_nm125_l20}"
GPU_CARD_TYPE="${GPU_CARD_TYPE:-l20}"
CLUSTER_FILE="nebula/eval_cluster.json"
ENTRY_FILE="nebula/eval_pi05.py"
ALGO_NAME="${ALGO_NAME:-pytorch280}"
AIHUB_MODEL_NAME="${AIHUB_MODEL_NAME:-behavior}"
EVAL_PROFILE="${EVAL_PROFILE:-smoke}"
case "$EVAL_PROFILE" in
    smoke|reduced|full) ;;
    *) echo "EVAL_PROFILE must be smoke, reduced, or full" >&2; exit 2 ;;
esac
CONFIRM_FULL="${CONFIRM_FULL:-0}"
if [[ "$EVAL_PROFILE" == full && "$CONFIRM_FULL" != 1 ]]; then
    echo "Full evaluation requires CONFIRM_FULL=1" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_ID:-pi05_joint_subtask_117541_${EVAL_PROFILE}_$STAMP}"
JOB_NAME="${JOB_NAME:-pi05-eval-${EVAL_PROFILE}-$STAMP}"
OUTPUT_BASE="${OUTPUT_BASE:-/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_closed_loop_aihub}"
PERSISTENT_OUTPUT="$OUTPUT_BASE/$RUN_ID"

EVAL_ENV_ARCHIVE="${EVAL_ENV_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-eval.tar.gz}"
PI05_ENV_ARCHIVE="${PI05_ENV_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05.tar.gz}"
OPENPI_CACHE_ARCHIVE="${OPENPI_CACHE_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05-openpi-cache.tar}"
ASSET_ROOT="${ASSET_ROOT:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/eval-assets/v1}"
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_two_tasks_joint_subtask/checkpoints/pi05_b1k/pi05_b1k_2tasks_joint_subtask_s1_117542_20260910_172411/117541}"
PI05_REPO_ID="${PI05_REPO_ID:-b1k_two_tasks_joint_subtask_s1}"
INSTANCE_INDICES="${INSTANCE_INDICES:-0 1 2 3 4}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
NEBULA_WORKER_DRY_RUN="${NEBULA_WORKER_DRY_RUN:-0}"
KEEP_RUNTIME="${KEEP_RUNTIME:-0}"

for path in "$EVAL_ENV_ARCHIVE" "$PI05_ENV_ARCHIVE" "$OPENPI_CACHE_ARCHIVE" \
    "$ASSET_ROOT/SHA256SUMS" "$CHECKPOINT_SOURCE/params/_METADATA" \
    "$CHECKPOINT_SOURCE/assets/$PI05_REPO_ID/norm_stats.json"; do
    [[ -s "$path" ]] || { echo "Required input missing or empty: $path" >&2; exit 1; }
done
(
    cd "$ASSET_ROOT"
    sha256sum -c SHA256SUMS
)
[[ ! -e "$PERSISTENT_OUTPUT" ]] || { echo "Output already exists: $PERSISTENT_OUTPUT" >&2; exit 1; }

CODE_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then CODE_DIRTY=1; else CODE_DIRTY=0; fi

BUNDLE_PARENT="${BUNDLE_PARENT:-${TMPDIR:-/tmp}}"
mkdir -p "$BUNDLE_PARENT"
BUNDLE_DIR="$(mktemp -d "$BUNDLE_PARENT/pi05-eval-nebula-bundle.XXXXXX")"
cleanup_bundle() {
    if [[ "${KEEP_BUNDLE:-0}" == 1 ]]; then
        printf 'Kept submission bundle: %s\n' "$BUNDLE_DIR"
    else
        rm -rf -- "$BUNDLE_DIR"
    fi
}
trap cleanup_bundle EXIT

mkdir -p \
    "$BUNDLE_DIR/openpi/packages/openpi-client" \
    "$BUNDLE_DIR/openpi/scripts/b1k" \
    "$BUNDLE_DIR/BEHAVIOR-1K/OmniGibson" \
    "$BUNDLE_DIR/BEHAVIOR-1K/bddl3" \
    "$BUNDLE_DIR/BEHAVIOR-1K/joylo" \
    "$BUNDLE_DIR/scripts" \
    "$BUNDLE_DIR/nebula"
cp -a "$REPO_ROOT/openpi/src" "$BUNDLE_DIR/openpi/"
cp -a "$REPO_ROOT/openpi/packages/openpi-client/src" "$BUNDLE_DIR/openpi/packages/openpi-client/"
cp -a "$REPO_ROOT/openpi/scripts/b1k/serve_b1k.py" "$BUNDLE_DIR/openpi/scripts/b1k/"
cp -a "$REPO_ROOT/BEHAVIOR-1K/OmniGibson/omnigibson" "$BUNDLE_DIR/BEHAVIOR-1K/OmniGibson/"
cp -a "$REPO_ROOT/BEHAVIOR-1K/bddl3/bddl" "$BUNDLE_DIR/BEHAVIOR-1K/bddl3/"
cp -a "$REPO_ROOT/BEHAVIOR-1K/joylo/gello" "$BUNDLE_DIR/BEHAVIOR-1K/joylo/"
cp -a \
    "$REPO_ROOT/scripts/eval_b1k.sh" \
    "$REPO_ROOT/scripts/b1k_tasks.txt" \
    "$REPO_ROOT/scripts/b1k_tasks_reduced.txt" \
    "$BUNDLE_DIR/scripts/"
cp -a \
    "$REPO_ROOT/nebula/eval_cluster.json" \
    "$REPO_ROOT/nebula/eval_pi05.py" \
    "$REPO_ROOT/nebula/eval_pi05_worker.sh" \
    "$REPO_ROOT/nebula/summarize_pi05_eval.py" \
    "$BUNDLE_DIR/nebula/"
find "$BUNDLE_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$BUNDLE_DIR" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete

write_setting() { printf 'export %s=%q\n' "$1" "$2"; }
{
    write_setting RUN_ID "$RUN_ID"
    write_setting EVAL_PROFILE "$EVAL_PROFILE"
    write_setting CONFIRM_FULL "$CONFIRM_FULL"
    write_setting EVAL_ENV_ARCHIVE "$EVAL_ENV_ARCHIVE"
    write_setting PI05_ENV_ARCHIVE "$PI05_ENV_ARCHIVE"
    write_setting OPENPI_CACHE_ARCHIVE "$OPENPI_CACHE_ARCHIVE"
    write_setting ASSET_ROOT "$ASSET_ROOT"
    write_setting CHECKPOINT_SOURCE "$CHECKPOINT_SOURCE"
    write_setting PI05_REPO_ID "$PI05_REPO_ID"
    write_setting INSTANCE_INDICES "$INSTANCE_INDICES"
    write_setting NUM_ROLLOUTS "$NUM_ROLLOUTS"
    write_setting MAX_STEPS "$MAX_STEPS"
    write_setting PERSISTENT_OUTPUT "$PERSISTENT_OUTPUT"
    write_setting NEBULA_WORKER_DRY_RUN "$NEBULA_WORKER_DRY_RUN"
    write_setting KEEP_RUNTIME "$KEEP_RUNTIME"
    write_setting CODE_COMMIT "$CODE_COMMIT"
    write_setting CODE_DIRTY "$CODE_DIRTY"
} > "$BUNDLE_DIR/nebula/eval_job.env"

BUNDLE_SIZE="$(du -sh "$BUNDLE_DIR" | cut -f1)"
printf 'AIHUB_MODEL_NAME: %s\n' "$AIHUB_MODEL_NAME"
printf 'JOB_NAME: %s\n' "$JOB_NAME"
printf 'RUN_ID: %s\n' "$RUN_ID"
printf 'QUEUE: %s (%s)\n' "$QUEUE" "$GPU_CARD_TYPE"
printf 'PROFILE: %s\n' "$EVAL_PROFILE"
printf 'CHECKPOINT: %s\n' "$CHECKPOINT_SOURCE"
printf 'EVAL_ENV: %s\n' "$EVAL_ENV_ARCHIVE"
printf 'ASSETS: %s\n' "$ASSET_ROOT"
printf 'OUTPUT: %s\n' "$PERSISTENT_OUTPUT"
printf 'BUNDLE: %s (%s)\n' "$BUNDLE_DIR" "$BUNDLE_SIZE"
printf 'GPU MAP: policy=0 simulator=1\n'

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: submission skipped."
    exit 0
fi

cd "$BUNDLE_DIR"
ai-hub-cli train mdl \
    "--token=$AI_HUB_TOKEN" \
    "--namespace=$AI_HUB_NAMESPACE" \
    "--name=$AIHUB_MODEL_NAME" \
    "--queue=$QUEUE" \
    "--entry=$ENTRY_FILE" \
    "--worker_count=1" \
    "--file.cluster_file=$CLUSTER_FILE" \
    "--oss_access_id=$OSS_ACCESS_ID" \
    "--oss_access_key=$OSS_ACCESS_KEY" \
    "--oss_bucket=$OSS_BUCKET" \
    "--oss_endpoint=$OSS_ENDPOINT" \
    "--job_name=$JOB_NAME" \
    "--algo_name=$ALGO_NAME" \
    "--gpu_card_type=$GPU_CARD_TYPE" \
    "--oss_appendable=true" \
    "--debug_mode" \
    "--ignore=__pycache__,*.pyc,*.pyo"
