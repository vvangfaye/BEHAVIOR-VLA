#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

assert_fixed() {
    local name="$1" expected="$2" current="${!1-}"
    if [[ -n "$current" && "$current" != "$expected" ]]; then
        echo "$name is fixed at $expected for GoalKeeper H20 submissions (got $current)" >&2
        exit 2
    fi
    printf -v "$name" '%s' "$expected"
    export "$name"
}
assert_fixed QUEUE demand_specific_na175_h20
assert_fixed WORKER_COUNT 1
assert_fixed NUM_GPUS 8
assert_fixed ALGO_NAME pytorch280
assert_fixed CLUSTER_FILE nebula/multihost_cluster.json
assert_fixed GLOBAL_BATCH_SIZE 64
assert_fixed BATCH_PER_RANK 8
ENTRY_FILE="nebula/train_goalkeeper_h20.py"
AIHUB_MODEL_NAME="${AIHUB_MODEL_NAME:-behavior}"
MODE="${MODE:-preflight}"
case "$MODE" in
    preflight|smoke|full) ;;
    *) echo "MODE must be preflight, smoke, or full" >&2; exit 2 ;;
esac

: "${FEATURE_ROOT:?Set FEATURE_ROOT to the prepared immutable GoalKeeper feature root}"
: "${EXPECTED_TREE_SHA256:?Set EXPECTED_TREE_SHA256 for FEATURE_ROOT}"
: "${EXPECTED_MANIFEST_SHA256:?Set EXPECTED_MANIFEST_SHA256 for manifest.jsonl}"
: "${EXPECTED_CONTRACT_SHA256:?Set EXPECTED_CONTRACT_SHA256 for dataset_contract.json}"
: "${RUN_ID:?Set a fresh RUN_ID explicitly}"
: "${ATTEMPT_ID:?Set a fresh ATTEMPT_ID explicitly}"
: "${JOB_NAME:?Set a fresh JOB_NAME explicitly}"
: "${OUTPUT_ROOT:?Set a fresh OUTPUT_ROOT explicitly}"
for value_name in RUN_ID ATTEMPT_ID JOB_NAME; do
    value="${!value_name}"
    [[ "$value" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]] || {
        echo "$value_name must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}" >&2
        exit 2
    }
done
for hash_name in EXPECTED_TREE_SHA256 EXPECTED_MANIFEST_SHA256 EXPECTED_CONTRACT_SHA256; do
    value="${!hash_name}"
    [[ "$value" =~ ^[0-9a-f]{64}$ ]] || { echo "$hash_name must be a lowercase SHA-256 digest" >&2; exit 2; }
done
[[ "$FEATURE_ROOT" == /* ]] || { echo "FEATURE_ROOT must be an absolute mounted path" >&2; exit 2; }
[[ "$OUTPUT_ROOT" == /* ]] || { echo "OUTPUT_ROOT must be an absolute persistent path" >&2; exit 2; }
[[ -d "$FEATURE_ROOT" ]] || { echo "FEATURE_ROOT is not a directory: $FEATURE_ROOT" >&2; exit 1; }
[[ ! -e "$OUTPUT_ROOT" ]] || { echo "Fresh OUTPUT_ROOT already exists: $OUTPUT_ROOT" >&2; exit 1; }
for value_name in FEATURE_ROOT OUTPUT_ROOT RUN_ID ATTEMPT_ID JOB_NAME; do
    value="${!value_name}"
    [[ "$value" != *','* && "$value" != *$'\n'* && "$value" != *$'\r'* ]] || {
        echo "$value_name contains a character unsupported by AIHub --env" >&2
        exit 2
    }
done
readonly QUEUE WORKER_COUNT NUM_GPUS ALGO_NAME CLUSTER_FILE GLOBAL_BATCH_SIZE BATCH_PER_RANK
readonly MODE FEATURE_ROOT EXPECTED_TREE_SHA256 EXPECTED_MANIFEST_SHA256 EXPECTED_CONTRACT_SHA256
readonly RUN_ID ATTEMPT_ID JOB_NAME OUTPUT_ROOT

python3 "$SCRIPT_DIR/train_goalkeeper_h20.py" validate-feature "$FEATURE_ROOT" \
    --expected-tree "$EXPECTED_TREE_SHA256" \
    --expected-manifest "$EXPECTED_MANIFEST_SHA256" \
    --expected-contract "$EXPECTED_CONTRACT_SHA256" \
    >/dev/null

BUNDLE_PARENT="${BUNDLE_PARENT:-${TMPDIR:-/tmp}}"
mkdir -p "$BUNDLE_PARENT"
BUNDLE_DIR="$(mktemp -d "$BUNDLE_PARENT/goalkeeper-h20-bundle.XXXXXX")"
cleanup_bundle() {
    if [[ "${KEEP_BUNDLE:-0}" == 1 ]]; then
        printf 'Kept submission bundle: %s\n' "$BUNDLE_DIR"
    else
        rm -rf -- "$BUNDLE_DIR"
    fi
}
trap cleanup_bundle EXIT
mkdir -p "$BUNDLE_DIR/goalkeeper" "$BUNDLE_DIR/nebula"
cp -a "$REPO_ROOT/goalkeeper/pyproject.toml" "$BUNDLE_DIR/goalkeeper/"
cp -a "$REPO_ROOT/goalkeeper/src" "$BUNDLE_DIR/goalkeeper/"
cp -a "$REPO_ROOT/goalkeeper/scripts" "$BUNDLE_DIR/goalkeeper/"
cp -a \
    "$REPO_ROOT/nebula/submit_goalkeeper_h20.sh" \
    "$REPO_ROOT/nebula/train_goalkeeper_h20.py" \
    "$REPO_ROOT/nebula/train_goalkeeper_h20.sh" \
    "$REPO_ROOT/nebula/multihost_cluster.json" \
    "$BUNDLE_DIR/nebula/"
find "$BUNDLE_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$BUNDLE_DIR" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
if find "$BUNDLE_DIR" -type l -print -quit | grep -q .; then
    echo "Refusing to submit a bundle containing symlinks" >&2
    exit 1
fi

BUNDLE_SIZE="$(du -sh "$BUNDLE_DIR" | cut -f1)"
printf 'AIHUB_MODEL_NAME: %s\n' "$AIHUB_MODEL_NAME"
printf 'JOB_NAME: %s\n' "$JOB_NAME"
printf 'RUN_ID: %s\n' "$RUN_ID"
printf 'ATTEMPT_ID: %s\n' "$ATTEMPT_ID"
printf 'MODE: %s\n' "$MODE"
printf 'QUEUE: %s\n' "$QUEUE"
printf 'ALGO_NAME: %s\n' "$ALGO_NAME"
printf 'TOPOLOGY: %s worker x %s H20 GPUs\n' "$WORKER_COUNT" "$NUM_GPUS"
printf 'BATCH: global=%s per-rank=%s\n' "$GLOBAL_BATCH_SIZE" "$BATCH_PER_RANK"
printf 'FEATURE_ROOT: %s\n' "$FEATURE_ROOT"
printf 'OUTPUT_ROOT: %s\n' "$OUTPUT_ROOT"
printf 'BUNDLE: %s (%s)\n' "$BUNDLE_DIR" "$BUNDLE_SIZE"

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: submission skipped."
    exit 0
fi

# Credentials remain outside the allowlisted bundle. Loading this local file is
# deferred until after dry-run so validation and bundle tests need no secrets.
if [[ -f "$SCRIPT_DIR/config.sh" ]]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/config.sh"
    set +x
fi
: "${AI_HUB_TOKEN:?AI_HUB_TOKEN is required}"
: "${AI_HUB_NAMESPACE:?AI_HUB_NAMESPACE is required}"
: "${OSS_ACCESS_ID:?OSS_ACCESS_ID is required}"
: "${OSS_ACCESS_KEY:?OSS_ACCESS_KEY is required}"
: "${OSS_BUCKET:?OSS_BUCKET is required}"
: "${OSS_ENDPOINT:?OSS_ENDPOINT is required}"
for secret_name in AI_HUB_TOKEN AI_HUB_NAMESPACE OSS_ACCESS_ID OSS_ACCESS_KEY OSS_BUCKET OSS_ENDPOINT; do
    secret="${!secret_name}"
    [[ "$secret" != *','* && "$secret" != *$'\n'* && "$secret" != *$'\r'* ]] || {
        echo "$secret_name contains a character unsupported by AIHub arguments" >&2
        exit 1
    }
done
AI_HUB_CLI="${AI_HUB_CLI:-ai-hub-cli}"
command -v "$AI_HUB_CLI" >/dev/null 2>&1 || { echo "AIHub CLI not found: $AI_HUB_CLI" >&2; exit 1; }

mdl_args=(
    "--token=$AI_HUB_TOKEN"
    "--namespace=$AI_HUB_NAMESPACE"
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
    "--ignore=__pycache__,*.pyc,*.pyo"
    "--env=MODE=$MODE"
    "--env=FEATURE_ROOT=$FEATURE_ROOT"
    "--env=EXPECTED_TREE_SHA256=$EXPECTED_TREE_SHA256"
    "--env=EXPECTED_MANIFEST_SHA256=$EXPECTED_MANIFEST_SHA256"
    "--env=EXPECTED_CONTRACT_SHA256=$EXPECTED_CONTRACT_SHA256"
    "--env=RUN_ID=$RUN_ID"
    "--env=ATTEMPT_ID=$ATTEMPT_ID"
    "--env=JOB_NAME=$JOB_NAME"
    "--env=OUTPUT_ROOT=$OUTPUT_ROOT"
)
cd "$BUNDLE_DIR"
"$AI_HUB_CLI" train mdl "${mdl_args[@]}"
