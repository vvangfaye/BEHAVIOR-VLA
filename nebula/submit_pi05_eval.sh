#!/usr/bin/env bash
set -euo pipefail
set +x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$REPO_ROOT"

QUEUE="${QUEUE:-aios_specific_nm125_l20}"
GPU_CARD_TYPE="${GPU_CARD_TYPE:-l20}"
ENTRY_FILE="nebula/eval_pi05.py"
PROVIDED_SUBTASK_AS_PROMPT="${PROVIDED_SUBTASK_AS_PROMPT:-0}"
ALGO_NAME="${ALGO_NAME:-pytorch2120}"
AIHUB_MODEL_NAME="${AIHUB_MODEL_NAME:-behavior}"
AI_HUB_CLI="${AI_HUB_CLI:-ai-hub-cli}"
PI05_MODEL_FORMAT="${PI05_MODEL_FORMAT:-jax}"
NEBULA_PIP_NO_DEPS="${NEBULA_PIP_NO_DEPS:-0}"
case "$PI05_MODEL_FORMAT" in
    jax|torch) ;;
    *) echo "PI05_MODEL_FORMAT must be jax or torch" >&2; exit 2 ;;
esac
EVAL_PROFILE="${EVAL_PROFILE:-smoke}"
case "$EVAL_PROFILE" in
    smoke|speed|two_tasks|reduced|full|full4) ;;
    *) echo "EVAL_PROFILE must be smoke, speed, two_tasks, reduced, full, or full4" >&2; exit 2 ;;
esac
if [[ "$EVAL_PROFILE" == speed || "$EVAL_PROFILE" == two_tasks || "$EVAL_PROFILE" == full4 ]]; then
    CLUSTER_FILE="${CLUSTER_FILE:-nebula/multihost_cluster_4gpu.json}"
    EVAL_WORKERS="${EVAL_WORKERS:-4}"
    BASE_PORT="${BASE_PORT:-8000}"
    PI05_MEM_FRACTION="${PI05_MEM_FRACTION:-0.35}"
    if [[ "$EVAL_PROFILE" == speed ]]; then
        WRITE_VIDEO="${WRITE_VIDEO:-0}"
        SPEED_TASK="${SPEED_TASK:-make_rose_centerpieces}"
        INSTANCE_INDICES="${INSTANCE_INDICES:-0 1 2 3}"
        SPEED_EXPECTED_INSTANCE_IDS="${SPEED_EXPECTED_INSTANCE_IDS:-301 302 303 304}"
        MAX_STEPS="${MAX_STEPS:-100}"
    elif [[ "$EVAL_PROFILE" == full4 ]]; then
        WRITE_VIDEO="${WRITE_VIDEO:-1}"
        SPEED_TASK=""
        SPEED_EXPECTED_INSTANCE_IDS=""
        INSTANCE_INDICES="${INSTANCE_INDICES:-0 1}"
        NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
        MAX_STEPS="${MAX_STEPS:-}"
    else
        WRITE_VIDEO="${WRITE_VIDEO:-1}"
        SPEED_TASK=""
        SPEED_EXPECTED_INSTANCE_IDS=""
        INSTANCE_INDICES="${INSTANCE_INDICES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
        NUM_ROLLOUTS="${NUM_ROLLOUTS:-3}"
        MAX_STEPS="${MAX_STEPS:-}"
    fi
else
    CLUSTER_FILE="${CLUSTER_FILE:-nebula/eval_cluster.json}"
    EVAL_WORKERS="${EVAL_WORKERS:-1}"
    BASE_PORT="${BASE_PORT:-8000}"
    PI05_MEM_FRACTION="${PI05_MEM_FRACTION:-0.85}"
    WRITE_VIDEO="${WRITE_VIDEO:-1}"
    SPEED_TASK="${SPEED_TASK:-}"
    SPEED_EXPECTED_INSTANCE_IDS="${SPEED_EXPECTED_INSTANCE_IDS:-}"
fi
if [[ "$EVAL_PROFILE" == smoke && ( "$PI05_MODEL_FORMAT" == torch || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ) ]]; then
    SMOKE_TASK="${SMOKE_TASK:-carrying_in_groceries}"
    SMOKE_EVAL_GPU="${SMOKE_EVAL_GPU:-0}"
else
    SMOKE_TASK="${SMOKE_TASK:-turning_on_radio}"
    SMOKE_EVAL_GPU="${SMOKE_EVAL_GPU:-1}"
fi
CONFIRM_FULL="${CONFIRM_FULL:-0}"
SUBTASK_PREDICTION="${SUBTASK_PREDICTION:-1}"
EXTERNAL_SUBTASK_SELECTOR="${EXTERNAL_SUBTASK_SELECTOR:-}"
MAX_FAILOVER_TIMES="${MAX_FAILOVER_TIMES:-0}"
if [[ ( "$EVAL_PROFILE" == full || "$EVAL_PROFILE" == full4 ) && "$CONFIRM_FULL" != 1 ]]; then
    echo "Full evaluation requires CONFIRM_FULL=1" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    RUN_ID="${RUN_ID:-pi05_strict_executor_114356_selector_${EVAL_PROFILE}_$STAMP}"
    JOB_NAME="${JOB_NAME:-pi05-strict-selector-${EVAL_PROFILE}-$STAMP}"
elif [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    RUN_ID="${RUN_ID:-pi05_selector_diagnostic_117541_${EVAL_PROFILE}_$STAMP}"
    JOB_NAME="${JOB_NAME:-pi05-selector-diagnostic-${EVAL_PROFILE}-$STAMP}"
elif [[ "$PI05_MODEL_FORMAT" == torch ]]; then
    RUN_ID="${RUN_ID:-pi05_torch_joint_subtask_${EVAL_PROFILE}_$STAMP}"
    JOB_NAME="${JOB_NAME:-pi05-torch-eval-${EVAL_PROFILE}-$STAMP}"
else
    RUN_ID="${RUN_ID:-pi05_joint_subtask_117541_${EVAL_PROFILE}_$STAMP}"
    JOB_NAME="${JOB_NAME:-pi05-eval-${EVAL_PROFILE}-$STAMP}"
fi
OUTPUT_BASE="${OUTPUT_BASE:-/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_closed_loop_aihub}"
PERSISTENT_OUTPUT="$OUTPUT_BASE/$RUN_ID"

EVAL_ENV_ARCHIVE="${EVAL_ENV_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-eval.tar.gz}"
if [[ "$PI05_MODEL_FORMAT" == torch ]]; then
    DEFAULT_PI05_ENV_ARCHIVE="/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05-torch.tar.gz"
else
    DEFAULT_PI05_ENV_ARCHIVE="/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05.tar.gz"
fi
PI05_ENV_ARCHIVE="${PI05_ENV_ARCHIVE:-$DEFAULT_PI05_ENV_ARCHIVE}"
OPENPI_CACHE_ARCHIVE="${OPENPI_CACHE_ARCHIVE:-/data/oss_bucket_0/wangfaye/conda_env/b1k-pi05-openpi-cache.tar}"
ASSET_ROOT="${ASSET_ROOT:-/data/oss_bucket_0/wangfaye/embodied/datasets/behavior-1k/eval-assets/v1}"
if [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    DEFAULT_CHECKPOINT_SOURCE="/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_multihost/staged_inference_pi05_b1k_two_tasks_subtask_only_strict_s1_114356_20260919_074849"
elif [[ "$PI05_MODEL_FORMAT" == torch ]]; then
    DEFAULT_CHECKPOINT_SOURCE=""
elif [[ "$EVAL_PROFILE" == two_tasks ]]; then
    DEFAULT_CHECKPOINT_SOURCE="/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_two_tasks_joint_subtask/staged_local_pi05_b1k_2tasks_joint_subtask_117541_20260915_151620"
else
    DEFAULT_CHECKPOINT_SOURCE="/data/oss_bucket_0/wangfaye/result/behavior-challenge/pi05_two_tasks_joint_subtask/checkpoints/pi05_b1k/pi05_b1k_2tasks_joint_subtask_s1_117542_20260910_172411/117541"
fi
CHECKPOINT_SOURCE="${CHECKPOINT_SOURCE:-$DEFAULT_CHECKPOINT_SOURCE}"
[[ -n "$CHECKPOINT_SOURCE" ]] || {
    echo "CHECKPOINT_SOURCE is required for PI05_MODEL_FORMAT=torch" >&2
    exit 2
}
if [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    DEFAULT_PI05_REPO_ID="b1k_two_tasks_subtask_only_strict_s1"
else
    DEFAULT_PI05_REPO_ID="b1k_two_tasks_joint_subtask_s1"
fi
PI05_REPO_ID="${PI05_REPO_ID:-$DEFAULT_PI05_REPO_ID}"
INSTANCE_INDICES="${INSTANCE_INDICES:-0 1 2 3 4}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
NEBULA_WORKER_DRY_RUN="${NEBULA_WORKER_DRY_RUN:-0}"
KEEP_RUNTIME="${KEEP_RUNTIME:-0}"
[[ "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_WORKERS must be a positive integer" >&2; exit 2; }
[[ "$NUM_ROLLOUTS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ROLLOUTS must be a positive integer" >&2; exit 2; }
[[ "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || { echo "BASE_PORT must be a positive integer" >&2; exit 2; }
[[ "$MAX_FAILOVER_TIMES" =~ ^[0-9]+$ ]] || { echo "MAX_FAILOVER_TIMES must be a non-negative integer" >&2; exit 2; }
[[ "$NEBULA_PIP_NO_DEPS" == 0 || "$NEBULA_PIP_NO_DEPS" == 1 ]] || {
    echo "NEBULA_PIP_NO_DEPS must be 0 or 1" >&2; exit 2; }
[[ "$WRITE_VIDEO" == 0 || "$WRITE_VIDEO" == 1 ]] || { echo "WRITE_VIDEO must be 0 or 1" >&2; exit 2; }
[[ "$SUBTASK_PREDICTION" == 0 || "$SUBTASK_PREDICTION" == 1 ]] || { echo "SUBTASK_PREDICTION must be 0 or 1" >&2; exit 2; }
[[ "$PROVIDED_SUBTASK_AS_PROMPT" == 0 || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]] || {
    echo "PROVIDED_SUBTASK_AS_PROMPT must be 0 or 1" >&2; exit 2; }
[[ "$SMOKE_EVAL_GPU" =~ ^[0-9]+$ ]] || { echo "SMOKE_EVAL_GPU must be a non-negative integer" >&2; exit 2; }
[[ "$PI05_MEM_FRACTION" =~ ^0\.[0-9]+$ ]] || { echo "PI05_MEM_FRACTION must be between 0 and 1" >&2; exit 2; }
if [[ "$SUBTASK_PREDICTION" == 1 && "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    echo "SUBTASK_PREDICTION and PROVIDED_SUBTASK_AS_PROMPT are mutually exclusive" >&2
    exit 2
fi
if [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    [[ "$PI05_MODEL_FORMAT" == jax ]] || {
        echo "PROVIDED_SUBTASK_AS_PROMPT currently supports JAX checkpoints only" >&2
        exit 2
    }
    [[ "$SUBTASK_PREDICTION" == 0 ]] || {
        echo "PROVIDED_SUBTASK_AS_PROMPT requires SUBTASK_PREDICTION=0" >&2
        exit 2
    }
    [[ "$EXTERNAL_SUBTASK_SELECTOR" == state_machine_v1 ]] || {
        echo "PROVIDED_SUBTASK_AS_PROMPT requires EXTERNAL_SUBTASK_SELECTOR=state_machine_v1" >&2
        exit 2
    }
fi
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    [[ "$EXTERNAL_SUBTASK_SELECTOR" == state_machine_v1 ]] || {
        echo "EXTERNAL_SUBTASK_SELECTOR must be state_machine_v1" >&2
        exit 2
    }
    [[ "$EVAL_PROFILE" == smoke || "$EVAL_PROFILE" == two_tasks ]] || {
        echo "EXTERNAL_SUBTASK_SELECTOR is supported only for EVAL_PROFILE=smoke or two_tasks" >&2
        exit 2
    }
    [[ "$SUBTASK_PREDICTION" == 1 || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]] || {
        echo "EXTERNAL_SUBTASK_SELECTOR requires a subtask conditioning mode" >&2
        exit 2
    }
    if [[ "$EVAL_PROFILE" == smoke ]]; then
        case "$SMOKE_TASK" in
            carrying_in_groceries|make_rose_centerpieces) ;;
            *) echo "selector smoke supports only carrying_in_groceries or make_rose_centerpieces" >&2; exit 2 ;;
        esac
    fi
    [[ -z "${EXTERNAL_SUBTASK:-}" ]] || {
        echo "EXTERNAL_SUBTASK and EXTERNAL_SUBTASK_SELECTOR are mutually exclusive" >&2
        exit 2
    }
fi
if [[ "$EVAL_PROFILE" == speed ]]; then
    (( EVAL_WORKERS == 4 )) || { echo "speed profile requires EVAL_WORKERS=4" >&2; exit 2; }
    (( NUM_ROLLOUTS == 1 )) || { echo "speed profile requires NUM_ROLLOUTS=1" >&2; exit 2; }
    read -ra speed_indices <<< "$INSTANCE_INDICES"
    read -ra speed_instance_ids <<< "$SPEED_EXPECTED_INSTANCE_IDS"
    (( ${#speed_indices[@]} == EVAL_WORKERS && ${#speed_instance_ids[@]} == EVAL_WORKERS )) || {
        echo "speed profile requires four indices and four expected instance IDs" >&2
        exit 2
    }
    [[ "$(printf '%s\n' "${speed_indices[@]}" | sort -u | grep -c .)" == "$EVAL_WORKERS" ]] || {
        echo "speed profile instance indices must be unique" >&2
        exit 2
    }
    [[ "$(printf '%s\n' "${speed_instance_ids[@]}" | sort -u | grep -c .)" == "$EVAL_WORKERS" ]] || {
        echo "speed profile expected instance IDs must be unique" >&2
        exit 2
    }
    [[ "${speed_indices[*]}" == "0 1 2 3" ]] || {
        echo "speed profile requires INSTANCE_INDICES='0 1 2 3'" >&2
        exit 2
    }
    [[ "${speed_instance_ids[*]}" == "301 302 303 304" ]] || {
        echo "speed profile requires SPEED_EXPECTED_INSTANCE_IDS='301 302 303 304'" >&2
        exit 2
    }
elif [[ "$EVAL_PROFILE" == two_tasks ]]; then
    (( EVAL_WORKERS == 4 )) || { echo "two_tasks profile requires EVAL_WORKERS=4" >&2; exit 2; }
    (( NUM_ROLLOUTS == 3 )) || { echo "two_tasks profile requires NUM_ROLLOUTS=3" >&2; exit 2; }
    [[ "$WRITE_VIDEO" == 1 ]] || { echo "two_tasks profile requires WRITE_VIDEO=1" >&2; exit 2; }
    [[ -z "$MAX_STEPS" ]] || { echo "two_tasks profile requires a full task timeout" >&2; exit 2; }
    read -ra two_task_indices <<< "$INSTANCE_INDICES"
    (( ${#two_task_indices[@]} == 20 )) || {
        echo "two_tasks profile requires all 20 public instance indices" >&2
        exit 2
    }
    [[ "${two_task_indices[*]}" == "0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19" ]] || {
        echo "two_tasks profile requires INSTANCE_INDICES='0 1 ... 19'" >&2
        exit 2
    }
elif [[ "$EVAL_PROFILE" == full4 ]]; then
    (( EVAL_WORKERS == 4 )) || { echo "full4 profile requires EVAL_WORKERS=4" >&2; exit 2; }
    (( NUM_ROLLOUTS == 1 )) || { echo "full4 profile requires NUM_ROLLOUTS=1" >&2; exit 2; }
    [[ "$WRITE_VIDEO" == 1 ]] || { echo "full4 profile requires WRITE_VIDEO=1" >&2; exit 2; }
    [[ -z "$MAX_STEPS" ]] || { echo "full4 profile requires a full task timeout" >&2; exit 2; }
    [[ -z "$EXTERNAL_SUBTASK_SELECTOR" ]] || { echo "full4 profile does not support EXTERNAL_SUBTASK_SELECTOR" >&2; exit 2; }
    read -ra full4_indices <<< "$INSTANCE_INDICES"
    (( ${#full4_indices[@]} >= 1 )) || { echo "full4 profile requires at least one instance index" >&2; exit 2; }
    for idx in "${full4_indices[@]}"; do
        [[ "$idx" =~ ^[0-9]+$ ]] && (( idx >= 0 && idx <= 19 )) || {
            echo "full4 instance indices must be integers in 0..19 (public test)" >&2; exit 2; }
    done
    [[ "$(printf '%s\n' "${full4_indices[@]}" | sort -un | grep -c .)" == "${#full4_indices[@]}" ]] || {
        echo "full4 instance indices must be unique" >&2; exit 2; }
    full4_task_count="$(grep -cvE '^[[:space:]]*(#|$)' "$REPO_ROOT/scripts/b1k_tasks.txt")"
    (( full4_task_count == 100 )) || {
        echo "full4 profile requires 100 tasks in scripts/b1k_tasks.txt, found $full4_task_count" >&2
        exit 2
    }
fi

for path in "$EVAL_ENV_ARCHIVE" "$PI05_ENV_ARCHIVE" "$OPENPI_CACHE_ARCHIVE" \
    "$ASSET_ROOT/SHA256SUMS" "$CHECKPOINT_SOURCE/assets/$PI05_REPO_ID/norm_stats.json"; do
    [[ -s "$path" ]] || { echo "Required input missing or empty: $path" >&2; exit 1; }
done
if [[ "$PI05_MODEL_FORMAT" == torch ]]; then
    for path in "$CHECKPOINT_SOURCE/model.safetensors" "$CHECKPOINT_SOURCE/metadata.json"; do
        [[ -s "$path" ]] || { echo "Torch checkpoint input missing or empty: $path" >&2; exit 1; }
    done
    [[ ! -e "$CHECKPOINT_SOURCE/params/_METADATA" ]] || {
        echo "Torch checkpoint must not contain JAX params/_METADATA" >&2
        exit 1
    }
else
    [[ -s "$CHECKPOINT_SOURCE/params/_METADATA" ]] || {
        echo "JAX checkpoint input missing or empty: $CHECKPOINT_SOURCE/params/_METADATA" >&2
        exit 1
    }
    [[ ! -e "$CHECKPOINT_SOURCE/model.safetensors" ]] || {
        echo "JAX checkpoint must not contain model.safetensors" >&2
        exit 1
    }
fi
if [[ "$EVAL_PROFILE" == two_tasks || "$PI05_MODEL_FORMAT" == torch || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    for path in "$CHECKPOINT_SOURCE/_CHECKPOINT_METADATA" \
        "$CHECKPOINT_SOURCE/SHA256SUMS.inference" "$CHECKPOINT_SOURCE/_READY"; do
        [[ -s "$path" ]] || { echo "Staged checkpoint input missing or empty: $path" >&2; exit 1; }
    done
    (
        cd "$CHECKPOINT_SOURCE"
        sha256sum -c SHA256SUMS.inference
    )
fi
if [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    python3 - "$CHECKPOINT_SOURCE/_READY" "$PI05_REPO_ID" <<'PY'
import json
import sys
from pathlib import Path

ready = json.loads(Path(sys.argv[1]).read_text())
expected_repo_id = sys.argv[2]
expected = {
    "conditioning_mode": "subtask_only",
    "strict_subtask_action_chunks": True,
    "subtask_executor": False,
    "action_horizon": 32,
    "repo_id": expected_repo_id,
}
for key, value in expected.items():
    if ready.get(key) != value:
        raise SystemExit(f"Strict checkpoint metadata mismatch for {key}: expected {value!r}, got {ready.get(key)!r}")
PY
fi
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
    "$BUNDLE_DIR/BEHAVIOR-1K/docs/assets" \
    "$BUNDLE_DIR/BEHAVIOR-1K/joylo" \
    "$BUNDLE_DIR/scripts" \
    "$BUNDLE_DIR/nebula"
cp -a "$REPO_ROOT/openpi/src" "$BUNDLE_DIR/openpi/"
cp -a "$REPO_ROOT/openpi/packages/openpi-client/src" "$BUNDLE_DIR/openpi/packages/openpi-client/"
cp -a "$REPO_ROOT/openpi/scripts/b1k/serve_b1k.py" "$BUNDLE_DIR/openpi/scripts/b1k/"
cp -a "$REPO_ROOT/openpi/scripts/install_transformers_patch.py" "$BUNDLE_DIR/openpi/scripts/"
cp -a "$REPO_ROOT/BEHAVIOR-1K/OmniGibson/omnigibson" "$BUNDLE_DIR/BEHAVIOR-1K/OmniGibson/"
cp -a "$REPO_ROOT/BEHAVIOR-1K/docs/assets/OmniGibson_logo.png" "$BUNDLE_DIR/BEHAVIOR-1K/docs/assets/"
cp -a "$REPO_ROOT/BEHAVIOR-1K/bddl3/bddl" "$BUNDLE_DIR/BEHAVIOR-1K/bddl3/"
cp -a "$REPO_ROOT/BEHAVIOR-1K/joylo/gello" "$BUNDLE_DIR/BEHAVIOR-1K/joylo/"
cp -a \
    "$REPO_ROOT/scripts/eval_b1k.sh" \
    "$REPO_ROOT/scripts/b1k_tasks.txt" \
    "$REPO_ROOT/scripts/b1k_tasks_reduced.txt" \
    "$BUNDLE_DIR/scripts/"
cp -a \
    "$REPO_ROOT/nebula/eval_pi05.py" \
    "$REPO_ROOT/nebula/eval_pi05_worker.sh" \
    "$REPO_ROOT/nebula/summarize_pi05_eval.py" \
    "$BUNDLE_DIR/nebula/"
cp -a "$REPO_ROOT/$CLUSTER_FILE" "$BUNDLE_DIR/$CLUSTER_FILE"
find "$BUNDLE_DIR" -type d -name __pycache__ -prune -exec rm -rf {} +
find "$BUNDLE_DIR" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
BOOTSTRAP_ENV_ARGS=()
if [[ "$NEBULA_PIP_NO_DEPS" == 1 ]]; then
    BOOTSTRAP_ENV_ARGS=("--env=PIP_NO_DEPS=1")
fi

write_setting() { printf 'export %s=%q\n' "$1" "$2"; }
{
    write_setting RUN_ID "$RUN_ID"
    write_setting EVAL_PROFILE "$EVAL_PROFILE"
    write_setting PI05_MODEL_FORMAT "$PI05_MODEL_FORMAT"
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
    write_setting EVAL_WORKERS "$EVAL_WORKERS"
    write_setting BASE_PORT "$BASE_PORT"
    write_setting PI05_MEM_FRACTION "$PI05_MEM_FRACTION"
    write_setting WRITE_VIDEO "$WRITE_VIDEO"
    write_setting SUBTASK_PREDICTION "$SUBTASK_PREDICTION"
    write_setting PROVIDED_SUBTASK_AS_PROMPT "$PROVIDED_SUBTASK_AS_PROMPT"
    write_setting EXTERNAL_SUBTASK_SELECTOR "$EXTERNAL_SUBTASK_SELECTOR"
    write_setting SMOKE_TASK "$SMOKE_TASK"
    write_setting SMOKE_EVAL_GPU "$SMOKE_EVAL_GPU"
    write_setting SPEED_TASK "$SPEED_TASK"
    write_setting SPEED_EXPECTED_INSTANCE_IDS "$SPEED_EXPECTED_INSTANCE_IDS"
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
printf 'ALGO_NAME: %s\n' "$ALGO_NAME"
printf 'NEBULA_PIP_NO_DEPS: %s\n' "$NEBULA_PIP_NO_DEPS"
printf 'PROFILE: %s\n' "$EVAL_PROFILE"
printf 'MODEL_FORMAT: %s\n' "$PI05_MODEL_FORMAT"
printf 'CLUSTER: %s\n' "$CLUSTER_FILE"
printf 'SUBTASK_PREDICTION: %s\n' "$SUBTASK_PREDICTION"
printf 'PROVIDED_SUBTASK_AS_PROMPT: %s\n' "$PROVIDED_SUBTASK_AS_PROMPT"
printf 'EXTERNAL_SUBTASK_SELECTOR: %s\n' "${EXTERNAL_SUBTASK_SELECTOR:-<disabled>}"
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    printf 'EVALUATION_KIND: selector diagnostic (label_format=legacy oracle=true privileged=true official_benchmark=false)\n'
    if [[ "$EVAL_PROFILE" == two_tasks ]]; then
        printf 'EXPECTED_ROLLOUTS: 120 (4 co-located workers)\n'
    else
        printf 'EXPECTED_ROLLOUTS: %s smoke rollout(s)\n' "$NUM_ROLLOUTS"
    fi
fi
printf 'MAX_FAILOVER_TIMES: %s\n' "$MAX_FAILOVER_TIMES"
printf 'CHECKPOINT: %s\n' "$CHECKPOINT_SOURCE"
printf 'POLICY_ENV: %s\n' "$PI05_ENV_ARCHIVE"
printf 'EVAL_ENV: %s\n' "$EVAL_ENV_ARCHIVE"
printf 'ASSETS: %s\n' "$ASSET_ROOT"
printf 'OUTPUT: %s\n' "$PERSISTENT_OUTPUT"
printf 'BUNDLE: %s (%s)\n' "$BUNDLE_DIR" "$BUNDLE_SIZE"
if [[ "$EVAL_PROFILE" == speed || "$EVAL_PROFILE" == two_tasks ]]; then
    TWO_TASK_SHARDS=("0 4 8 12 16" "1 5 9 13 17" "2 6 10 14 18" "3 7 11 15 19")
    for (( worker = 0; worker < EVAL_WORKERS; worker++ )); do
        if [[ "$EVAL_PROFILE" == two_tasks ]]; then
            printf 'WORKER %d: tasks=carrying_in_groceries,make_rose_centerpieces indices=[%s] rollouts=30 gpu=%d port=%d\n' \
                "$worker" "${TWO_TASK_SHARDS[worker]}" "$worker" "$((BASE_PORT + worker))"
        else
            printf 'WORKER %d: policy=%d simulator=%d port=%d\n' \
                "$worker" "$worker" "$worker" "$((BASE_PORT + worker))"
        fi
    done
elif [[ "$EVAL_PROFILE" == full4 ]]; then
    mapfile -t FULL4_TASKS < <(grep -vE '^[[:space:]]*(#|$)' "$REPO_ROOT/scripts/b1k_tasks.txt" | awk '{print $1}')
    read -ra FULL4_IDX <<< "$INSTANCE_INDICES"
    n_idx="${#FULL4_IDX[@]}"
    for (( worker = 0; worker < EVAL_WORKERS; worker++ )); do
        shard_count=0
        shard_first=""
        for (( idx = worker; idx < ${#FULL4_TASKS[@]}; idx += EVAL_WORKERS )); do
            [[ -z "$shard_first" ]] && shard_first="${FULL4_TASKS[idx]}"
            shard_count=$((shard_count + 1))
        done
        printf 'WORKER %d: tasks=%d (first=%s) indices=[%s] rollouts=%d gpu=%d port=%d\n' \
            "$worker" "$shard_count" "$shard_first" "$INSTANCE_INDICES" \
            "$((shard_count * n_idx * NUM_ROLLOUTS))" "$worker" "$((BASE_PORT + worker))"
    done
    printf 'FULL4 TOTAL: tasks=%d instances_per_task=%d rollouts=%d videos=%d\n' \
        "${#FULL4_TASKS[@]}" "$n_idx" \
        "$(( ${#FULL4_TASKS[@]} * n_idx * NUM_ROLLOUTS ))" "$(( ${#FULL4_TASKS[@]} * n_idx * NUM_ROLLOUTS ))"
elif [[ "$EVAL_PROFILE" == smoke ]]; then
    printf 'GPU MAP: policy=0 simulator=%s task=%s\n' "$SMOKE_EVAL_GPU" "$SMOKE_TASK"
else
    printf 'GPU MAP: policy=0 simulator=1\n'
fi

if [[ "${DRY_RUN:-0}" == 1 ]]; then
    echo "DRY_RUN=1: submission skipped."
    exit 0
fi

command -v "$AI_HUB_CLI" >/dev/null 2>&1 || {
    echo "AIHub CLI not found: $AI_HUB_CLI" >&2
    exit 1
}
RECEIPT_ROOT="${RECEIPT_ROOT:-$OUTPUT_BASE/.submission_receipts}"
umask 077
mkdir -p "$RECEIPT_ROOT"
RECEIPT_PATH="$RECEIPT_ROOT/$RUN_ID.json"
CLI_OUTPUT_PATH="$RECEIPT_ROOT/$RUN_ID.stdout.log"
[[ ! -e "$RECEIPT_PATH" && ! -e "$CLI_OUTPUT_PATH" ]] || {
    echo "Submission receipt already exists for run: $RUN_ID" >&2
    exit 1
}

cd "$BUNDLE_DIR"
set +e
"$AI_HUB_CLI" train mdl \
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
    "--max_failover_times=$MAX_FAILOVER_TIMES" \
    "${BOOTSTRAP_ENV_ARGS[@]}" \
    "--debug_mode" \
    "--ignore=__pycache__,*.pyc,*.pyo" 2>&1 | tee "$CLI_OUTPUT_PATH"
submit_status=${PIPESTATUS[0]}
set -e
python3 - "$CLI_OUTPUT_PATH" "$RECEIPT_PATH" "$RUN_ID" "$JOB_NAME" \
    "$EVAL_PROFILE" "$PERSISTENT_OUTPUT" "$EXTERNAL_SUBTASK_SELECTOR" "$PI05_MODEL_FORMAT" \
    "$PI05_ENV_ARCHIVE" "$CHECKPOINT_SOURCE" "$SUBTASK_PREDICTION" \
    "$PROVIDED_SUBTASK_AS_PROMPT" "$ALGO_NAME" "$NEBULA_PIP_NO_DEPS" "$submit_status" <<'PY'
import json
import re
import sys
from pathlib import Path

(
    output_path,
    receipt_path,
    run_id,
    job_name,
    profile,
    persistent_output,
    selector,
    model_format,
    policy_env_archive,
    checkpoint_source,
    subtask_prediction,
    provided_subtask_as_prompt,
    algo_name,
    nebula_pip_no_deps,
    exit_code,
) = sys.argv[1:]
text = Path(output_path).read_text(errors="replace")
identifiers = set()
urls = set(re.findall(r"https?://[^\s<>\"']+", text))
for line in text.splitlines():
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, dict):
        for key in ("job_id", "jobId", "task_id", "taskId", "instance_id", "instanceId"):
            if value.get(key) is not None:
                identifiers.add(str(value[key]))
    for match in re.finditer(
        r"(?i)\b(?:job|task|instance)[_ -]?id\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9._:-]*)",
        line,
    ):
        identifiers.add(match.group(1))
ready_path = Path(checkpoint_source) / "_READY"
ready = json.loads(ready_path.read_text()) if ready_path.is_file() else {}
receipt = {
    "run_id": run_id,
    "job_name": job_name,
    "profile": profile,
    "model_format": model_format,
    "algo_name": algo_name,
    "nebula_pip_no_deps": nebula_pip_no_deps == "1",
    "policy_env_archive": policy_env_archive,
    "checkpoint_source": checkpoint_source,
    "checkpoint_manifest_sha256": ready.get("manifest_sha256"),
    "conditioning_mode": (
        "provided_subtask_as_prompt"
        if provided_subtask_as_prompt == "1"
        else "joint_subtask_prediction"
        if subtask_prediction == "1"
        else "whole_task"
    ),
    "persistent_output": persistent_output,
    "exit_code": int(exit_code),
    "job_identifiers": sorted(identifiers),
    "urls": sorted(urls),
    "stdout_stderr": output_path,
}
if selector:
    receipt["evaluation_kind"] = "selector_diagnostic"
    receipt["selector_diagnostics"] = {
        "selector": selector,
        "label_format": "legacy",
        "selector_labels_sha256": "fd785101426544b24ce756a84e72e2f5971776cfe6ebf7bde18e5ff7fb58ac04",
        "legacy_segments_sha256": "999c58103ad91046d3a1215e202ab2022c645b55524261b3eeed6d66e7f1322d",
        "training_annotation_sha256": "870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d",
        "oracle": True,
        "privileged": True,
        "official_benchmark": False,
    }
Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
printf 'Submission receipt: %s\n' "$RECEIPT_PATH"
printf 'Submission output: %s\n' "$CLI_OUTPUT_PATH"
exit "$submit_status"
