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
    smoke|speed|two_tasks|reduced|full|full4|subtasks100) ;;
    *) echo "EVAL_PROFILE must be smoke, speed, two_tasks, reduced, full, full4, or subtasks100" >&2; exit 2 ;;
esac
if [[ "$EVAL_PROFILE" == two_tasks ]]; then
    EVAL_WORKERS="${EVAL_WORKERS:-4}"
    if [[ "$EVAL_WORKERS" == 8 ]]; then
        CLUSTER_FILE="${CLUSTER_FILE:-nebula/multihost_cluster.json}"
    else
        CLUSTER_FILE="${CLUSTER_FILE:-nebula/multihost_cluster_4gpu.json}"
    fi
    BASE_PORT="${BASE_PORT:-8000}"
    PI05_MEM_FRACTION="${PI05_MEM_FRACTION:-0.35}"
    WRITE_VIDEO="${WRITE_VIDEO:-1}"
    SPEED_TASK=""
    SPEED_EXPECTED_INSTANCE_IDS=""
    INSTANCE_INDICES="${INSTANCE_INDICES:-0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19}"
    NUM_ROLLOUTS="${NUM_ROLLOUTS:-3}"
    MAX_STEPS="${MAX_STEPS:-}"
elif [[ "$EVAL_PROFILE" == speed || "$EVAL_PROFILE" == full4 || "$EVAL_PROFILE" == subtasks100 ]]; then
    EVAL_WORKERS="${EVAL_WORKERS:-4}"
    if [[ ( "$EVAL_PROFILE" == full4 || "$EVAL_PROFILE" == subtasks100 ) && "$EVAL_WORKERS" == 8 ]]; then
        CLUSTER_FILE="${CLUSTER_FILE:-nebula/multihost_cluster.json}"
    else
        CLUSTER_FILE="${CLUSTER_FILE:-nebula/multihost_cluster_4gpu.json}"
    fi
    BASE_PORT="${BASE_PORT:-8000}"
    PI05_MEM_FRACTION="${PI05_MEM_FRACTION:-0.35}"
    if [[ "$EVAL_PROFILE" == speed ]]; then
        WRITE_VIDEO="${WRITE_VIDEO:-0}"
        SPEED_TASK="${SPEED_TASK:-make_rose_centerpieces}"
        INSTANCE_INDICES="${INSTANCE_INDICES:-0 1 2 3}"
        SPEED_EXPECTED_INSTANCE_IDS="${SPEED_EXPECTED_INSTANCE_IDS:-301 302 303 304}"
        MAX_STEPS="${MAX_STEPS:-100}"
    else
        WRITE_VIDEO="${WRITE_VIDEO:-1}"
        SPEED_TASK=""
        SPEED_EXPECTED_INSTANCE_IDS=""
        INSTANCE_INDICES="${INSTANCE_INDICES:-0 1}"
        NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
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
if [[ "$EVAL_PROFILE" == smoke && "$PI05_MODEL_FORMAT" == torch ]]; then
    SMOKE_TASK="${SMOKE_TASK:-carrying_in_groceries}"
    SMOKE_EVAL_GPU="${SMOKE_EVAL_GPU:-0}"
else
    SMOKE_TASK="${SMOKE_TASK:-turning_on_radio}"
    SMOKE_EVAL_GPU="${SMOKE_EVAL_GPU:-1}"
fi
CONFIRM_FULL="${CONFIRM_FULL:-0}"
SUBTASK_PREDICTION="${SUBTASK_PREDICTION:-1}"
SUBTASK_INTERVAL="${SUBTASK_INTERVAL:-}"
SUBTASK_VOTE_WINDOW="${SUBTASK_VOTE_WINDOW:-}"
SUBTASK_VOTE_THRESHOLD="${SUBTASK_VOTE_THRESHOLD:-}"
EXECUTE_HORIZON="${EXECUTE_HORIZON:-16}"
EXTERNAL_SUBTASK_SELECTOR="${EXTERNAL_SUBTASK_SELECTOR:-}"
SELECTOR_DEBOUNCE_REFRESHES="${SELECTOR_DEBOUNCE_REFRESHES:-}"
MAX_FAILOVER_TIMES="${MAX_FAILOVER_TIMES:-0}"
if [[ ( "$EVAL_PROFILE" == full || "$EVAL_PROFILE" == full4 || "$EVAL_PROFILE" == subtasks100 ) && "$CONFIRM_FULL" != 1 ]]; then
    echo "Full evaluation requires CONFIRM_FULL=1" >&2
    exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    RUN_ID="${RUN_ID:-pi05_selector_diagnostic_117541_${EVAL_PROFILE}_$STAMP}"
    JOB_NAME="${JOB_NAME:-pi05-selector-diagnostic-${EVAL_PROFILE}-$STAMP}"
elif [[ -n "$SUBTASK_VOTE_WINDOW" || -n "$SUBTASK_VOTE_THRESHOLD" ]]; then
    RUN_ID="${RUN_ID:-pi05_subtask_vote_${EVAL_PROFILE}_$STAMP}"
    JOB_NAME="${JOB_NAME:-pi05-subtask-vote-${EVAL_PROFILE}-$STAMP}"
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
if [[ "$PI05_MODEL_FORMAT" == torch ]]; then
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
PI05_REPO_ID="${PI05_REPO_ID:-b1k_two_tasks_joint_subtask_s1}"
INSTANCE_INDICES="${INSTANCE_INDICES:-0 1 2 3 4}"
NUM_ROLLOUTS="${NUM_ROLLOUTS:-1}"
MAX_STEPS="${MAX_STEPS:-}"
NEBULA_WORKER_DRY_RUN="${NEBULA_WORKER_DRY_RUN:-0}"
KEEP_RUNTIME="${KEEP_RUNTIME:-0}"
[[ "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "EVAL_WORKERS must be a positive integer" >&2; exit 2; }
[[ "$NUM_ROLLOUTS" =~ ^[1-9][0-9]*$ ]] || { echo "NUM_ROLLOUTS must be a positive integer" >&2; exit 2; }
[[ "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || { echo "BASE_PORT must be a positive integer" >&2; exit 2; }
(( BASE_PORT + EVAL_WORKERS - 1 <= 65535 )) || { echo "worker port range exceeds 65535" >&2; exit 2; }
[[ "$MAX_FAILOVER_TIMES" =~ ^[0-9]+$ ]] || { echo "MAX_FAILOVER_TIMES must be a non-negative integer" >&2; exit 2; }
[[ "$NEBULA_PIP_NO_DEPS" == 0 || "$NEBULA_PIP_NO_DEPS" == 1 ]] || {
    echo "NEBULA_PIP_NO_DEPS must be 0 or 1" >&2; exit 2; }
[[ "$WRITE_VIDEO" == 0 || "$WRITE_VIDEO" == 1 ]] || { echo "WRITE_VIDEO must be 0 or 1" >&2; exit 2; }
[[ "$SUBTASK_PREDICTION" == 0 || "$SUBTASK_PREDICTION" == 1 ]] || { echo "SUBTASK_PREDICTION must be 0 or 1" >&2; exit 2; }
[[ "$EXECUTE_HORIZON" =~ ^[1-9][0-9]*$ ]] || { echo "EXECUTE_HORIZON must be a positive integer" >&2; exit 2; }
if [[ -n "$SUBTASK_INTERVAL" ]]; then
    [[ "$SUBTASK_INTERVAL" =~ ^[1-9][0-9]*$ ]] || {
        echo "SUBTASK_INTERVAL must be a positive integer" >&2; exit 2; }
    [[ "$SUBTASK_PREDICTION" == 1 ]] || {
        echo "SUBTASK_INTERVAL requires SUBTASK_PREDICTION=1" >&2; exit 2; }
fi
if [[ -n "$SUBTASK_VOTE_WINDOW" || -n "$SUBTASK_VOTE_THRESHOLD" ]]; then
    [[ "$SUBTASK_VOTE_WINDOW" =~ ^[1-9][0-9]*$ ]] && (( SUBTASK_VOTE_WINDOW >= 2 )) || {
        echo "SUBTASK_VOTE_WINDOW must be an integer of at least 2" >&2; exit 2; }
    [[ "$SUBTASK_VOTE_THRESHOLD" =~ ^[1-9][0-9]*$ ]] \
        && (( SUBTASK_VOTE_THRESHOLD >= 2 && SUBTASK_VOTE_THRESHOLD <= SUBTASK_VOTE_WINDOW )) || {
        echo "SUBTASK_VOTE_THRESHOLD must be between 2 and SUBTASK_VOTE_WINDOW" >&2; exit 2; }
    (( SUBTASK_VOTE_THRESHOLD * 2 > SUBTASK_VOTE_WINDOW )) || {
        echo "SUBTASK_VOTE_THRESHOLD must be a strict majority of SUBTASK_VOTE_WINDOW" >&2; exit 2; }
    [[ "$SUBTASK_PREDICTION" == 1 && -n "$SUBTASK_INTERVAL" ]] || {
        echo "Temporal subtask voting requires SUBTASK_PREDICTION=1 and an explicit SUBTASK_INTERVAL" >&2; exit 2; }
    [[ -z "$EXTERNAL_SUBTASK_SELECTOR" && -z "${EXTERNAL_SUBTASK:-}" ]] || {
        echo "Temporal subtask voting cannot be combined with external subtask conditioning" >&2; exit 2; }
fi
[[ "$SMOKE_EVAL_GPU" =~ ^[0-9]+$ ]] || { echo "SMOKE_EVAL_GPU must be a non-negative integer" >&2; exit 2; }
[[ "$PI05_MEM_FRACTION" =~ ^0\.[0-9]+$ ]] || { echo "PI05_MEM_FRACTION must be between 0 and 1" >&2; exit 2; }
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    [[ "$EXTERNAL_SUBTASK_SELECTOR" == state_machine_v1 ]] || {
        echo "EXTERNAL_SUBTASK_SELECTOR must be state_machine_v1" >&2
        exit 2
    }
    [[ "$EVAL_PROFILE" == smoke || "$EVAL_PROFILE" == two_tasks ]] || {
        echo "EXTERNAL_SUBTASK_SELECTOR is supported only for EVAL_PROFILE=smoke or two_tasks" >&2
        exit 2
    }
    [[ "$SUBTASK_PREDICTION" == 1 ]] || {
        echo "EXTERNAL_SUBTASK_SELECTOR requires SUBTASK_PREDICTION=1" >&2
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
if [[ -n "$SELECTOR_DEBOUNCE_REFRESHES" ]]; then
    [[ "$SELECTOR_DEBOUNCE_REFRESHES" =~ ^[1-9][0-9]*$ ]] || {
        echo "SELECTOR_DEBOUNCE_REFRESHES must be a positive integer" >&2
        exit 2
    }
    [[ "$EXTERNAL_SUBTASK_SELECTOR" == state_machine_v1 ]] || {
        echo "SELECTOR_DEBOUNCE_REFRESHES requires EXTERNAL_SUBTASK_SELECTOR=state_machine_v1" >&2
        exit 2
    }
fi
SELECTOR_DEBOUNCE_EFFECTIVE=""
if [[ "$EXTERNAL_SUBTASK_SELECTOR" == state_machine_v1 ]]; then
    SELECTOR_DEBOUNCE_EFFECTIVE="${SELECTOR_DEBOUNCE_REFRESHES:-3}"
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
    (( EVAL_WORKERS == 4 || EVAL_WORKERS == 8 )) || {
        echo "two_tasks profile requires EVAL_WORKERS=4 or 8" >&2; exit 2; }
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
    (( EVAL_WORKERS == 4 || EVAL_WORKERS == 8 )) || {
        echo "full4 profile requires EVAL_WORKERS=4 or 8" >&2; exit 2; }
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

SUBTASK_VALIDATION_PLAN_PATH="${SUBTASK_VALIDATION_PLAN_PATH:-}"
SUBTASK_COLLECTION_PLAN_PATH="${SUBTASK_COLLECTION_PLAN_PATH:-}"
SUBTASK_NATIVE_INPUT_ROOT="${SUBTASK_NATIVE_INPUT_ROOT:-}"
[[ -z "$SUBTASK_VALIDATION_PLAN_PATH" || -z "$SUBTASK_COLLECTION_PLAN_PATH" ]] || { echo "Validation and collection plans are mutually exclusive" >&2; exit 2; }
if [[ -n "${PI05_PAIRED_RETEST_SEED:-}" ]]; then
    [[ "$EVAL_PROFILE" == subtasks100 && -z "$SUBTASK_VALIDATION_PLAN_PATH" && -z "$SUBTASK_COLLECTION_PLAN_PATH" ]] || { echo "Paired retest requires the full-panel profile" >&2; exit 2; }
    python3 - "$PI05_PAIRED_RETEST_SEED" <<'RETEST_SEED'
import sys
seed = int(sys.argv[1])
if not 0 <= seed < 2**32:
    raise SystemExit("Paired retest seed must be a uint32 integer")
RETEST_SEED
fi
if [[ "$EVAL_PROFILE" == subtasks100 ]]; then
    if [[ -n "$SUBTASK_VALIDATION_PLAN_PATH" || -n "$SUBTASK_COLLECTION_PLAN_PATH" ]]; then
        (( EVAL_WORKERS == 1 )) || { echo "Subtask validation requires one worker/GPU" >&2; exit 2; }
    else
        (( EVAL_WORKERS == 4 || EVAL_WORKERS == 8 )) || { echo "subtasks100 requires 4 or 8 workers" >&2; exit 2; }
    fi
    [[ "$PI05_MODEL_FORMAT" == jax && "$SUBTASK_PREDICTION" == 1 && "$WRITE_VIDEO" == 1 ]] || {
        echo "subtasks100 requires a joint JAX checkpoint and videos" >&2; exit 2; }
    [[ -z "$EXTERNAL_SUBTASK_SELECTOR" && -z "$SUBTASK_INTERVAL" && -z "$SUBTASK_VOTE_WINDOW" ]] || {
        echo "subtasks100 uses fixed annotated subtasks without selector/voting" >&2; exit 2; }
    : "${SUBTASK_PANEL_PATH:?set SUBTASK_PANEL_PATH to the frozen full100 panel}"
    : "${SUBTASK_RAW_INVENTORY_PATH:?set SUBTASK_RAW_INVENTORY_PATH}"
    : "${SUBTASK_RAW_SOURCE:?set SUBTASK_RAW_SOURCE to the complete raw dataset OSS prefix}"
    [[ -s "$SUBTASK_PANEL_PATH" && -s "$SUBTASK_RAW_INVENTORY_PATH" ]] || {
        echo "Frozen subtask inputs are missing" >&2; exit 2; }
    SUBTASK_SAMPLE_MAX_STEPS="${SUBTASK_SAMPLE_MAX_STEPS:-2048}"
    [[ "$SUBTASK_SAMPLE_MAX_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "Subtask maximum steps must be positive" >&2; exit 2; }
    if [[ -n "$SUBTASK_VALIDATION_PLAN_PATH" ]]; then
        python3 - "$SUBTASK_PANEL_PATH" "$SUBTASK_VALIDATION_PLAN_PATH" "$SUBTASK_SAMPLE_MAX_STEPS" "$SUBTASK_NATIVE_INPUT_ROOT" <<'PROBE_PREFLIGHT'
import sys
sys.path.insert(0, "scripts")
from subtask_sample_panel import load_panel
from eval_b1k_subtask_probe import load_plan
plan, samples = load_plan(sys.argv[2], load_panel(sys.argv[1]))
if plan["max_steps"] != int(sys.argv[3]): raise SystemExit("Validation step limit differs from plan")
if plan['kind']=='subtask_native_snapshot_model_validation':
    if not sys.argv[4]:raise SystemExit('Native snapshot validation requires frozen snapshot inputs')
    from validate_b1k_native_snapshot_model import validate_input_artifacts
    validate_input_artifacts(sys.argv[4],load_panel(sys.argv[1]),plan)
elif sys.argv[4]:raise SystemExit('Native snapshot inputs require an explicit native validation plan')
print(f"SUBTASK VALIDATION: samples={len(samples)} tasks={len({s['task'] for s in samples})} max_steps={plan['max_steps']} workers=1")
PROBE_PREFLIGHT
    fi
    if [[ -n "$SUBTASK_COLLECTION_PLAN_PATH" ]]; then
        python3 - "$SUBTASK_PANEL_PATH" "$SUBTASK_COLLECTION_PLAN_PATH" "$SUBTASK_SAMPLE_MAX_STEPS" <<'COLLECTION_PREFLIGHT'
import sys
sys.path.insert(0, "scripts")
from subtask_sample_panel import load_panel
from subtask_improvement import load_plan
plan, samples = load_plan(sys.argv[2], load_panel(sys.argv[1]))
if plan["max_steps"] != int(sys.argv[3]) or not plan.get("checkpoint_identity"):
    raise SystemExit("Collection limit or verified checkpoint identity is missing")
if plan["expected_trials"] > 300: raise SystemExit("Pilot collection is capped at 300 trials")
selected = [samples[k] for k in {t["sample_id"] for t in plan["trials"]}]
print(f"SUBTASK COLLECTION: trials={plan['expected_trials']} starts={len(selected)} max_steps={plan['max_steps']} workers=1")
COLLECTION_PREFLIGHT
    fi
fi

SUBMIT_STORAGE_MODE="${SUBMIT_STORAGE_MODE:-oss}"
OSS_MOUNT_ROOT="${OSS_MOUNT_ROOT:-/data/oss_bucket_0}"
export OSS_ACCESS_ID OSS_ACCESS_KEY OSS_BUCKET OSS_ENDPOINT OSS_MOUNT_ROOT
case "$SUBMIT_STORAGE_MODE" in
    oss|mounted) ;;
    *) echo "SUBMIT_STORAGE_MODE must be oss or mounted" >&2; exit 2 ;;
esac
runtime_path() {
    case "$1" in
        "oss://$OSS_BUCKET/"*) printf '%s/%s\n' "${OSS_MOUNT_ROOT%/}" "${1#oss://$OSS_BUCKET/}" ;;
        oss://*) echo "Input URI must use configured OSS_BUCKET" >&2; return 1 ;;
        *) printf '%s\n' "$1" ;;
    esac
}
for variable in EVAL_ENV_ARCHIVE PI05_ENV_ARCHIVE OPENPI_CACHE_ARCHIVE ASSET_ROOT CHECKPOINT_SOURCE OUTPUT_BASE; do
    runtime_value="$(runtime_path "${!variable}")" || exit 1
    printf -v "$variable" '%s' "$runtime_value"
done
PERSISTENT_OUTPUT="$OUTPUT_BASE/$RUN_ID"
if [[ "$EVAL_PROFILE" == subtasks100 ]]; then SUBTASK_RAW_SOURCE="$(runtime_path "$SUBTASK_RAW_SOURCE")"; fi
OSS_RECEIPT_PREFIX="$OUTPUT_BASE/.submission_receipts/$RUN_ID"
PREFLIGHT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/pi05-submit-preflight.XXXXXX")"
READY_CACHE="$PREFLIGHT_DIR/ready.json"
cleanup_submission() {
    (( BASH_SUBSHELL == 0 )) || return 0
    cd "$REPO_ROOT"
    rm -rf -- "$PREFLIGHT_DIR"
    if [[ -n "${BUNDLE_DIR:-}" ]]; then
        if [[ "${KEEP_BUNDLE:-0}" == 1 ]]; then
            printf 'Kept submission bundle: %s\n' "$BUNDLE_DIR"
        else
            rm -rf -- "$BUNDLE_DIR"
        fi
    fi
}
trap cleanup_submission EXIT
# Fail on missing/corrupt runtime wheels before remote checks or run reservation.
python3 "$SCRIPT_DIR/eval_runtime.py" verify --wheel-dir "$SCRIPT_DIR/runtime-wheels"
if [[ "$SUBMIT_STORAGE_MODE" == oss ]]; then
    SUBMIT_PYTHON="${NEBULA_SUBMIT_PYTHON:-python3}"
    if ! "$SUBMIT_PYTHON" -c 'import oss2' >/dev/null 2>&1; then
        if [[ -z "${NEBULA_SUBMIT_PYTHON:-}" && -x "$REPO_ROOT/../.venv/bin/python" ]] \
            && "$REPO_ROOT/../.venv/bin/python" -c 'import oss2' >/dev/null 2>&1; then
            SUBMIT_PYTHON="$REPO_ROOT/../.venv/bin/python"
        else
            echo "Install nebula/requirements-submit.txt and set NEBULA_SUBMIT_PYTHON to that Python" >&2
            exit 1
        fi
    fi
    PREFLIGHT_ARGS=(preflight --checkpoint "$CHECKPOINT_SOURCE" --assets "$ASSET_ROOT"
        --repo-id "$PI05_REPO_ID" --model-format "$PI05_MODEL_FORMAT"
        --archive "$EVAL_ENV_ARCHIVE" --archive "$PI05_ENV_ARCHIVE" --archive "$OPENPI_CACHE_ARCHIVE"
        --output "$PERSISTENT_OUTPUT" --receipt-prefix "$OSS_RECEIPT_PREFIX" --ready-out "$READY_CACHE")
    if [[ "$EVAL_PROFILE" == two_tasks || "$EVAL_PROFILE" == full4 || "$EVAL_PROFILE" == subtasks100 || "$PI05_MODEL_FORMAT" == torch ]]; then
        PREFLIGHT_ARGS+=(--staged)
    fi
    "$SUBMIT_PYTHON" "$SCRIPT_DIR/submit_oss.py" "${PREFLIGHT_ARGS[@]}"
    if [[ "$EVAL_PROFILE" == subtasks100 ]]; then
        "$SUBMIT_PYTHON" "$SCRIPT_DIR/subtask_eval.py" preflight --panel "$SUBTASK_PANEL_PATH" \
            --inventory "$SUBTASK_RAW_INVENTORY_PATH" --raw-source "$SUBTASK_RAW_SOURCE"
    fi
else
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
    if [[ "$EVAL_PROFILE" == two_tasks || "$EVAL_PROFILE" == full4 || "$EVAL_PROFILE" == subtasks100 || "$PI05_MODEL_FORMAT" == torch ]]; then
        for path in "$CHECKPOINT_SOURCE/_CHECKPOINT_METADATA" \
            "$CHECKPOINT_SOURCE/SHA256SUMS.inference" "$CHECKPOINT_SOURCE/_READY"; do
            [[ -s "$path" ]] || { echo "Staged checkpoint input missing or empty: $path" >&2; exit 1; }
        done
        (
            cd "$CHECKPOINT_SOURCE"
            sha256sum -c SHA256SUMS.inference
        )
    fi
    (
        cd "$ASSET_ROOT"
        sha256sum -c SHA256SUMS
    )
    [[ ! -e "$PERSISTENT_OUTPUT" ]] || { echo "Output already exists: $PERSISTENT_OUTPUT" >&2; exit 1; }
    python3 - "$CHECKPOINT_SOURCE/_READY" "$READY_CACHE" <<'CACHE_PY'
import json
import sys
from pathlib import Path
source = Path(sys.argv[1])
Path(sys.argv[2]).write_text(source.read_text() if source.is_file() else '{}\n')
CACHE_PY
fi

if [[ -n "$SUBTASK_COLLECTION_PLAN_PATH" ]]; then
    python3 - "$READY_CACHE" "$SUBTASK_COLLECTION_PLAN_PATH" <<'COLLECTION_CHECKPOINT'
import json,sys
ready=json.load(open(sys.argv[1]));plan=json.load(open(sys.argv[2]));identity=plan.get('checkpoint_identity') or {}
if identity.get('checkpoint_step') != ready.get('step') or identity.get('checkpoint_manifest_sha256') != ready.get('manifest_sha256'):
    raise SystemExit('Collection checkpoint differs from its verified baseline identity')
COLLECTION_CHECKPOINT
fi

if [[ -n "$SUBTASK_VALIDATION_PLAN_PATH" ]]; then
    python3 - "$READY_CACHE" "$SUBTASK_VALIDATION_PLAN_PATH" <<'NATIVE_CHECKPOINT'
import json,sys
ready=json.load(open(sys.argv[1]));plan=json.load(open(sys.argv[2]))
if plan.get('require_recorded_native_state') or plan.get('kind') == 'subtask_native_snapshot_model_validation':
    if plan.get('checkpoint_step') != ready.get('step') or plan.get('checkpoint_manifest_sha256') != ready.get('manifest_sha256'):
        raise SystemExit('Native validation checkpoint differs from its frozen plan')
NATIVE_CHECKPOINT
fi

CODE_COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then CODE_DIRTY=1; else CODE_DIRTY=0; fi

BUNDLE_PARENT="${BUNDLE_PARENT:-${TMPDIR:-/tmp}}"
mkdir -p "$BUNDLE_PARENT"
BUNDLE_DIR="$(mktemp -d "$BUNDLE_PARENT/pi05-eval-nebula-bundle.XXXXXX")"

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
if [[ "$EVAL_PROFILE" == subtasks100 ]]; then
    for file in eval_b1k_dynamic.py eval_b1k_subtask_dynamic.py eval_b1k_queue.py eval_b1k_subtask_samples.py \
        pi05_eval_queue.py subtask_sample_panel.py subtask_completion.py recorded_native_state.py subtask_source_exclusion.py summarize_subtask_closed_loop.py eval_b1k_subtask_probe.py diagnose_b1k_grasp_restore.py validate_b1k_grasp_reconstruction.py validate_b1k_native_snapshot_model.py \
        subtask_improvement.py subtask_trajectory.py collect_b1k_subtasks.py export_subtask_collection.py verify_subtask_collection_export.py subtask_paired_retest.py; do
        cp -a "$REPO_ROOT/scripts/$file" "$BUNDLE_DIR/scripts/"
    done
    cp -a "$SCRIPT_DIR/subtask_eval.py" "$SCRIPT_DIR/submit_oss.py" "$BUNDLE_DIR/nebula/"
    cp "$SUBTASK_PANEL_PATH" "$BUNDLE_DIR/nebula/subtask_panel.json"
    cp "$SUBTASK_RAW_INVENTORY_PATH" "$BUNDLE_DIR/nebula/subtask_raw_inventory.json"
    if [[ -n "$SUBTASK_COLLECTION_PLAN_PATH" ]]; then cp "$SUBTASK_COLLECTION_PLAN_PATH" "$BUNDLE_DIR/nebula/subtask_collection_plan.json"; fi
    if [[ -n "$SUBTASK_VALIDATION_PLAN_PATH" ]]; then cp "$SUBTASK_VALIDATION_PLAN_PATH" "$BUNDLE_DIR/nebula/subtask_validation_plan.json"; fi
    if [[ -n "$SUBTASK_NATIVE_INPUT_ROOT" ]]; then
        python3 - "$SUBTASK_NATIVE_INPUT_ROOT" "$BUNDLE_DIR/nebula/native_snapshot_input" "$SUBTASK_PANEL_PATH" "$SUBTASK_VALIDATION_PLAN_PATH" <<'NATIVE_INPUT_PACK'
import sys
sys.path.insert(0,'scripts')
from subtask_sample_panel import load_panel
from eval_b1k_subtask_probe import load_plan
from validate_b1k_native_snapshot_model import pack_input_artifacts
panel=load_panel(sys.argv[3]);plan,_=load_plan(sys.argv[4],panel)
pack_input_artifacts(sys.argv[1],sys.argv[2],panel,plan)
NATIVE_INPUT_PACK
    fi
fi
cp -a "$SCRIPT_DIR/eval_storage.py" "$SCRIPT_DIR/eval_runtime.py" "$SCRIPT_DIR/check_eval_runtime.py" \
    "$SCRIPT_DIR/requirements-eval-runtime.txt" "$BUNDLE_DIR/nebula/"
mkdir -p "$BUNDLE_DIR/nebula/runtime-wheels"
cp "$SCRIPT_DIR/runtime-wheels/warp_lang-1.12.0-py3-none-manylinux_2_28_x86_64.whl" \
    "$BUNDLE_DIR/nebula/runtime-wheels/"
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
    write_setting CLUSTER_FILE "$CLUSTER_FILE"
    write_setting PI05_MEM_FRACTION "$PI05_MEM_FRACTION"
    write_setting WRITE_VIDEO "$WRITE_VIDEO"
    write_setting SUBTASK_PREDICTION "$SUBTASK_PREDICTION"
    write_setting SUBTASK_INTERVAL "$SUBTASK_INTERVAL"
    write_setting SUBTASK_VOTE_WINDOW "$SUBTASK_VOTE_WINDOW"
    write_setting SUBTASK_VOTE_THRESHOLD "$SUBTASK_VOTE_THRESHOLD"
    write_setting EXECUTE_HORIZON "$EXECUTE_HORIZON"
    write_setting EXTERNAL_SUBTASK_SELECTOR "$EXTERNAL_SUBTASK_SELECTOR"
    write_setting SELECTOR_DEBOUNCE_REFRESHES "$SELECTOR_DEBOUNCE_REFRESHES"
    if [[ "$EVAL_PROFILE" == subtasks100 ]]; then
        write_setting SUBTASK_RAW_SOURCE "$SUBTASK_RAW_SOURCE"
        write_setting SUBTASK_SAMPLE_MAX_STEPS "$SUBTASK_SAMPLE_MAX_STEPS"
        write_setting PI05_PAIRED_RETEST_SEED "${PI05_PAIRED_RETEST_SEED:-}"
        write_setting SUBTASK_COLLECTION_ENABLED "$([[ -n "$SUBTASK_COLLECTION_PLAN_PATH" ]] && echo 1 || echo 0)"
        write_setting SUBTASK_VALIDATION_ENABLED "$([[ -n "$SUBTASK_VALIDATION_PLAN_PATH" ]] && echo 1 || echo 0)"
    fi
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
printf 'SUBMIT_STORAGE_MODE: %s\n' "$SUBMIT_STORAGE_MODE"
printf 'QUEUE: %s (%s)\n' "$QUEUE" "$GPU_CARD_TYPE"
printf 'ALGO_NAME: %s\n' "$ALGO_NAME"
printf 'NEBULA_PIP_NO_DEPS: %s\n' "$NEBULA_PIP_NO_DEPS"
printf 'PROFILE: %s\n' "$EVAL_PROFILE"
printf 'MODEL_FORMAT: %s\n' "$PI05_MODEL_FORMAT"
printf 'CLUSTER: %s\n' "$CLUSTER_FILE"
printf 'SUBTASK_PREDICTION: %s\n' "$SUBTASK_PREDICTION"
printf 'SUBTASK_INTERVAL: %s\n' "${SUBTASK_INTERVAL:-<policy-default>}"
printf 'SUBTASK_VOTE_WINDOW: %s\n' "${SUBTASK_VOTE_WINDOW:-<disabled>}"
printf 'SUBTASK_VOTE_THRESHOLD: %s\n' "${SUBTASK_VOTE_THRESHOLD:-<disabled>}"
printf 'EXECUTE_HORIZON: %s\n' "$EXECUTE_HORIZON"
printf 'EXTERNAL_SUBTASK_SELECTOR: %s\n' "${EXTERNAL_SUBTASK_SELECTOR:-<disabled>}"
printf 'SELECTOR_DEBOUNCE_REFRESHES: %s\n' "${SELECTOR_DEBOUNCE_REFRESHES:-<selector-default>}"
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    printf 'SELECTOR_DEBOUNCE_EFFECTIVE: %s\n' "$SELECTOR_DEBOUNCE_EFFECTIVE"
    printf 'SELECTOR_STALL_STEPS: 2048\n'
    printf 'EVALUATION_KIND: selector diagnostic (label_format=legacy oracle=true privileged=true official_benchmark=false)\n'
elif [[ -n "$SUBTASK_VOTE_WINDOW" ]]; then
    printf 'EVALUATION_KIND: temporal subtask voting (oracle=false privileged=false official_benchmark=false inference_ablation=true)\n'
fi
if [[ "$EVAL_PROFILE" == two_tasks ]]; then
    printf 'EXPECTED_ROLLOUTS: 120 (%d co-located workers, %d each)\n' \
        "$EVAL_WORKERS" "$((120 / EVAL_WORKERS))"
elif [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    printf 'EXPECTED_ROLLOUTS: %s smoke rollout(s)\n' "$NUM_ROLLOUTS"
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
            if (( EVAL_WORKERS == 4 )); then
                worker_task=carrying_in_groceries,make_rose_centerpieces
                shard_index="$worker"
                worker_rollouts=30
            elif (( worker < 4 )); then
                worker_task=carrying_in_groceries
                shard_index="$worker"
                worker_rollouts=15
            else
                worker_task=make_rose_centerpieces
                shard_index="$((worker - 4))"
                worker_rollouts=15
            fi
            printf 'WORKER %d: tasks=%s indices=[%s] rollouts=%d gpu=%d port=%d\n' \
                "$worker" "$worker_task" "${TWO_TASK_SHARDS[shard_index]}" "$worker_rollouts" \
                "$worker" "$((BASE_PORT + worker))"
        else
            printf 'WORKER %d: policy=%d simulator=%d port=%d\n' \
                "$worker" "$worker" "$worker" "$((BASE_PORT + worker))"
        fi
    done
elif [[ "$EVAL_PROFILE" == subtasks100 ]]; then
    python3 - "$BUNDLE_DIR/nebula/subtask_panel.json" "$EVAL_WORKERS" "$SUBTASK_SAMPLE_MAX_STEPS" <<'SUBTASK_CONTRACT'
import json,sys
panel=json.load(open(sys.argv[1]))
print(f"PARENT SUBTASK PANEL: tasks={len(panel['selection']['tasks'])} episode_jobs={len(panel['episodes'])} rollouts={len(panel['samples'])} unsupported={len(panel['exclusions'])} max_steps={sys.argv[3]} workers={sys.argv[2]}")
print("CONDITIONING: joint checkpoint with provided fixed subtask; training-state diagnostic; official_benchmark=false")
SUBTASK_CONTRACT
elif [[ "$EVAL_PROFILE" == full4 ]]; then
    FULL4_TASKS=()
    while IFS= read -r task; do FULL4_TASKS+=("$task"); done \
        < <(grep -vE '^[[:space:]]*(#|$)' "$REPO_ROOT/scripts/b1k_tasks.txt" | awk '{print $1}')
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
if [[ "$SUBMIT_STORAGE_MODE" == oss ]]; then
    RECEIPT_ROOT="${RECEIPT_ROOT:-$REPO_ROOT/eval_results/.submission_receipts}"
    [[ "$RECEIPT_ROOT" != oss://* && "$RECEIPT_ROOT" != "$OSS_MOUNT_ROOT"/* ]] || {
        echo "In OSS mode RECEIPT_ROOT must be a local writable directory" >&2; exit 2; }
else
    RECEIPT_ROOT="${RECEIPT_ROOT:-$OUTPUT_BASE/.submission_receipts}"
fi
umask 077
mkdir -p "$RECEIPT_ROOT"
RECEIPT_PATH="$RECEIPT_ROOT/$RUN_ID.json"
CLI_OUTPUT_PATH="$RECEIPT_ROOT/$RUN_ID.stdout.log"
CLAIM_PATH="$RECEIPT_ROOT/$RUN_ID.claim.json"
[[ ! -e "$RECEIPT_PATH" && ! -e "$CLI_OUTPUT_PATH" && ! -e "$CLAIM_PATH" ]] || {
    echo "Submission receipt already exists for run: $RUN_ID" >&2
    exit 1
}

if [[ "$SUBMIT_STORAGE_MODE" == oss ]]; then
    python3 - "$CLAIM_PATH" "$RUN_ID" "$JOB_NAME" "$CHECKPOINT_SOURCE" <<'CLAIM_PY'
import json
import sys
from pathlib import Path
with Path(sys.argv[1]).open('x') as stream:
    json.dump({'run_id': sys.argv[2], 'job_name': sys.argv[3], 'checkpoint': sys.argv[4],
               'state': 'submission_reserved', 'resubmit_allowed': False}, stream)
    stream.write('\n')
CLAIM_PY
    "$SUBMIT_PYTHON" "$SCRIPT_DIR/submit_oss.py" reserve \
        --source "$CLAIM_PATH" --destination "$OSS_RECEIPT_PREFIX.claim.json"
fi
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
    ${BOOTSTRAP_ENV_ARGS[@]+"${BOOTSTRAP_ENV_ARGS[@]}"} \
    "--debug_mode" \
    "--ignore=__pycache__,*.pyc,*.pyo" 2>&1 | tee "$CLI_OUTPUT_PATH"
submit_status=${PIPESTATUS[0]}
set -e
python3 - "$CLI_OUTPUT_PATH" "$RECEIPT_PATH" "$RUN_ID" "$JOB_NAME" \
    "$EVAL_PROFILE" "$PERSISTENT_OUTPUT" "$EXTERNAL_SUBTASK_SELECTOR" "$PI05_MODEL_FORMAT" \
    "$PI05_ENV_ARCHIVE" "$CHECKPOINT_SOURCE" "$SUBTASK_PREDICTION" "$SUBTASK_INTERVAL" \
    "$SUBTASK_VOTE_WINDOW" "$SUBTASK_VOTE_THRESHOLD" "$EXECUTE_HORIZON" "$SELECTOR_DEBOUNCE_REFRESHES" \
    "$ALGO_NAME" "$NEBULA_PIP_NO_DEPS" "$EVAL_WORKERS" \
    "$BASE_PORT" "$CLUSTER_FILE" "$submit_status" "$READY_CACHE" <<'PY'
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
    subtask_interval,
    subtask_vote_window,
    subtask_vote_threshold,
    execute_horizon,
    selector_debounce_refreshes,
    algo_name,
    nebula_pip_no_deps,
    eval_workers,
    base_port,
    cluster_file,
    exit_code,
    ready_cache,
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
        for key in ("job_id", "jobId", "task_id", "taskId", "instance_id", "instanceId", "iid", "IID"):
            if value.get(key) is not None:
                identifiers.add(str(value[key]))
    for match in re.finditer(
        r"(?i)\b(?:iid|(?:job|task|instance)[_ -]?id)\s*[:=]\s*([A-Za-z0-9][A-Za-z0-9._:-]*)",
        line,
    ):
        identifiers.add(match.group(1))
ready_path = Path(ready_cache)
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
        "joint_subtask_prediction"
        if subtask_prediction == "1"
        else "whole_task"
    ),
    "subtask_interval": int(subtask_interval) if subtask_interval else None,
    "subtask_interval_source": "explicit" if subtask_interval else "policy_default",
    "action_horizon": int(execute_horizon),
    "subtask_voting": (
        {
            "mode": "temporal_majority",
            "action_horizon": int(execute_horizon),
            "interval": int(subtask_interval),
            "window": int(subtask_vote_window),
            "threshold": int(subtask_vote_threshold),
            "oracle": False,
            "privileged": False,
            "official_benchmark": False,
            "inference_ablation": True,
        }
        if subtask_vote_window
        else None
    ),
    "eval_workers": int(eval_workers),
    "base_port": int(base_port),
    "cluster_file": cluster_file,
    "persistent_output": persistent_output,
    "exit_code": int(exit_code),
    "job_identifiers": sorted(identifiers),
    "urls": sorted(urls),
    "stdout_stderr": output_path,
}
if profile == "subtasks100":
    panel = json.loads(Path("nebula/subtask_panel.json").read_text())
    receipt["evaluation_kind"] = "training_subtask_closed_loop"
    receipt["conditioning_mode"] = "joint_fixed_annotated_subtask"
    receipt["official_benchmark"] = False
    receipt["training_set_diagnostic"] = True
    receipt["checkpoint_step"] = ready.get("step")
    receipt["panel_fingerprint"] = panel["fingerprint"]
    receipt["expected_tasks"] = len(panel["selection"]["tasks"])
    receipt["expected_episode_jobs"] = len(panel["episodes"])
    receipt["expected_segment_rollouts"] = len(panel["samples"])
    receipt["expected_exclusions"] = len(panel["exclusions"])
    probe_path = Path("nebula/subtask_validation_plan.json")
    if probe_path.is_file():
        plan = json.loads(probe_path.read_text())
        selected = [s for s in panel["samples"] if s["sample_id"] in plan["sample_ids"]]
        receipt.update(evaluation_kind=plan["kind"], validation_only=True,
                       expected_tasks=len({s["task"] for s in selected}),
                       expected_episode_jobs=len({s["episode_index"] for s in selected}),
                       expected_segment_rollouts=len(selected), expected_exclusions=0,
                       validation_plan=plan)
    collection_path = Path("nebula/subtask_collection_plan.json")
    if collection_path.is_file():
        plan = json.loads(collection_path.read_text())
        identity = plan.get("checkpoint_identity") or {}
        if identity.get("checkpoint_step") != ready.get("step") or identity.get("checkpoint_manifest_sha256") != ready.get("manifest_sha256"):
            raise SystemExit("Collection checkpoint differs from baseline identity")
        selected_ids = {t["sample_id"] for t in plan["trials"]}
        selected = [s for s in panel["samples"] if s["sample_id"] in selected_ids]
        receipt.update(evaluation_kind="subtask_success_collection", collection_only=True,
                       expected_tasks=len({s["task"] for s in selected}),
                       expected_episode_jobs=len({s["episode_index"] for s in selected}),
                       expected_segment_rollouts=plan["expected_trials"], expected_exclusions=0,
                       collection_plan=plan)
elif subtask_vote_window:
    receipt["evaluation_kind"] = "temporal_subtask_voting"
elif selector:
    receipt["evaluation_kind"] = "selector_diagnostic"
    receipt["selector_diagnostics"] = {
        "selector": selector,
        "label_format": "legacy",
        "selector_labels_sha256": "fd785101426544b24ce756a84e72e2f5971776cfe6ebf7bde18e5ff7fb58ac04",
        "legacy_segments_sha256": "999c58103ad91046d3a1215e202ab2022c645b55524261b3eeed6d66e7f1322d",
        "training_annotation_sha256": "870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d",
        "requested_debounce_refreshes": (
            int(selector_debounce_refreshes) if selector_debounce_refreshes else None
        ),
        "effective_debounce_refreshes": int(selector_debounce_refreshes or 3),
        "stall_steps": 2048,
        "oracle": True,
        "privileged": True,
        "official_benchmark": False,
    }
Path(receipt_path).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
PY
if [[ "$SUBMIT_STORAGE_MODE" == oss ]]; then
    if ! "$SUBMIT_PYTHON" "$SCRIPT_DIR/submit_oss.py" sync \
        --receipt "$RECEIPT_PATH" --stdout "$CLI_OUTPUT_PATH" --prefix "$OSS_RECEIPT_PREFIX"; then
        echo "Receipt kept locally; OSS sync failed. Inspect the receipt and task status; do not resubmit." >&2
        if (( submit_status == 0 )); then submit_status=1; fi
    fi
fi
printf 'Submission receipt: %s\n' "$RECEIPT_PATH"
printf 'Submission output: %s\n' "$CLI_OUTPUT_PATH"
exit "$submit_status"
