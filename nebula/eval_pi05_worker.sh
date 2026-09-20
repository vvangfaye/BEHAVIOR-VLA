#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/eval_job.env"
EXTERNAL_SUBTASK_SELECTOR="${EXTERNAL_SUBTASK_SELECTOR:-}"
PI05_MODEL_FORMAT="${PI05_MODEL_FORMAT:-jax}"
PROVIDED_SUBTASK_AS_PROMPT="${PROVIDED_SUBTASK_AS_PROMPT:-0}"
SMOKE_TASK="${SMOKE_TASK:-turning_on_radio}"
SMOKE_EVAL_GPU="${SMOKE_EVAL_GPU:-1}"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_file() { [[ -s "$1" ]] || fail "required file missing or empty: $1"; }
require_dir() { [[ -d "$1" ]] || fail "required directory missing: $1"; }
check_free_gb() {
    local path="$1" need="$2" available
    available="$(df -Pk "$path" | tr -s ' ' | cut -d' ' -f4 | tail -1)"
    (( available >= need * 1024 * 1024 )) || fail "$path has less than ${need} GiB free"
}

case "$PI05_MODEL_FORMAT" in
    jax|torch) ;;
    *) fail "PI05_MODEL_FORMAT must be jax or torch" ;;
esac
case "$EVAL_PROFILE" in
    smoke|speed|two_tasks|reduced|full) ;;
    *) fail "EVAL_PROFILE must be smoke, speed, two_tasks, reduced, or full" ;;
esac
if [[ "$EVAL_PROFILE" == full && "$CONFIRM_FULL" != 1 ]]; then
    fail "full evaluation requires CONFIRM_FULL=1"
fi
[[ "$EVAL_WORKERS" =~ ^[1-9][0-9]*$ ]] || fail "EVAL_WORKERS must be a positive integer"
[[ "$BASE_PORT" =~ ^[1-9][0-9]*$ ]] || fail "BASE_PORT must be a positive integer"
[[ "$WRITE_VIDEO" == 0 || "$WRITE_VIDEO" == 1 ]] || fail "WRITE_VIDEO must be 0 or 1"
[[ "$SUBTASK_PREDICTION" == 0 || "$SUBTASK_PREDICTION" == 1 ]] || fail "SUBTASK_PREDICTION must be 0 or 1"
[[ "$PROVIDED_SUBTASK_AS_PROMPT" == 0 || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]] \
    || fail "PROVIDED_SUBTASK_AS_PROMPT must be 0 or 1"
[[ "$SMOKE_EVAL_GPU" =~ ^[0-9]+$ ]] || fail "SMOKE_EVAL_GPU must be a non-negative integer"
if [[ "$SUBTASK_PREDICTION" == 1 && "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    fail "SUBTASK_PREDICTION and PROVIDED_SUBTASK_AS_PROMPT are mutually exclusive"
fi
if [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    [[ "$PI05_MODEL_FORMAT" == jax ]] || fail "PROVIDED_SUBTASK_AS_PROMPT currently supports JAX checkpoints only"
    [[ "$SUBTASK_PREDICTION" == 0 ]] || fail "PROVIDED_SUBTASK_AS_PROMPT requires SUBTASK_PREDICTION=0"
    [[ "$EXTERNAL_SUBTASK_SELECTOR" == state_machine_v1 ]] \
        || fail "PROVIDED_SUBTASK_AS_PROMPT requires EXTERNAL_SUBTASK_SELECTOR=state_machine_v1"
fi
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    [[ "$EXTERNAL_SUBTASK_SELECTOR" == state_machine_v1 ]] \
        || fail "EXTERNAL_SUBTASK_SELECTOR must be state_machine_v1"
    [[ "$EVAL_PROFILE" == smoke || "$EVAL_PROFILE" == two_tasks ]] \
        || fail "EXTERNAL_SUBTASK_SELECTOR is supported only for EVAL_PROFILE=smoke or two_tasks"
    [[ "$SUBTASK_PREDICTION" == 1 || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]] \
        || fail "EXTERNAL_SUBTASK_SELECTOR requires a subtask conditioning mode"
    if [[ "$EVAL_PROFILE" == smoke ]]; then
        case "$SMOKE_TASK" in
            carrying_in_groceries|make_rose_centerpieces) ;;
            *) fail "selector smoke supports only carrying_in_groceries or make_rose_centerpieces" ;;
        esac
    fi
    [[ -z "${EXTERNAL_SUBTASK:-}" ]] \
        || fail "EXTERNAL_SUBTASK and EXTERNAL_SUBTASK_SELECTOR are mutually exclusive"
fi
if [[ ( "$EVAL_PROFILE" == speed || "$EVAL_PROFILE" == two_tasks ) && "$EVAL_WORKERS" != 4 ]]; then
    fail "$EVAL_PROFILE profile requires EVAL_WORKERS=4"
fi
if [[ "$EVAL_PROFILE" == two_tasks ]]; then
    [[ "$NUM_ROLLOUTS" == 3 ]] || fail "two_tasks profile requires NUM_ROLLOUTS=3"
    [[ "$WRITE_VIDEO" == 1 ]] || fail "two_tasks profile requires WRITE_VIDEO=1"
    [[ -z "$MAX_STEPS" ]] || fail "two_tasks profile requires a full task timeout"
fi
if [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    POLICY_MAX_TOKEN_LEN=200
else
    POLICY_MAX_TOKEN_LEN=512
fi

GLIBC_VERSION="$(ldd --version 2>&1 | sed -n '1{s/.* //;p;}')"
IFS=. read -r GLIBC_MAJOR GLIBC_MINOR _ <<< "$GLIBC_VERSION"
if [[ ! "$GLIBC_MAJOR" =~ ^[0-9]+$ || ! "$GLIBC_MINOR" =~ ^[0-9]+$ ]]; then
    fail "could not parse glibc version: $GLIBC_VERSION"
fi
if (( GLIBC_MAJOR < 2 || (GLIBC_MAJOR == 2 && GLIBC_MINOR < 34) )); then
    fail "Isaac Sim 5.1 requires glibc >= 2.34; worker image provides $GLIBC_VERSION"
fi
log "glibc $GLIBC_VERSION satisfies Isaac Sim requirement"

for path in "$EVAL_ENV_ARCHIVE" "$PI05_ENV_ARCHIVE" "$OPENPI_CACHE_ARCHIVE" \
    "$ASSET_ROOT/SHA256SUMS" "$ASSET_ROOT/behavior-1k-assets-3.9.0.zip" \
    "$ASSET_ROOT/omnigibson-robot-assets-3.8.2.zip" \
    "$ASSET_ROOT/2026-challenge-task-instances.zip" "$ASSET_ROOT/omnigibson.key" \
    "$CHECKPOINT_SOURCE/assets/$PI05_REPO_ID/norm_stats.json"; do
    require_file "$path"
done
if [[ "$PI05_MODEL_FORMAT" == torch ]]; then
    require_file "$CHECKPOINT_SOURCE/model.safetensors"
    require_file "$CHECKPOINT_SOURCE/metadata.json"
    [[ ! -e "$CHECKPOINT_SOURCE/params/_METADATA" ]] || fail "Torch checkpoint must not contain JAX params/_METADATA"
else
    require_file "$CHECKPOINT_SOURCE/params/_METADATA"
    [[ ! -e "$CHECKPOINT_SOURCE/model.safetensors" ]] || fail "JAX checkpoint must not contain model.safetensors"
fi
if [[ "$EVAL_PROFILE" == two_tasks || "$PI05_MODEL_FORMAT" == torch || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    require_file "$CHECKPOINT_SOURCE/_CHECKPOINT_METADATA"
    require_file "$CHECKPOINT_SOURCE/SHA256SUMS.inference"
    require_file "$CHECKPOINT_SOURCE/_READY"
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
expected = {
    "conditioning_mode": "subtask_only",
    "strict_subtask_action_chunks": True,
    "subtask_executor": False,
    "action_horizon": 32,
    "repo_id": sys.argv[2],
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

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader)
REQUIRED_GPUS=2
[[ "$EVAL_PROFILE" == speed || "$EVAL_PROFILE" == two_tasks ]] && REQUIRED_GPUS="$EVAL_WORKERS"
(( ${#GPU_ROWS[@]} >= REQUIRED_GPUS )) || fail "closed-loop evaluation requires at least $REQUIRED_GPUS GPUs"
printf '%s\n' "${GPU_ROWS[@]}"
for (( gpu = 0; gpu < REQUIRED_GPUS; gpu++ )); do
    [[ "${GPU_ROWS[gpu]}" == *L20* ]] || fail "GPU $gpu must be L20"
done
DRIVER="$(cut -d',' -f3 <<< "${GPU_ROWS[0]}" | tr -d ' ')"
case "$DRIVER" in
    535.*) fail "NVIDIA driver $DRIVER is incompatible with the archived RTX shader cache" ;;
    580.*) ;;
    *) log "WARNING: driver $DRIVER is unverified; smoke evaluation will validate RTX startup" ;;
esac
check_free_gb /dev/shm 40
check_free_gb /var/tmp 120

log "run id: $RUN_ID"
log "profile: $EVAL_PROFILE"
log "model format: $PI05_MODEL_FORMAT"
log "provided subtask as prompt: $PROVIDED_SUBTASK_AS_PROMPT"
log "checkpoint: $CHECKPOINT_SOURCE"
log "policy environment: $PI05_ENV_ARCHIVE"
log "persistent output: $PERSISTENT_OUTPUT"
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    log "selector diagnostic: selector=$EXTERNAL_SUBTASK_SELECTOR label_format=legacy oracle=true privileged=true official_benchmark=false"
fi
if [[ "$NEBULA_WORKER_DRY_RUN" == 1 ]]; then
    log "worker dry-run passed; no runtime directories were created"
    exit 0
fi

RUNTIME_ROOT="/var/tmp/wangfaye/aihub-b1k-eval/$RUN_ID"
ENV_ROOT="/dev/shm/wangfaye/aihub-b1k-eval/$RUN_ID"
PI05_ENV="$ENV_ROOT/b1k-pi05"
EVAL_ENV="$ENV_ROOT/b1k-eval"
OPENPI_DATA_HOME="$RUNTIME_ROOT/openpi-cache"
OMNIGIBSON_DATA_PATH="$RUNTIME_ROOT/omnigibson-data"
LOCAL_CHECKPOINT="$RUNTIME_ROOT/checkpoint"
LOCAL_RESULTS="$RUNTIME_ROOT/results"
LOCAL_ASSETS="$RUNTIME_ROOT/asset-archives"

[[ ! -e "$RUNTIME_ROOT" && ! -e "$ENV_ROOT" ]] || fail "run-specific local path already exists for $RUN_ID"
[[ ! -e "$PERSISTENT_OUTPUT" ]] || fail "persistent output already exists: $PERSISTENT_OUTPUT"
mkdir -p "$RUNTIME_ROOT" "$ENV_ROOT" "$PERSISTENT_OUTPUT"
exec > >(tee -a "$PERSISTENT_OUTPUT/staging.log") 2>&1

EVAL_PIDS=()
stop_eval_processes() {
    local pid second active
    for pid in "${EVAL_PIDS[@]}"; do
        kill -TERM -- "-$pid" 2>/dev/null || true
    done
    for (( second = 0; second < 20; second++ )); do
        active=0
        for pid in "${EVAL_PIDS[@]}"; do
            kill -0 -- "-$pid" 2>/dev/null && active=1
        done
        (( active == 0 )) && break
        sleep 1
    done
    for pid in "${EVAL_PIDS[@]}"; do
        kill -0 -- "-$pid" 2>/dev/null && kill -KILL -- "-$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    EVAL_PIDS=()
}
cleanup() {
    local status=$? preserve_ok=1
    set +e
    stop_eval_processes
    rm -f "$PERSISTENT_OUTPUT/_RUNNING" || preserve_ok=0
    if (( status != 0 )); then
        if [[ -d "$LOCAL_RESULTS" ]]; then
            cp -a "$LOCAL_RESULTS/." "$PERSISTENT_OUTPUT/" 2>/dev/null || preserve_ok=0
        fi
        printf 'exit_code=%s\n' "$status" > "$PERSISTENT_OUTPUT/_FAILED" || preserve_ok=0
    fi
    if [[ "$KEEP_RUNTIME" != 1 && ( $status -eq 0 || $preserve_ok -eq 1 ) ]]; then
        rm -rf -- "$RUNTIME_ROOT" "$ENV_ROOT"
    else
        log "kept runtime: $RUNTIME_ROOT $ENV_ROOT"
    fi
}
handle_signal() {
    local status="$1"
    stop_eval_processes
    exit "$status"
}
trap cleanup EXIT
trap 'handle_signal 143' TERM
trap 'handle_signal 130' INT

restore_env() {
    local archive="$1" target="$2"
    mkdir -p "$target"
    tar -xzf "$archive" -C "$target"
    require_file "$target/bin/python"
    if [[ -x "$target/bin/conda-unpack" ]]; then
        "$target/bin/conda-unpack"
    fi
}

log "restoring policy and simulator environments"
restore_env "$PI05_ENV_ARCHIVE" "$PI05_ENV"
restore_env "$EVAL_ENV_ARCHIVE" "$EVAL_ENV"

log "binding bundled source trees"
PI05_SITE="$(env -u PYTHONPATH "$PI05_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
EVAL_SITE="$(env -u PYTHONPATH "$EVAL_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
printf '%s\n%s\n' "$REPO_ROOT/openpi/src" "$REPO_ROOT/openpi/packages/openpi-client/src" \
    > "$PI05_SITE/00-aihub-b1k-eval-source.pth"
printf '%s\n%s\n%s\n' "$REPO_ROOT/BEHAVIOR-1K/OmniGibson" "$REPO_ROOT/BEHAVIOR-1K/bddl3" \
    "$REPO_ROOT/BEHAVIOR-1K/joylo" > "$EVAL_SITE/00-aihub-b1k-eval-source.pth"
if [[ "$PI05_MODEL_FORMAT" == torch ]]; then
    ln -s "$PI05_ENV" "$REPO_ROOT/openpi/.venv-torch"
else
    ln -s "$PI05_ENV" "$REPO_ROOT/openpi/.venv"
fi

log "restoring OpenPI cache"
mkdir -p "$OPENPI_DATA_HOME"
tar -xf "$OPENPI_CACHE_ARCHIVE" -C "$OPENPI_DATA_HOME"
require_file "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
require_file "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params/_METADATA"

log "copying inference checkpoint from OSS"
mkdir -p "$LOCAL_CHECKPOINT"
if [[ "$PI05_MODEL_FORMAT" == torch ]]; then
    cp "$CHECKPOINT_SOURCE/model.safetensors" "$CHECKPOINT_SOURCE/metadata.json" "$LOCAL_CHECKPOINT/"
    cp -a "$CHECKPOINT_SOURCE/assets" "$LOCAL_CHECKPOINT/"
    require_file "$LOCAL_CHECKPOINT/model.safetensors"
    require_file "$LOCAL_CHECKPOINT/metadata.json"
else
    cp -a "$CHECKPOINT_SOURCE/params" "$CHECKPOINT_SOURCE/assets" "$LOCAL_CHECKPOINT/"
    require_file "$LOCAL_CHECKPOINT/params/_METADATA"
fi
if [[ -f "$CHECKPOINT_SOURCE/_CHECKPOINT_METADATA" ]]; then
    cp "$CHECKPOINT_SOURCE/_CHECKPOINT_METADATA" "$LOCAL_CHECKPOINT/"
fi
require_file "$LOCAL_CHECKPOINT/assets/$PI05_REPO_ID/norm_stats.json"
if [[ "$EVAL_PROFILE" == two_tasks || "$PI05_MODEL_FORMAT" == torch || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
    cp "$CHECKPOINT_SOURCE/SHA256SUMS.inference" "$LOCAL_CHECKPOINT/"
    (
        cd "$LOCAL_CHECKPOINT"
        sha256sum -c SHA256SUMS.inference
    )
fi

log "copying and extracting simulator assets"
mkdir -p "$LOCAL_ASSETS" "$OMNIGIBSON_DATA_PATH"
cp "$ASSET_ROOT/behavior-1k-assets-3.9.0.zip" \
    "$ASSET_ROOT/omnigibson-robot-assets-3.8.2.zip" \
    "$ASSET_ROOT/2026-challenge-task-instances.zip" \
    "$ASSET_ROOT/omnigibson.key" "$LOCAL_ASSETS/"
cp "$ASSET_ROOT/SHA256SUMS" "$LOCAL_ASSETS/"
(
    cd "$LOCAL_ASSETS"
    sha256sum -c SHA256SUMS
)
mkdir -p "$OMNIGIBSON_DATA_PATH/behavior-1k-assets" \
    "$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets" \
    "$OMNIGIBSON_DATA_PATH/2026-challenge-task-instances"
unzip -q "$LOCAL_ASSETS/behavior-1k-assets-3.9.0.zip" -d "$OMNIGIBSON_DATA_PATH/behavior-1k-assets"
unzip -q "$LOCAL_ASSETS/omnigibson-robot-assets-3.8.2.zip" -d "$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets"
unzip -q "$LOCAL_ASSETS/2026-challenge-task-instances.zip" -d "$OMNIGIBSON_DATA_PATH/2026-challenge-task-instances"
cp "$LOCAL_ASSETS/omnigibson.key" "$OMNIGIBSON_DATA_PATH/omnigibson.key"
require_file "$OMNIGIBSON_DATA_PATH/behavior-1k-assets/VERSION"
require_file "$OMNIGIBSON_DATA_PATH/omnigibson-robot-assets/VERSION"
require_file "$OMNIGIBSON_DATA_PATH/2026-challenge-task-instances/metadata/task.jsonl"

log "writing Vulkan and EGL manifests"
DRIVER_LIB=""
for candidate in "/usr/local/nvidia/lib64/libEGL_nvidia.so.$DRIVER" \
    "/usr/lib64/libEGL_nvidia.so.$DRIVER" "/lib64/libEGL_nvidia.so.$DRIVER"; do
    if [[ -f "$candidate" ]]; then DRIVER_LIB="$candidate"; break; fi
done
[[ -n "$DRIVER_LIB" ]] || fail "libEGL_nvidia for driver $DRIVER was not found"
mkdir -p "$EVAL_ENV/share/vulkan/icd.d" "$EVAL_ENV/share/glvnd/egl_vendor.d"
printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "%s", "api_version": "1.3.242"}\n}\n' \
    "$DRIVER_LIB" > "$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json"
printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {"library_path": "%s"}\n}\n' \
    "$DRIVER_LIB" > "$EVAL_ENV/share/glvnd/egl_vendor.d/10_nvidia.json"

log "verifying policy and simulator runtimes"
if [[ "$PI05_MODEL_FORMAT" == torch ]]; then
    JAX_PLATFORMS=cpu "$PI05_ENV/bin/python" "$REPO_ROOT/openpi/scripts/install_transformers_patch.py" --check
    CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cpu OPENPI_DATA_HOME="$OPENPI_DATA_HOME" \
        "$PI05_ENV/bin/python" -c 'import openpi, openpi_client, torch, transformers; assert torch.cuda.is_available(); assert transformers.__version__ == "5.5.4", transformers.__version__; print(openpi.__file__, torch.__version__, torch.cuda.get_device_name())'
else
    CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME="$OPENPI_DATA_HOME" XLA_PYTHON_CLIENT_PREALLOCATE=false \
        "$PI05_ENV/bin/python" -c 'import jax, openpi, openpi_client; devices = jax.devices(); assert len(devices) == 1 and devices[0].platform == "gpu", devices; print(openpi.__file__, devices)'
fi
IMPORT_CHECK_RUNTIME="$RUNTIME_ROOT/omnigibson-import-check"
IMPORT_CHECK_GPU=1
if [[ "$EVAL_PROFILE" == smoke && ( "$PI05_MODEL_FORMAT" == torch || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ) ]]; then
    IMPORT_CHECK_GPU="$SMOKE_EVAL_GPU"
fi
mkdir -p "$IMPORT_CHECK_RUNTIME"
env CUDA_VISIBLE_DEVICES="$IMPORT_CHECK_GPU" PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH" \
    OMNIGIBSON_APPDATA_PATH="$IMPORT_CHECK_RUNTIME" \
    OMNIGIBSON_GPU_ID="$IMPORT_CHECK_GPU" OMNIGIBSON_HEADLESS=1 OMNIGIBSON_DISABLE_TORCH_COMPILE=1 \
    LD_LIBRARY_PATH="$EVAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    VK_ICD_FILENAMES="$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json" \
    __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_ENV/share/glvnd/egl_vendor.d/10_nvidia.json" \
    "$EVAL_ENV/bin/python" -c 'import bddl, isaacsim, omnigibson, torch; assert torch.cuda.is_available(); print(omnigibson.__file__, torch.cuda.get_device_name())'
LD_LIBRARY_PATH="$EVAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
VK_ICD_FILENAMES="$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json" \
__EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_ENV/share/glvnd/egl_vendor.d/10_nvidia.json" \
    "$EVAL_ENV/bin/vulkaninfo" --summary | grep -E 'Vulkan Instance Version|deviceName|driverInfo'

case "$EVAL_PROFILE" in
    smoke)
        TASK_FILE=""
        TASK_SPECS=("$SMOKE_TASK|0")
        MAX_STEPS_VALUE="${MAX_STEPS:-100}"
        ;;
    speed)
        TASK_FILE=""
        TASK_SPECS=()
        read -ra speed_indices <<< "$INSTANCE_INDICES"
        (( ${#speed_indices[@]} == EVAL_WORKERS )) || fail "speed profile requires one instance index per worker"
        for index in "${speed_indices[@]}"; do
            TASK_SPECS+=("$SPEED_TASK|$index")
        done
        MAX_STEPS_VALUE="${MAX_STEPS:-100}"
        ;;
    two_tasks)
        TASK_FILE=""
        TASK_SPECS=(
            "carrying_in_groceries make_rose_centerpieces|0 4 8 12 16"
            "carrying_in_groceries make_rose_centerpieces|1 5 9 13 17"
            "carrying_in_groceries make_rose_centerpieces|2 6 10 14 18"
            "carrying_in_groceries make_rose_centerpieces|3 7 11 15 19"
        )
        MAX_STEPS_VALUE=""
        ;;
    reduced)
        TASK_FILE="$REPO_ROOT/scripts/b1k_tasks_reduced.txt"
        TASK_SPECS=()
        MAX_STEPS_VALUE="${MAX_STEPS:-}"
        ;;
    full)
        TASK_FILE="$REPO_ROOT/scripts/b1k_tasks.txt"
        TASK_SPECS=()
        MAX_STEPS_VALUE="${MAX_STEPS:-}"
        ;;
esac

if [[ -n "$TASK_FILE" ]]; then
    require_file "$TASK_FILE"
    while read -r task ninst _weight _rest; do
        [[ -n "$task" && "$task" != \#* ]] || continue
        if [[ -n "${ninst:-}" && "$ninst" =~ ^[1-9][0-9]*$ ]]; then
            indices="$(seq 0 $((ninst - 1)) | tr '\n' ' ')"
        else
            indices="$INSTANCE_INDICES"
        fi
        TASK_SPECS+=("$task|$indices")
    done < "$TASK_FILE"
fi
(( ${#TASK_SPECS[@]} )) || fail "evaluation profile selected no tasks"

EXPECTED_ROLLOUTS=0
for spec in "${TASK_SPECS[@]}"; do
    tasks="${spec%%|*}"
    indices="${spec#*|}"
    read -ra task_array <<< "$tasks"
    read -ra index_array <<< "$indices"
    EXPECTED_ROLLOUTS=$((EXPECTED_ROLLOUTS + ${#task_array[@]} * ${#index_array[@]} * NUM_ROLLOUTS))
done
if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" && "$EVAL_PROFILE" == two_tasks && "$EXPECTED_ROLLOUTS" != 120 ]]; then
    fail "two-task selector diagnostic requires exactly 120 rollouts, got $EXPECTED_ROLLOUTS"
fi

mkdir -p "$LOCAL_RESULTS" "$PERSISTENT_OUTPUT/status"
"$PI05_ENV/bin/python" - "$PERSISTENT_OUTPUT/manifest.json" <<PY
import json, pathlib
profile = "$EVAL_PROFILE"
selector = "$EXTERNAL_SUBTASK_SELECTOR" or None
indices = [int(value) for value in "$INSTANCE_INDICES".split()]
expected_ids = [int(value) for value in "$SPEED_EXPECTED_INSTANCE_IDS".split()]
if profile == "speed":
    workers = [
        {
            "worker": worker,
            "server_gpu": worker,
            "eval_gpu": worker,
            "port": $BASE_PORT + worker,
            "tasks": ["$SPEED_TASK"],
            "instance_indices": [indices[worker]],
            "expected_instance_ids": [expected_ids[worker]],
            "expected_rollouts": 1,
        }
        for worker in range($EVAL_WORKERS)
    ]
elif profile == "two_tasks":
    shards = ([0, 4, 8, 12, 16], [1, 5, 9, 13, 17], [2, 6, 10, 14, 18], [3, 7, 11, 15, 19])
    workers = [
        {
            "worker": worker,
            "server_gpu": worker,
            "eval_gpu": worker,
            "port": $BASE_PORT + worker,
            "tasks": ["carrying_in_groceries", "make_rose_centerpieces"],
            "instance_indices": list(shards[worker]),
            "expected_instance_ids": [301 + index for index in shards[worker]],
            "expected_rollouts": 30,
        }
        for worker in range($EVAL_WORKERS)
    ]
elif profile == "smoke":
    workers = [{
        "worker": 0,
        "server_gpu": 0,
        "eval_gpu": $SMOKE_EVAL_GPU,
        "port": $BASE_PORT,
        "tasks": ["$SMOKE_TASK"],
        "instance_indices": [0],
        "expected_instance_ids": [301],
        "expected_rollouts": $NUM_ROLLOUTS,
    }]
else:
    workers = [{"worker": 0, "server_gpu": 0, "eval_gpu": 1, "port": $BASE_PORT}]
ready_path = pathlib.Path("$CHECKPOINT_SOURCE/_READY")
ready = json.loads(ready_path.read_text()) if ready_path.is_file() else {}
path = pathlib.Path("$PERSISTENT_OUTPUT/manifest.json")
path.write_text(json.dumps({
    "run_id": "$RUN_ID",
    "profile": profile,
    "model_format": "$PI05_MODEL_FORMAT",
    "mode": "public_test",
    "prompt_mode": "whole_task",
    "checkpoint": "$CHECKPOINT_SOURCE",
    "checkpoint_manifest_sha256": ready.get("manifest_sha256"),
    "policy_env_archive": "$PI05_ENV_ARCHIVE",
    "repo_id": "$PI05_REPO_ID",
    "conditioning_mode": (
        "provided_subtask_as_prompt"
        if bool($PROVIDED_SUBTASK_AS_PROMPT)
        else "joint_subtask_prediction"
        if bool($SUBTASK_PREDICTION)
        else "whole_task"
    ),
    "subtask_prediction": bool($SUBTASK_PREDICTION),
    "provided_subtask_as_prompt": bool($PROVIDED_SUBTASK_AS_PROMPT),
    "max_token_len": (200 if bool($PROVIDED_SUBTASK_AS_PROMPT) else 512 if bool($SUBTASK_PREDICTION) else None),
    "subtask_interval": (1 if bool($SUBTASK_PREDICTION) else None),
    "subtask_temperature": (0.0 if bool($SUBTASK_PREDICTION) else None),
    "external_subtask_selector": selector,
    "selector_diagnostics": ({
        "selector": selector,
        "label_format": "legacy",
        "selector_labels_sha256": "fd785101426544b24ce756a84e72e2f5971776cfe6ebf7bde18e5ff7fb58ac04",
        "legacy_segments_sha256": "999c58103ad91046d3a1215e202ab2022c645b55524261b3eeed6d66e7f1322d",
        "training_annotation_sha256": "870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d",
        "oracle": True,
        "privileged": True,
        "official_benchmark": False,
    } if selector else None),
    "tasks": (["carrying_in_groceries", "make_rose_centerpieces"] if profile == "two_tasks" else None),
    "instance_indices": indices,
    "num_rollouts": $NUM_ROLLOUTS,
    "max_steps": (int("${MAX_STEPS_VALUE}") if "${MAX_STEPS_VALUE}" else None),
    "expected_rollouts": $EXPECTED_ROLLOUTS,
    "expected_videos": ($EXPECTED_ROLLOUTS if bool($WRITE_VIDEO) else 0),
    "require_subtask_events": bool($SUBTASK_PREDICTION) or bool($PROVIDED_SUBTASK_AS_PROMPT),
    "require_selector_events": bool(selector),
    "driver": "$DRIVER",
    "pi05_mem_fraction": $PI05_MEM_FRACTION,
    "write_video": bool($WRITE_VIDEO),
    "workers": workers,
    "code_commit": "$CODE_COMMIT",
    "code_dirty": "$CODE_DIRTY",
}, indent=2, sort_keys=True) + "\n")
PY
touch "$PERSISTENT_OUTPUT/_RUNNING"

status=0
EVAL_WALL_TIME=""
if [[ "$EVAL_PROFILE" == speed || "$EVAL_PROFILE" == two_tasks ]]; then
    PARALLEL_START="$(date +%s)"
    WORKER_PIDS=()
    WORKER_TASKS=()
    WORKER_INDICES=()
    read -ra speed_expected_ids <<< "$SPEED_EXPECTED_INSTANCE_IDS"
    for worker in "${!TASK_SPECS[@]}"; do
        spec="${TASK_SPECS[worker]}"
        tasks="${spec%%|*}"
        indices="${spec#*|}"
        read -ra task_array <<< "$tasks"
        port="$((BASE_PORT + worker))"
        shard_root="$LOCAL_RESULTS/shards/worker_$worker"
        worker_log="$shard_root/worker.log"
        runtime_root="$RUNTIME_ROOT/omnigibson-runtime/worker_$worker"
        mkdir -p "$shard_root" "$runtime_root"
        log "launching $EVAL_PROFILE worker=$worker tasks=[$tasks] indices=[$indices] gpu=$worker port=$port"
        setsid env MODEL=pi05 PI05_CKPT="$LOCAL_CHECKPOINT" PI05_REPO_ID="$PI05_REPO_ID" \
            PI05_ENV="$PI05_ENV" SUBTASK_PREDICTION="$SUBTASK_PREDICTION" \
            PROVIDED_SUBTASK_AS_PROMPT="$PROVIDED_SUBTASK_AS_PROMPT" \
            EXTERNAL_SUBTASK_SELECTOR="$EXTERNAL_SUBTASK_SELECTOR" \
            MAX_TOKEN_LEN="$POLICY_MAX_TOKEN_LEN" MAX_SUBTASK_LEN=64 SUBTASK_INTERVAL=1 SUBTASK_TEMPERATURE=0 \
            PROMPT_MODE=whole_task MODE=public_test \
            SERVER_GPU="$worker" EVAL_GPU="$worker" PORT="$port" EVAL_ENV="$EVAL_ENV" \
            EVAL_RUNTIME_ROOT="$runtime_root" PI05_MEM_FRACTION="$PI05_MEM_FRACTION" \
            OPENPI_DATA_HOME="$OPENPI_DATA_HOME" OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH" \
            OMNIGIBSON_DISABLE_TORCH_COMPILE=1 WRITE_VIDEO="$WRITE_VIDEO" \
            OUTPUT_ROOT="$shard_root" INSTANCE_INDICES="$indices" NUM_ROLLOUTS="$NUM_ROLLOUTS" \
            MAX_STEPS="$MAX_STEPS_VALUE" bash "$REPO_ROOT/scripts/eval_b1k.sh" "${task_array[@]}" \
            > "$worker_log" 2>&1 &
        WORKER_PIDS+=("$!")
        WORKER_TASKS+=("$tasks")
        WORKER_INDICES+=("$indices")
        EVAL_PIDS+=("$!")
    done

    set +e
    for worker in "${!WORKER_PIDS[@]}"; do
        wait "${WORKER_PIDS[worker]}"
        worker_status=$?
        if [[ "$EVAL_PROFILE" == speed ]]; then
            printf '{"worker":%d,"task":"%s","instance_index":%d,"expected_instance_id":%d,"gpu":%d,"port":%d,"exit_code":%d}\n' \
                "$worker" "$SPEED_TASK" "${speed_indices[worker]}" "${speed_expected_ids[worker]}" \
                "$worker" "$((BASE_PORT + worker))" "$worker_status" \
                > "$PERSISTENT_OUTPUT/status/worker-$worker.json"
        else
            "$PI05_ENV/bin/python" - "$PERSISTENT_OUTPUT/status/worker-$worker.json" \
                "$LOCAL_RESULTS/shards/worker_$worker" "$worker" "${WORKER_TASKS[worker]}" \
                "${WORKER_INDICES[worker]}" "$((BASE_PORT + worker))" "$worker_status" <<'PY'
import json
from pathlib import Path
import sys

output, shard_root, worker, tasks, indices, port, exit_code = sys.argv[1:]
root = Path(shard_root)
results = []
for path in root.rglob("json/*.json"):
    try:
        results.append(json.loads(path.read_text()))
    except Exception:
        pass
status = {
    "worker": int(worker),
    "tasks": tasks.split(),
    "instance_indices": [int(value) for value in indices.split()],
    "gpu": int(worker),
    "port": int(port),
    "expected_rollouts": 30,
    "completed_rollouts": len(results),
    "completed_videos": sum(1 for path in root.rglob("videos/*.mp4") if path.stat().st_size > 0),
    "rollouts_with_subtask_events": sum(bool(result.get("subtask_events")) for result in results),
    "rollouts_with_selector_events": sum(
        bool(result.get("selector_events")) and bool(result.get("selector_summary")) for result in results
    ),
    "exit_code": int(exit_code),
}
Path(output).write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
PY
        fi
        (( worker_status == 0 )) || status=1
    done
    set -e
    EVAL_PIDS=()
    EVAL_WALL_TIME=$(( $(date +%s) - PARALLEL_START ))
    (( EVAL_WALL_TIME > 0 )) || EVAL_WALL_TIME=1
    printf '{"workers":%d,"wall_time_seconds":%d}\n' "$EVAL_WORKERS" "$EVAL_WALL_TIME" \
        > "$LOCAL_RESULTS/parallel_metrics.json"
    cp -a "$LOCAL_RESULTS/." "$PERSISTENT_OUTPUT/"
else
    SERIAL_EVAL_GPU=1
    [[ "$EVAL_PROFILE" == smoke ]] && SERIAL_EVAL_GPU="$SMOKE_EVAL_GPU"
    for spec in "${TASK_SPECS[@]}"; do
        task="${spec%%|*}"
        indices="${spec#*|}"
        log "evaluating $task on public indices [$indices]"
        set +e
        setsid env MODEL=pi05 PI05_CKPT="$LOCAL_CHECKPOINT" PI05_REPO_ID="$PI05_REPO_ID" \
            PI05_ENV="$PI05_ENV" SUBTASK_PREDICTION="$SUBTASK_PREDICTION" \
            PROVIDED_SUBTASK_AS_PROMPT="$PROVIDED_SUBTASK_AS_PROMPT" \
            EXTERNAL_SUBTASK_SELECTOR="$EXTERNAL_SUBTASK_SELECTOR" \
            MAX_TOKEN_LEN="$POLICY_MAX_TOKEN_LEN" MAX_SUBTASK_LEN=64 SUBTASK_INTERVAL=1 SUBTASK_TEMPERATURE=0 \
            PROMPT_MODE=whole_task MODE=public_test \
            SERVER_GPU=0 EVAL_GPU="$SERIAL_EVAL_GPU" PORT="$BASE_PORT" EVAL_ENV="$EVAL_ENV" \
            EVAL_RUNTIME_ROOT="$RUNTIME_ROOT/omnigibson-runtime" PI05_MEM_FRACTION="$PI05_MEM_FRACTION" \
            OPENPI_DATA_HOME="$OPENPI_DATA_HOME" OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH" \
            OMNIGIBSON_DISABLE_TORCH_COMPILE=1 WRITE_VIDEO="$WRITE_VIDEO" \
            OUTPUT_ROOT="$LOCAL_RESULTS" INSTANCE_INDICES="$indices" NUM_ROLLOUTS="$NUM_ROLLOUTS" \
            MAX_STEPS="$MAX_STEPS_VALUE" bash "$REPO_ROOT/scripts/eval_b1k.sh" "$task" &
        EVAL_PIDS=("$!")
        wait "${EVAL_PIDS[0]}"
        task_status=$?
        EVAL_PIDS=()
        set -e
        cp -a "$LOCAL_RESULTS/." "$PERSISTENT_OUTPUT/"
        printf '{"task":"%s","exit_code":%d}\n' "$task" "$task_status" > "$PERSISTENT_OUTPUT/status/$task.json"
        (( task_status == 0 )) || status=1
    done
fi

if [[ "$EVAL_PROFILE" == speed ]]; then
    if ! "$PI05_ENV/bin/python" - "$LOCAL_RESULTS" "$SPEED_TASK" "$SPEED_EXPECTED_INSTANCE_IDS" "$EVAL_WORKERS" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
task = sys.argv[2]
expected_ids = [int(value) for value in sys.argv[3].split()]
workers = int(sys.argv[4])
observed = []
for worker in range(workers):
    paths = list((root / "shards" / f"worker_{worker}" / "pi05" / task / "json").glob("*.json"))
    if len(paths) != 1:
        raise RuntimeError(f"worker {worker} produced {len(paths)} rollout JSON files")
    result = json.loads(paths[0].read_text())
    key = (result["task"], int(result["instance_id"]), int(result["rollout_id"]))
    expected = (task, expected_ids[worker], 0)
    if key != expected:
        raise RuntimeError(f"worker {worker} produced {key}, expected {expected}")
    observed.append(key)
if len(set(observed)) != workers:
    raise RuntimeError(f"speed workers produced duplicate rollouts: {observed}")
print(json.dumps({"validated_rollouts": [list(key) for key in observed]}, sort_keys=True))
PY
    then
        status=1
    fi
fi
if [[ "$SUBTASK_PREDICTION" == 1 && -z "$EXTERNAL_SUBTASK_SELECTOR" \
    && ( "$EVAL_PROFILE" == speed || "$EVAL_PROFILE" == two_tasks ) ]]; then
    for (( worker = 0; worker < EVAL_WORKERS; worker++ )); do
        server_log="$LOCAL_RESULTS/shards/worker_$worker/pi05/server_$((BASE_PORT + worker)).log"
        if ! grep -q 'Generated subtask:' "$server_log"; then
            log "$EVAL_PROFILE worker $worker server log contains no generated subtask"
            status=1
        fi
    done
fi

SUMMARY_ARGS=(
    --results-root "$LOCAL_RESULTS"
    --output "$LOCAL_RESULTS/summary.json"
    --run-id "$RUN_ID"
    --profile "$EVAL_PROFILE"
    --checkpoint "$CHECKPOINT_SOURCE"
    --expected-rollouts "$EXPECTED_ROLLOUTS"
    --require-complete
)
if [[ -n "$EVAL_WALL_TIME" ]]; then
    SUMMARY_ARGS+=(--wall-time-seconds "$EVAL_WALL_TIME")
fi
if [[ "$EVAL_PROFILE" == speed ]]; then
    SUMMARY_ARGS+=(
        --expected-task "$SPEED_TASK"
        --expected-instance-ids "$SPEED_EXPECTED_INSTANCE_IDS"
        --expected-workers "$EVAL_WORKERS"
        --status-root "$PERSISTENT_OUTPUT/status"
    )
elif [[ "$EVAL_PROFILE" == two_tasks ]]; then
    SUMMARY_ARGS+=(
        --expected-tasks "carrying_in_groceries make_rose_centerpieces"
        --expected-instance-ids "301 302 303 304 305 306 307 308 309 310 311 312 313 314 315 316 317 318 319 320"
        --expected-rollout-ids "0 1 2"
        --expected-workers "$EVAL_WORKERS"
        --status-root "$PERSISTENT_OUTPUT/status"
        --require-videos
    )
    if [[ "$SUBTASK_PREDICTION" == 1 || "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
        SUMMARY_ARGS+=(--require-subtask-events)
    fi
    if [[ -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
        SUMMARY_ARGS+=(--require-selector-events)
    elif [[ "$SUBTASK_PREDICTION" == 1 ]]; then
        SUMMARY_ARGS+=(--require-generated-subtasks)
    fi
elif [[ "$EVAL_PROFILE" == smoke && -n "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
    SUMMARY_ARGS+=(--require-subtask-events --require-selector-events)
elif [[ "$EVAL_PROFILE" == smoke && "$SUBTASK_PREDICTION" == 1 ]]; then
    SUMMARY_ARGS+=(--require-subtask-events --require-generated-subtasks)
fi
set +e
"$PI05_ENV/bin/python" "$SCRIPT_DIR/summarize_pi05_eval.py" "${SUMMARY_ARGS[@]}"
summary_status=$?
set -e
cp "$LOCAL_RESULTS/summary.json" "$PERSISTENT_OUTPUT/summary.json"
(( summary_status == 0 )) || status=1

if [[ "$EVAL_PROFILE" == smoke ]]; then
    [[ -n "$(find "$LOCAL_RESULTS/pi05" -path '*/json/*.json' -type f -size +0c -print -quit)" ]] \
        || fail "smoke produced no nonempty rollout JSON"
    [[ -n "$(find "$LOCAL_RESULTS/pi05" -path '*/videos/*.mp4' -type f -size +0c -print -quit)" ]] \
        || fail "smoke produced no nonempty video"
    if [[ "$SUBTASK_PREDICTION" == 1 && -z "$EXTERNAL_SUBTASK_SELECTOR" ]]; then
        grep -q 'Generated subtask:' "$LOCAL_RESULTS/pi05/server_$BASE_PORT.log" \
            || fail "smoke server log contains no generated subtask"
    elif [[ "$PROVIDED_SUBTASK_AS_PROMPT" == 1 ]]; then
        server_log="$LOCAL_RESULTS/pi05/server_$BASE_PORT.log"
        grep -q 'Loaded policy class=Policy conditioning=provided_subtask_as_prompt max_token_len=200' "$server_log" \
            || fail "smoke server did not load the strict ordinary Policy"
        if grep -q 'Generated subtask:' "$server_log"; then
            fail "strict provided-subtask smoke unexpectedly generated a subtask"
        fi
    fi
fi

if (( status == 0 )); then
    touch "$PERSISTENT_OUTPUT/_SUCCESS"
    rm -f "$PERSISTENT_OUTPUT/_FAILED"
    log "evaluation complete: $PERSISTENT_OUTPUT"
else
    fail "one or more evaluation tasks failed"
fi
