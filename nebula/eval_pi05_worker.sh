#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/eval_job.env"

log() { printf '[%(%F %T)T] %s\n' -1 "$*"; }
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
require_file() { [[ -s "$1" ]] || fail "required file missing or empty: $1"; }
require_dir() { [[ -d "$1" ]] || fail "required directory missing: $1"; }
check_free_gb() {
    local path="$1" need="$2" available
    available="$(df -Pk "$path" | tr -s ' ' | cut -d' ' -f4 | tail -1)"
    (( available >= need * 1024 * 1024 )) || fail "$path has less than ${need} GiB free"
}

case "$EVAL_PROFILE" in
    smoke|reduced|full) ;;
    *) fail "EVAL_PROFILE must be smoke, reduced, or full" ;;
esac
if [[ "$EVAL_PROFILE" == full && "$CONFIRM_FULL" != 1 ]]; then
    fail "full evaluation requires CONFIRM_FULL=1"
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
    "$CHECKPOINT_SOURCE/params/_METADATA" \
    "$CHECKPOINT_SOURCE/assets/$PI05_REPO_ID/norm_stats.json"; do
    require_file "$path"
done
(
    cd "$ASSET_ROOT"
    sha256sum -c SHA256SUMS
)

mapfile -t GPU_ROWS < <(nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader)
(( ${#GPU_ROWS[@]} >= 2 )) || fail "closed-loop evaluation requires at least two GPUs"
printf '%s\n' "${GPU_ROWS[@]}"
[[ "${GPU_ROWS[0]}" == *L20* && "${GPU_ROWS[1]}" == *L20* ]] || fail "the first two GPUs must be L20"
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
log "checkpoint: $CHECKPOINT_SOURCE"
log "persistent output: $PERSISTENT_OUTPUT"
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

EVAL_PID=""
cleanup() {
    local status=$? preserve_ok=1
    set +e
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
    if [[ -n "$EVAL_PID" ]] && kill -0 "$EVAL_PID" 2>/dev/null; then
        kill -TERM -- "-$EVAL_PID" 2>/dev/null || true
        wait "$EVAL_PID" 2>/dev/null || true
    fi
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
ln -s "$PI05_ENV" "$REPO_ROOT/openpi/.venv"

log "restoring OpenPI cache"
mkdir -p "$OPENPI_DATA_HOME"
tar -xf "$OPENPI_CACHE_ARCHIVE" -C "$OPENPI_DATA_HOME"
require_file "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
require_file "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_base/params/_METADATA"

log "copying inference checkpoint from OSS"
mkdir -p "$LOCAL_CHECKPOINT"
cp -a "$CHECKPOINT_SOURCE/params" "$CHECKPOINT_SOURCE/assets" "$LOCAL_CHECKPOINT/"
if [[ -f "$CHECKPOINT_SOURCE/_CHECKPOINT_METADATA" ]]; then
    cp "$CHECKPOINT_SOURCE/_CHECKPOINT_METADATA" "$LOCAL_CHECKPOINT/"
fi
require_file "$LOCAL_CHECKPOINT/params/_METADATA"
require_file "$LOCAL_CHECKPOINT/assets/$PI05_REPO_ID/norm_stats.json"

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
CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME="$OPENPI_DATA_HOME" XLA_PYTHON_CLIENT_PREALLOCATE=false \
    "$PI05_ENV/bin/python" -c 'import jax, openpi, openpi_client; devices = jax.devices(); assert len(devices) == 1 and devices[0].platform == "gpu", devices; print(openpi.__file__, devices)'
IMPORT_CHECK_RUNTIME="$RUNTIME_ROOT/omnigibson-import-check"
mkdir -p "$IMPORT_CHECK_RUNTIME"
env CUDA_VISIBLE_DEVICES=1 PYTHONNOUSERSITE=1 OMNI_KIT_ACCEPT_EULA=YES \
    OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH" \
    OMNIGIBSON_APPDATA_PATH="$IMPORT_CHECK_RUNTIME" \
    OMNIGIBSON_GPU_ID=1 OMNIGIBSON_HEADLESS=1 OMNIGIBSON_DISABLE_TORCH_COMPILE=1 \
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
        TASK_SPECS=("turning_on_radio|0")
        MAX_STEPS_VALUE="${MAX_STEPS:-100}"
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
    indices="${spec#*|}"
    read -ra index_array <<< "$indices"
    EXPECTED_ROLLOUTS=$((EXPECTED_ROLLOUTS + ${#index_array[@]} * NUM_ROLLOUTS))
done

mkdir -p "$LOCAL_RESULTS" "$PERSISTENT_OUTPUT/status"
"$PI05_ENV/bin/python" - "$PERSISTENT_OUTPUT/manifest.json" <<PY
import json, pathlib
path = pathlib.Path("$PERSISTENT_OUTPUT/manifest.json")
path.write_text(json.dumps({
    "run_id": "$RUN_ID",
    "profile": "$EVAL_PROFILE",
    "checkpoint": "$CHECKPOINT_SOURCE",
    "repo_id": "$PI05_REPO_ID",
    "subtask_prediction": True,
    "subtask_interval": 1,
    "subtask_temperature": 0.0,
    "instance_indices": "$INSTANCE_INDICES",
    "num_rollouts": $NUM_ROLLOUTS,
    "max_steps": "${MAX_STEPS_VALUE}",
    "expected_rollouts": $EXPECTED_ROLLOUTS,
    "driver": "$DRIVER",
    "server_gpu": 0,
    "eval_gpu": 1,
    "code_commit": "$CODE_COMMIT",
    "code_dirty": "$CODE_DIRTY",
}, indent=2, sort_keys=True) + "\n")
PY

status=0
for spec in "${TASK_SPECS[@]}"; do
    task="${spec%%|*}"
    indices="${spec#*|}"
    log "evaluating $task on public indices [$indices]"
    set +e
    setsid env MODEL=pi05 PI05_CKPT="$LOCAL_CHECKPOINT" PI05_REPO_ID="$PI05_REPO_ID" \
        SUBTASK_PREDICTION=1 MAX_TOKEN_LEN=512 MAX_SUBTASK_LEN=64 \
        SUBTASK_INTERVAL=1 SUBTASK_TEMPERATURE=0 PROMPT_MODE=whole_task \
        SERVER_GPU=0 EVAL_GPU=1 PORT=8000 EVAL_ENV="$EVAL_ENV" \
        EVAL_RUNTIME_ROOT="$RUNTIME_ROOT/omnigibson-runtime" \
        OPENPI_DATA_HOME="$OPENPI_DATA_HOME" OMNIGIBSON_DATA_PATH="$OMNIGIBSON_DATA_PATH" \
        OMNIGIBSON_DISABLE_TORCH_COMPILE=1 \
        OUTPUT_ROOT="$LOCAL_RESULTS" INSTANCE_INDICES="$indices" NUM_ROLLOUTS="$NUM_ROLLOUTS" \
        MAX_STEPS="$MAX_STEPS_VALUE" "$REPO_ROOT/scripts/eval_b1k.sh" "$task" &
    EVAL_PID=$!
    wait "$EVAL_PID"
    task_status=$?
    EVAL_PID=""
    set -e
    cp -a "$LOCAL_RESULTS/." "$PERSISTENT_OUTPUT/"
    printf '{"task":"%s","exit_code":%d}\n' "$task" "$task_status" > "$PERSISTENT_OUTPUT/status/$task.json"
    (( task_status == 0 )) || status=1
done

set +e
"$PI05_ENV/bin/python" "$SCRIPT_DIR/summarize_pi05_eval.py" \
    --results-root "$LOCAL_RESULTS" --output "$LOCAL_RESULTS/summary.json" \
    --run-id "$RUN_ID" --profile "$EVAL_PROFILE" --checkpoint "$CHECKPOINT_SOURCE" \
    --expected-rollouts "$EXPECTED_ROLLOUTS" --require-complete
summary_status=$?
set -e
cp "$LOCAL_RESULTS/summary.json" "$PERSISTENT_OUTPUT/summary.json"
(( summary_status == 0 )) || status=1

if [[ "$EVAL_PROFILE" == smoke ]]; then
    [[ -n "$(find "$LOCAL_RESULTS/pi05" -path '*/json/*.json' -type f -size +0c -print -quit)" ]] \
        || fail "smoke produced no nonempty rollout JSON"
    [[ -n "$(find "$LOCAL_RESULTS/pi05" -path '*/videos/*.mp4' -type f -size +0c -print -quit)" ]] \
        || fail "smoke produced no nonempty video"
    grep -q 'Generated subtask:' "$LOCAL_RESULTS/pi05/server_8000.log" \
        || fail "smoke server log contains no generated subtask"
fi

if (( status == 0 )); then
    touch "$PERSISTENT_OUTPUT/_SUCCESS"
    rm -f "$PERSISTENT_OUTPUT/_FAILED"
    log "evaluation complete: $PERSISTENT_OUTPUT"
else
    fail "one or more evaluation tasks failed"
fi
