#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-restore}"
if [[ "$MODE" != "restore" && "$MODE" != "check" ]]; then
    echo "usage: $0 [restore|check]" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/data/oss_bucket_0/wangfaye/conda_env}"
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
PI05_ENV="$ENV_ROOT/b1k-pi05"
EVAL_ENV="$ENV_ROOT/b1k-eval"
DATA_PATH="${DATA_PATH:-/dev/shm/wangfaye/omnigibson-data}"
ASSET_CACHE="${ASSET_CACHE:-$REPO_ROOT/BEHAVIOR-1K/.downloads/hf-assets}"
DATA_LINK="$REPO_ROOT/BEHAVIOR-1K/omnigibson-data"
KEY_PATH="$REPO_ROOT/BEHAVIOR-1K/datasets/omnigibson.key"

log() { printf '[restore-pi05] %s\n' "$*"; }
fail() { printf '[restore-pi05] ERROR: %s\n' "$*" >&2; exit 1; }

restore_env() {
    local name="$1" target="$2" archive
    archive="$ARCHIVE_ROOT/$name.tar.gz"
    if [[ -x "$target/bin/python" ]]; then
        log "$name already present: $target"
        return
    fi
    [[ "$MODE" == "restore" ]] || fail "$name is missing: $target"
    [[ -s "$archive" ]] || fail "archive is missing or empty: $archive"
    if [[ -d "$target" && -n "$(ls -A "$target" 2>/dev/null)" ]]; then
        fail "target exists but is incomplete; inspect it before moving or removing it: $target"
    fi
    mkdir -p "$target"
    log "extracting $archive"
    tar xzf "$archive" -C "$target"
    [[ -x "$target/bin/python" ]] || fail "archive did not produce a usable environment: $target"
    if [[ -x "$target/bin/conda-unpack" ]]; then
        log "relocating $name"
        "$target/bin/conda-unpack"
    fi
}

link_pi05_env() {
    local link="$REPO_ROOT/openpi/.venv"
    if [[ -e "$link" && ! -L "$link" && "$(readlink -f "$link")" != "$(readlink -f "$PI05_ENV")" ]]; then
        fail "$link is a real directory; inspect it before replacing it"
    fi
    [[ "$MODE" == "restore" ]] && ln -sfn "$PI05_ENV" "$link"
    [[ "$(readlink -f "$link")" == "$(readlink -f "$PI05_ENV")" ]] || fail "openpi/.venv does not point to $PI05_ENV"
}

write_gpu_manifests() {
    local driver lib="" path
    driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | sed -n '1{s/ //g;p;}')"
    [[ -n "$driver" ]] || fail "could not determine NVIDIA driver version"
    for path in \
        "/usr/local/nvidia/lib64/libEGL_nvidia.so.$driver" \
        "/usr/lib64/libEGL_nvidia.so.$driver" \
        "/lib64/libEGL_nvidia.so.$driver"; do
        if [[ -f "$path" ]]; then lib="$path"; break; fi
    done
    [[ -n "$lib" ]] || fail "libEGL_nvidia for driver $driver was not found"

    if [[ "$MODE" == "restore" ]]; then
        mkdir -p "$EVAL_ENV/share/vulkan/icd.d" "$EVAL_ENV/share/glvnd/egl_vendor.d" "$EVAL_ENV/etc/conda/activate.d"
        printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {\n    "library_path": "%s",\n    "api_version": "1.3.242"\n  }\n}\n' "$lib" \
            > "$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json"
        printf '{\n  "file_format_version": "1.0.0",\n  "ICD": {\n    "library_path": "%s"\n  }\n}\n' "$lib" \
            > "$EVAL_ENV/share/glvnd/egl_vendor.d/10_nvidia.json"
        cat > "$EVAL_ENV/etc/conda/activate.d/99-b1k-gpu-manifests.sh" <<'EOF'
export VK_ICD_FILENAMES="$CONDA_PREFIX/share/vulkan/icd.d/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$CONDA_PREFIX/share/glvnd/egl_vendor.d/10_nvidia.json"
EOF
    fi
    grep -qF "$lib" "$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json" || fail "Vulkan manifest does not target $lib"
    log "NVIDIA driver $driver; Vulkan ICD $lib"
}

extract_asset() {
    local name="$1" archive_name="$2" marker="$3"
    local target="$DATA_PATH/$name" archive="$ASSET_CACHE/$archive_name"
    if [[ -e "$target/$marker" ]]; then
        log "$name already extracted"
        return
    fi
    [[ "$MODE" == "restore" ]] || fail "$name is missing under $DATA_PATH"
    [[ -s "$archive" ]] || fail "asset archive is missing or empty: $archive"
    log "verifying $archive_name"
    unzip -tq "$archive" >/dev/null
    log "extracting $archive_name"
    mkdir -p "$target"
    unzip -q -o "$archive" -d "$target"
    [[ -e "$target/$marker" ]] || fail "asset marker missing after extraction: $target/$marker"
}

restore_assets() {
    mkdir -p "$DATA_PATH"
    extract_asset "omnigibson-robot-assets" "omnigibson-robot-assets-3.8.2.zip" "VERSION"
    extract_asset "2026-challenge-task-instances" "2026-challenge-task-instances.zip" "metadata/task.jsonl"
    extract_asset "behavior-1k-assets" "behavior-1k-assets-3.9.0.zip" "VERSION"

    if [[ "$MODE" == "restore" ]]; then
        if [[ -e "$DATA_LINK" && ! -L "$DATA_LINK" ]]; then
            fail "$DATA_LINK is a real directory; inspect it before replacing it"
        fi
        ln -sfn "$DATA_PATH" "$DATA_LINK"
    fi
    [[ "$(readlink -f "$DATA_LINK")" == "$(readlink -f "$DATA_PATH")" ]] || fail "asset link is incorrect: $DATA_LINK"
}

ensure_key() {
    local runtime_key="$DATA_PATH/omnigibson.key"
    if [[ ! -s "$KEY_PATH" ]]; then
        [[ "$MODE" == "restore" ]] || fail "OmniGibson key is missing: $KEY_PATH"
        log "downloading the small OmniGibson decryption key"
        OMNIGIBSON_NO_OMNIVERSE=1 "$EVAL_ENV/bin/python" -c \
            'from omnigibson.utils.asset_utils import download_key; download_key()'
    fi
    [[ -s "$KEY_PATH" ]] || fail "OmniGibson key is missing: $KEY_PATH"
    if [[ "$MODE" == "restore" ]]; then
        if [[ -e "$runtime_key" && ! -L "$runtime_key" && ! -s "$runtime_key" ]]; then
            fail "$runtime_key exists but is empty; inspect it before replacing it"
        fi
        if [[ ! -e "$runtime_key" || -L "$runtime_key" ]]; then
            ln -sfn "$KEY_PATH" "$runtime_key"
        fi
    fi
    [[ -s "$runtime_key" ]] || fail "runtime OmniGibson key is missing: $runtime_key"
    log "OmniGibson key present"
}

repair_eval_build_packages() {
    if env -u PYTHONPATH PYTHONNOUSERSITE=1 "$EVAL_ENV/bin/python" \
        -c 'import packaging, setuptools' >/dev/null 2>&1; then
        log "eval build packages are complete"
        return
    fi
    [[ "$MODE" == "restore" ]] || fail "eval build packages are incomplete; run restore mode to repair them"
    command -v uv >/dev/null || fail "uv is required to repair eval build packages"
    log "repairing the archived packaging and setuptools packages"
    UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-http://yum.tbsite.net/pypi/simple}" \
    UV_LINK_MODE=copy \
        uv pip install --python "$EVAL_ENV/bin/python" --reinstall \
            "packaging==25.0" "setuptools==80.10.2"
    env -u PYTHONPATH PYTHONNOUSERSITE=1 "$EVAL_ENV/bin/python" \
        -c 'import packaging, setuptools' || fail "eval build-package repair failed"
}

verify_runtime() {
    XLA_PYTHON_CLIENT_PREALLOCATE=false "$PI05_ENV/bin/python" - <<'PY'
import jax
import lerobot
import openpi
import openpi_client
import websockets
print(f"[restore-pi05] pi05 imports ok; JAX devices={len(jax.devices())}; websockets={websockets.__version__}")
PY

    env -u PYTHONPATH \
    PYTHONNOUSERSITE=1 \
    OMNI_KIT_ACCEPT_EULA=YES \
    OMNIGIBSON_HEADLESS=1 \
    LD_LIBRARY_PATH="$EVAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    VK_ICD_FILENAMES="$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json" \
    __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_ENV/share/glvnd/egl_vendor.d/10_nvidia.json" \
        "$EVAL_ENV/bin/python" - <<'PY'
import bddl
import isaacsim
import omnigibson
import torch
print(f"[restore-pi05] eval imports ok; OmniGibson={omnigibson.__version__}; torch={torch.__version__}; cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable in b1k-eval")
PY

    LD_LIBRARY_PATH="$EVAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    VK_ICD_FILENAMES="$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json" \
    __EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_ENV/share/glvnd/egl_vendor.d/10_nvidia.json" \
        "$EVAL_ENV/bin/vulkaninfo" --summary 2>/dev/null \
        | grep -E 'Vulkan Instance Version|deviceName|driverInfo' \
        | sed 's/^/[restore-pi05] /'
}

restore_env b1k-pi05 "$PI05_ENV"
restore_env b1k-eval "$EVAL_ENV"
link_pi05_env
write_gpu_manifests
restore_assets
ensure_key
repair_eval_build_packages
verify_runtime
log "$MODE complete"
