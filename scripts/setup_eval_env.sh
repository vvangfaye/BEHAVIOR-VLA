#!/usr/bin/env bash
# Build the OmniGibson + Isaac Sim 5.1 closed-loop evaluation environment as a
# prefix-based conda env, mirroring scripts/setup_envs.sh conventions.
#
# Usage:
#   ./scripts/setup_eval_env.sh
#   EVAL_ENV=/somewhere/b1k-eval ./scripts/setup_eval_env.sh
#
# This is the stage bootstrap.sh deliberately skips: Isaac Sim needs a GPU with
# RT cores, which the H20 training hosts lack. The L20 eval host has them.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
EVAL_ENV="${EVAL_ENV:-$ENV_ROOT/b1k-eval}"
CONDA="${CONDA:-/opt/conda/bin/conda}"
# setup.sh hard-requires exactly 3.11, and the Isaac Sim wheels are cp311-only.
EVAL_PYTHON="${EVAL_PYTHON:-3.11.13}"
WHEEL_DIR="${WHEEL_DIR:-/dev/shm/wangfaye/wheels/isaacsim}"
B1K="$REPO_ROOT/BEHAVIOR-1K"

export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-http://yum.tbsite.net/pypi/simple}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/wangfaye/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export OMNI_KIT_ACCEPT_EULA=YES
export GIT_LFS_SKIP_SMUDGE=1

# /checkpoint/binary/train_package sits on PYTHONPATH in this image and shadows
# packages installed into the environment.
export PYTHONNOUSERSITE=1
unset PYTHONPATH
# setup.sh refuses to run if a foreign Isaac Sim is already pointed at.
unset EXP_PATH CARB_APP_PATH ISAAC_PATH 2>/dev/null || true

PY="$EVAL_ENV/bin/python"

# Isaac Sim's libcarb.so needs GLIBCXX_3.4.29; /lib64 stops at 3.4.28 and comes
# first in the ambient LD_LIBRARY_PATH, so put this env's 3.4.34 copy ahead of it
# for the verification steps below (the activation hook does the same at runtime).
export LD_LIBRARY_PATH="$EVAL_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

ISAAC_VERSION=5.1.0.0
KIT_VERSION=107.3.1.206797
ISAAC_PACKAGES=(
    "omniverse_kit-$KIT_VERSION"
    "isaacsim_kernel-$ISAAC_VERSION"
    "isaacsim_app-$ISAAC_VERSION"
    "isaacsim_core-$ISAAC_VERSION"
    "isaacsim_gui-$ISAAC_VERSION"
    "isaacsim_utils-$ISAAC_VERSION"
    "isaacsim_storage-$ISAAC_VERSION"
    "isaacsim_asset-$ISAAC_VERSION"
    "isaacsim_sensor-$ISAAC_VERSION"
    "isaacsim_robot_motion-$ISAAC_VERSION"
    "isaacsim_robot-$ISAAC_VERSION"
    "isaacsim_benchmark-$ISAAC_VERSION"
    "isaacsim_code_editor-$ISAAC_VERSION"
    "isaacsim_ros1-$ISAAC_VERSION"
    "isaacsim_cortex-$ISAAC_VERSION"
    "isaacsim_example-$ISAAC_VERSION"
    "isaacsim_replicator-$ISAAC_VERSION"
    "isaacsim_rl-$ISAAC_VERSION"
    "isaacsim_robot_setup-$ISAAC_VERSION"
    "isaacsim_ros2-$ISAAC_VERSION"
    "isaacsim_template-$ISAAC_VERSION"
    "isaacsim_test-$ISAAC_VERSION"
    "isaacsim-$ISAAC_VERSION"
    "isaacsim_extscache_physics-$ISAAC_VERSION"
    "isaacsim_extscache_kit-$ISAAC_VERSION"
    "isaacsim_extscache_kit_sdk-$ISAAC_VERSION"
)

log() { echo "[$(date '+%F %T')] $*"; }

conda_pack_scripts() {
    local tool_dir
    tool_dir="$(dirname "$(readlink -f "$(command -v conda-pack)")")"
    find "${tool_dir%/bin}" -type d -path "*/conda_pack/scripts/posix" -print -quit
}

add_activation_support() {
    # A prefix-based conda env has no bin/activate, but eval_b1k.sh and the
    # packed tarball rely on `source <env>/bin/activate`.
    local prefix="$1" src
    src="$(conda_pack_scripts)"
    if [[ -z "$src" ]]; then
        echo "conda-pack scripts not found; install it with: uv tool install conda-pack" >&2
        exit 1
    fi
    install -m 755 "$src/activate" "$prefix/bin/activate"
    install -m 755 "$src/deactivate" "$prefix/bin/deactivate"
    install -m 644 "$src/activate.fish" "$prefix/bin/activate.fish"

    mkdir -p "$prefix/etc/conda/activate.d"
    cat > "$prefix/etc/conda/activate.d/00-b1k-eval.sh" <<'EOF'
export PYTHONNOUSERSITE=1
unset PYTHONPATH

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# Prepend, not append: Isaac Sim's libcarb.so needs GLIBCXX_3.4.29 and the
# container's /lib64/libstdc++.so.6 stops at 3.4.28, while this env ships 3.4.34.
# The ambient LD_LIBRARY_PATH lists /lib64 first, so appending loses.
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# Accepted once at install time; kit re-checks on every launch.
export OMNI_KIT_ACCEPT_EULA=YES
# huggingface.co is unreachable from this network.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# This container's driver mount has the graphics libraries but not the JSON
# manifests the nvidia container runtime normally installs, so the Vulkan loader
# reports "Found no drivers" and Kit cannot start. See write_gpu_manifests.
export VK_ICD_FILENAMES="$CONDA_PREFIX/share/vulkan/icd.d/nvidia_icd.json"
export __EGL_VENDOR_LIBRARY_FILENAMES="$CONDA_PREFIX/share/glvnd/egl_vendor.d/10_nvidia.json"
EOF
}

write_gpu_manifests() {
    # The Vulkan ICD must point at libEGL_nvidia, not libGLX_nvidia: both export
    # the vk_icd* entrypoints, but libGLX_nvidia's
    # vk_icdNegotiateLoaderICDInterfaceVersion returns VK_ERROR_INITIALIZATION_FAILED
    # and every proc lookup yields NULL, while libEGL_nvidia negotiates cleanly and
    # enumerates all 8 L20s.
    local prefix="$1" driver_lib
    driver_lib="$(ls /usr/local/nvidia/lib64/libEGL_nvidia.so.*.* 2>/dev/null | head -1)"
    [[ -n "$driver_lib" ]] || driver_lib="$(ls /lib64/libEGL_nvidia.so.*.* 2>/dev/null | head -1)"
    if [[ -z "$driver_lib" ]]; then
        echo "libEGL_nvidia not found; this host cannot run Isaac Sim" >&2
        exit 1
    fi

    mkdir -p "$prefix/share/vulkan/icd.d" "$prefix/share/glvnd/egl_vendor.d"
    cat > "$prefix/share/vulkan/icd.d/nvidia_icd.json" <<EOF
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "$driver_lib",
        "api_version": "1.3.242"
    }
}
EOF
    cat > "$prefix/share/glvnd/egl_vendor.d/10_nvidia.json" <<'EOF'
{
    "file_format_version": "1.0.0",
    "ICD": {
        "library_path": "libEGL_nvidia.so.0"
    }
}
EOF
    log "wrote Vulkan/EGL manifests for $driver_lib"

    VK_ICD_FILENAMES="$prefix/share/vulkan/icd.d/nvidia_icd.json" \
        "$PY" - <<'PY'
import ctypes, sys

vk = ctypes.CDLL("libvulkan.so.1")

class AppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p), ("applicationVersion", ctypes.c_uint32),
                ("pEngineName", ctypes.c_char_p), ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32)]

class CreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p), ("flags", ctypes.c_uint32),
                ("pApplicationInfo", ctypes.POINTER(AppInfo)),
                ("enabledLayerCount", ctypes.c_uint32), ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32), ("ppEnabledExtensionNames", ctypes.c_void_p)]

app = AppInfo(0, None, b"probe", 1, b"none", 1, (1 << 22) | (3 << 12))
ci = CreateInfo(1, None, 0, ctypes.pointer(app), 0, None, 0, None)
inst = ctypes.c_void_p()
if vk.vkCreateInstance(ctypes.byref(ci), None, ctypes.byref(inst)) != 0:
    sys.exit("vkCreateInstance failed: Isaac Sim will not start")

n = ctypes.c_uint32()
vk.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
if not n.value:
    sys.exit("Vulkan reports no physical devices")
print(f"  Vulkan ok: {n.value} device(s)")
PY
}

fetch_isaac_wheels() {
    # The published wheels are tagged manylinux_2_35 but the binaries run on
    # glibc 2.32; upstream setup.sh retags them to manylinux_2_31 for exactly
    # this case. Download in parallel: a serial curl loop wastes minutes on the
    # ~10 GB of extscache wheels.
    mkdir -p "$WHEEL_DIR"
    local glibc_old=0
    ldd --version 2>&1 | grep -qE "2\.(31|32|33|34)" && glibc_old=1

    local jobs=() pkg name src_tag dst_tag file url
    src_tag="manylinux_2_35_x86_64"
    dst_tag="$src_tag"
    (( glibc_old )) && dst_tag="manylinux_2_31_x86_64"
    (( glibc_old )) && log "glibc $(ldd --version | head -1 | grep -oE '2\.[0-9]+') detected: retagging wheels to manylinux_2_31"

    for pkg in "${ISAAC_PACKAGES[@]}"; do
        name="$(echo "${pkg%-*}" | tr '_' '-')"
        file="${pkg}-cp311-none-${dst_tag}.whl"
        url="https://pypi.nvidia.com/${name}/${pkg}-cp311-none-${src_tag}.whl"
        if "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).namelist()" "$WHEEL_DIR/$file" 2>/dev/null; then
            continue
        fi
        jobs+=("$url"$'\t'"$WHEEL_DIR/$file")
    done

    if (( ${#jobs[@]} )); then
        log "downloading ${#jobs[@]} Isaac Sim wheels"
        # curl 7.61 on this image predates --retry-all-errors.
        printf '%s\n' "${jobs[@]}" | xargs -P 6 -d '\n' -I{} bash -c '
            IFS=$'"'"'\t'"'"' read -r url out <<< "{}"
            curl -sL --retry 5 --retry-delay 2 --retry-connrefused --max-time 1800 -o "$out.part" "$url" \
                && mv "$out.part" "$out" \
                && echo "  got $(basename "$out")"
        '
    fi

    ISAAC_WHEELS=()
    for pkg in "${ISAAC_PACKAGES[@]}"; do
        file="$WHEEL_DIR/${pkg}-cp311-none-${dst_tag}.whl"
        "$PY" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).namelist()" "$file" \
            || { echo "corrupt or missing wheel: $file" >&2; exit 1; }
        ISAAC_WHEELS+=("$file")
    done
}

# --- 1. environment -----------------------------------------------------------
if [[ -x "$PY" ]]; then
    log "reusing $EVAL_ENV ($("$PY" -V))"
else
    log "creating $EVAL_ENV (python $EVAL_PYTHON)"
    "$CONDA" create --prefix "$EVAL_ENV" "python=$EVAL_PYTHON" pip -y
fi
uv pip install --python "$PY" -U setuptools wheel

# --- 2. bddl3 then OmniGibson ------------------------------------------------
# OmniGibson requires bddl~=3.7.0; the in-tree bddl3 satisfies it, so it must be
# installed first or pip resolves the older PyPI release instead.
log "installing bddl3"
uv pip install --python "$PY" -e "$B1K/bddl3"

log "installing OmniGibson[eval]"
uv pip install --python "$PY" -e "$B1K/OmniGibson[eval]" --no-build-isolation

# omnigibson.eval.evaluator imports gello.utils.og_teleop_cfg, so joylo is a hard
# requirement for evaluation (upstream setup.sh rejects --eval without --joylo).
# --no-deps: the modules eval reaches need only torch/numpy/yaml/omnigibson, while
# joylo's declared dependencies are teleoperation hardware libraries.
log "installing joylo (gello) for the evaluator"
uv pip install --python "$PY" -e "$B1K/joylo" --no-deps

# Dependencies of Isaac Sim extensions that normally come from the
# omni.services.pip_archive prebundle. OmniGibson's kit configuration never
# enables that extension, so its prebundle is not on sys.path and the
# extensions fail to start:
#   botocore -> omni.replicator.core (S3 backend), needed for render products
#   osqp     -> isaacsim.robot.wheeled_robots (HolonomicController)
log "installing Isaac Sim extension dependencies"
uv pip install --python "$PY" botocore osqp

# --- 3. Isaac Sim 5.1 --------------------------------------------------------
if "$PY" -c "import isaacsim" 2>/dev/null; then
    log "Isaac Sim already installed"
else
    fetch_isaac_wheels
    log "installing ${#ISAAC_WHEELS[@]} Isaac Sim wheels"
    # Upstream uses pip here. These wheels are unusual (multi-GB extscache trees,
    # namespace packages), so stay on the path NVIDIA and BEHAVIOR test.
    "$PY" -m pip install --no-deps "${ISAAC_WHEELS[@]}"
    "$PY" -c "import isaacsim" || { echo "isaacsim import failed" >&2; exit 1; }
fi

# --- 4. post-install conflict fixups (from BEHAVIOR-1K/setup.sh) -------------
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
if [[ -d "$SITE/isaacsim/extscache" ]]; then
    log "removing bundled websockets that shadow the pip version"
    find "$SITE/isaacsim/extscache" -type d -name websockets -path "*/pip_prebundle/*" \
        -exec rm -rf {} + 2>/dev/null || true
fi
# isaacsim pins packaging 23.0 while omni.kit ships 25.0.
PKG_CONFLICT="$SITE/isaacsim/extscache/omni.services.pip_archive-0.16.0+107.0.3.lx64.cp311/pip_prebundle/packaging"
[[ -d "$PKG_CONFLICT" ]] && { log "removing conflicting bundled packaging"; rm -rf "$PKG_CONFLICT"; }

log "pinning cffi and websockets that Isaac Sim downgraded"
uv pip install --python "$PY" --reinstall cffi==1.17.1
uv pip install --python "$PY" --reinstall "websockets>=15.0.1"

# --- 5. eval extra: torch-cluster matching the installed torch ---------------
TORCH_VERSION="$("$PY" -c 'import torch; print(torch.__version__.split("+")[0])' 2>/dev/null || true)"
if [[ -n "$TORCH_VERSION" ]]; then
    log "installing torch-cluster for torch $TORCH_VERSION"
    uv pip install --python "$PY" torch-cluster \
        --find-links "https://data.pyg.org/whl/torch-${TORCH_VERSION}.html" \
        --no-build-isolation || log "  torch-cluster unavailable for torch $TORCH_VERSION (only needed by some eval metrics)"
else
    log "torch not installed; skipping torch-cluster"
fi

add_activation_support "$EVAL_ENV"

# The container ships no Vulkan loader (no libvulkan.so.1 on the system), which
# Isaac Sim needs to reach the NVIDIA ICD. The activation hook already prepends
# $CONDA_PREFIX/lib, so installing the loader here makes it resolvable.
if [[ ! -e "$EVAL_ENV/lib/libvulkan.so.1" ]]; then
    log "installing the Vulkan loader"
    "$CONDA" install --prefix "$EVAL_ENV" -c conda-forge vulkan-tools -y --freeze-installed
fi

write_gpu_manifests "$EVAL_ENV"

log "verifying"
VK_ICD_FILENAMES="$EVAL_ENV/share/vulkan/icd.d/nvidia_icd.json" \
__EGL_VENDOR_LIBRARY_FILENAMES="$EVAL_ENV/share/glvnd/egl_vendor.d/10_nvidia.json" \
"$PY" - <<'PY'
import isaacsim, omnigibson, torch, bddl
print("isaacsim   ", isaacsim.__file__)
print("omnigibson ", omnigibson.__version__)
print("bddl       ", bddl.__version__ if hasattr(bddl, "__version__") else "ok")
print("torch      ", torch.__version__, "cuda:", torch.cuda.is_available())
PY

log "eval env ready: $EVAL_ENV"
