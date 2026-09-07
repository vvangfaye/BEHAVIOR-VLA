#!/usr/bin/env bash
# Build the pi0.5 (openpi) environment
# as prefix-based conda environments that conda-pack can turn into relocatable
# tarballs. See scripts/pack_envs.sh for packing.
#
# Usage:
#   ./scripts/setup_envs.sh            # build pi05
#   ./scripts/setup_envs.sh pi05
#
# The environments live outside the repository (tmpfs by default) because they
# are ~15-25 GB each and must not be committed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
CONDA="${CONDA:-/opt/conda/bin/conda}"
PI05_ENV="${PI05_ENV:-$ENV_ROOT/b1k-pi05}"
PI05_PYTHON="${PI05_PYTHON:-3.11.13}"

# The internal mirror proxies all of PyPI and is ~25x faster than pypi.org from
# this network. Keep a single index: uv's default first-index strategy then
# needs one metadata request per package instead of one per index per package.
export UV_DEFAULT_INDEX="${UV_DEFAULT_INDEX:-http://yum.tbsite.net/pypi/simple}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/dev/shm/wangfaye/.cache/uv}"
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-600}"
export GIT_LFS_SKIP_SMUDGE=1

# /checkpoint/binary/train_package sits on PYTHONPATH in this image and shadows
# packages installed into the environment.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

TARGETS=("$@")
if [[ ${#TARGETS[@]} -eq 0 ]]; then
    TARGETS=(pi05)
fi

conda_pack_scripts() {
    local tool_dir
    tool_dir="$(dirname "$(readlink -f "$(command -v conda-pack)")")"
    find "${tool_dir%/bin}" -type d -path "*/conda_pack/scripts/posix" -print -quit
}

add_activation_support() {
    # A prefix-based conda env has no bin/activate, but the training scripts
    # (and the packed tarball) rely on `source <env>/bin/activate`. Reuse the
    # scripts conda-pack itself ships so local and packed envs behave alike.
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
    cat > "$prefix/etc/conda/activate.d/00-b1k.sh" <<'EOF'
# Keep the interpreter isolated from the container's global Python state.
# PYTHONPATH points at /checkpoint/binary/train_package on these notebook
# images, which shadows packages installed in this environment.
export PYTHONNOUSERSITE=1
unset PYTHONPATH

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

# torchcodec needs libavutil.so.56 (FFmpeg 4). The image ships FFmpeg 6
# (libavutil.so.58), which torchcodec 0.4.0 does not support, so FFmpeg 4 is
# installed into this environment. Append so system libraries still win for
# names that exist in both.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$CONDA_PREFIX/lib"

EOF
}

create_env() {
    local prefix="$1" pyver="$2"
    if [[ -x "$prefix/bin/python" ]]; then
        echo "Reusing $prefix ($("$prefix/bin/python" -V))"
        return
    fi
    echo "Creating $prefix (python $pyver)"
    "$CONDA" create --prefix "$prefix" "python=$pyver" pip -y
}

install_ffmpeg() {
    # torchcodec loads a prebuilt shared library that links against a specific
    # FFmpeg major version. It accepts libavutil 56/57/59/60, while the image
    # ships FFmpeg 6 (libavutil.so.58), so an FFmpeg it supports goes into the
    # environment.
    #
    # Prefer FFmpeg 5 and preserve the existing Python version.
    local prefix="$1" pyver
    if [[ -e "$prefix/lib/libavutil.so.56" || -e "$prefix/lib/libavutil.so.57" ]]; then
        echo "Supported FFmpeg already present in $prefix"
        return
    fi
    pyver="$("$prefix/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
    for spec in "ffmpeg=5" "ffmpeg=4.2.2"; do
        if "$CONDA" install --prefix "$prefix" "python==$pyver" "$spec" -y --freeze-installed; then
            return
        fi
        echo "  $spec did not solve against Python $pyver; trying the next" >&2
    done
    echo "No supported FFmpeg could be installed into $prefix" >&2
    exit 1
}

for target in "${TARGETS[@]}"; do
    case "$target" in
        pi05)
            create_env "$PI05_ENV" "$PI05_PYTHON"
            echo "Installing openpi into $PI05_ENV"
            cd "$REPO_ROOT/openpi"
            uv pip install --python "$PI05_ENV/bin/python" -e . -e packages/openpi-client
            # lerobot leaves torchcodec unpinned; the latest release is built
            # against a newer torch than 2.7.1 and fails to load.
            uv pip install --python "$PI05_ENV/bin/python" "torchcodec==0.4.0"
            install_ffmpeg "$PI05_ENV"
            add_activation_support "$PI05_ENV"
            ln -sfn "$PI05_ENV" "$REPO_ROOT/openpi/.venv"
            ;;
        *)
            echo "Unknown target: $target (expected pi05)" >&2
            exit 2
            ;;
    esac
done

echo "Done."
