#!/usr/bin/env bash
# Build the π0.5 (openpi) and GR00T N1.7 (Isaac-GR00T) training environments as
# prefix-based conda environments that conda-pack can turn into relocatable
# tarballs. See scripts/pack_envs.sh for packing.
#
# Usage:
#   ./scripts/setup_envs.sh            # build both
#   ./scripts/setup_envs.sh pi05
#   ./scripts/setup_envs.sh groot
#
# The environments live outside the repository (tmpfs by default) because they
# are ~15-25 GB each and must not be committed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
CONDA="${CONDA:-/opt/conda/bin/conda}"
PI05_ENV="${PI05_ENV:-$ENV_ROOT/b1k-pi05}"
GROOT_ENV="${GROOT_ENV:-$ENV_ROOT/b1k-groot}"
PI05_PYTHON="${PI05_PYTHON:-3.11.13}"
GROOT_PYTHON="${GROOT_PYTHON:-3.10.13}"
WHEEL_DIR="${WHEEL_DIR:-/dev/shm/wangfaye/wheels}"

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
    TARGETS=(pi05 groot)
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

FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.7cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
# Prefixes that take the full GitHub URL as a path suffix, tried in order after
# GitHub itself. Some hosts in this fleet cannot reach github.com at all.
FLASH_ATTN_MIRRORS="${FLASH_ATTN_MIRRORS:-https://ghproxy.net/ https://gh-proxy.com/}"

resolve_flash_attn_url() {
    # Report the first candidate that answers with a size. The size is reused by
    # the chunked download below. `|| true` keeps a failed probe from tripping
    # `set -e` through pipefail, which would otherwise abort before the error
    # message below could explain what went wrong.
    local candidate size prefix
    for prefix in "" $FLASH_ATTN_MIRRORS; do
        candidate="${prefix}${FLASH_ATTN_WHEEL_URL}"
        size="$(curl -sIL --max-time 30 "$candidate" 2>/dev/null \
            | awk 'tolower($1)=="content-length:"{n=$2} END{print n+0}' | tr -dc '0-9' || true)"
        if [[ -n "$size" && "$size" -gt 0 ]]; then
            FLASH_ATTN_URL="$candidate"
            FLASH_ATTN_SIZE="$size"
            [[ -n "$prefix" ]] && echo "  reaching flash-attn through $prefix"
            return 0
        fi
    done
    echo "flash-attn wheel unreachable via github.com or mirrors ($FLASH_ATTN_MIRRORS)." >&2
    echo "Set FLASH_ATTN_MIRRORS to a reachable prefix, or place the wheel at" >&2
    echo "  $WHEEL_DIR/${FLASH_ATTN_WHEEL_URL##*/}" >&2
    return 1
}

fetch_flash_attn() {
    # gr00t pins flash-attn to a GitHub release asset. GitHub gives this network
    # ~76 KB/s on a single connection, so uv stalls for over an hour on the
    # 395 MB wheel. Parallel range requests reach ~1.2 MB/s. Writes the override
    # file that makes uv install the local copy instead of refetching the URL.
    local wheel="$WHEEL_DIR/${FLASH_ATTN_WHEEL_URL##*/}"
    local overrides="$WHEEL_DIR/flash-attn-override.txt"
    mkdir -p "$WHEEL_DIR"

    if ! "$GROOT_ENV/bin/python" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).namelist()" "$wheel" 2>/dev/null; then
        echo "Fetching $(basename "$wheel")"
        resolve_flash_attn_url
        local total="$FLASH_ATTN_SIZE" chunk parts=12
        chunk=$(( (total + parts - 1) / parts ))
        rm -f "$WHEEL_DIR"/flash-attn-part_*
        for ((i = 0; i < parts; i++)); do
            local start=$((i * chunk)) end=$((i * chunk + chunk - 1))
            (( end >= total )) && end=$((total - 1))
            curl -sL --retry 5 --retry-all-errors --max-time 1800 \
                -r "$start-$end" -o "$WHEEL_DIR/flash-attn-part_$(printf %02d "$i")" \
                "$FLASH_ATTN_URL" &
        done
        wait
        cat "$WHEEL_DIR"/flash-attn-part_* > "$wheel"
        rm -f "$WHEEL_DIR"/flash-attn-part_*
        local got
        got="$(stat -c %s "$wheel")"
        if (( got != total )); then
            echo "flash-attn wheel size mismatch: got $got want $total" >&2
            exit 1
        fi
        "$GROOT_ENV/bin/python" -c "import zipfile,sys; zipfile.ZipFile(sys.argv[1]).namelist()" "$wheel"
    fi

    echo "flash-attn @ file://$wheel" > "$overrides"
    FLASH_ATTN_OVERRIDE="$overrides"
}

install_ffmpeg() {
    # torchcodec loads a prebuilt shared library that links against a specific
    # FFmpeg major version. It supports libavutil 56/57/59/60, while the image
    # ships FFmpeg 6 (libavutil.so.58), so FFmpeg 4 goes into the environment.
    local prefix="$1"
    if [[ -e "$prefix/lib/libavutil.so.56" ]]; then
        echo "FFmpeg 4 already present in $prefix"
        return
    fi
    "$CONDA" install --prefix "$prefix" ffmpeg=4.2.2 -y --freeze-installed
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
        groot)
            create_env "$GROOT_ENV" "$GROOT_PYTHON"
            fetch_flash_attn
            echo "Installing gr00t into $GROOT_ENV"
            cd "$REPO_ROOT/Isaac-GR00T"
            uv pip install --python "$GROOT_ENV/bin/python" -e . \
                --override "$FLASH_ATTN_OVERRIDE"
            install_ffmpeg "$GROOT_ENV"
            add_activation_support "$GROOT_ENV"
            ln -sfn "$GROOT_ENV" "$REPO_ROOT/Isaac-GR00T/.venv"
            ;;
        *)
            echo "Unknown target: $target (expected pi05 or groot)" >&2
            exit 2
            ;;
    esac
done

echo "Done."
