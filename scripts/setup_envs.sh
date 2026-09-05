#!/usr/bin/env bash
# Build the π0.5 (openpi), GR00T N1.7 (Isaac-GR00T) and GalaxeaVLA G0.5 environments
# as prefix-based conda environments that conda-pack can turn into relocatable
# tarballs. See scripts/pack_envs.sh for packing.
#
# Usage:
#   ./scripts/setup_envs.sh            # build pi05 and groot (the default pair)
#   ./scripts/setup_envs.sh pi05
#   ./scripts/setup_envs.sh groot
#   ./scripts/setup_envs.sh g05        # GalaxeaVLA, not built by default
#
# The environments live outside the repository (tmpfs by default) because they
# are ~15-25 GB each and must not be committed.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
CONDA="${CONDA:-/opt/conda/bin/conda}"
PI05_ENV="${PI05_ENV:-$ENV_ROOT/b1k-pi05}"
GROOT_ENV="${GROOT_ENV:-$ENV_ROOT/b1k-groot}"
G05_ENV="${G05_ENV:-$ENV_ROOT/b1k-g05}"
PI05_PYTHON="${PI05_PYTHON:-3.11.13}"
GROOT_PYTHON="${GROOT_PYTHON:-3.10.13}"
# GalaxeaVLA pins requires-python = ">=3.10.16,<3.11".
G05_PYTHON="${G05_PYTHON:-3.10.16}"
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

# The g05 build of torchcodec also links libnppicc.so.12 for GPU decoding. NPP
# is not among the CUDA libraries torch pulls in, so nvidia-npp-cu12 is
# installed separately and its lib directory has to be on the path. Globbed for
# whichever Python version this env has; a no-op where NPP is absent.
for _npp in "$CONDA_PREFIX"/lib/python*/site-packages/nvidia/npp/lib; do
    [[ -d "$_npp" ]] && export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$_npp"
done
unset _npp
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
    # FFmpeg major version. It accepts libavutil 56/57/59/60, while the image
    # ships FFmpeg 6 (libavutil.so.58), so an FFmpeg it supports goes into the
    # environment.
    #
    # Prefer FFmpeg 5 (libavutil.so.57). ffmpeg 4.2.2 is an ancient build and its
    # solve fights the interpreter: unpinned it silently took g05 from Python
    # 3.10.16 to 3.10.14, below GalaxeaVLA's >=3.10.16, and pinned it becomes
    # unsatisfiable through a chain of zlib conflicts. FFmpeg 5 solves cleanly
    # against a pinned interpreter.
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
        g05)
            create_env "$G05_ENV" "$G05_PYTHON"
            echo "Installing GalaxeaVLA (g05) into $G05_ENV"
            cd "$REPO_ROOT/GalaxeaVLA"
            # No flash-attn workaround needed here: every dependency, flash-attn-4
            # included, resolves to an ordinary PyPI wheel, so there is no GitHub
            # release asset to fetch. Dependencies are near-fully "==" pinned in
            # pyproject.toml, so a fresh resolve lands on the same set as uv.lock.
            uv pip install --python "$G05_ENV/bin/python" -e .
            # torchcodec's build here links libnppicc.so.12, and NPP is not one of
            # the CUDA libraries torch pulls in. Without it torchcodec fails to
            # load, reporting the same missing library under every FFmpeg version,
            # which reads like an FFmpeg problem but is not one.
            uv pip install --python "$G05_ENV/bin/python" nvidia-npp-cu12
            # g05 pins torchcodec 0.4.0, which needs FFmpeg 4 like the others.
            install_ffmpeg "$G05_ENV"
            add_activation_support "$G05_ENV"
            # The repo ships a .venv built by another user on another host; its
            # interpreter path does not exist here, so bin/python is a dead
            # symlink. Keep it out of the way rather than deleting someone's work.
            if [[ -e "$REPO_ROOT/GalaxeaVLA/.venv" && ! -L "$REPO_ROOT/GalaxeaVLA/.venv" ]]; then
                mv "$REPO_ROOT/GalaxeaVLA/.venv" "$REPO_ROOT/GalaxeaVLA/.venv.foreign"
            fi
            ln -sfn "$G05_ENV" "$REPO_ROOT/GalaxeaVLA/.venv"
            ;;
        *)
            echo "Unknown target: $target (expected pi05, groot or g05)" >&2
            exit 2
            ;;
    esac
done

echo "Done."
