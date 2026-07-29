#!/usr/bin/env bash
set -euo pipefail

PROXY_URL="${PROXY_URL:-http://127.0.0.1:7890}"
PERSISTENT_CACHE_ROOT="${PRETRAINED_CACHE_ROOT:-/mnt/afs/visitor34/devhome/cache}"
CACHE_ROOT="${OPENPI_CACHE_ROOT:-${PERSISTENT_CACHE_ROOT}/openpi}"
REMOTE_PATH="gs://openpi-assets/checkpoints/pi05_base/params"
CHECKPOINT_ROOT="${CACHE_ROOT}/openpi-assets/checkpoints/pi05_base"
FINAL_PATH="${CHECKPOINT_ROOT}/params"
PARTIAL_PATH="${CHECKPOINT_ROOT}/params.partial"

export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"

if [[ -d "${FINAL_PATH}" ]]; then
    echo "π0.5 base checkpoint already exists:"
    du -sh "${FINAL_PATH}"
    exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "Error: uv is not available in PATH." >&2
    exit 1
fi

if ! command -v gsutil >/dev/null 2>&1; then
    echo "Installing gsutil with uv..."
    uv tool install gsutil
    UV_TOOL_BIN="$(uv tool dir --bin)"
    export PATH="${UV_TOOL_BIN}:${PATH}"
fi

if ! command -v gsutil >/dev/null 2>&1; then
    echo "Error: gsutil installation succeeded but gsutil is not in PATH." >&2
    echo "Try: export PATH=\"$(uv tool dir --bin):\$PATH\"" >&2
    exit 1
fi

mkdir -p "${PARTIAL_PATH}"

echo "Downloading ${REMOTE_PATH}"
echo "Destination: ${PARTIAL_PATH}"
echo "Proxy: ${PROXY_URL}"
echo "Run this in another terminal to inspect size:"
echo "  watch -n 2 'du -sh ${PARTIAL_PATH}'"

gsutil -m \
    -o GSUtil:parallel_process_count=4 \
    -o GSUtil:parallel_thread_count=8 \
    rsync -r \
    "${REMOTE_PATH}" \
    "${PARTIAL_PATH}"

mv "${PARTIAL_PATH}" "${FINAL_PATH}"

echo "Download completed:"
du -sh "${FINAL_PATH}"
