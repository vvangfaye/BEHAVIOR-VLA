#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-nvidia/GR00T-N1.7-3B}"
PROXY_URL="${PROXY_URL:-http://127.0.0.1:7890}"
PERSISTENT_CACHE_ROOT="${PRETRAINED_CACHE_ROOT:-/mnt/afs/visitor34/devhome/cache}"
HF_HOME="${HF_HOME:-${PERSISTENT_CACHE_ROOT}/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

export http_proxy="${PROXY_URL}"
export https_proxy="${PROXY_URL}"
export HTTP_PROXY="${PROXY_URL}"
export HTTPS_PROXY="${PROXY_URL}"
export NO_PROXY="${NO_PROXY:-127.0.0.1,localhost}"
export HF_HOME
export HF_HUB_CACHE
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

if command -v hf >/dev/null 2>&1; then
    HF_CLI="hf"
elif [[ -x /root/.local/bin/hf ]]; then
    HF_CLI="/root/.local/bin/hf"
else
    echo "Error: Hugging Face CLI (hf) was not found." >&2
    echo "Install it with: uv tool install 'huggingface-hub[cli]'" >&2
    exit 1
fi

mkdir -p "${HF_HUB_CACHE}"

echo "Model: ${MODEL_ID}"
echo "Cache: ${HF_HUB_CACHE}"
echo "Proxy: ${PROXY_URL}"
echo "Existing cache size:"
du -sh "${HF_HUB_CACHE}/models--nvidia--GR00T-N1.7-3B" 2>/dev/null || echo "  0 B"

"${HF_CLI}" download \
    "${MODEL_ID}" \
    --cache-dir "${HF_HUB_CACHE}"

echo "Download completed:"
du -sh "${HF_HUB_CACHE}/models--nvidia--GR00T-N1.7-3B"
