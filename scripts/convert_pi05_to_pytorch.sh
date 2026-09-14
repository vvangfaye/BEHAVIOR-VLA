#!/usr/bin/env bash
# Convert a JAX checkpoint root (containing params/) into Torch model.safetensors.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$REPO_ROOT/openpi}"
PYTORCH_ENV="${PYTORCH_ENV:-$OPENPI_DIR/.venv-torch}"
if [[ $# -ne 2 ]]; then
    echo "Usage: $0 /path/to/jax-checkpoint /path/to/torch-checkpoint" >&2
    exit 2
fi
JAX_CHECKPOINT="$1"
TORCH_CHECKPOINT="$2"
for name in OPENPI_DIR PYTORCH_ENV JAX_CHECKPOINT TORCH_CHECKPOINT; do
    value="${!name}"
    if [[ "$value" != /* ]]; then printf -v "$name" '%s/%s' "$PWD" "$value"; fi
done
"$PYTORCH_ENV/bin/python" "$OPENPI_DIR/scripts/install_transformers_patch.py" --check
cd "$OPENPI_DIR"
export JAX_PLATFORMS=cpu
exec "$PYTORCH_ENV/bin/python" examples/convert_jax_model_to_pytorch.py \
    --config-name pi05_b1k --checkpoint-dir "$JAX_CHECKPOINT" --output-path "$TORCH_CHECKPOINT" \
    --precision "${CONVERT_PRECISION:-float32}"
