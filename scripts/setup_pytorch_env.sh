#!/usr/bin/env bash
# Build a separate environment so Transformers patches do not alter the JAX environment.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_ROOT="${ENV_ROOT:-/dev/shm/wangfaye/conda_envs}"
PYTORCH_ENV="${PYTORCH_ENV:-$ENV_ROOT/b1k-pi05-torch}"
PI05_ENV="$PYTORCH_ENV" OPENPI_ENV_LINK="$REPO_ROOT/openpi/.venv-torch" \
    bash "$REPO_ROOT/scripts/setup_envs.sh" pi05
"$PYTORCH_ENV/bin/python" "$REPO_ROOT/openpi/scripts/install_transformers_patch.py"
JAX_PLATFORMS=cpu "$PYTORCH_ENV/bin/python" - <<'PY'
import torch
from transformers.models.gemma.modeling_gemma import GemmaRMSNorm
layer = GemmaRMSNorm(8, cond_dim=8)
output, gate = layer(torch.zeros(1, 2, 8), torch.zeros(1, 8))
assert output.shape == (1, 2, 8)
print("Torch environment ready:", torch.__version__, "CUDA:", torch.cuda.is_available())
PY
