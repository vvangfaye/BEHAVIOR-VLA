#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUBTASK_PREDICTION=1
export WHOLE_TASK_PROB="${WHOLE_TASK_PROB:-0}"
export EXP_NAME="${EXP_NAME:-pi05_b1k_subtask_torch_${1:-check}_$(date +%Y%m%d_%H%M%S)}"
exec bash "$REPO_ROOT/scripts/train_pi05_pytorch_b1k.sh" "${1:-check}"
