#!/usr/bin/env bash
# Train high-level task -> subtask language jointly with subtask-conditioned actions.
set -euo pipefail
SUBTASK_MODE="${1:-check}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SUBTASK_PREDICTION=1
export WHOLE_TASK_PROB="${WHOLE_TASK_PROB:-0}"
export EXP_NAME="${EXP_NAME:-pi05_b1k_subtask_${SUBTASK_MODE}_$(date +%Y%m%d_%H%M%S)}"
case "$SUBTASK_MODE" in
    check)
        export PREFLIGHT_ONLY=1
        exec bash "$CODE_ROOT/scripts/train_pi05_b1k.sh" full
        ;;
    smoke|full)
        exec bash "$CODE_ROOT/scripts/train_pi05_b1k.sh" "$SUBTASK_MODE"
        ;;
    *)
        echo "Usage: $0 [check|smoke|full]" >&2
        exit 2
        ;;
esac
