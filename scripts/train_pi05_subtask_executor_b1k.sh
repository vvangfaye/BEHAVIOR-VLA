#!/usr/bin/env bash
# Train a pure current-subtask executor with strict temporal action masking.
set -euo pipefail

MODE="${1:-check}"
CODE_ROOT="${CODE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export SUBTASK_EXECUTOR=1
export SUBTASK_PREDICTION=0
export WHOLE_TASK_PROB=0
export MAX_SAMPLES_PER_SEGMENT="${MAX_SAMPLES_PER_SEGMENT:-64}"
export SUBTASK_PROMPTS="${SUBTASK_PROMPTS:-${DATASET_ROOT:-/data/2026-challenge-demos}/meta/subtask_segments.parquet}"
export EXP_NAME="${EXP_NAME:-pi05_b1k_subtask_executor_${MODE}_$(date +%Y%m%d_%H%M%S)}"

case "$MODE" in
    check)
        export PREFLIGHT_ONLY=1
        exec bash "$CODE_ROOT/scripts/train_pi05_b1k.sh" full
        ;;
    smoke|full)
        exec bash "$CODE_ROOT/scripts/train_pi05_b1k.sh" "$MODE"
        ;;
    *)
        echo "Usage: $0 [check|smoke|full]" >&2
        exit 2
        ;;
esac
