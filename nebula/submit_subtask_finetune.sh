#!/usr/bin/env bash
set -euo pipefail
set +x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_QUEUE_REQUEST="${TRAIN_QUEUE:-${QUEUE:-demand_specific_na175_h20}}"
TRAIN_CARD_REQUEST="${TRAIN_GPU_CARD_TYPE:-${GPU_CARD_TYPE:-h20}}"
# Only this launcher sources credentials; never copy or display config.sh.
source "$SCRIPT_DIR/config.sh"
export TRAIN_QUEUE="$TRAIN_QUEUE_REQUEST" TRAIN_GPU_CARD_TYPE="$TRAIN_CARD_REQUEST"
export AI_HUB_TOKEN AI_HUB_NAMESPACE OSS_ACCESS_ID OSS_ACCESS_KEY OSS_BUCKET OSS_ENDPOINT
for setting in QUEUE GPU_CARD_TYPE ALGO_NAME AI_HUB_CLI AIHUB_MODEL_NAME; do
    if [[ -n "${!setting:-}" ]]; then export "$setting"; fi
done
exec "${NEBULA_SUBMIT_PYTHON:-python3}" "$SCRIPT_DIR/subtask_finetune.py" "$@"
