#!/usr/bin/env bash
# Explicit local preparation/submission; never imports full100 or L20 defaults.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${NEBULA_SUBMIT_PYTHON:-python3}" "$SCRIPT_DIR/subtask_mainline.py" "$@"
