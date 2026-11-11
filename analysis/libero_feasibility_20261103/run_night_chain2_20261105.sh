#!/bin/bash
# Night chain v2 (2026-11-05 20:4x, after the first video run failed on word-split paths): fixed videos, then the
# agent-manager pilot (seed 3), then the seed-3 rule/random manager controls. Strictly sequential, one RLinf job at a time.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
bash run_videos_libero.sh > "$HERE/results/videos_20261105.stdout.txt" 2>&1
bash run_agent_manager_seed3.sh > "$HERE/results/agent_manager_seed3.stdout.txt" 2>&1
log "agent-manager pilot script exited rc=$?"
bash run_manager_controls_seed3.sh > "$HERE/results/manager_controls_seed3.stdout.txt" 2>&1
log "night chain v2 finished"
