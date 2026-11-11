#!/bin/bash
# Chain v3 (2026-11-06 morning, after the machine crash): waits for the PPO seed-3 step-15 reference eval, then the
# agent-manager v2 pilot (seed 3), then the seed-3 rule/random manager controls (manager v2 code). Sequential.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
while pgrep -f "run_eval_ckpt.sh eval_balanced_seed3_s15_c10" >/dev/null; do sleep 30; done
bash run_agent_manager_v2_seed3.sh > "$HERE/results/agent_manager_v2_seed3.stdout.txt" 2>&1
log "agent-manager v2 pilot script exited rc=$?"
bash run_manager_controls_seed3.sh > "$HERE/results/manager_controls_seed3.stdout.txt" 2>&1
log "chain v3 finished"
