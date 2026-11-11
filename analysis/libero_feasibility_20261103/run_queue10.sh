#!/bin/bash
# Queue v10 (2026-11-04 23:0x): replaces v9's tail. The user found more LIBERO seeds unnecessary: the third seed runs on
# the cluster (libero_seed4_multi, actor.seed=45 / env.train.seed=3), so the local seed-3 run is dropped. v9's bash was
# stopped; its rsi_local driver keeps running and this queue waits for it to exit before using the GPUs.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
ckpt() { find "$HERE/results/$1" -path "*checkpoints/global_step_$2/actor/model_state_dict/full_weights.pt" 2>/dev/null | head -1; }
run_step() {
    local name="$1"; shift
    wait_idle
    log "START $name: $*"
    local t0=$(date +%s)
    "$@"
    local rc=$?
    log "END   $name rc=$rc after $(( ($(date +%s) - t0) / 60 )) min"
    return $rc
}
DRIVER="${1:-}"
log "queue v10 started (pid $$): waiting for the rsi_local driver (pid ${DRIVER:-none}) to finish; then LIBERO learning-curve evals; no local seed 3"
if [ -n "$DRIVER" ]; then while kill -0 "$DRIVER" 2>/dev/null; do sleep 60; done; fi
log "rsi_local driver finished"
export EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10"
for s in 5 10 15; do
    B=$(ckpt ppo_balanced $s)
    [ -n "$B" ] && run_step eval_libero_balanced_s$s bash run_eval_ckpt.sh eval_balanced_s${s}_c10 "$B" all
done
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s5_c10/console.log results/eval_balanced_s10_c10/console.log \
    results/eval_balanced_s15_c10/console.log results/eval_balanced_s20_c10/console.log 2>&1 | tee results/paired_libero_curve_c10.txt | tee -a "$QLOG"
log "queue v10 finished"
