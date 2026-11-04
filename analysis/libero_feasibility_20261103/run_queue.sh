#!/bin/bash
# Sequential local GPU queue for the CVPR RL validation (EXPERIMENT_PLAN.md), started 2026-11-04.
# Strictly one RLinf job at a time (RLinf's ray.init(address='auto') attaches to any running Ray cluster).
# Stops at the first failed step; every step is logged to results/queue_log.txt.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
# After a job exits, clear any leftover Ray head of OURS (only this user runs Ray on this host, checked 2026-11-04)
# so the next job cannot attach to a stale cluster through address="auto".
wait_idle() { while busy; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
ckpt() { find "$HERE/results/$1" -path "*checkpoints/global_step_$2/actor/model_state_dict/full_weights.pt" 2>/dev/null | head -1; }
require_ckpt() {
    local c; c=$(ckpt "$1" "$2")
    if [ -z "$c" ]; then log "STOP: $1 has no global_step_$2 checkpoint"; exit 1; fi
    echo "$c"
}
step() {  # step <name> <command...>
    local name="$1"; shift
    wait_idle
    log "START $name: $*"
    local t0=$(date +%s)
    "$@"
    local rc=$?
    log "END   $name rc=$rc after $(( ($(date +%s) - t0) / 60 )) min"
    if [ $rc -ne 0 ]; then log "STOP: $name failed"; exit 1; fi
}
CACHE=(+actor.prefix_cache=True +actor.prefix_cache_pin=False +actor.prefix_cache_verify_every=64)

log "queue started (pid $$)"
wait_idle
require_ckpt ppo_balanced 20 >/dev/null

# LIBERO-10: token-mean control, same budget as ppo_balanced.
step libero_tokenmean env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero.sh ppo_tokenmean \
    runner.max_epochs=20 "${CACHE[@]}" +algorithm.length_normalization=False
B=$(require_ckpt ppo_balanced 20); T=$(require_ckpt ppo_tokenmean 20)
step eval_libero_balanced bash run_eval_ckpt.sh eval_balanced_s20 "$B" all
step eval_libero_tokenmean bash run_eval_ckpt.sh eval_tokenmean_s20 "$T" all
python3 paired.py results/full500/console.log results/eval_balanced_s20/console.log results/eval_tokenmean_s20/console.log \
    | tee results/paired_libero_s20.txt | tee -a "$QLOG"
python3 paired.py results/eval_tokenmean_s20/console.log results/eval_balanced_s20/console.log | tee -a "$QLOG"

# CALVIN ABC->D: SFT reference eval (also checks the env + per-chain logging), then the two arms.
step eval_calvin_sft bash run_calvin_eval_ckpt.sh calvin_eval_sft none
CCACHE=("${CACHE[@]}" +actor.prefix_cache_max_gib=100)
step calvin_balanced env CONFIG=calvin_abc_d_ppo_pi05_local2gpu_all bash run_ppo_local_calvin.sh calvin_balanced \
    runner.max_epochs=20 "${CCACHE[@]}"
step calvin_tokenmean env CONFIG=calvin_abc_d_ppo_pi05_local2gpu_all bash run_ppo_local_calvin.sh calvin_tokenmean \
    runner.max_epochs=20 "${CCACHE[@]}" +algorithm.length_normalization=False
CB=$(require_ckpt calvin_balanced 20); CT=$(require_ckpt calvin_tokenmean 20)
step eval_calvin_balanced bash run_calvin_eval_ckpt.sh calvin_eval_balanced_s20 "$CB"
step eval_calvin_tokenmean bash run_calvin_eval_ckpt.sh calvin_eval_tokenmean_s20 "$CT"
python3 calvin_paired.py results/calvin_eval_sft/console.log results/calvin_eval_balanced_s20/console.log \
    results/calvin_eval_tokenmean_s20/console.log | tee results/paired_calvin_s20.txt | tee -a "$QLOG"
log "queue finished"
