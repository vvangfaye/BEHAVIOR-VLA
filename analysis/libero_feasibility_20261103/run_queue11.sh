#!/bin/bash
# Queue v11 (2026-11-05 00:5x): user decision after the negative L4: stop the PPO RSI round 2 (L5 near-certainly null,
# its candidates are all small step sizes around the failed Phi_1); the local GPUs run the LIBERO learning-curve evals and
# the third LIBERO seed (actor.seed=44, env.train.seed=2) so it lands before the 15:00 report; the queued Spot seed job
# (libero_seed4_multi) is cancelled, so LIBERO keeps 3 seeds in total.
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
CACHE=(+actor.prefix_cache=True +actor.prefix_cache_pin=False +actor.prefix_cache_verify_every=64)
LIB=(env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero2.sh)
log "queue v11 started (pid $$): LIBERO learning-curve evals, then the third LIBERO seed (local)"
export EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10"
for s in 5 10 15; do
    B=$(ckpt ppo_balanced $s)
    [ -n "$B" ] && run_step eval_libero_balanced_s$s bash run_eval_ckpt.sh eval_balanced_s${s}_c10 "$B" all
done
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s5_c10/console.log results/eval_balanced_s10_c10/console.log \
    results/eval_balanced_s15_c10/console.log results/eval_balanced_s20_c10/console.log 2>&1 | tee results/paired_libero_curve_c10.txt | tee -a "$QLOG"
unset EVAL_EXTRA_OVERRIDES
run_step libero_balanced_seed3 "${LIB[@]}" ppo_balanced_seed3 runner.max_epochs=20 actor.seed=44 env.train.seed=2 "${CACHE[@]}"
export EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10"
S3=$(ckpt ppo_balanced_seed3 20)
[ -n "$S3" ] && run_step eval_libero_balanced_seed3 bash run_eval_ckpt.sh eval_balanced_seed3_s20_c10 "$S3" all
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s20_c10/console.log results/eval_balanced_seed3_s20_c10/console.log 2>&1 | tee results/paired_libero_seeds_c10.txt | tee -a "$QLOG"
log "queue v11 finished"
