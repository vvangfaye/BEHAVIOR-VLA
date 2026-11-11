#!/bin/bash
# Queue v8 (2026-11-04 ~15:0x): replaces v7 after its RSI decision (PPO loop runs locally). Restarts the local RSI
# driver with the external in-loop verifier (gpt-6.1-sol via Codex CLI, <=500 calls/day; user-approved): each
# source-fold candidate gets a 30-episode frame-dump eval judged by the verifier, which selects Phi_k.
# The driver resumes: the in-flight run (rsil_ppo_r1_c0, orphaned on purpose) is waited for, not repeated.
# Then v6 leftovers: ppo_balanced s5/s10/s15 deployment evals, third PPO seed + eval.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
RSI="$HERE/../rsi_20261104"
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

log "queue v8 started (pid $$): restarting rsi_local PPO loop with the gpt-6.1-sol verifier"
python3 "$RSI/rsi_local.py" run --base ppo --state rsi_state_ppo_local.json >> "$RSI/rsi_local_ppo.log" 2>&1
log "END   rsi_local PPO loop rc=$?"

export EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10"
for s in 5 10 15; do
    B=$(ckpt ppo_balanced $s)
    [ -n "$B" ] && run_step eval_libero_balanced_s$s bash run_eval_ckpt.sh eval_balanced_s${s}_c10 "$B" all
done
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s5_c10/console.log results/eval_balanced_s10_c10/console.log \
    results/eval_balanced_s15_c10/console.log results/eval_balanced_s20_c10/console.log 2>&1 | tee results/paired_libero_curve_c10.txt | tee -a "$QLOG"
run_step libero_balanced_seed3 "${LIB[@]}" ppo_balanced_seed3 runner.max_epochs=20 actor.seed=44 env.train.seed=2 "${CACHE[@]}"
S3=$(ckpt ppo_balanced_seed3 20)
[ -n "$S3" ] && run_step eval_libero_balanced_seed3 bash run_eval_ckpt.sh eval_balanced_seed3_s20_c10 "$S3" all
log "queue v8 finished"
