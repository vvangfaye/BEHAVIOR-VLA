#!/bin/bash
# Queue v6 (2026-11-04 ~09:5x). Takes over from run_queue5.sh while its ppo_tokenmean job runs (only the queue's bash
# process is replaced; the running job is untouched). Changes vs v5: the CALVIN steps moved to the cluster (5090:
# calvin_balanced_5090_r3 + calvin_sft_eval_5090, evaluated there with one config), so locally:
#   1. deployment-protocol eval (ODE, chunk 10, 500 init states) of ppo_tokenmean step 20;
#   2. the same eval for ppo_balanced steps 5/10/15 (learning curve under the deployment protocol; step 20 is done);
#   3. a third LIBERO RLinf-PPO (balanced) seed, same local config as ppo_balanced (seed 2 runs on the cluster), + eval.
# Strictly one RLinf job at a time. Logged to results/queue_log.txt.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
ckpt() { find "$HERE/results/$1" -path "*checkpoints/global_step_$2/actor/model_state_dict/full_weights.pt" 2>/dev/null | head -1; }
run_step() {  # run_step <name> <command...>; returns the command's exit code (does not stop the queue)
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

log "queue v6 started (pid $$); waiting for ppo_tokenmean started by queue v5"
wait_idle
export EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10"

# 1. token-mean control, step 20
T=$(ckpt ppo_tokenmean 20)
if [ -n "$T" ]; then run_step eval_libero_tokenmean bash run_eval_ckpt.sh eval_tokenmean_s20_c10 "$T" all
else log "WARNING: ppo_tokenmean has no global_step_20 checkpoint"; fi

# 2. ppo_balanced learning curve under the deployment protocol
for s in 5 10 15; do
    B=$(ckpt ppo_balanced $s)
    [ -n "$B" ] && run_step eval_libero_balanced_s$s bash run_eval_ckpt.sh eval_balanced_s${s}_c10 "$B" all
done
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s5_c10/console.log results/eval_balanced_s10_c10/console.log \
    results/eval_balanced_s15_c10/console.log results/eval_balanced_s20_c10/console.log results/eval_tokenmean_s20_c10/console.log \
    2>&1 | tee results/paired_libero_c10_v6.txt | tee -a "$QLOG"

# 3. third PPO seed (seed 1 = ppo_balanced: actor.seed 42, env.train.seed 0; seed 2 on the cluster: 43/1)
run_step libero_balanced_seed3 "${LIB[@]}" ppo_balanced_seed3 runner.max_epochs=20 actor.seed=44 env.train.seed=2 "${CACHE[@]}"
S3=$(ckpt ppo_balanced_seed3 20)
[ -n "$S3" ] && run_step eval_libero_balanced_seed3 bash run_eval_ckpt.sh eval_balanced_seed3_s20_c10 "$S3" all
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s20_c10/console.log results/eval_balanced_seed3_s20_c10/console.log \
    2>&1 | tee -a results/paired_libero_c10_v6.txt | tee -a "$QLOG"
log "queue v6 finished"
