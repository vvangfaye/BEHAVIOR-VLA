#!/bin/bash
# LIBERO subgoal progress reward (potential-based shaping on BDDL goal predicates, coef 0.5; success reward 1.0),
# 2026-11-05 (user-chosen after ESPO failed). PPO exactly as seed 3 (actor.seed=44, env.train.seed=2, 20 iterations,
# all 10 tasks) plus +env.train.subgoal_reward_coef=0.5. Paired 500-episode eval vs seed 3 (45.8%). Queued after the
# AR v2 seed-3 run (one local RLinf job at a time).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy || pgrep -f "^bash run_rsi_ar_v2_seed3.sh" >/dev/null; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
TAG=ppo_balanced_seed3_subgoal
wait_idle
log "START $TAG (subgoal shaping coef 0.5, PPO as seed 3: 10 tasks, 20 iters, local)"
t0=$(date +%s)
env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero2.sh $TAG runner.max_epochs=20 actor.seed=44 env.train.seed=2 \
    +actor.prefix_cache=True +actor.prefix_cache_pin=False +actor.prefix_cache_verify_every=64 +env.train.subgoal_reward_coef=0.5
log "END   $TAG rc=$? after $(( ($(date +%s) - t0) / 60 )) min"
CK=$(find "$HERE/results/$TAG" -path "*checkpoints/global_step_20/actor/model_state_dict/full_weights.pt" | head -1)
[ -n "$CK" ] || { log "$TAG: no checkpoint"; exit 1; }
wait_idle
log "START eval_${TAG}_s20_c10"
EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10" bash run_eval_ckpt.sh eval_${TAG}_s20_c10 "$CK" all
log "END   eval_${TAG}_s20_c10 rc=$?"
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_seed3_s20_c10/console.log results/eval_${TAG}_s20_c10/console.log 2>&1 | tee results/paired_seed3_subgoal_c10.txt | tee -a "$QLOG"
python3 paired.py results/eval_balanced_seed3_s20_c10/console.log results/eval_${TAG}_s20_c10/console.log 2>&1 | tail -2 | sed 's/^/subgoal vs seed 3 (same seeds): /' | tee -a "$QLOG"
log "subgoal seed-3 run finished"
