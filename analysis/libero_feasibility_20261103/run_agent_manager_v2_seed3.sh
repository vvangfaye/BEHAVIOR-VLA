#!/bin/bash
# Agent-managed RL, manager v2 (2026-11-06; v1 pilot crashed with the machine at 00:54 and had two design flaws:
# weights inflated to the 4.0 cap = uniform sampling by iteration 6; best-window regression stats flagged 6-10 tasks per
# iteration). v2: unbiased prev/start-window stats, weights renormalized to mean 1, at most 3 threshold bumps per
# iteration (rlinf/b1k_tools/agent_manager.py MANAGER_VERSION v2). PPO exactly as seed 3 (actor.seed=44, env.train.seed=2,
# 20 iterations, all 10 tasks); seed 3 without manager: 229/500 = 45.8%. Paired 500-episode eval.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy || pgrep -f "^bash run_rsi_ar_local.sh" >/dev/null; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
TAG=ppo_balanced_seed3_agent_v2
LOGD=/mnt/data1/faye/agent_mgr_logs/$TAG   # no spaces: overrides are word-split
TS=/tmp/b1k_ts_${TAG}_$(date +%s)
mkdir -p "$LOGD"
AM="+env.train.task_stats_dir=$TS +env.train.agent_weights_file=$LOGD/task_weights.json +algorithm.agent_manager.enable=True +algorithm.agent_manager.mode=agent +algorithm.agent_manager.out_dir=$LOGD/am +algorithm.agent_manager.seed=0 +algorithm.agent_manager.wait_s=180"
wait_idle
log "START $TAG (agent training manager v2: gpt-6.1-sol sets task sampling weights + lr multiplier every iteration from training-rollout stats; PPO as seed 3: 10 tasks, 20 iters, local)"
t0=$(date +%s)
env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero2.sh $TAG runner.max_epochs=20 actor.seed=44 env.train.seed=2 \
    +actor.prefix_cache=True +actor.prefix_cache_pin=False +actor.prefix_cache_verify_every=64 $AM
log "END   $TAG rc=$? after $(( ($(date +%s) - t0) / 60 )) min; manager decisions: $(wc -l < $LOGD/am/decisions.jsonl 2>/dev/null)"
CK=$(find "$HERE/results/$TAG" -path "*checkpoints/global_step_20/actor/model_state_dict/full_weights.pt" | head -1)
[ -n "$CK" ] || { log "$TAG: no checkpoint"; exit 1; }
wait_idle
log "START eval_${TAG}_s20_c10"
EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10" bash run_eval_ckpt.sh eval_${TAG}_s20_c10 "$CK" all
log "END   eval_${TAG}_s20_c10 rc=$?"
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_seed3_s20_c10/console.log results/eval_${TAG}_s20_c10/console.log results/eval_balanced_s20_c10/console.log 2>&1 | tee results/paired_seed3_agent_v2_c10.txt | tee -a "$QLOG"
python3 paired.py results/eval_balanced_seed3_s20_c10/console.log results/eval_${TAG}_s20_c10/console.log 2>&1 | tail -2 | sed 's/^/agent manager v2 vs seed 3 (same seeds): /' | tee -a "$QLOG"
log "agent-manager v2 seed-3 run finished"
