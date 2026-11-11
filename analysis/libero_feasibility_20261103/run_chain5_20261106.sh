#!/bin/bash
# Chain 5 (2026-11-06 ~13:55): seed-3 attribution controls for the agent-v2 result (62.6% vs PPO 45.8%), in priority order.
# The rule-manager run (started 13:40 by run_manager_controls_seed3.sh, whose parent bash was stopped so the order can
# change) finishes on its own; this chain then evaluates it, runs the lr-schedule replay control (agent v2's per-iteration
# lr multipliers, uniform task weights), then the random manager, each 20 iters + 500-episode eval, then the paired table.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy || pgrep -f "run_ppo_local_libero2.sh ppo_balanced_seed3_mgr_rule" >/dev/null; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
CKREL=checkpoints/global_step_20/actor/model_state_dict/full_weights.pt
REPLAY_SRC=/mnt/data1/faye/agent_mgr_logs/ppo_balanced_seed3_agent_v2/am/decisions.jsonl
evaluate() {  # $1 = tag
    local TAG=$1 CK
    CK=$(find "$HERE/results/$TAG" -path "*$CKREL" | head -1)
    if [ -z "$CK" ]; then log "$TAG: no checkpoint"; return 1; fi
    wait_idle
    log "START eval_${TAG}_s20_c10"
    EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10" bash run_eval_ckpt.sh eval_${TAG}_s20_c10 "$CK" all
    log "END   eval_${TAG}_s20_c10 rc=$?"
    python3 paired.py results/eval_balanced_seed3_s20_c10/console.log results/eval_${TAG}_s20_c10/console.log 2>&1 | tail -2 | sed "s/^/$TAG vs PPO seed 3 (same seeds): /" | tee -a "$QLOG"
}
train() {  # $1 = tag, rest = manager overrides
    local TAG=$1; shift
    local LOGD=/mnt/data1/faye/agent_mgr_logs/$TAG TS=/tmp/b1k_ts_${TAG}_$(date +%s)
    mkdir -p "$LOGD"
    wait_idle
    log "START $TAG ($*; PPO as seed 3: 10 tasks, 20 iters, local)"
    local t0=$(date +%s)
    env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero2.sh $TAG runner.max_epochs=20 actor.seed=44 env.train.seed=2 \
        +actor.prefix_cache=True +actor.prefix_cache_pin=False +actor.prefix_cache_verify_every=64 \
        +env.train.task_stats_dir=$TS +env.train.agent_weights_file=$LOGD/task_weights.json +algorithm.agent_manager.enable=True \
        +algorithm.agent_manager.out_dir=$LOGD/am +algorithm.agent_manager.seed=0 +algorithm.agent_manager.wait_s=180 "$@"
    log "END   $TAG rc=$? after $(( ($(date +%s) - t0) / 60 )) min; manager decisions: $(wc -l < $LOGD/am/decisions.jsonl 2>/dev/null)"
}
wait_idle
log "rule-manager run (seed 3) finished training (started 13:40 by run_manager_controls_seed3.sh)"
evaluate ppo_balanced_seed3_mgr_rule
train ppo_balanced_seed3_mgr_replay_lr +algorithm.agent_manager.mode=replay +algorithm.agent_manager.replay_from=$REPLAY_SRC +algorithm.agent_manager.replay_lr=True +algorithm.agent_manager.replay_weights=False
evaluate ppo_balanced_seed3_mgr_replay_lr
train ppo_balanced_seed3_mgr_random +algorithm.agent_manager.mode=random
evaluate ppo_balanced_seed3_mgr_random
ARMS=(results/eval_sft_c10/console.log results/eval_balanced_seed3_s20_c10/console.log)
for t in ppo_balanced_seed3_agent_v2 ppo_balanced_seed3_mgr_rule ppo_balanced_seed3_mgr_replay_lr ppo_balanced_seed3_mgr_random ppo_balanced_seed3_subgoal; do
    f=results/eval_${t}_s20_c10/console.log; [ -f "$f" ] && ARMS+=("$f")
done
python3 paired.py "${ARMS[@]}" 2>&1 | tee results/paired_seed3_managers_c10.txt | tail -14 | tee -a "$QLOG"
log "chain 5 finished (paired table: results/paired_seed3_managers_c10.txt)"
