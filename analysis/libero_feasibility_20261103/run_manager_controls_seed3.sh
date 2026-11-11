#!/bin/bash
# Manager controls on seed 3 (local, 11/6 early morning): the rule manager, then the random manager, each exactly like
# the agent pilot (run_agent_manager_seed3.sh: PPO as seed 3 = actor.seed 44 / env.train.seed 2, config lr, 20 iters,
# all 10 tasks) except algorithm.agent_manager.mode. Same seed, same hardware as the agent arm and as PPO seed 3 without
# a manager (45.8%) -> clean 4-arm comparison. Waits for run_night_chain_20261105.sh (never two RLinf jobs at once).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy || pgrep -f "run_night_chain_20261105.sh|run_agent_manager_seed3.sh" >/dev/null; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
CKREL=checkpoints/global_step_20/actor/model_state_dict/full_weights.pt
for MODE in rule random; do
    TAG=ppo_balanced_seed3_mgr_$MODE
    LOGD=/mnt/data1/faye/agent_mgr_logs/$TAG   # no spaces: overrides are word-split
    TS=/tmp/b1k_ts_${TAG}_$(date +%s)
    mkdir -p "$LOGD"
    AM="+env.train.task_stats_dir=$TS +env.train.agent_weights_file=$LOGD/task_weights.json +algorithm.agent_manager.enable=True +algorithm.agent_manager.mode=$MODE +algorithm.agent_manager.out_dir=$LOGD/am +algorithm.agent_manager.seed=0 +algorithm.agent_manager.wait_s=180"
    wait_idle
    log "START $TAG (manager control: $MODE; PPO as seed 3: 10 tasks, 20 iters, local)"
    t0=$(date +%s)
    env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero2.sh $TAG runner.max_epochs=20 actor.seed=44 env.train.seed=2 \
        +actor.prefix_cache=True +actor.prefix_cache_pin=False +actor.prefix_cache_verify_every=64 $AM
    log "END   $TAG rc=$? after $(( ($(date +%s) - t0) / 60 )) min; manager decisions: $(wc -l < $LOGD/am/decisions.jsonl 2>/dev/null)"
    CK=$(find "$HERE/results/$TAG" -path "*$CKREL" | head -1)
    if [ -z "$CK" ]; then log "$TAG: no checkpoint"; continue; fi
    wait_idle
    log "START eval_${TAG}_s20_c10"
    EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10" bash run_eval_ckpt.sh eval_${TAG}_s20_c10 "$CK" all
    log "END   eval_${TAG}_s20_c10 rc=$?"
    python3 paired.py results/eval_balanced_seed3_s20_c10/console.log results/eval_${TAG}_s20_c10/console.log 2>&1 | tail -2 | sed "s/^/$MODE manager vs seed 3 (same seeds): /" | tee -a "$QLOG"
done
ARMS=(results/eval_sft_c10/console.log results/eval_balanced_seed3_s20_c10/console.log)
for t in ppo_balanced_seed3_agent_v2 ppo_balanced_seed3_mgr_rule ppo_balanced_seed3_mgr_random ppo_balanced_seed3_subgoal; do
    f=results/eval_${t}_s20_c10/console.log; [ -f "$f" ] && ARMS+=("$f")
done
python3 paired.py "${ARMS[@]}" 2>&1 | tee results/paired_seed3_managers_c10.txt | tail -12 | tee -a "$QLOG"
log "seed-3 manager controls finished (paired table: results/paired_seed3_managers_c10.txt)"
