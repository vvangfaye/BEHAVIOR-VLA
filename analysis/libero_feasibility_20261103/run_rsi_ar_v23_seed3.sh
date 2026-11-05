#!/bin/bash
# RSI option 1, AR v2.3 on the full LIBERO-10 setting (2026-11-05 11:3x; v2.3 = v2.1 + re-measurement of the restored anchor, opt-in remeasure=True): PPO exactly as seed 3 (actor.seed=44,
# env.train.seed=2, config lr, 20 iterations, all 10 tasks) plus verified accept/rollback with the best-anchor reference
# (algorithm.accept_rollback.reference=best). Seed 3 without AR: 229/500 = 45.8% (learned task 5 49/50 but regressed
# tasks 1, 2, 7, 9). Question: does the rollback rule keep the task-5 gain without the regressions? Paired 500-episode eval.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy || pgrep -f "^bash run_rsi_ar_local.sh" >/dev/null; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
TAG=ppo_balanced_seed3_arv23
LOGD=/mnt/data1/faye/rsi_ar_logs/$TAG   # no spaces: overrides are word-split
TS=/tmp/b1k_ts_${TAG}_$(date +%s)
mkdir -p "$LOGD"
AR="+algorithm.accept_rollback.enable=True +algorithm.accept_rollback.reference=best +algorithm.accept_rollback.delta_task=0.15 +algorithm.accept_rollback.delta_mean=0.08 +algorithm.accept_rollback.z=1.64 +algorithm.accept_rollback.alpha=0.2 +algorithm.accept_rollback.remeasure=True +algorithm.accept_rollback.min_n=8 +algorithm.accept_rollback.lr_decay=0.5 +algorithm.accept_rollback.wait_s=180 +algorithm.accept_rollback.log_path=$LOGD/ar_decisions.jsonl +env.train.task_stats_dir=$TS"
wait_idle
log "START $TAG (AR v2.3 = v2.1 + post-rollback re-measurement pooled into the anchor (remeasure=True) + v2.2 memory fix, PPO as seed 3: 10 tasks, 20 iters, local)"
t0=$(date +%s)
env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero2.sh $TAG runner.max_epochs=20 actor.seed=44 env.train.seed=2 \
    +actor.prefix_cache=True +actor.prefix_cache_pin=False +actor.prefix_cache_verify_every=64 $AR
log "END   $TAG rc=$? after $(( ($(date +%s) - t0) / 60 )) min; AR: $(python3 -c "import json; d=[json.loads(l) for l in open('$LOGD/ar_decisions.jsonl')]; print(len(d), 'checks,', sum(1 for x in d if not x['ar/accepted']), 'rollbacks, final lr', d[-1]['ar/lr'])" 2>/dev/null)"
CK=$(find "$HERE/results/$TAG" -path "*checkpoints/global_step_20/actor/model_state_dict/full_weights.pt" | head -1)
[ -n "$CK" ] || { log "$TAG: no checkpoint"; exit 1; }
wait_idle
log "START eval_${TAG}_s20_c10"
EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10" bash run_eval_ckpt.sh eval_${TAG}_s20_c10 "$CK" all
log "END   eval_${TAG}_s20_c10 rc=$?"
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_seed3_s20_c10/console.log results/eval_${TAG}_s20_c10/console.log results/eval_balanced_s20_c10/console.log 2>&1 | tee results/paired_seed3_arv23_c10.txt | tee -a "$QLOG"
python3 paired.py results/eval_balanced_seed3_s20_c10/console.log results/eval_${TAG}_s20_c10/console.log 2>&1 | tail -2 | sed 's/^/AR v2.3 vs seed 3 (same seeds): /' | tee -a "$QLOG"
log "RSI-AR v2.3 seed-3 run finished"
