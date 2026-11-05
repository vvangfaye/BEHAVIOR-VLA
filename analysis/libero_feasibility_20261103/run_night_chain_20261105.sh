#!/bin/bash
# Night chain 2026-11-05 (local machine, one RLinf job at a time). Waits for run_videos_libero.sh (which itself waits for
# the subgoal seed-3 chain), then:
#  1. capability tests 2/3 (subgoal judgment + failure localization): SFT eval, 100 episodes (tasks 0-9 x trials 0-9,
#     chunk 10), third-person frames + BDDL goal-predicate truth every 70 steps and at the end
#     (+env.eval.dump_goal_preds, goal_preds_patch.py); gpt-6.1-sol judges every subgoal from 6 frames
#     (analysis/rsi_20261104/sol_subgoal_judge.py, ~100 calls under the daily ledger);
#  2. the agent training-manager pilot (run_agent_manager_seed3.sh: PPO as seed 3 + Sol manager, 20 iters, 500-ep eval).
# A failure in step 1 is logged and does not block step 2.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
RSI="/mnt/data1/faye/code/Behavior-1k Challenge/analysis/rsi_20261104"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy || pgrep -f "run_subgoal_seed3.sh|run_videos_libero.sh" >/dev/null; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
FR=/mnt/data1/faye/rsi_frames/capability_subgoal_sft_c10   # no spaces: EVAL_EXTRA_OVERRIDES is word-split
OUT="$RSI/agent_capability_20261105_subgoal"
TAG=capability_subgoal_sft
if [ -e "$FR" ]; then FR="${FR}_$(date +%H%M)"; fi   # never mix two dumps in one directory

wait_idle
log "START $TAG (capability tests 2/3: SFT, 100 episodes, chunk 10, frames + goal predicates every 70 steps -> $FR)"
EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10 +env.eval.dump_frames_dir=$FR +env.eval.dump_frames_every=70 +env.eval.dump_goal_preds=True" \
    bash run_eval_ckpt.sh "$TAG" none all 50 2 > "$HERE/results/$TAG.stdout.txt" 2>&1
rc=$?
neps=$(find "$FR" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l)
njs=$(find "$FR" -name 'step*.json' 2>/dev/null | wc -l)
log "END   $TAG rc=$rc ($(grep -a -c '\[libero eval\]' "$HERE/results/$TAG/console.log" 2>/dev/null) episodes in console, $neps frame dirs, $njs predicate files)"
if [ "$neps" -gt 0 ] && [ "$njs" -gt 0 ]; then
    python3 "$RSI/sol_subgoal_judge.py" "$FR" "$OUT" --workers 8 --console "$HERE/results/$TAG/console.log" > "$OUT.stdout.txt" 2>&1
    jrc=$?
    log "capability 2/3 subgoal judge finished rc=$jrc: $(python3 -c '
import json, sys
r = json.load(open(sys.argv[1]))
keys = ("episodes", "subgoal_instances", "subgoal_accuracy", "subgoal_kappa", "precision(holds)", "recall(holds)",
        "exact_localization_on_partial", "exact_localization_on_failures", "baseline_localization_on_failures(task mode, no images)",
        "progress_vs_true_fraction_pearson", "success_agreement")
print("; ".join(f"{k} {r.get(k)}" for k in keys))' "$OUT/report.json" 2>&1 | tail -1)"
else
    log "capability 2/3: no frames/predicates dumped (see results/$TAG.stdout.txt); skipped the judge"
fi

wait_idle
bash run_agent_manager_seed3.sh > "$HERE/results/agent_manager_seed3.stdout.txt" 2>&1
log "night chain finished (agent-manager pilot rc=$?)"
