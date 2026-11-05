#!/bin/bash
# LIBERO-Long comparison videos (2026-11-05; the user asked for videos). Same initial states for every policy: trials 0 and
# 1 of tasks 1, 5 and 7 (task-id filter, trial-major interleave), deployment protocol (chunk 10), 6 episodes per policy:
# SFT, PPO seed 1 (56.2%), PPO seed 3 (45.8%), PPO + subgoal reward seed 3 (step 20). RLinf's RecordVideo tiles the 3 envs of
# each env worker, so every MP4 is one trial with tasks 1/5/7 side by side; make_comparison_videos.py then stacks the
# policies as rows. Runs after the local queue (never two RLinf jobs at once).
set -uo pipefail
# Fix 11/5 20:4x: the first run failed because paths with a space ("Behavior-1k Challenge") were word-split (Hydra
# override parse error for video_base_dir; checkpoint paths cut by `set -- $a`). Videos now go to a space-free dir
# (EVAL_EXTRA_OVERRIDES is word-split by run_eval_ckpt.sh) and arms/checkpoints are parallel arrays; the stacked
# comparison videos are copied to results/videos_20261105/ at the end.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy; do sleep 30; done; sleep 10; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
OUTV=/mnt/data1/faye/videos_20261105     # no spaces
OUT="$HERE/results/videos_20261105"
mkdir -p "$OUTV" "$OUT"
CK=checkpoints/global_step_20/actor/model_state_dict/full_weights.pt
ARMS=(sft ppo_seed1 ppo_seed3 subgoal_seed3)
CKPTS=(none
       "$HERE/results/ppo_balanced/libero_10_ppo_openpi_pi05/$CK"
       "$HERE/results/ppo_balanced_seed3/libero_10_ppo_openpi_pi05/$CK"
       "$HERE/results/ppo_balanced_seed3_subgoal/libero_10_ppo_openpi_pi05/$CK")
wait_idle
log "START videos (tasks 1/5/7 x trials 0/1, chunk 10): ${ARMS[*]}"
for i in "${!ARMS[@]}"; do
    ARM=${ARMS[$i]}; CKPT=${CKPTS[$i]}
    if [ "$CKPT" != "none" ] && [ ! -f "$CKPT" ]; then log "videos: $ARM checkpoint missing ($CKPT)"; continue; fi
    rm -rf "$OUTV/$ARM"
    wait_idle
    EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10 +env.eval.task_id_filter=[1,5,7] env.eval.video_cfg.save_video=True env.eval.video_cfg.video_base_dir=$OUTV/$ARM" \
        bash run_eval_ckpt.sh "video_$ARM" "$CKPT" all 6 1 > "$OUT/$ARM.stdout.txt" 2>&1
    log "videos: $ARM rc=$? ($(grep -a -c '\[libero eval\]' "$HERE/results/video_$ARM/console.log" 2>/dev/null) episodes, $(find "$OUTV/$ARM" -name '*.mp4' 2>/dev/null | wc -l) mp4)"
done
/mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/python "$HERE/make_comparison_videos.py" "$OUTV" "${ARMS[@]}" 2>&1 | tail -5 | tee -a "$QLOG"
cp "$OUTV"/comparison_trial*.mp4 "$OUT"/ 2>/dev/null
log "videos finished: $(ls "$OUT"/comparison_trial*.mp4 2>/dev/null | wc -l) comparison videos in $OUT"
