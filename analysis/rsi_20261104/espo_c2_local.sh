#!/bin/bash
# Local ESPO step-size test (2026-11-04 ~16:5x), replacing the cluster candidate rsi_espo_v3_r1_c2 that was stopped
# after the platform notice (cluster submissions frozen). Same design as the ESPO RSI loop's round-1 candidate c2:
# ESPO v3 (signed advantages, prox 1.0, M=4, eta 3, ODE rollouts, critic warmup 12) with lr 1.2e-5 (4x Phi_0) on the
# source fold S1 = LIBERO-10 tasks [0,1,7], 8 iterations x 128 episodes, seeds 42/0; then the 48-episode verifier
# eval with frame dump and gpt-6.1-sol judging (sol_ver.py), as for the PPO loop's candidates.
# Runs while HOLD_LOCAL keeps the local RSI PPO driver waiting; removes HOLD_LOCAL on exit (success or failure).
set -uo pipefail
RSI="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LF="$RSI/../libero_feasibility_20261103"
QLOG="$LF/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
trap 'rm -f "$RSI/HOLD_LOCAL"; log "espo_c2_local: released HOLD_LOCAL"' EXIT
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
TAG=rsil_espo_r1_c2
OV=$(python3 - << 'EOF'
import importlib.util
spec = importlib.util.spec_from_file_location("R", "/mnt/data1/faye/code/Behavior-1k Challenge/analysis/rsi_20261104/rsi_loop.py")
R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
print(" ".join(R.train_overrides({"base": "espo", "espo_variant": "v3", "iters": 8, "knob": "actor.optim.lr"}, "S1", 1.2e-5)))
EOF
)
log "espo_c2_local: waiting for the GPUs (PPO r1_c1 in flight); overrides: $OV"
wait_idle
cd "$LF"
log "START $TAG (local ESPO v3, lr 1.2e-5, S1, 8 iters)"
t0=$(date +%s)
env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero2.sh $TAG $OV
log "END   $TAG rc=$? after $(( ($(date +%s) - t0) / 60 )) min"
CK=$(find "$LF/results/$TAG" -path "*checkpoints/global_step_8/actor/model_state_dict/full_weights.pt" | head -1)
if [ -z "$CK" ]; then log "espo_c2_local: no checkpoint, skipping verifier"; exit 1; fi
wait_idle
FR=/mnt/data1/faye/rsi_frames/${TAG}_ver
log "START ${TAG}_ver (48 episodes on S1, frame dump)"
EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10 +env.eval.task_id_filter=[0,1,7] +env.eval.dump_frames_dir=$FR +env.eval.dump_frames_every=70" \
    bash run_eval_ckpt.sh ${TAG}_ver "$CK" all 24 2
log "END   ${TAG}_ver rc=$?"
python3 "$RSI/sol_ver.py" judge "$FR" "$LF/results/${TAG}_ver/sol" --workers 6 > "$LF/results/${TAG}_ver/sol_summary.txt" 2>&1
log "espo_c2_local verifier: $(tail -1 "$LF/results/${TAG}_ver/sol_summary.txt" | cut -c1-200)"
