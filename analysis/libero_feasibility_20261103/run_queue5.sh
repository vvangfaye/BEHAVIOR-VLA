#!/bin/bash
# Queue v5 (2026-11-04 07:1x): as v4, but LIBERO evals use replan horizon 10 (piRL Appendix J, H'=10 for
# LIBERO-Long; ESPO v2 warmup rollouts with the SFT policy, ODE + chunk 10, gave 45.3% vs 27.8% under RLinf's chunk-5
# eval config), plus a chunk-10 SFT reference. Takes over while espo_v2 runs. Earlier: Queue v4 (2026-11-04 06:4x). ESPO v1 (Flow-SDE rollouts, noise 0.5) dropped to 12.5% training success after one
# update: its rollouts came from the SDE sampler while ESPO optimizes the ODE (deployment) sampler. ESPO v2 sets
# actor.model.openpi.noise_level=0.0, so rollouts, model samples and evaluation all use the same native ODE sampler
# (as in the toy and the registered method). Earlier header: Queue v3 (2026-11-04 05:3x), took over from run_queue2.sh.
# Change vs v2: launchers run_ppo_local_*2.sh set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (ESPO peaked at
# ~47/48 GB per GPU in the smoke test). Previously: Queue v2 (2026-11-04 02:2x), replaced run_queue.sh before it started any job. Reordered so the registered CVPR
# method (ESPO, loss_type=espo) is compared with RLinf-PPO on LIBERO-10 at the same interaction budget first; the
# token-mean control and CALVIN follow. Strictly one RLinf job at a time. Logged to results/queue_log.txt.
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
ESPO=(algorithm.loss_type=espo +algorithm.espo.model_samples=2 +algorithm.espo.eta=3.0 +algorithm.espo.coef=1.0
      +algorithm.espo.stop_frac=0.5 actor.model.openpi.noise_level=0.0 actor.optim.lr=1e-6 actor.optim.critic_warmup_steps=12)
LIB=(env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero2.sh)
CAL=(env CONFIG=calvin_abc_d_ppo_pi05_local2gpu_all bash run_ppo_local_calvin2.sh)

log "queue v5 started (pid $$); waiting for espo_v2 started by queue v4"
wait_idle
B=$(ckpt ppo_balanced 20)
[ -n "$B" ] || log "WARNING: ppo_balanced has no global_step_20 checkpoint"

# 1. ESPO on LIBERO-10: 2-iteration smoke test, then the full 20-iteration arm (same budget as ppo_balanced).
ESPO_OK=1  # code path validated by espo_smoke (2 iterations) and the first iterations of espo_v1
wait_idle  # espo_v2 was launched by queue v4

# 2. LIBERO evals of the step-20 checkpoints (500 init states each, both GPUs).
export EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10"
run_step eval_libero_sft_c10 bash run_eval_ckpt.sh eval_sft_c10 none all
[ -n "$B" ] && run_step eval_libero_balanced bash run_eval_ckpt.sh eval_balanced_s20_c10 "$B" all
E=$(ckpt espo_v2 20)
[ -n "$E" ] && run_step eval_libero_espo bash run_eval_ckpt.sh eval_espo_s20_c10 "$E" all
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s20_c10/console.log results/eval_espo_s20_c10/console.log \
    2>&1 | tee results/paired_libero_s20_c10.txt | tee -a "$QLOG"

# 3. Token-mean control on LIBERO-10 and its eval.
run_step libero_tokenmean "${LIB[@]}" ppo_tokenmean runner.max_epochs=20 "${CACHE[@]}" +algorithm.length_normalization=False
T=$(ckpt ppo_tokenmean 20)
[ -n "$T" ] && run_step eval_libero_tokenmean bash run_eval_ckpt.sh eval_tokenmean_s20_c10 "$T" all
unset EVAL_EXTRA_OVERRIDES
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s20_c10/console.log results/eval_espo_s20_c10/console.log \
    results/eval_tokenmean_s20_c10/console.log 2>&1 | tee results/paired_libero_s20_c10.txt | tee -a "$QLOG"

# 4. CALVIN ABC->D: SFT reference eval, RLinf-PPO (balanced) arm, ESPO arm, evals.
CCACHE=("${CACHE[@]}" +actor.prefix_cache_max_gib=100)
run_step eval_calvin_sft bash run_calvin_eval_ckpt.sh calvin_eval_sft none
run_step calvin_balanced "${CAL[@]}" calvin_balanced runner.max_epochs=20 "${CCACHE[@]}"
[ "$ESPO_OK" = 1 ] && run_step calvin_espo "${CAL[@]}" calvin_espo runner.max_epochs=20 "${CCACHE[@]}" "${ESPO[@]}"
CB=$(ckpt calvin_balanced 20); CE=$(ckpt calvin_espo 20)
[ -n "$CB" ] && run_step eval_calvin_balanced bash run_calvin_eval_ckpt.sh calvin_eval_balanced_s20 "$CB"
[ -n "$CE" ] && run_step eval_calvin_espo bash run_calvin_eval_ckpt.sh calvin_eval_espo_s20 "$CE"
python3 calvin_paired.py results/calvin_eval_sft/console.log results/calvin_eval_balanced_s20/console.log \
    results/calvin_eval_espo_s20/console.log 2>&1 | tee results/paired_calvin_s20.txt | tee -a "$QLOG"
log "queue v5 finished"
