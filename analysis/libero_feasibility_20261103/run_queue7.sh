#!/bin/bash
# Queue v7 (2026-11-04 ~13:2x), takes over from run_queue6.sh while ppo_tokenmean runs (only the queue bash is replaced).
# 1. deployment-protocol eval of ppo_tokenmean step 20 (as in v6);
# 2. Physical-RSI PPO loop: the cluster copy (rsi_state_ppo.json) has waited >90 min in the 5090 Spot queue. If none of
#    its jobs is RUNNING when the local GPUs free up, cancel the cluster PPO loop jobs and run the same loop locally and
#    sequentially (analysis/rsi_20261104/rsi_local.py, all PPO-loop runs on this machine: no hardware confound).
#    Otherwise keep the cluster loop and continue with v6's remaining steps.
# 3. v6 leftovers: ppo_balanced s5/s10/s15 evals, third PPO seed + eval.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
RSI="$HERE/../rsi_20261104"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
ckpt() { find "$HERE/results/$1" -path "*checkpoints/global_step_$2/actor/model_state_dict/full_weights.pt" 2>/dev/null | head -1; }
run_step() {
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

log "queue v7 started (pid $$); waiting for ppo_tokenmean (started by queue v5)"
wait_idle
log "END   libero_tokenmean (detected by queue v7)"
export EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10"
T=$(ckpt ppo_tokenmean 20)
if [ -n "$T" ]; then run_step eval_libero_tokenmean bash run_eval_ckpt.sh eval_tokenmean_s20_c10 "$T" all
else log "WARNING: ppo_tokenmean has no global_step_20 checkpoint"; fi
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s20_c10/console.log results/eval_tokenmean_s20_c10/console.log \
    2>&1 | tee results/paired_libero_tokenmean_c10.txt | tee -a "$QLOG"
unset EVAL_EXTRA_OVERRIDES

# 2. decide where the PPO RSI loop runs
RUNNING=$(python3 - << 'EOF'
import json, subprocess
st = json.load(open("/mnt/data1/faye/code/Behavior-1k Challenge/analysis/rsi_20261104/rsi_state_ppo.json"))
runs = [v["train"] for v in st["ee"].values()] + [c for r in st["rounds"].values() for c in r["cands"]]
n = 0
for r in runs:
    name = json.load(open(f"/mnt/data1/faye/code/Behavior-1k Challenge/analysis/cluster_probe_20261104/runs/{r['tag']}.json"))["name"]
    out = subprocess.run(["/home/faye/.local/bin/sco", "acp", "jobs", "describe", name, "--workspace-name=share-space", "-o", "json"],
                         capture_output=True, text=True, timeout=60).stdout
    try:
        s = json.loads(out).get("state")
        s = s.get("state") if isinstance(s, dict) else s
    except Exception:
        s = "?"
    n += s in ("RUNNING", "SUCCEEDED")
print(n)
EOF
)
if [ "${RUNNING:-0}" = "0" ]; then
    log "RSI PPO loop: no cluster job running -> cancel the cluster PPO loop and run it locally (rsi_local.py)"
    touch "$RSI/rsi_state_ppo.json.disabled"; sleep 360   # let any in-flight 'step' of the cluster loop finish first
    for N in b1k-rl-rsi-ppo-ee-phi0 b1k-rl-rsi-ppo-r1-c0 b1k-rl-rsi-ppo-r1-c1 b1k-rl-rsi-ppo-r1-c2; do
        timeout 90 /home/faye/.local/bin/sco acp jobs stop --workspace-name=share-space "$N" 2>&1 | tail -1 | tee -a "$QLOG"
    done
    cp "$RSI/rsi_state_ppo.json" "$RSI/rsi_state_ppo.cancelled_$(date +%H%M).json"
    log "START rsi_local PPO loop"
    python3 "$RSI/rsi_local.py" run --base ppo --state rsi_state_ppo_local.json >> "$RSI/rsi_local_ppo.log" 2>&1
    log "END   rsi_local PPO loop rc=$?"
else
    log "RSI PPO loop: $RUNNING cluster job(s) running -> keep it on the cluster"
fi

# 3. v6 leftovers
export EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10"
for s in 5 10 15; do
    B=$(ckpt ppo_balanced $s)
    [ -n "$B" ] && run_step eval_libero_balanced_s$s bash run_eval_ckpt.sh eval_balanced_s${s}_c10 "$B" all
done
python3 paired.py results/eval_sft_c10/console.log results/eval_balanced_s5_c10/console.log results/eval_balanced_s10_c10/console.log \
    results/eval_balanced_s15_c10/console.log results/eval_balanced_s20_c10/console.log 2>&1 | tee results/paired_libero_curve_c10.txt | tee -a "$QLOG"
run_step libero_balanced_seed3 "${LIB[@]}" ppo_balanced_seed3 runner.max_epochs=20 actor.seed=44 env.train.seed=2 "${CACHE[@]}"
S3=$(ckpt ppo_balanced_seed3 20)
[ -n "$S3" ] && run_step eval_libero_balanced_seed3 bash run_eval_ckpt.sh eval_balanced_seed3_s20_c10 "$S3" all
log "queue v7 finished"
