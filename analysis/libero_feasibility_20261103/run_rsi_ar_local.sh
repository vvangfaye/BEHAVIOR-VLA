#!/bin/bash
# RSI option 1 pilot, local arm (2026-11-05 06:3x, user-approved): verified accept/rollback (per-task regression guard,
# delta_task 0.15, delta_mean 0.08, z 1.64, lr x0.5 on rollback) with 4x the base step size (lr 2e-5) on the held-out EE
# fold [2,3,5,9], 8 iterations, same seeds as the RSI runs (42/0); then the 200-episode EE eval. Compared on the same
# platform with EE-Phi0 (lr 5e-6, no AR: 117/200) and EE-Phi1 (1.25e-6: 98/200); SFT 97/200.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
QLOG="$HERE/results/queue_log.txt"
log() { echo "$(date '+%F %T') $*" | tee -a "$QLOG"; }
busy() { ps -eo cmd | grep -q -E "^python (examples/embodiment/train_embodied_agent.py|evaluations/eval_embodied_agent.py)"; }
wait_idle() { while busy || pgrep -f "^bash run_queue11.sh" >/dev/null; do sleep 60; done; sleep 20; /mnt/data1/faye/code/libero_rlinf_20261103/RLinf/.venv/bin/ray stop --force >/dev/null 2>&1; sleep 5; }
TAG=rsil_ppo_ee_ar
TS=/tmp/b1k_ts_${TAG}_$(date +%s)
mkdir -p results/$TAG /mnt/data1/faye/rsi_ar_logs/$TAG  # no spaces: overrides are word-split
AR="+algorithm.accept_rollback.enable=True +algorithm.accept_rollback.delta_task=0.15 +algorithm.accept_rollback.delta_mean=0.08 +algorithm.accept_rollback.z=1.64 +algorithm.accept_rollback.min_n=8 +algorithm.accept_rollback.lr_decay=0.5 +algorithm.accept_rollback.wait_s=180 +algorithm.accept_rollback.log_path=/mnt/data1/faye/rsi_ar_logs/$TAG/ar_decisions.jsonl +env.train.task_stats_dir=$TS"
wait_idle
log "START $TAG (RSI option 1 pilot: EE fold, lr 2e-5 + verified accept/rollback, 8 iters, local)"
t0=$(date +%s)
env CONFIG=libero_10_ppo_pi05_local2gpu_all bash run_ppo_local_libero2.sh $TAG +actor.prefix_cache=True +actor.prefix_cache_pin=False +actor.prefix_cache_verify_every=64 actor.seed=42 env.train.seed=0 runner.max_epochs=8 runner.save_interval=8 +env.train.task_id_filter=[2,3,5,9] actor.optim.lr=2e-05 $AR
log "END   $TAG rc=$? after $(( ($(date +%s) - t0) / 60 )) min; AR decisions: $(python3 -c "import json; d=[json.loads(l) for l in open('/mnt/data1/faye/rsi_ar_logs/$TAG/ar_decisions.jsonl')]; print(len(d), 'checks,', sum(1 for x in d if not x['ar/accepted']), 'rollbacks, final lr', d[-1]['ar/lr'])" 2>/dev/null)"
CK=$(find "$HERE/results/$TAG" -path "*checkpoints/global_step_8/actor/model_state_dict/full_weights.pt" | head -1)
[ -n "$CK" ] || { log "$TAG: no checkpoint"; exit 1; }
wait_idle
log "START ${TAG}_eval (EE fold, 200 episodes)"
EVAL_EXTRA_OVERRIDES="rollout.model.num_action_chunks=10 +env.eval.task_id_filter=[2,3,5,9]" bash run_eval_ckpt.sh ${TAG}_eval "$CK" all 20 10
log "END   ${TAG}_eval rc=$?"
python3 - << 'PY' 2>&1 | tee -a "$QLOG"
import importlib.util
from math import comb
from pathlib import Path
spec = importlib.util.spec_from_file_location("R", "../rsi_20261104/rsi_loop.py"); R = importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
L = Path("results")
ar, e0, e1, sft = (R.ee_eval_result_path(L / n / "console.log") for n in ("rsil_ppo_ee_ar_eval", "rsil_ppo_ee_phi0_eval", "rsil_ppo_ee_phi1_eval", "eval_sft_c10"))
def mcn(a, b):
    k = sorted(set(a) & set(b)); x = sum(1 for i in k if a[i] and not b[i]); y = sum(1 for i in k if b[i] and not a[i]); n = x + y
    return len(k), sum(a[i] for i in k), sum(b[i] for i in k), (min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2 ** n) if n else 1.0)
for name, ref in (("EE-Phi0 (5e-6)", e0), ("EE-Phi1 (1.25e-6)", e1), ("SFT", sft)):
    n, a, b, p = mcn(ar, ref); print(f"RSI-AR pilot (EE, 2e-5 + AR) vs {name}: {a}/{n} vs {b}/{n}, McNemar p={p:.4f}")
print("per task (AR / Phi0 / SFT):", {t: (sum(v for k, v in ar.items() if int(k.split(':')[0]) == t), sum(v for k, v in e0.items() if int(k.split(':')[0]) == t), sum(sft.get(k, 0) for k in ar if int(k.split(':')[0]) == t)) for t in (2, 3, 5, 9)})
PY
log "RSI-AR local pilot finished"
