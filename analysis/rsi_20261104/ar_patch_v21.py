#!/usr/bin/env python3
"""AR v2.1 (2026-11-05 09:4x): two fixes found in the first AR v2 run on 10 tasks (2 rollbacks in 6 iterations, both
just over z=1.64 on one task with ~13 episodes).
1. Multiple comparisons: a task triggers a rollback only if its drop exceeds delta_task AND its one-sided z exceeds the
   Bonferroni threshold z = Phi^-1(1 - alpha / K), K = number of tasks with >= min_n episodes in both rollouts
   (alpha = algorithm.accept_rollback.alpha, default 0.05; K=10 -> 2.58, K=4 -> 2.24, K=3 -> 2.13). The overall-mean test
   keeps algorithm.accept_rollback.z (one test).
2. Learning rate on rollback: optimizer.load_state_dict restores the snapshot's lr, so the decay took effect one
   iteration late; now the lr in force before the rollback is multiplied by lr_decay after the restore.
Usage: ar_patch_v21.py <RLinf root> (requires ar_patch.py + ar_patch_v2.py; idempotent)."""
import sys
from pathlib import Path

ACT = Path(sys.argv[1]) / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
a = ACT.read_text()
if "accept_rollback v2.1" in a:
    print("v2.1 already applied"); sys.exit(0)


def sub(text, old, new, label):
    if text.count(old) != 1:
        raise SystemExit(f"{label}: anchor found {text.count(old)} times")
    return text.replace(old, new)


a = sub(a, """        worst = (0.0, 0.0, None)  # (drop, z, task)
        for t, (n1, s1) in ref.items():
            n2, s2 = new.get(t, (0, 0))
            if n1 >= min_n and n2 >= min_n:
                d = s1 / n1 - s2 / n2
                if d > worst[0]:
                    worst = (d, z(s1 / n1, n1, s2 / n2, n2), t)""", """        worst = (0.0, 0.0, None)  # (drop, z, task)
        per_task = []  # accept_rollback v2.1: every eligible task's (drop, z, task) for the Bonferroni test
        for t, (n1, s1) in ref.items():
            n2, s2 = new.get(t, (0, 0))
            if n1 >= min_n and n2 >= min_n:
                d = s1 / n1 - s2 / n2
                per_task.append((d, z(s1 / n1, n1, s2 / n2, n2), t))
                if d > worst[0]:
                    worst = (d, z(s1 / n1, n1, s2 / n2, n2), t)
        EmbodiedFSDPActor._ar_last_per_task = per_task""", "per-task list")
a = sub(a, """                zc = float(acfg.get("z", 1.64))
                bad_task = worst[0] > float(acfg.get("delta_task", 0.15)) and worst[1] > zc""", """                zc = float(acfg.get("z", 1.64))
                # accept_rollback v2.1: Bonferroni-corrected per-task threshold over the K eligible tasks
                from statistics import NormalDist
                per_task = getattr(EmbodiedFSDPActor, "_ar_last_per_task", [])
                zc_task = NormalDist().inv_cdf(1.0 - float(acfg.get("alpha", 0.05)) / max(len(per_task), 1))
                dt = float(acfg.get("delta_task", 0.15))
                hits = [p for p in per_task if p[0] > dt and p[1] > zc_task]
                bad_task = bool(hits)
                if hits:
                    worst = max(hits, key=lambda p: p[1])""", "bonferroni")
a = sub(a, """            self.optimizer.load_state_dict(copy.deepcopy(st["snap"][1]))
            gamma = float(acfg.get("lr_decay", 0.5))
            self.optimizer.param_groups[0]["lr"] *= gamma""", """            gamma = float(acfg.get("lr_decay", 0.5))
            lr_now = self.optimizer.param_groups[0]["lr"]  # accept_rollback v2.1: decay the lr in force, not the snapshot's
            self.optimizer.load_state_dict(copy.deepcopy(st["snap"][1]))
            self.optimizer.param_groups[0]["lr"] = lr_now * gamma""", "lr restore")
ACT.write_text(a)
print("v2.1 applied")
