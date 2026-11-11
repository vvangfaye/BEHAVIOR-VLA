#!/usr/bin/env python3
"""AR v2.3 (2026-11-05 11:3x): re-measurement of the anchor after a rollback (opt-in: algorithm.accept_rollback.remeasure).
Found in the local seed-3 v2.1 run: the anchor (best accepted rollout, 58/128, task 2 = 10/16) was a lucky draw; after a
justified-looking rollback (11:09), the next rollout came from the restored anchor policy itself (identical parameters)
and was tested against the anchor's own lucky measurement: task 2 = 4/16 -> "drop" 0.375, z 2.14 -> a second rollback of
the same policy (no update, lr halved again); the third rollout of that policy got task 2 = 2/8.
With remeasure=True the rollout right after a rollback is treated as what it is, a fresh measurement of the anchor policy:
it is accepted without a test and pooled into the anchor's reference counts (per task n, s), so later candidates are
tested against the pooled, less noisy and less biased (best-of-many) estimate; the snapshot stays (same parameters).
Usage: ar_patch_v23.py <RLinf root>  (requires v2.1 and v2.2; idempotent)."""
import sys
from pathlib import Path

ACT = Path(sys.argv[1]) / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
a = ACT.read_text()
if "accept_rollback v2.3" in a:
    print("v2.3 already applied"); sys.exit(0)
assert "accept_rollback v2.2" in a, "apply v2.2 first"


def sub(text, old, new, label):
    if text.count(old) != 1:
        raise SystemExit(f"{label}: anchor found {text.count(old)} times")
    return text.replace(old, new)


a = sub(a, """        dec = torch.zeros(7, device=self.device)  # accepted, n, worst drop, worst z, mean drop, mean z, move anchor""",
        """        dec = torch.zeros(8, device=self.device)  # accepted, n, worst drop, worst z, mean drop, mean z, move, remeasure""", "dec size")
a = sub(a, """            accepted, worst, mean = True, (0.0, 0.0, None), (0.0, 0.0)
            if st["ref"] is not None and recs:
                worst, mean = self._ar_drop(st["ref"], new, int(acfg.get("min_n", 8)))""",
        """            accepted, worst, mean = True, (0.0, 0.0, None), (0.0, 0.0)
            # accept_rollback v2.3: the rollout right after a rollback comes from the restored anchor policy itself -> pool it
            # into the anchor's reference and accept it without a test (opt-in: remeasure=True)
            remeasure = bool(acfg.get("remeasure", False)) and bool(st.get("restored")) and st["ref"] is not None and bool(recs)
            if remeasure:
                pooled = {t: list(v) for t, v in st["ref"].items()}
                for t, v in new.items():
                    pv = pooled.setdefault(t, [0, 0])
                    pv[0] += v[0]
                    pv[1] += v[1]
                st["ref"] = {t: tuple(v) for t, v in pooled.items()}
            elif st["ref"] is not None and recs:
                worst, mean = self._ar_drop(st["ref"], new, int(acfg.get("min_n", 8)))""", "remeasure branch")
a = sub(a, """            move = accepted
            if accepted and recs and str(acfg.get("reference", "last")) == "best" and st["ref"] is not None:""",
        """            move = accepted and not remeasure
            if move and recs and str(acfg.get("reference", "last")) == "best" and st["ref"] is not None:""", "move")
a = sub(a, """            dec = torch.tensor([float(accepted), len(recs), worst[0], worst[1], mean[0], mean[1], float(move)],
                               device=self.device)
            info = {"tasks": {str(t): v for t, v in sorted(new.items())}, "worst_task": worst[2]}""",
        """            dec = torch.tensor([float(accepted), len(recs), worst[0], worst[1], mean[0], mean[1], float(move),
                                float(remeasure)], device=self.device)
            info = {"tasks": {str(t): v for t, v in sorted(new.items())}, "worst_task": worst[2],
                    "ref_tasks": {str(t): list(v) for t, v in sorted((st["ref"] or {}).items())}}""", "dec values")
a = sub(a, """        metrics = {"ar/accepted": float(accepted), "ar/n_records": float(dec[1].item()),""",
        """        st["restored"] = not accepted  # accept_rollback v2.3: the next rollout re-measures the restored anchor
        metrics = {"ar/accepted": float(accepted), "ar/n_records": float(dec[1].item()), "ar/remeasure": float(dec[7].item()),""",
        "restored flag")
ACT.write_text(a)
print("v2.3 applied")
