#!/usr/bin/env python3
"""AR v2 (2026-11-05 08:0x): optional best-anchor reference. algorithm.accept_rollback.reference = 'last' (v1, default:
compare with the last accepted rollout, which drifts down with slow declines) | 'best' (compare with the best accepted
rollout so far by overall success; the snapshot / rollback point is that best policy and only moves when a rollout is at
least as good overall). Usage: ar_patch_v2.py <RLinf root> (requires ar_patch.py applied; idempotent)."""
import sys
from pathlib import Path

ACT = Path(sys.argv[1]) / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
a = ACT.read_text()
if "accept_rollback v2" in a:
    print("v2 already applied"); sys.exit(0)
old = """            if accepted and recs:
                st["ref"] = {t: tuple(v) for t, v in new.items()}
            dec = torch.tensor([float(accepted), len(recs), worst[0], worst[1], mean[0], mean[1]], device=self.device)"""
new = """            # accept_rollback v2 (2026-11-05): reference='best' keeps the best accepted rollout as the anchor (no downward
            # drift); the snapshot below only moves with the anchor (flag 6 of dec)
            move = accepted
            if accepted and recs and str(acfg.get("reference", "last")) == "best" and st["ref"] is not None:
                rate = lambda d: sum(v[1] for v in d.values()) / max(sum(v[0] for v in d.values()), 1)
                move = rate(new) >= rate(st["ref"])
            if move and recs:
                st["ref"] = {t: tuple(v) for t, v in new.items()}
            dec = torch.tensor([float(accepted), len(recs), worst[0], worst[1], mean[0], mean[1], float(move)],
                               device=self.device)"""
assert a.count(old) == 1, "anchor 1"
a = a.replace(old, new)
old2 = """        dec = torch.zeros(6, device=self.device)  # accepted, n, worst drop, worst z, mean drop, mean z"""
new2 = """        dec = torch.zeros(7, device=self.device)  # accepted, n, worst drop, worst z, mean drop, mean z, move anchor"""
assert a.count(old2) == 1, "anchor 2"
a = a.replace(old2, new2)
old3 = """        if accepted or st["snap"] is None:
            # this rollout's policy is the new fallback point
            st["snap"] = ([p.detach().to("cpu", copy=True) for p in params],
                          self._ar_to_cpu(copy.deepcopy(self.optimizer.state_dict())))
            st["accepts"] += 1
            accepted = True"""
new3 = """        if accepted or st["snap"] is None:
            if dec[6].item() > 0.5 or st["snap"] is None:
                # this rollout's policy is the new fallback point (v2 'best': only when it is the new anchor)
                st["snap"] = ([p.detach().to("cpu", copy=True) for p in params],
                              self._ar_to_cpu(copy.deepcopy(self.optimizer.state_dict())))
            st["accepts"] += 1
            accepted = True"""
assert a.count(old3) == 1, "anchor 3"
a = a.replace(old3, new3)
ACT.write_text(a)
print("v2 applied")
