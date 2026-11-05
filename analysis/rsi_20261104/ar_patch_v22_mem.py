#!/usr/bin/env python3
"""AR v2.2 (memory only, 2026-11-05 11:1x; no change to the accept/rollback rule).
A 5090 job (240 GiB host RAM, 32 LIBERO envs) was OOM-killed by Ray exactly when the best anchor moved: taking the new
snapshot deep-copied the optimizer state on the GPU (one extra copy of the Adam moments in GPU memory) and kept the old
snapshot alive until the new one existed (two host copies). Rollbacks deep-copied the host snapshot once more.
Now: the old snapshot is dropped before the new one is taken, tensors go straight to host memory (_ar_to_cpu already
copies every tensor), and a rollback loads the host snapshot directly (load_state_dict copies the moments onto the GPU
parameters' device); CPU tensors that Optimizer.load_state_dict keeps by reference ('step') are cloned so that Adam's
in-place step increments cannot alter the snapshot.
Usage: ar_patch_v22_mem.py <RLinf root> (requires ar_patch.py + v2 + v2.1; idempotent)."""
import sys
from pathlib import Path

ACT = Path(sys.argv[1]) / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
a = ACT.read_text()
if "accept_rollback v2.2" in a:
    print("v2.2 already applied"); sys.exit(0)
assert "accept_rollback v2.1" in a, "apply v2.1 first"


def sub(text, old, new, label):
    if text.count(old) != 1:
        raise SystemExit(f"{label}: anchor found {text.count(old)} times")
    return text.replace(old, new)


a = sub(a, """                # this rollout's policy is the new fallback point (v2 'best': only when it is the new anchor)
                st["snap"] = ([p.detach().to("cpu", copy=True) for p in params],
                              self._ar_to_cpu(copy.deepcopy(self.optimizer.state_dict())))""",
        """                # this rollout's policy is the new fallback point (v2 'best': only when it is the new anchor)
                # accept_rollback v2.2 (memory): drop the old snapshot first, copy tensors straight to host memory
                st["snap"] = None
                import gc

                gc.collect()
                st["snap"] = ([p.detach().to("cpu", copy=True) for p in params],
                              self._ar_to_cpu(self.optimizer.state_dict()))""", "snapshot")
a = sub(a, """            self.optimizer.load_state_dict(copy.deepcopy(st["snap"][1]))""",
        """            # accept_rollback v2.2 (memory): no host deepcopy; load_state_dict copies the moments to the GPU params'
            # device, and CPU tensors it keeps by reference ('step') are cloned so the snapshot stays intact
            self.optimizer.load_state_dict(st["snap"][1])
            for _st in self.optimizer.state.values():
                for _k, _v in list(_st.items()):
                    if torch.is_tensor(_v) and _v.device.type == "cpu":
                        _st[_k] = _v.clone()""", "rollback load")
ACT.write_text(a)
print("v2.2 applied")
