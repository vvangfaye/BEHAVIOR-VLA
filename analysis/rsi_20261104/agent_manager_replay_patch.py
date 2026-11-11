#!/usr/bin/env python3
"""Manager mode=replay (2026-11-06, attribution control for the agent-v2 result 62.6% vs PPO 45.8% on seed 3):
replays a finished run's decisions.jsonl - the effective lr multiplier per iteration (replay_lr, default on) and/or the
task weights (replay_weights, default off -> uniform weights). lr-only replay = "is it just the agent's higher lr?".
Overrides: +algorithm.agent_manager.mode=replay +algorithm.agent_manager.replay_from=<decisions.jsonl>
           [+algorithm.agent_manager.replay_lr=True] [+algorithm.agent_manager.replay_weights=False]
Usage: agent_manager_replay_patch.py <RLinf root> (idempotent)."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
AMF = root / "rlinf/b1k_tools/agent_manager.py"
ACT = root / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
MARK = "manager replay (local addition 2026-11-06)"


def sub(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: anchor found {n} times")
    return text.replace(old, new)


m = AMF.read_text()
if MARK not in m:
    m = sub(m, '    def __init__(self, mode, out_dir, total_iters, seed=0, ntask=10, stats_dir=None, episodes_per_iter=128, version="v2"):',
            '    def __init__(self, mode, out_dir, total_iters, seed=0, ntask=10, stats_dir=None, episodes_per_iter=128, version="v2",\n'
            '                 replay_from=None, replay_lr=True, replay_weights=False):\n'
            '        # ' + MARK + '\n'
            '        self.replay_rows = [json.loads(l) for l in open(replay_from) if l.strip()] if replay_from else []\n'
            '        self.replay_lr, self.replay_weights = bool(replay_lr), bool(replay_weights)', "init")
    m = sub(m, '        elif self.mode == "random":',
            '        elif self.mode == "replay":  # ' + MARK + '\n'
            '            if not self.replay_rows:\n'
            '                raise RuntimeError("mode=replay needs algorithm.agent_manager.replay_from")\n'
            '            row = self.replay_rows[min(it, len(self.replay_rows)) - 1]\n'
            '            info["replayed_iteration"] = row.get("iteration")\n'
            '            if self.replay_weights:\n'
            '                self.weights = [float(x) for x in row["weights"]]\n'
            '            if self.replay_lr:\n'
            '                mult = float(row["lr_multiplier"])\n'
            '        elif self.mode == "random":', "decide")
    AMF.write_text(m)
    print("manager replay added")
else:
    print("manager already has replay")
a = ACT.read_text()
if MARK not in a:
    a = sub(a, '                               version=str(mcfg.get("version", "v2")))  # manager v3 (local addition 2026-11-06)',
            '                               version=str(mcfg.get("version", "v2")),  # manager v3 (local addition 2026-11-06)\n'
            '                               replay_from=mcfg.get("replay_from", None), replay_lr=bool(mcfg.get("replay_lr", True)),\n'
            '                               replay_weights=bool(mcfg.get("replay_weights", False)))  # ' + MARK, "actor")
    ACT.write_text(a)
    print("actor patched")
else:
    print("actor already patched")
