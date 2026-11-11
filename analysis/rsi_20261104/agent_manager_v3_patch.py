#!/usr/bin/env python3
"""Manager v3 as an opt-in version (2026-11-06 late morning; default stays v2 so running/queued v2 runs are unchanged).
Why: both v2 runs (seed 3 local, seed 2 cluster) pushed the lr to 1.5-1.95x in the first iterations and poured samples into
collapsed / never-solved tasks; their training success fell (seed 2 macro 0.41 -> 0.20 by iteration 8, same-seed rule
manager at lr 1x recovered to 0.37). v3: lr steps {0.75, 1, 1.25}, cumulative scale [0.5, 1.25]; weights [0.5, 2], a
flagged task gets >= 1.5x the mean; the prompt drops the 'do not be conservative' prior and states what was observed.
Select with +algorithm.agent_manager.version=v3. Usage: agent_manager_v3_patch.py <RLinf root> (idempotent)."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
AMF = root / "rlinf/b1k_tools/agent_manager.py"
ACT = root / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
MARK = "manager v3 (local addition 2026-11-06)"


def sub(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: anchor found {n} times")
    return text.replace(old, new)


m = AMF.read_text()
if MARK not in m:
    i = m.index('PROMPT = """You manage online')
    j = m.index('Answer with JSON only."""', i) + len('Answer with JSON only."""')
    v2_prompt = m[i:j]
    v3_prompt = v2_prompt.replace('PROMPT = """', 'PROMPT_V3 = """', 1)
    v3_prompt = v3_prompt.replace("(0.5, 0.75, 1, 1.25 or 1.5; the cumulative learning rate is kept within 0.25x..2x of the default)",
                                  "(0.75, 1 or 1.25; the cumulative learning rate is kept within 0.5x..1.25x of the default)")
    v3_prompt = v3_prompt.replace("Use the range 0.25..4: 1 = normal share, 2 = double, 0.5 = half.",
                                  "Use the range 0.5..2: 1 = normal share, 2 = double, 0.5 = half.")
    v3_prompt = v3_prompt.replace(
        "- Being too conservative (tiny learning rates, ignoring weak tasks) was the main failure of earlier automatic managers; halving the learning rate repeatedly freezes learning.",
        "- In earlier runs of this setup, raising the learning rate to 1.5-2x the default coincided with broad drops in success, and halving it repeatedly froze learning; keep the default unless the statistics clearly call for a change.\n"
        "- A task that has collapsed to near-zero success did not recover by sampling it much more; moderate extra practice (up to 2x) is enough.")
    assert v3_prompt != v2_prompt.replace('PROMPT = """', 'PROMPT_V3 = """', 1) and "PROMPT_V3" in v3_prompt
    assert "0.5x..1.25x" in v3_prompt and "0.5..2" in v3_prompt and "coincided" in v3_prompt
    m = m[:j] + "\n\n# " + MARK + ": opt-in (algorithm.agent_manager.version=v3)\n" + v3_prompt + """
VERSIONS = {
    "v2": {"prompt": "PROMPT", "lr_choices": (0.5, 0.75, 1.0, 1.25, 1.5), "scale": (0.25, 2.0), "w": (0.25, 4.0), "bump": 2.0},
    "v3": {"prompt": "PROMPT_V3", "lr_choices": (0.75, 1.0, 1.25), "scale": (0.5, 1.25), "w": (0.5, 2.0), "bump": 1.5},
}


def schema_version(n, version="v2"):
    \"\"\"schema_for(n) with the version's weight and lr bounds (v2: identical to schema_for).\"\"\"
    s = schema_for(n)
    if version == "v3":
        for k in ("minimum", "maximum"):
            s["properties"]["task_weights"]["items"][k] = VERSIONS["v3"]["w"][0 if k == "minimum" else 1]
        s["properties"]["lr_multiplier"]["minimum"], s["properties"]["lr_multiplier"]["maximum"] = 0.75, 1.25
    return s
""" + m[j:]
    m = sub(m, "    def __init__(self, mode, out_dir, total_iters, seed=0, ntask=10, stats_dir=None, episodes_per_iter=128):",
            "    def __init__(self, mode, out_dir, total_iters, seed=0, ntask=10, stats_dir=None, episodes_per_iter=128, version=\"v2\"):\n"
            "        self.version = version if version in VERSIONS else \"v2\"  # " + MARK + "\n"
            "        self.V = VERSIONS[self.version]", "init sig")
    m = sub(m, "            prompt = PROMPT.format(tasks=", "            prompt = globals()[self.V[\"prompt\"]].format(tasks=", "prompt select")
    m = sub(m, '                ans = ask_agent(prompt, self.out / f"iter{it:03d}", schema=schema_for(self.ntask))',
            '                ans = ask_agent(prompt, self.out / f"iter{it:03d}", schema=schema_version(self.ntask, self.version))', "agent schema")
    m = sub(m, '                ans = ask_relay(prompt, self.out / f"iter{it:03d}", key=self.out.parent.name, schema=schema_for(self.ntask))',
            '                ans = ask_relay(prompt, self.out / f"iter{it:03d}", key=self.out.parent.name, schema=schema_version(self.ntask, self.version))',
            "relay schema")
    m = sub(m, "                w = [min(4.0, max(0.25, float(x))) for x in ans[\"task_weights\"]]",
            "                lo, hi = self.V[\"w\"]\n                w = [min(hi, max(lo, float(x))) for x in ans[\"task_weights\"]]", "agent clip")
    m = sub(m, "                    w[t] = max(w[t], 2.0 * mean_w)", "                    w[t] = max(w[t], self.V[\"bump\"] * mean_w)", "bump")
    m = sub(m, "                self.weights = normalize(w)\n                mult = min(LR_CHOICES,",
            "                self.weights = normalize(w, *self.V[\"w\"])\n                mult = min(self.V[\"lr_choices\"],", "agent norm+lr")
    m = sub(m, "            self.weights = normalize(w)\n        elif self.mode == \"random\":",
            "            self.weights = normalize(w, *self.V[\"w\"])\n        elif self.mode == \"random\":", "rule norm")
    m = sub(m, "            self.weights = normalize([math.exp(self.rng.uniform(math.log(0.5), math.log(2.0))) for _ in range(self.ntask)])\n            mult = self.rng.choice(LR_CHOICES)",
            "            self.weights = normalize([math.exp(self.rng.uniform(math.log(0.5), math.log(2.0))) for _ in range(self.ntask)], *self.V[\"w\"])\n            mult = self.rng.choice(self.V[\"lr_choices\"])",
            "random")
    m = sub(m, "        new_scale = min(2.0, max(0.25, self.lr_scale * mult))",
            "        new_scale = min(self.V[\"scale\"][1], max(self.V[\"scale\"][0], self.lr_scale * mult))", "scale")
    m = sub(m, "        info.update(version=MANAGER_VERSION,", "        info.update(version=self.version,", "version log")
    AMF.write_text(m)
    print("manager v3 added")
else:
    print("manager already has v3")

a = ACT.read_text()
if MARK not in a:
    a = sub(a, """                               episodes_per_iter=int(self.cfg.env.train.total_num_envs)
                               * int(self.cfg.env.train.get("rollout_epoch", 1)))""",
            """                               episodes_per_iter=int(self.cfg.env.train.total_num_envs)
                               * int(self.cfg.env.train.get("rollout_epoch", 1)),
                               version=str(mcfg.get("version", "v2")))  # """ + MARK, "actor version")
    ACT.write_text(a)
    print("actor patched")
else:
    print("actor already patched")
