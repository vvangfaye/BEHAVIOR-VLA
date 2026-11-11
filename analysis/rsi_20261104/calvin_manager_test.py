#!/usr/bin/env python3
"""Offline test of agent_manager_calvin_patch.py (2026-11-06). Applies the patch to a scratch copy of the three files
and checks
  1. LIBERO/RoboCasa unchanged: for modes agent (stubbed Sol), rule, random, replay and none, the patched manager
     gives byte-identical prompts, schemas, weights, lr multipliers and decisions.jsonl rows (time field excluded),
     with and without a RoboCasa-style tasks.json; env_weights() == weights;
  2. CALVIN manager: prompt formatting with the real 1000-chain pool, 34 type weights -> 1000 chain weights, the
     achieved attempt shares follow the type weights, uniform type weights give uniform chains, resume;
  3. CALVIN env methods (run with .venv-calvin and --env): tasks.json, per-chain records, weighted chain sampling;
  4. actor counting snippet with "sub" records.
Usage: calvin_manager_test.py <RLinf root> <chains.json> [--env]"""
import importlib.util
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT, CHAINS = Path(sys.argv[1]), Path(sys.argv[2])
HERE = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp(prefix="calvin_mgr_test_"))
files = ["rlinf/envs/sim/calvin/calvin_gym_env.py", "rlinf/workers/actor/embodied_fsdp_actor_worker.py",
         "rlinf/b1k_tools/agent_manager.py"]
for f in files:
    (tmp / f).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / f, tmp / f)
r = subprocess.run([sys.executable, str(HERE / "agent_manager_calvin_patch.py"), str(tmp)], capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip())
assert r.returncode == 0
r2 = subprocess.run([sys.executable, str(HERE / "agent_manager_calvin_patch.py"), str(tmp)], capture_output=True, text=True)
assert "already patched" in r2.stdout and r2.returncode == 0, r2.stdout  # idempotent
for f in files:
    subprocess.run([sys.executable, "-m", "py_compile", str(tmp / f)], check=True)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


OLD = load(ROOT / files[2], "am_old")
NEW = load(tmp / files[2], "am_new")


def stub(mod, store, answer_fn):
    def fake(prompt, ev_dir, timeout=180, schema=None):
        store.append((prompt, json.dumps(schema, sort_keys=True)))
        return answer_fn(len(store))
    mod.ask_agent = fake


def synthetic_counts(ntask, it, seed):
    rnd = random.Random(seed * 1000 + it)
    return {t: (n, rnd.randint(0, n)) for t in range(ntask) for n in [rnd.randint(5, 20)]}


# ---------------------------------------------------------------- 1. LIBERO / RoboCasa unchanged
def run(mod, mode, version, out, ntask=10, tasks_json=None, replay=None):
    sd = out / "stats"
    sd.mkdir(parents=True, exist_ok=True)
    if tasks_json:
        (sd / "tasks.json").write_text(json.dumps(tasks_json))
    store = []
    rnd = random.Random(7)
    stub(mod, store, lambda k: {"task_weights": [round(rnd.uniform(0.3, 3), 2) for _ in range(ntask)],
                                "lr_multiplier": rnd.choice([0.5, 1.0, 1.4]),
                                "prob_regressed": [round(rnd.random(), 2) for _ in range(ntask)], "reason": "x"})
    M = mod.Manager(mode, out / "am", 12, seed=3, stats_dir=str(sd), episodes_per_iter=128, version=version,
                    replay_from=replay, replay_lr=True, replay_weights=True)
    res = []
    for it in range(12):
        w, m, _ = M.decide(synthetic_counts(ntask, it, 1))
        env_w = M.env_weights() if hasattr(M, "env_weights") else w
        res.append((list(w), m, list(env_w)))
    rows = [json.loads(l) for l in (out / "am" / "decisions.jsonl").read_text().splitlines()]
    for row in rows:
        row.pop("time")
    return res, store, rows


rc_tasks = {"suite": "RoboCasa365 atomic", "tasks": [f"task {i}" for i in range(6)]}
n_checked = 0
for mode in ("agent", "relay_skip", "rule", "random", "none", "replay"):
    if mode == "relay_skip":
        continue
    for version in ("v2", "v3"):
        for tj, ntask in ((None, 10), (rc_tasks, 6)):
            replay = None
            if mode == "replay":
                base = tmp / f"replay_src_{version}_{ntask}"
                run(OLD, "random", version, base, ntask, tj)
                replay = str(base / "am" / "decisions.jsonl")
            a = run(OLD, mode, version, tmp / f"old_{mode}_{version}_{ntask}", ntask, tj, replay)
            b = run(NEW, mode, version, tmp / f"new_{mode}_{version}_{ntask}", ntask, tj, replay)
            assert [x[:2] for x in a[0]] == [x[:2] for x in b[0]], (mode, version, ntask)
            assert all(x[2] == x[0] for x in b[0]), "env_weights must equal weights without chains"
            assert a[1] == b[1], f"prompt/schema differ: {mode} {version} {ntask}"
            assert a[2] == b[2], f"decisions differ: {mode} {version} {ntask}"
            n_checked += 1
print(f"1. LIBERO/RoboCasa byte-identical in {n_checked} configurations (modes agent/rule/random/none/replay x v2/v3 x 10/6 tasks)")

# ---------------------------------------------------------------- 2. CALVIN manager
cd = json.loads(CHAINS.read_text())
names = cd["names"]
idx = {n: i for i, n in enumerate(names)}
chains = [[idx[t] for t in c] for c in cd["chains"]]
calvin_tasks = {"suite": "CALVIN ABC->D (training in scenes A, B, C)", "kind": "calvin_chains", "tasks": names, "chains": chains}
weak = ["push_blue_block_right", "push_red_block_right", "push_pink_block_right", "rotate_pink_block_right", "push_into_drawer"]
sft = {"push_blue_block_right": .24, "push_red_block_right": .31, "push_pink_block_right": .36, "rotate_pink_block_right": .49,
       "push_into_drawer": .66}


def calvin_counts(it, rnd):
    """128 chains drawn uniformly; per-type success from the SFT eval profile (0.92 elsewhere)."""
    c = {}
    for _ in range(128):
        ch = chains[rnd.randrange(len(chains))]
        for s in ch:
            ok = rnd.random() < sft.get(names[s], 0.92)
            v = c.setdefault(s, [0, 0])
            v[0] += 1
            v[1] += int(ok)
            if not ok:
                break
    return {t: tuple(v) for t, v in c.items()}


for version in ("v2", "v3"):
    out = tmp / f"calvin_{version}"
    sd = out / "stats"
    sd.mkdir(parents=True)
    (sd / "tasks.json").write_text(json.dumps(calvin_tasks))
    store = []
    lo, hi = NEW.VERSIONS[version]["w"]
    stub(NEW, store, lambda k: {"task_weights": [min(hi, 2.0) if names[t] in weak else 1.0 for t in range(len(names))],
                                "lr_multiplier": 1.0, "prob_regressed": [0.0] * len(names), "reason": "weak x2"})
    M = NEW.Manager("agent", out / "am", 20, seed=0, stats_dir=str(sd), episodes_per_iter=128, version=version)
    assert M.ntask == 34 and M.chains is not None and len(M.chains) == 1000
    assert M.env_weights() == [1.0] * 1000
    rnd = random.Random(0)
    for it in range(4):
        w, m, info = M.decide(calvin_counts(it, rnd))
    p, sch = store[-1]
    assert "subtask 0: " in p and "chain of 5 language-instructed subtasks" in p and "pool of 1000 chains" in p
    assert "task 0:" not in p.replace("subtask 0:", "")
    assert json.loads(sch)["properties"]["task_weights"]["minItems"] == 34
    cw = M.env_weights()
    assert len(cw) == 1000 and abs(sum(cw) / 1000 - 1) < 1e-9 and min(cw) >= 0.1 - 1e-9
    ach = info["chain_weights"]["attempt_share_vs_uniform"]
    wk = [ach[idx[n]] for n in weak]
    other = [ach[t] for t in range(34) if names[t] not in weak]
    assert min(wk) > 1.6 and max(other) < 1.0, (wk, max(other))
    row = json.loads((out / "am" / "decisions.jsonl").read_text().splitlines()[-1])
    assert row["chain_weights"]["ess_frac"] > 0.3
    print(f"2. CALVIN {version}: prompt {len(p)} chars; weak-type attempt share x{min(wk):.2f}..x{max(wk):.2f}, others "
          f"x{min(other):.2f}..x{max(other):.2f}; chain w [{min(cw):.2f}, {max(cw):.2f}], ESS {row['chain_weights']['ess_frac']}")
    if version == "v2":
        print("---- CALVIN v2 prompt (iteration 4), first 1600 chars ----\n" + p[:1600] + "\n....\n" + p[-700:] + "\n----")

# uniform type weights -> uniform chains; rule / random / none / replay run with chains
for mode in ("none", "rule", "random"):
    out = tmp / f"calvin_{mode}"
    sd = out / "stats"
    sd.mkdir(parents=True)
    (sd / "tasks.json").write_text(json.dumps(calvin_tasks))
    M = NEW.Manager(mode, out / "am", 20, seed=0, stats_dir=str(sd), episodes_per_iter=128)
    rnd = random.Random(1)
    for it in range(3):
        w, m, info = M.decide(calvin_counts(it, rnd))
    cw = M.env_weights()
    if mode == "none":
        assert max(abs(x - 1) for x in cw) < 1e-9
    print(f"   mode {mode}: chain w [{min(cw):.2f}, {max(cw):.2f}] ESS {info['chain_weights']['ess_frac']}, lr x{m}")
# resume
out = tmp / "calvin_rule"
(out / "am" / "resume_from.jsonl").write_text((out / "am" / "decisions.jsonl").read_text())
M2 = NEW.Manager("rule", out / "am", 20, seed=0, stats_dir=str(out / "stats"), episodes_per_iter=128)
assert len(M2.hist) == 3 and M2.chains is not None and M2.env_weights() == [1.0] * 1000
w, m, info = M2.decide(calvin_counts(3, random.Random(2)))
assert len(M2.env_weights()) == 1000
print("   resume: history 3 -> decision 4 ok")

# ---------------------------------------------------------------- 4. actor counting snippet
src = (tmp / files[1]).read_text()
assert 'for t, s in (r["sub"] if "sub" in r else [(r["task"], r["s"])]):' in src and "self._am.env_weights()" in src
recs = [{"task": 3, "s": 0, "sub": [[1, 1], [2, 1], [3, 0]], "trial": 5, "done": 2}, {"task": 4, "s": 1}]
counts = {}
for r_ in recs:
    for t, s in (r_["sub"] if "sub" in r_ else [(r_["task"], r_["s"])]):
        v = counts.setdefault(int(t), [0, 0])
        v[0] += 1
        v[1] += int(s)
assert counts == {1: [1, 1], 2: [1, 1], 3: [1, 0], 4: [1, 1]}
print("4. actor counting ok")

# ---------------------------------------------------------------- 3. env methods (needs .venv-calvin)
if "--env" in sys.argv:
    import numpy as np
    from omegaconf import OmegaConf

    sys.path.insert(0, str(ROOT))
    E = load(tmp / files[0], "calvin_env_new")
    from rlinf.envs.sim.calvin import CalvinBenchmark

    sdir = tmp / "env_stats"
    wfile = tmp / "env_w.json"
    env = object.__new__(E.CalvinEnv)
    env.cfg = OmegaConf.create({"is_eval": False, "task_stats_dir": str(sdir), "agent_weights_file": str(wfile),
                                "task_suite_name": "calvin_abc"})
    env._generator = np.random.default_rng(0)
    env.task_suite = CalvinBenchmark("calvin_abc", env._generator)
    env.num_tasks, env.task_num_trials = env.task_suite.get_num_tasks(), env.task_suite.get_task_num_trials()
    env._compute_total_num_group_envs()
    env.num_envs = 4
    env._calvin_write_tasks_json()
    tj = json.loads((sdir / "tasks.json").read_text())
    assert tj["kind"] == "calvin_chains" and tj["tasks"] == names and tj["chains"] == chains, "env chain pool != offline pool"
    # records
    env.trial_ids = np.array([0, 1, 2, 3])
    env.task_sequence = [env.task_suite.get_task_sequence(t) for t in env.trial_ids]
    env.current_task_idx = [0, 2, 5, 4]
    env._ts_recorded = np.zeros(4, dtype=bool)
    env._record_task_stats(np.array([True, False, True, False]))
    env.current_task_idx = [1, 2, 5, 4]
    env._record_task_stats(np.array([True, True, True, True]))  # env 0 and 2 already recorded this epoch
    lines = [json.loads(l) for f in sdir.glob("env_*.jsonl") for l in f.read_text().splitlines()]
    assert len(lines) == 4, lines
    by = {l["trial"]: l for l in lines}
    assert by[0]["done"] == 0 and by[0]["sub"] == [[chains[0][0], 0]] and by[0]["s"] == 0
    assert by[2]["done"] == 5 and by[2]["s"] == 1 and len(by[2]["sub"]) == 5 and all(x[1] == 1 for x in by[2]["sub"])
    assert by[1]["sub"] == [[chains[1][0], 1], [chains[1][1], 1], [chains[1][2], 0]]
    assert by[3]["sub"][-1] == [chains[3][4], 0] and len(by[3]["sub"]) == 5
    # sampling: no file -> uniform integers (original RNG path); with weights -> proportional
    g_ref = np.random.default_rng(5).integers(low=0, high=1000, size=(8,))
    env._generator = np.random.default_rng(5)
    assert (env._get_random_reset_state_ids(8) == g_ref).all()
    w = [0.0] * 1000
    for t in (7, 70, 700):
        w[t] = 1.0
    wfile.write_text(json.dumps({str(i): x for i, x in enumerate(w)}))
    ids = env._get_random_reset_state_ids(300)
    assert set(ids.tolist()) == {7, 70, 700} and ids.dtype == np.int64
    env.cfg.is_eval = True
    assert env._agent_task_weights() is None
    print("3. env: tasks.json matches the offline pool; records ok; sampling uniform without weights, proportional with")
shutil.rmtree(tmp)
print("ALL OK")
