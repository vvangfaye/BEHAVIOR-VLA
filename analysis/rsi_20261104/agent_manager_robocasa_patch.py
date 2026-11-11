#!/usr/bin/env python3
"""Agent training manager for RoboCasa365 (2026-11-05 night; opt-in, default off). Extends agent_manager_patch.py:

env    rlinf/envs/sim/robocasa365/robocasa365_env.py (train envs only):
       * env.train.task_stats_dir: one (task, success) record per env per rollout epoch (first episode end: success
         termination = 1, truncation = 0), plus <dir>/tasks.json = {"suite", "tasks": [task descriptions]} for the manager;
       * env.train.agent_weights_file: random task sampling with probability proportional to the weights.
manager rlinf/b1k_tools/agent_manager.py: task list, suite name and episodes per iteration are no longer hard-coded to
       LIBERO-Long; they come from <task_stats_dir>/tasks.json and the actor (total_num_envs x rollout_epoch). Without a
       tasks.json the prompt is byte-identical to the LIBERO version used on 11/5 (checked by the test below).
actor  embodied_fsdp_actor_worker.py: passes stats_dir and episodes_per_iter to Manager.
Usage: agent_manager_robocasa_patch.py <RLinf root>   (idempotent; needs agent_manager_patch.py applied first)."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
ENV = root / "rlinf/envs/sim/robocasa365/robocasa365_env.py"
ACT = root / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
AMF = root / "rlinf/b1k_tools/agent_manager.py"
MARK = "agent manager robocasa (local addition 2026-11-05)"


def sub(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: anchor found {n} times")
    return text.replace(old, new)


# ---------------------------------------------------------------- env
e = ENV.read_text()
if MARK not in e:
    e = sub(e, """        del options
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
""", """        del options
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
            # agent manager robocasa (local addition 2026-11-05): a full reset starts a rollout epoch
            if not bool(self.cfg.get("is_eval", False)) and self.cfg.get("task_stats_dir", None):
                self._ts_recorded = np.zeros(self.num_envs, dtype=bool)
""", "env reset")
    e = sub(e, """        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
""", """        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        # agent manager robocasa (local addition 2026-11-05): per-task training statistics
        if not bool(self.cfg.get("is_eval", False)) and self.cfg.get("task_stats_dir", None):
            self._record_task_stats(past_terminations.cpu().numpy(), past_dones.cpu().numpy())
""", "env chunk_step")
    e = sub(e, """        return self._generator.integers(
            low=0,
            high=self.num_tasks,""", """        _aw = self._agent_task_weights()  # agent manager robocasa (local addition 2026-11-05)
        if _aw is not None:
            w = np.array([max(float(_aw.get(t, 1.0)), 0.0) for t in range(self.num_tasks)], dtype=np.float64)
            if w.sum() <= 0:
                w[:] = 1.0
            return self._generator.choice(self.num_tasks, size=num_groups, p=w / w.sum()).astype(np.int32)
        return self._generator.integers(
            low=0,
            high=self.num_tasks,""", "env sampler")
    e = sub(e, """    def _set_next_task_ids(
        self, env_idx: Optional[Union[int, list[int], np.ndarray]] = None
    ) -> np.ndarray:""", '''    def _agent_task_weights(self):
        """agent manager robocasa (local addition 2026-11-05): task weights from env.train.agent_weights_file."""
        path = None if bool(self.cfg.get("is_eval", False)) else self.cfg.get("agent_weights_file", None)
        if not path or not os.path.exists(path):
            return None
        try:
            import json as _json

            mt = os.path.getmtime(path)
            cache = getattr(self, "_aw_cache", (None, None))
            if cache[0] != mt:
                with open(path) as f:
                    self._aw_cache = (mt, {int(k): float(v) for k, v in _json.load(f).items()})
            return self._aw_cache[1]
        except Exception:
            return None

    def _record_task_stats(self, terms, dones):
        """agent manager robocasa (local addition 2026-11-05): append the first episode end of each env in this
        rollout epoch (success termination = 1, truncation = 0) to <task_stats_dir>/env_<pid>.jsonl; the first call
        also writes <task_stats_dir>/tasks.json (task descriptions in task-id order) for the manager's prompt."""
        import json as _json

        d = self.cfg.get("task_stats_dir")
        os.makedirs(d, exist_ok=True)
        tj = os.path.join(d, "tasks.json")
        if not getattr(self, "_ts_tasks_written", False):
            if not os.path.exists(tj):
                tmp = f"{tj}.{os.getpid()}.tmp"
                with open(tmp, "w") as f:
                    _json.dump({"suite": "RoboCasa365 " + str(self.cfg.get("task_mode", "atomic")),
                                "tasks": [str(s.get("task_description", s.get("metadata_view", {}).get("task_name", "?")))
                                          for s in self.task_specs]}, f)
                os.replace(tmp, tj)
            self._ts_tasks_written = True
        rec = getattr(self, "_ts_recorded", None)
        if rec is None or len(rec) != self.num_envs:
            rec = self._ts_recorded = np.zeros(self.num_envs, dtype=bool)
        new = np.asarray(dones, dtype=bool) & ~rec
        if not new.any():
            return
        with open(os.path.join(d, f"env_{os.getpid()}.jsonl"), "a") as f:
            for i in np.nonzero(new)[0]:
                f.write(_json.dumps({"task": int(self.task_ids[i]), "s": int(bool(terms[i]))}) + "\\n")
        rec |= new

    def _set_next_task_ids(
        self, env_idx: Optional[Union[int, list[int], np.ndarray]] = None
    ) -> np.ndarray:''', "env methods")
    if "\nimport os\n" not in e:
        e = sub(e, "\nimport copy\n", "\nimport copy\nimport os\n", "env import os")
    ENV.write_text(e)
    print("env patched")
else:
    print("env already patched")

# ---------------------------------------------------------------- manager
m = AMF.read_text()
if MARK not in m:
    m = sub(m, "After every iteration you set (1) the sampling weight of each task for the next iteration (relative, 0.25..4; 128 training episodes per iteration are split in proportion)",
            "After every iteration you set (1) the sampling weight of each task for the next iteration (relative, 0.25..4; {eps} training episodes per iteration are split in proportion)", "prompt eps")
    m = sub(m, "- Each task gets only ~13 episodes per iteration,", "- Each task gets only ~{eps_task} episodes per iteration,", "prompt eps per task")
    m = sub(m, "ONE robot policy on the 10 LIBERO-Long tasks below.", "ONE robot policy on the {ntask} {suite} tasks below.", "prompt suite")
    m = sub(m, """    def __init__(self, mode, out_dir, total_iters, seed=0, ntask=10):
        self.mode, self.out = mode, Path(out_dir)""", """    def __init__(self, mode, out_dir, total_iters, seed=0, ntask=10, stats_dir=None, episodes_per_iter=128):
        # agent manager robocasa (local addition 2026-11-05): task list / suite from <stats_dir>/tasks.json if present
        self.stats_dir, self.eps = stats_dir, int(episodes_per_iter)
        self.task_desc, self.suite = list(LIBERO10), "LIBERO-Long"
        try:
            tj = Path(stats_dir) / "tasks.json" if stats_dir else None
            if tj is not None and tj.exists():
                d = json.loads(tj.read_text())
                self.task_desc, self.suite = [str(x) for x in d["tasks"]], str(d.get("suite", "robot"))
                ntask = len(self.task_desc)
        except Exception:  # noqa: BLE001
            pass
        self.mode, self.out = mode, Path(out_dir)""", "manager init")
    m = sub(m, """            prompt = PROMPT.format(tasks="\\n".join(f"{t}: {LIBERO10[t]}" for t in range(self.ntask)), history=hist_rows,
                                   stats=stat_rows, weights=[round(w, 2) for w in self.weights], lr_scale=self.lr_scale,
                                   it=it, total=self.total)""", """            prompt = PROMPT.format(tasks="\\n".join(f"{t}: {self.task_desc[t]}" for t in range(self.ntask)), history=hist_rows,
                                   stats=stat_rows, weights=[round(w, 2) for w in self.weights], lr_scale=self.lr_scale,
                                   it=it, total=self.total, eps=self.eps, eps_task=round(self.eps / max(self.ntask, 1)),
                                   ntask=self.ntask, suite=self.suite)""", "manager prompt format")
    m = sub(m, """PROMPT = \"\"\"You manage online""", """def schema_for(n):
    \"\"\"agent manager robocasa (local addition 2026-11-05): SCHEMA with n tasks (schema_for(10) == SCHEMA).\"\"\"
    s = json.loads(json.dumps(SCHEMA))
    for k in ("task_weights", "prob_regressed"):
        s["properties"][k]["minItems"] = s["properties"][k]["maxItems"] = int(n)
    return s


PROMPT = \"\"\"You manage online""", "schema_for")
    m = sub(m, "def ask_agent(prompt, ev_dir: Path, timeout=180):", "def ask_agent(prompt, ev_dir: Path, timeout=180, schema=None):", "ask_agent sig")
    m = sub(m, '    (ev_dir / "schema.json").write_text(json.dumps(SCHEMA))', '    (ev_dir / "schema.json").write_text(json.dumps(schema or SCHEMA))', "ask_agent schema")
    m = sub(m, "def ask_relay(prompt, ev_dir: Path, key: str, timeout: float | None = None, poll_s: float = 15.0):",
            "def ask_relay(prompt, ev_dir: Path, key: str, timeout: float | None = None, poll_s: float = 15.0, schema=None):", "ask_relay sig")
    m = sub(m, '"prompt": prompt, "schema": SCHEMA}).encode()', '"prompt": prompt, "schema": schema or SCHEMA}).encode()', "ask_relay body")
    m = sub(m, '                ans = ask_agent(prompt, self.out / f"iter{it:03d}")',
            '                ans = ask_agent(prompt, self.out / f"iter{it:03d}", schema=schema_for(self.ntask))', "decide agent schema")
    m = sub(m, '                ans = ask_relay(prompt, self.out / f"iter{it:03d}", key=self.out.parent.name)',
            '                ans = ask_relay(prompt, self.out / f"iter{it:03d}", key=self.out.parent.name, schema=schema_for(self.ntask))',
            "decide relay schema")
    m = m.replace('"""Agent-managed RL training (local addition 2026-11-05;', '"""Agent-managed RL training (' + MARK + '; local addition 2026-11-05;', 1)
    AMF.write_text(m)
    print("manager patched")
else:
    print("manager already patched")

# ---------------------------------------------------------------- actor
a = ACT.read_text()
if MARK not in a:
    a = sub(a, """            self._am = Manager(str(mcfg.get("mode", "agent")), str(mcfg.get("out_dir")),
                               int(self.cfg.runner.max_epochs), seed=int(mcfg.get("seed", 0)))""",
            """            self._am = Manager(str(mcfg.get("mode", "agent")), str(mcfg.get("out_dir")),
                               int(self.cfg.runner.max_epochs), seed=int(mcfg.get("seed", 0)),
                               stats_dir=self.cfg.env.train.get("task_stats_dir"),  # """ + MARK + """
                               episodes_per_iter=int(self.cfg.env.train.total_num_envs)
                               * int(self.cfg.env.train.get("rollout_epoch", 1)))""", "actor manager args")
    ACT.write_text(a)
    print("actor patched")
else:
    print("actor already patched")
