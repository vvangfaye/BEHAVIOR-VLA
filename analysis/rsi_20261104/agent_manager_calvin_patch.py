#!/usr/bin/env python3
"""Agent training manager for CALVIN (2026-11-06 evening; opt-in, default off). Extends the LIBERO/RoboCasa manager.

CALVIN has one "task" with a fixed pool of 1000 chains of 5 subtasks (34 subtask types); an episode stops at the first
failed subtask. The manager therefore works on SUBTASK TYPES and turns its per-type practice weights into chain
sampling probabilities:

env    rlinf/envs/sim/calvin/calvin_gym_env.py (train envs only, when env.train.task_stats_dir is set):
       * __init__ writes <task_stats_dir>/tasks.json = {"suite", "kind": "calvin_chains", "tasks": [34 subtask names,
         sorted], "chains": [[subtask index] x 5] x 1000} (no RNG is consumed, so sampling is unchanged without weights);
       * one record per env per rollout epoch at its first episode end: {"task": last attempted subtask, "s": chain
         completed, "sub": [[subtask, 1], ..., [failed subtask, 0]], "trial": chain id, "done": completed subtasks};
       * env.train.agent_weights_file = {chain id: weight}: random chain sampling with probability proportional to it.
manager rlinf/b1k_tools/agent_manager.py: with a "calvin_chains" tasks.json the prompt describes chains and subtask
       types (PROMPT_CALVIN / PROMPT_V3_CALVIN, same structure and statistics as the LIBERO prompts) and
       chain_weights() solves for chain sampling weights whose expected per-type attempts (with reach probabilities
       from the recent training success rates) are proportional to the type weights (multiplicative raking, chain
       weights clipped to [0.1, 10]); Manager.env_weights() returns the chain weights (or the task weights otherwise).
actor  embodied_fsdp_actor_worker.py: a record with "sub" counts each attempted subtask; the weights file receives
       Manager.env_weights().
Without a "calvin_chains" tasks.json everything is byte-identical to before (checked by calvin_manager_test.py).
Usage: agent_manager_calvin_patch.py <RLinf root>   (idempotent; needs the robocasa, v3 and replay patches first)."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
ENV = root / "rlinf/envs/sim/calvin/calvin_gym_env.py"
ACT = root / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
AMF = root / "rlinf/b1k_tools/agent_manager.py"
MARK = "agent manager calvin (local addition 2026-11-06)"


def sub(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: anchor found {n} times")
    return text.replace(old, new)


# ---------------------------------------------------------------- env
e = ENV.read_text()
if MARK not in e:
    e = sub(e, """        self.video_cfg = cfg.video_cfg
""", """        self.video_cfg = cfg.video_cfg
        # agent manager calvin (local addition 2026-11-06): subtask index + chain pool for the manager
        if not bool(self.cfg.get("is_eval", False)) and self.cfg.get("task_stats_dir", None):
            self._calvin_write_tasks_json()
""", "env init")
    e = sub(e, """        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if reset_state_ids is None:
""", """        if env_idx is None:
            env_idx = np.arange(self.num_envs)
            # agent manager calvin (local addition 2026-11-06): a full reset starts a rollout epoch
            if not bool(self.cfg.get("is_eval", False)) and self.cfg.get("task_stats_dir", None):
                self._ts_recorded = np.zeros(self.num_envs, dtype=bool)

        if reset_state_ids is None:
""", "env reset")
    e = sub(e, """    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
""", """    def _get_random_reset_state_ids(self, num_reset_states):
        _aw = self._agent_task_weights()  # agent manager calvin (local addition 2026-11-06): chain weights
        if _aw is not None:
            w = np.array([max(float(_aw.get(i, 1.0)), 0.0) for i in range(self.total_num_group_envs)], dtype=np.float64)
            if w.sum() <= 0:
                w[:] = 1.0
            return self._generator.choice(self.total_num_group_envs, size=num_reset_states, p=w / w.sum()).astype(np.int64)
        reset_state_ids = self._generator.integers(
""", "env sampling")
    e = sub(e, """        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)
""", """        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        # agent manager calvin (local addition 2026-11-06): per-subtask training statistics, before any auto-reset
        if not bool(self.cfg.get("is_eval", False)) and self.cfg.get("task_stats_dir", None):
            self._record_task_stats(past_dones.cpu().numpy())
""", "env chunk_step")
    e = sub(e, """    def _calc_step_reward(self, terminations):
""", """    # ---------------- agent manager calvin (local addition 2026-11-06) ----------------
    def _calvin_write_tasks_json(self):
        \"\"\"Sorted subtask-type names and the chain pool as subtask indices -> <task_stats_dir>/tasks.json.\"\"\"
        import json as _json

        chains = [list(self.task_suite.get_task_sequence(t)) for t in range(self.total_num_group_envs)]
        names = sorted({s for c in chains for s in c})
        self._calvin_sub_index = {s: i for i, s in enumerate(names)}
        d = self.cfg.get("task_stats_dir")
        os.makedirs(d, exist_ok=True)
        tj = os.path.join(d, "tasks.json")
        tmp = f"{tj}.{os.getpid()}.tmp"
        suite = {"calvin_abc": "CALVIN ABC->D (training in scenes A, B, C)", "calvin_d": "CALVIN D->D",
                 "calvin_abcd": "CALVIN ABCD->D"}.get(str(self.cfg.task_suite_name), "CALVIN")
        with open(tmp, "w") as f:
            _json.dump({"suite": suite, "kind": "calvin_chains", "tasks": names,
                        "chains": [[self._calvin_sub_index[s] for s in c] for c in chains]}, f)
        os.replace(tmp, tj)

    def _agent_task_weights(self):
        \"\"\"Chain sampling weights {chain id: weight} from env.train.agent_weights_file (train envs only).\"\"\"
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

    def _record_task_stats(self, dones):
        \"\"\"First episode end of each env in this rollout epoch -> <task_stats_dir>/env_<pid>.jsonl: the attempted
        subtasks of its chain (completed ones = 1, the failed or truncated one = 0).\"\"\"
        import json as _json

        if not hasattr(self, "_calvin_sub_index"):
            self._calvin_write_tasks_json()
        rec = getattr(self, "_ts_recorded", None)
        if rec is None or len(rec) != self.num_envs:
            rec = self._ts_recorded = np.zeros(self.num_envs, dtype=bool)
        new = np.asarray(dones, dtype=bool) & ~rec
        if not new.any():
            return
        d = self.cfg.get("task_stats_dir")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"env_{os.getpid()}.jsonl"), "a") as f:
            for i in np.nonzero(new)[0]:
                chain = [self._calvin_sub_index[s] for s in self.task_sequence[i]]
                k = int(self.current_task_idx[i])
                subs = [[chain[j], 1] for j in range(min(k, 5))] + ([[chain[k], 0]] if k < 5 else [])
                f.write(_json.dumps({"task": int(subs[-1][0]), "s": int(k >= 5), "sub": subs,
                                     "trial": int(self.trial_ids[i]), "done": k}) + "\\n")
        rec |= new

    def _calc_step_reward(self, terminations):
""", "env methods")
    ENV.write_text(e)
    print("env patched")
else:
    print("env already patched")

# ---------------------------------------------------------------- actor
a = ACT.read_text()
if MARK not in a:
    a = sub(a, """            counts = {}
            for r in recs:
                v = counts.setdefault(int(r["task"]), [0, 0])
                v[0] += 1
                v[1] += int(r["s"])
            weights, m, _info = self._am.decide({t: tuple(v) for t, v in counts.items()})
            write_weights(self.cfg.env.train.agent_weights_file, weights)
""", """            counts = {}
            for r in recs:
                # agent manager calvin (local addition 2026-11-06): a chain record counts every attempted subtask
                for t, s in (r["sub"] if "sub" in r else [(r["task"], r["s"])]):
                    v = counts.setdefault(int(t), [0, 0])
                    v[0] += 1
                    v[1] += int(s)
            weights, m, _info = self._am.decide({t: tuple(v) for t, v in counts.items()})
            write_weights(self.cfg.env.train.agent_weights_file,
                          self._am.env_weights() if hasattr(self._am, "env_weights") else weights)
""", "actor counts")
    ACT.write_text(a)
    print("actor patched")
else:
    print("actor already patched")

# ---------------------------------------------------------------- manager
m = AMF.read_text()
if MARK not in m:
    m = sub(m, """def schema_version(n, version="v2"):""", '''# agent manager calvin (local addition 2026-11-06): CALVIN prompts (chains of subtasks; same statistics and fields)
PROMPT_CALVIN = """You manage online reinforcement-learning fine-tuning (PPO) of ONE robot policy on {suite}. Every training episode is a chain of 5 language-instructed subtasks of the {ntask} subtask types below; the episode stops at the first failed subtask, so the later subtasks of a chain are attempted only when the earlier ones succeed. Goal: the highest final number of completed subtasks per chain; learning weak subtask types matters as much as not losing the ones the policy can already do.
After every iteration you set (1) the practice weight of each subtask type for the next iteration and (2) a learning-rate multiplier for the next update (0.5, 0.75, 1, 1.25 or 1.5; the cumulative learning rate is kept within 0.25x..2x of the default).
The weights are RELATIVE: they are rescaled to mean 1 after your answer, and the {eps} training chains of the next iteration are drawn from a fixed pool of {nchains} chains so that the expected number of attempts of each subtask type is proportional to its weight, as far as the pool allows. Use the range 0.25..4: 1 = normal share of attempts, 2 = double, 0.5 = half.
Facts from earlier experiments with this kind of setup (use them):
- Subtask types get very different numbers of attempts per iteration (from about 2 to about 50), so per-type counts are very noisy, above all for rare types; judge changes from pooled windows and the statistics given, not from a single iteration.
- Subtask types with low success so far can still be learned through exploration; do not give up on them.
- Being too conservative (tiny learning rates, ignoring weak subtask types) was the main failure of earlier automatic managers; halving the learning rate repeatedly freezes learning.
- With {ntask} subtask types, a z around 1.6 for some type is expected by chance in most iterations; a real drop shows up as a large z that persists over iterations.

Subtask types:
{tasks}

Per-subtask-type training success (successes/attempts) of the last iterations, oldest first:
{history}

Statistics: the recent window (last 3 iterations, pooled) against the previous 3 iterations (prev) and against the first 3 iterations (start); drop > 0 means worse now, z is the one-sided two-proportion z for a drop:
{stats}

Current relative weights: {weights}; current cumulative learning-rate scale: {lr_scale:.3f}; iteration {it} of {total}.
For each subtask type also give your probability (0..1) that its true success rate is now really lower than in the previous window (give 0 when there is no previous window yet). Answer with JSON only."""
PROMPT_V3_CALVIN = """You manage online reinforcement-learning fine-tuning (PPO) of ONE robot policy on {suite}. Every training episode is a chain of 5 language-instructed subtasks of the {ntask} subtask types below; the episode stops at the first failed subtask, so the later subtasks of a chain are attempted only when the earlier ones succeed. Goal: the highest final number of completed subtasks per chain; learning weak subtask types matters as much as not losing the ones the policy can already do.
After every iteration you set (1) the practice weight of each subtask type for the next iteration and (2) a learning-rate multiplier for the next update (0.75, 1 or 1.25; the cumulative learning rate is kept within 0.5x..1.25x of the default).
The weights are RELATIVE: they are rescaled to mean 1 after your answer, and the {eps} training chains of the next iteration are drawn from a fixed pool of {nchains} chains so that the expected number of attempts of each subtask type is proportional to its weight, as far as the pool allows. Use the range 0.5..2: 1 = normal share of attempts, 2 = double, 0.5 = half.
Facts from earlier experiments with this kind of setup (use them):
- Subtask types get very different numbers of attempts per iteration (from about 2 to about 50), so per-type counts are very noisy, above all for rare types; judge changes from pooled windows and the statistics given, not from a single iteration.
- Subtask types with low success so far can still be learned through exploration; do not give up on them.
- In earlier runs of this kind of setup, raising the learning rate to 1.5-2x the default coincided with broad drops in success, and halving it repeatedly froze learning; keep the default unless the statistics clearly call for a change.
- A task that had collapsed to near-zero success did not recover by sampling it much more; moderate extra practice (up to 2x) is enough.
- With {ntask} subtask types, a z around 1.6 for some type is expected by chance in most iterations; a real drop shows up as a large z that persists over iterations.

Subtask types:
{tasks}

Per-subtask-type training success (successes/attempts) of the last iterations, oldest first:
{history}

Statistics: the recent window (last 3 iterations, pooled) against the previous 3 iterations (prev) and against the first 3 iterations (start); drop > 0 means worse now, z is the one-sided two-proportion z for a drop:
{stats}

Current relative weights: {weights}; current cumulative learning-rate scale: {lr_scale:.3f}; iteration {it} of {total}.
For each subtask type also give your probability (0..1) that its true success rate is now really lower than in the previous window (give 0 when there is no previous window yet). Answer with JSON only."""


def chain_weights(chains, w, p, iters=60, lo=0.1, hi=10.0):
    """agent manager calvin: chain sampling weights (mean 1) whose expected attempts per subtask type are proportional
    to w[type] x the attempts under uniform sampling. Reach probabilities come from the per-type success estimates p
    (the k-th subtask of a chain is attempted with probability prod of p over the earlier ones). Multiplicative raking:
    each chain is scaled by the attempt-weighted geometric mean of its types' target/current ratios; chain weights are
    clipped to [lo, hi]. Returns (weights, achieved attempt-share ratio per type vs uniform sampling)."""
    S, N = len(w), len(chains)
    A = []
    for c in chains:
        r, a = 1.0, {}
        for s in c:
            a[s] = a.get(s, 0.0) + r
            r *= p[s]
        A.append(a)

    def mass(pi):
        out = [0.0] * S
        for x, a in zip(pi, A):
            for s, v in a.items():
                out[s] += x * v
        return out

    f = mass([1.0] * N)
    g = [max(float(w[s]), 0.0) * f[s] for s in range(S)]
    tg = sum(g) or 1.0
    pi = [1.0] * N
    for _ in range(iters):
        mm = mass(pi)
        tm = sum(mm) or 1.0
        lr = [math.log((g[s] / tg) / (mm[s] / tm)) if mm[s] > 0 and g[s] > 0 else 0.0 for s in range(S)]
        pi = [x * math.exp(sum(v * lr[s] for s, v in a.items()) / max(sum(a.values()), 1e-12)) for x, a in zip(pi, A)]
        mean = sum(pi) / N
        pi = [min(hi, max(lo, x / mean)) for x in pi]
    mean = sum(pi) / N
    pi = [x / mean for x in pi]
    mm, tf = mass(pi), sum(f) or 1.0
    tm = sum(mm) or 1.0
    ach = [((mm[s] / tm) / (f[s] / tf)) if f[s] > 0 else 1.0 for s in range(S)]
    return pi, ach


def schema_version(n, version="v2"):''', "manager prompts")
    m = sub(m, """                self.task_desc, self.suite = [str(x) for x in d["tasks"]], str(d.get("suite", "robot"))
                ntask = len(self.task_desc)
""", """                self.task_desc, self.suite = [str(x) for x in d["tasks"]], str(d.get("suite", "robot"))
                ntask = len(self.task_desc)
                if d.get("kind") == "calvin_chains":  # agent manager calvin (local addition 2026-11-06)
                    self.chains = [[int(s) for s in c] for c in d["chains"]]
""", "manager init")
    m = sub(m, """        self.stats_dir, self.eps = stats_dir, int(episodes_per_iter)
""", """        self.stats_dir, self.eps = stats_dir, int(episodes_per_iter)
        self.chains, self.chain_w = None, None  # agent manager calvin (local addition 2026-11-06)
""", "manager chains attr")
    m = sub(m, """            hist_rows = "\\n".join(
                f"task {t}: " + ", ".join(f"{h.get(t, (0, 0))[1]}/{h.get(t, (0, 0))[0]}" for h in self.hist[-6:])
                for t in range(self.ntask))
            stat_rows = "\\n".join(
                f"task {s['task']}: recent""", """            unit = "subtask" if self.chains else "task"  # agent manager calvin (local addition 2026-11-06)
            hist_rows = "\\n".join(
                f"{unit} {t}: " + ", ".join(f"{h.get(t, (0, 0))[1]}/{h.get(t, (0, 0))[0]}" for h in self.hist[-6:])
                for t in range(self.ntask))
            stat_rows = "\\n".join(
                f"{unit} {s['task']}: recent""", "manager rows")
    m = sub(m, """            prompt = globals()[self.V["prompt"]].format(""", """            prompt = globals()[self.V["prompt"] + ("_CALVIN" if self.chains else "")].format(nchains=len(self.chains or ()),""", "manager prompt choice")
    m = sub(m, """        new_scale = min(self.V["scale"][1], max(self.V["scale"][0], self.lr_scale * mult))
""", """        if self.chains:  # agent manager calvin (local addition 2026-11-06): type weights -> chain sampling weights
            k = len(self.hist)
            pr = []
            tot_n = sum(pooled(self.hist, t, max(0, k - 3), k)[0] for t in range(self.ntask))
            tot_s = sum(pooled(self.hist, t, max(0, k - 3), k)[1] for t in range(self.ntask))
            pbar = (tot_s + 1) / (tot_n + 2)
            for t in range(self.ntask):
                n, s = pooled(self.hist, t, max(0, k - 3), k)
                pr.append((s + 2 * pbar) / (n + 2))
            self.chain_w, ach = chain_weights(self.chains, self.weights, pr)
            ess = sum(self.chain_w) ** 2 / max(sum(x * x for x in self.chain_w), 1e-12) / len(self.chain_w)
            info["chain_weights"] = {"min": round(min(self.chain_w), 3), "max": round(max(self.chain_w), 3),
                                     "ess_frac": round(ess, 3), "attempt_share_vs_uniform": [round(x, 3) for x in ach]}
        new_scale = min(self.V["scale"][1], max(self.V["scale"][0], self.lr_scale * mult))
""", "manager chain weights")
    m = sub(m, """    def decide(self, counts):
""", """    def env_weights(self):
        \"\"\"agent manager calvin (local addition 2026-11-06): what the train envs sample by - chain weights for
        CALVIN (uniform before the first decision), the task weights otherwise.\"\"\"
        if self.chains:
            return self.chain_w if self.chain_w is not None else [1.0] * len(self.chains)
        return self.weights

    def decide(self, counts):
""", "manager env_weights")
    AMF.write_text(m)
    print("manager patched")
else:
    print("manager already patched")
