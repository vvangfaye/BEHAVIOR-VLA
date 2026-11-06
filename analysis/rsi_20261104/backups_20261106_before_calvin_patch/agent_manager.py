"""Agent-managed RL training (agent manager robocasa (local addition 2026-11-05); local addition 2026-11-05; user-chosen direction: "an agent in the RL loop").

After every PPO iteration the training manager reads the per-task success counts of the latest training rollout
(written by the train envs to env.train.task_stats_dir, one record per env per rollout epoch) and decides
  * task sampling weights for the next rollout (relative, clipped to [0.25, 4]; read by the LIBERO train env from
    env.train.agent_weights_file at every reset), and
  * a learning-rate multiplier for the next update (from {0.5, 0.75, 1, 1.25, 1.5}; the cumulative scale stays within
    [0.25, 2] x the configured lr).
Modes (algorithm.agent_manager.mode):
  agent   gpt-6.1-sol through the local Codex CLI (same restrictions and daily ledger as the Sol verifier); the prompt
          carries precomputed statistics, asks for regression probabilities (thresholded outside: p >= 0.5 doubles the
          weight) and states the priors learned in the offline capability tests (2026-11-05): small samples are
          noisy, and tasks with zero successes can still be learned through exploration.
  relay   same prompt and post-processing as agent, for cluster jobs without internet: the prompt goes to AOSS through a
          presigned PUT URL and the answer comes back through a presigned GET URL (env B1K_RELAY_URLS_B64, one request/
          answer object pair per stage, key = the stage directory name = parent of out_dir); the local relay
          (analysis/rsi_20261104/sol_relay.py) calls Sol. No answer within B1K_RELAY_TIMEOUT s (default 900) = no change.
  rule    z-test manager: weight x2 for a task whose pooled recent rate dropped below its best by > 0.15 with z > 1.64,
          x0.5 for tasks above 90%, x1.5 for tasks between 10% and 70%; lr unchanged.
  random  log-uniform weights in [0.5, 2] and a random lr multiplier (seeded) - control for "any perturbation helps".
  none    weights 1, lr 1 (logs only).
Everything the manager sees is training-rollout data (never evaluation results).
"""
from __future__ import annotations

import datetime
import json
import math
import os
import random
import re
import subprocess
import time
from pathlib import Path

LIBERO10 = [
    "put both the alphabet soup and the tomato sauce in the basket",
    "put both the cream cheese box and the butter in the basket",
    "turn on the stove and put the moka pot on it",
    "put the black bowl in the bottom drawer of the cabinet and close it",
    "put the white mug on the left plate and put the yellow and white mug on the right plate",
    "pick up the book and place it in the back compartment of the caddy",
    "put the white mug on the plate and put the chocolate pudding to the right of the plate",
    "put both the alphabet soup and the cream cheese box in the basket",
    "put both moka pots on the stove",
    "put the yellow and white mug in the microwave and close it",
]
LR_CHOICES = (0.5, 0.75, 1.0, 1.25, 1.5)
CODEX = os.environ.get("B1K_CODEX", "/home/faye/.local/bin/codex")
LEDGER = Path(os.environ.get("B1K_SOL_LEDGER", "/mnt/data1/faye/code/Behavior-1k Challenge/analysis/rsi_20261104/sol_ledger.json"))
DAILY_CAP = 500
DISABLED = ("shell_tool", "apps", "plugins", "multi_agent", "multi_agent_v2", "browser_use", "browser_use_external",
            "computer_use", "image_generation", "code_mode", "code_mode_host", "view_image", "goals", "sleep_tool",
            "skill_search", "tool_suggest", "workspace_dependencies")
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["task_weights", "lr_multiplier", "prob_regressed", "reason"],
          "properties": {"task_weights": {"type": "array", "items": {"type": "number", "minimum": 0.25, "maximum": 4},
                                          "minItems": 10, "maxItems": 10},
                         "lr_multiplier": {"type": "number", "minimum": 0.5, "maximum": 1.5},
                         "prob_regressed": {"type": "array", "items": {"type": "number", "minimum": 0, "maximum": 1},
                                            "minItems": 10, "maxItems": 10},
                         "reason": {"type": "string", "maxLength": 400}}}
def schema_for(n):
    """agent manager robocasa (local addition 2026-11-05): SCHEMA with n tasks (schema_for(10) == SCHEMA)."""
    s = json.loads(json.dumps(SCHEMA))
    for k in ("task_weights", "prob_regressed"):
        s["properties"][k]["minItems"] = s["properties"][k]["maxItems"] = int(n)
    return s


MANAGER_VERSION = "v2"
# v2 regression threshold, from offline test A (11/5, 170 no-change / 20 true-drop task instances, probability prompt):
# p >= 0.94 -> false alarm 4.1% (7/170), recall 70%; v1 used p >= 0.5, which there had 51% false alarms (87/170) and
# flagged 6-10 of 10 tasks per iteration online.
REG_THRESHOLD = 0.94
PROMPT = """You manage online reinforcement-learning fine-tuning (PPO) of ONE robot policy on the {ntask} {suite} tasks below. Goal: the highest final success on ALL tasks; learning new tasks matters as much as not losing tasks the policy can already do.
After every iteration you set (1) the sampling weight of each task for the next iteration and (2) a learning-rate multiplier for the next update (0.5, 0.75, 1, 1.25 or 1.5; the cumulative learning rate is kept within 0.25x..2x of the default).
The weights are RELATIVE: they are rescaled to mean 1 after your answer, and {eps} training episodes per iteration are split in proportion to them. Use the range 0.25..4: 1 = normal share, 2 = double, 0.5 = half.
Facts from earlier experiments with this setup (use them):
- Each task gets only ~{eps_task} episodes per iteration, so per-task counts are very noisy; judge changes from pooled windows and the statistics given, not from a single iteration.
- Tasks with zero successes so far can still be learned through exploration; do not give up on them.
- Being too conservative (tiny learning rates, ignoring weak tasks) was the main failure of earlier automatic managers; halving the learning rate repeatedly freezes learning.
- With {ntask} tasks, a z around 1.6 for some task is expected by chance in most iterations; a real drop shows up as a large z that persists over iterations.

Tasks:
{tasks}

Per-task training success (successes/episodes) of the last iterations, oldest first:
{history}

Statistics: the recent window (last 3 iterations, pooled) against the previous 3 iterations (prev) and against the first 3 iterations (start); drop > 0 means worse now, z is the one-sided two-proportion z for a drop:
{stats}

Current relative weights: {weights}; current cumulative learning-rate scale: {lr_scale:.3f}; iteration {it} of {total}.
For each task also give your probability (0..1) that its true success rate is now really lower than in the previous window (give 0 when there is no previous window yet). Answer with JSON only."""

# manager v3 (local addition 2026-11-06): opt-in (algorithm.agent_manager.version=v3)
PROMPT_V3 = """You manage online reinforcement-learning fine-tuning (PPO) of ONE robot policy on the {ntask} {suite} tasks below. Goal: the highest final success on ALL tasks; learning new tasks matters as much as not losing tasks the policy can already do.
After every iteration you set (1) the sampling weight of each task for the next iteration and (2) a learning-rate multiplier for the next update (0.75, 1 or 1.25; the cumulative learning rate is kept within 0.5x..1.25x of the default).
The weights are RELATIVE: they are rescaled to mean 1 after your answer, and {eps} training episodes per iteration are split in proportion to them. Use the range 0.5..2: 1 = normal share, 2 = double, 0.5 = half.
Facts from earlier experiments with this setup (use them):
- Each task gets only ~{eps_task} episodes per iteration, so per-task counts are very noisy; judge changes from pooled windows and the statistics given, not from a single iteration.
- Tasks with zero successes so far can still be learned through exploration; do not give up on them.
- In earlier runs of this setup, raising the learning rate to 1.5-2x the default coincided with broad drops in success, and halving it repeatedly froze learning; keep the default unless the statistics clearly call for a change.
- A task that has collapsed to near-zero success did not recover by sampling it much more; moderate extra practice (up to 2x) is enough.
- With {ntask} tasks, a z around 1.6 for some task is expected by chance in most iterations; a real drop shows up as a large z that persists over iterations.

Tasks:
{tasks}

Per-task training success (successes/episodes) of the last iterations, oldest first:
{history}

Statistics: the recent window (last 3 iterations, pooled) against the previous 3 iterations (prev) and against the first 3 iterations (start); drop > 0 means worse now, z is the one-sided two-proportion z for a drop:
{stats}

Current relative weights: {weights}; current cumulative learning-rate scale: {lr_scale:.3f}; iteration {it} of {total}.
For each task also give your probability (0..1) that its true success rate is now really lower than in the previous window (give 0 when there is no previous window yet). Answer with JSON only."""
VERSIONS = {
    "v2": {"prompt": "PROMPT", "lr_choices": (0.5, 0.75, 1.0, 1.25, 1.5), "scale": (0.25, 2.0), "w": (0.25, 4.0), "bump": 2.0},
    "v3": {"prompt": "PROMPT_V3", "lr_choices": (0.75, 1.0, 1.25), "scale": (0.5, 1.25), "w": (0.5, 2.0), "bump": 1.5},
}


def schema_version(n, version="v2"):
    """schema_for(n) with the version's weight and lr bounds (v2: identical to schema_for)."""
    s = schema_for(n)
    if version == "v3":
        for k in ("minimum", "maximum"):
            s["properties"]["task_weights"]["items"][k] = VERSIONS["v3"]["w"][0 if k == "minimum" else 1]
        s["properties"]["lr_multiplier"]["minimum"], s["properties"]["lr_multiplier"]["maximum"] = 0.75, 1.25
    return s



def pooled(hist, t, lo, hi):
    n = sum(h.get(t, (0, 0))[0] for h in hist[lo:hi])
    s = sum(h.get(t, (0, 0))[1] for h in hist[lo:hi])
    return n, s


def _cmp(rn, rs, bn, bs):
    """Reference (bn, bs) minus recent (rn, rs): drop and one-sided pooled two-proportion z."""
    if not rn or not bn:
        return 0.0, 0.0
    pr, pb = rs / rn, bs / bn
    p = (rs + bs) / (rn + bn)
    se = math.sqrt(max(p * (1 - p) * (1 / rn + 1 / bn), 1e-12))
    return pb - pr, (pb - pr) / se


def task_stats(hist, ntask=10, window=3):
    """v2 (2026-11-06): the recent pooled window (last `window` iterations) against two disjoint, unselected references,
    the window just before it (prev) and the first window (start). v1 compared against each task's best window so far;
    a maximum over noisy windows is biased upward, and in the 11/5 pilot 6-10 of 10 tasks looked regressed every
    iteration from iteration 5 on."""
    out = []
    k = len(hist)
    for t in range(ntask):
        rn, rs = pooled(hist, t, max(0, k - window), k)
        qn, qs = pooled(hist, t, max(0, k - 2 * window), max(0, k - window))
        sn, ss = pooled(hist, t, 0, min(window, max(0, k - window)))
        dp, zp = _cmp(rn, rs, qn, qs)
        ds, zs = _cmp(rn, rs, sn, ss)
        out.append({"task": t, "recent": (rn, rs), "prev": (qn, qs), "start": (sn, ss), "drop_prev": dp, "z_prev": zp,
                    "drop_start": ds, "z_start": zs, "rate": rs / rn if rn else 0.0})
    return out


def normalize(w, lo=0.25, hi=4.0):
    """v2: relative weights, mean 1, clipped to [lo, hi] (two passes)."""
    w = [float(x) for x in w]
    for _ in range(2):
        m = sum(w) / max(len(w), 1)
        w = [min(hi, max(lo, x / m)) for x in w] if m > 0 else [1.0] * len(w)
    return w


def _ledger_reserve():
    import fcntl

    today = datetime.date.today().isoformat()
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(str(LEDGER) + ".lock", "w") as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            led = json.loads(LEDGER.read_text()) if LEDGER.exists() else {}
            if led.get(today, 0) >= DAILY_CAP:
                return False
            led[today] = led.get(today, 0) + 1
            LEDGER.write_text(json.dumps(led, indent=1))
            return True
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)


def _codex_args():
    env = {k: v for k, v in os.environ.items() if k not in {"CODEX_API_KEY", "OPENAI_API_KEY"}}
    args = ["--config", 'web_search="disabled"']
    for f in DISABLED:
        args += ["--disable", f]
    try:
        mcp = subprocess.run([CODEX, "mcp", "list", "--json", *args], env=env, capture_output=True, text=True, timeout=30)
        for s in json.loads(mcp.stdout or "[]"):
            name = s.get("name") if isinstance(s, dict) else None
            if isinstance(name, str) and re.fullmatch(r"[A-Za-z0-9_-]{1,256}", name):
                args += ["--config", f"mcp_servers.{name}.enabled=false"]
    except Exception:  # noqa: BLE001
        pass
    return env, args


def ask_agent(prompt, ev_dir: Path, timeout=180, schema=None):
    ev_dir.mkdir(parents=True, exist_ok=True)
    if not _ledger_reserve():
        return {"error": "daily_cap_reached"}
    env, restr = _codex_args()
    (ev_dir / "schema.json").write_text(json.dumps(schema or SCHEMA))
    (ev_dir / "prompt.txt").write_text(prompt)
    argv = [CODEX, "exec", "-", "--ephemeral", "--sandbox", "read-only", "--skip-git-repo-check",
            "--output-schema", str(ev_dir / "schema.json"), "--output-last-message", str(ev_dir / "last.txt"),
            "--color", "never", "--json", "--model", "gpt-6.1-sol", "--config", 'model_reasoning_effort="high"',
            "--config", 'service_tier="ultrafast"', *restr]
    t0 = time.time()
    try:
        subprocess.run(argv, input=prompt, env=env, cwd=ev_dir, capture_output=True, text=True, timeout=timeout)
        ans = json.loads((ev_dir / "last.txt").read_text())
        ans["_seconds"] = round(time.time() - t0, 1)
        return ans
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}"}


def ask_relay(prompt, ev_dir: Path, key: str, timeout: float | None = None, poll_s: float = 15.0, schema=None):
    """Cluster side of the AOSS relay: PUT {nonce, prompt} to the stage's request object, poll the answer object until
    it carries the same nonce (the previous iteration's answer stays there until the relay overwrites it)."""
    import base64
    import urllib.request
    import uuid

    ev_dir.mkdir(parents=True, exist_ok=True)
    try:
        urls = json.loads(base64.b64decode(os.environ.get("B1K_RELAY_URLS_B64", "")).decode() or "{}").get(key)
    except Exception:  # noqa: BLE001
        urls = None
    if not urls:
        return {"error": "relay_unconfigured"}
    timeout = float(os.environ.get("B1K_RELAY_TIMEOUT", 900)) if timeout is None else timeout
    nonce = uuid.uuid4().hex
    (ev_dir / "prompt.txt").write_text(prompt)
    body = json.dumps({"nonce": nonce, "time": time.time(), "key": key, "prompt": prompt, "schema": schema or SCHEMA}).encode()
    t0 = time.time()
    try:
        urllib.request.urlopen(urllib.request.Request(urls["req_put"], data=body, method="PUT"), timeout=60).read()
    except Exception as e:  # noqa: BLE001
        return {"error": f"relay_put_{type(e).__name__}"}
    while time.time() - t0 < timeout:
        time.sleep(poll_s)
        try:
            with urllib.request.urlopen(urls["ans_get"], timeout=60) as r:
                a = json.loads(r.read().decode())
        except Exception:  # noqa: BLE001  (404 until the first answer exists)
            continue
        if a.get("nonce") == nonce:
            ans = a.get("answer") or {"error": "relay_empty"}
            ans["_seconds"] = round(time.time() - t0, 1)
            return ans
    return {"error": "relay_timeout"}


class Manager:
    def __init__(self, mode, out_dir, total_iters, seed=0, ntask=10, stats_dir=None, episodes_per_iter=128, version="v2",
                 replay_from=None, replay_lr=True, replay_weights=False):
        # manager replay (local addition 2026-11-06)
        self.replay_rows = [json.loads(l) for l in open(replay_from) if l.strip()] if replay_from else []
        self.replay_lr, self.replay_weights = bool(replay_lr), bool(replay_weights)
        self.version = version if version in VERSIONS else "v2"  # manager v3 (local addition 2026-11-06)
        self.V = VERSIONS[self.version]
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
        self.mode, self.out = mode, Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.total, self.ntask = int(total_iters), ntask
        self.hist = []  # list of {task: (n, s)}
        self.weights = [1.0] * ntask
        self.lr_scale = 1.0
        self.rng = random.Random(seed)
        # resume (local addition 2026-11-06, after the 00:54 machine crash): <out_dir>/resume_from.jsonl = the decisions
        # up to the checkpoint the training resumes from; history, weights and lr scale continue from its last row
        rf = self.out / "resume_from.jsonl"
        if rf.exists():
            rows = [json.loads(line) for line in rf.read_text().splitlines() if line.strip()]
            if rows:
                self.hist = [{int(t): tuple(v) for t, v in r["counts"].items()} for r in rows]
                self.weights = [float(x) for x in rows[-1]["weights"]]
                self.lr_scale = float(rows[-1]["lr_scale"])

    def decide(self, counts):
        """counts: {task: (n, s)} of the latest training rollout. Returns (weights, lr_multiplier, info)."""
        self.hist.append({int(t): tuple(v) for t, v in counts.items()})
        it = len(self.hist)
        st = task_stats(self.hist, self.ntask)
        info = {"iteration": it, "counts": {str(t): list(v) for t, v in sorted(self.hist[-1].items())},
                "stats": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()} for s in st]}
        mult = 1.0
        if self.mode in ("agent", "relay"):
            hist_rows = "\n".join(
                f"task {t}: " + ", ".join(f"{h.get(t, (0, 0))[1]}/{h.get(t, (0, 0))[0]}" for h in self.hist[-6:])
                for t in range(self.ntask))
            stat_rows = "\n".join(
                f"task {s['task']}: recent {s['recent'][1]}/{s['recent'][0]} ({s['rate']:.2f}); "
                + (f"prev {s['prev'][1]}/{s['prev'][0]}, drop {s['drop_prev']:+.2f}, z {s['z_prev']:+.2f}; " if s['prev'][0] else "prev n/a; ")
                + (f"start {s['start'][1]}/{s['start'][0]}, drop {s['drop_start']:+.2f}, z {s['z_start']:+.2f}" if s['start'][0] else "start n/a")
                for s in st)
            prompt = globals()[self.V["prompt"]].format(tasks="\n".join(f"{t}: {self.task_desc[t]}" for t in range(self.ntask)), history=hist_rows,
                                   stats=stat_rows, weights=[round(w, 2) for w in self.weights], lr_scale=self.lr_scale,
                                   it=it, total=self.total, eps=self.eps, eps_task=round(self.eps / max(self.ntask, 1)),
                                   ntask=self.ntask, suite=self.suite)
            if self.mode == "agent":
                ans = ask_agent(prompt, self.out / f"iter{it:03d}", schema=schema_version(self.ntask, self.version))
            else:  # relay: key = stage directory name (out_dir = <run>/<stage>/am)
                ans = ask_relay(prompt, self.out / f"iter{it:03d}", key=self.out.parent.name, schema=schema_version(self.ntask, self.version))
            info["agent"] = ans
            if "error" not in ans:
                lo, hi = self.V["w"]
                w = [min(hi, max(lo, float(x))) for x in ans["task_weights"]]
                pr = [float(x) for x in ans["prob_regressed"]]
                # calibrated decision outside the agent (capability test: thresholding probabilities works, its own
                # binary calls are too conservative): tasks with p >= REG_THRESHOLD (0.94, offline false alarm 4.1%) and
                # a previous window get at least twice the mean weight, at most the 3 most probable per iteration;
                # weights are renormalized to mean 1 (v1: p >= 0.5, no cap, no renormalization -> weights at the 4.0
                # cap = uniform sampling by iteration 6)
                mean_w = sum(w) / len(w)
                flag = sorted((p, t) for t, p in enumerate(pr) if p >= REG_THRESHOLD and st[t]["prev"][0] > 0)[-3:]
                for _p, t in flag:
                    w[t] = max(w[t], self.V["bump"] * mean_w)
                info["flagged"] = sorted(t for _p, t in flag)
                self.weights = normalize(w, *self.V["w"])
                mult = min(self.V["lr_choices"], key=lambda c: abs(c - float(ans["lr_multiplier"])))  # snap to the allowed set
        elif self.mode == "rule":
            w = []
            for s in st:
                x = 1.0
                if s["drop_prev"] > 0.15 and s["z_prev"] > 1.64:
                    x = 2.0
                elif s["rate"] > 0.9:
                    x = 0.5
                elif 0.1 <= s["rate"] <= 0.7:
                    x = 1.5
                w.append(x)
            self.weights = normalize(w, *self.V["w"])
        elif self.mode == "replay":  # manager replay (local addition 2026-11-06)
            if not self.replay_rows:
                raise RuntimeError("mode=replay needs algorithm.agent_manager.replay_from")
            row = self.replay_rows[min(it, len(self.replay_rows)) - 1]
            info["replayed_iteration"] = row.get("iteration")
            if self.replay_weights:
                self.weights = [float(x) for x in row["weights"]]
            if self.replay_lr:
                mult = float(row["lr_multiplier"])
        elif self.mode == "random":
            self.weights = normalize([math.exp(self.rng.uniform(math.log(0.5), math.log(2.0))) for _ in range(self.ntask)], *self.V["w"])
            mult = self.rng.choice(self.V["lr_choices"])
        new_scale = min(self.V["scale"][1], max(self.V["scale"][0], self.lr_scale * mult))
        mult = new_scale / self.lr_scale
        self.lr_scale = new_scale
        info.update(version=self.version, weights=[round(w, 3) for w in self.weights], lr_multiplier=round(mult, 4),
                    lr_scale=round(self.lr_scale, 4))
        with open(self.out / "decisions.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%H:%M:%S"), "mode": self.mode, **info}, default=str) + "\n")
        return self.weights, mult, info


def write_weights(path, weights):
    """Atomic write of {task: weight} for the train envs."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps({str(t): float(w) for t, w in enumerate(weights)}))
    os.replace(tmp, p)
