#!/usr/bin/env python3
"""LIBERO subgoal progress reward (potential-based shaping), local addition 2026-11-05 (user-chosen after ESPO failed).

Phi(s) = fraction of satisfied BDDL goal predicates (parsed_problem['goal_state'] evaluated by the problem's own
_eval_predicate, i.e. exactly the conjuncts of LIBERO's success check). At the end of every action chunk of a TRAIN env
the reward of the chunk's last step gets  coef * (Phi_end - Phi_prev); the first chunk after a reset adds 0, so the
per-episode bonus telescopes to coef * (Phi_final - Phi_after_first_chunk). Default off: +env.train.subgoal_reward_coef.
Queries go through a new subprocess command 'goal_frac' (the 'reconfigure' command rebuilds envs without env_fn, so a
wrapper around the env would be lost after the first training reset).
Usage: subgoal_patch.py <RLinf root>   (idempotent; refuses if an anchor is not found exactly once)
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
VENV = root / "rlinf/envs/sim/libero/venv.py"
ENV = root / "rlinf/envs/sim/libero/libero_env.py"
MARK = "subgoal reward (local addition 2026-11-05)"


def sub(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: anchor found {n} times")
    return text.replace(old, new)


v = VENV.read_text()
if MARK not in v:
    v = sub(v, """def _worker(
    parent: connection.Connection,""", """def _goal_frac(env) -> float:
    \"\"\"subgoal reward (local addition 2026-11-05): fraction of satisfied BDDL goal predicates of a LIBERO env.\"\"\"
    dom = getattr(env, "env", env)
    goal = dom.parsed_problem["goal_state"]
    return float(sum(bool(dom._eval_predicate(g)) for g in goal)) / max(len(goal), 1)


def _worker(
    parent: connection.Connection,""", "venv helper")
    v = sub(v, """            elif cmd == "check_success":
                p.send(env.check_success())""", """            elif cmd == "check_success":
                p.send(env.check_success())
            elif cmd == "goal_frac":
                p.send(_goal_frac(env))""", "venv worker cmd")
    v = sub(v, """    def reconfigure_env_fns(self, env_fns, id=None):""", """    def goal_frac(self) -> np.ndarray:
        \"\"\"subgoal reward (local addition 2026-11-05): goal-predicate fraction of every env (queries sent first,
        then collected, so the workers evaluate in parallel).\"\"\"
        for w in self.workers:
            w.parent_remote.send(["goal_frac", None])
        return np.array([w.parent_remote.recv() for w in self.workers], dtype=np.float32)

    def reconfigure_env_fns(self, env_fns, id=None):""", "venv method")
    VENV.write_text(v)
    print("venv patched")
else:
    print("venv already patched")

e = ENV.read_text()
if MARK not in e:
    e = sub(e, """        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward""", """        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        # subgoal reward (local addition 2026-11-05): potential-based shaping from the BDDL goal predicates
        self.subgoal_coef = float(cfg.get("subgoal_reward_coef", 0.0) or 0.0)
        self._prev_goal_frac = None""", "env init")
    e = sub(e, """                self._ts_recorded = np.zeros(self.num_envs, dtype=bool)

        if self.is_start:""", """                self._ts_recorded = np.zeros(self.num_envs, dtype=bool)
        if getattr(self, "_prev_goal_frac", None) is not None:  # subgoal reward: no shaping across a reset
            self._prev_goal_frac[np.asarray(env_idx)] = np.nan

        if self.is_start:""", "env reset (after the accept_rollback block)")
    e = sub(e, """        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None""", """        # subgoal reward (local addition 2026-11-05): coef * (Phi_end - Phi_prev) on the chunk's last step
        if not self.is_eval and getattr(self, "subgoal_coef", 0.0) > 0:
            gf = self.env.goal_frac()
            if self._prev_goal_frac is None or len(self._prev_goal_frac) != len(gf):
                self._prev_goal_frac = np.full(len(gf), np.nan, dtype=np.float32)
            shaping = np.where(np.isnan(self._prev_goal_frac), 0.0, gf - self._prev_goal_frac).astype(np.float32)
            chunk_rewards[:, -1] = chunk_rewards[:, -1] + self.subgoal_coef * torch.as_tensor(
                shaping, dtype=chunk_rewards.dtype, device=chunk_rewards.device)
            self._prev_goal_frac = gf.copy()

        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None""", "env chunk_step")
    ENV.write_text(e)
    print("env patched")
else:
    print("env already patched")
