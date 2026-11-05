#!/usr/bin/env python3
"""RoboCasa365 progress reward (potential-based shaping), local addition 2026-11-05 — the RoboCasa counterpart of the
LIBERO subgoal reward (subgoal_patch.py; CALVIN already rewards every completed subtask).

RLinf's RoboCasa reward is success-only. Phi(s) in [0, 1] is read from the task's own state, the same quantities its
_check_success() thresholds (1.0 once the task succeeds):
  SlideDishwasherRack   0.9 * rack position (toward the requested direction)      success: rack >= 0.95 (or <= 0.05)
  OpenStandMixerHead    0.9 * head joint state                                    success: head > 0.99
  TurnOnElectricKettle  0.9 * switch joint / 0.95                                 success: switch >= 0.95 latches on
  TurnOnSinkFaucet      0.9 * handle joint / 0.40 rad                             success: 0.40 < handle < pi
  PickPlaceCounterToStove, CoffeeSetupMug: 0.3 once the object is grasped, 0.6 once it is in the receptacle / placed
                        for pouring, 1.0 on success (placed + gripper released)
  other tasks: None (no shaping). Only the 6 atomic training tasks of the RoboCasa PPO runs need a model.
Each train step adds coef * (Phi_t - Phi_{t-1}); the first step after a (auto-)reset adds 0, so the episode bonus
telescopes to coef * (Phi_end - Phi_first). Eval envs never shape. Opt-in: +env.train.subgoal_reward_coef=<coef>; the
env sets B1K_RC_PROGRESS=1 before its subprocesses start, and only then do they compute Phi (info["progress"]).
Usage: robocasa_progress_patch.py <RLinf root>   (idempotent; refuses if an anchor is not found exactly once)
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
VENV = root / "rlinf/envs/sim/robocasa/venv.py"
ENV = root / "rlinf/envs/sim/robocasa365/robocasa365_env.py"
MARK = "progress reward (local addition 2026-11-05)"


def sub(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: anchor found {n} times")
    return text.replace(old, new)


v = VENV.read_text()
if MARK not in v:
    v = sub(v, "from multiprocessing import Pipe, connection\n", "import os\nfrom multiprocessing import Pipe, connection\n",
            "venv import os")
    v = sub(v, """def _worker(
    parent: connection.Connection,""", '''def _rc_progress(env):
    """progress reward (local addition 2026-11-05): task progress in [0, 1] from the task's own state (1.0 on success,
    None for tasks without a progress model). See analysis/rsi_20261104/robocasa_progress_patch.py."""
    try:
        if env._check_success():
            return 1.0
        name = type(env).__name__

        def clip(x):
            return float(min(max(float(x), 0.0), 1.0))

        if name == "SlideDishwasherRack":
            pos = float(env.dishwasher.get_state(env)["rack"])
            return 0.9 * clip(pos if env.should_pull else 1.0 - pos)
        if name == "OpenStandMixerHead":
            return 0.9 * clip(env.stand_mixer.get_state(env)["head"])
        if name == "TurnOnElectricKettle":
            return 0.9 * clip((env.electric_kettle.get_state(env).get("switch") or 0.0) / 0.95)
        if name == "TurnOnSinkFaucet":
            return 0.9 * clip(env.sink.get_handle_state(env=env).get("handle_joint", 0.0) / 0.40)
        from robocasa.utils import object_utils as OU

        if name == "PickPlaceCounterToStove":
            placed = OU.check_obj_in_receptacle(env, "obj", "container", th=0.07)
        elif name == "CoffeeSetupMug":
            placed = env.coffee_machine.check_receptacle_placement_for_pouring(env, "obj")
        else:
            return None
        if placed:
            return 0.6
        return 0.3 if OU.check_obj_grasped(env, "obj") else 0.0
    except Exception:
        return None


def _worker(
    parent: connection.Connection,''', "venv helper")
    v = sub(v, """                if hasattr(env, "_check_success"):
                    env_return = _check_success(env, env_return)""", """                if hasattr(env, "_check_success"):
                    env_return = _check_success(env, env_return)
                # progress reward (local addition 2026-11-05): only when the train env asked for it
                if os.environ.get("B1K_RC_PROGRESS") == "1":
                    env_return[-1]["progress"] = _rc_progress(env)""", "venv step")
    VENV.write_text(v)
    print("venv patched")
else:
    print("venv already patched")

e = ENV.read_text()
if MARK not in e:
    e = sub(e, "import copy\n", "import copy\nimport os\n", "env import os")
    e = sub(e, """        self._init_reset_state_ids()
        self.update_reset_state_ids()
        self._init_env()
""", """        self._init_reset_state_ids()
        self.update_reset_state_ids()
        # progress reward (local addition 2026-11-05): potential-based shaping from the task's own state (train only)
        self.subgoal_coef = float(cfg.get("subgoal_reward_coef", 0.0) or 0.0)
        self._prev_progress = None
        if self.subgoal_coef > 0 and not bool(cfg.get("is_eval", False)):
            os.environ["B1K_RC_PROGRESS"] = "1"  # read by the env subprocesses (rlinf/envs/sim/robocasa/venv.py)
        self._init_env()
""", "env init")
    e = sub(e, """        if isinstance(env_idx, int):
            env_idx = [env_idx]

        raw_obs, info_list = self.env.reset(id=env_idx)""", """        if isinstance(env_idx, int):
            env_idx = [env_idx]
        if getattr(self, "_prev_progress", None) is not None:  # progress reward: no shaping across a reset
            self._prev_progress[np.asarray(env_idx, dtype=np.int64)] = np.nan

        raw_obs, info_list = self.env.reset(id=env_idx)""", "env reset")
    e = sub(e, """        step_reward = self._calc_step_reward(terminations)

        infos = list_of_dict_to_dict_of_list(info_lists)""", """        step_reward = self._calc_step_reward(terminations)
        # progress reward (local addition 2026-11-05): coef * (Phi_t - Phi_{t-1}) on train envs
        if getattr(self, "subgoal_coef", 0.0) > 0 and not bool(self.cfg.get("is_eval", False)):
            prog = np.array([np.nan if info.get("progress") is None else float(info["progress"]) for info in info_lists],
                            dtype=np.float32)
            if self._prev_progress is None or len(self._prev_progress) != len(prog):
                self._prev_progress = np.full(len(prog), np.nan, dtype=np.float32)
            shaping = np.where(np.isnan(self._prev_progress) | np.isnan(prog), 0.0, prog - self._prev_progress)
            step_reward = step_reward + self.subgoal_coef * shaping.astype(np.float32)
            self._prev_progress = prog

        infos = list_of_dict_to_dict_of_list(info_lists)""", "env step")
    ENV.write_text(e)
    print("env patched")
else:
    print("env already patched")
