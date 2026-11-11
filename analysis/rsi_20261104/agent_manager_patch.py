#!/usr/bin/env python3
"""Hook the agent training manager (rlinf/b1k_tools/agent_manager.py) into RLinf (2026-11-05; opt-in, default off).

env   LIBERO train env: if env.train.agent_weights_file exists, every random reset samples the task with probability
      proportional to its weight (trial uniform within the task); eval envs never read it.
actor before each PPO update (rank 0): read the new per-task records of env.train.task_stats_dir (the accept_rollback
      env hook writes them), call Manager.decide, write the weights file, broadcast the lr multiplier and apply it to
      the optimizer lr and the scheduler base lr.
Overrides: +env.train.task_stats_dir=<dir> +env.train.agent_weights_file=<file>
           +algorithm.agent_manager.enable=True +algorithm.agent_manager.mode=agent|rule|random|none
           +algorithm.agent_manager.out_dir=<dir> [+algorithm.agent_manager.seed=0 +algorithm.agent_manager.wait_s=180]
Usage: agent_manager_patch.py <RLinf root>  (requires ar_patch.py for the task-stats env hook; idempotent)."""
import sys
from pathlib import Path

root = Path(sys.argv[1])
ENV = root / "rlinf/envs/sim/libero/libero_env.py"
ACT = root / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
MARK = "agent manager (local addition 2026-11-05)"


def sub(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: anchor found {n} times")
    return text.replace(old, new)


e = ENV.read_text()
if MARK not in e:
    assert "accept_rollback (local addition 2026-11-05)" in e, "apply ar_patch.py first (task-stats hook)"
    e = sub(e, """    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:""", '''    def _agent_task_weights(self):
        """agent manager (local addition 2026-11-05): task sampling weights from env.train.agent_weights_file."""
        path = None if self.is_eval else self.cfg.get("agent_weights_file", None)
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

    def _weighted_reset_state_ids(self, num_reset_states, weights):
        """agent manager: task ~ weights (restricted to the task filter), trial uniform within the task."""
        if self.task_id_filter is not None:
            tids = sorted({int(t) for t in self.task_id_filter})
        else:
            tids = list(range(len(self.trial_id_bins)))
        w = np.array([max(float(weights.get(t, 1.0)), 0.0) for t in tids], dtype=np.float64)
        if w.sum() <= 0:
            w[:] = 1.0
        tasks = self._generator.choice(np.array(tids), size=num_reset_states, p=w / w.sum())
        ids = []
        for t in tasks:
            t = int(t)
            start = int(self.cumsum_trial_id_bins[t - 1]) if t > 0 else 0
            ids.append(start + int(self._generator.integers(0, self.trial_id_bins[t])))
        return np.array(ids, dtype=np.int64)

    def _get_random_reset_state_ids(self, num_reset_states):
        _aw = None if self.specific_reset_id is not None else self._agent_task_weights()
        if _aw is not None:  # agent manager (local addition 2026-11-05)
            return self._weighted_reset_state_ids(num_reset_states, _aw)
        if self.specific_reset_id is not None:''', "env sampler")
    ENV.write_text(e)
    print("env patched")
else:
    print("env already patched")

a = ACT.read_text()
if MARK not in a:
    a = sub(a, """        # accept_rollback (local addition 2026-11-05): verified acceptance of the last update (see _ar_check)""",
            """        # agent manager (local addition 2026-11-05): per-iteration training manager (rlinf/b1k_tools/agent_manager.py)
        am_metrics = None
        if self.cfg.algorithm.get("agent_manager", {}).get("enable", False):
            am_metrics = self._agent_manage()

        # accept_rollback (local addition 2026-11-05): verified acceptance of the last update (see _ar_check)""",
            "actor call")
    a = sub(a, """        if ar_metrics is not None:
            mean_metric_dict.update(ar_metrics)
""", """        if ar_metrics is not None:
            mean_metric_dict.update(ar_metrics)
        if am_metrics is not None:
            mean_metric_dict.update(am_metrics)
""", "actor metrics")
    a = sub(a, """    # ---------------- accept_rollback (local addition 2026-11-05; RSI option 1) ----------------""",
            '''    # ---------------- agent manager (local addition 2026-11-05) ----------------
    def _am_read_new(self, d, expected, wait_s):
        """New (task, success) records of the train envs since the last call (own offsets; AR keeps its own)."""
        import glob
        import json as _json
        import time

        if not hasattr(self, "_am_offsets"):
            self._am_offsets = {}
        t0 = time.time()
        while True:
            recs = []
            offsets = dict(self._am_offsets)
            for path in sorted(glob.glob(f"{d}/env_*.jsonl")):
                with open(path) as f:
                    f.seek(offsets.get(path, 0))
                    chunk = f.read()
                cut = chunk.rfind("\\n") + 1
                for line in chunk[:cut].splitlines():
                    if line.strip():
                        recs.append(_json.loads(line))
                offsets[path] = offsets.get(path, 0) + len(chunk[:cut].encode())
            if len(recs) >= expected or time.time() - t0 > wait_s:
                self._am_offsets = offsets
                return recs
            time.sleep(2.0)

    def _agent_manage(self) -> dict:
        mcfg = self.cfg.algorithm.agent_manager
        if not hasattr(self, "_am"):
            from rlinf.b1k_tools.agent_manager import Manager

            self._am = Manager(str(mcfg.get("mode", "agent")), str(mcfg.get("out_dir")),
                               int(self.cfg.runner.max_epochs), seed=int(mcfg.get("seed", 0)))
        mult = torch.ones(1, device=self.device)
        if self._rank == 0:
            from rlinf.b1k_tools.agent_manager import write_weights

            d = self.cfg.env.train.get("task_stats_dir")
            expected = int(self.cfg.env.train.total_num_envs) * int(self.cfg.env.train.get("rollout_epoch", 1))
            recs = self._am_read_new(d, expected, float(mcfg.get("wait_s", 180)))
            counts = {}
            for r in recs:
                v = counts.setdefault(int(r["task"]), [0, 0])
                v[0] += 1
                v[1] += int(r["s"])
            weights, m, _info = self._am.decide({t: tuple(v) for t, v in counts.items()})
            write_weights(self.cfg.env.train.agent_weights_file, weights)
            mult = torch.tensor([float(m)], device=self.device)
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(mult, src=0)
        m = float(mult.item())
        if abs(m - 1.0) > 1e-9:
            self.optimizer.param_groups[0]["lr"] *= m
            if hasattr(self.lr_scheduler, "base_lrs"):
                self.lr_scheduler.base_lrs[0] *= m
        return {"am/lr_multiplier": m, "am/lr": float(self.optimizer.param_groups[0]["lr"])}

    # ---------------- accept_rollback (local addition 2026-11-05; RSI option 1) ----------------''', "actor methods")
    ACT.write_text(a)
    print("actor patched")
else:
    print("actor already patched")
