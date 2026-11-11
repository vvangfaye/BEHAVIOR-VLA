#!/usr/bin/env python3
"""Patch an RLinf tree with the verified accept/rollback (AR) option for embodied PPO (RSI option 1, 2026-11-05).

Default off. Enabled by two overrides:
  +env.train.task_stats_dir=<dir>            LIBERO train env writes one (task, success) record per env per rollout
                                              epoch (first episode end: success termination = 1, 480-step truncation = 0)
  +algorithm.accept_rollback.enable=True     actor checks each new rollout against the last accepted one before updating
Knobs (algorithm.accept_rollback.*): delta_task (0.15), delta_mean (0.08), z (1.64), min_n (8), lr_decay (0.5),
wait_s (180), log_path (optional JSONL of decisions).
Rule: reject when some task with >= min_n episodes in both rollouts drops by more than delta_task with a one-sided
two-proportion z above z, or the overall success drops by more than delta_mean with z above z. On reject: restore the
trainable parameters and the optimizer state saved at the last acceptance, multiply the actor learning rate by lr_decay,
skip this update (the next rollout comes from the restored policy). On accept: save parameters + optimizer state (CPU),
make this rollout the reference, run the normal update.
Usage: ar_patch.py <RLinf root>   (idempotent; refuses if the anchors are not found exactly once)
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
ENV = root / "rlinf/envs/sim/libero/libero_env.py"
ACT = root / "rlinf/workers/actor/embodied_fsdp_actor_worker.py"
MARK = "accept_rollback (local addition 2026-11-05)"


def sub(text, old, new, label):
    n = text.count(old)
    if n != 1:
        raise SystemExit(f"{label}: anchor found {n} times")
    return text.replace(old, new)


# ---------------------------------------------------------------- env: per-epoch first-episode records
s = ENV.read_text()
if MARK not in s:
    s = sub(s, """        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:""", """        if env_idx is None:
            env_idx = np.arange(self.num_envs)
            # accept_rollback (local addition 2026-11-05): a full reset starts a rollout epoch
            if not self.is_eval and self.cfg.get("task_stats_dir", None):
                self._ts_recorded = np.zeros(self.num_envs, dtype=bool)

        if self.is_start:""", "env reset")
    s = sub(s, """        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None""", """        # accept_rollback (local addition 2026-11-05): one (task, success) record per env per rollout epoch
        if not self.is_eval and self.cfg.get("task_stats_dir", None):
            self._record_task_stats(past_terminations.cpu().numpy(), past_dones.cpu().numpy())

        # eval_count_mask: per-env bool, True if this completion counts toward eval metrics.
        eval_count_mask = None""", "env chunk_step")
    s = sub(s, """    def _dump_eval_frames(self, obs, dones):""", """    def _record_task_stats(self, terms, dones):
        \"\"\"accept_rollback (local addition 2026-11-05): append the first episode end of each env in this rollout
        epoch (success termination = 1, truncation = 0) to <task_stats_dir>/env_<pid>.jsonl, before any auto-reset.\"\"\"
        import json as _json
        import os as _os

        rec = getattr(self, "_ts_recorded", None)
        if rec is None or len(rec) != self.num_envs:
            rec = self._ts_recorded = np.zeros(self.num_envs, dtype=bool)
        new = np.asarray(dones, dtype=bool) & ~rec
        if not new.any():
            return
        d = self.cfg.get("task_stats_dir")
        _os.makedirs(d, exist_ok=True)
        with open(_os.path.join(d, f"env_{_os.getpid()}.jsonl"), "a") as f:
            for i in np.nonzero(new)[0]:
                f.write(_json.dumps({"task": int(self.task_ids[i]), "s": int(bool(terms[i]))}) + "\\n")
        rec |= new

    def _dump_eval_frames(self, obs, dones):""", "env helper")
    ENV.write_text(s)
    print("env patched")
else:
    print("env already patched")

# ---------------------------------------------------------------- actor: accept / rollback before the update
a = ACT.read_text()
if MARK not in a:
    a = sub(a, """        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        if self.cfg.algorithm.loss_type == "opd":""", """        if self.is_weight_offloaded:
            self.load_param_and_grad(self.device)
        if self.is_optimizer_offloaded:
            self.load_optimizer(self.device)

        # accept_rollback (local addition 2026-11-05): verified acceptance of the last update (see _ar_check)
        if self.cfg.algorithm.get("accept_rollback", {}).get("enable", False):
            ar_metrics = self._ar_check()
            if not ar_metrics["ar/accepted"]:
                self.optimizer.zero_grad()
                clear_memory()
                return ar_metrics
        else:
            ar_metrics = None

        if self.cfg.algorithm.loss_type == "opd":""", "actor run_training head")
    a = sub(a, """        if prefix_cache_stats is not None:
            mean_metric_dict.update(
                {f"prefix_cache/{k}": float(v) for k, v in prefix_cache_stats.items()}
            )

        return mean_metric_dict""", """        if prefix_cache_stats is not None:
            mean_metric_dict.update(
                {f"prefix_cache/{k}": float(v) for k, v in prefix_cache_stats.items()}
            )
        if ar_metrics is not None:
            mean_metric_dict.update(ar_metrics)

        return mean_metric_dict""", "actor run_training tail")
    a = sub(a, """    def _actor_prefix_store_for_training(self):""", '''    # ---------------- accept_rollback (local addition 2026-11-05; RSI option 1) ----------------
    def _ar_read_new(self, d, expected, wait_s):
        """New (task, success) records written by the train envs since the last call (all env_*.jsonl files)."""
        import glob
        import json as _json
        import time

        st = self._ar_state
        t0 = time.time()
        while True:
            recs = []
            offsets = dict(st["offsets"])
            for path in sorted(glob.glob(f"{d}/env_*.jsonl")):
                with open(path) as f:
                    f.seek(offsets.get(path, 0))
                    chunk = f.read()
                cut = chunk.rfind("\\n") + 1  # complete lines only
                for line in chunk[:cut].splitlines():
                    if line.strip():
                        recs.append(_json.loads(line))
                offsets[path] = offsets.get(path, 0) + len(chunk[:cut].encode())
            if len(recs) >= expected or time.time() - t0 > wait_s:
                st["offsets"] = offsets
                return recs
            time.sleep(2.0)

    @staticmethod
    def _ar_drop(ref, new, min_n):
        """Largest one-sided drop (new below ref) among tasks with >= min_n episodes in both, plus the overall drop."""
        import math

        def z(p1, n1, p2, n2):
            p = (p1 * n1 + p2 * n2) / max(n1 + n2, 1)
            se = math.sqrt(max(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)), 1e-12))
            return (p1 - p2) / se

        worst = (0.0, 0.0, None)  # (drop, z, task)
        for t, (n1, s1) in ref.items():
            n2, s2 = new.get(t, (0, 0))
            if n1 >= min_n and n2 >= min_n:
                d = s1 / n1 - s2 / n2
                if d > worst[0]:
                    worst = (d, z(s1 / n1, n1, s2 / n2, n2), t)
        N1, S1 = sum(v[0] for v in ref.values()), sum(v[1] for v in ref.values())
        N2, S2 = sum(v[0] for v in new.values()), sum(v[1] for v in new.values())
        mean = (S1 / max(N1, 1) - S2 / max(N2, 1), z(S1 / max(N1, 1), N1, S2 / max(N2, 1), N2) if N1 and N2 else 0.0)
        return worst, mean

    def _ar_check(self) -> dict:
        import copy
        import json as _json
        import time

        acfg = self.cfg.algorithm.accept_rollback
        d = self.cfg.env.train.get("task_stats_dir")
        expected = int(self.cfg.env.train.total_num_envs) * int(self.cfg.env.train.get("rollout_epoch", 1))
        if not hasattr(self, "_ar_state"):
            self._ar_state = {"offsets": {}, "ref": None, "snap": None, "accepts": 0, "rollbacks": 0}
        st = self._ar_state
        dec = torch.zeros(6, device=self.device)  # accepted, n, worst drop, worst z, mean drop, mean z
        info = {}
        if self._rank == 0:
            recs = self._ar_read_new(d, expected, float(acfg.get("wait_s", 180)))
            new = {}
            for r in recs:
                v = new.setdefault(int(r["task"]), [0, 0])
                v[0] += 1
                v[1] += int(r["s"])
            accepted, worst, mean = True, (0.0, 0.0, None), (0.0, 0.0)
            if st["ref"] is not None and recs:
                worst, mean = self._ar_drop(st["ref"], new, int(acfg.get("min_n", 8)))
                zc = float(acfg.get("z", 1.64))
                bad_task = worst[0] > float(acfg.get("delta_task", 0.15)) and worst[1] > zc
                bad_mean = mean[0] > float(acfg.get("delta_mean", 0.08)) and mean[1] > zc
                accepted = not (bad_task or bad_mean)
            if accepted and recs:
                st["ref"] = {t: tuple(v) for t, v in new.items()}
            dec = torch.tensor([float(accepted), len(recs), worst[0], worst[1], mean[0], mean[1]], device=self.device)
            info = {"tasks": {str(t): v for t, v in sorted(new.items())}, "worst_task": worst[2]}
        if torch.distributed.is_initialized():
            torch.distributed.broadcast(dec, src=0)
        accepted = bool(dec[0].item() > 0.5)
        params = [p for p in self.model.parameters() if p.requires_grad]
        if accepted or st["snap"] is None:
            # this rollout's policy is the new fallback point
            st["snap"] = ([p.detach().to("cpu", copy=True) for p in params],
                          self._ar_to_cpu(copy.deepcopy(self.optimizer.state_dict())))
            st["accepts"] += 1
            accepted = True
        else:
            with torch.no_grad():
                for p, s in zip(params, st["snap"][0]):
                    p.data.copy_(s.to(p.device, non_blocking=True))
            self.optimizer.load_state_dict(copy.deepcopy(st["snap"][1]))
            gamma = float(acfg.get("lr_decay", 0.5))
            self.optimizer.param_groups[0]["lr"] *= gamma
            if hasattr(self.lr_scheduler, "base_lrs"):
                self.lr_scheduler.base_lrs[0] *= gamma
            st["rollbacks"] += 1
        metrics = {"ar/accepted": float(accepted), "ar/n_records": float(dec[1].item()),
                   "ar/worst_task_drop": float(dec[2].item()), "ar/worst_task_z": float(dec[3].item()),
                   "ar/mean_drop": float(dec[4].item()), "ar/mean_z": float(dec[5].item()),
                   "ar/rollbacks": float(st["rollbacks"]), "ar/lr": float(self.optimizer.param_groups[0]["lr"])}
        if self._rank == 0 and acfg.get("log_path", None):
            with open(acfg.log_path, "a") as f:
                f.write(_json.dumps({"time": time.strftime("%H:%M:%S"), **metrics, **info}) + "\\n")
        return metrics

    @staticmethod
    def _ar_to_cpu(obj):
        if torch.is_tensor(obj):
            return obj.detach().to("cpu", copy=True)
        if isinstance(obj, dict):
            return {k: EmbodiedFSDPActor._ar_to_cpu(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(EmbodiedFSDPActor._ar_to_cpu(v) for v in obj)
        return obj

    def _actor_prefix_store_for_training(self):''', "actor methods")
    ACT.write_text(a)
    print("actor patched")
else:
    print("actor already patched")
