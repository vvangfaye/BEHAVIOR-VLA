"""Fixed +80-step landmark on full updated-policy rollouts, with paired continuations.

The optional primary runner hook calls observe_boundary only for the new policy's
actual root rollout. It does not rerun root inference or change actions or rewards.
"""
from collections import deque


class StallProbe:
    def __init__(self, seeds, horizon):
        self.seeds = list(seeds)
        self.horizon = horizon
        self.histories = [deque(maxlen=4) for _ in seeds]
        self.snapshots = {}
        self.rows = [
            {"seed": seed, "split": "formal_holdout", "eligible": False, "flagged": None,
             "first_in_step": None, "landmark_step": None, "window": [],
             "net_closing_m_30": None, "ineligible_reason": None,
             "old_success": None, "new_success": None,
             "first_success_old": None, "first_success_new": None}
            for seed in seeds
        ]

    def observe_boundary(self, index, metric, env):
        """Observe a true ten-action boundary before success. No action modification."""
        t = metric["step"]
        if t % 10:
            raise ValueError(f"Expected chunk boundary, got simulator step {t}")
        row, history = self.rows[index], self.histories[index]
        history.append({k: metric[k] for k in ("step", "bowl_in", "drawer_closed", "drawer_open_m")})
        if row["first_in_step"] is None and metric["bowl_in"] and not metric["drawer_closed"]:
            row["first_in_step"] = t
        if row["first_in_step"] is None or t != row["first_in_step"] + 80:
            return
        row["landmark_step"] = t
        row["window"] = list(history)
        if [m["step"] for m in history] != [t - 30, t - 20, t - 10, t]:
            raise RuntimeError("Incomplete or nonconsecutive pre-landmark geometry")
        if t >= self.horizon:
            row["ineligible_reason"] = "no_remaining_horizon"
            return
        if not metric["bowl_in"] or metric["drawer_closed"]:
            row["ineligible_reason"] = "bowl_out_or_drawer_closed_at_landmark"
            return
        closing_m = history[0]["drawer_open_m"] - metric["drawer_open_m"]
        row.update(eligible=True, net_closing_m_30=closing_m,
                   flagged=all(m["bowl_in"] for m in history)
                   and metric["drawer_open_m"] > 0.04 and closing_m < 0.01)
        snapshot = env.snapshot()
        if snapshot["success_once"] or snapshot["step"] != t:
            raise RuntimeError("Stall landmark history or step does not match its snapshot")
        row["state_hash"] = snapshot["state_hash"]
        row["observation_hash"] = snapshot["observation_hash"]
        self.snapshots[index] = snapshot

    def finish(self, root_new, envs, old, new, out, roll, save):
        directory = out / "stall"
        directory.mkdir(exist_ok=True)
        for i, (row, root) in enumerate(zip(self.rows, root_new, strict=True)):
            if row["seed"] != root["seed"]:
                raise RuntimeError("Supplementary row does not reference the actual primary root")
            row.update(root_new_success=root["success_once"],
                       root_first_success_step=root["first_success_step"],
                       initial_hash=root["initial_hash"])
            row["handoff_step"] = row["first_in_step"]
            if row["eligible"]:
                snap = self.snapshots[i]
                if snap["initial_hash"] != root["initial_hash"]:
                    raise RuntimeError("Supplementary snapshot initial-state mismatch")
                save(directory / f'prefix_{row["seed"]}.json', snap)
            elif row["ineligible_reason"] is None:
                if row["first_in_step"] is None:
                    row["ineligible_reason"] = "no_open_inside_boundary_before_success_or_horizon"
                elif root["success_once"]:
                    row["ineligible_reason"] = "success_before_or_at_landmark"
                else:
                    row["ineligible_reason"] = "landmark_after_horizon"
        selected = list(self.snapshots)
        if selected:
            use_envs = [envs[i] for i in selected]
            ids = [self.seeds[i] for i in selected]
            starts = [self.rows[i]["landmark_step"] for i in selected]
            snaps = [self.snapshots[i] for i in selected]
            left, _ = roll(use_envs, ids, old, starts, snaps, directory, "old_stall")
            right, _ = roll(use_envs, ids, new, starts, snaps, directory, "new_stall")
            for i, old_row, new_row in zip(selected, left, right, strict=True):
                if old_row["start_metrics"] != new_row["start_metrics"]:
                    raise RuntimeError("Old/new supplemental branch state mismatch")
                self.rows[i].update(old_success=old_row["success_once"], new_success=new_row["success_once"],
                                    first_success_old=old_row["first_success_step"],
                                    first_success_new=new_row["first_success_step"])
        for row in self.rows:
            save(directory / f'paired_{row["seed"]}.json', row)
