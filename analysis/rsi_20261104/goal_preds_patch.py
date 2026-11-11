#!/usr/bin/env python3
"""Ground truth for subgoal judgments (2026-11-05, capability tests 2/3): a subprocess command 'goal_preds' returns
every BDDL goal predicate of a LIBERO env with its truth value; with +env.eval.dump_goal_preds=True the eval frame dump
writes step<t>.json (one [predicate, bool] pair per goal conjunct) next to every step<t>.png. Default off.
Usage: goal_preds_patch.py <RLinf root>  (requires subgoal_patch.py; idempotent)."""
import sys
from pathlib import Path
root = Path(sys.argv[1])
VENV = root / "rlinf/envs/sim/libero/venv.py"
ENV = root / "rlinf/envs/sim/libero/libero_env.py"
MARK = "goal_preds (local addition 2026-11-05)"


def sub(text, old, new, label):
    if text.count(old) != 1:
        raise SystemExit(f"{label}: anchor found {text.count(old)} times")
    return text.replace(old, new)


v = VENV.read_text()
if MARK not in v:
    v = sub(v, """def _worker(
    parent: connection.Connection,""", '''def _goal_preds(env):
    """goal_preds (local addition 2026-11-05): [predicate text, truth] for every BDDL goal conjunct."""
    dom = getattr(env, "env", env)
    return [[" ".join(str(x) for x in g), bool(dom._eval_predicate(g))] for g in dom.parsed_problem["goal_state"]]


def _worker(
    parent: connection.Connection,''', "venv helper")
    v = sub(v, """            elif cmd == "goal_frac":
                p.send(_goal_frac(env))""", """            elif cmd == "goal_frac":
                p.send(_goal_frac(env))
            elif cmd == "goal_preds":
                p.send(_goal_preds(env))""", "venv cmd")
    v = sub(v, """    def reconfigure_env_fns(self, env_fns, id=None):""", '''    def goal_preds(self):
        """goal_preds (local addition 2026-11-05): per env, [predicate text, truth] of every goal conjunct."""
        for w in self.workers:
            w.parent_remote.send(["goal_preds", None])
        return [w.parent_remote.recv() for w in self.workers]

    def reconfigure_env_fns(self, env_fns, id=None):''', "venv method")
    VENV.write_text(v)
    print("venv patched")
else:
    print("venv already patched")
e = ENV.read_text()
if MARK not in e:
    e = sub(e, """        imgs = imgs.detach().cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs)
        for i in range(self.num_envs):
            t = int(self._elapsed_steps[i])
            if not (bool(dones[i]) or (t > 0 and t % every == 0)):
                continue""", """        imgs = imgs.detach().cpu().numpy() if torch.is_tensor(imgs) else np.asarray(imgs)
        preds = None
        if self.cfg.get("dump_goal_preds", False):  # goal_preds (local addition 2026-11-05): subgoal ground truth
            due = [bool(dones[i]) or (int(self._elapsed_steps[i]) > 0 and int(self._elapsed_steps[i]) % every == 0)
                   for i in range(self.num_envs)]
            if any(due):
                preds = self.env.goal_preds()
        for i in range(self.num_envs):
            t = int(self._elapsed_steps[i])
            if not (bool(dones[i]) or (t > 0 and t % every == 0)):
                continue""", "env dump head")
    e = sub(e, """            imageio.imwrite(os.path.join(d, f"step{t:04d}.png"), img)""", """            imageio.imwrite(os.path.join(d, f"step{t:04d}.png"), img)
            if preds is not None:
                import json as _json

                with open(os.path.join(d, f"step{t:04d}.json"), "w") as f:
                    _json.dump({"preds": preds[i], "done": bool(dones[i])}, f)""", "env dump json")
    ENV.write_text(e)
    print("env patched")
else:
    print("env already patched")
