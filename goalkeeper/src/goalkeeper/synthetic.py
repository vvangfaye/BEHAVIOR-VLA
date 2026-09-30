"""Causal binary toy data for checking the software, never benchmark evidence."""
from pathlib import Path
import json
import numpy as np
from .goals import compile_bddl


TOY_BDDL = """; SYNTHETIC software fixture, not BEHAVIOR.
(define (problem binary_controls)
 (:objects first - switch second - switch)
 (:init (not (on first)) (not (on second)))
 (:goal (and (on first) (on second))))
"""


def make_dataset(output, groups=120, windows=3, seed=7):
    if groups < 20 or windows < 1: raise ValueError("Use >=20 groups and >=1 window")
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    (root / "task.bddl").write_text(TOY_BDDL)
    program = compile_bddl(TOY_BDDL)
    rng = np.random.default_rng(seed)
    rows = []
    for group in range(groups):
        for window in range(windows):
            truth = rng.integers(0, 2, (3, 2)).astype(np.float32)
            # Two visual patches identify two controllable switches.
            visual = np.zeros((3, 3, 2, 8), np.float32)
            visual[..., :2] = np.eye(2)[None, None]
            visual[..., 2] = truth[:, None]
            visual += rng.normal(0, .015, visual.shape).astype(np.float32)
            actions = np.zeros((4, 4, 2), np.float32)
            for candidate, target in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                actions[candidate] = np.array(target) * 2 - 1
            future = (actions[:, -1] > 0).astype(np.float32)
            query = np.zeros((2, 8), np.float32); query[:, :2] = np.eye(2)
            rid = f"g{group:04d}-w{window:03d}"
            np.savez_compressed(root / f"{rid}.npz", visual=visual, proprio=truth,
                                queries=query, actions=actions, truth=truth,
                                visibility=np.ones_like(truth), future_truth=future,
                                atom_keys=np.array([a.key for a in program.atoms]))
            rows.append({"id": rid, "group_id": f"synthetic-root-{group}", "source": "synthetic_test",
                         "record": f"{rid}.npz", "bddl": "task.bddl",
                         "feature_id": "synthetic-switches-v1", "action_spec": "synthetic-binary-v1"})
    manifest = root / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(manifest)
