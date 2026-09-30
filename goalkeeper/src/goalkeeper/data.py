"""Offline feature/label records with immutable group-level splits.

One NPZ record is a history window at a training initial state. Its K action
prefixes may contain matched branch rollouts. Future labels must describe the
actual end of each recorded H-step action prefix, not full-episode success.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import torch
from .goals import compile_bddl


def group_split(group_id: str, seed: str = "goalkeeper-v1") -> str:
    bucket = int(hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()[:8], 16) % 10
    return "calibration" if bucket == 0 else "dev" if bucket == 1 else "train"


def load_manifest(path: str | Path) -> list[dict]:
    path = Path(path)
    rows, ids = [], set()
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        required = {"id", "group_id", "source", "record", "bddl", "feature_id", "action_spec"}
        if not required <= row.keys():
            raise ValueError(f"Missing manifest fields: {required - row.keys()}")
        if row["id"] in ids:
            raise ValueError("Duplicate record ID")
        ids.add(row["id"])
        if row["source"] not in ("training_demo", "training_rollout", "synthetic_test"):
            raise ValueError("Training inputs must not contain public_test or hidden_test instances")
        for key in ("record", "bddl"):
            row[key] = str((path.parent / row[key]).resolve())
        row["split"] = group_split(row["group_id"])
        rows.append(row)
    if not rows:
        raise ValueError("Empty dataset")
    if len({r["feature_id"] for r in rows}) != 1 or len({r["action_spec"] for r in rows}) != 1:
        raise ValueError("Mixed feature encoders or robot action specifications")
    return rows


def load_record(row: dict) -> dict:
    goal = compile_bddl(Path(row["bddl"]).read_text())
    if not goal.atoms or not goal.clauses or any(not c for c in goal.clauses):
        raise ValueError("Training records require a nontrivial supported goal")
    with np.load(row["record"], allow_pickle=False) as f:
        required = {"visual", "proprio", "queries", "actions", "truth", "visibility", "future_truth", "atom_keys"}
        if not required <= set(f.files):
            raise ValueError(f"Missing NPZ fields: {required - set(f.files)}")
        arrays = {k: f[k].copy() for k in required}
    keys = arrays.pop("atom_keys").tolist()
    if keys != [a.key for a in goal.atoms]:
        raise ValueError("Labels/features must use exactly the compiler's sorted grounded atoms")
    v, s, q, a = (arrays[k] for k in ["visual", "proprio", "queries", "actions"])
    if v.ndim != 4 or s.ndim != 2 or q.ndim != 2 or a.ndim != 3:
        raise ValueError("Invalid record dimensions")
    t, n, k = len(v), len(goal.atoms), len(a)
    if t < 1 or k < 1 or s.shape[0] != t or q.shape[0] != n:
        raise ValueError("Empty or mismatched feature dimensions")
    for field, shape in [("truth", (t, n)), ("visibility", (t, n)), ("future_truth", (k, n))]:
        if arrays[field].shape != shape:
            raise ValueError(f"Incorrect shape: {field}")
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise ValueError("Nonfinite training record")
    for field in ["truth", "future_truth"]:
        if not np.isin(arrays[field], [0, 1]).all():
            raise ValueError("Goal truth must be binary")
    if not np.isin(arrays["visibility"], [-1, 0, 1]).all():
        raise ValueError("Visibility labels must be -1 (unknown), 0 or 1")
    atom_index = {a: i for i, a in enumerate(goal.atoms)}
    clauses = np.zeros((len(goal.clauses), n), np.float32)
    for j, clause in enumerate(goal.clauses):
        for literal in clause:
            clauses[j, atom_index[literal.atom]] = 1 if literal.positive else -1
    arrays["clauses"] = clauses
    return {k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in arrays.items()}


def collate(records: list[dict]) -> dict:
    """Pad only N, K and clause axes; history and action horizon must match."""
    n = max(r["queries"].shape[0] for r in records)
    k = max(r["actions"].shape[0] for r in records)
    c = max(r["clauses"].shape[0] for r in records)
    batched = {}
    for key in records[0]:
        values = []
        for r in records:
            x = r[key]
            shape = list(x.shape)
            if key == "queries": shape[0] = n
            elif key in ["truth", "visibility"]: shape[1] = n
            elif key == "actions": shape[0] = k
            elif key == "future_truth": shape = [k, n]
            elif key == "clauses": shape = [c, n]
            y = torch.full(shape, -1. if key == "visibility" else 0.)
            y[tuple(slice(0, s) for s in x.shape)] = x
            values.append(y)
        batched[key] = torch.stack(values)
    batched["atom_mask"] = torch.tensor([[i < r["queries"].shape[0] for i in range(n)] for r in records])
    batched["candidate_mask"] = torch.tensor([[i < r["actions"].shape[0] for i in range(k)] for r in records])
    return batched
