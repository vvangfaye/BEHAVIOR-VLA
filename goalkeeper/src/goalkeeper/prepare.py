"""Export decision-spaced training features from replayed, labeled episodes.

This module is OFFLINE ONLY. It does not infer simulator labels from videos.
Input truth must be measured by a training-data replay/annotation process.
"""
from pathlib import Path
import json
import tempfile
import numpy as np
from .goals import compile_bddl
from .data import merge_manifests, write_record_archive


def export_episode(episode, bddl, output, *, group_id, features, action_spec,
                   source="training_demo", horizon=16, history=4):
    if source not in ("training_demo", "training_rollout"):
        raise ValueError("Only training episodes may be exported")
    if min(horizon, history) < 1 or not group_id or not action_spec:
        raise ValueError("Positive horizons and explicit provenance are required")
    program = compile_bddl(Path(bddl).read_text())
    with np.load(episode, allow_pickle=False) as data:
        state, actions, truth, visibility = [data[k] for k in
                                            ("proprio", "actions", "truth", "visibility")]
        if "source" in data and str(data["source"].item()) != source:
            raise ValueError("Episode source disagrees with requested training source")
        if "group_id" in data and str(data["group_id"].item()) != group_id:
            raise ValueError("Episode group provenance disagrees with --group-id")
        if "rgb" in data:
            rgb = data["rgb"]
            if rgb.ndim != 5 or rgb.shape[1] != 3: raise ValueError("rgb must have shape [T+1,3,H,W,3]")
            cameras = tuple(rgb[:, i] for i in range(3))
        else:
            cameras = tuple(data[k] for k in ("rgb_head", "rgb_left", "rgb_right"))
        if data["atom_keys"].tolist() != [a.key for a in program.atoms]:
            raise ValueError("Episode truth does not match public grounded goal")
    # T actions connect T+1 observations: action[t] maps obs[t] -> obs[t+1].
    steps, n = len(actions), len(program.atoms)
    if any(x.ndim != 4 or len(x) != steps + 1 or x.shape[-1] != 3 or x.dtype != np.uint8 for x in cameras):
        raise ValueError("Each RGB camera must be uint8 [T+1,height,width,3]")
    if state.ndim != 2 or len(state) != steps + 1 or actions.ndim != 2:
        raise ValueError("Invalid episode proprio/action dimensions")
    if truth.shape != (steps + 1, n) or visibility.shape != truth.shape:
        raise ValueError("Invalid training label shape")
    if not np.isin(truth, [0, 1]).all() or not np.isin(visibility, [-1, 0, 1]).all():
        raise ValueError("Invalid labels")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite episode")
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    task_path = root / f"{program.source_sha256}.bddl"
    task_path.write_text(Path(bddl).read_text())
    queries = features.encode_queries([program.query(a) for a in program.atoms])
    # Match inference: one recurrent update per H-action decision, not each frame.
    decision_times = list(range(0, steps - horizon + 1, horizon))
    cache = {t: features.encode_images(tuple(x[t] for x in cameras)) for t in decision_times}
    rows, records = [], []
    import hashlib
    episode_id = hashlib.sha256(f"{group_id}:{Path(episode).name}".encode()).hexdigest()[:20]
    archive_name = f"{episode_id}.npz"
    for j, t in enumerate(decision_times):
        if j < history - 1: continue  # No repeated padding pretending to be evidence.
        times = decision_times[j - history + 1:j + 1]
        rid = f"{episode_id}-{t:07d}"
        record_index = len(records)
        records.append({"visual": np.stack([cache[s] for s in times]),
                        "proprio": state[times], "actions": actions[None, t:t + horizon],
                        "truth": truth[times], "visibility": visibility[times],
                        "future_truth": truth[None, t + horizon]})
        rows.append({"id": rid, "group_id": group_id, "source": source,
                     "record": archive_name, "record_index": record_index, "bddl": task_path.name,
                     "feature_id": features.feature_id, "action_spec": action_spec})
    if not rows: raise ValueError("Episode is too short for this history and horizon")
    write_record_archive(root / archive_name, records, queries,
                         np.array([a.key for a in program.atoms]))
    manifest = root / f"{episode_id}.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(manifest)


def export_batch(spec, output, *, features, horizon=16, history=4):
    """Export an explicit JSON/JSONL list of approved episode NPZ files and seal it."""
    spec_path = Path(spec).resolve()
    text = spec_path.read_text()
    items = json.loads(text) if spec_path.suffix == ".json" else [json.loads(line) for line in text.splitlines() if line.strip()]
    if not isinstance(items, list) or not items:
        raise ValueError("Batch specification must contain at least one episode")
    required = {"episode", "bddl", "group_id", "action_spec", "source"}
    manifests = []
    with tempfile.TemporaryDirectory(prefix="goalkeeper-export-") as temporary:
        for index, item in enumerate(items):
            if not isinstance(item, dict) or not required <= item.keys():
                raise ValueError(f"Batch item {index} is missing: {required - set(item) if isinstance(item, dict) else required}")
            if item["source"] not in ("training_demo", "training_rollout"):
                raise ValueError("Batch export accepts approved training sources only")
            episode = (spec_path.parent / item["episode"]).resolve()
            bddl = (spec_path.parent / item["bddl"]).resolve()
            manifests.append(export_episode(
                episode, bddl, Path(temporary) / f"episode-{index:06d}",
                group_id=item["group_id"], features=features, action_spec=item["action_spec"],
                source=item["source"], horizon=horizon, history=history))
        return merge_manifests(manifests, output, seal=True)
