#!/usr/bin/env python3
"""Export verified Codex-assisted full-task HDF5 attempts to a fresh LeRobot root.

Export is a preparation step. This command does not mark data ready for training:
real LeRobot reader/video-codec and training-label readback remain required.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import h5py
import pyarrow as pa
import pyarrow.parquet as pq

from codex_collection_trajectory import CAMERAS, KIND, inspect_codex_collection_trajectory


def canonical_task_prompts():
    """Read the literal registry without importing model or GPU dependencies."""
    path = Path(__file__).resolve().parents[1] / "openpi/src/openpi/configs/tasks/b1k.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TASKS" for t in node.targets):
            prompts = ast.literal_eval(node.value)
            if isinstance(prompts, dict) and prompts:
                return prompts
    raise ValueError("Canonical BEHAVIOR task prompt registry was not found")


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _features(info):
    features = {
        "action": {"dtype": "float32", "shape": (23,), "names": ["action"]},
        "observation.state": {"dtype": "float32", "shape": tuple(info["state_shape"]), "names": ["proprio"]},
    }
    for name, shape in info["camera_shapes"].items():
        features[name] = {"dtype": "video", "shape": tuple(shape), "names": ["height", "width", "channels"]}
    return features


def export_codex_collection(trajectory_paths, output, repo_id, *, task_prompts=None, dataset_class=None):
    """Export only explicitly named, unique successes; never launch training.

    ``dataset_class`` is an injection point for CPU contract tests, not a route
    to claiming real codec/reader validation. The report remains pending readback.
    """
    output = Path(output)
    if output.exists():
        raise ValueError("Export requires a fresh output root")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id is required")
    paths = [Path(path).resolve() for path in trajectory_paths]
    if not paths:
        raise ValueError("Provide explicit successful trajectory paths")
    prompts = canonical_task_prompts() if task_prompts is None else dict(task_prompts)
    checked, identities, content_hashes, task_ids, used_prompts = [], set(), set(), {}, {}
    for path in paths:
        info = inspect_codex_collection_trajectory(path, require_success=True)
        identity = info["metadata"]
        task = identity["task"]
        key = (task, identity["instance_id"], identity["attempt_id"])
        if key in identities:
            raise ValueError(f"Duplicate task/instance/attempt identity: {key}")
        if info["content_sha256"] in content_hashes:
            raise ValueError("Duplicate executed trajectory content")
        identities.add(key)
        content_hashes.add(info["content_sha256"])
        prompt = prompts.get(task)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Missing whole-task prompt: {task}")
        source_prompt = identity.get("task_prompt")
        if not isinstance(source_prompt, str) or not source_prompt.strip():
            raise ValueError(f"Missing nonempty source task_prompt: {path}")
        if prompt != source_prompt:
            raise ValueError(f"Export whole-task prompt differs from collected task_prompt: {task}")
        if prompt in used_prompts and used_prompts[prompt] != task:
            raise ValueError("Different tasks must not share an identical dataset prompt")
        used_prompts[prompt] = task
        task_ids.setdefault(task, len(task_ids))
        if checked and (identity["fps"] != checked[0][1]["metadata"]["fps"] or _features(info) != _features(checked[0][1])):
            raise ValueError("Collection feature shapes and control frequency must match")
        checked.append((path, info))

    real_writer = dataset_class is None
    if dataset_class is None:
        from lerobot.datasets import LeRobotDataset
        dataset_class = LeRobotDataset
    features = _features(checked[0][1])
    fps = checked[0][1]["metadata"]["fps"]
    dataset = dataset_class.create(repo_id=repo_id, root=output, fps=fps, features=features,
                                   robot_type="r1pro", use_videos=True)
    provenance, segments = [], []
    try:
        for episode, (path, info) in enumerate(checked):
            identity = info["metadata"]
            task = identity["task"]
            with h5py.File(path, "r") as stream:
                for step in range(info["steps"]):
                    dataset.add_frame({**{name: stream[name][step] for name in features}, "task": prompts[task]})
            dataset.save_episode()
            segments.extend({"episode_index": episode, **segment} for segment in info["segments"])
            provenance.append({
                "episode_index": episode, "task_index": task_ids[task], "task_name": task,
                "task_instance_id": identity["instance_id"], "attempt_id": identity["attempt_id"],
                "length": info["steps"], "source_kind": KIND, "source_trajectory": str(path),
                "content_sha256": info["content_sha256"], "subtask_sha256": info["subtask_sha256"],
                "checkpoint": identity["checkpoint"], "conditioning": identity["conditioning"],
                "source_metadata": identity, "environment_result": info["result"],
            })
    finally:
        dataset.finalize()

    by_episode = {row["episode_index"]: row for row in provenance}
    seen = set()
    episode_files = sorted((output / "meta/episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise ValueError("LeRobot did not finalize v3 episode metadata")
    for path in episode_files:
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            episode = row["episode_index"]
            if episode in seen or episode not in by_episode:
                raise ValueError("Unexpected or repeated exported episode metadata")
            seen.add(episode)
            source = by_episode[episode]
            if row.get("length") != source["length"]:
                raise ValueError("Exported length differs from executed controls")
            for key in ("task_index", "task_name", "task_instance_id", "attempt_id", "source_kind"):
                row[key] = source[key]
        pq.write_table(pa.Table.from_pylist(rows), path)
    if seen != set(by_episode):
        raise ValueError("Exported episode metadata is incomplete")

    pq.write_table(pa.Table.from_pylist(segments), output / "meta/subtask_segments.parquet")
    _write_json(output / "meta/collection_provenance.json", provenance)
    (output / "meta/tasks.jsonl").write_text("".join(
        json.dumps({"task_index": index, "task_name": task, "task": prompts[task]}) + "\n"
        for task, index in task_ids.items()
    ))
    (output / "retained_episodes.txt").write_text("".join(f"{episode}\n" for episode in by_episode))
    report = {
        "schema_version": 1, "kind": "codex_teacher_full_task_export", "repo_id": repo_id,
        "episodes": len(provenance), "frames": sum(row["length"] for row in provenance),
        "segments": len(segments), "fps": fps, "conditioning": "task_and_subtask",
        "source_groups": len({(row["task_name"], row["task_instance_id"]) for row in provenance}),
        "source_sha256": _digest(provenance), "label_sha256": _digest(segments),
        "real_lerobot_writer": real_writer, "real_codec_reader_verified": False,
        "ready_for_training": False, "gpu_training_verified": False, "official_benchmark": False,
        "status": "exported_pending_real_reader_and_codec_readback",
        "verification_required": ["real LeRobot video decoding", "source observation/action alignment",
                                  "training reader action horizons and dynamic subtask label coverage"],
    }
    _write_json(output / "export_report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--task-prompts", type=Path, help="Optional JSON object mapping task names to full task prompts")
    args = parser.parse_args(argv)
    prompts = json.loads(args.task_prompts.read_text()) if args.task_prompts else None
    report = export_codex_collection(args.trajectories, args.output, args.repo_id, task_prompts=prompts)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
