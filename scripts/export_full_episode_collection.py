#!/usr/bin/env python3
"""Export native whole-task successes, then verify and index their weak labels.

Every executed frame stays in the exported episode. Joint training samples only
nonempty, policy-reported subtask intervals containing a complete action chunk.
These labels are self-generated supervision, not human annotations or evidence
that an individual subtask succeeded. Failed/truncated attempts remain raw data.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from full_episode_trajectory import KIND, inspect_full_episode_trajectory
from subtask_trajectory import CAMERAS, array

EXPORT_KIND = "native_full_episode_export"
LABEL_SEMANTICS = "self_generated_weak_subtask"
VERIFICATION_KIND = "native_full_episode_export_readback"
VERIFIED_MARKER = "_VERIFIED_FULL_EPISODE_EXPORT"


def canonical_task_prompts():
    """Read the literal task registry without model or teacher dependencies."""
    path = Path(__file__).resolve().parents[1] / "openpi/src/openpi/configs/tasks/b1k.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "TASKS"
                                               for target in node.targets):
            prompts = ast.literal_eval(node.value)
            if isinstance(prompts, dict) and prompts:
                return prompts
    raise ValueError("Canonical BEHAVIOR whole-task prompt registry was not found")


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_scope(training_instance_min, held_out_instances):
    if type(training_instance_min) is not int or training_instance_min not in (301, 311):
        raise ValueError("Training instance minimum must be 301 (all public test) or 311")
    held = set()
    for row in held_out_instances:
        if (not isinstance(row, (list, tuple)) or len(row) != 2 or not isinstance(row[0], str)
                or not row[0].strip() or type(row[1]) is not int or row[1] < 0):
            raise ValueError("Invalid held-out task/instance pair")
        pair = tuple(row)
        if pair in held:
            raise ValueError("Duplicate held-out task/instance pair")
        held.add(pair)
    return held


def features_for(info):
    return {
        "action": {"dtype": "float32", "shape": (23,), "names": ["action"]},
        "observation.state": {"dtype": "float32", "shape": tuple(info["state_shape"]), "names": ["proprio"]},
        **{name: {"dtype": "video", "shape": tuple(shape), "names": ["height", "width", "channels"]}
           for name, shape in info["camera_shapes"].items()},
    }


def label_rows(info, episode):
    rows, cursor = [], 0
    for segment in info["segments"]:
        start, end = segment["start"], segment["end"]
        if type(start) is not int or type(end) is not int or start != cursor or end <= start:
            raise ValueError("Executed labels must tile the episode with [start, end) intervals")
        label, source = segment["subtask"], segment["source"]
        if label is not None and not isinstance(label, str):
            raise ValueError("Executed subtask must be text or null")
        if source not in (None, "generated", "cached"):
            raise ValueError("Native labels cannot be teacher/oracle annotations")
        rows.append({"episode_index": episode, "start": start, "end": end, "subtask": label,
                     "subtask_source": source, "label_semantics": LABEL_SEMANTICS,
                     "is_human_annotated": False, "subtask_success_verified": False})
        cursor = end
    if cursor != info["steps"]:
        raise ValueError("Executed labels do not cover the complete episode")
    return rows


def training_intervals(rows, action_horizon=32):
    """Merge unchanged labels across generated/cached source transitions only.

The full source trace keeps those transitions. They do not change the low-level
language target, so they are not action boundaries. Empty labels break a span.
"""
    merged = []
    for row in rows:
        label = row["subtask"]
        if not label or not label.strip():
            continue
        if label != label.strip():
            raise ValueError("Nonempty native labels must be canonical text; raw labels are never silently rewritten")
        if (merged and merged[-1]["episode_index"] == row["episode_index"]
                and merged[-1]["subtask"] == label and merged[-1]["end"] == row["start"]):
            merged[-1]["end"] = row["end"]
        else:
            merged.append({key: row[key] for key in ("episode_index", "start", "end", "subtask")})
    return [row for row in merged if row["end"] - row["start"] >= action_horizon]


def inspect_inputs(trajectory_paths, *, training_instance_min=301, held_out_instances=(), action_horizon=32):
    """Report raw attempts and BC selection without creating output or reading videos."""
    held = validate_scope(training_instance_min, held_out_instances)
    if type(action_horizon) is not int or action_horizon <= 0:
        raise ValueError("Action horizon must be a positive integer")
    paths = [Path(path).resolve() for path in trajectory_paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("Provide explicit unique raw trajectory paths")
    rows, identities, contents = [], set(), set()
    for path in paths:
        info = inspect_full_episode_trajectory(path, require_success=False)
        metadata, result = info["metadata"], info["result"]
        identity = tuple(metadata[key] for key in ("task", "instance_id", "rollout_id", "run_id", "attempt_id"))
        if identity in identities or info["content_sha256"] in contents:
            raise ValueError("Duplicate native attempt identity or trajectory content")
        identities.add(identity)
        contents.add(info["content_sha256"])
        segments = label_rows(info, 0)
        eligible = (result.get("success") is True and result.get("terminated") is True
                    and result.get("truncated") is False and metadata["instance_id"] >= training_instance_min
                    and (metadata["task"], metadata["instance_id"]) not in held)
        anchors = sum(row["end"] - row["start"] - action_horizon + 1
                      for row in training_intervals(segments, action_horizon))
        rows.append({"path": str(path), "task": metadata["task"], "instance_id": metadata["instance_id"],
                     "success": result["success"], "terminated": result["terminated"],
                     "truncated": result["truncated"], "frames": info["steps"],
                     "bc_episode_eligible": eligible, "valid_labelled_anchors": anchors if eligible else 0,
                     "unlabelled_frames": sum(row["end"] - row["start"] for row in segments
                                               if not row["subtask"] or not row["subtask"].strip()),
                     "content_sha256": info["content_sha256"]})
    return {"kind": "native_full_episode_preflight", "attempts": len(rows),
            "raw_frames": sum(row["frames"] for row in rows),
            "bc_eligible_episodes": sum(row["bc_episode_eligible"] for row in rows),
            "bc_eligible_frames": sum(row["frames"] for row in rows if row["bc_episode_eligible"]),
            "valid_labelled_anchors": sum(row["valid_labelled_anchors"] for row in rows),
            "training_instance_min": training_instance_min, "held_out_instances": sorted(map(list, held)),
            "label_semantics": LABEL_SEMANTICS, "official_benchmark": False, "attempt_rows": rows}


def export_full_episode_collection(trajectory_paths, output, repo_id, *, task_prompts=None,
                                   dataset_class=None, training_instance_min=301, held_out_instances=(),
                                   writer_options=None):
    output = Path(output)
    if output.exists():
        raise ValueError("Export requires a fresh output root")
    if not isinstance(repo_id, str) or not repo_id.strip():
        raise ValueError("repo_id is required")
    preflight = inspect_inputs(trajectory_paths, training_instance_min=training_instance_min,
                              held_out_instances=held_out_instances)
    if preflight["bc_eligible_episodes"] != preflight["attempts"]:
        raise ValueError("Export inputs must all be complete successes in the explicit training split")
    prompts = canonical_task_prompts() if task_prompts is None else dict(task_prompts)
    checked, task_ids, prompt_owners = [], {}, {}
    for row in preflight["attempt_rows"]:
        path = Path(row["path"])
        info = inspect_full_episode_trajectory(path, require_success=True)
        metadata = info["metadata"]
        task, prompt = metadata["task"], prompts.get(metadata["task"])
        if not isinstance(prompt, str) or not prompt.strip() or metadata.get("task_prompt") != prompt:
            raise ValueError("Whole-task prompt must exactly match the collected task_prompt")
        if prompt in prompt_owners and prompt_owners[prompt] != task:
            raise ValueError("Different canonical tasks must not share an identical prompt")
        prompt_owners[prompt] = task
        if metadata["fps"] != 30:
            raise ValueError("Three-source B1K training requires the unchanged 30 Hz control timebase")
        if checked and features_for(info) != features_for(checked[0][1]):
            raise ValueError("Collected feature shapes differ across native episodes")
        task_ids.setdefault(task, len(task_ids))
        checked.append((path, info))
    real_writer = dataset_class is None
    writer_options = {} if writer_options is None else dict(writer_options)
    if set(writer_options) - {'streaming_encoding', 'encoder_threads', 'camera_encoder'}:
        raise ValueError('Writer options may only tune CPU video encoding')
    if dataset_class is None:
        from lerobot.datasets import LeRobotDataset
        dataset_class = LeRobotDataset
    features = features_for(checked[0][1])
    writer = dataset_class.create(repo_id=repo_id, root=output, fps=30, features=features,
                                  robot_type="r1pro", use_videos=True, **writer_options)
    provenance, segments = [], []
    try:
        for episode, (path, info) in enumerate(checked):
            metadata = info["metadata"]
            with h5py.File(path, "r") as source:
                for step in range(info["steps"]):
                    writer.add_frame({**{key: source[key][step] for key in features},
                                      "task": prompts[metadata["task"]]})
            writer.save_episode()
            segments.extend(label_rows(info, episode))
            provenance.append({"episode_index": episode, "task_index": task_ids[metadata["task"]],
                               "task_name": metadata["task"], "task_instance_id": metadata["instance_id"],
                               "rollout_id": metadata["rollout_id"], "attempt_id": metadata["attempt_id"],
                               "length": info["steps"], "source_kind": KIND,
                               "source_trajectory": str(path), "content_sha256": info["content_sha256"],
                               "subtask_sha256": info["subtask_sha256"], "source_metadata": metadata,
                               "terminal_observation_sha256": info.get("terminal_observation_sha256"),
                               "terminal_observation_recorded": info.get("terminal_observation_recorded", False),
                               "environment_result": info["result"], "training_seen": True,
                               "independent_evaluation_eligible": False, "label_semantics": LABEL_SEMANTICS})
    finally:
        writer.finalize()
    by_episode = {row["episode_index"]: row for row in provenance}
    seen = set()
    episode_files = sorted((output / "meta/episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise ValueError("LeRobot did not finalize v3 episode metadata")
    for path in episode_files:
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            number = row["episode_index"]
            if number in seen or number not in by_episode or row.get("length") != by_episode[number]["length"]:
                raise ValueError("Export episode metadata differs from executed native trajectories")
            seen.add(number)
            source = by_episode[number]
            for key in ("task_index", "task_name", "task_instance_id", "rollout_id", "attempt_id", "source_kind",
                        "training_seen", "independent_evaluation_eligible", "label_semantics"):
                row[key] = source[key]
        pq.write_table(pa.Table.from_pylist(rows), path)
    if seen != set(by_episode):
        raise ValueError("Exported native episodes are incomplete")
    pq.write_table(pa.Table.from_pylist(segments), output / "meta/subtask_segments.parquet")
    write_json(output / "meta/collection_provenance.json", provenance)
    (output / "meta/tasks.jsonl").write_text("".join(
        json.dumps({"task_index": number, "task_name": task, "task": prompts[task]}) + "\n"
        for task, number in task_ids.items()))
    (output / "retained_episodes.txt").write_text("".join(f"{number}\n" for number in by_episode))
    report = {"schema_version": 1, "kind": EXPORT_KIND, "repo_id": repo_id, "episodes": len(provenance),
              "frames": sum(row["length"] for row in provenance), "segments": len(segments), "fps": 30,
              "source_sha256": digest(provenance), "label_sha256": digest(segments),
              "source_groups": len({(row["task_name"], row["task_instance_id"]) for row in provenance}),
              "training_instance_min": training_instance_min, "held_out_instances": preflight["held_out_instances"],
              "training_seen_instances": sorted(map(list, {(r["task_name"], r["task_instance_id"]) for r in provenance})),
              "label_semantics": LABEL_SEMANTICS, "subtask_success_verified": False,
              "action_boundary_policy": "full_horizon_within_nonempty_policy_label_interval",
              "unlabelled_frames": sum(row["unlabelled_frames"] for row in preflight["attempt_rows"]),
              "valid_labelled_anchors": preflight["valid_labelled_anchors"], "action_horizon": 32,
              "real_lerobot_writer": real_writer, "real_codec_reader_verified": False,
              "ready_for_training": False, "official_benchmark": False, "gpu_training_verified": False,
              "status": "exported_pending_real_reader_and_codec_readback"}
    write_json(output / "export_report.json", report)
    return report


def load_export(root):
    root = Path(root)
    report = json.loads((root / "export_report.json").read_text())
    provenance = json.loads((root / "meta/collection_provenance.json").read_text())
    segments = pq.read_table(root / "meta/subtask_segments.parquet").to_pylist()
    if (report.get("kind") != EXPORT_KIND or report.get("source_sha256") != digest(provenance)
            or report.get("label_sha256") != digest(segments)):
        raise ValueError("Native export provenance or executed labels changed")
    if (report.get("episodes") != len(provenance) or report.get("frames") != sum(row["length"] for row in provenance)
            or [row["episode_index"] for row in provenance] != list(range(len(provenance)))):
        raise ValueError("Native export episode counts or identity changed")
    return report, provenance, segments


def metadata_hashes(root):
    paths = [root / name for name in ("meta/info.json", "meta/tasks.jsonl", "meta/collection_provenance.json",
                                      "meta/subtask_segments.parquet", "export_report.json")]
    paths.extend(sorted((root / "meta/episodes").glob("chunk-*/*.parquet")))
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}


def verify_export(root, *, dataset_class=None, action_horizon=32):
    """Compare full state/action arrays and decode representative real RGB frames."""
    root = Path(root)
    report, provenance, segments = load_export(root)
    if action_horizon != 32:
        raise ValueError("The frozen joint training action horizon is 32")
    real_reader = dataset_class is None
    if dataset_class is None:
        from lerobot.datasets import LeRobotDataset
        dataset_class = LeRobotDataset
    reader = dataset_class(repo_id=report["repo_id"], root=root, download_videos=False, video_backend="pyav",
                           delta_timestamps={"action": [step / report["fps"] for step in range(action_horizon)]})
    if len(reader) != report["frames"]:
        raise ValueError("Reader frame count differs from native export")
    tasks = {row["task_index"]: row for row in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
    offset, decoded = 0, 0
    for source in provenance:
        episode, count = source["episode_index"], source["length"]
        inspected = inspect_full_episode_trajectory(source["source_trajectory"], require_success=True)
        if (inspected["content_sha256"] != source["content_sha256"] or inspected["metadata"] != source["source_metadata"]
                or inspected.get("terminal_observation_sha256") != source["terminal_observation_sha256"]
                or inspected["result"] != source["environment_result"] or label_rows(inspected, episode)
                != [row for row in segments if row["episode_index"] == episode]):
            raise ValueError("Native source identity or label trace differs from exported provenance")
        raw = reader.hf_dataset.select_columns(["action", "observation.state", "episode_index", "frame_index", "task_index"])[offset:offset + count]
        actions, states = array(raw["action"]), array(raw["observation.state"])
        with h5py.File(source["source_trajectory"], "r") as stream:
            if not np.array_equal(actions, stream["action"][:]) or not np.array_equal(states, stream["observation.state"][:]):
                raise ValueError("Export changed pre-step observation/executed-action alignment")
        if (array(raw["episode_index"]).reshape(-1).tolist() != [episode] * count
                or array(raw["frame_index"]).reshape(-1).tolist() != list(range(count))
                or array(raw["task_index"]).reshape(-1).tolist() != [source["task_index"]] * count):
            raise ValueError("LeRobot episode/frame/task identity changed")
        sample_frames = {0, count // 2, count - 1}
        for segment in segments:
            if segment["episode_index"] == episode and segment["subtask"] and segment["end"] - segment["start"] >= action_horizon:
                sample_frames.update((segment["start"], segment["end"] - action_horizon))
        for frame in sorted(sample_frames):
            row = reader[offset + frame]
            expected = actions[np.minimum(frame + np.arange(action_horizon), count - 1)]
            if not np.array_equal(array(row["action"]), expected) or not np.array_equal(array(row["observation.state"]), states[frame]):
                raise ValueError("Reader action horizon or paired state crosses an episode boundary")
            if row["task"] != tasks[source["task_index"]]["task"]:
                raise ValueError("Reader whole-task prompt changed")
            for key, shape in inspected["camera_shapes"].items():
                pixels = array(row[key])
                if (pixels.shape != (shape[2], shape[0], shape[1]) or not np.isfinite(pixels).all()
                        or pixels.min() < 0 or pixels.max() > 1):
                    raise ValueError("Onboard RGB cannot decode to finite CHW training observations")
            decoded += 1
        offset += count
    verified = {"kind": VERIFICATION_KIND, "complete": True, "episodes": len(provenance), "frames": offset,
                "source_sha256": report["source_sha256"], "label_sha256": report["label_sha256"],
                "action_horizon": action_horizon, "decoded_samples": decoded,
                "aligned_source_frames_verified": offset, "real_lerobot_reader": real_reader,
                "real_codec_reader_verified": real_reader, "ready_for_training": real_reader,
                "label_semantics": LABEL_SEMANTICS, "dataset_metadata_sha256": metadata_hashes(root),
                "gpu_training_verified": False, "official_benchmark": False}
    # Injected test readers prove contracts only, never codec/training readiness.
    if real_reader:
        write_json(root / "native_format_verification.json", verified)
        write_json(root / VERIFIED_MARKER, {"source_sha256": report["source_sha256"], "label_sha256": report["label_sha256"],
                                           "verification_sha256": file_sha256(root / "native_format_verification.json")})
    return verified


def prepare_training_index(root, *, root_id, held_out_instances, action_horizon=32):
    """Freeze a standard collected-source index after native real-codec readback.

This explicit native adapter keeps empty intervals in the source episode. The
older fixed-subtask ``prepare_subtask_recovery`` exporter contract deliberately
does not accept native weak labels or partially unlabelled collected episodes.
"""
    root = Path(root).resolve()
    report, provenance, segments = load_export(root)
    if not isinstance(root_id, str) or not root_id.strip() or action_horizon != 32:
        raise ValueError("A nonempty root_id and the unchanged 32-step horizon are required")
    marker = json.loads((root / VERIFIED_MARKER).read_text())
    verification = json.loads((root / "native_format_verification.json").read_text())
    if (verification.get("kind") != VERIFICATION_KIND or verification.get("complete") is not True
            or verification.get("real_lerobot_reader") is not True
            or verification.get("real_codec_reader_verified") is not True
            or verification.get("aligned_source_frames_verified") != report["frames"]
            or marker.get("verification_sha256") != file_sha256(root / "native_format_verification.json")
            or any(value.get(key) != report[key] for value in (marker, verification)
                   for key in ("source_sha256", "label_sha256"))):
        raise ValueError("Native data requires matching real-reader and codec readback before indexing")
    if verification.get("dataset_metadata_sha256") != metadata_hashes(root):
        raise ValueError("Native dataset metadata changed after real-reader readback")
    held = validate_scope(report["training_instance_min"], held_out_instances)
    if not set(map(tuple, report["held_out_instances"])) <= held:
        raise ValueError("Index must preserve every previously excluded held-out instance")
    intervals, records = [], []
    for source in provenance:
        if (source["task_name"], source["task_instance_id"]) in held:
            raise ValueError("Native training contains a held-out instance")
        inspected = inspect_full_episode_trajectory(source["source_trajectory"], require_success=True)
        if (inspected["content_sha256"] != source["content_sha256"]
                or inspected.get("terminal_observation_sha256") != source["terminal_observation_sha256"]
                or inspected["metadata"] != source["source_metadata"] or inspected["result"] != source["environment_result"]):
            raise ValueError("Native source changed after readback")
        eligible = training_intervals([row for row in segments if row["episode_index"] == source["episode_index"]], action_horizon)
        if not eligible:
            continue
        intervals.extend(eligible)
        records.append({key: source[key] for key in ("episode_index", "length", "task_name", "task_instance_id", "content_sha256")})
    if not intervals:
        raise ValueError("No nonempty weak-label interval has a complete 32-step action window")
    files = [(root / "meta/info.json", "metadata"), (root / "meta/tasks.jsonl", "metadata"),
             *[(p, "metadata") for p in sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))],
             (root / "meta/subtask_segments.parquet", "labels"),
             *[(root / name, "collection_evidence") for name in (
                 "meta/collection_provenance.json", "export_report.json", "native_format_verification.json", VERIFIED_MARKER)]]
    evidence = [{"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path), "role": role} for path, role in files]
    payload = {"schema_version": 1, "kind": "b1k_valid_intervals", "source_kind": "collected",
               "root_id": root_id, "repo_id": report["repo_id"], "dataset_root": str(root),
               "action_horizon": action_horizon, "held_out_instances": sorted(map(list, held)),
               "episodes": records, "intervals": intervals,
               "provenance": {"source_files": evidence, "native_collection_kind": KIND,
                              "label_semantics": LABEL_SEMANTICS, "is_human_annotated": False,
                              "subtask_success_verified": False, "training_seen": True,
                              "independent_evaluation_eligible": False, "source_sha256": report["source_sha256"],
                              "action_boundary_policy": "full_horizon_within_nonempty_policy_label_interval",
                              "unlabelled_frames": report["unlabelled_frames"], "excluded_no_legal_anchor_episodes": len(provenance) - len(records),
                              "valid_anchors": sum(row["end"] - row["start"] - action_horizon + 1 for row in intervals)}}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openpi/src"))
    from openpi.training.recovery_index import canonical_digest, verify_source_files
    index = {**payload, "fingerprint": canonical_digest(payload)}
    verify_source_files(index)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "export"):
        command = sub.add_parser(name)
        command.add_argument("--trajectories", type=Path, nargs="+", required=True)
        command.add_argument("--training-instance-min", type=int, choices=(301, 311), default=301,
                             help="Default 301 uses all public test instances as training_seen; 311 optionally preserves 301..310")
        command.add_argument("--held-out-instances", type=Path, help="Additional JSON list of task/instance pairs")
    sub.choices["export"].add_argument("--output", type=Path, required=True)
    sub.choices["export"].add_argument("--repo-id", required=True)
    for name in ("verify", "index"):
        sub.add_parser(name).add_argument("--dataset-root", type=Path, required=True)
    sub.choices["index"].add_argument("--root-id", required=True)
    sub.choices["index"].add_argument("--held-out-instances", type=Path, required=True,
                                     help="Explicit JSON task/instance validation split, shared with the other mixture sources")
    sub.choices["index"].add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command in ("inspect", "export"):
        held = json.loads(args.held_out_instances.read_text()) if args.held_out_instances else []
        if args.command == "inspect":
            result = inspect_inputs(args.trajectories, training_instance_min=args.training_instance_min, held_out_instances=held)
        else:
            result = export_full_episode_collection(args.trajectories, args.output, args.repo_id,
                training_instance_min=args.training_instance_min, held_out_instances=held)
    elif args.command == "verify":
        result = verify_export(args.dataset_root)
    else:
        if args.output.exists():
            parser.error("Frozen index output already exists")
        held = json.loads(args.held_out_instances.read_text())
        index = prepare_training_index(args.dataset_root, root_id=args.root_id, held_out_instances=held)
        index["provenance"]["held_out_input"] = {"path": str(args.held_out_instances.resolve()),
                                                   "sha256": file_sha256(args.held_out_instances)}
        from openpi.training.recovery_index import canonical_digest
        index["fingerprint"] = canonical_digest({key: value for key, value in index.items() if key != "fingerprint"})
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n")
        result = {"index": str(args.output), "fingerprint": index["fingerprint"], "episodes": len(index["episodes"]),
                  **{key: value for key, value in index["provenance"].items() if key != "source_files"}}
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
