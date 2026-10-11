#!/usr/bin/env python3
"""Mix explicit original demonstrations and successful Codex collection episodes.

This is separate from the older fixed-subtask collection experiment. A real
LeRobot decode, source alignment and 32-action training-window check is required
before its output may be used for training.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from codex_collection_trajectory import CAMERAS, KIND, inspect_codex_collection_trajectory
from export_codex_collection import canonical_task_prompts, _digest, _features, _write_json
from export_subtask_collection import image_hwc
from subtask_trajectory import array

EXPORT_KIND = "codex_teacher_mixed_training_export"
# Match pi05_b1k's training config and the original demonstration decoder.
READER_TOLERANCE_S = 5e-3


def read(path):
    return json.loads(Path(path).read_text())


def labels_for(rows, episode, length):
    selected = sorted((r for r in rows if r["episode_index"] == episode), key=lambda r: r["start"])
    end = 0
    for row in selected:
        if type(row["start"]) is not int or type(row["end"]) is not int or row["start"] != end or row["end"] <= end or not isinstance(row["subtask"], str) or not row["subtask"].strip():
            raise ValueError("Subtask labels must be nonempty and exactly tile each episode")
        end = row["end"]
    if not selected or end != length:
        raise ValueError("Subtask labels do not cover the complete episode")
    return [{k: r[k] for k in ("start", "end", "subtask")} for r in selected]


def original_sources(root, segments, episodes, tasks, prompts):
    root = Path(root).resolve()
    if not episodes or len(episodes) != len(set(episodes)) or any(type(e) is not int or e < 0 for e in episodes):
        raise ValueError("Original replay requires explicit unique episode IDs")
    all_rows = [r for p in sorted((root / "meta/episodes").glob("chunk-*/*.parquet")) for r in pq.read_table(p).to_pylist()]
    metadata = {r["episode_index"]: r for r in all_rows}
    if len(metadata) != len(all_rows):
        raise ValueError("Duplicate original episode metadata")
    label_rows = pq.read_table(segments).to_pylist()
    task_rows = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
    task_map = {row["task_index"]: row for row in task_rows}
    if len(task_map) != len(task_rows) or len({row["task_name"] for row in task_rows}) != len(task_rows):
        raise ValueError("Original task mapping has repeated IDs or task names")
    result = []
    for episode in episodes:
        row = metadata[episode]
        definition = task_map[row["task_index"]]
        task, prompt = definition["task_name"], definition["task"]
        # Released B1K stores names in tasks.parquet but full prompts in jsonl.
        # Accept either representation only after matching the same explicit ID
        # to the canonical registry; never infer identity from arbitrary text.
        if prompt != prompts.get(task) or row.get("tasks") not in ([task], [prompt]) or (row.get("task_name") is not None and row["task_name"] != task):
            raise ValueError("Original task ID, episode name and canonical prompt disagree")
        instance = row["task_instance_id"]
        if task not in tasks or type(instance) is not int or not 0 <= instance < 301:
            raise ValueError("Original replay must be selected-task training data and exclude public test instances")
        result.append(dict(task=task, task_index=row["task_index"], task_prompt=prompt,
                           instance=instance, episode=episode, length=row["length"],
                           labels=labels_for(label_rows, episode, row["length"])))
    if {r["task"] for r in result} != tasks:
        raise ValueError("Every collected task needs original replay demonstrations")
    return result


def replay_frame(row, features):
    frame = {}
    for key, feature in features.items():
        value = image_hwc(row[key]) if key in CAMERAS else array(row[key]).astype(np.float32)
        if key in CAMERAS and tuple(value.shape) != tuple(feature["shape"]):
            from PIL import Image
            h, w, _ = feature["shape"]
            value = np.asarray(Image.fromarray(value).resize((w, h)))
        if tuple(value.shape) != tuple(feature["shape"]) or not np.isfinite(value).all():
            raise ValueError("Original replay feature dimensions differ from collected data")
        frame[key] = value
    return frame


def validate_original_row(row, source, frame):
    if row["task"] not in (source["task"], source["task_prompt"]):
        raise ValueError("Original decoded task differs from its verified task mapping")
    for key, expected in (("task_index", source["task_index"]), ("episode_index", source["episode"]), ("frame_index", frame)):
        value = array(row[key]).reshape(-1)
        if value.shape != (1,) or value[0] != expected:
            raise ValueError("Original decoded row identity differs from selected source metadata")


def export_mixed(trajectories, output, repo_id, *, base_dataset, base_repo_id, base_segments,
                 base_episodes, dataset_class=None):
    output = Path(output).resolve()
    if output.exists() or not repo_id.strip() or not base_repo_id.strip():
        raise ValueError("Mixed export requires a fresh root and explicit repository IDs")
    prompts = canonical_task_prompts()
    checked, identities, hashes = [], set(), set()
    for source in trajectories:
        path = Path(source).resolve()
        info = inspect_codex_collection_trajectory(path, require_success=True)
        meta = info["metadata"]
        identity = (meta["task"], meta["instance_id"], meta["attempt_id"])
        if identity in identities or info["content_sha256"] in hashes:
            raise ValueError("Duplicate teacher attempt or trajectory")
        if meta.get("task_prompt") != prompts.get(meta["task"]):
            raise ValueError("Collected task prompt differs from the canonical training prompt")
        if checked and (_features(info) != _features(checked[0][1]) or meta["fps"] != checked[0][1]["metadata"]["fps"]):
            raise ValueError("Collected feature or frequency contracts differ")
        identities.add(identity); hashes.add(info["content_sha256"]); checked.append((path, info))
    if not checked:
        raise ValueError("At least one verified successful teacher trajectory is required")
    task_ids = {task: i for i, task in enumerate(dict.fromkeys(info["metadata"]["task"] for _, info in checked))}
    sources = original_sources(base_dataset, base_segments, list(base_episodes), set(task_ids), prompts)
    real_writer = dataset_class is None
    if dataset_class is None:
        from lerobot.datasets import LeRobotDataset
        dataset_class = LeRobotDataset
    features, fps = _features(checked[0][1]), checked[0][1]["metadata"]["fps"]
    ds = dataset_class.create(repo_id=repo_id, root=output, fps=fps, features=features, robot_type="r1pro", use_videos=True)
    provenance, segments = [], []

    def commit(task, instance, length, labels, **source):
        episode = len(provenance)
        ds.save_episode()
        segments.extend(dict(episode_index=episode, **r) for r in labels)
        provenance.append(dict(episode_index=episode, task_index=task_ids[task], task_name=task,
                               task_instance_id=instance, length=length, **source))

    try:
        for path, info in checked:
            meta = info["metadata"]
            with h5py.File(path) as stream:
                for step in range(info["steps"]):
                    ds.add_frame({**{key: stream[key][step] for key in features}, "task": prompts[meta["task"]]})
            commit(meta["task"], meta["instance_id"], info["steps"], info["segments"], source_kind=KIND,
                   source_trajectory=str(path), content_sha256=info["content_sha256"],
                   subtask_sha256=info["subtask_sha256"], source_metadata=meta,
                   environment_result=info["result"], attempt_id=meta["attempt_id"])
        for source in sources:
            base = dataset_class(repo_id=base_repo_id, root=Path(base_dataset), episodes=[source["episode"]],
                                 download_videos=False, video_backend="pyav", tolerance_s=READER_TOLERANCE_S)
            if base.meta.fps != fps or len(base) != source["length"]:
                raise ValueError("Original replay episode length or frequency differs")
            for step in range(len(base)):
                row = base[step]
                validate_original_row(row, source, step)
                ds.add_frame({**replay_frame(row, features), "task": source["task_prompt"]})
            commit(source["task"], source["instance"], len(base), source["labels"], source_kind="original",
                   source_dataset=str(Path(base_dataset).resolve()), source_repo_id=base_repo_id,
                   source_episode_index=source["episode"], source_task_index=source["task_index"],
                   source_task_prompt=source["task_prompt"], source_segments=str(Path(base_segments).resolve()))
    finally:
        ds.finalize()
    expected = {r["episode_index"]: r for r in provenance}
    seen = set()
    for path in sorted((output / "meta/episodes").glob("chunk-*/*.parquet")):
        rows = pq.read_table(path).to_pylist()
        for row in rows:
            ep = row["episode_index"]
            if ep in seen or ep not in expected or row["length"] != expected[ep]["length"]:
                raise ValueError("Finalized episode metadata differs from source")
            seen.add(ep)
            row.update({key: expected[ep][key] for key in ("task_index", "task_name", "task_instance_id", "source_kind")})
        pq.write_table(pa.Table.from_pylist(rows), path)
    if seen != set(expected):
        raise ValueError("Finalized metadata is missing episodes")
    pq.write_table(pa.Table.from_pylist(segments), output / "meta/subtask_segments.parquet")
    _write_json(output / "meta/collection_provenance.json", provenance)
    (output / "meta/tasks.jsonl").write_text("".join(json.dumps(dict(task_index=i, task_name=t, task=prompts[t])) + "\n" for t, i in task_ids.items()))
    (output / "retained_episodes.txt").write_text("".join(f"{i}\n" for i in expected))
    teacher_frames = sum(r["length"] for r in provenance if r["source_kind"] == KIND)
    total = sum(r["length"] for r in provenance)
    report = dict(schema_version=1, kind=EXPORT_KIND, repo_id=repo_id, episodes=len(provenance), frames=total,
                  new_frames=teacher_frames, original_frames=total - teacher_frames,
                  new_frame_fraction=teacher_frames / total, tasks=list(task_ids),
                  conditioning="task_and_subtask", fps=fps, reader_tolerance_s=READER_TOLERANCE_S,
                  source_sha256=_digest(provenance),
                  label_sha256=_digest(segments), real_lerobot_writer=real_writer,
                  ready_for_training=False, real_codec_reader_verified=False,
                  collection_instances=list(range(311, 321)), evaluation_instances=list(range(301, 311)))
    report["fingerprint"] = _digest(report)
    _write_json(output / "export_report.json", report)
    return report


def verify_mixed(root, *, dataset_class=None, horizon=32):
    root = Path(root)
    report, provenance = read(root / "export_report.json"), read(root / "meta/collection_provenance.json")
    segments = pq.read_table(root / "meta/subtask_segments.parquet").to_pylist()
    if horizon != 32 or report.get("kind") != EXPORT_KIND or report.get("fingerprint") != _digest({k: v for k, v in report.items() if k != "fingerprint"}):
        raise ValueError("Wrong or changed Codex mixed export contract")
    if report.get("reader_tolerance_s") != READER_TOLERANCE_S:
        raise ValueError("Mixed export reader tolerance differs from the training reader")
    if _digest(provenance) != report["source_sha256"] or _digest(segments) != report["label_sha256"]:
        raise ValueError("Source provenance or subtask labels changed")
    if [r["episode_index"] for r in provenance] != list(range(report["episodes"])) or (root / "retained_episodes.txt").read_text().splitlines() != [str(i) for i in range(report["episodes"])]:
        raise ValueError("Mixed export episode selection is not complete and dense")
    tasks = {r["task_index"]: r for r in (json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines())}
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "openpi/src"))
    from openpi.training.subtask_data import SubtaskFromSegments, episode_lengths_in_dataset_order
    lengths = episode_lengths_in_dataset_order(root)
    labels = SubtaskFromSegments.from_parquet(root / "meta/subtask_segments.parquet")
    labels.validate_coverage(lengths)
    if set(lengths) != set(range(report["episodes"])) or sum(lengths.values()) != report["frames"]:
        raise ValueError("Actual metadata counts differ from report")
    real_reader = dataset_class is None
    if dataset_class is None:
        from lerobot.datasets import LeRobotDataset
        dataset_class = LeRobotDataset
    reader = dataset_class(repo_id=report["repo_id"], root=root, download_videos=False, video_backend="pyav",
                           tolerance_s=READER_TOLERANCE_S,
                           delta_timestamps={"action": [i / report["fps"] for i in range(horizon)]})
    if len(reader) != report["frames"]:
        raise ValueError("Real dataset length differs from export")
    offset = decoded = aligned = original_frames = 0
    for source in provenance:
        ep, n = source["episode_index"], source["length"]
        if n != lengths[ep] or tasks[source["task_index"]]["task_name"] != source["task_name"]:
            raise ValueError("Source episode length or task mapping changed")
        ep_labels = labels_for(segments, ep, n)
        numeric = reader.hf_dataset.select_columns(["action", "observation.state", "episode_index", "frame_index", "task_index"])[offset:offset + n]
        actions, states = array(numeric["action"]).astype(np.float32), array(numeric["observation.state"]).astype(np.float32)
        if array(numeric["episode_index"]).reshape(-1).tolist() != [ep] * n or array(numeric["frame_index"]).reshape(-1).tolist() != list(range(n)) or array(numeric["task_index"]).reshape(-1).tolist() != [source["task_index"]] * n:
            raise ValueError("Exported row identities do not align")
        if actions.shape != (n, 23) or states.ndim != 2 or not 57 <= states.shape[1] <= 512 or not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError("Nonfinite or malformed exported state/action arrays")
        stream = None
        try:
            if source["source_kind"] == KIND:
                info = inspect_codex_collection_trajectory(source["source_trajectory"], require_success=True)
                if info["content_sha256"] != source["content_sha256"] or info["subtask_sha256"] != source["subtask_sha256"] or info["metadata"] != source["source_metadata"] or info["segments"] != ep_labels or not 311 <= source["task_instance_id"] <= 320 or info["metadata"]["fps"] != report["fps"]:
                    raise ValueError("Teacher source content or labels changed")
                stream = h5py.File(source["source_trajectory"])
                source_actions, source_states = stream["action"][:], stream["observation.state"][:]
                image_source = lambda frame: {key: stream[key][frame] for key in CAMERAS}
                aligned += n
            elif source["source_kind"] == "original":
                if not 0 <= source["task_instance_id"] < 301:
                    raise ValueError("Original replay contains public test instances")
                canonical = canonical_task_prompts()
                checked = original_sources(source["source_dataset"], source["source_segments"], [source["source_episode_index"]], {source["task_name"]}, canonical)[0]
                if checked["instance"] != source["task_instance_id"] or checked["length"] != n or checked["labels"] != ep_labels or checked["task_index"] != source["source_task_index"] or checked["task_prompt"] != source["source_task_prompt"]:
                    raise ValueError("Original replay metadata or labels changed")
                base = dataset_class(repo_id=source["source_repo_id"], root=Path(source["source_dataset"]),
                                     episodes=[source["source_episode_index"]], download_videos=False,
                                     video_backend="pyav", tolerance_s=READER_TOLERANCE_S)
                if base.meta.fps != report["fps"] or len(base) != n:
                    raise ValueError("Original replay frequency or length changed")
                values = base.hf_dataset.select_columns(["action", "observation.state"] )[:]
                source_actions, source_states = array(values["action"]).astype(np.float32), array(values["observation.state"]).astype(np.float32)
                source_features = {key: reader.meta.features[key] for key in (*CAMERAS, "action", "observation.state")}
                def image_source(frame):
                    row = base[frame]
                    validate_original_row(row, checked, frame)
                    return replay_frame(row, source_features)
                original_frames += n
            else:
                raise ValueError("Unknown source kind")
            if not np.array_equal(actions, source_actions) or not np.array_equal(states, source_states):
                raise ValueError("Export changed source observation/action alignment")
            samples = {0, n // 2, n - 1} | {r["start"] for r in ep_labels} | {r["end"] - 1 for r in ep_labels}
            for frame in sorted(samples):
                row = reader[offset + frame]
                if not np.array_equal(array(row["action"]), actions[np.minimum(frame + np.arange(horizon), n - 1)]) or not np.array_equal(array(row["observation.state"]), states[frame]):
                    raise ValueError("Training action window crosses episode bounds or shifts state")
                if labels(row)["subtask"] != labels.lookup(ep, frame)[0] or row["task"] != tasks[source["task_index"]]["task"]:
                    raise ValueError("Training task/subtask labels differ")
                expected_images = image_source(frame)
                for key in CAMERAS:
                    pixels = image_hwc(row[key])
                    expected = expected_images[key]
                    if pixels.shape != expected.shape or np.abs(pixels.astype(float) - expected.astype(float)).mean() > 12:
                        raise ValueError("Decoded RGB differs from source beyond codec tolerance")
                decoded += 1
        finally:
            if stream is not None:
                stream.close()
        offset += n
    if not aligned or not original_frames or aligned != report["new_frames"] or original_frames != report["original_frames"]:
        raise ValueError("Mixed export needs both verified teacher and original controls")
    return dict(complete=real_reader and report["real_lerobot_writer"], real_lerobot_reader=real_reader,
                real_lerobot_writer=report["real_lerobot_writer"], export_fingerprint=report["fingerprint"],
                action_horizon=horizon, reader_tolerance_s=READER_TOLERANCE_S, episodes=len(provenance), frames=offset,
                aligned_source_frames_verified=aligned, original_source_frames_verified=original_frames,
                decoded_frame_samples=decoded, label_coverage=True, gpu_training_verified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export")
    export.add_argument("--trajectories", type=Path, nargs="+", required=True)
    for name in ("output", "base-dataset", "base-segments", "base-episodes"):
        export.add_argument("--" + name, type=Path, required=True)
    export.add_argument("--repo-id", required=True); export.add_argument("--base-repo-id", required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--dataset-root", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "export":
        result = export_mixed(args.trajectories, args.output, args.repo_id, base_dataset=args.base_dataset,
                              base_repo_id=args.base_repo_id, base_segments=args.base_segments,
                              base_episodes=[int(line) for line in args.base_episodes.read_text().splitlines() if line.strip()])
    else:
        result = verify_mixed(args.dataset_root)
        with args.report.open("x") as stream:
            json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
