#!/usr/bin/env python3
"""Small visual demonstrations for a collection-only teacher.

Original examples require matching released LeRobot metadata, retained training
episode membership, and a task-success raw recording. New examples require a
verified 311--320 whole-task trajectory. Evaluation instances 301--310 are never
admitted. A montage is an observation artifact, not a synthesized robot image.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image, ImageDraw

from codex_collection_trajectory import inspect_codex_collection_trajectory
from subtask_trajectory import CAMERAS, array

KIND = "codex_collection_demonstrations_v1"


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _published_raw_identity(raw, row, expected_sha256, actual_sha256):
    if expected_sha256 is None:
        return None
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)
            or expected_sha256 != actual_sha256):
        raise ValueError("Original raw SHA256 differs from the frozen source selection")
    if any(type(row.get(key)) is not int or row[key] < 0 for key in ("task_index", "raw_episode_id")):
        raise ValueError("Published metadata needs explicit task_index and raw_episode_id")
    # Official replay_obs.py infers the task ID from the raw demo ID this way.
    if row["raw_episode_id"] // 10000 != row["task_index"]:
        raise ValueError("Published raw episode ID encodes a different task index")
    parent = f"task-{row['task_index']:04d}"
    filename = f"episode_{row['raw_episode_id']:08d}.hdf5"
    if raw.parent.name != parent or raw.name != filename:
        raise ValueError("Original raw path differs from the published metadata identity")
    if ("annotation_path" in row
            and row["annotation_path"] != f"annotations/{parent}/{Path(filename).stem}.json"):
        raise ValueError("Published raw and annotation identities disagree")
    return {"identity_source": "published_metadata_raw_id_and_sha", "task_index": row["task_index"],
            "raw_episode_id": row["raw_episode_id"], "published_raw_relative_path": f"{parent}/{filename}",
            "expected_raw_sha256": expected_sha256}


def _raw_task_config(data, task, instance, *, published_identity):
    """Read recording identity, never simulator state, from the standard config."""
    if "config" not in data.attrs:
        return None
    try:
        config = json.loads(data.attrs["config"])
        recorded = config["task"]
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Original raw task config is invalid") from error
    if (not isinstance(recorded, dict) or recorded.get("type") != "BehaviorTask"
            or recorded.get("activity_name") != task
            or type(recorded.get("activity_instance_id")) is not int
            or (recorded["activity_instance_id"] != instance
                and not (published_identity is not None and recorded["activity_instance_id"] == 0))):
        raise ValueError("Original raw config task or instance differs from metadata")
    for key in ("instance_id", "task_instance_id"):
        if key in recorded and (type(recorded[key]) is not int or recorded[key] != instance):
            raise ValueError("Original raw config has conflicting instance identity")
    return recorded


def _raw_success_proof(data, task, instance, length, *, published_identity):
    recorded = _raw_task_config(data, task, instance, published_identity=published_identity)
    matches = []
    for name, group in data.items():
        if not isinstance(group, h5py.Group) or not name.startswith("demo_"):
            continue
        if "task_name" in group.attrs and group.attrs["task_name"] != task:
            continue
        identities = [group.attrs[key] for key in ("instance_id", "task_instance_id") if key in group.attrs]
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
               or int(value) != instance for value in identities):
            continue
        if "action" not in group or not isinstance(group["action"], h5py.Dataset) or group["action"].shape != (length, 23):
            continue
        if "completion_reason" in group.attrs:
            # An explicit failure cannot fall through to the standard-schema path.
            if (group.attrs["completion_reason"] != "task_success"
                    or group.attrs.get("task_name") != task or not identities):
                continue
            proof = {"success_source": "original_raw_completion_reason_task_success"}
        else:
            if recorded is None or published_identity is None:
                raise ValueError("Standard original schema requires frozen raw SHA256 and published metadata identity")
            termination = recorded.get("termination_config", {})
            if (not isinstance(termination, dict) or set(termination) - {"max_steps"}
                    or ("max_steps" in termination and
                        (type(termination["max_steps"]) is not int or termination["max_steps"] <= 0))):
                raise ValueError("Unsupported original BehaviorTask termination config")
            signals = []
            for key in ("terminated", "truncated"):
                if (key not in group or not isinstance(group[key], h5py.Dataset)
                        or group[key].shape != (length,) or group[key].dtype.kind != "b"):
                    raise ValueError("Original termination signals must be action-aligned boolean sequences")
                signals.append(group[key][:])
            terminated, truncated = signals
            if (length <= 0 or not np.any(terminated & ~truncated)
                    or not bool(terminated[-1]) or bool(truncated[-1])):
                continue
            proof = {**published_identity,
                     "recorded_config_instance": recorded["activity_instance_id"],
                     "recorded_config_instance_is_default": recorded["activity_instance_id"] == 0,
                     "success_source": "original_behavior_task_terminated_without_truncation",
                     "termination_evidence": {"samples": length,
                         "terminated_count": int(terminated.sum()), "truncated_count": int(truncated.sum()),
                         "final_terminated": True, "final_truncated": False,
                         "termination_config": termination},
                     "success_semantics_audit": {
                         "reference": "BehaviorTask predicate/timeout and Joylo TaskSuccessMetric",
                         "reference_vendor_revision": "640e2418f9ffd148f94c04623ae965ce0e0b1767",
                         "exact_recording_source_semantics_verified": False,
                         "limitation": "Reference semantics were audited; the exact recording source revision was not verified."}}
        matches.append((group.name, proof))
    if len(matches) != 1:
        raise ValueError("No unique matching task-success original raw demonstration")
    group_name, proof = matches[0]
    version = git_hash = None
    if "scene_file" in data.attrs:
        try:
            # Only the version fields leave this helper. Scene state is never
            # returned to the teacher or used to infer success.
            scene = json.loads(data.attrs["scene_file"])
            recording = scene.get("versions", {}).get("omnigibson", {})
            version, git_hash = recording.get("version"), recording.get("git_hash")
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("Original recording version metadata is invalid") from error
        if any(value is not None and (not isinstance(value, str) or not value.strip() or len(value) > 128)
               for value in (version, git_hash)):
            raise ValueError("Original recording version metadata is invalid")
    proof.update(recording_omnigibson_version=version, recording_omnigibson_git_hash=git_hash)
    return group_name, proof


def _split(source_kind, instance):
    if type(instance) is not int:
        raise ValueError("Demonstration instance ID must be an integer")
    allowed = ((source_kind == "original_train" and 0 <= instance <= 300)
               or (source_kind == "verified_collection" and 311 <= instance <= 320))
    if not allowed:
        raise ValueError("Demonstrations permit original train 0..300 or verified collection 311..320 only")


def _segments(rows, length):
    rows = sorted(rows, key=lambda row: row["start"])
    end = 0
    clean = []
    for row in rows:
        if (type(row.get("start")) is not int or type(row.get("end")) is not int
                or row["start"] != end or not end < row["end"] <= length
                or not isinstance(row.get("subtask"), str) or not row["subtask"].strip()):
            raise ValueError("Demonstration labels must exactly tile the source episode")
        clean.append({key: row[key] for key in ("start", "end", "subtask")})
        end = row["end"]
    if not clean or end != length:
        raise ValueError("Demonstration labels must exactly tile the source episode")
    return clean


def _image(value):
    value = array(value)
    if value.ndim == 3 and value.shape[0] == 3 and value.shape[-1] != 3:
        value = value.transpose(1, 2, 0)
    if value.ndim != 3 or value.shape[-1] != 3 or not np.isfinite(value).all():
        raise ValueError("Invalid demonstration RGB")
    if np.issubdtype(value.dtype, np.floating):
        if value.min() < 0 or value.max() > 1:
            raise ValueError("Float demonstration RGB must be in [0,1]")
        value = np.rint(value * 255).astype(np.uint8)
    if value.dtype != np.uint8:
        raise ValueError("Demonstration RGB must decode to uint8")
    return Image.fromarray(value).resize((224, 224))


def _numbers(value, *, action=False):
    value = array(value).reshape(-1)
    if (not np.isfinite(value).all() or (action and value.shape != (23,))
            or (not action and not 23 <= len(value) <= 512)):
        raise ValueError("Invalid demonstration state or executed action")
    return np.round(value.astype(np.float64), 5).tolist()


def _example_segment_indices(labels, max_examples, required_subtask=None):
    selected = sorted(set(np.linspace(0, len(labels) - 1,
                                     min(max_examples, len(labels)), dtype=int).tolist()))
    if required_subtask is None:
        return selected
    if not isinstance(required_subtask, str) or not required_subtask.strip():
        raise ValueError("Required demonstration subtask must be nonempty text")
    required = [i for i, row in enumerate(labels) if row["subtask"] == required_subtask]
    if not required or len(required) > max_examples:
        raise ValueError("Required demonstration subtask is absent or exceeds the example budget")
    return sorted(required + [i for i in selected if i not in required][:max_examples - len(required)])


def _gripper_phase(actions, start, end):
    """Select observations around the first recorded full-close command.

    This selects a control event, not a successful grasp or simulator predicate.
    All observations are still captured before the action at their frame.
    """
    for frame in range(start, end):
        arms = [arm for arm, index in (("left", 14), ("right", 22))
                if actions[frame, index] <= -.95
                and (frame == 0 or actions[frame - 1, index] > -.95)]
        if arms:
            frames = sorted({max(start, frame - 15), frame, min(end - 1, frame + 15)})
            return frames, frame, dict(kind="recorded_full_close_command_v1", frame=frame,
                                      arms=arms, window_steps=15, grasp_success_inferred=False)
    return [start, end - 1], start, dict(kind="segment_endpoints_no_close_onset",
                                        grasp_success_inferred=False)


def _prepare(output_dir, metadata, labels, length, reader, *, max_examples=6, action_steps=4,
             required_subtask=None, gripper_phase_actions=None):
    _split(metadata["source_kind"], metadata["instance_id"])
    if type(max_examples) is not int or not 1 <= max_examples <= 6:
        raise ValueError("max_examples must be 1..6")
    if type(action_steps) is not int or not 1 <= action_steps <= 8:
        raise ValueError("action_steps must be 1..8")
    labels = _segments(labels, length)
    if gripper_phase_actions is not None:
        gripper_phase_actions = np.asarray(gripper_phase_actions)
        if gripper_phase_actions.shape != (length, 23) or not np.isfinite(gripper_phase_actions).all():
            raise ValueError("Gripper phase references require finite source-aligned 23D actions")
    # Cover the beginning, middle and end instead of choosing favorable outcomes.
    # A predeclared local target may reserve examples before that uniform coverage.
    selected = _example_segment_indices(labels, max_examples, required_subtask)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    examples = []
    for number, index in enumerate(selected):
        segment = labels[index]
        start, end = segment["start"], segment["end"]
        # Both images precede actual actions. Never describe the final image as
        # post-action success: success is verified separately for the full task.
        frames = [start, end - 1]
        action_start, selection = start, None
        if gripper_phase_actions is not None:
            frames, action_start, selection = _gripper_phase(gripper_phase_actions, start, end)
        montage = Image.new("RGB", (3 * 224, len(frames) * 244), "black")
        draw = ImageDraw.Draw(montage)
        observations = []
        for row, frame in enumerate(frames):
            observed = reader(frame)
            if gripper_phase_actions is not None:
                if not np.allclose(array(observed['action']).reshape(-1), gripper_phase_actions[frame],
                                   rtol=0, atol=1e-5):
                    raise ValueError("Released reference actions differ from the raw control event")
            observations.append({"frame": frame, "state": _numbers(observed["observation.state"])})
            for col, (key, camera) in enumerate(zip(CAMERAS, ("head", "left wrist", "right wrist"))):
                montage.paste(_image(observed[key]), (col * 224, row * 244 + 20))
                draw.text((col * 224 + 4, row * 244 + 3), f"frame {frame} / {camera}", fill="white")
        name = f"example-{number:02d}.png"
        montage.save(output / name)
        actions = [_numbers(reader(frame)["action"], action=True)
                   for frame in range(action_start, min(end, action_start + action_steps))]
        examples.append({"segment_index": index, **segment, "observations": observations,
                         "executed_actions_start": action_start, "executed_actions": actions,
                         "image": name, "image_sha256": _sha(output / name),
                         "example_kind": "successful_episode_segment",
                         "recovery_verified": False})
        if selection is not None:
            examples[-1]['reference_selection'] = selection
    result = {"kind": KIND, "metadata": metadata, "steps": length,
              "subtask_sequence": [row["subtask"] for row in labels], "examples": examples,
              "image_layout": "Rows are earlier/later pre-action observations; columns head/left wrist/right wrist.",
              "action_semantics": "Recorded 23D R1Pro controls from this source; examples are not commands to replay verbatim."}
    if required_subtask is not None:
        result["required_subtask"] = required_subtask
    if gripper_phase_actions is not None:
        result['reference_frame_policy'] = 'recorded_full_close_command_v1'
        result['image_layout'] = ("Rows are chronological pre-action observations around a recorded "
            "full-close command, or segment endpoints if no close onset exists; columns head/left wrist/right wrist. "
            "A close command does not prove a successful grasp.")
    result["fingerprint"] = _digest(result)
    (output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def prepare_original_demonstrations(dataset_root, segments_path, raw_path, episode_index,
                                    output_dir, *, retained_episodes_path, dataset_class=None,
                                    max_examples=6, expected_raw_sha256=None, required_subtask=None,
                                    gripper_phase_reference=False):
    """Read one explicitly retained original LeRobot episode plus raw success proof.

    ``dataset_class`` is a LeRobotDataset-compatible test seam. Production uses
    the real decoder and never substitutes raw simulator state for proprioception.
    """
    import pyarrow.parquet as pq
    root, raw = Path(dataset_root), Path(raw_path)
    if type(gripper_phase_reference) is not bool:
        raise ValueError("Gripper phase reference must be an explicit boolean")
    if type(episode_index) is not int or episode_index < 0:
        raise ValueError("Explicit original episode_index is required")
    retained = {int(line) for line in Path(retained_episodes_path).read_text().splitlines() if line.strip()}
    if episode_index not in retained:
        raise ValueError("Example is absent from the original retained training episode list")
    metadata_paths = sorted((root / "meta/episodes").glob("chunk-*/*.parquet"))
    matches = [(path, row) for path in metadata_paths for row in pq.read_table(path).to_pylist()
               if row["episode_index"] == episode_index]
    if len(matches) != 1:
        raise ValueError("Original episode metadata is missing or duplicated")
    metadata_path, row = matches[0]
    tasks = [json.loads(line) for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
    task_rows = [task for task in tasks if task["task_index"] == row["task_index"]]
    if len(task_rows) != 1:
        raise ValueError("Original task mapping is missing or duplicated")
    task = task_rows[0]["task_name"]
    prompt = task_rows[0].get("task")
    if (not isinstance(task, str) or not task.strip() or not isinstance(prompt, str) or not prompt.strip()
            or row.get("tasks") not in ([task], [prompt])
            or row.get("task_name", task) != task):
        raise ValueError("Original episode task text differs from its verified task index mapping")
    instance = row["task_instance_id"]
    _split("original_train", instance)
    raw_sha256 = _sha(raw)
    published_identity = _published_raw_identity(raw, row, expected_raw_sha256, raw_sha256)
    # Original raw files are action/state recordings. Their serialized state is
    # deliberately never exposed as an observation to the teacher.
    with h5py.File(raw, "r") as stream:
        raw_group, success_proof = _raw_success_proof(stream["data"], task, instance, row["length"],
                                                    published_identity=published_identity)
        phase_actions = stream[raw_group]['action'][:] if gripper_phase_reference else None
    if dataset_class is None:
        from lerobot.datasets import LeRobotDataset
        dataset_class = LeRobotDataset
    # Match the existing pi05_b1k training reader's 5 ms timestamp tolerance.
    reader_tolerance_s = 5e-3
    dataset = dataset_class(repo_id="original/teacher_examples", root=root,
                            episodes=[episode_index], download_videos=False, video_backend="pyav",
                            tolerance_s=reader_tolerance_s)
    if len(dataset) != row["length"]:
        raise ValueError("Decoded original episode length differs from metadata")
    labels = [label for label in pq.read_table(segments_path).to_pylist()
              if label["episode_index"] == episode_index]
    # Pin source files used for identity, labels and decoded observations. Large
    # shared video files are hashed once during preparation, not at every call.
    from stage_subtask_replay import required_files
    paths, _ = required_files(root, [episode_index], [])
    inventory = [{"path": path, "sha256": _sha(root / path)} for path in paths]
    metadata = {"source_kind": "original_train", "task": task, "instance_id": instance,
                "episode_index": episode_index, "task_prompt": prompt,
                "source_dataset": str(root.resolve()), "source_raw": str(raw.resolve()),
                "source_raw_group": raw_group, "source_raw_sha256": raw_sha256,
                "reader_tolerance_s": reader_tolerance_s,
                "source_files": inventory, "episode_metadata_sha256": _sha(metadata_path),
                "segments_sha256": _sha(segments_path), "retained_episodes_sha256": _sha(retained_episodes_path),
                **success_proof}
    return _prepare(output_dir, metadata, labels, len(dataset), dataset.__getitem__, max_examples=max_examples,
                    required_subtask=required_subtask, gripper_phase_actions=phase_actions)


def prepare_collection_demonstrations(trajectory_path, output_dir, *, max_examples=6):
    """Add an environment-confirmed 311--320 success to the reusable example pool."""
    info = inspect_codex_collection_trajectory(trajectory_path, require_success=True)
    source = info["metadata"]
    metadata = {"source_kind": "verified_collection", "task": source["task"],
                "instance_id": source["instance_id"], "attempt_id": source["attempt_id"],
                "task_prompt": source.get("task_prompt"), "source_trajectory": str(Path(trajectory_path).resolve()),
                "source_sha256": _sha(trajectory_path), "content_sha256": info["content_sha256"],
                "success_source": "environment_task_and_final_goal"}
    with h5py.File(trajectory_path, "r") as stream:
        def reader(frame):
            return {key: stream[key][frame] for key in ("observation.state", "action", *CAMERAS)}
        return _prepare(output_dir, metadata, info["segments"], info["steps"], reader,
                        max_examples=max_examples)


def demonstration_context(manifest_path, task, previous_subtask=None, *, limit=1):
    """Return small JSON context and its montage paths, checked before inference."""
    if type(limit) is not int or not 1 <= limit <= 2:
        raise ValueError("At most two visual examples per teacher decision")
    path = Path(manifest_path)
    data = json.loads(path.read_text())
    if data.get("kind") != KIND or data.get("fingerprint") != _digest({k: v for k, v in data.items() if k != "fingerprint"}):
        raise ValueError("Demonstration manifest changed or has unknown format")
    metadata = data["metadata"]
    _split(metadata["source_kind"], metadata["instance_id"])
    if metadata["task"] != task:
        raise ValueError("Demonstration task differs from collection task")
    words = set((previous_subtask or "").lower().split())
    ranked = sorted(data["examples"], key=lambda row: (-len(words & set(row["subtask"].lower().split())), row["start"]))
    examples, images = [], []
    for example in ranked[:limit]:
        image = path.parent / example["image"]
        if image.parent.resolve() != path.parent.resolve() or _sha(image) != example["image_sha256"]:
            raise ValueError("Demonstration montage changed or escapes package")
        images.append(image.resolve())
        examples.append({key: value for key, value in example.items() if key not in {"image", "image_sha256"}})
    context = {"demonstration_fingerprint": data["fingerprint"], "source_kind": metadata["source_kind"],
               "source_instance_id": metadata["instance_id"], "subtask_sequence": data["subtask_sequence"],
               "image_layout": data["image_layout"], "action_semantics": data["action_semantics"],
               "examples": examples}
    return context, images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    original = subs.add_parser("original")
    for name in ("dataset-root", "segments-path", "raw-path", "retained-episodes-path"):
        original.add_argument("--" + name, required=True, type=Path)
    original.add_argument("--episode-index", required=True, type=int)
    original.add_argument("--expected-raw-sha256", help="Frozen raw source SHA256; mandatory for the standard release schema")
    original.add_argument("--required-subtask", help="Predeclared local label whose segments must appear among the bounded examples")
    original.add_argument("--gripper-phase-reference", action="store_true",
                          help="Show recorded observations around the first full-close command in each example")
    collected = subs.add_parser("collection")
    collected.add_argument("--trajectory-path", required=True, type=Path)
    for child in (original, collected):
        child.add_argument("--output-dir", required=True, type=Path)
        child.add_argument("--max-examples", type=int, default=6)
    args = vars(parser.parse_args())
    command = args.pop("command")
    result = (prepare_original_demonstrations if command == "original" else prepare_collection_demonstrations)(**args)
    print(json.dumps({"fingerprint": result["fingerprint"], "examples": len(result["examples"]),
                      "metadata": result["metadata"]}, indent=2))


if __name__ == "__main__":
    main()
