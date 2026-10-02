#!/usr/bin/env python3
"""Replay approved BEHAVIOR training demos into GoalKeeper records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np

_PACKAGE = Path(__file__).resolve().parents[1] / "src"
if str(_PACKAGE) not in sys.path:
    sys.path.insert(0, str(_PACKAGE))

from goalkeeper.data import write_record_archive
from goalkeeper.goals import compile_bddl

APPROVED_TASKS = {"carrying_in_groceries": 14, "make_rose_centerpieces": 56}
DEFAULT_KEYS = {
    "rgb_head_key": "robot::robot:zed_link:Camera:0::rgb",
    "rgb_left_key": "robot::robot:left_realsense_link:Camera:0::rgb",
    "rgb_right_key": "robot::robot:right_realsense_link:Camera:0::rgb",
    "proprio_key": "robot::robot:proprio::proprio",
}


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evaluate_compiled_atoms(task, program) -> np.ndarray:
    """Evaluate compiler-grounded atoms through the official simulator backend."""
    from bddl.predicates import TOKEN_TO_PREDICATE

    values = []
    for atom in program.atoms:
        predicate = TOKEN_TO_PREDICATE.get(atom.predicate)
        if predicate is None:
            raise ValueError(f"No official BDDL predicate class for {atom.predicate!r}")
        values.append(bool(task._evaluate_predicate(predicate, *atom.arguments)))
    result = np.asarray(values, dtype=np.float32)
    if result.shape != (len(program.atoms),) or not np.isin(result, [0, 1]).all():
        raise ValueError("Simulator must return one boolean value per compiled atom")
    return result


def _numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _rgb(value) -> np.ndarray:
    array = _numpy(value)
    if array.ndim == 4 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 3 or array.shape[-1] not in (3, 4):
        raise ValueError("RGB observations must have shape [height,width,3 or 4]")
    array = array[..., :3]
    if np.issubdtype(array.dtype, np.floating):
        if not np.isfinite(array).all() or array.min() < 0 or array.max() > 1:
            raise ValueError("Floating RGB observations must be finite in [0,1]")
        array = np.rint(array * 255)
    return np.asarray(array, dtype=np.uint8)


def _stable_group_id(approval: dict, initial_state: np.ndarray) -> str:
    provenance = {
        "task": approval["task"],
        "task_id": approval["task_id"],
        "scene_model": approval["scene_model"],
        "scene_instance": approval["scene_instance"],
        "instance_id": approval["instance_id"],
        "initial_state_sha256": hashlib.sha256(np.ascontiguousarray(initial_state).view(np.uint8)).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]
    return f"b1k-training-{approval['task_id']:03d}-{digest}"


def validate_approval(
    input_path: str | Path,
    bddl: str | Path,
    approval_path: str | Path,
    task_name: str,
    episode_id: int,
) -> tuple[dict, str]:
    """Bind replay to a reviewed training file, task, demo group, and initial state."""
    input_path = Path(input_path).resolve()
    bddl = Path(bddl).resolve()
    approval_path = Path(approval_path).resolve()
    approval = json.loads(approval_path.read_text())
    required = {
        "approved",
        "source",
        "task",
        "task_id",
        "episode_id",
        "scene_model",
        "scene_instance",
        "instance_id",
        "demo_sha256",
        "bddl_sha256",
    }
    if not isinstance(approval, dict) or not required <= approval.keys():
        missing = required - set(approval) if isinstance(approval, dict) else required
        raise ValueError(f"Approval is missing required fields: {missing}")
    expected_id = APPROVED_TASKS.get(task_name)
    if approval["approved"] is not True or approval["source"] != "training_demo":
        raise ValueError("Only explicitly approved training_demo inputs may be replayed")
    if task_name not in APPROVED_TASKS or approval["task"] != task_name or approval["task_id"] != expected_id:
        raise ValueError("Replay is limited to the configured task name/ID pairs")
    if type(episode_id) is not int or episode_id < 0 or approval["episode_id"] != episode_id:
        raise ValueError("Explicit demo group ID does not match approval")
    if _sha256(input_path) != approval["demo_sha256"] or _sha256(bddl) != approval["bddl_sha256"]:
        raise ValueError("Approved input or BDDL content hash mismatch")
    if any(not isinstance(approval[key], str) or not approval[key]
           for key in ("scene_model", "scene_instance")):
        raise ValueError("Approval requires stable scene model and instance provenance")
    if not isinstance(approval["instance_id"], (str, int)):
        raise ValueError("Approval requires a training initial-state ID")

    import h5py

    with h5py.File(input_path, "r") as data:
        root = data["data"]
        config = json.loads(root.attrs["config"])
        recorded_task = config.get("task", {}).get("activity_name")
        if recorded_task and recorded_task != task_name:
            raise ValueError("Recorded HDF5 task disagrees with approval")
        scene = config.get("scene", {})
        if (scene.get("scene_model"), scene.get("scene_instance")) != (
            approval["scene_model"], approval["scene_instance"]
        ):
            raise ValueError("Recorded HDF5 scene model/instance disagrees with approval")
        key = f"demo_{episode_id}"
        if key not in root:
            raise ValueError(f"Approved demo group does not exist: {key}")
        demo = root[key]
        state = np.asarray(demo["state"][0])
        if "state_size" in demo:
            state = state[: int(demo["state_size"][0])]
    return approval, _stable_group_id(approval, state)


def _challenge_scene(task_name: str, scene_model: str, scene_instance: str) -> tuple[str, list[str], str]:
    import yaml
    from omnigibson.macros import gm

    metadata = Path(gm.DATA_PATH) / "2026-challenge-task-instances" / "metadata"
    available = yaml.safe_load((metadata / "available_tasks.yaml").read_text())
    if task_name not in available:
        raise KeyError(f"Task is absent from challenge metadata: {task_name}")
    allowed_models = {item["scene_model"] for item in available[task_name]}
    if scene_model not in allowed_models:
        raise ValueError(f"Scene model {scene_model!r} is not approved for {task_name}")
    scene_path = (
        Path(gm.DATA_PATH)
        / "2026-challenge-task-instances"
        / "scenes"
        / scene_model
        / "json"
        / f"{scene_instance}.json"
    )
    if not scene_path.is_file():
        raise FileNotFoundError(f"Recorded full scene file is missing: {scene_path}")
    rooms = None
    with (metadata / "B100_task_misc.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row.get("Task") == task_name:
                value = row.get("Room Instance") or row.get("Room Instances") or ""
                rooms = [item for item in value.strip().splitlines() if item]
                break
    if not rooms:
        with (metadata / "B100_task_misc.csv").open(newline="", encoding="utf-8") as stream:
            for row in csv.reader(stream):
                if len(row) > 2 and task_name == row[1]:
                    rooms = [item for item in row[2].strip().splitlines() if item]
                    break
    if not rooms:
        raise ValueError(f"No room instances recorded for {task_name}")
    return str(scene_path), rooms, scene_model


def _make_wrapper(program, observation_keys, *, capture_stride: int, features=None):
    from omnigibson.envs.hdf5_data_wrapper import HDF5PlaybackWrapper

    class GoalKeeperPlaybackWrapper(HDF5PlaybackWrapper):
        def __init__(self, *args, **kwargs):
            self.goalkeeper_frames = []
            self._goalkeeper_capture_active = False
            self._goalkeeper_capture_calls = 0
            super().__init__(*args, **kwargs)

        def playback_episode(self, *args, **kwargs):
            self.goalkeeper_frames = []
            self._goalkeeper_capture_calls = 0
            self._goalkeeper_capture_active = True
            try:
                return super().playback_episode(*args, **kwargs)
            finally:
                self._goalkeeper_capture_active = False

        def _process_obs(self, obs, info):
            processed = super()._process_obs(obs, info)
            if not self._goalkeeper_capture_active:
                return {}
            call_index = self._goalkeeper_capture_calls
            self._goalkeeper_capture_calls += 1
            if call_index == 0:
                return {}
            step = call_index - 1
            if step % capture_stride:
                return {}
            missing = [key for key in observation_keys if key not in processed]
            if missing:
                raise KeyError(f"Missing replay observations: {missing}; available={sorted(processed)}")
            head, left, right, proprio = (processed[key] for key in observation_keys)
            images = (_rgb(head), _rgb(left), _rgb(right))
            frame = {
                "step": step,
                "proprio": _numpy(proprio).astype(np.float32).reshape(-1),
                "truth": evaluate_compiled_atoms(self.env.task, program),
            }
            if features is None:
                frame.update(rgb_head=images[0], rgb_left=images[1], rgb_right=images[2])
            else:
                frame["visual"] = features.encode_images(images)
            self.goalkeeper_frames.append(frame)
            return {}

    return GoalKeeperPlaybackWrapper


def _open_wrapper(input_path, output_path, task, program, keys, *, scene_model, scene_instance,
                  capture_stride, features=None, render_iterations=1):
    import omnigibson as og
    from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES
    from omnigibson.macros import gm

    gm.ENABLE_TRANSITION_RULES = False
    full_scene_file, rooms, scene_model = _challenge_scene(task, scene_model, scene_instance)
    sensor_config = {
        "VisionSensor": {"sensor_kwargs": {"image_height": 480, "image_width": 480}},
        "zed_link:Camera:0": {
            "sensor_kwargs": {"horizontal_aperture": 40.0, "image_height": 720, "image_width": 720}
        },
    }
    wrapper_type = _make_wrapper(program, keys, capture_stride=capture_stride, features=features)
    wrapper = wrapper_type.create_from_hdf5(
        input_path=str(input_path),
        output_path=str(output_path),
        full_scene_file=full_scene_file,
        load_room_instances=rooms,
        robot_sensor_config=sensor_config,
        robot_proprio_keys=list(PROPRIOCEPTION_INDICES["R1Pro"].keys()),
        robot_obs_modalities=("rgb", "proprio"),
        n_render_iterations=render_iterations,
        overwrite=True,
        only_successes=False,
        include_task=True,
        include_robot_control=False,
        include_contacts=False,
    )
    return wrapper, scene_model, og


def _load_actions(input_path: Path, episode_id: int) -> np.ndarray:
    import h5py

    with h5py.File(input_path, "r") as data:
        return np.asarray(data["data"][f"demo_{episode_id}"]["action"], dtype=np.float32)


def _close_wrapper(wrapper, og) -> None:
    if wrapper is None:
        return
    try:
        wrapper.close_dataset()
    finally:
        wrapper.input_hdf5.close()
        if hasattr(wrapper.env, "close"):
            wrapper.env.close()
        if hasattr(og, "shutdown"):
            og.shutdown()


def replay(
    input_path: str | Path,
    output: str | Path,
    bddl: str | Path,
    approval: str | Path,
    *,
    task: str,
    episode_id: int,
    rgb_head_key: str = DEFAULT_KEYS["rgb_head_key"],
    rgb_left_key: str = DEFAULT_KEYS["rgb_left_key"],
    rgb_right_key: str = DEFAULT_KEYS["rgb_right_key"],
    proprio_key: str = DEFAULT_KEYS["proprio_key"],
    render_iterations: int = 1,
    max_steps: int = 2048,
) -> dict:
    """Emit a diagnostic dense episode NPZ; formal long episodes use replay_features."""
    input_path, output, bddl = Path(input_path).resolve(), Path(output).resolve(), Path(bddl).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace replay output: {output}")
    approval_data, group_id = validate_approval(input_path, bddl, approval, task, episode_id)
    program = compile_bddl(bddl.read_text())
    actions = _load_actions(input_path, episode_id)
    if len(actions) > max_steps:
        raise ValueError("Dense replay would retain too many RGB frames; use --encoder for streaming feature export")
    keys = (rgb_head_key, rgb_left_key, rgb_right_key, proprio_key)
    wrapper = og = None
    with tempfile.TemporaryDirectory(prefix="goalkeeper-replay-") as temporary:
        wrapper, scene_model, og = _open_wrapper(
            input_path, Path(temporary) / "observations.hdf5", task, program, keys,
            scene_model=approval_data["scene_model"], scene_instance=approval_data["scene_instance"],
            capture_stride=1, render_iterations=render_iterations,
        )
        if scene_model != approval_data["scene_model"]:
            raise ValueError("Challenge scene metadata disagrees with approval")
        try:
            wrapper.playback_episode(episode_id=episode_id, record_data=True)
            frames = wrapper.goalkeeper_frames
        finally:
            _close_wrapper(wrapper, og)
    if len(frames) != len(actions) + 1:
        raise ValueError(f"Replay produced {len(frames)} observations for {len(actions)} actions")
    arrays = {
        name: np.stack([frame[name] for frame in frames])
        for name in ("rgb_head", "rgb_left", "rgb_right", "proprio", "truth")
    }
    visibility = np.full(arrays["truth"].shape, -1, dtype=np.int8)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(
        temporary_output,
        **arrays,
        actions=actions,
        visibility=visibility,
        atom_keys=np.asarray([atom.key for atom in program.atoms]),
        group_id=np.asarray(group_id),
        source=np.asarray("training_demo"),
        task=np.asarray(task),
        task_id=np.asarray(APPROVED_TASKS[task], dtype=np.int64),
        instance_id=np.asarray(str(approval_data["instance_id"])),
        scene_model=np.asarray(approval_data["scene_model"]),
        scene_instance=np.asarray(approval_data["scene_instance"]),
        input_sha256=np.asarray(approval_data["demo_sha256"]),
        bddl_sha256=np.asarray(approval_data["bddl_sha256"]),
        episode_id=np.asarray(episode_id, dtype=np.int64),
    )
    temporary_output.replace(output)
    return {
        "episode": str(output),
        "group_id": group_id,
        "actions": len(actions),
        "observations": len(actions) + 1,
        "atoms": len(program.atoms),
    }


def replay_features(
    input_path: str | Path,
    output: str | Path,
    bddl: str | Path,
    approval: str | Path,
    *,
    task: str,
    episode_id: int,
    features,
    action_spec: str,
    horizon: int = 16,
    history: int = 4,
    rgb_head_key: str = DEFAULT_KEYS["rgb_head_key"],
    rgb_left_key: str = DEFAULT_KEYS["rgb_left_key"],
    rgb_right_key: str = DEFAULT_KEYS["rgb_right_key"],
    proprio_key: str = DEFAULT_KEYS["proprio_key"],
    render_iterations: int = 1,
) -> dict:
    """Stream one long replay into compact, decision-spaced feature records."""
    input_path, output, bddl = Path(input_path).resolve(), Path(output).resolve(), Path(bddl).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace feature output: {output}")
    if min(horizon, history) < 1 or not action_spec:
        raise ValueError("Positive horizon/history and explicit action_spec are required")
    approval_data, group_id = validate_approval(input_path, bddl, approval, task, episode_id)
    program = compile_bddl(bddl.read_text())
    actions = _load_actions(input_path, episode_id)
    keys = (rgb_head_key, rgb_left_key, rgb_right_key, proprio_key)
    wrapper = og = None
    with tempfile.TemporaryDirectory(prefix="goalkeeper-feature-replay-") as temporary:
        wrapper, scene_model, og = _open_wrapper(
            input_path, Path(temporary) / "observations.hdf5", task, program, keys,
            scene_model=approval_data["scene_model"], scene_instance=approval_data["scene_instance"],
            capture_stride=horizon, features=features, render_iterations=render_iterations,
        )
        if scene_model != approval_data["scene_model"]:
            raise ValueError("Challenge scene metadata disagrees with approval")
        try:
            wrapper.playback_episode(episode_id=episode_id, record_data=True)
            frames = list(wrapper.goalkeeper_frames)
        finally:
            _close_wrapper(wrapper, og)
    expected_steps = list(range(0, len(actions) + 1, horizon))
    observed_steps = [frame["step"] for frame in frames]
    if observed_steps != expected_steps:
        raise ValueError(f"Decision-step replay mismatch: observed={observed_steps[:5]} expected={expected_steps[:5]}")

    output.mkdir(parents=True)
    task_path = output / f"{program.source_sha256}.bddl"
    task_path.write_text(bddl.read_text())
    queries = features.encode_queries([program.query(atom) for atom in program.atoms])
    rows, records = [], []
    episode_key = hashlib.sha256(f"{group_id}:{input_path.name}:{episode_id}".encode()).hexdigest()[:20]
    archive_name = f"{episode_key}.npz"
    for index, frame in enumerate(frames[:-1]):
        if index < history - 1:
            continue
        t = frame["step"]
        if t + horizon > len(actions):
            continue
        window = frames[index - history + 1 : index + 1]
        future = frames[index + 1]
        if future["step"] != t + horizon:
            raise ValueError("Future label is not aligned to the executed action prefix")
        record_id = f"{episode_key}-{t:07d}"
        record_index = len(records)
        records.append(
            {
                "visual": np.stack([item["visual"] for item in window]),
                "proprio": np.stack([item["proprio"] for item in window]),
                "actions": actions[None, t : t + horizon],
                "truth": np.stack([item["truth"] for item in window]),
                "visibility": np.full((history, len(program.atoms)), -1, dtype=np.int8),
                "future_truth": future["truth"][None],
            }
        )
        rows.append(
            {
                "id": record_id,
                "group_id": group_id,
                "source": "training_demo",
                "record": archive_name,
                "record_index": record_index,
                "bddl": task_path.name,
                "feature_id": features.feature_id,
                "action_spec": action_spec,
            }
        )
    if not rows:
        raise ValueError("Episode is too short for the requested history and horizon")
    write_record_archive(
        output / archive_name,
        records,
        queries,
        np.asarray([atom.key for atom in program.atoms]),
    )
    manifest = output / f"{episode_key}.jsonl"
    manifest.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return {
        "manifest": str(manifest),
        "group_id": group_id,
        "actions": len(actions),
        "decision_frames": len(frames),
        "records": len(rows),
        "atoms": len(program.atoms),
        "feature_id": features.feature_id,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay an approved BEHAVIOR training HDF5 demo")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--bddl", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--task", required=True, choices=sorted(APPROVED_TASKS))
    parser.add_argument("--episode-id", required=True, type=int)
    parser.add_argument("--rgb-head-key", default=DEFAULT_KEYS["rgb_head_key"])
    parser.add_argument("--rgb-left-key", default=DEFAULT_KEYS["rgb_left_key"])
    parser.add_argument("--rgb-right-key", default=DEFAULT_KEYS["rgb_right_key"])
    parser.add_argument("--proprio-key", default=DEFAULT_KEYS["proprio_key"])
    parser.add_argument("--render-iterations", type=int, default=1)
    parser.add_argument("--encoder")
    parser.add_argument("--revision")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-spec")
    parser.add_argument("--horizon", type=int, default=16)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--max-raw-steps", type=int, default=2048)
    args = parser.parse_args(argv)
    if args.render_iterations < 1:
        parser.error("--render-iterations must be positive")
    common = {
        "input_path": args.input,
        "output": args.output,
        "bddl": args.bddl,
        "approval": args.approval,
        "task": args.task,
        "episode_id": args.episode_id,
        "rgb_head_key": args.rgb_head_key,
        "rgb_left_key": args.rgb_left_key,
        "rgb_right_key": args.rgb_right_key,
        "proprio_key": args.proprio_key,
        "render_iterations": args.render_iterations,
    }
    if args.encoder:
        if not args.revision or not args.action_spec:
            parser.error("--encoder requires --revision and --action-spec")
        from goalkeeper.features import SiglipFeatures

        result = replay_features(
            **common,
            features=SiglipFeatures(args.encoder, args.revision, args.device),
            action_spec=args.action_spec,
            horizon=args.horizon,
            history=args.history,
        )
    else:
        if args.revision or args.action_spec:
            parser.error("--revision/--action-spec require --encoder")
        result = replay(**common, max_steps=args.max_raw_steps)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
