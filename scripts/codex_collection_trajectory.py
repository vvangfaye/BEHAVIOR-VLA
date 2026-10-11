"""Aligned, continuous whole-task demonstrations collected with a Codex teacher.

The caller checks the actual policy response or executed teacher correction.
Only environment goal evidence can make a completed attempt eligible for export.
This schema is deliberately separate from the fixed-subtask collection schema.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from subtask_trajectory import CAMERAS, TrajectoryWriter, array

KIND = "codex_teacher_full_task_collection"
SCHEMA_VERSION = 2
FRAME_FIELDS = ("observation.state", *CAMERAS)
ARRAY_FIELDS = (*FRAME_FIELDS, "action", "next.terminated", "next.truncated")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    if any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must not contain control characters")
    return " ".join(value.split())


def _metadata(metadata):
    result = json.loads(_json(metadata))
    if not isinstance(result, dict):
        raise ValueError("metadata must be a dictionary")
    for key in ("task", "attempt_id"):
        result[key] = _text(result.get(key), key)
    instance = result.get("instance_id")
    if type(instance) is not int or not 311 <= instance <= 320:
        raise ValueError("Teacher collection requires instance_id 311..320")
    checkpoint = result.get("checkpoint")
    if isinstance(checkpoint, str):
        result["checkpoint"] = _text(checkpoint, "checkpoint")
    elif not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("checkpoint must identify the collected policy")
    if result.get("conditioning") != "task_and_subtask":
        raise ValueError("conditioning must be task_and_subtask")
    if type(result.get("fps")) is not int or result["fps"] <= 0:
        raise ValueError("fps must be a positive integer")
    if result.get("official_benchmark", False) is not False:
        raise ValueError("Teacher collection cannot be an official benchmark result")
    result["official_benchmark"] = False
    mode = result.get("collection_mode", "teacher_conditioned")
    if mode not in {"teacher_conditioned", "native_with_corrections"}:
        raise ValueError("Unknown collection mode")
    source = "mixed_native_and_teacher" if mode == "native_with_corrections" else "provided"
    if result.get("subtask_source", source) != source:
        raise ValueError("Subtask source differs from collection mode")
    # Do not add a missing mode to historical metadata: it is part of the exact
    # manifest hash in legacy schema-2 files. New collectors set it explicitly.
    result["subtask_source"] = source
    return result


def _frame(frame):
    if set(frame) != set(FRAME_FIELDS):
        raise ValueError("Frame must contain proprioception and exactly three onboard RGB cameras")
    result = {key: array(frame[key]) for key in FRAME_FIELDS}
    state = result["observation.state"]
    if state.ndim != 1 or not 57 <= len(state) <= 512 or state.dtype != np.float32:
        raise ValueError("Expected complete float32 R1Pro proprioception")
    if not np.isfinite(state).all():
        raise ValueError("Nonfinite proprioception")
    for key in CAMERAS:
        image = result[key]
        if (image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8
                or any(not 1 <= size <= 224 for size in image.shape[:2])):
            raise ValueError(f"Invalid RGB camera {key}; use the default 224x224 wrapper")
    return result


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _result(result, count, terminal):
    result = json.loads(_json(result))
    if type(result.get("steps")) is not int or result["steps"] != count:
        raise ValueError("Result steps must equal the number of executed actions")
    task_success = _boolean(result.get("task_success"), "environment task_success")
    goal = result.get("final_goal_state")
    if not isinstance(goal, dict):
        raise ValueError("Environment final_goal_state is required")
    goal_success = _boolean(goal.get("success"), "final_goal_state.success")
    if task_success != goal_success:
        raise ValueError("Environment task_success and final goal success disagree")
    if "success" in result and _boolean(result["success"], "success") != task_success:
        raise ValueError("Claimed success differs from environment whole-task success")
    flags = tuple(_boolean(result.get(key), key) for key in ("terminated", "truncated"))
    if flags != terminal:
        raise ValueError("Result terminal flags differ from the final executed step")
    if task_success and (count == 0 or not flags[0]):
        raise ValueError("Whole-task success requires a nonempty environment-terminated trajectory")
    result["success"] = task_success
    result["success_source"] = "environment_task_and_final_goal"
    result["official_benchmark"] = False
    return result


def subtask_segments(labels):
    """Return the exact [start, end) tiling of actual executed labels."""
    segments = []
    for index, label in enumerate(labels):
        text = _text(label, "subtask")
        if segments and segments[-1]["subtask"] == text:
            segments[-1]["end"] = index + 1
        else:
            segments.append({"start": index, "end": index + 1, "subtask": text})
    return segments


class CodexCollectionTrajectory:
    """One continuous attempt; a simulator reset must start a new file.

    ``record`` is called after a successful environment step, using its preceding
    observation, executed action and actual policy-reported subtask. Inference
    candidates and responses not yet executed must never be passed here.
    """

    def __init__(self, path, metadata):
        self.metadata = _metadata(metadata)
        self._writer = TrajectoryWriter(path, self.metadata)
        self.path = Path(path)
        self._labels = []
        self._terminal = (False, False)
        self._shapes = None
        stream = self._writer.file
        stream.attrs["kind"] = KIND
        stream.attrs["schema_version"] = SCHEMA_VERSION
        stream.create_dataset("subtask", shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"))
        stream.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype=np.int64)

    @property
    def count(self):
        return self._writer.count

    def record(self, frame, action, terminated, truncated, subtask):
        if self._writer.closed:
            raise ValueError("Trajectory is already closed")
        if any(self._terminal):
            raise ValueError("Cannot append after a terminal step; start a new attempt")
        label = _text(subtask, "subtask")
        terminal = (_boolean(terminated, "terminated"), _boolean(truncated, "truncated"))
        prepared = _frame(frame)
        shapes = {key: value.shape for key, value in prepared.items()}
        if self._shapes is not None and shapes != self._shapes:
            raise ValueError("Observation dimensions changed within the attempt")
        executed = array(action).reshape(-1).astype(np.float32)
        if executed.shape != (23,) or not np.isfinite(executed).all():
            raise ValueError("Executed action must be a finite 23-dimensional control")
        self._writer.record(prepared, executed, *terminal)
        for name, value in (("subtask", label), ("frame_index", self.count - 1)):
            dataset = self._writer.file[name]
            dataset.resize(self.count, axis=0)
            dataset[self.count - 1] = value
        self._labels.append(label)
        self._shapes = shapes
        self._terminal = terminal

    def finish(self, result):
        if self._writer.closed:
            raise ValueError("Trajectory is already closed")
        result = _result(result, self.count, self._terminal)
        if len(self._labels) != self.count:
            raise ValueError("Missing executed subtask labels")
        segments = subtask_segments(self._labels)
        stream = self._writer.file
        stream.attrs["segments"] = _json(segments)
        stream.attrs["subtask_sha256"] = _digest(self._labels)
        stream.attrs["manifest_sha256"] = _digest([self.metadata, result, segments])
        self._writer.finish(result)
        return result

    def close(self):
        """Close without finish: the attempt remains explicitly incomplete."""
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def inspect_codex_collection_trajectory(path, *, require_success=True):
    """Validate continuity, labels, environment outcome and unchanged arrays."""
    with h5py.File(path, "r") as stream:
        if (stream.attrs.get("kind") != KIND or stream.attrs.get("schema_version") != SCHEMA_VERSION
                or not stream.attrs.get("complete")):
            raise ValueError("Unfinished or unknown Codex whole-task trajectory")
        raw_metadata = json.loads(stream.attrs["metadata"])
        metadata = _metadata(raw_metadata)
        if metadata != raw_metadata:
            raise ValueError("Noncanonical trajectory metadata")
        count = int(stream.attrs["control_steps"])
        if count < 0 or "subtask" not in stream or "frame_index" not in stream:
            raise ValueError("Missing trajectory labels or frame indices")
        if stream["subtask"].shape != (count,) or stream["frame_index"].shape != (count,):
            raise ValueError("Subtask labels and executed frames are not aligned")
        if not np.array_equal(stream["frame_index"][:], np.arange(count)):
            raise ValueError("Skipped or repeated frame index")
        labels = stream["subtask"].asstr()[:].tolist()
        segments = subtask_segments(labels)
        if json.loads(stream.attrs["segments"]) != segments or _digest(labels) != stream.attrs["subtask_sha256"]:
            raise ValueError("Subtask labels or segment coverage changed")
        for name in ARRAY_FIELDS:
            if count and (name not in stream or len(stream[name]) != count):
                raise ValueError(f"Missing or unaligned trajectory array: {name}")
        hasher = hashlib.sha256()
        terminal = (False, False)
        shapes = None
        for index in range(count):
            if any(terminal):
                raise ValueError("Trajectory contains actions after termination or reset")
            frame = _frame({key: stream[key][index] for key in FRAME_FIELDS})
            shapes = {key: list(value.shape) for key, value in frame.items()}
            action = stream["action"][index]
            if action.dtype != np.float32 or action.shape != (23,) or not np.isfinite(action).all():
                raise ValueError("Invalid executed action")
            flags = [stream[key][index] for key in ("next.terminated", "next.truncated")]
            if any(flag.shape != (1,) or flag.dtype != np.bool_ for flag in flags):
                raise ValueError("Invalid per-step terminal flags")
            terminal = tuple(bool(flag[0]) for flag in flags)
            for key in ARRAY_FIELDS:
                hasher.update(key.encode())
                hasher.update(stream[key][index].tobytes())
        digest = hasher.hexdigest()
        if digest != stream.attrs["content_sha256"]:
            raise ValueError("Trajectory arrays changed after collection")
        raw_result = json.loads(stream.attrs["result"])
        result = _result(raw_result, count, terminal)
        if result != raw_result or _digest([metadata, result, segments]) != stream.attrs["manifest_sha256"]:
            raise ValueError("Trajectory outcome or provenance changed after collection")
        if require_success and not result["success"]:
            raise ValueError("Only environment-confirmed whole-task successes can be training data")
        return {
            "metadata": metadata, "result": result, "segments": segments, "steps": count,
            "content_sha256": digest, "arrays_sha256": digest,
            "subtask_sha256": _digest(labels),
            "state_shape": shapes["observation.state"] if shapes else None,
            "camera_shapes": {key: shapes[key] for key in CAMERAS} if shapes else {},
        }
