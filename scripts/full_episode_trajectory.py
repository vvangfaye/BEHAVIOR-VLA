"""Continuous native-policy episodes, separate from fixed-subtask and teacher data.

Each row pairs the observation before an environment step with the action that
was actually executed. Model-generated subtask text is retained as a weak label;
missing or empty text is never filled from privileged simulator annotations.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

from subtask_trajectory import CAMERAS, TrajectoryWriter, array, capture_observation

KIND = "native_full_episode_collection"
SCHEMA_VERSION = 3
MANIFEST_KIND = "native_full_episode_collection_manifest"
CONDITIONING = "task_and_native_generated_subtask"
FRAME_FIELDS = ("observation.state", *CAMERAS)
ARRAY_FIELDS = (*FRAME_FIELDS, "action", "next.terminated", "next.truncated")


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _text(value, name):
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 32 for char in value):
        raise ValueError(f"{name} must be nonempty text without control characters")
    return value


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be a lowercase SHA256")
    return value


def load_collection_manifest(path):
    raw = Path(path).read_bytes()
    manifest = json.loads(raw)
    if manifest.get("kind") != MANIFEST_KIND:
        raise ValueError("Unknown full-episode collection manifest")
    _text(manifest.get("run_id"), "run_id")
    _sha(manifest.get("source_manifest_sha256"), "source_manifest_sha256")
    if not isinstance(manifest.get("checkpoint"), dict) or not manifest["checkpoint"]:
        raise ValueError("Manifest must identify an immutable checkpoint")
    if not isinstance(manifest.get("task_prompts"), dict) or not manifest["task_prompts"]:
        raise ValueError("Manifest must contain whole-task prompts")
    for task, prompt in manifest["task_prompts"].items():
        _text(task, "task"); _text(prompt, "task_prompt")
    return manifest, hashlib.sha256(raw).hexdigest()


def _metadata(value):
    result = json.loads(_json(value))
    if not isinstance(result, dict):
        raise ValueError("metadata must be a dictionary")
    for key in ("task", "run_id", "attempt_id", "task_prompt"):
        _text(result.get(key), key)
    if type(result.get("instance_id")) is not int or not 301 <= result["instance_id"] <= 320:
        raise ValueError("Full-episode collection requires public instance 301..320")
    if type(result.get("rollout_id")) is not int or result["rollout_id"] < 0:
        raise ValueError("rollout_id must be a nonnegative integer")
    if not isinstance(result.get("checkpoint"), dict) or not result["checkpoint"]:
        raise ValueError("checkpoint must identify the collected policy")
    _sha(result.get("source_manifest_sha256"), "source_manifest_sha256")
    if "collection_manifest_sha256" in result:
        _sha(result["collection_manifest_sha256"], "collection_manifest_sha256")
    if type(result.get("fps")) is not int or result["fps"] <= 0:
        raise ValueError("fps must be a positive integer")
    expected = {"conditioning": CONDITIONING,
                "recording_scope": "reset_to_environment_terminal", "official_benchmark": False}
    for key, required in expected.items():
        if result.get(key, required) != required:
            raise ValueError(f"Invalid native full-episode {key}")
        result[key] = required
    return result


def _frame(frame):
    if set(frame) != set(FRAME_FIELDS):
        raise ValueError("Frame requires complete proprioception and three onboard RGB cameras")
    values = {key: array(frame[key]) for key in FRAME_FIELDS}
    state = values["observation.state"]
    if (state.ndim != 1 or state.dtype != np.float32 or not 57 <= len(state) <= 512
            or not np.isfinite(state).all()):
        raise ValueError("Invalid complete R1Pro proprioception")
    for key in CAMERAS:
        image = values[key]
        if (image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8
                or any(not 1 <= size <= 224 for size in image.shape[:2])):
            raise ValueError("Invalid onboard RGB frame: " + key)
    return values


def _boolean(value, name):
    if type(value) is not bool:
        raise ValueError(name + " must be a boolean")
    return value


def _label(label, source):
    if label is not None and not isinstance(label, str):
        raise ValueError("Native subtask must be text or null")
    if source not in (None, "generated", "cached"):
        raise ValueError("Only actual native generated/cached subtask sources are allowed")
    return [label, source]


def subtask_segments(labels):
    """Tile every executed row, retaining empty labels and their actual sources."""
    segments = []
    for index, (label, source) in enumerate(labels):
        _label(label, source)
        if segments and (segments[-1]["subtask"], segments[-1]["source"]) == (label, source):
            segments[-1]["end"] = index + 1
        else:
            segments.append({"start": index, "end": index + 1, "subtask": label, "source": source})
    return segments


def _result(value, metadata, count, terminal):
    result = json.loads(_json(value))
    for key in ("task", "instance_id", "rollout_id"):
        if type(result.get(key)) is not type(metadata[key]) or result.get(key) != metadata[key]:
            raise ValueError("Episode result identity differs from metadata: " + key)
    if type(result.get("steps")) is not int or result["steps"] != count or count <= 0:
        raise ValueError("Result steps must equal a nonempty executed episode")
    flags = tuple(_boolean(result.get(key), key) for key in ("terminated", "truncated"))
    if flags != terminal or not any(flags):
        raise ValueError("Complete episode requires the actual final environment terminal flags")
    success = _boolean(result.get("success"), "success")
    goal = result.get("final_goal_state")
    if not isinstance(goal, dict) or _boolean(goal.get("success"), "final_goal_state.success") != success:
        raise ValueError("Environment task success and final goal success disagree")
    if _boolean(result.get("task_success", success), "task_success") != success:
        raise ValueError("task_success differs from environment success")
    if success and not flags[0]:
        raise ValueError("Successful complete episode requires environment termination")
    conditioning = result.get("conditioning", {})
    if (conditioning.get("mode") != "whole_task" or conditioning.get("privileged") is not False
            or conditioning.get("oracle") is not False):
        raise ValueError("Full-episode collection requires native, nonprivileged whole-task conditioning")
    result.update(task_success=success, success_source="environment_task_and_final_goal", official_benchmark=False)
    return result


class NativeFullEpisodeTrajectory:
    """One reset-to-terminal attempt; interrupted files stay explicitly incomplete."""

    def __init__(self, path, metadata):
        self.metadata = _metadata(metadata)
        self.path = Path(path)
        self._writer = TrajectoryWriter(path, self.metadata)
        self._labels = []
        self._terminal = (False, False)
        self._shapes = None
        stream = self._writer.file
        stream.attrs.update(kind=KIND, schema_version=SCHEMA_VERSION)
        for name in ("subtask", "subtask_source"):
            stream.create_dataset(name, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"))
        stream.create_dataset("frame_index", shape=(0,), maxshape=(None,), dtype=np.int64)

    @property
    def count(self):
        return self._writer.count

    def record(self, frame, action, terminated, truncated, subtask, subtask_source):
        if self._writer.closed or any(self._terminal):
            raise ValueError("Cannot append after a closed or terminal episode")
        values = _frame(frame)
        shapes = {key: value.shape for key, value in values.items()}
        if self._shapes is not None and self._shapes != shapes:
            raise ValueError("Observation shapes changed within an episode")
        labels = _label(subtask, subtask_source)
        flags = tuple(_boolean(value, name) for value, name in
                      ((terminated, "terminated"), (truncated, "truncated")))
        executed = array(action)
        if executed.shape != (23,):
            raise ValueError("Only the one executed 23-dimensional action may be recorded")
        self._writer.record(values, executed, *flags)
        stream = self._writer.file
        for name, value in (("subtask", _json(subtask)), ("subtask_source", _json(subtask_source)),
                            ("frame_index", self.count - 1)):
            stream[name].resize(self.count, axis=0)
            stream[name][self.count - 1] = value
        self._labels.append(labels)
        self._terminal, self._shapes = flags, shapes

    def finish(self, result, terminal_observation=None):
        if self._writer.closed:
            raise ValueError("Episode is already closed")
        result = _result(result, self.metadata, self.count, self._terminal)
        segments = subtask_segments(self._labels)
        stream = self._writer.file
        terminal_hash = None
        if terminal_observation is not None:
            values = _frame(terminal_observation)
            if {key: value.shape for key, value in values.items()} != self._shapes:
                raise ValueError("Terminal observation shapes changed")
            digest = hashlib.sha256()
            for key, value in values.items():
                stream.create_dataset("terminal_observation/" + key, data=value, compression="gzip")
                digest.update(key.encode()); digest.update(value.tobytes())
            terminal_hash = digest.hexdigest()
        stream.attrs["segments"] = _json(segments)
        stream.attrs["subtask_sha256"] = _digest(self._labels)
        stream.attrs["terminal_observation_sha256"] = _json(terminal_hash)
        stream.attrs["manifest_sha256"] = _digest([self.metadata, result, segments, terminal_hash])
        self._writer.finish(result)
        return result

    def close(self):
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def inspect_full_episode_trajectory(path, *, require_success=True):
    with h5py.File(path, "r") as stream:
        if (stream.attrs.get("kind") != KIND or stream.attrs.get("schema_version") != SCHEMA_VERSION
                or not stream.attrs.get("complete")):
            raise ValueError("Unfinished or unknown native full episode")
        raw_metadata = json.loads(stream.attrs["metadata"])
        metadata = _metadata(raw_metadata)
        if metadata != raw_metadata:
            raise ValueError("Noncanonical full-episode metadata")
        count = int(stream.attrs["control_steps"])
        for key in (*ARRAY_FIELDS, "subtask", "subtask_source", "frame_index"):
            if key not in stream or len(stream[key]) != count:
                raise ValueError("Missing or unaligned full-episode array: " + key)
        if not np.array_equal(stream["frame_index"][:], np.arange(count)):
            raise ValueError("Skipped or repeated episode frame index")
        labels = [_label(json.loads(label), json.loads(source)) for label, source in
                  zip(stream["subtask"].asstr()[:], stream["subtask_source"].asstr()[:])]
        segments = subtask_segments(labels)
        if _digest(labels) != stream.attrs["subtask_sha256"] or json.loads(stream.attrs["segments"]) != segments:
            raise ValueError("Native subtask labels changed after recording")
        digest = hashlib.sha256()
        terminal, shapes = (False, False), None
        for index in range(count):
            if any(terminal):
                raise ValueError("Executed actions exist after an environment terminal step")
            values = _frame({key: stream[key][index] for key in FRAME_FIELDS})
            current_shapes = {key: list(value.shape) for key, value in values.items()}
            if shapes is not None and current_shapes != shapes:
                raise ValueError("Observation dimensions changed")
            shapes = current_shapes
            action = stream["action"][index]
            if action.shape != (23,) or action.dtype != np.float32 or not np.isfinite(action).all():
                raise ValueError("Invalid executed action")
            flags = [stream[key][index] for key in ("next.terminated", "next.truncated")]
            if any(flag.shape != (1,) or flag.dtype != np.bool_ for flag in flags):
                raise ValueError("Invalid per-step terminal flags")
            terminal = tuple(bool(flag[0]) for flag in flags)
            for key in ARRAY_FIELDS:
                digest.update(key.encode()); digest.update(stream[key][index].tobytes())
        if digest.hexdigest() != stream.attrs["content_sha256"]:
            raise ValueError("Full-episode arrays changed after collection")
        terminal_hash = json.loads(stream.attrs["terminal_observation_sha256"])
        if terminal_hash is not None:
            final = _frame({key: stream["terminal_observation/" + key][:] for key in FRAME_FIELDS})
            if {key: list(value.shape) for key, value in final.items()} != shapes:
                raise ValueError("Terminal observation dimensions changed")
            final_digest = hashlib.sha256()
            for key, value in final.items():
                final_digest.update(key.encode()); final_digest.update(value.tobytes())
            if final_digest.hexdigest() != terminal_hash:
                raise ValueError("Terminal observation changed after collection")
        elif "terminal_observation" in stream:
            raise ValueError("Unbound terminal observation")
        raw_result = json.loads(stream.attrs["result"])
        result = _result(raw_result, metadata, count, terminal)
        if (result != raw_result or _digest([metadata, result, segments, terminal_hash])
                != stream.attrs["manifest_sha256"]):
            raise ValueError("Full-episode outcome or provenance changed")
        if require_success and not result["success"]:
            raise ValueError("Only environment-confirmed full-episode successes can be BC data")
        return {"metadata": metadata, "result": result, "steps": count, "segments": segments,
                "content_sha256": digest.hexdigest(), "arrays_sha256": digest.hexdigest(),
                "subtask_sha256": _digest(labels), "terminal_observation_sha256": terminal_hash,
                "terminal_observation_recorded": terminal_hash is not None,
                "state_shape": shapes["observation.state"],
                "camera_shapes": {key: shapes[key] for key in CAMERAS}}
