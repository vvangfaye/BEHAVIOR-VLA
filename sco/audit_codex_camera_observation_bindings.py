"""Supplement frozen-run audits with local observation/reference cross-checks.

This does not replace source, qualification, native-monitor, or video audits.
Inputs are read-only; only an explicitly named, previously absent report is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


CAMERAS = (
    "observation.rgb.zed_link_camera_0",
    "observation.rgb.left_realsense_link_camera_0",
    "observation.rgb.right_realsense_link_camera_0",
)
TARGET = "digital_camera_87"
TASK = "attach_a_camera_to_a_tripod"
REFERENCE_SHA256 = "e4ee702ddd522a319a25cd08b54caeb2a057c5aa8fd9d46fbad2b1413984c74c"
REFERENCE_FINGERPRINT = "16c39fa12f0a95617948275ac7b997dc581ed62413a0f0c5a350d4cfc0d9b72f"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def local_file(directory, name):
    require(isinstance(name, str) and Path(name).name == name, "Image path is not a local filename")
    path = directory / name
    require(path.resolve().parent == directory.resolve(), "Image escapes its evidence directory")
    return path


def audit_bindings(attempt, reference_manifest, *, require_success=False,
                   expected_reference_sha256=REFERENCE_SHA256,
                   expected_reference_fingerprint=REFERENCE_FINGERPRINT):
    attempt, reference_manifest = Path(attempt), Path(reference_manifest)
    reference = load(reference_manifest)
    fingerprint = hashlib.sha256(json.dumps(
        {k: v for k, v in reference.items() if k != "fingerprint"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    require(reference.get("fingerprint") == fingerprint == expected_reference_fingerprint,
            "Reference manifest fingerprint differs from the frozen selection")
    require(reference.get("kind") == "codex_collection_demonstrations_v1"
            and reference["metadata"]["task"] == TASK
            and reference["metadata"]["source_kind"] == "original_train",
            "Reference is not the original training camera demonstration")
    examples = reference["examples"]
    require(len(examples) == 1, "Expected exactly one frozen reference image")
    example = examples[0]
    require(example["image_sha256"] == expected_reference_sha256
            == sha(local_file(reference_manifest.parent, example["image"])),
            "Reference PNG differs from the frozen SHA256")
    completed = load(attempt / "completed.json")
    result = completed["result"]
    matched, unavailable, terminal_indices = 0, [], {}
    folders = sorted((attempt / "teacher").glob("step-*"))
    require(bool(folders), "No teacher requests to bind")
    with h5py.File(attempt / "trajectory.hdf5", "r") as stream:
        require(bool(stream.attrs.get("complete")), "Trajectory is incomplete")
        require(json.loads(stream.attrs["result"]) == result, "Completed and HDF5 results differ")
        metadata = json.loads(stream.attrs["metadata"])
        n = stream.attrs["control_steps"]
        require(n > 0 and n == result["steps"], "Executed control count differs")
        require(result["task"] == TASK and metadata["sample"]["task"] == TASK,
                "Attempt is not the camera task")
        predicate = result["predicate"]
        require(predicate["operation"] == "grasped" and predicate["targets"] == [TARGET]
                and metadata["sample"]["predicate"] == predicate, "Wrong native grasp target")
        monitor = result["monitor"]
        require(monitor["predicate"] == predicate, "Monitor predicate differs")
        require(result["setup_evidence"]["initial_predicate_value"] is False
                and monitor["setup_evidence"]["initial_predicate_value"] is False,
                "Initial predicate must be false")
        initial, final = monitor["initial_snapshot"], monitor["final_snapshot"]
        require(initial["operation"] == "grasped"
                and initial["targets"] == [{"object": TARGET, "grasped": False}],
                "Initial native target snapshot is not false")
        require(final["operation"] == "grasped" and len(final["targets"]) == 1
                and final["targets"][0]["object"] == TARGET
                and type(final["targets"][0]["grasped"]) is bool,
                "Final native target identity is invalid")
        require(type(final["step"]) is int and final["step"] == n,
                "Final native snapshot is not from the final executed control")
        for key in ("observation.state", *CAMERAS, "action"):
            require(len(stream[key]) == n, "Observation/action array length differs")
        for name in ("terminated", "truncated"):
            flags = stream["next." + name][:]
            require(flags.shape == (n, 1) and flags.dtype == np.bool_, "Invalid terminal flag array")
            indices = np.flatnonzero(flags[:, 0]).tolist()
            require(not indices or indices == [n - 1], "Terminal flag occurs before the final control")
            require(type(result[name]) is bool and result[name] == bool(flags[-1, 0]),
                    "Final terminal flag differs from result")
            terminal_indices[name] = indices
        for folder in folders:
            outer = load(folder / "request.json")
            requests = list((folder / "decision").glob("*/request.json"))
            require(len(requests) == 1, "Ambiguous frozen teacher request")
            request_path = requests[0]
            envelope = load(request_path)
            require(envelope["request"] == outer, "Outer and frozen teacher requests differ")
            require(outer["camera_order"] == ["head", "left_wrist", "right_wrist"],
                    "Current camera order differs")
            images, contexts = envelope["images"], outer["context_images"]
            require(len(images) == 3 + len(contexts) and len(images) >= 4,
                    "Image list does not match current and context images")
            reference_indices = [i for i, c in enumerate(contexts)
                                 if c.get("kind") == "successful_demonstration"]
            require(reference_indices == [0] and contexts[0].get("current_observation") is False,
                    "Frozen reference must be the fourth, non-current image")
            require(outer["demonstration"]["demonstration_fingerprint"] == fingerprint,
                    "Teacher reference fingerprint differs")
            paths = [local_file(request_path.parent, item["file"]) for item in images]
            require(all(sha(path) == item["sha256"] for path, item in zip(paths, images)),
                    "Frozen teacher image SHA256 differs")
            require(images[3]["sha256"] == expected_reference_sha256,
                    "Teacher did not receive the frozen reference PNG")
            step = outer["step"]
            require(type(step) is int and 0 <= step <= n, "Teacher step is outside trajectory")
            if step == n:
                # HDF5 stores N pre-action observations, not observation N after the last action.
                unavailable.append({"step": step, "request_id": envelope["request_id"],
                                    "reason": "post_final_observation_not_recorded"})
                continue
            state = np.asarray(outer["proprio"], dtype=np.float64)
            require(np.isfinite(state).all() and np.array_equal(state, stream["observation.state"][step]),
                    f"Teacher proprio differs from pre-action observation at step {step}")
            for key, path in zip(CAMERAS, paths[:3]):
                with Image.open(path) as image:
                    require(image.mode == "RGB" and np.array_equal(np.asarray(image), stream[key][step]),
                            f"Teacher {key} differs from pre-action observation at step {step}")
            matched += 1
        require(type(result["success"]) is bool, "Invalid result success flag")
        if result["success"] or require_success:
            require(result["success"] is True and result["metric_eligible"] is True
                    and completed["training_eligible"] is True, "Attempt is not a qualified success")
            require(not unavailable, "Successful attempt has an unverifiable post-final teacher observation")
            require(monitor["satisfied"] is True and monitor["debounce_steps"] == 3
                    and monitor["dwell"] >= 3 and final["targets"][0]["grasped"] is True,
                    "Successful attempt lacks final native grasp dwell")
    return dict(kind="codex_camera_observation_bindings_v1",
                state="bindings_verified_with_unavailable_observations" if unavailable else "bindings_verified",
                result_success=result["success"], claimed_training_eligible=completed["training_eligible"],
                full_success_acceptance=False, attempt=str(attempt.resolve()),
                reference_manifest=str(reference_manifest.resolve()), reference_manifest_sha256=sha(reference_manifest),
                reference_png_sha256=expected_reference_sha256, reference_fingerprint=fingerprint,
                executed_steps=int(n), teacher_requests=len(folders), matched_requests=matched,
                matched_current_images=matched * 3, unavailable_observations=unavailable,
                terminal_flag_indices=terminal_indices, final_snapshot_step=final["step"],
                limitations=["Supplement only: run frozen source/qualification/action/native-monitor and video audits.",
                             "Final native dwell is reported by the frozen monitor; per-step native truth is not exported."])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("attempt", type=Path)
    parser.add_argument("reference_manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--require-success", action="store_true")
    args = parser.parse_args()
    require(not args.output.exists(), "Preserve existing audit report")
    report = audit_bindings(args.attempt, args.reference_manifest, require_success=args.require_success)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
