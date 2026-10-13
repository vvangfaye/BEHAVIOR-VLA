"""Synthetic binding fixtures do not claim a real robot success."""
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_codex_camera_observation_bindings import CAMERAS, TARGET, TASK, audit_bindings


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def case(tmp_path):
    attempt = tmp_path / "attempt"
    folder = attempt / "teacher/step-00000001-decision-0000"
    decision = folder / "decision/example-request"
    decision.mkdir(parents=True)
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir()
    rgb = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    Image.fromarray(rgb).save(reference_dir / "example.png")
    reference_sha = hashlib.sha256((reference_dir / "example.png").read_bytes()).hexdigest()
    reference = dict(kind="codex_collection_demonstrations_v1",
                     metadata=dict(task=TASK, source_kind="original_train"),
                     examples=[dict(image="example.png", image_sha256=reference_sha)])
    fingerprint = hashlib.sha256(json.dumps(reference, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    reference["fingerprint"] = fingerprint
    manifest = reference_dir / "manifest.json"
    dump(manifest, reference)
    predicate = dict(operation="grasped", targets=[TARGET])
    setup = dict(initial_predicate_value=False)
    result = dict(steps=2, task=TASK, predicate=predicate, success=False, metric_eligible=True,
                  setup_evidence=setup, terminated=False, truncated=False,
                  monitor=dict(predicate=predicate, setup_evidence=setup, satisfied=False,
                               debounce_steps=3, dwell=0,
                               initial_snapshot=dict(operation="grasped", targets=[dict(object=TARGET, grasped=False)]),
                               final_snapshot=dict(operation="grasped", targets=[dict(object=TARGET, grasped=False)], step=2)))
    completed = dict(result=result, training_eligible=False)
    dump(attempt / "completed.json", completed)
    state = np.arange(6, dtype=np.float32).reshape(2, 3)
    with h5py.File(attempt / "trajectory.hdf5", "w") as stream:
        stream.attrs.update(complete=True, control_steps=2, result=json.dumps(result),
                            metadata=json.dumps(dict(sample=dict(task=TASK, predicate=predicate))))
        stream["observation.state"] = state
        stream["action"] = np.zeros((2, 23), dtype=np.float32)
        for camera in CAMERAS:
            stream[camera] = np.stack([np.zeros_like(rgb), rgb])
        stream["next.terminated"] = stream["next.truncated"] = np.zeros((2, 1), dtype=bool)
    request = dict(step=1, proprio=state[1].tolist(), camera_order=["head", "left_wrist", "right_wrist"],
                   context_images=[dict(kind="successful_demonstration", current_observation=False)],
                   demonstration=dict(demonstration_fingerprint=fingerprint))
    images = []
    for i in range(4):
        image = decision / f"camera_{i}.png"
        Image.fromarray(rgb).save(image)
        images.append(dict(file=image.name, sha256=hashlib.sha256(image.read_bytes()).hexdigest()))
    envelope = dict(request=request, request_id="example-request", images=images)
    dump(folder / "request.json", request)
    dump(decision / "request.json", envelope)
    return dict(attempt=attempt, reference_manifest=manifest,
                expected_reference_sha256=reference_sha, expected_reference_fingerprint=fingerprint)


def change_request(case, change):
    outer = next((case["attempt"] / "teacher").glob("step-*/request.json"))
    inner = next((case["attempt"] / "teacher").glob("step-*/decision/*/request.json"))
    envelope = json.loads(inner.read_text())
    change(envelope)
    dump(inner, envelope)
    dump(outer, envelope["request"])


def change_result(case, change):
    path = case["attempt"] / "completed.json"
    completed = json.loads(path.read_text())
    change(completed)
    dump(path, completed)
    with h5py.File(case["attempt"] / "trajectory.hdf5", "r+") as stream:
        stream.attrs["result"] = json.dumps(completed["result"])


def make_success(completed):
    completed["training_eligible"] = True
    result = completed["result"]
    result["success"] = True
    result["monitor"].update(satisfied=True, dwell=3)
    result["monitor"]["final_snapshot"]["targets"][0]["grasped"] = True


def test_failure_binds_without_claiming_success(case):
    report = audit_bindings(**case)
    assert report["matched_requests"] == 1 and report["matched_current_images"] == 3
    assert report["result_success"] is report["full_success_acceptance"] is False
    with pytest.raises(ValueError, match="not a qualified success"):
        audit_bindings(**case, require_success=True)


@pytest.mark.parametrize("damage", ["proprio", "current_image", "reference_image"])
def test_wrong_but_self_consistent_observation_is_rejected(case, damage):
    def change(envelope):
        if damage == "proprio":
            envelope["request"]["proprio"][0] += 1
        else:
            index = 0 if damage == "current_image" else 3
            image = next((case["attempt"] / "teacher").glob(f"step-*/decision/*/camera_{index}.png"))
            Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(image)
            envelope["images"][index]["sha256"] = hashlib.sha256(image.read_bytes()).hexdigest()
    change_request(case, change)
    with pytest.raises(ValueError, match="proprio differs|differs from pre-action|frozen reference PNG"):
        audit_bindings(**case)


@pytest.mark.parametrize("damage", ["initial", "target", "final_step", "early_terminal"])
def test_invalid_native_or_terminal_boundary_is_rejected(case, damage):
    if damage == "early_terminal":
        with h5py.File(case["attempt"] / "trajectory.hdf5", "r+") as stream:
            stream["next.terminated"][0] = True
    else:
        def change(completed):
            result = completed["result"]
            if damage == "initial":
                result["setup_evidence"]["initial_predicate_value"] = True
            elif damage == "target":
                result["monitor"]["final_snapshot"]["targets"][0]["object"] = "other_object"
            else:
                result["monitor"]["final_snapshot"]["step"] = 1
        change_result(case, change)
    with pytest.raises(ValueError, match="Initial predicate|target identity|final executed|before the final"):
        audit_bindings(**case)


def test_post_final_observation_is_explicit_gap_and_never_valid_success(case):
    change_request(case, lambda envelope: envelope["request"].update(step=2))
    report = audit_bindings(**case)
    assert report["matched_requests"] == 0
    assert report["unavailable_observations"][0]["reason"] == "post_final_observation_not_recorded"
    change_result(case, make_success)
    for require_success in (False, True):
        with pytest.raises(ValueError, match="unverifiable post-final"):
            audit_bindings(**case, require_success=require_success)


def test_success_requires_all_bindings_and_dwell(case):
    change_result(case, make_success)
    assert audit_bindings(**case, require_success=True)["result_success"] is True
    change_result(case, lambda completed: completed["result"]["monitor"].update(dwell=2))
    with pytest.raises(ValueError, match="native grasp dwell"):
        audit_bindings(**case, require_success=True)
