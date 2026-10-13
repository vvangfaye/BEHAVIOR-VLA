"""Robot-only reference poses and measured correction motion for a teacher.

Reference poses are observed during the existing original-history replay. They
are historical robot poses, never queried target-object poses or commands to
replay. All transforms use named current-runtime frames, not old proprio indices.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import numpy as np

CONTRACT = "recorded_robot_reference_and_motion_v1"
REFERENCE_CONTRACT = "recorded_robot_world_anchors_v1"
MOTION_CONTRACT = "executed_robot_eef_burst_motion_v1"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def prepare_reference_capture(manifest, plan, cache):
    from codex_collection_demonstrations import demonstration_context
    selected = plan["teacher_robot_guidance"]
    sample = plan["sample"]
    context, _ = demonstration_context(manifest, sample["task"], sample["subtask"], limit=1)
    data = json.loads(Path(manifest).read_text())
    frames = sorted({o["frame"] for e in context["examples"] for o in e["observations"]})
    if (context["demonstration_fingerprint"] != selected["demonstration_fingerprint"]
            or frames != selected["frames"]
            or data["metadata"].get("source_raw_sha256") != cache["sha256"]
            or data["metadata"].get("episode_index") != sample["episode_index"]
            or data["metadata"].get("instance_id") != sample["task_instance_id"]):
        raise ValueError("Robot anchors must come from the exact bound visual demonstration")
    cache["robot_reference_capture_frames"] = frames


def build_reference_receipt(cache, plan):
    selection = plan["teacher_robot_guidance"]
    frames = selection["frames"]
    anchors = cache.get("robot_reference_anchors", {})
    if set(anchors) != set(frames):
        raise ValueError("Original replay did not capture every requested robot reference")
    receipt = dict(contract=REFERENCE_CONTRACT, plan_fingerprint=plan["fingerprint"],
        source_raw_sha256=cache["sha256"], sample_id=plan["sample"]["sample_id"],
        demonstration_fingerprint=selection["demonstration_fingerprint"],
        capture_protocol="recorded_actions_before_start_native_grasp_recovery_v1",
        anchors=[anchors[f] for f in frames], training_eligible=False,
        object_ground_truth_disclosed=False)
    receipt["fingerprint"] = digest(receipt)
    validate_reference_receipt(receipt, plan)
    return receipt


def validate_reference_receipt(receipt, plan):
    keys = {"contract", "plan_fingerprint", "source_raw_sha256", "sample_id",
            "demonstration_fingerprint", "capture_protocol", "anchors", "training_eligible",
            "object_ground_truth_disclosed", "fingerprint"}
    selection, sample = plan["teacher_robot_guidance"], plan["sample"]
    relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
    if (set(receipt) != keys or receipt["contract"] != REFERENCE_CONTRACT
            or receipt["fingerprint"] != digest({k:v for k,v in receipt.items() if k != "fingerprint"})
            or receipt["plan_fingerprint"] != plan["fingerprint"]
            or receipt["source_raw_sha256"] != plan["source_plan"]["raw_source_sha256"][relative]
            or receipt["sample_id"] != sample["sample_id"]
            or receipt["demonstration_fingerprint"] != selection["demonstration_fingerprint"]
            or receipt["capture_protocol"] != "recorded_actions_before_start_native_grasp_recovery_v1"
            or receipt["training_eligible"] is not False
            or receipt["object_ground_truth_disclosed"] is not False
            or [a.get("frame") for a in receipt["anchors"]] != selection["frames"]):
        raise ValueError("Robot reference receipt differs from its frozen source")
    for anchor in receipt["anchors"]:
        if (set(anchor) != {"frame", "coordinate_frame", "base_footprint", "arms", "recorded_state_sha256"}
                or anchor["coordinate_frame"] != "world" or set(anchor["arms"]) != {"left", "right"}
                or len(anchor["recorded_state_sha256"]) != 64):
            raise ValueError("Robot reference contains unknown observations")
        for p in [anchor["base_footprint"], *anchor["arms"].values()]:
            if set(p) != {"position_m", "quaternion_xyzw"}:
                raise ValueError("Only named robot poses may be in a reference")
            pose(p["position_m"], p["quaternion_xyzw"])


def vector(value, size):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError("Robot guidance requires a complete finite vector")
    return result


def quaternion(value):
    q = vector(value, 4)
    norm = np.linalg.norm(q)
    if not math.isclose(float(norm), 1., abs_tol=1e-3):
        raise ValueError("Robot guidance quaternion must be normalized")
    return q / norm


def conjugate(q):
    return np.r_[-q[:3], q[3]]


def multiply(a, b):
    return np.r_[a[3]*b[:3] + b[3]*a[:3] + np.cross(a[:3], b[:3]),
                 a[3]*b[3] - np.dot(a[:3], b[:3])]


def rotation(q):
    x, y, z, w = quaternion(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rotation_error(goal, current):
    """Shortest left-multiplying axis-angle error, in the common base frame."""
    error = multiply(quaternion(goal), conjugate(quaternion(current)))
    if error[3] < 0:
        error = -error
    size = float(np.linalg.norm(error[:3]))
    if size < 1e-10:
        return (2*error[:3]).tolist()
    return (error[:3] * (2*math.atan2(size, float(error[3])) / size)).tolist()


def pose(position, orientation):
    return dict(position_m=vector(position, 3).tolist(),
                quaternion_xyzw=quaternion(orientation).tolist())


def capture_robot_anchor(robot, frame):
    """Read robot links only; no object states or scene state are exposed."""
    if type(frame) is not int or frame < 0:
        raise ValueError("Original reference frame must be a nonnegative integer")
    base = pose(*robot.get_position_orientation())
    arms = {arm: pose(*robot.eef_links[arm].get_position_orientation())
            for arm in ("left", "right")}
    return dict(frame=frame, coordinate_frame="world", base_footprint=base, arms=arms)


def validate_selection(selection, sample, history_protocol):
    if not isinstance(selection, dict) or set(selection) != {
            "contract", "frames", "demonstration_fingerprint"}:
        raise ValueError("Robot guidance needs an exact source selection")
    frames = selection["frames"]
    fingerprint = selection["demonstration_fingerprint"]
    if (selection["contract"] != CONTRACT
            or sample["task"] != "attach_a_camera_to_a_tripod"
            or history_protocol != "recorded_actions_before_start_native_grasp_recovery_v1"
            or not isinstance(frames, list) or not 1 <= len(frames) <= 8
            or any(type(f) is not int or not sample["start"] <= f < sample["end"] for f in frames)
            or frames != sorted(set(frames))
            or not isinstance(fingerprint, str) or len(fingerprint) != 64
            or any(c not in "0123456789abcdef" for c in fingerprint)):
        raise ValueError("Invalid training robot reference selection")


def guidance_from_anchors(receipt, robot, control_state, *, receipt_sha256):
    """Transform original world EEF poses into the *current* base footprint."""
    if receipt["contract"] != REFERENCE_CONTRACT:
        raise ValueError("Unexpected original robot anchor contract")
    if control_state["frame"] != "robot_base_footprint":
        raise ValueError("Reference guidance requires the actual base footprint frame")
    base = pose(*robot.get_position_orientation())
    bp, bq = vector(base["position_m"], 3), quaternion(base["quaternion_xyzw"])
    inverse = rotation(bq).T
    result = []
    for anchor in receipt["anchors"]:
        arms = {}
        for arm in ("left", "right"):
            reference = anchor["arms"][arm]
            position = inverse @ (vector(reference["position_m"], 3)-bp)
            orientation = multiply(conjugate(bq), quaternion(reference["quaternion_xyzw"]))
            live = control_state["arms"][arm]
            # Independently check that the live link and correction adapter use
            # the advertised same frame. Fail before the teacher if they diverge.
            wp, wq = robot.eef_links[arm].get_position_orientation()
            if (np.linalg.norm(inverse @ (vector(wp, 3)-bp)-vector(live["eef_position_m"], 3)) > 1e-3
                    or np.linalg.norm(rotation_error(multiply(conjugate(bq), quaternion(wq)),
                                                       live["eef_quaternion_xyzw"])) > 1e-3):
                raise ValueError("Live link world pose disagrees with correction frame")
            arms[arm] = dict(reference_position_m=position.tolist(),
                reference_quaternion_xyzw=quaternion(orientation).tolist(),
                position_error_m=(position-vector(live["eef_position_m"], 3)).tolist(),
                rotation_error_axis_angle_rad=rotation_error(orientation, live["eef_quaternion_xyzw"]))
        result.append(dict(source_frame=anchor["frame"], arms=arms))
    return dict(contract=REFERENCE_CONTRACT, source_receipt_sha256=receipt_sha256,
        frame="robot_base_footprint", current_base_world_pose=base, references=result,
        historical_robot_reference=True, object_ground_truth_disclosed=False,
        interpretation="Historical robot EEF poses transformed into the current base frame. "
        "Errors are reference minus current robot pose, not observed object locations. "
        "Objects may have moved; use RGB and tactile feedback. These are not executable commands.")


def burst_motion(before, after, correction, *, start_step, end_step, decision_index, request_id):
    count = end_step-start_step
    if (type(start_step) is not int or type(end_step) is not int
            or not 0 <= count <= correction["steps"]):
        raise ValueError("Burst feedback must count actually executed correction controls")
    if before["frame"] != "robot_base_footprint" or after["frame"] != before["frame"]:
        raise ValueError("Burst feedback requires matching named robot frames")
    arm = correction["arm"]
    begin, end = before["arms"][arm], after["arms"][arm]
    wanted = vector(correction["delta_position_m"], 3)*count
    reached = vector(end["eef_position_m"], 3)-vector(begin["eef_position_m"], 3)
    norm2 = float(np.dot(wanted, wanted))
    return dict(contract=MOTION_CONTRACT, decision_index=decision_index, request_id=request_id,
        start_step=start_step, end_step=end_step, arm=arm, frame=before["frame"],
        requested_controls=correction["steps"], executed_controls=count,
        start_eef_pose=copy.deepcopy(begin), end_eef_pose=copy.deepcopy(end),
        requested_translation_m=wanted.tolist(), achieved_translation_m=reached.tolist(),
        achieved_rotation_axis_angle_rad=rotation_error(end["eef_quaternion_xyzw"],
                                                        begin["eef_quaternion_xyzw"]),
        translation_projection_ratio=float(np.dot(wanted, reached))/norm2 if norm2 else None,
        object_ground_truth_disclosed=False,
        interpretation="Measured robot self motion over actual contiguous teacher controls. "
        "Executed does not mean the commanded distance was reached. Includes dynamics, "
        "coupling and possible base drift; gives no object identity or grasp truth.")


def motion_matches(actual, expected):
    """Allow only roundoff in derived floats, never identities or recorded poses.

    BLAS dot products on the cluster and the audit host can differ by one ULP.
    Frozen request bytes/hashes are still checked separately without tolerance.
    """
    if not isinstance(actual, dict) or set(actual) != set(expected):
        return False
    derived = {'requested_translation_m', 'achieved_translation_m',
               'achieved_rotation_axis_angle_rad', 'translation_projection_ratio'}
    for key, wanted in expected.items():
        value = actual[key]
        if key not in derived:
            if type(value) is not type(wanted) or value != wanted:
                return False
        elif wanted is None:
            if value is not None:
                return False
        elif isinstance(wanted, list):
            if (not isinstance(value, list) or len(value) != len(wanted)
                    or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
                    or not np.allclose(value, wanted, atol=1e-12, rtol=1e-12)):
                return False
        elif (type(value) not in (int, float) or not math.isfinite(value)
              or not math.isclose(value, wanted, abs_tol=1e-12, rel_tol=1e-12)):
            return False
    return True


def validate_teacher_guidance(request):
    fields = {"robot_guidance_protocol", "robot_reference_guidance", "previous_correction_motion"}
    if not fields.intersection(request):
        return
    if (not fields <= request.keys() or request["robot_guidance_protocol"] != CONTRACT
            or request["task"] != "attach_a_camera_to_a_tripod"
            or request.get("gripper_execution", {}).get("contract") != "robot_gripper_execution_v1"):
        raise ValueError("Incomplete or unsupported robot guidance")
    guidance, current = request["robot_reference_guidance"], request["robot_control_state"]
    expected = {"contract", "source_receipt_sha256", "frame", "current_base_world_pose",
                "references", "historical_robot_reference", "object_ground_truth_disclosed", "interpretation"}
    if (set(guidance) != expected or guidance["contract"] != REFERENCE_CONTRACT
            or guidance["frame"] != "robot_base_footprint" or current["frame"] != guidance["frame"]
            or guidance["historical_robot_reference"] is not True
            or guidance["object_ground_truth_disclosed"] is not False
            or len(guidance["source_receipt_sha256"]) != 64
            or not 1 <= len(guidance["references"]) <= 8):
        raise ValueError("Invalid historical robot reference guidance")
    base = guidance["current_base_world_pose"]
    if set(base) != {"position_m", "quaternion_xyzw"}:
        raise ValueError("Only a robot base pose may be disclosed")
    pose(base["position_m"], base["quaternion_xyzw"])
    for row in guidance["references"]:
        if (set(row) != {"source_frame", "arms"} or type(row["source_frame"]) is not int
                or row["source_frame"] < 0 or set(row["arms"]) != {"left", "right"}):
            raise ValueError("Invalid robot reference frame")
        for arm, value in row["arms"].items():
            if set(value) != {"reference_position_m", "reference_quaternion_xyzw",
                              "position_error_m", "rotation_error_axis_angle_rad"}:
                raise ValueError("Only robot EEF reference errors may be disclosed")
            p = vector(value["reference_position_m"], 3)
            q = quaternion(value["reference_quaternion_xyzw"])
            actual = current["arms"][arm]
            if (not np.allclose(vector(value["position_error_m"], 3),
                                p-vector(actual["eef_position_m"], 3), atol=1e-8, rtol=0)
                    or not np.allclose(vector(value["rotation_error_axis_angle_rad"], 3),
                                       rotation_error(q, actual["eef_quaternion_xyzw"]), atol=1e-8, rtol=0)):
                raise ValueError("Reference error does not describe the current robot pose")
    motion = request["previous_correction_motion"]
    if motion is None:
        return
    previous = request["recent_decisions"][-1] if request["recent_decisions"] else {}
    if (previous.get("action") != "correct" or motion.get("contract") != MOTION_CONTRACT
            or motion.get("request_id") != previous.get("request_id")
            or motion.get("decision_index") != previous.get("decision_index")
            or motion.get("start_step") != previous.get("step") or motion.get("end_step") != request["step"]):
        raise ValueError("Motion feedback is not the previous executed correction")
    arm = previous["correction"]["arm"]
    before_arm = motion["start_eef_pose"]
    if set(before_arm) != set(current["arms"][arm]):
        raise ValueError("Unknown robot motion observations")
    for name, value in before_arm.items():
        if name == "eef_link":
            if value != current["arms"][arm][name]:
                raise ValueError("Motion feedback changed the robot link")
        else:
            vector(value, len(current["arms"][arm][name]))
    before = dict(frame=current["frame"], arms={arm:before_arm})
    expected_motion = burst_motion(before, current, previous["correction"],
        start_step=previous["step"], end_step=request["step"],
        decision_index=previous["decision_index"], request_id=previous["request_id"])
    if not motion_matches(motion, expected_motion):
        raise ValueError("Motion feedback differs from measured robot self poses")


def audit_attempt_guidance(directory, plan, action_rows):
    """Bind teacher reference/motion fields to immutable original receipts and controls."""
    directory = Path(directory)
    selection = plan.get('teacher_robot_guidance')
    reference = None
    if selection:
        path = directory.parent/'qualification/robot_reference.json'
        reference = json.loads(path.read_text())
        validate_reference_receipt(reference, plan)
        reference_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    previous = None
    for folder in sorted((directory/'teacher').glob('step-*')):
        request = json.loads((folder/'request.json').read_text())
        present = request.get('robot_guidance_protocol')
        if not selection:
            if present or 'robot_reference_guidance' in request or 'previous_correction_motion' in request:
                raise ValueError('Unplanned robot guidance in a teacher request')
            continue
        validate_teacher_guidance(request)
        if present != CONTRACT:
            raise ValueError('Teacher request lost its planned robot guidance')
        g = request['robot_reference_guidance']
        if (g['source_receipt_sha256'] != reference_sha
                or request.get('demonstration', {}).get('demonstration_fingerprint') != selection['demonstration_fingerprint']
                or [row['source_frame'] for row in g['references']] != selection['frames']):
            raise ValueError('Teacher robot references differ from the frozen source selection')
        base = g['current_base_world_pose']
        bp, bq = vector(base['position_m'],3), quaternion(base['quaternion_xyzw'])
        for row, anchor in zip(g['references'], reference['anchors']):
            for arm in ('left','right'):
                actual, source = row['arms'][arm], anchor['arms'][arm]
                expected_p = rotation(bq).T @ (vector(source['position_m'],3)-bp)
                expected_q = multiply(conjugate(bq), quaternion(source['quaternion_xyzw']))
                if (not np.allclose(actual['reference_position_m'],expected_p,atol=1e-8,rtol=0)
                        or np.linalg.norm(rotation_error(actual['reference_quaternion_xyzw'],expected_q))>1e-8):
                    raise ValueError('Teacher reference was not transformed from the recorded robot anchor')
        motion = request['previous_correction_motion']
        if previous and previous[1].get('action') == 'correct':
            before, decision = previous
            emitted = action_rows[before['step']:request['step']]
            if not all(row['source']=='teacher_cartesian_correction'
                       and row['request_id']==decision['request_id']
                       and row['decision_index']==before['decision_index'] for row in emitted):
                raise ValueError('Motion feedback includes controls outside the preceding correction')
            expected = burst_motion(before['robot_control_state'],request['robot_control_state'],
                decision['correction'],start_step=before['step'],end_step=request['step'],
                decision_index=before['decision_index'],request_id=decision['request_id'])
            if not motion_matches(motion, expected):
                raise ValueError('Teacher motion feedback differs from the preceding request and emitted controls')
        elif motion is not None:
            raise ValueError('Teacher motion feedback has no preceding correction')
        responses = list((folder/'decision').glob('*/response.json'))
        if len(responses)>1:
            raise ValueError('Ambiguous teacher correction evidence')
        decision = json.loads(responses[0].read_text()) if responses else {'action':'rejected'}
        previous = request, decision
