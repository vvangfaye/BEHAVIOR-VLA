"""Independent contract for fixed-label, teacher-assisted training subtasks.

Original accepted-rate plans remain immutable. This contract separately binds
their source proof and the checkpoint actually used for assisted collection.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

KIND = "codex_assisted_training_subtask_v1"
MODE = "subtask_with_corrections"
TASK = "freeze_fruit"
FIXED_SUBTASK = "place on tupperware to countertop"
RADIO_TASK = "turning_on_radio"
RADIO_SUBTASK = "place on radio to coffee table koagbh"
CAMERA_TASK = "attach_a_camera_to_a_tripod"
CAMERA_SUBTASK = "pick up from digital camera to floors nlswvt"
SELECTED_SUBTASKS = {(TASK, FIXED_SUBTASK), (RADIO_TASK, RADIO_SUBTASK), (CAMERA_TASK, CAMERA_SUBTASK)}
NATIVE = "recorded_native_fields_v1"
HISTORY = "recorded_actions_before_start_native_grasp_recovery_v1"
SOURCES = {"policy_fixed_subtask", "teacher_cartesian_correction"}


def digest(value):
    # Match the immutable source-plan digest convention (ASCII JSON escaping).
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def file_sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _checkpoint(identity):
    if (not isinstance(identity, dict)
            or type(identity.get("checkpoint_step")) is not int
            or identity["checkpoint_step"] <= 0
            or not _sha(identity.get("checkpoint_manifest_sha256"))
            or not isinstance(identity.get("checkpoint_source"), str)
            or not identity["checkpoint_source"].strip()):
        raise ValueError("A verified collection checkpoint identity is required")


def freeze_plan(panel, source_plan, sample_id, checkpoint_identity, *, checkpoint_id,
                raw_data_root, attempt_id, policy_seed, max_steps=2048,
                max_decisions=32, teacher_interval=64, max_seconds=7200,
                step_timeout=120, teacher_feedback=None, teacher_execution=None,
                teacher_robot_guidance=None, teacher_scene_ground_truth=None,
                teacher_motion_validation=None, teacher_configuration=None,
                teacher_strategy=None, teacher_access=None):
    """Bind one original training start, not a replayed success or new scene."""
    from full_subtask_collection import validate as validate_full
    from accepted_subtask_collection import validate as validate_native

    samples = {s["sample_id"]: s for s in panel["samples"]}
    if source_plan.get("panel_fingerprint") != panel["fingerprint"]:
        raise ValueError("Source collection plan belongs to another training panel")
    if source_plan.get("fingerprint") != digest({k: v for k, v in source_plan.items() if k != "fingerprint"}):
        raise ValueError("Source collection plan changed")
    validator = validate_full if source_plan.get("require_recorded_source_state") else validate_native
    validator(source_plan, samples)
    if sample_id not in samples:
        raise ValueError("Unknown original training start")
    selected = samples[sample_id]
    version = 1 if selected["task"] == TASK else 2
    payload = dict(schema_version=version, kind=KIND, mode="train", official_benchmark=False,
        collection_mode=MODE, sample=samples[sample_id], source_plan=source_plan,
        panel_fingerprint=panel["fingerprint"], checkpoint_id=checkpoint_id,
        checkpoint_identity=checkpoint_identity,
        baseline_checkpoint_identity=source_plan["checkpoint_identity"],
        raw_data_root=str(raw_data_root), attempt_id=attempt_id, policy_seed=policy_seed,
        budgets=dict(max_steps=max_steps, max_decisions=max_decisions,
                     teacher_interval=teacher_interval, max_seconds=max_seconds,
                     step_timeout=step_timeout),
        action_semantics="executed_R1Pro_control_23d",
        pairing="observation_before_executed_action",
        conditioning="task_and_subtask", label_policy="fixed_original_annotation",
        initialization_replay_is_training_data=False)
    if teacher_feedback is not None:
        payload['teacher_feedback'] = teacher_feedback
    if teacher_execution is not None:
        payload['teacher_execution'] = teacher_execution
    if teacher_robot_guidance is not None:
        payload['teacher_robot_guidance'] = teacher_robot_guidance
    if teacher_scene_ground_truth is not None:
        payload['teacher_scene_ground_truth'] = teacher_scene_ground_truth
    if teacher_motion_validation is not None:
        payload['teacher_motion_validation'] = teacher_motion_validation
    if teacher_configuration is not None:
        payload['teacher_configuration'] = teacher_configuration
    if teacher_strategy is not None:
        payload["teacher_strategy"] = teacher_strategy
    if teacher_access is not None:
        payload["teacher_access"] = teacher_access
    # Detach all caller-owned dictionaries and reject nonfinite serialization.
    plan = json.loads(json.dumps(payload, allow_nan=False))
    plan["fingerprint"] = digest(plan)
    validate_plan(plan)
    return plan


def validate_plan(plan):
    if (not isinstance(plan, dict) or plan.get("kind") != KIND
            or plan.get("schema_version") not in (1, 2) or plan.get("mode") != "train"
            or plan.get("official_benchmark") is not False
            or plan.get("collection_mode") != MODE
            or plan.get("conditioning") != "task_and_subtask"
            or plan.get("label_policy") != "fixed_original_annotation"
            or plan.get("initialization_replay_is_training_data") is not False
            or plan.get("fingerprint") != digest({k: v for k, v in plan.items() if k != "fingerprint"})):
        raise ValueError("Invalid or changed assisted subtask contract")
    sample, source = plan["sample"], plan["source_plan"]
    sid = sample.get("sample_id")
    instance = sample.get("task_instance_id")
    pair = (sample.get("task"), sample.get("subtask"))
    if plan.get('teacher_access') not in (None, 'shared_home_per_call_v1'):
        raise ValueError('Unsupported teacher access protocol')
    if plan.get('teacher_strategy') is not None:
        if (plan['teacher_strategy'] not in ('gt_short_feedback_v1', 'reference_staged_v1')
                or pair != (CAMERA_TASK, CAMERA_SUBTASK)
                or not plan.get('teacher_scene_ground_truth')
                or not plan.get('teacher_robot_guidance')):
            raise ValueError('Method comparison requires the camera GT and reference contracts')
    if 'teacher_configuration' in plan:
        from codex_collection_teacher import validate_teacher_configuration
        validate_teacher_configuration(plan['teacher_configuration'])
    if plan.get('teacher_scene_ground_truth') is not None:
        from codex_scene_ground_truth import validate_selection as validate_gt_selection
        validate_gt_selection(plan['teacher_scene_ground_truth'], sample)
    if plan.get('teacher_motion_validation') is not None:
        from codex_jacobian_validation import CONTRACT, RUNTIME_CONTRACT
        if (plan['teacher_motion_validation'] not in (CONTRACT,RUNTIME_CONTRACT) or
                pair != (CAMERA_TASK, CAMERA_SUBTASK)):
            raise ValueError('Unsupported diagnostic Jacobian validation')
    if plan.get('teacher_robot_guidance') is not None:
        from codex_robot_guidance import validate_selection
        validate_selection(plan['teacher_robot_guidance'], sample,
            source.get('source_protocol_by_sample', {}).get(sid, NATIVE))
        if plan.get('teacher_feedback') != 'robot_gripper_execution_v1':
            raise ValueError('Robot reference guidance requires executed tactile feedback')
    if plan.get('teacher_execution') is not None:
        from codex_collection_actions import EXTENDED_BURST_PROTOCOL
        if (plan['teacher_execution'] != EXTENDED_BURST_PROTOCOL
                or pair != (CAMERA_TASK, CAMERA_SUBTASK)
                or plan.get('teacher_feedback') != 'robot_gripper_execution_v1'):
            raise ValueError('Extended execution requires the camera task and robot tactile feedback')
    if (plan.get('teacher_feedback') is not None
            and (plan['teacher_feedback'] != 'robot_gripper_execution_v1'
                 or pair != (CAMERA_TASK, CAMERA_SUBTASK))):
        raise ValueError('Unsupported task or protocol for auxiliary teacher feedback')
    if (pair not in SELECTED_SUBTASKS
            or (plan["schema_version"] == 1 and pair != (TASK, FIXED_SUBTASK))
            or type(instance) is not int or not 0 <= instance <= 300
            or not isinstance(sid, str) or not sid
            or [sample.get("task"), instance] in source.get("held_out_instances", [])):
        raise ValueError("Only selected non-held-out training subtasks are authorized")
    if (source.get("fingerprint") != digest({k: v for k, v in source.items() if k != "fingerprint"})
            or source.get("panel_fingerprint") != plan.get("panel_fingerprint")
            or source.get("mode") != "train"
            or source.get("official_benchmark") is not False
            or sid not in {t["sample_id"] for t in source.get("trials", [])}
            or sid in source.get("baseline_setup_excluded_sample_ids", [])
            or sid in source.get("baseline_source_excluded_sample_ids", [])):
        raise ValueError("Unverified or excluded source collection start")
    groups = [g for g in source.get("selected_groups", [])
              if (g.get("task"), g.get("subtask")) == pair]
    if (len(groups) != 1 or type(groups[0].get("eligible")) is not int
            or groups[0]["eligible"] <= 0 or groups[0].get("successes") != 0
            or groups[0].get("rate") != 0):
        raise ValueError("This pilot requires an accepted nonempty zero-success group")
    protocol = source.get("source_protocol_by_sample", {}).get(sid)
    if protocol is None and source.get("require_recorded_native_state"):
        protocol = NATIVE
    if protocol not in {NATIVE, HISTORY}:
        raise ValueError("Unsupported original physical-state restoration protocol")
    relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
    expected_predicate = dict(contract="annotation_roles_v1", kind="role_bound",
        operation="placed", references=["countertop"], state="OnTop", targets=["tupperware"])
    if pair == (RADIO_TASK, RADIO_SUBTASK):
        expected_predicate.update(targets=["radio_89"], references=["coffee_table_koagbh_0"])
    elif pair == (CAMERA_TASK, CAMERA_SUBTASK):
        expected_predicate = dict(contract="annotation_roles_v1", kind="role_bound",
                                  operation="grasped", targets=["digital_camera_87"])
    if (not _sha(source.get("raw_source_sha256", {}).get(relative))
            or not _sha(source.get("selected_state_sha256", {}).get(sid))
            or not isinstance(sample.get("predicate"), dict)
            or sample["predicate"] != expected_predicate
            or type(sample.get("start")) is not int or sample["start"] < 0
            or type(sample.get("end")) is not int or sample["end"] <= sample["start"]):
        raise ValueError("Original source hashes, interval and local predicate are required")
    if protocol == HISTORY and not _sha(source.get("prefix_actions_sha256", {}).get(sid)):
        raise ValueError("Recorded-history restoration requires a frozen action-prefix hash")
    _checkpoint(plan.get("checkpoint_identity"))
    if plan.get("baseline_checkpoint_identity") != source.get("checkpoint_identity"):
        raise ValueError("Original zero-success checkpoint identity changed")
    if (not isinstance(plan.get("checkpoint_id"), str) or not plan["checkpoint_id"].strip()
            or not isinstance(plan.get("attempt_id"), str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", plan["attempt_id"]) is None
            or not isinstance(plan.get("raw_data_root"), str) or not plan["raw_data_root"]
            or type(plan.get("policy_seed")) is not int or not 0 <= plan["policy_seed"] < 2**32):
        raise ValueError("Invalid actual checkpoint, attempt, raw root or policy seed")
    budgets = plan.get("budgets", {})
    # The floor-camera pilot needs enough decisions to finish a failed grasp
    # and retry. Keep the original cap for the other selected pilots.
    decision_limit = 64 if pair == (CAMERA_TASK, CAMERA_SUBTASK) else 32
    for key, maximum in (("max_steps", 2048), ("max_decisions", decision_limit),
                         ("teacher_interval", 2048), ("step_timeout", 120)):
        if type(budgets.get(key)) is not int or not 1 <= budgets[key] <= maximum:
            raise ValueError("Invalid finite assisted collection budget: " + key)
    seconds = budgets.get("max_seconds")
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or not 0 < seconds <= 7200:
        raise ValueError("Invalid assisted collection wall-clock budget")
    return plan


def load_plan(path):
    return validate_plan(json.loads(Path(path).read_text()))


def validate_restoration_proof(result, plan):
    """Keep before-action state evidence separate from the mixed action audit."""
    sample, source = plan["sample"], plan["source_plan"]
    sid = sample["sample_id"]
    protocol = source.get("source_protocol_by_sample", {}).get(sid, NATIVE)
    key = "recorded_native_source" if protocol == NATIVE else "history_recovery_source"
    proof = result.get("restoration", {}).get(key, {})
    relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
    if (proof.get("contract") != protocol or proof.get("sample_id") != sid
            or proof.get("source_start") != sample["start"]
            or proof.get("raw_sha256") != source["raw_source_sha256"][relative]
            or proof.get("selected_state_sha256") != source["selected_state_sha256"][sid]):
        raise ValueError("Mixed trajectory lacks its verified before-action source identity")
    if protocol == NATIVE:
        if (proof.get("recorded_native_fields_verified_before_policy") is not True
                or proof.get("full_robot_joints_match") is not True
                or proof.get("unrecorded_physics_caches_or_timers_verified") is not False):
            raise ValueError("Native restoration proof must retain its verified-field limits")
        saved, actual = proof.get("recorded_grasps", []), proof.get("actual_restored_grasps", [])
        by_arm = {row["arm"]: row for row in actual}
        if len(saved) != len(actual):
            raise ValueError("Restored native grasp count differs from the source")
        for row in saved:
            restored = by_arm.get({0: "left", 1: "right"}.get(row.get("arm_index")), {})
            if any(restored.get(k) != row.get(k) for k in ("target_object", "target_uuid", "joint_type",
                    "parent_frame_pos", "parent_frame_orn", "child_frame_pos", "child_frame_orn")):
                raise ValueError("Mixed trajectory changed its original native grasp")
    elif (proof.get("runtime_history_snapshot_loaded_before_policy") is not True
            or proof.get("prefix_actions_sha256") != source["prefix_actions_sha256"][sid]
            or proof.get("future_actions_used_to_prepare_start") is not False
            or proof.get("manual_grasp_establishment_used") is not False
            or proof.get("exact_original_grasp_anchor_bytes_verified") is not False):
        raise ValueError("Recorded-history restoration proof changed")
    binding = result.get("setup_evidence", {}).get("source_role_binding")
    if result.get("metric_eligible") is True:
        if (not isinstance(binding, dict) or binding.get("sample_id") != sid
                or binding.get("raw_sha256") != proof["raw_sha256"]
                or binding.get("source_state_sha256") != proof["selected_state_sha256"]
                or binding.get("desired_predicate_truth_used") is not False):
            raise ValueError("Local success lacks original annotation-to-object binding")
    return proof


def inspect_attempt(directory, *, require_success=True):
    """Verify mixed actions without calling the old model-only action audit."""
    from subtask_trajectory import inspect_trajectory
    directory = Path(directory)
    info = inspect_trajectory(directory / "trajectory.hdf5", require_success=require_success)
    metadata, result = info["metadata"], info["result"]
    plan = metadata["assisted_plan"]
    validate_plan(plan)
    if plan["schema_version"] >= 2:
        from codex_subtask_qualification import verify_qualification
        qualification = verify_qualification(directory.parent / "qualification/qualification.json", plan)
        if metadata.get("qualification", {}).get("sha256") != qualification["sha256"]:
            raise ValueError("Assisted attempt qualification differs")
    if (metadata.get("kind") != KIND or metadata.get("sample") != plan["sample"]
            or metadata.get("checkpoint") != plan["checkpoint_id"]
            or metadata.get("checkpoint_identity") != plan["checkpoint_identity"]
            or result.get("plan_fingerprint") != plan["fingerprint"]
            or result.get("official_benchmark") is not False
            or result.get("action_execution") != "teacher_mixed"
            or result.get("predicate") != plan["sample"]["predicate"]):
        raise ValueError("Mixed subtask trajectory provenance differs")
    validate_restoration_proof(result, plan)
    action_path = directory / "action_sources.jsonl"
    rows = [json.loads(line) for line in action_path.read_text().splitlines()]
    if (len(rows) != info["steps"] or [r.get("step") for r in rows] != list(range(info["steps"]))
            or any(r.get("source") not in SOURCES or r.get("subtask") != plan["sample"]["subtask"] for r in rows)
            or file_sha256(action_path) != result.get("action_sources_sha256")):
        raise ValueError("Mixed executed-action provenance is incomplete or changed")
    counts = {source: sum(r["source"] == source for r in rows) for source in SOURCES}
    if (result.get("actual_model_control_steps") != counts["policy_fixed_subtask"]
            or result.get("actual_teacher_control_steps") != counts["teacher_cartesian_correction"]
            or result.get("steps") != sum(counts.values())):
        raise ValueError("Teacher actions must not be reported as model actions")
    if plan.get('teacher_robot_guidance'):
        from codex_robot_guidance import validate_reference_receipt
        reference_path = directory.parent / 'qualification/robot_reference.json'
        validate_reference_receipt(json.loads(reference_path.read_text()), plan)
        if (metadata.get('teacher_robot_guidance') != plan['teacher_robot_guidance']
                or result.get('teacher_robot_guidance') != plan['teacher_robot_guidance']
                or metadata.get('robot_reference_sha256') != file_sha256(reference_path)
                or result.get('robot_reference_sha256') != file_sha256(reference_path)):
            raise ValueError('Recorded robot reference provenance differs')
    from codex_robot_guidance import audit_attempt_guidance
    audit_attempt_guidance(directory, plan, rows)
    from codex_scene_ground_truth import audit_attempt as audit_scene_ground_truth
    audit_scene_ground_truth(directory, plan, metadata, result)
    if plan.get('teacher_execution'):
        events_path = directory / 'correction_interruptions.jsonl'
        if (metadata.get('teacher_execution_protocol') != plan['teacher_execution']
                or result.get('teacher_execution_protocol') != plan['teacher_execution']
                or not events_path.is_file()
                or file_sha256(events_path) != result.get('correction_interruptions_sha256')):
            raise ValueError('Extended burst interruption provenance differs')
        seen = set()
        for line in events_path.read_text().splitlines():
            event = json.loads(line)
            step, arm = event.get('step'), event.get('arm')
            if (type(step) is not int or step in seen or not 0 <= step < len(rows)
                    or arm not in ('left', 'right') or event.get('executed') is not True
                    or event.get('burst_interrupted') is not True
                    or event.get('interruption_reason') != 'robot_finger_environment_contact'
                    or event.get('next_teacher_step') != step+1
                    or rows[step]['source'] != 'teacher_cartesian_correction'
                    or any(rows[step].get(k) != event.get(k) for k in ('request_id', 'decision_index'))):
                raise ValueError('Contact interruption lacks its actual executed action')
            contact = event.get('post_action_feedback', {})
            flags = contact.get('arms', {}).get(arm, {}).get('finger_environment_contact', [])
            if (contact.get('contract') != 'robot_gripper_execution_v1'
                    or contact.get('executed_control_steps') != step+1
                    or contact.get('contact_object_identity_disclosed') is not False
                    or contact.get('target_grasp_truth_disclosed') is not False
                    or len(flags) != 2 or any(type(flag) is not bool for flag in flags) or not any(flags)):
                raise ValueError('Contact interruption lacks robot-only post-action tactile feedback')
            seen.add(step)
    monitor = result.get("monitor", {})
    if require_success and (monitor.get("predicate") != plan["sample"]["predicate"]
            or monitor.get("debounce_steps") != 3 or monitor.get("satisfied") is not True
            or not (monitor.get("dwell", 0) >= 3
                    or (result.get("terminal_credit") is True and result.get("terminated") is True))):
        raise ValueError("Success lacks the original local completion predicate evidence")
    video = directory / "rollout.mp4"
    if result.get("metric_eligible") is True and (not video.is_file() or video.stat().st_size <= 0
            or file_sha256(video) != result.get("video_sha256")):
        raise ValueError("Mixed subtask trajectory lacks its verified video")
    return {**info, "action_source_counts": counts}
