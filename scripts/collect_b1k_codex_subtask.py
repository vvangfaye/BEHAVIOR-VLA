#!/usr/bin/env python3
"""Collect one new, fixed-label training-subtask attempt with a Codex coach.

Demonstration replay is initialization only. Every recorded action is newly
executed after restoration; a local environment predicate determines success.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import copy
import json
import math
from pathlib import Path
import sys
import time
from types import SimpleNamespace

from codex_subtask_collection import (KIND, MODE, FIXED_SUBTASK, NATIVE, HISTORY,
                                      file_sha256, load_plan, validate_plan, inspect_attempt)


class FixedSubtaskPolicy:
    """Mix short executed corrections with honestly attributed student actions."""
    def __init__(self, policy, subtask, correction):
        self.policy, self.subtask, self.correction = policy, subtask, correction
        self.prepared = None
        self.resume_reset = False
        self.action_source = None

    def __getattr__(self, name):
        return getattr(self.policy, name)

    def reset(self):
        self.prepared = None
        self.resume_reset = False
        self.action_source = None
        self.correction.cancel()
        return self.policy.reset()

    @property
    def current_subtask(self):
        return self.subtask if self.action_source == "teacher_cartesian_correction" else self.policy.current_subtask

    @property
    def current_subtask_source(self):
        return "provided" if self.action_source == "teacher_cartesian_correction" else self.policy.current_subtask_source

    def prepare_correction(self):
        # Called before Evaluator.step: a rejected IK must not advance its state.
        if self.prepared is not None:
            raise RuntimeError("Unconsumed correction action")
        if self.correction.remaining_steps:
            self.prepared = self.correction.next_action()

    def cancel_correction(self):
        self.prepared = None
        self.correction.cancel()

    def forward(self, obs):
        if self.prepared is not None:
            action, self.prepared = self.prepared, None
            self.action_source = "teacher_cartesian_correction"
            self.resume_reset = True
            return action
        if self.correction.remaining_steps:
            raise RuntimeError("Correction must be prepared before environment stepping")
        if self.resume_reset:
            self.policy.reset()  # Flush predicted chunks, never reset the scene.
            self.resume_reset = False
        self.action_source = "policy_fixed_subtask"
        action = self.policy.forward(obs={**obs, "subtask": self.subtask})
        actual = getattr(self.policy, "current_subtask", None)
        if (getattr(self.policy, "current_subtask_source", None) != "provided"
                or not isinstance(actual, str) or " ".join(actual.split()) != self.subtask):
            raise RuntimeError("Student did not acknowledge the fixed original subtask")
        return action


def make_monitor(evaluator, sample, cache, restoration):
    from subtask_completion import RoleCompletionMonitor
    source = restoration.get("recorded_native_source") or restoration.get("history_recovery_source")
    if source is None:
        raise ValueError("Local success needs verified original physical-state context")
    return RoleCompletionMonitor(evaluator, sample["predicate"], sample,
                                 source_context=(cache, source))


def _restore(evaluator, cache, sample, materialized, protocol):
    import eval_b1k_subtask_samples as runner
    restoration = runner.restore_materialized_sample_state(evaluator, cache, materialized)
    if protocol == HISTORY:
        from legacy_history_recovery import restored_context
        restoration.update(restored_context(evaluator, cache, sample, materialized))
    else:
        from recorded_native_state import decode_start, validate_runtime
        restoration["recorded_native_source"] = validate_runtime(decode_start(cache, sample), evaluator)
    return restoration


def collect_attempt(evaluator, teacher, plan, cache, materialized, output, *,
                    demonstration_manifest, correction_factory=None,
                    restore_fn=_restore, monitor_factory=make_monitor, feedback_factory=None):
    """One fresh attempt. Injectable seams permit CPU tests, never fake GPU proof."""
    from PIL import Image
    from subtask_trajectory import CAMERAS, TrajectoryWriter, capture_observation
    from codex_collection_actions import BoundedCartesianCorrection, RecoverableCorrectionError
    from codex_collection_demonstrations import demonstration_context
    from codex_collection_teacher import RejectedTeacherCorrection
    from collect_b1k_codex import time_limit

    validate_plan(plan)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sample, budget = plan["sample"], plan["budgets"]
    qualification = None
    if plan["schema_version"] >= 2:
        from codex_subtask_qualification import verify_qualification
        qualification = verify_qualification(output.parent / "qualification/qualification.json", plan)
    source = plan["source_plan"]
    protocol = source.get("source_protocol_by_sample", {}).get(sample["sample_id"], NATIVE)
    original_policy = evaluator.policy
    writer = None
    recording = False
    steps = decisions = 0
    terminated = truncated = success = terminal_credit = False
    stop_reason = "step_budget"
    history, previous_head = [], None
    last_outcome = None
    previous_correction = None
    active = {"decision_index": None, "request_id": None}
    counts = {"policy_fixed_subtask": 0, "teacher_cartesian_correction": 0}
    usage_rows = []
    started = time.monotonic()
    try:
        with time_limit(budget["max_seconds"]):
            restoration = restore_fn(evaluator, cache, sample, materialized, protocol)
            monitor = monitor_factory(evaluator, sample, cache, restoration)
            execution_protocol = plan.get('teacher_execution')
            correction_kwargs = {"execution_protocol": execution_protocol} if execution_protocol else {}
            correction = (correction_factory or BoundedCartesianCorrection)(evaluator.robot, **correction_kwargs)
            feedback = None
            if plan.get('teacher_feedback') == 'robot_gripper_execution_v1':
                from codex_gripper_feedback import GripperExecutionFeedback
                feedback = (feedback_factory or GripperExecutionFeedback.from_evaluator)(evaluator)
            policy = FixedSubtaskPolicy(original_policy, sample["subtask"], correction)
            evaluator.policy = policy
            evaluator.external_subtask = sample["subtask"]
            if hasattr(original_policy, "next_seed"):
                original_policy.next_seed = plan["policy_seed"]
            metadata = dict(kind=KIND, assisted_plan=plan, plan_fingerprint=plan["fingerprint"],
                checkpoint=plan["checkpoint_id"], checkpoint_identity=plan["checkpoint_identity"],
                sample=sample, conditioning="task_and_subtask", fps=int(evaluator.env.env_config["rendering_frequency"]),
                action_execution="teacher_mixed", initialization_replay_is_training_data=False,
                action_semantics="executed_R1Pro_control_23d", pairing="observation_before_executed_action")
            if cache.get('preparation_checkpoint'):
                metadata['preparation_checkpoint'] = cache['preparation_checkpoint']
            if qualification is not None:
                metadata["qualification"] = qualification
            if feedback is not None:
                metadata['teacher_auxiliary_observations'] = 'robot_gripper_execution_v1'
            if execution_protocol:
                metadata['teacher_execution_protocol'] = execution_protocol
            gt_selection = plan.get('teacher_scene_ground_truth')
            if gt_selection is not None:
                metadata['teacher_scene_ground_truth'] = gt_selection
                metadata['policy_observation_includes_ground_truth'] = False
            reference_receipt = None
            if plan.get('teacher_robot_guidance'):
                from codex_robot_guidance import validate_reference_receipt
                reference_path = output.parent / 'qualification/robot_reference.json'
                reference_receipt = json.loads(reference_path.read_text())
                validate_reference_receipt(reference_receipt, plan)
                metadata['teacher_robot_guidance'] = plan['teacher_robot_guidance']
                metadata['robot_reference_sha256'] = file_sha256(reference_path)
            writer = TrajectoryWriter(output / "trajectory.hdf5", metadata)
            next_teacher_step = 0
            action_path = output / "action_sources.jsonl"
            interruption_path = output / "correction_interruptions.jsonl"
            with action_path.open("x", buffering=1) as actions, \
                    (output / "correction_rejections.jsonl").open("x", buffering=1) as rejections, \
                    (interruption_path.open("x", buffering=1) if execution_protocol else nullcontext(None)) as interruptions, \
                    ((output / 'scene_ground_truth.jsonl').open('x', buffering=1)
                     if gt_selection is not None else nullcontext(None)) as gt_log:
                if monitor.metric_eligible:
                    evaluator.start_recording(str(output / "rollout.mp4"), rate=metadata["fps"])
                    recording = True
                else:
                    stop_reason = "setup_excluded"
                while monitor.metric_eligible and steps < budget["max_steps"]:
                    frame = capture_observation(evaluator.obs)
                    if steps >= next_teacher_step:
                        if decisions >= budget["max_decisions"]:
                            stop_reason = "teacher_decision_budget"
                            break
                        snapshot = output / "teacher" / f"step-{steps:08d}-decision-{decisions:04d}"
                        snapshot.mkdir(parents=True)
                        images = []
                        for index, key in enumerate(CAMERAS):
                            path = snapshot / f"camera-{index}.png"
                            Image.fromarray(frame[key]).save(path)
                            images.append(path)
                        request = dict(task=sample["task"], instance_id=sample["task_instance_id"],
                            sample_id=sample["sample_id"], attempt_id=plan["attempt_id"], step=steps,
                            decision_index=decisions, collection_mode=MODE,
                            fixed_subtask=sample["subtask"], target_subtask=sample["subtask"],
                            previous_subtask=sample["subtask"], proprio=frame["observation.state"].tolist(),
                            camera_order=["head", "left_wrist", "right_wrist"],
                            recent_decisions=history[-8:], context_images=[],
                            next_observation_after_steps=budget["teacher_interval"],
                            robot_control_state=correction.snapshot(),
                            last_correction_diagnostics=correction.last_diagnostics,
                            last_correction_outcome=last_outcome)
                        if plan.get("teacher_strategy"):
                            request["teacher_strategy"] = plan["teacher_strategy"]
                        if feedback is not None:
                            request['gripper_execution'] = feedback.snapshot(evaluator.robot)
                        if execution_protocol:
                            request['teacher_execution_protocol'] = execution_protocol
                        if gt_selection is not None:
                            from codex_scene_ground_truth import CONTRACT as GT_CONTRACT, capture as capture_gt
                            request['scene_ground_truth_protocol'] = GT_CONTRACT
                            request['scene_ground_truth'] = capture_gt(evaluator, plan, request)
                            gt_log.write(json.dumps(request['scene_ground_truth'], allow_nan=False) + '\n')
                        if reference_receipt is not None:
                            from codex_robot_guidance import CONTRACT, guidance_from_anchors, burst_motion
                            request['robot_guidance_protocol'] = CONTRACT
                            request['robot_reference_guidance'] = guidance_from_anchors(reference_receipt,
                                evaluator.robot, request['robot_control_state'],
                                receipt_sha256=metadata['robot_reference_sha256'])
                            request['previous_correction_motion'] = None
                            if previous_correction is not None:
                                request['previous_correction_motion'] = burst_motion(
                                    previous_correction['control_state'], request['robot_control_state'],
                                    previous_correction['correction'],
                                    start_step=previous_correction['start_step'], end_step=steps,
                                    decision_index=previous_correction['decision_index'],
                                    request_id=previous_correction['request_id'])
                                previous_correction = None
                        context, examples = demonstration_context(demonstration_manifest, sample["task"],
                                                                  sample["subtask"], limit=1)
                        request["demonstration"] = context
                        for path in examples:
                            images.append(path)
                            request["context_images"].append(dict(kind="successful_demonstration", current_observation=False))
                        if previous_head is not None:
                            images.append(previous_head[1])
                            request["context_images"].append(dict(kind="previous_head_observation",
                                step=previous_head[0], current_observation=False))
                        previous_head = (steps, images[0])
                        (snapshot / "request.json").write_text(json.dumps(request, indent=2) + "\n")
                        try:
                            decision = teacher.choose(request, images, snapshot / "decision")
                        except RejectedTeacherCorrection as error:
                            # A new observation/feedback request consumes a decision,
                            # never a physics step or the rejected evidence directory.
                            active = dict(decision_index=decisions, request_id=error.request_id)
                            usage_path = error.evidence_dir / "receipt.json"
                            usage = json.loads(usage_path.read_text()).get("usage")
                            usage_rows.append(dict(**active, usage=usage))
                            decisions += 1
                            last_outcome = dict(status="rejected", step=steps, **active, executed=False,
                                reason_code="invalid_teacher_correction", diagnostics=error.diagnostics)
                            rejections.write(json.dumps(last_outcome, allow_nan=False) + "\n")
                            history.append(dict(step=steps, **active, subtask=sample["subtask"],
                                action="rejected", reason="Correction contract rejected; no action executed",
                                last_rejection=last_outcome))
                            next_teacher_step = steps
                            continue
                        decisions += 1
                        usage_path = snapshot / "decision" / decision["request_id"] / "receipt.json"
                        usage = json.loads(usage_path.read_text()).get("usage") if usage_path.is_file() else None
                        usage_rows.append(dict(decision_index=decisions - 1, request_id=decision["request_id"], usage=usage))
                        if (decision.get("subtask") != sample["subtask"]
                                or decision.get("action") not in {"continue", "correct", "abort"}):
                            raise ValueError("Teacher must preserve the original fixed subtask")
                        active = dict(decision_index=decisions - 1, request_id=decision["request_id"])
                        history.append(dict(step=steps, **active, **{k: decision.get(k) for k in
                            ("subtask", "action", "correction", "reason")}))
                        next_teacher_step = steps + budget["teacher_interval"]
                        if decision["action"] == "abort":
                            stop_reason = "teacher_abort"
                            break
                        if decision["action"] == "correct":
                            correction.begin(decision["correction"], evaluator.robot_action if steps else None)
                            if reference_receipt is not None:
                                previous_correction = dict(control_state=copy.deepcopy(request['robot_control_state']),
                                    correction=copy.deepcopy(decision['correction']), start_step=steps, **active)
                            next_teacher_step = steps + correction.remaining_steps
                    try:
                        with time_limit(budget["step_timeout"]):
                            policy.prepare_correction()
                    except RecoverableCorrectionError as error:
                        policy.cancel_correction()
                        last_outcome = dict(status="rejected", step=steps, **active, executed=False,
                            reason_code=error.reason_code, diagnostics=error.diagnostics)
                        rejections.write(json.dumps(last_outcome, allow_nan=False) + "\n")
                        history[-1]["last_rejection"] = last_outcome
                        next_teacher_step = steps
                        continue
                    evaluator.external_subtask = sample["subtask"]
                    with time_limit(budget["step_timeout"]):
                        terminated, truncated = evaluator.step()
                    terminated, truncated = bool(terminated), bool(truncated)
                    writer.record(frame, evaluator.robot_action, terminated, truncated)
                    source_name = policy.action_source
                    if source_name not in counts:
                        raise RuntimeError("Unknown actually executed action source")
                    counts[source_name] += 1
                    actions.write(json.dumps(dict(step=steps, **active, source=source_name,
                        subtask=sample["subtask"], subtask_source="provided",
                        diagnostics=correction.last_diagnostics if source_name == "teacher_cartesian_correction" else None),
                        allow_nan=False) + "\n")
                    if source_name == "teacher_cartesian_correction":
                        last_outcome = dict(status="executed", step=steps, **active,
                                            executed=True, diagnostics=correction.last_diagnostics)
                    steps += 1
                    if feedback is not None:
                        feedback.observe_executed_action(evaluator.robot_action)
                    if execution_protocol and source_name == 'teacher_cartesian_correction':
                        contact_feedback = feedback.snapshot(evaluator.robot)
                        if correction.interrupt_on_contact(contact_feedback):
                            last_outcome.update(burst_interrupted=True,
                                interruption_reason='robot_finger_environment_contact')
                            history[-1]['last_outcome'] = copy.deepcopy(last_outcome)
                            interruptions.write(json.dumps(dict(**last_outcome,
                                next_teacher_step=steps, arm=history[-1]['correction']['arm'],
                                post_action_feedback=contact_feedback), allow_nan=False) + '\n')
                            next_teacher_step = steps
                    success = bool(monitor.refresh(steps))
                    if success:
                        stop_reason = "local_predicate_success"
                        break
                    if terminated or truncated:
                        if terminated and monitor.satisfied:
                            success = terminal_credit = True
                            stop_reason = "local_predicate_terminal_credit"
                        else:
                            stop_reason = "environment_terminal"
                        break
            if recording:
                evaluator.stop_recording()
                recording = False
            result = dict(kind=KIND, official_benchmark=False, plan_fingerprint=plan["fingerprint"],
                task=sample["task"], instance_id=sample["task_instance_id"], sample_id=sample["sample_id"],
                subtask=sample["subtask"], predicate=sample["predicate"], steps=steps,
                success=success, metric_eligible=bool(monitor.metric_eligible),
                setup_exclusion_reason=monitor.setup_exclusion_reason,
                setup_evidence=copy.deepcopy(monitor.setup_evidence), monitor=monitor.result(),
                restoration=restoration, terminal_credit=terminal_credit,
                terminated=terminated, truncated=truncated, task_success=bool(evaluator.env.task.success),
                stop_reason=stop_reason, teacher_decisions=decisions,
                actual_model_control_steps=counts["policy_fixed_subtask"],
                actual_teacher_control_steps=counts["teacher_cartesian_correction"],
                action_execution="teacher_mixed", conditioning=dict(source="provided", subtask_source="provided",
                    subtask_action_conditioning="task_and_subtask", official_benchmark=False, training_set_diagnostic=True),
                action_sources_sha256=file_sha256(action_path),
                video_sha256=file_sha256(output / "rollout.mp4") if monitor.metric_eligible else None,
                teacher_usage=usage_rows,
                wall_seconds=time.monotonic() - started)
            if execution_protocol:
                result['teacher_execution_protocol'] = execution_protocol
                result['correction_interruptions_sha256'] = file_sha256(interruption_path)
            if gt_selection is not None:
                result['teacher_scene_ground_truth'] = gt_selection
                result['policy_observation_includes_ground_truth'] = False
                result['scene_ground_truth_sha256'] = file_sha256(output / 'scene_ground_truth.jsonl')
            if reference_receipt is not None:
                result['teacher_robot_guidance'] = plan['teacher_robot_guidance']
                result['robot_reference_sha256'] = metadata['robot_reference_sha256']
            writer.finish(result)
            inspected = inspect_attempt(output, require_success=False) if steps else None
            receipt = dict(kind=KIND, plan_fingerprint=plan["fingerprint"], result=result,
                official_benchmark=False, training_eligible=bool(success and monitor.metric_eligible and steps),
                content_sha256=inspected["content_sha256"] if inspected else None,
                action_sources_sha256=result["action_sources_sha256"], video_sha256=result["video_sha256"])
            if receipt["training_eligible"]:
                inspect_attempt(output, require_success=True)
            (output / "completed.json").write_text(json.dumps(receipt, indent=2) + "\n")
            return receipt
    except BaseException as error:
        (output / "failed.json").write_text(json.dumps(dict(kind=KIND, plan_fingerprint=plan["fingerprint"],
            steps=steps, error_type=type(error).__name__, training_eligible=False,
            official_benchmark=False), indent=2) + "\n")
        raise
    finally:
        if recording:
            evaluator.stop_recording()
        if writer is not None:
            writer.close()
        evaluator.policy = original_policy


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-host", default="127.0.0.1")
    parser.add_argument("--policy-port", type=int, required=True)
    parser.add_argument("--policy-server-token", required=True)
    parser.add_argument("--teacher-codex", action="store_true", required=True)
    parser.add_argument("--codex-binary", default="codex")
    parser.add_argument("--codex-timeout", type=float, default=120)
    parser.add_argument("--codex-capacity-retries", type=int, choices=range(3), default=0)
    parser.add_argument("--reuse-qualification-root", type=Path)
    parser.add_argument("--persist-preparation", action="store_true")
    parser.add_argument("--resume-preparation", type=Path)
    parser.add_argument("--demonstration-manifest", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.policy_port <= 65535 or not args.policy_server_token.strip():
        parser.error("A valid policy port and private server identity token are required")
    if not math.isfinite(args.codex_timeout) or not 0 < args.codex_timeout <= 120:
        parser.error("Codex timeout must be finite, positive and at most 120 seconds")
    if args.reuse_qualification_root and not args.persist_preparation:
        parser.error("Qualification reuse requires a durable preparation checkpoint")
    if args.resume_preparation and not args.reuse_qualification_root:
        parser.error("Preparation resume requires the original verified qualification")
    return args


def main(argv=None):
    args = parse_args(argv)
    plan = load_plan(args.plan)
    sample, budget = plan["sample"], plan["budgets"]
    stage = "initialization"
    try:
        import eval_b1k_subtask_samples as runner
        from collect_b1k_codex import connect_collection_policy, collection_evaluator, time_limit
        from collect_b1k_subtasks import SeededClient
        from codex_subtask_teacher import CodexSubtaskTeacher
        teacher = CodexSubtaskTeacher(codex_binary=args.codex_binary, timeout_seconds=args.codex_timeout,
                capacity_retries=args.codex_capacity_retries,
                shared_home=(__import__('os').environ['CODEX_HOME']
                    if plan.get('teacher_access') == 'shared_home_per_call_v1' else None),
                **plan.get('teacher_configuration', {}))
        base = runner._load_base_module()
        base_args = base.parse_args(["--task-name", sample["task"], "--mode", "train",
            "--prompt-mode", "whole_task", "--num-rollouts", "1", "--max-steps", str(budget["max_steps"]),
            "--host", args.policy_host, "--port", str(args.policy_port),
            "--expected-server-token", args.policy_server_token,
            "--robot-config", str(args.robot_config), "--output-dir", str(args.output_dir),
            "--write-video", "--headless"])
        base_args.external_subtask = sample["subtask"]
        base.gm.HEADLESS = True
        base.seed_everything(plan["policy_seed"])
        with time_limit(budget["max_seconds"]):
            stage = "policy_handshake"
            connected, server_metadata = connect_collection_policy(SimpleNamespace(
                host=args.policy_host, port=args.policy_port, expected_server_token=args.policy_server_token,
                step_timeout=budget["step_timeout"], max_seconds=budget["max_seconds"]))
            if server_metadata.get("collection_seeding") is not True:
                raise ValueError("Student server must support explicit collection policy seeds")
            class ConnectedEvaluator(base.Evaluator):
                def load_policy(self):
                    return connected
            proxy = SimpleNamespace(**vars(base))
            proxy.Evaluator = ConnectedEvaluator
            stage = "source_validation"
            samples = [sample]
            if plan["schema_version"] >= 2:
                from codex_subtask_qualification import control_sample
                samples.append(control_sample(sample))
            cache = runner.load_episode_cache(runner.raw_hdf5_path(plan["raw_data_root"], sample), samples,
                                              allow_unverified_grasp_state=True)
            if plan.get('teacher_robot_guidance'):
                from codex_robot_guidance import prepare_reference_capture
                prepare_reference_capture(args.demonstration_manifest, plan, cache)
            source = plan["source_plan"]
            relative = f"task-{sample['task_index']:04d}/episode_{sample['raw_episode_id']:08d}.hdf5"
            if cache["sha256"] != source["raw_source_sha256"][relative]:
                raise ValueError("Original raw source changed")
            protocol = source.get("source_protocol_by_sample", {}).get(sample["sample_id"], NATIVE)
            legacy = cache["grasp_serialization_audit"]["legacy_missing_assisted_grasp_state"]
            if legacy != (protocol == HISTORY):
                raise ValueError("Source grasp serialization disagrees with frozen protocol")
            cfg = runner._build_evaluator_cfg(base, base_args, plan["policy_seed"])
            reused = None
            if args.reuse_qualification_root:
                from codex_preparation_checkpoint import reuse_qualification
                stage = "verify_reused_qualification"
                reused = reuse_qualification(args.reuse_qualification_root,
                    args.output_dir.parent / 'qualification', plan, Path(__file__).resolve().parent.parent)
                cache['preparation_runtime_signature'] = reused['verified_source_files_sha256']
                print('QUALIFICATION_REUSED=' + json.dumps(reused), flush=True)
            stage = "simulator_startup"
            def on_failure(error):
                args.output_dir.parent.mkdir(parents=True, exist_ok=True)
                path = args.output_dir.parent / ("collector_failure_" + plan["attempt_id"] + ".json")
                if not path.exists():
                    path.write_text(json.dumps(dict(kind=KIND, stage=stage, error_type=type(error).__name__,
                        plan_fingerprint=plan["fingerprint"], training_eligible=False, official_benchmark=False)) + "\n")
            with collection_evaluator(lambda: runner.create_sample_evaluator(proxy, cfg, cache, sample,
                    budget["max_steps"]), on_failure) as evaluator:
                from codex_jacobian_validation import RUNTIME_CONTRACT
                if plan.get('teacher_motion_validation') == RUNTIME_CONTRACT and reused is None:
                    from codex_jacobian_validation import run_initial
                    stage = 'runtime_jacobian_validation'
                    run_initial(evaluator, plan, cache,
                                args.output_dir.parent/'qualification/jacobian_validation.json')
                stage = "source_materialization"
                if args.resume_preparation:
                    from codex_preparation_checkpoint import load_preparation
                    states, restored_cache, proof = load_preparation(args.resume_preparation, plan, cache['sha256'],
                        runtime_signature=cache.get('preparation_runtime_signature', {}))
                    cache.update(restored_cache)
                    cache['preparation_checkpoint'] = proof
                    state_hash = cache['sample_evidence'][sample['sample_id']]['selected_state_sha256']
                    print('PREPARATION_RESUMED=' + json.dumps(proof), flush=True)
                elif protocol == HISTORY:
                    from legacy_history_recovery import enabled, materialize
                    if not enabled():
                        raise ValueError("Recorded-history source needs its explicit restoration protocol")
                    states = materialize(evaluator, cache, samples)
                    if states[sample["sample_id"]]["restoration"]["prefix_actions_sha256"] != source["prefix_actions_sha256"][sample["sample_id"]]:
                        raise ValueError("Original initialization action prefix changed")
                    state_hash = cache["sample_evidence"][sample["sample_id"]]["selected_state_sha256"]
                else:
                    from recorded_native_state import decode_start
                    state_hash = decode_start(cache, sample)["selected_state_sha256"]
                    states = runner.materialize_episode_start_states(evaluator, cache, samples)
                if state_hash != source["selected_state_sha256"][sample["sample_id"]]:
                    raise ValueError("Original selected source state changed")
                if args.persist_preparation and not args.resume_preparation:
                    from codex_preparation_checkpoint import save_preparation
                    stage = 'persist_preparation'
                    states, restored_cache, proof = save_preparation(
                        args.output_dir.parent / 'preparation', plan, states, cache, source_reuse=reused)
                    cache.update(restored_cache)
                    cache['preparation_checkpoint'] = proof
                    print('PREPARATION_READY=' + json.dumps(proof), flush=True)
                evaluator.policy = SeededClient(evaluator.policy)
                stage = "assisted_collection"
                if plan["schema_version"] >= 2 and reused is None:
                    from codex_subtask_qualification import qualify
                    stage = "paired_qualification"
                    qualification = qualify(base, evaluator, plan, cache, states,
                                            args.output_dir.parent / "qualification")
                    if not qualification["qualified"]:
                        print("SUBTASK_QUALIFICATION=" + json.dumps(qualification), flush=True)
                        return 0
                    if plan.get('teacher_robot_guidance'):
                        from codex_robot_guidance import build_reference_receipt
                        reference_path = args.output_dir.parent / 'qualification/robot_reference.json'
                        with reference_path.open('x') as stream:
                            json.dump(build_reference_receipt(cache, plan), stream, indent=2, allow_nan=False)
                            stream.write('\n')
                    stage = "assisted_collection"
                receipt = collect_attempt(evaluator, teacher, plan, cache, states[sample["sample_id"]],
                    args.output_dir, demonstration_manifest=args.demonstration_manifest)
                print("SUBTASK_COLLECTION_RECEIPT=" + json.dumps(receipt), flush=True)
        return 0
    except BaseException as error:
        # A startup error must not create the fresh attempt directory.
        args.output_dir.parent.mkdir(parents=True, exist_ok=True)
        path = args.output_dir.parent / ("collector_failure_" + plan["attempt_id"] + ".json")
        if not path.exists():
            path.write_text(json.dumps(dict(kind=KIND, stage=stage, error_type=type(error).__name__,
                plan_fingerprint=plan["fingerprint"], training_eligible=False, official_benchmark=False)) + "\n")
        raise


if __name__ == "__main__":
    sys.exit(main())
