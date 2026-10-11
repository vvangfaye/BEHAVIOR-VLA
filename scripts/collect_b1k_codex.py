#!/usr/bin/env python3
"""Collect ONE whole-task attempt on 311–320 with Codex subtask guidance.

This is training-data collection, never a challenge evaluation entry point.
The policy server and simulator must already be provisioned separately.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
import json
import math
from pathlib import Path
import signal
import sys
import time

COLLECTION_MODES = ("teacher_conditioned", "native_with_corrections")


def task_prompt(task):
    path = Path(__file__).resolve().parents[1] / "openpi/src/openpi/configs/tasks/b1k.py"
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "TASKS" for t in node.targets):
            return ast.literal_eval(node.value)[task]
    raise ValueError("Canonical B1K task prompts not found")


class TeacherPolicy:
    """Validate conditioning BEFORE the evaluator executes the returned action."""
    def __init__(self, policy, prompt, collection_mode="teacher_conditioned"):
        if collection_mode not in COLLECTION_MODES:
            raise ValueError("Unknown collection mode")
        if getattr(policy, "_action_chunk_size", 0) != 0:
            raise ValueError("Teacher collection requires client action_chunk_size=0")
        self.policy = policy
        self.prompt = prompt
        self.collection_mode = collection_mode
        self.subtask = None
        self.correction = None
        self._resume_reset = False
        self._prepared_correction_action = None
        self.action_source = None

    def __getattr__(self, name):
        return getattr(self.policy, name)

    @property
    def current_subtask(self):
        return self.subtask if self.action_source == "teacher_cartesian_correction" else self.policy.current_subtask

    @property
    def current_subtask_source(self):
        return "provided" if self.action_source == "teacher_cartesian_correction" else self.policy.current_subtask_source

    def reset(self):
        self.subtask = None
        self._resume_reset = False
        self._prepared_correction_action = None
        self.action_source = None
        return self.policy.reset()

    def prepare_correction(self):
        """Compute before evaluator.step; forward must never solve IK inside env stepping."""
        if self._prepared_correction_action is not None:
            raise RuntimeError("A prepared correction must be consumed exactly once")
        if self.correction is not None and self.correction.remaining_steps:
            self._prepared_correction_action = self.correction.next_action()

    def cancel_correction(self):
        self._prepared_correction_action = None
        self.correction.cancel()

    def forward(self, obs):
        if self.collection_mode == "teacher_conditioned" and not self.subtask:
            raise ValueError("A teacher decision is required before executing actions")
        if self._prepared_correction_action is not None:
            self.action_source = "teacher_cartesian_correction"
            self._resume_reset = True
            action, self._prepared_correction_action = self._prepared_correction_action, None
            return action
        if self.correction is not None and self.correction.remaining_steps:
            raise RuntimeError("Correction action must be prepared before evaluator.step")
        if self._resume_reset:
            # A correction moved the robot outside the last predicted action chunk.
            # Flush only the policy cache, never the simulator or task state.
            self.policy.reset()
            self._resume_reset = False
        if self.collection_mode == "native_with_corrections":
            # Match the native evaluator input. Neither a teacher label nor an
            # explicit refresh flag may replace the server's subtask/chunk state.
            native_obs = {**obs, "prompt": self.prompt}
            native_obs.pop("subtask", None)
            native_obs.pop("need_new_action", None)
            self.action_source = "policy_native_subtask"
            action = self.policy.forward(obs=native_obs)
            actual = getattr(self.policy, "current_subtask", None)
            if (getattr(self.policy, "current_subtask_source", None) not in {"generated", "cached"}
                    or not isinstance(actual, str) or not actual.strip()
                    or any(ord(char) < 32 for char in " ".join(actual.split()))):
                raise RuntimeError("Policy did not report an actual native subtask; action not executed")
            return action
        self.action_source = "policy_conditioned_on_teacher_subtask"
        action = self.policy.forward(obs={**obs, "prompt": self.prompt,
                                          "subtask": self.subtask, "need_new_action": True})
        actual = getattr(self.policy, "current_subtask", None)
        if (getattr(self.policy, "current_subtask_source", None) != "provided"
                or not isinstance(actual, str) or " ".join(actual.split()) != self.subtask):
            raise RuntimeError("Policy did not acknowledge the exact teacher subtask; action not executed")
        return action


@contextmanager
def time_limit(seconds):
    """Interrupt a blocked socket call on the simulator's POSIX main thread.

The containing GPU job must also have a platform wall-time limit: Python signal
handling alone cannot stop an uninterruptible native simulator/driver call.
"""
    if seconds <= 0:
        raise ValueError("Time budget must be positive")
    previous_handler = signal.getsignal(signal.SIGALRM)
    started = time.monotonic()
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    def expired(*_):
        raise TimeoutError("Collection wall-time or policy step budget exceeded")
    signal.signal(signal.SIGALRM, expired)
    duration = min(seconds, previous_timer[0]) if previous_timer[0] > 0 else seconds
    signal.setitimer(signal.ITIMER_REAL, duration)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            remaining = previous_timer[0] - (time.monotonic() - started)
            if remaining > 0:
                signal.setitimer(signal.ITIMER_REAL, remaining, previous_timer[1])


def collect_attempt(evaluator, teacher, output, metadata, *, max_steps, teacher_interval,
                    max_decisions, max_seconds=1800, step_timeout=120,
                    correction_factory=None, demonstration_manifest=None,
                    collection_mode="teacher_conditioned"):
    with time_limit(max_seconds):
        return _collect_attempt(evaluator, teacher, output, metadata, max_steps=max_steps,
                                teacher_interval=teacher_interval, max_decisions=max_decisions,
                                step_timeout=step_timeout, max_seconds=max_seconds,
                                correction_factory=correction_factory,
                                demonstration_manifest=demonstration_manifest,
                                collection_mode=collection_mode)


def _collect_attempt(evaluator, teacher, output, metadata, *, max_steps, teacher_interval,
                     max_decisions, step_timeout, max_seconds, correction_factory=None,
                     demonstration_manifest=None, collection_mode="teacher_conditioned"):
    from PIL import Image
    from subtask_trajectory import CAMERAS, capture_observation
    from codex_collection_trajectory import CodexCollectionTrajectory, inspect_codex_collection_trajectory
    from codex_collection_actions import RecoverableCorrectionError

    if type(metadata["instance_id"]) is not int or metadata["instance_id"] not in range(311, 321):
        raise ValueError("Only 311–320 may enter teacher-assisted collection")
    if collection_mode not in COLLECTION_MODES or metadata.get("collection_mode", collection_mode) != collection_mode:
        raise ValueError("Unknown or conflicting collection mode")
    if any(type(v) is not int or v < 1 for v in (max_steps, teacher_interval, max_decisions)):
        raise ValueError("All collection budgets must be positive integers")
    if getattr(evaluator, "prompt_mode", None) != "whole_task" or getattr(evaluator, "external_subtask_selector", None):
        raise ValueError("Use whole_task without another external selector")
    if getattr(evaluator, "external_subtask", None):
        raise ValueError("Do not combine Codex collection with a fixed external subtask")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    prompt = task_prompt(metadata["task"])
    metadata = {**metadata, "task_prompt": prompt, "teacher": "codex_chatgpt_login",
                "official_benchmark": False, "teacher_interval": teacher_interval,
                "max_decisions": max_decisions, "max_steps": max_steps,
                "max_seconds": max_seconds, "step_timeout": step_timeout,
                "collection_mode": collection_mode}
    (output / "collection_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    original_policy = evaluator.policy
    policy = TeacherPolicy(original_policy, prompt, collection_mode)
    evaluator.policy = policy
    writer = None
    action_log = None
    rejection_log = None
    recording = False
    steps = decisions = 0
    terminated = truncated = False
    stop_reason = "step_budget"
    history = []
    native = collection_mode == "native_with_corrections"
    next_teacher_step = teacher_interval if native else 0
    last_executed_subtask = None
    previous_head = None
    last_correction_outcome = None
    active_decision = {"decision_index": None, "request_id": None}
    started = time.monotonic()
    try:
        evaluator.reset(prepare_for_task_instance=True)
        evaluator.load_task_instance(metadata["instance_id"])
        evaluator.reset()
        if correction_factory is not None:
            policy.correction = correction_factory(evaluator.robot)
        writer = CodexCollectionTrajectory(output / "trajectory.hdf5", metadata)
        action_log = (output / "action_sources.jsonl").open("x", buffering=1)
        rejection_log = (output / "correction_rejections.jsonl").open("x", buffering=1)
        evaluator.start_recording(str(output / "rollout.mp4"), rate=metadata["fps"])
        recording = True
        while steps < max_steps:
            frame = capture_observation(evaluator.obs)
            if steps >= next_teacher_step:
                if decisions >= max_decisions:
                    stop_reason = "teacher_decision_budget"
                    break
                decision_index = decisions
                snapshot = output / "teacher" / f"step-{steps:08d}-decision-{decision_index:04d}"
                snapshot.mkdir(parents=True)
                images = []
                for index, key in enumerate(CAMERAS):
                    path = snapshot / f"camera-{index}.png"
                    Image.fromarray(frame[key]).save(path)
                    images.append(path)
                request = {"task": metadata["task"], "task_prompt": prompt,
                           "instance_id": metadata["instance_id"], "attempt_id": metadata["attempt_id"],
                           "step": steps, "decision_index": decision_index,
                           "collection_mode": collection_mode,
                           "previous_subtask": last_executed_subtask if native else policy.subtask,
                           "camera_order": ["head", "left_wrist", "right_wrist"],
                           "proprio": frame["observation.state"].tolist(),
                           "recent_decisions": history[-8:], "next_observation_after_steps": teacher_interval}
                if native:
                    request["current_native_subtask"] = original_policy.current_subtask
                    request["current_native_subtask_source"] = original_policy.current_subtask_source
                request["context_images"] = []
                if policy.correction is not None:
                    request["robot_control_state"] = policy.correction.snapshot()
                    request["last_correction_diagnostics"] = policy.correction.last_diagnostics
                    request["last_correction_outcome"] = last_correction_outcome
                if demonstration_manifest is not None:
                    from codex_collection_demonstrations import demonstration_context
                    context, examples = demonstration_context(demonstration_manifest, metadata["task"],
                                                               request["previous_subtask"], limit=1)
                    request["demonstration"] = context
                    for path in examples:
                        images.append(path)
                        request["context_images"].append({"kind": "successful_demonstration",
                                                           "current_observation": False})
                if previous_head is not None:
                    images.append(previous_head[1])
                    request["context_images"].append({"kind": "previous_head_observation",
                                                       "step": previous_head[0], "current_observation": False})
                previous_head = (steps, images[0])
                (snapshot / "request.json").write_text(json.dumps(request, indent=2) + "\n")
                decision = teacher.choose(request, images, snapshot / "decision")
                decisions += 1
                active_decision = {"decision_index": decision_index, "request_id": decision["request_id"]}
                if decision["action"] == "abort":
                    stop_reason = "teacher_abort"
                    break
                if decision["action"] not in ("continue", "correct") or not isinstance(decision["subtask"], str):
                    raise ValueError("Malformed teacher decision")
                label = " ".join(decision["subtask"].split())
                native_continue = native and decision["action"] == "continue"
                if native_continue and (decision["subtask"] != "" or decision.get("correction") is not None):
                    raise ValueError("Native continue must not supply a subtask or correction")
                if not native_continue and (not label or len(label) > 256):
                    raise ValueError("Invalid teacher subtask")
                if not native_continue:
                    policy.subtask = label
                next_teacher_step = steps + teacher_interval
                if decision["action"] == "correct":
                    if policy.correction is None:
                        raise ValueError("Correction is unavailable in this collector")
                    policy.correction.begin(decision["correction"], evaluator.robot_action if steps > 0 else None)
                    next_teacher_step = steps + policy.correction.remaining_steps
                history.append({"step": steps, "subtask": label, "request_id": decision["request_id"],
                                "decision_index": decision_index,
                                "action": decision["action"], "correction": decision.get("correction"),
                                "reason": decision["reason"]})
            # A rejected IK candidate must never enter Evaluator.step(), whose
            # own bookkeeping may advance before policy.forward returns.
            try:
                with time_limit(step_timeout):
                    policy.prepare_correction()
            except RecoverableCorrectionError as error:
                policy.cancel_correction()
                last_correction_outcome = {"status": "rejected", "step": steps, **active_decision,
                    "error_type": type(error).__name__, "reason_code": error.reason_code,
                    "executed": False, "diagnostics": error.diagnostics}
                rejection_log.write(json.dumps(last_correction_outcome, allow_nan=False) + "\n")
                history[-1]["last_rejection"] = last_correction_outcome
                next_teacher_step = steps
                continue
            with time_limit(step_timeout):
                terminated, truncated = evaluator.step()
            terminated, truncated = bool(terminated), bool(truncated)
            executed_label = " ".join(policy.current_subtask.split())
            writer.record(frame, evaluator.robot_action, terminated, truncated, executed_label)
            last_executed_subtask = executed_label
            action_row = {"step": steps, **active_decision, "source": policy.action_source,
                "diagnostics": policy.correction.last_diagnostics if policy.correction is not None
                    and policy.action_source == "teacher_cartesian_correction" else None}
            if native:
                action_row.update(subtask=executed_label, subtask_source=policy.current_subtask_source)
            action_log.write(json.dumps(action_row) + "\n")
            if policy.action_source == "teacher_cartesian_correction":
                last_correction_outcome = {"status": "executed", "step": steps, **active_decision,
                                          "executed": True, "diagnostics": policy.correction.last_diagnostics}
            steps += 1
            if terminated or truncated:
                stop_reason = "environment_terminal"
                break
        evaluator.stop_recording()
        recording = False
        result = {"steps": steps, "task_success": bool(evaluator.env.task.success),
                  "final_goal_state": evaluator.final_goal_state,
                  "terminated": terminated, "truncated": truncated,
                  "stop_reason": stop_reason, "teacher_decisions": decisions,
                  "wall_seconds": time.monotonic() - started, "official_benchmark": False}
        # A zero-action rejection budget is a bounded non-successful completion.
        # Keep its HDF5 incomplete; no unexecuted action becomes training data.
        if steps == 0 and stop_reason == "teacher_decision_budget":
            result["success"] = False
            (output / "completed.json").write_text(json.dumps({"result": result, "content_sha256": None,
                "training_eligible": False, "trajectory_complete": False, "official_benchmark": False,
                "video_path": "rollout.mp4"}, indent=2) + "\n")
            return result
        # A pre-action abort creates an incomplete zero-step attempt, not training data.
        if steps == 0:
            raise RuntimeError("Attempt ended before any action; no trajectory accepted")
        result = writer.finish(result)
        info = inspect_codex_collection_trajectory(output / "trajectory.hdf5", require_success=False)
        (output / "subtask_segments.json").write_text(json.dumps(info["segments"], indent=2) + "\n")
        (output / "completed.json").write_text(json.dumps({
            "result": result, "content_sha256": info["content_sha256"],
            "training_eligible": result["success"], "official_benchmark": False,
            "video_path": "rollout.mp4",
        }, indent=2, default=float) + "\n")
        return result
    except Exception as error:
        (output / "failed.json").write_text(json.dumps({
            "steps": steps, "error_type": type(error).__name__, "message": str(error),
            "training_eligible": False, "official_benchmark": False,
        }, indent=2) + "\n")
        raise
    finally:
        if recording:
            evaluator.stop_recording()
        if writer is not None:
            writer.close()
        if action_log is not None:
            action_log.close()
        if rejection_log is not None:
            rejection_log.close()
        evaluator.policy = original_policy


def parse_args(argv=None):
    """Validate backend selection before importing the simulator or launching it."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--instance-id", type=int, choices=range(311, 321), required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--checkpoint-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--max-decisions", type=int, required=True)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--step-timeout", type=int, default=120)
    parser.add_argument("--teacher-interval", type=int, default=64)
    parser.add_argument("--collection-mode", choices=COLLECTION_MODES, default="teacher_conditioned")
    parser.add_argument("--teacher-url", help="HTTP teacher URL; only with --teacher-token-file")
    transport = parser.add_mutually_exclusive_group(required=True)
    transport.add_argument("--teacher-token-file", type=Path)
    transport.add_argument("--teacher-mailbox-config", type=Path)
    transport.add_argument("--teacher-codex", action="store_true",
                           help="Run the ChatGPT-authenticated Codex CLI on this machine")
    parser.add_argument("--codex-binary", help="Direct backend CLI executable (default: codex)")
    parser.add_argument("--teacher-timeout", type=float,
                        help="Direct backend seconds per decision (default: 120; maximum: 180)")
    parser.add_argument("--demonstration-manifest", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--expected-server-token", required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(v < 1 for v in (args.max_steps, args.max_decisions, args.teacher_interval, args.max_seconds, args.step_timeout)):
        parser.error("Collection budgets must be positive")
    if args.teacher_url is not None and args.teacher_token_file is None:
        parser.error("--teacher-url requires --teacher-token-file")
    if not args.teacher_codex and (args.codex_binary is not None or args.teacher_timeout is not None):
        parser.error("--codex-binary and --teacher-timeout require --teacher-codex")
    if args.teacher_codex:
        args.codex_binary = args.codex_binary if args.codex_binary is not None else "codex"
        if not args.codex_binary.strip():
            parser.error("--codex-binary must name a CLI executable")
        args.teacher_timeout = args.teacher_timeout if args.teacher_timeout is not None else 120.0
        if not math.isfinite(args.teacher_timeout) or not 0 < args.teacher_timeout <= 180:
            parser.error("--teacher-timeout must be finite, positive, and at most 180 seconds")
        args.teacher_backend = "codex_local"
    elif args.teacher_mailbox_config:
        args.teacher_backend = "aoss_mailbox"
    else:
        args.teacher_backend = "remote_http"
        args.teacher_url = args.teacher_url or "http://127.0.0.1:8766"
    task_prompt(args.task_name)  # Reject unknown task before GPU initialization.
    if args.output_dir.exists():
        parser.error("Use a fresh output directory for each independent attempt")
    return args


def build_teacher(args):
    """Reuse one strict teacher protocol for each explicitly selected transport.

    Direct decisions keep their evidence in the same per-step decision directory
    as remote decisions. CodexTeacher verifies the existing ChatGPT login, strips
    API keys from children, and pins the workload's model and reasoning effort.
    """
    if args.teacher_backend == "codex_local":
        from codex_collection_teacher import CodexTeacher
        return CodexTeacher(codex_binary=args.codex_binary, timeout_seconds=args.teacher_timeout)
    if args.teacher_backend == "aoss_mailbox":
        from codex_teacher_mailbox import WorkerAossTeacher
        return WorkerAossTeacher(args.teacher_mailbox_config)
    if args.teacher_backend == "remote_http":
        from codex_teacher_bridge import RemoteTeacher
        return RemoteTeacher(args.teacher_url, args.teacher_token_file)
    raise ValueError("Unknown teacher backend")


def connect_collection_policy(args):
    """Use the vendored client's reset handshake before loading a scene."""
    from omnigibson.eval.policies import WebsocketPolicy
    policy = WebsocketPolicy(host=args.host, port=args.port, allow_reconnect=False,
                             expected_server_token=args.expected_server_token, action_chunk_size=0)
    # The real client is lazy: construction/get_server_metadata do not connect.
    # reset() establishes the socket, validates its identity, then sends reset.
    with time_limit(min(60, args.step_timeout, args.max_seconds)):
        policy.reset()
    metadata = policy.policy.get_server_metadata()
    if (not isinstance(metadata, dict)
            or metadata.get("server_instance_token") != args.expected_server_token
            or metadata.get("subtask_generation") is not True
            or metadata.get("subtask_action_conditioning") != "task_and_subtask"):
        raise ValueError("Collection policy server identity or joint conditioning contract differs")
    return policy, metadata


def record_main_failure(args, stage, error):
    """Keep a safe root-cause receipt even when simulator shutdown exits Python."""
    receipt = dict(kind="codex_collection_main_failure", stage=stage, error_type=type(error).__name__,
                   task=args.task_name, instance_id=args.instance_id, attempt_id=args.attempt_id,
                   training_eligible=False, official_benchmark=False)
    # A startup failure must not create the attempt directory that collect_attempt
    # requires to be fresh. Never replace an earlier failure's evidence.
    path = args.output_dir.parent / "collector_failure.json"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x") as stream:
            json.dump(receipt, stream, indent=2); stream.write("\n")
    except FileExistsError:
        return
    except OSError as write_error:
        receipt["failure_receipt_write_error"] = type(write_error).__name__
    print("COLLECTOR_FAILURE_JSON=" + json.dumps(receipt), file=sys.stderr, flush=True)


@contextmanager
def collection_evaluator(factory, on_failure):
    """Preserve a rollout exception across Evaluator.__exit__/shutdown exit(0)."""
    evaluator = factory()
    try:
        evaluator.__enter__()
        yield evaluator
    except BaseException as error:
        on_failure(error)  # Persist before shutdown can terminate the interpreter.
        try:
            # The safe receipt is the diagnostic; avoid logging arbitrary exception
            # text through the vendored evaluator's traceback printer.
            evaluator.__exit__(None, None, None)
        except BaseException:
            pass  # The original collection failure takes precedence over cleanup.
        raise
    else:
        try:
            evaluator.__exit__(None, None, None)
        except SystemExit as error:
            if error.code not in (None, 0):
                raise


def main(argv=None):
    args = parse_args(argv)
    stage = "initialization"
    try:
        teacher = build_teacher(args)
        from eval_b1k_queue import _load_base_module, _build_evaluator_cfg
        base = _load_base_module()
        base_args = base.parse_args([
            "--task-name", args.task_name, "--mode", "public_test", "--prompt-mode", "whole_task",
            "--instance-indices", str(args.instance_id - 301), "--num-rollouts", "1",
            "--max-steps", str(args.max_steps), "--host", args.host, "--port", str(args.port),
            "--expected-server-token", args.expected_server_token,
            "--robot-config", str(args.robot_config), "--output-dir", str(args.output_dir),
            "--write-video", "--headless",
        ])
        base.gm.HEADLESS = True
        seed = base.seed_everything(base.DEFAULT_EVAL_SEED)
        with time_limit(args.max_seconds):
            stage = "policy_handshake"
            policy, server_metadata = connect_collection_policy(args)

            class ConnectedEvaluator(base.Evaluator):
                def load_policy(self):
                    # Reuse this same verified socket; do not create another lazy client.
                    return policy

            stage = "simulator_startup"
            factory = lambda: ConnectedEvaluator(_build_evaluator_cfg(base, base_args, seed))
            with collection_evaluator(factory, lambda error: record_main_failure(args, stage, error)) as evaluator:
                stage = "collection"
                fps = int(evaluator.env.env_config["rendering_frequency"])
                from codex_collection_actions import BoundedCartesianCorrection
                result = collect_attempt(evaluator, teacher, args.output_dir, {
                    "task": args.task_name, "instance_id": args.instance_id, "attempt_id": args.attempt_id,
                    "checkpoint": args.checkpoint_id, "conditioning": "task_and_subtask", "fps": fps,
                    "teacher_backend": args.teacher_backend,
                    "policy_contract": {key: server_metadata[key] for key in
                                        ("subtask_generation", "subtask_action_conditioning")},
                }, max_steps=args.max_steps, teacher_interval=args.teacher_interval, max_decisions=args.max_decisions,
                   max_seconds=args.max_seconds, step_timeout=args.step_timeout,
                   correction_factory=BoundedCartesianCorrection, demonstration_manifest=args.demonstration_manifest,
                   collection_mode=args.collection_mode)
                print(json.dumps(result, indent=2, default=float), flush=True)
                stage = "shutdown"
        return result
    except BaseException as error:
        record_main_failure(args, stage, error)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
