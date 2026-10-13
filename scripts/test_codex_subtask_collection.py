"""CPU contract/loop checks; synthetic frames never certify a robot success."""
from pathlib import Path
import copy
import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
import codex_subtask_collection as contract
from collect_b1k_codex_subtask import collect_attempt, FixedSubtaskPolicy, parse_args
from codex_collection_actions import RecoverableCorrectionError
from subtask_trajectory import CAMERAS


def fixture_plan(*, camera=False, **budgets):
    from full_subtask_collection import freeze
    sample = dict(sample_id="freeze_fruit-instance-33-episode-10020-segment-3-interval-0",
        task="freeze_fruit", task_instance_id=33, task_index=50, episode_index=10020,
        raw_episode_id=500330, rollout_id=3, source_segment_index=3, source_interval_index=0,
        start=1620, end=1830, segment_frames=210, subtask=contract.FIXED_SUBTASK,
        predicate=dict(kind="role_bound", contract="annotation_roles_v1", operation="placed",
                       targets=["tupperware"], references=["countertop"], state="OnTop"))
    if camera:
        sample.update(sample_id="attach_a_camera_to_a_tripod-instance-4-episode-7003-segment-1-interval-0",
            task=contract.CAMERA_TASK, task_instance_id=4, task_index=35,
            episode_index=7003, raw_episode_id=350040, source_segment_index=1,
            start=520, end=1679, segment_frames=1159, subtask=contract.CAMERA_SUBTASK,
            predicate=dict(kind="role_bound", contract="annotation_roles_v1",
                           operation="grasped", targets=["digital_camera_87"]))
    panel = dict(fingerprint="c" * 64, samples=[sample])
    identity = dict(checkpoint_step=1375000, checkpoint_manifest_sha256="b" * 64,
                    checkpoint_source="/baseline/1375000")
    group = dict(task=sample["task"], subtask=sample["subtask"], assignment_index=9,
                 eligible=5, successes=0, rate=0.0, underpowered=True)
    proofs = {sample["sample_id"]: dict(baseline_metric_eligible=True, raw_sha256="d" * 64,
        selected_state_sha256="e" * 64, baseline_result_sha256="f" * 64, state_protocol=contract.NATIVE)}
    source = freeze(panel, group, [sample], proofs, identity, [[sample["task"], 100]],
                    accepted_input_sha256={"metrics": "a" * 64}, evidence_manifest_sha256="9" * 64,
                    repeats=1, max_steps=2048)
    actual = dict(checkpoint_step=1688977, checkpoint_manifest_sha256="1" * 64,
                  checkpoint_source="/actual/1688977")
    plan = contract.freeze_plan(panel, source, sample["sample_id"], actual,
        checkpoint_id="pi05-1688977", raw_data_root="/original/raw", attempt_id="pilot-001",
        policy_seed=9, **budgets)
    return plan


def rehash(plan):
    plan["source_plan"]["fingerprint"] = contract.digest({k: v for k, v in plan["source_plan"].items() if k != "fingerprint"})
    plan["fingerprint"] = contract.digest({k: v for k, v in plan.items() if k != "fingerprint"})
    return plan


def test_actual_checkpoint_does_not_relabel_old_zero_success_baseline(tmp_path):
    plan = fixture_plan()
    assert plan["checkpoint_identity"]["checkpoint_step"] == 1688977
    assert plan["baseline_checkpoint_identity"]["checkpoint_step"] == 1375000
    assert plan["source_plan"]["checkpoint_identity"]["checkpoint_step"] == 1375000
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(plan))
    assert contract.load_plan(path) == plan


@pytest.mark.parametrize("change", ["heldout", "public_eval", "wrong_target", "zero_denominator", "not_zero", "changed_source", "unverified_state"])
def test_scope_and_source_gates_cannot_be_bypassed_by_outer_rehash(change):
    plan = fixture_plan()
    if change == "heldout": plan["source_plan"]["held_out_instances"].append(["freeze_fruit", 33])
    if change == "public_eval": plan["sample"]["task_instance_id"] = 301
    if change == "wrong_target": plan["sample"]["subtask"] = "grasp tupperware"
    if change == "zero_denominator": plan["source_plan"]["selected_groups"][0]["eligible"] = 0
    if change == "not_zero": plan["source_plan"]["selected_groups"][0]["successes"] = 1
    if change == "changed_source": plan["source_plan"]["mode"] = "public_test"
    if change == "unverified_state": plan["source_plan"]["selected_state_sha256"].clear()
    with pytest.raises(ValueError): contract.validate_plan(rehash(plan))


@pytest.mark.parametrize("budgets", [dict(max_steps=2049), dict(max_decisions=33), dict(max_seconds=7201),
    dict(max_steps=True), dict(step_timeout=121), dict(teacher_interval=0), dict(max_seconds=float("nan"))])
def test_finite_budget_validation(budgets):
    with pytest.raises(ValueError): fixture_plan(**budgets)


def test_camera_retry_budget_is_finite_and_keeps_other_limits():
    plan = fixture_plan(camera=True, max_decisions=64)
    assert plan["budgets"]["max_decisions"] == 64
    assert plan["budgets"]["max_steps"] == 2048
    assert plan["budgets"]["max_seconds"] == 7200
    with pytest.raises(ValueError):
        fixture_plan(camera=True, max_decisions=65)


def test_robot_gripper_feedback_is_explicitly_frozen_and_camera_only():
    plan = fixture_plan(camera=True, teacher_feedback='robot_gripper_execution_v1')
    contract.validate_plan(plan)
    assert plan['teacher_feedback'] == 'robot_gripper_execution_v1'
    with pytest.raises(ValueError):fixture_plan(teacher_feedback='robot_gripper_execution_v1')
    with pytest.raises(ValueError):fixture_plan(camera=True,teacher_feedback='unverified_feedback')


class FakePolicy:
    def __init__(self):
        self.resets = 0
        self.model_calls = 0
        self.current_subtask = contract.FIXED_SUBTASK
        self.current_subtask_source = "provided"
    def reset(self): self.resets += 1
    def forward(self, obs):
        assert obs["subtask"] == contract.FIXED_SUBTASK
        self.model_calls += 1
        return np.full(23, 90, np.float32)


class FakeCorrection:
    def __init__(self, robot):
        self.remaining_steps = 0
        self.last_diagnostics = {}
        self.reject = getattr(robot, "reject", False)
    def snapshot(self): return {"frame": "base", "limits": {"max_steps": 5}}
    def begin(self, correction, previous): self.remaining_steps = correction["steps"]
    def cancel(self): self.remaining_steps = 0
    def next_action(self):
        if self.reject:
            self.reject = False
            raise RecoverableCorrectionError("synthetic IK rejection", reason_code="ik_residual", diagnostics={"arm": "right", "residual": 1})
        self.remaining_steps -= 1
        self.last_diagnostics = {"fixture": True}
        return np.full(23, 7, np.float32)


class FakeEvaluator:
    def __init__(self, *, reject=False, terminal_at=None):
        self.policy = FakePolicy()
        self.robot = SimpleNamespace(reject=reject)
        self.env = SimpleNamespace(task=SimpleNamespace(success=False), env_config={"rendering_frequency": 30})
        self.robot_action = None
        self.steps = 0
        self.terminal_at = terminal_at
        self.recording = None
        self.obs = self.observation()
    def observation(self):
        return {"robot_r1::proprio": np.full(61, self.steps, np.float32),
                **{key: np.full((2, 3, 4), self.steps, np.uint8) for key in CAMERAS.values()}}
    def step(self):
        self.robot_action = self.policy.forward(obs={"subtask": self.external_subtask})
        self.steps += 1
        self.obs = self.observation()
        return self.steps == self.terminal_at, False
    def start_recording(self, path, rate):
        self.recording = Path(path)
        self.recording.write_bytes(b"synthetic test video only")
    def stop_recording(self): pass


class FakeMonitor:
    def __init__(self, *, success_at=3, eligible=True, satisfied_at=None):
        self.metric_eligible = eligible
        self.setup_exclusion_reason = None if eligible else "predicate_already_true"
        self.setup_evidence = {"initial_predicate_value": not eligible}
        self.satisfied = False
        self.dwell = 0
        self.success_at = success_at
        self.satisfied_at = satisfied_at
    def refresh(self, step):
        self.satisfied = step >= (self.satisfied_at if self.satisfied_at is not None else self.success_at - 2)
        self.dwell = self.dwell + 1 if self.satisfied else 0
        return self.dwell >= 3
    def result(self):
        return dict(satisfied=self.satisfied, dwell=self.dwell, debounce_steps=3,
                    metric_eligible=self.metric_eligible, predicate=dict(kind="role_bound",
                        contract="annotation_roles_v1", operation="placed", targets=["tupperware"],
                        references=["countertop"], state="OnTop"))


class FakeTeacher:
    def __init__(self, actions=()): self.actions, self.requests = list(actions), []
    def choose(self, request, images, output):
        self.requests.append(copy.deepcopy(request))
        output.mkdir(parents=True)
        action = self.actions.pop(0) if self.actions else "continue"
        return dict(request_id=str(len(self.requests)), action=action, subtask=contract.FIXED_SUBTASK,
            reason="CPU fixture", correction=dict(steps=1) if action == "correct" else None)


def test_feedback_advances_only_after_actual_steps_across_ik_rejection(tmp_path, monkeypatch):
    import codex_collection_demonstrations
    import codex_subtask_qualification
    monkeypatch.setattr(codex_collection_demonstrations,'demonstration_context',lambda *a,**k:({},[]))
    # This is an isolated loop test, never evidence of actual paired qualification or success.
    monkeypatch.setattr(codex_subtask_qualification,'verify_qualification',lambda *a,**k:{'sha256':'0'*64})
    plan=fixture_plan(camera=True,teacher_feedback='robot_gripper_execution_v1',max_steps=3,max_decisions=3,teacher_interval=1)
    evaluator=FakeEvaluator(reject=True)
    class CameraPolicy(FakePolicy):
        def forward(self,obs):
            assert obs['subtask']==contract.CAMERA_SUBTASK
            self.current_subtask=contract.CAMERA_SUBTASK;self.model_calls+=1
            return np.full(23,90,np.float32)
    evaluator.policy=CameraPolicy()
    class CameraTeacher(FakeTeacher):
        def choose(self,*args):
            answer=super().choose(*args);answer['subtask']=contract.CAMERA_SUBTASK;return answer
    teacher=CameraTeacher(['correct','correct','continue'])
    class Feedback:
        count=0
        def snapshot(self,robot):return {'contract':'CPU-loop-fixture','executed_control_steps':self.count}
        def observe_executed_action(self,action):self.count+=1
    feedback=Feedback();monitor=FakeMonitor(success_at=999)
    def restore(*args):
        monitor.setup_evidence['source_role_binding']=dict(sample_id=plan['sample']['sample_id'],
            raw_sha256='d'*64,source_state_sha256='e'*64,desired_predicate_truth_used=False)
        return dict(recorded_native_source=dict(contract=contract.NATIVE,sample_id=plan['sample']['sample_id'],
            source_start=520,raw_sha256='d'*64,selected_state_sha256='e'*64,
            recorded_native_fields_verified_before_policy=True,full_robot_joints_match=True,
            unrecorded_physics_caches_or_timers_verified=False,recorded_grasps=[],actual_restored_grasps=[]),
            initialization_replay_is_training_data=False)
    receipt=collect_attempt(evaluator,teacher,plan,{}, {},tmp_path/'attempt',demonstration_manifest='CPU-fixture',
        correction_factory=FakeCorrection,restore_fn=restore,monitor_factory=lambda *a:monitor,
        feedback_factory=lambda ev:feedback)
    assert [r['step'] for r in teacher.requests]==[0,0,1]
    assert [r['gripper_execution']['executed_control_steps'] for r in teacher.requests]==[0,0,1]
    assert feedback.count==evaluator.steps==receipt['result']['steps']==2
    assert receipt['training_eligible'] is False


@pytest.fixture
def run(tmp_path, monkeypatch):
    import codex_collection_demonstrations
    monkeypatch.setattr(codex_collection_demonstrations, "demonstration_context",
                        lambda *args, **kw: ({"fixture_reference": True}, []))
    def execute(*, evaluator=None, teacher=None, monitor=None, budgets=None, path=None):
        plan = fixture_plan(**(budgets or {}))
        evaluator = evaluator or FakeEvaluator()
        teacher = teacher or FakeTeacher()
        monitor = monitor or FakeMonitor()
        original = evaluator.policy
        output = path or tmp_path / "attempt"
        def restore(ev, cache, sample, materialized, protocol):
            # Synthetic restore performs no evaluator/model actions or recording.
            assert ev.steps == 0 and ev.recording is None
            monitor.setup_evidence["source_role_binding"] = dict(sample_id=sample["sample_id"],
                raw_sha256="d" * 64, source_state_sha256="e" * 64, desired_predicate_truth_used=False)
            return dict(recorded_native_source=dict(contract=contract.NATIVE,
                sample_id=sample["sample_id"], source_start=sample["start"], raw_sha256="d" * 64,
                selected_state_sha256="e" * 64, recorded_native_fields_verified_before_policy=True,
                full_robot_joints_match=True, unrecorded_physics_caches_or_timers_verified=False,
                recorded_grasps=[], actual_restored_grasps=[]), initialization_replay_is_training_data=False)
        receipt = collect_attempt(evaluator, teacher, plan, {}, {}, output,
            demonstration_manifest="unused-fixture", correction_factory=FakeCorrection,
            restore_fn=restore, monitor_factory=lambda *a: monitor)
        return receipt, evaluator, teacher, original, output
    return execute


def test_local_success_without_whole_task_success_and_action_pairing(run):
    receipt, evaluator, teacher, original, output = run()
    assert receipt["training_eligible"] and receipt["result"]["success"]
    assert receipt["result"]["task_success"] is False
    assert receipt["result"]["actual_model_control_steps"] == 3
    assert receipt["result"]["actual_teacher_control_steps"] == 0
    assert teacher.requests[0]["step"] == 0 and teacher.requests[0]["instance_id"] == 33
    assert receipt["result"]["teacher_usage"][0]["usage"] is None
    assert evaluator.policy is original
    with h5py.File(output / "trajectory.hdf5") as f:
        assert f["observation.state"][:, 0].tolist() == [0, 1, 2]
        assert f["action"][:, 0].tolist() == [90, 90, 90]
    contract.inspect_attempt(output)


def test_teacher_actions_are_recorded_as_executed_not_model_actions(run):
    receipt, evaluator, teacher, original, output = run(teacher=FakeTeacher(["correct", "continue"]))
    assert receipt["result"]["actual_teacher_control_steps"] == 1
    assert receipt["result"]["actual_model_control_steps"] == original.model_calls == 2
    assert original.resets == 1
    assert [r["step"] for r in teacher.requests] == [0, 1]
    with h5py.File(output / "trajectory.hdf5") as f:
        assert f["action"][:, 0].tolist() == [7, 90, 90]
    rows = [json.loads(line) for line in (output / "action_sources.jsonl").read_text().splitlines()]
    assert rows[0]["source"] == "teacher_cartesian_correction"
    assert all(row["subtask"] == contract.FIXED_SUBTASK for row in rows)


def test_rejected_ik_replans_without_advancing_or_recording_physical_step(run):
    receipt, evaluator, teacher, original, output = run(evaluator=FakeEvaluator(reject=True),
        teacher=FakeTeacher(["correct", "continue"]))
    assert [r["step"] for r in teacher.requests] == [0, 0]
    assert teacher.requests[1]["last_correction_outcome"]["executed"] is False
    assert receipt["result"]["steps"] == evaluator.steps == 3
    assert receipt["result"]["actual_teacher_control_steps"] == 0
    assert len((output / "correction_rejections.jsonl").read_text().splitlines()) == 1


def test_invalid_teacher_correction_gets_feedback_and_consumes_only_decision_budget(run):
    from codex_collection_teacher import RejectedTeacherCorrection
    class InvalidThenContinue(FakeTeacher):
        def choose(self, request, images, output):
            if not self.requests:
                self.requests.append(copy.deepcopy(request))
                evidence = output / "invalid-response"
                evidence.mkdir(parents=True)
                (evidence / "receipt.json").write_text(json.dumps({"usage":{"input_tokens":17}}))
                raise RejectedTeacherCorrection(request_id="invalid-response", evidence_dir=evidence,
                    diagnostics={"proposed_translation_norm_m":0.0283,"translation_norm_limit_m":0.02})
            return super().choose(request, images, output)
    receipt, evaluator, teacher, _, output = run(teacher=InvalidThenContinue())
    assert [r["step"] for r in teacher.requests] == [0, 0]
    assert [r["decision_index"] for r in teacher.requests] == [0, 1]
    rejection = teacher.requests[1]["last_correction_outcome"]
    assert rejection["executed"] is False and rejection["reason_code"] == "invalid_teacher_correction"
    assert receipt["result"]["teacher_decisions"] == 2
    assert receipt["result"]["teacher_usage"][0]["usage"] == {"input_tokens":17}
    assert receipt["result"]["steps"] == evaluator.steps == 3
    assert receipt["result"]["actual_teacher_control_steps"] == 0
    assert len((output / "action_sources.jsonl").read_text().splitlines()) == 3
    assert len((output / "correction_rejections.jsonl").read_text().splitlines()) == 1


def test_teacher_budget_is_bounded_and_not_a_success(run):
    receipt, evaluator, *_ = run(monitor=FakeMonitor(success_at=999),
                                 budgets=dict(teacher_interval=1, max_decisions=2))
    assert receipt["result"]["steps"] == 2
    assert receipt["result"]["stop_reason"] == "teacher_decision_budget"
    assert receipt["training_eligible"] is False


def test_setup_already_true_is_excluded_without_model_or_teacher_actions(run):
    receipt, evaluator, teacher, original, _ = run(monitor=FakeMonitor(eligible=False))
    assert receipt["training_eligible"] is False
    assert receipt["result"]["metric_eligible"] is False
    assert receipt["result"]["steps"] == 0 and original.model_calls == 0 and not teacher.requests


def test_terminal_local_predicate_credit(run):
    receipt, *_ = run(evaluator=FakeEvaluator(terminal_at=1), monitor=FakeMonitor(success_at=3))
    assert receipt["training_eligible"] and receipt["result"]["terminal_credit"]
    assert receipt["result"]["monitor"]["dwell"] == 1


def test_fresh_attempt_never_overwrites_evidence(run, tmp_path):
    output = tmp_path / "attempt"
    run(path=output)
    original = (output / "trajectory.hdf5").read_bytes()
    with pytest.raises(FileExistsError): run(path=output)
    assert (output / "trajectory.hdf5").read_bytes() == original


def test_no_teacher_subtask_switch_is_accepted(run):
    class BadTeacher(FakeTeacher):
        def choose(self, *args):
            return {**super().choose(*args), "subtask": "move somewhere else"}
    with pytest.raises(ValueError, match="fixed subtask"): run(teacher=BadTeacher())


def test_mixed_source_tampering_and_false_model_count_rejected(run):
    _, _, _, _, output = run(teacher=FakeTeacher(["correct", "continue"]))
    with h5py.File(output / "trajectory.hdf5", "r+") as f:
        result = json.loads(f.attrs["result"])
        result["actual_model_control_steps"] = result["steps"]
        f.attrs["result"] = json.dumps(result)
    with pytest.raises(ValueError, match="Teacher actions"): contract.inspect_attempt(output)


def test_changed_restoration_or_unsustained_success_cannot_be_exported(run):
    _, _, _, _, output = run()
    with h5py.File(output / "trajectory.hdf5", "r+") as f:
        original = json.loads(f.attrs["result"])
        result = copy.deepcopy(original)
        result["restoration"]["recorded_native_source"]["selected_state_sha256"] = "0" * 64
        f.attrs["result"] = json.dumps(result)
    with pytest.raises(ValueError, match="source identity"): contract.inspect_attempt(output)
    with h5py.File(output / "trajectory.hdf5", "r+") as f:
        result = copy.deepcopy(original)
        result["monitor"]["dwell"] = 1
        f.attrs["result"] = json.dumps(result)
    with pytest.raises(ValueError, match="completion predicate"): contract.inspect_attempt(output)


def test_teacher_abort_does_not_fabricate_success(run):
    receipt, evaluator, teacher, original, _ = run(teacher=FakeTeacher(["abort"]))
    assert receipt["training_eligible"] is False and receipt["result"]["steps"] == 0
    assert original.model_calls == 0 and receipt["result"]["stop_reason"] == "teacher_abort"


def test_student_cannot_silently_ignore_fixed_label_before_action_execution():
    original = FakePolicy()
    original.current_subtask = "wrong label"
    wrapped = FixedSubtaskPolicy(original, contract.FIXED_SUBTASK, FakeCorrection(SimpleNamespace()))
    with pytest.raises(RuntimeError, match="acknowledge"): wrapped.forward({})


@pytest.mark.parametrize("value", ["0", "121", "nan", "inf"])
def test_cli_rejects_unbounded_teacher_timeout_before_gpu_import(value):
    with pytest.raises(SystemExit):
        parse_args(["--plan", "plan.json", "--output-dir", "fresh", "--policy-port", "8930",
            "--policy-server-token", "private-fixture", "--teacher-codex", "--codex-timeout", value,
            "--demonstration-manifest", "demo.json", "--robot-config", "r1pro.yaml"])


def test_teacher_configuration_is_frozen_and_validated():
    plan = fixture_plan(camera=True, teacher_configuration=dict(model='gpt-6-astra', reasoning_effort='high', service_tier='ultrafast'))
    contract.validate_plan(plan)
    changed = copy.deepcopy(plan)
    changed['teacher_configuration']['service_tier'] = 'bogus'
    changed['fingerprint'] = contract.digest({k: v for k, v in changed.items() if k != 'fingerprint'})
    with pytest.raises(ValueError, match='Unsupported'):
        contract.validate_plan(changed)
