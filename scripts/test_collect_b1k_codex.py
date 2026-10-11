"""CPU protocol tests; fake simulator results are NEVER collection data."""
import json
from pathlib import Path
from types import SimpleNamespace
import time

import h5py
import numpy as np
import pytest

from codex_collection_trajectory import inspect_codex_collection_trajectory
from collect_b1k_codex import TeacherPolicy, build_teacher, collect_attempt, parse_args, time_limit
from subtask_trajectory import CAMERAS


class Policy:
    _action_chunk_size = 0
    current_subtask = None
    current_subtask_source = None
    def reset(self):
        self.current_subtask = self.current_subtask_source = None
    def forward(self, obs):
        assert obs["need_new_action"] is True
        self.current_subtask = obs["subtask"]
        self.current_subtask_source = "provided"
        return np.ones(23, np.float32) * float(obs["robot_r1::proprio"][0])


class Evaluator:
    prompt_mode = "whole_task"
    external_subtask_selector = external_subtask = None
    def __init__(self, terminal_step=3, success=True):
        self.policy = Policy()
        self.env = SimpleNamespace(task=SimpleNamespace(success=False))
        self.terminal_step, self.end_success = terminal_step, success
        self.steps = 0
        self.executed = []
        self.reset_calls = []
    @property
    def final_goal_state(self):
        return {"success": self.env.task.success}
    def observation(self):
        return {"robot_r1::proprio": np.ones(57, np.float32) * self.steps,
                **{source: np.full((4, 4, 3), self.steps, dtype=np.uint8) for source in CAMERAS.values()}}
    def reset(self, **kwargs):
        self.reset_calls.append(kwargs)
        self.steps = 0
        self.obs = self.observation()
        self.policy.reset()
    def load_task_instance(self, instance):
        self.instance = instance
    def start_recording(self, path, rate):
        self.recording = True
        Path(path).write_bytes(b"fake-test-video")
    def stop_recording(self):
        self.recording = False
    def step(self):
        self.robot_action = self.policy.forward(self.obs)
        self.executed.append(self.robot_action.copy())
        self.steps += 1
        self.obs = self.observation()
        done = self.steps >= self.terminal_step
        self.env.task.success = done and self.end_success
        return done, False


class Teacher:
    def __init__(self, abort=False):
        self.requests = []
        self.abort = abort
    def choose(self, request, images, output):
        assert 3 <= len(images) <= 5 and all(p.is_file() for p in images)
        assert len(request["context_images"]) == len(images)-3
        self.requests.append(request)
        return {"request_id": f"fake-{len(self.requests)}", "action": "abort" if self.abort else "continue",
                "subtask": "move to the radio" if request["step"] == 0 else "turn on the radio", "reason": "fixture"}


def metadata(instance=311):
    return dict(task="turning_on_radio", instance_id=instance, attempt_id="test-only", checkpoint="fixture",
                conditioning="task_and_subtask", fps=30)


def run(tmp_path, evaluator=None, teacher=None, **kwargs):
    return collect_attempt(evaluator or Evaluator(), teacher or Teacher(), tmp_path / "attempt", metadata(),
                           max_steps=kwargs.pop("max_steps", 10), teacher_interval=2,
                           max_decisions=kwargs.pop("max_decisions", 5), **kwargs)


class NativePolicy(Policy):
    """A strict native input contract, with observable label/cache changes."""
    def __init__(self):
        self.reset_count = 0
        self.forward_steps = []
        self.observations = []
    def reset(self):
        self.reset_count += 1
        return Policy.reset(self)
    def forward(self, obs):
        assert "subtask" not in obs and "need_new_action" not in obs
        step = int(obs["robot_r1::proprio"][0])
        self.forward_steps.append(step)
        self.observations.append(dict(obs))
        self.current_subtask = "native reach radio" if step < 1 else "native turn dial"
        self.current_subtask_source = "generated" if step in (0, 1) else "cached"
        return np.full(23, step, np.float32)


class NativeTeacher(Teacher):
    def choose(self, request, images, output):
        assert request["collection_mode"] == "native_with_corrections"
        assert request["current_native_subtask_source"] in {"generated", "cached"}
        self.requests.append(request)
        return {"request_id": f"native-{len(self.requests)}", "action": "continue",
                "subtask": "", "correction": None, "reason": "Observe native progress"}


def test_native_initial_interval_and_continue_preserve_labels_and_cache(tmp_path):
    evaluator, teacher = Evaluator(terminal_step=5), NativeTeacher()
    evaluator.policy = original = NativePolicy()
    result = run(tmp_path, evaluator, teacher, collection_mode="native_with_corrections")
    assert result["success"] and result["teacher_decisions"] == 2
    assert [row["step"] for row in teacher.requests] == [2, 4]
    assert teacher.requests[0]["previous_subtask"] == "native turn dial"
    assert teacher.requests[0]["current_native_subtask"] == "native turn dial"
    assert original.reset_count == 2 and original.forward_steps == list(range(5))
    info = inspect_codex_collection_trajectory(tmp_path / "attempt/trajectory.hdf5")
    assert info["metadata"]["collection_mode"] == "native_with_corrections"
    assert info["metadata"]["subtask_source"] == "mixed_native_and_teacher"
    assert info["segments"] == [dict(start=0,end=1,subtask="native reach radio"),
                                  dict(start=1,end=5,subtask="native turn dial")]
    rows = [json.loads(line) for line in (tmp_path / "attempt/action_sources.jsonl").read_text().splitlines()]
    assert [row["decision_index"] for row in rows] == [None, None, 0, 0, 1]
    assert [row["request_id"] for row in rows] == [None, None, "native-1", "native-1", "native-2"]
    assert all(row["source"] == "policy_native_subtask" for row in rows)
    assert [row["subtask_source"] for row in rows] == ["generated", "generated", "cached", "cached", "cached"]
    assert [row["subtask"] for row in rows] == ["native reach radio"] + ["native turn dial"] * 4


def test_native_can_finish_before_first_teacher_observation(tmp_path):
    evaluator, teacher = Evaluator(terminal_step=1), NativeTeacher()
    evaluator.policy = NativePolicy()
    result = run(tmp_path, evaluator, teacher, collection_mode="native_with_corrections")
    assert result["success"] and result["teacher_decisions"] == 0 and teacher.requests == []
    assert not (tmp_path / "attempt/teacher").exists()
    rows=[json.loads(line) for line in (tmp_path / "attempt/action_sources.jsonl").read_text().splitlines()]
    assert rows[0]["decision_index"] is None and rows[0]["request_id"] is None
    assert inspect_codex_collection_trajectory(tmp_path / "attempt/trajectory.hdf5")["result"]["teacher_decisions"] == 0


def test_native_decision_budget_stops_without_unbounded_continue(tmp_path):
    evaluator, teacher = Evaluator(terminal_step=100), NativeTeacher()
    evaluator.policy = NativePolicy()
    result = run(tmp_path, evaluator, teacher, collection_mode="native_with_corrections", max_decisions=1)
    assert result["steps"] == 4 and result["teacher_decisions"] == 1
    assert result["stop_reason"] == "teacher_decision_budget" and result["success"] is False
    assert [row["step"] for row in teacher.requests] == [2]


@pytest.mark.parametrize("source,label", [("provided", "native label"), (None, "native label"),
                                          ("generated", ""), ("cached", None)])
def test_native_invalid_report_never_executes_or_records(tmp_path, source, label):
    evaluator = Evaluator()
    class InvalidNative(NativePolicy):
        def forward(self, obs):
            action = super().forward(obs)
            self.current_subtask, self.current_subtask_source = label, source
            return action
    evaluator.policy = InvalidNative()
    with pytest.raises(RuntimeError, match="actual native subtask"):
        run(tmp_path, evaluator, NativeTeacher(), collection_mode="native_with_corrections")
    assert evaluator.steps == 0 and evaluator.executed == []


def test_native_long_actual_label_is_not_teacher_truncated(tmp_path):
    evaluator = Evaluator(terminal_step=1)
    label = "actual policy wording " * 20
    class LongNative(NativePolicy):
        def forward(self, obs):
            action = super().forward(obs);self.current_subtask = label;return action
    evaluator.policy = LongNative()
    run(tmp_path, evaluator, NativeTeacher(), collection_mode="native_with_corrections")
    with h5py.File(tmp_path / "attempt/trajectory.hdf5") as stream:
        assert stream["subtask"].asstr()[0] == label.strip()


def test_native_teacher_cannot_override_subtask_through_continue(tmp_path):
    evaluator = Evaluator(terminal_step=5);evaluator.policy = NativePolicy()
    with pytest.raises(ValueError, match="must not supply"):
        run(tmp_path, evaluator, Teacher(), collection_mode="native_with_corrections")
    assert evaluator.steps == 2 and len(evaluator.executed) == 2


def test_whole_task_dynamic_labels_preceding_frames_and_true_actions(tmp_path):
    evaluator, teacher = Evaluator(), Teacher()
    original = evaluator.policy
    result = run(tmp_path, evaluator, teacher)
    assert result["success"] is True and result["steps"] == 3
    info = inspect_codex_collection_trajectory(tmp_path / "attempt/trajectory.hdf5")
    assert info["segments"] == [dict(start=0, end=2, subtask="move to the radio"),
                                 dict(start=2, end=3, subtask="turn on the radio")]
    with h5py.File(tmp_path / "attempt/trajectory.hdf5") as f:
        assert f["observation.state"][:, 0].tolist() == [0, 1, 2]
        assert f["action"][:, 0].tolist() == [0, 1, 2]
    assert evaluator.reset_calls == [{"prepare_for_task_instance": True}, {}]
    assert [r["step"] for r in teacher.requests] == [0, 2]
    assert teacher.requests[1]["previous_subtask"] == "move to the radio"
    assert evaluator.policy is original and not evaluator.recording


def test_policy_mismatch_never_executes_action(tmp_path):
    evaluator = Evaluator()
    class WrongPolicy(Policy):
        def forward(self, obs):
            action = super().forward(obs)
            self.current_subtask_source = "generated"
            return action
    evaluator.policy = WrongPolicy()
    with pytest.raises(RuntimeError, match="acknowledge"):
        run(tmp_path, evaluator)
    assert not evaluator.executed
    assert not (tmp_path / "attempt/completed.json").exists()
    assert json.loads((tmp_path / "attempt/failed.json").read_text())["training_eligible"] is False
    with h5py.File(tmp_path / "attempt/trajectory.hdf5") as f:
        assert not f.attrs["complete"]


def test_terminal_failure_is_saved_but_rejected_for_training(tmp_path):
    result = run(tmp_path, Evaluator(success=False))
    assert result["success"] is False
    with pytest.raises(ValueError):
        inspect_codex_collection_trajectory(tmp_path / "attempt/trajectory.hdf5")


def test_budget_does_not_claim_complete_task(tmp_path):
    result = run(tmp_path, Evaluator(terminal_step=20), max_decisions=1)
    assert result["steps"] == 2 and result["stop_reason"] == "teacher_decision_budget"
    assert result["success"] is False


def test_teacher_abort_is_not_success(tmp_path):
    with pytest.raises(RuntimeError, match="before any action"):
        run(tmp_path, teacher=Teacher(abort=True))
    assert not (tmp_path / "attempt/completed.json").exists()


@pytest.mark.parametrize("instance", [301, 310, 321, True])
def test_eval_instances_never_reset_or_call_teacher(tmp_path, instance):
    evaluator, teacher = Evaluator(), Teacher()
    with pytest.raises(ValueError, match="311"):
        collect_attempt(evaluator, teacher, tmp_path / "attempt", metadata(instance),
                        max_steps=10, teacher_interval=2, max_decisions=2)
    assert evaluator.reset_calls == [] and teacher.requests == []
    assert not (tmp_path / "attempt").exists()


def test_preexisting_attempt_not_overwritten(tmp_path):
    run(tmp_path)
    with pytest.raises(FileExistsError):
        run(tmp_path)


def test_client_chunk_rejected():
    policy = Policy()
    policy._action_chunk_size = 16
    with pytest.raises(ValueError, match="chunk"):
        TeacherPolicy(policy, "task")


def test_stuck_step_times_out_without_completed_receipt(tmp_path):
    evaluator = Evaluator()
    evaluator.step = lambda: time.sleep(1)
    with pytest.raises(TimeoutError):
        run(tmp_path, evaluator, step_timeout=0.03)
    assert not (tmp_path / "attempt/completed.json").exists()
    with h5py.File(tmp_path / "attempt/trajectory.hdf5") as f:
        assert not f.attrs["complete"]


def test_nested_time_limit_preserves_outer_deadline():
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        with time_limit(0.04):
            with time_limit(1):
                time.sleep(1)
    assert time.monotonic() - started < 0.5


def test_expired_outer_deadline_does_not_interrupt_failure_cleanup():
    with time_limit(0.02):
        try:
            with time_limit(1):
                time.sleep(1)
        except TimeoutError:
            time.sleep(0.03)  # Cleanup must not receive a second expired alarm.


class CountingPolicy(Policy):
    def __init__(self):
        self.reset_count = 0
        self.forward_steps = []
    def reset(self):
        self.reset_count += 1
        return super().reset()
    def forward(self, obs):
        self.forward_steps.append(int(obs["robot_r1::proprio"][0]))
        return super().forward(obs)


class CorrectionFixture:
    """Distinct fixture actions let us verify the actually executed action source."""
    def __init__(self, robot):
        assert robot.evaluator.reset_calls == [{"prepare_for_task_instance": True}, {}]
        assert robot.evaluator.instance == 311
        self.evaluator = robot.evaluator
        self.remaining_steps = 0
        self.last_diagnostics = None
        self.starts = []
    def snapshot(self):
        return {"contract": "test-only", "step": self.evaluator.steps}
    def begin(self, correction, last_action=None):
        assert self.remaining_steps == 0
        if self.evaluator.steps == 0:
            assert last_action is None, "No previously executed action exists at step zero"
        else:
            np.testing.assert_equal(last_action, self.evaluator.executed[-1])
        self.starts.append((self.evaluator.steps, correction))
        self.remaining_steps = correction["steps"]
    def next_action(self):
        assert self.remaining_steps > 0
        self.remaining_steps -= 1
        self.last_diagnostics = {"fixture_step": self.evaluator.steps}
        return np.full(23, 100 + self.evaluator.steps, np.float32)
    def cancel(self):
        self.remaining_steps = 0


def correction_decision(step, *, first=4, label="turn on the radio"):
    return {"request_id": f"fixture-{step}", "action": "correct" if step == first else "continue",
            "subtask": label, "reason": "fixture only",
            "correction": {"arm": "left", "delta_position_m": [0.01, 0, 0],
                           "delta_rotation_axis_angle_rad": [0, 0, 0], "gripper": "keep", "steps": 2}
            if step == first else None}


def test_correction_actions_recorded_with_immediate_feedback_and_same_label_cache_reset(tmp_path):
    evaluator = Evaluator(terminal_step=7)
    evaluator.policy = original = CountingPolicy()
    evaluator.robot = SimpleNamespace(evaluator=evaluator)
    requests = []
    class BurstTeacher:
        def choose(self, request, images, output):
            requests.append(request)
            return correction_decision(request["step"])
    result = collect_attempt(evaluator, BurstTeacher(), tmp_path / "attempt", metadata(),
        max_steps=10, teacher_interval=4, max_decisions=3, correction_factory=CorrectionFixture)
    assert result["success"] is True
    assert [row["step"] for row in requests] == [0, 4, 6]  # feedback after 2-step correction, not step 8
    assert requests[2]["last_correction_diagnostics"] == {"fixture_step": 5}
    assert requests[2]["previous_subtask"] == requests[1]["previous_subtask"] == "turn on the radio"
    assert original.forward_steps == [0, 1, 2, 3, 6]
    assert original.reset_count == 3  # initial resets + one cache flush before post-correction inference
    assert evaluator.policy is original
    with h5py.File(tmp_path / "attempt/trajectory.hdf5") as stream:
        assert stream["observation.state"][:, 0].tolist() == list(range(7))
        assert stream["action"][:, 0].tolist() == [0, 1, 2, 3, 104, 105, 6]
        np.testing.assert_equal(stream["action"][:], evaluator.executed)
    sources = [json.loads(line) for line in (tmp_path / "attempt/action_sources.jsonl").read_text().splitlines()]
    assert [row["step"] for row in sources if row["source"] == "teacher_cartesian_correction"] == [4, 5]
    assert sources[4]["diagnostics"] == {"fixture_step": 4}
    assert sources[6]["diagnostics"] is None


def test_step_zero_correction_does_not_inherit_unexecuted_evaluator_action(tmp_path):
    evaluator = Evaluator(terminal_step=3)
    evaluator.robot = SimpleNamespace(evaluator=evaluator)
    evaluator.robot_action = {}  # Real Evaluator initialization, not an executed 23-D command.
    evaluator.policy = CountingPolicy()
    class ImmediateTeacher:
        def choose(self, request, images, output):
            return correction_decision(request["step"], first=0)
    result = collect_attempt(evaluator, ImmediateTeacher(), tmp_path / "attempt", metadata(),
        max_steps=5, teacher_interval=64, max_decisions=2, correction_factory=CorrectionFixture)
    assert result["steps"] == 3 and result["success"]
    assert [float(action[0]) for action in evaluator.executed] == [100, 101, 2]


def test_wrapper_reports_correction_label_to_evaluator_without_policy_inference():
    base = CountingPolicy()
    base.current_subtask, base.current_subtask_source = "old label", "provided"
    wrapper = TeacherPolicy(base, "task")
    wrapper.subtask = "regrasp the radio"
    class OneCorrection:
        remaining_steps = 1
        def next_action(self):
            self.remaining_steps = 0
            return np.zeros(23, np.float32)
    wrapper.correction = OneCorrection()
    wrapper.prepare_correction()
    wrapper.forward({})
    assert base.forward_steps == []
    assert wrapper.current_subtask == "regrasp the radio"
    assert wrapper.current_subtask_source in ("provided", "teacher_cartesian_correction")


def test_forward_cannot_compute_unprepared_ik_inside_evaluator():
    wrapper = TeacherPolicy(CountingPolicy(), 'task')
    wrapper.subtask = 'align the radio'
    wrapper.correction = SimpleNamespace(remaining_steps=1, next_action=lambda: pytest.fail('IK ran inside forward'))
    with pytest.raises(RuntimeError, match='prepared before'):wrapper.forward({})


class RejectedCorrectionFixture(CorrectionFixture):
    def next_action(self):
        from codex_collection_actions import RecoverableCorrectionError
        raise RecoverableCorrectionError('IK linearized residual too large; correction was not executed',
            reason_code='ik_residual', diagnostics=dict(arm='left',linearized_position_residual_m=0.01,
                linearized_position_limit_m=0.0025,linearized_orientation_residual_rad=0.0,
                linearized_orientation_limit_rad=0.01))


class PrebookkeepingEvaluator(Evaluator):
    """Real evaluator bookkeeping may happen before policy.forward; detect any rejected step entry."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_entries = 0
        self.robot = SimpleNamespace(evaluator=self)
    def step(self):
        self.step_entries += 1
        return super().step()


def test_recoverable_preflight_replans_same_physical_step_without_fake_frames(tmp_path):
    from codex_collection_teacher import request_identity
    evaluator = PrebookkeepingEvaluator(terminal_step=2)
    requests, directories, identities = [], [], []
    class ReplanningTeacher:
        def choose(self, request, images, output):
            requests.append(request);directories.append(output.parent.name)
            identities.append(request_identity(request,images))
            decision=correction_decision(request['step'],first=0 if request['decision_index']==0 else -1)
            decision['request_id']=identities[-1]
            return decision
    result=collect_attempt(evaluator,ReplanningTeacher(),tmp_path/'attempt',metadata(),max_steps=3,
        teacher_interval=8,max_decisions=3,correction_factory=RejectedCorrectionFixture)
    assert result['steps']==2 and result['teacher_decisions']==2 and result['success']
    assert [r['step'] for r in requests]==[0,0] and [r['decision_index'] for r in requests]==[0,1]
    assert len(set(directories))==2 and len(set(identities))==2
    assert directories==['step-00000000-decision-0000','step-00000000-decision-0001']
    assert evaluator.step_entries==2 and len(evaluator.executed)==2
    feedback=requests[1]['last_correction_outcome']
    assert feedback['executed'] is False and feedback['decision_index']==0 and feedback['request_id']==identities[0]
    assert feedback['diagnostics']['linearized_position_residual_m']==0.01
    events=[json.loads(line) for line in (tmp_path/'attempt/correction_rejections.jsonl').read_text().splitlines()]
    assert events==[feedback]
    with h5py.File(tmp_path/'attempt/trajectory.hdf5') as stream:
        assert stream['observation.state'][:,0].tolist()==[0,1]
        np.testing.assert_equal(stream['action'][:],evaluator.executed)
    actions=[json.loads(line) for line in (tmp_path/'attempt/action_sources.jsonl').read_text().splitlines()]
    assert [row['decision_index'] for row in actions]==[1,1]
    assert all(row['source']=='policy_conditioned_on_teacher_subtask' for row in actions)


@pytest.mark.parametrize('max_decisions',[1,3,32])
def test_rejection_budget_ends_zero_action_attempt_without_failure_or_fake_trajectory(tmp_path,max_decisions):
    evaluator=PrebookkeepingEvaluator(terminal_step=2);requests=[]
    class AlwaysRejectTeacher:
        def choose(self,request,images,output):
            requests.append(request)
            return {**correction_decision(0,first=0),'request_id':f"fixture-{request['decision_index']}"}
    result=collect_attempt(evaluator,AlwaysRejectTeacher(),tmp_path/'attempt',metadata(),max_steps=3,
        teacher_interval=8,max_decisions=max_decisions,correction_factory=RejectedCorrectionFixture)
    assert result['steps']==0 and result['teacher_decisions']==max_decisions and result['success'] is False
    assert result['stop_reason']=='teacher_decision_budget' and evaluator.step_entries==0
    assert [row['decision_index'] for row in requests]==list(range(max_decisions))
    assert len(list((tmp_path/'attempt/teacher').glob('step-*-decision-*')))==max_decisions
    receipt=json.loads((tmp_path/'attempt/completed.json').read_text())
    assert receipt['training_eligible'] is False and receipt['trajectory_complete'] is False
    assert not (tmp_path/'attempt/failed.json').exists()
    assert (tmp_path/'attempt/action_sources.jsonl').read_text()==''
    with h5py.File(tmp_path/'attempt/trajectory.hdf5') as stream:
        assert not stream.attrs['complete'] and stream['frame_index'].shape==(0,)


def test_mid_burst_rejection_preserves_only_executed_actions_and_resumes_policy(tmp_path):
    from codex_collection_actions import RecoverableCorrectionError
    evaluator=PrebookkeepingEvaluator(terminal_step=4)
    evaluator.policy=policy=CountingPolicy();requests=[]
    class PartialCorrection(CorrectionFixture):
        def next_action(self):
            if self.evaluator.steps==1:
                raise RecoverableCorrectionError('bounds',reason_code='ik_joint_bounds',
                    diagnostics=dict(arm='left',max_joint_delta_rad=0.2,joint_delta_limit_rad=0.15))
            return super().next_action()
    class ReplanTeacher:
        def choose(self,request,images,output):
            requests.append(request)
            return correction_decision(request['step'],first=0)
    result=collect_attempt(evaluator,ReplanTeacher(),tmp_path/'attempt',metadata(),max_steps=5,
        teacher_interval=8,max_decisions=3,correction_factory=PartialCorrection)
    assert result['success'] and [r['step'] for r in requests]==[0,1]
    assert evaluator.step_entries==4
    assert requests[1]['last_correction_outcome']['step']==1
    assert [float(action[0]) for action in evaluator.executed]==[100,1,2,3]
    assert policy.reset_count==3
    with h5py.File(tmp_path/'attempt/trajectory.hdf5') as stream:
        assert stream['observation.state'][:,0].tolist()==[0,1,2,3]
        np.testing.assert_equal(stream['action'][:],evaluator.executed)


def test_hard_correction_error_remains_fatal_before_any_evaluator_step(tmp_path):
    from codex_collection_actions import CorrectionError
    evaluator=PrebookkeepingEvaluator();requests=[]
    class HardFailure(CorrectionFixture):
        def next_action(self):raise CorrectionError('Current joints or joint limits are invalid')
    class ImmediateTeacher:
        def choose(self,request,images,output):
            requests.append(request);return correction_decision(0,first=0)
    with pytest.raises(CorrectionError,match='Current joints'):
        collect_attempt(evaluator,ImmediateTeacher(),tmp_path/'attempt',metadata(),max_steps=3,
            teacher_interval=8,max_decisions=3,correction_factory=HardFailure)
    assert evaluator.step_entries==0 and len(requests)==1
    assert (tmp_path/'attempt/correction_rejections.jsonl').read_text()==''
    assert (tmp_path/'attempt/failed.json').exists() and not (tmp_path/'attempt/completed.json').exists()


def test_actual_jparse_rejection_replans_and_only_accepted_action_reaches_step(tmp_path):
    from codex_collection_actions import BoundedCartesianCorrection
    from test_codex_collection_actions import Robot, Backend
    evaluator=PrebookkeepingEvaluator(terminal_step=1);evaluator.robot=Robot()
    backend=Backend();backend.jac=np.c_[np.diag([0,1,1,1,1,1]),np.zeros(6)]
    requests=[]
    class ChangeDirectionTeacher:
        def choose(self,request,images,output):
            requests.append(request)
            response=correction_decision(0,first=0)
            response['request_id']=f"numerical-fixture-{request['decision_index']}"
            response['correction']['steps']=1
            if request['decision_index']==1:response['correction']['delta_position_m']=[0,0.01,0]
            return response
    result=collect_attempt(evaluator,ChangeDirectionTeacher(),tmp_path/'attempt',metadata(),max_steps=2,
        teacher_interval=8,max_decisions=3,
        correction_factory=lambda robot:BoundedCartesianCorrection(robot,backend=backend))
    assert result['success'] and result['teacher_decisions']==2
    assert evaluator.step_entries==1 and [r['step'] for r in requests]==[0,0]
    assert requests[1]['last_correction_outcome']['reason_code']=='ik_residual'
    assert requests[1]['last_correction_outcome']['diagnostics']['linearized_position_residual_m']==pytest.approx(0.01)
    assert evaluator.executed[0][8]==pytest.approx(0.01)
    with h5py.File(tmp_path/'attempt/trajectory.hdf5') as stream:
        assert stream['action'].shape==(1,23)
        np.testing.assert_equal(stream['action'][:],evaluator.executed)


def test_rejection_at_final_decision_finishes_prior_executed_frames_as_non_success(tmp_path):
    evaluator=PrebookkeepingEvaluator(terminal_step=10)
    class LateRejectTeacher:
        def choose(self,request,images,output):return correction_decision(request['step'],first=2)
    result=collect_attempt(evaluator,LateRejectTeacher(),tmp_path/'attempt',metadata(),max_steps=5,
        teacher_interval=2,max_decisions=2,correction_factory=RejectedCorrectionFixture)
    assert result['steps']==2 and result['teacher_decisions']==2 and result['success'] is False
    assert result['stop_reason']=='teacher_decision_budget' and evaluator.step_entries==2
    assert not (tmp_path/'attempt/failed.json').exists()
    info=inspect_codex_collection_trajectory(tmp_path/'attempt/trajectory.hdf5',require_success=False)
    assert info['steps']==2 and info['result']['success'] is False
    assert json.loads((tmp_path/'attempt/completed.json').read_text())['training_eligible'] is False


def test_demonstration_and_previous_head_images_are_labeled_and_ordered(tmp_path, monkeypatch):
    from PIL import Image
    import codex_collection_demonstrations
    from codex_collection_teacher import request_identity
    montage = tmp_path / "example.png"
    Image.fromarray(np.full((4, 4, 3), 222, dtype=np.uint8)).save(montage)
    demo_calls = []
    def context(manifest, task, previous, *, limit):
        demo_calls.append((manifest, task, previous, limit))
        return {"source_kind": "original_train", "source_instance_id": 1}, [montage]
    monkeypatch.setattr(codex_collection_demonstrations, "demonstration_context", context)
    received = []
    class ContextTeacher:
        def choose(self, request, images, output):
            request_identity(request, images)  # Actual v2 count/context validation.
            received.append((request, [np.array(Image.open(path))[0, 0, 0].item() for path in images]))
            return {"request_id": "fixture-only", "action": "continue", "subtask": "turn on the radio", "reason": "fixture"}
    evaluator = Evaluator(terminal_step=3)
    collect_attempt(evaluator, ContextTeacher(), tmp_path / "attempt", metadata(), max_steps=5,
                    teacher_interval=2, max_decisions=2, demonstration_manifest=tmp_path / "manifest.json")
    first, second = received
    assert first[1] == [0, 0, 0, 222]
    assert second[1] == [2, 2, 2, 222, 0]
    assert first[0]["camera_order"] == ["head", "left_wrist", "right_wrist"]
    assert first[0]["context_images"] == [{"kind": "successful_demonstration", "current_observation": False}]
    assert second[0]["context_images"] == [
        {"kind": "successful_demonstration", "current_observation": False},
        {"kind": "previous_head_observation", "step": 0, "current_observation": False}]
    assert demo_calls[0][2:] == (None, 1)
    assert demo_calls[1][2:] == ("turn on the radio", 1)


def collection_cli(tmp_path):
    return ["--task-name", "turning_on_radio", "--instance-id", "311", "--attempt-id", "test-only",
            "--checkpoint-id", "fixture", "--output-dir", str(tmp_path / "attempt"),
            "--max-steps", "10", "--max-decisions", "2", "--max-seconds", "300",
            "--demonstration-manifest", str(tmp_path / "demo.json"),
            "--expected-server-token", "fixture", "--robot-config", str(tmp_path / "robot.yaml")]


@pytest.mark.parametrize("flags", [
    [], ["--teacher-codex", "--teacher-token-file", "token"],
    ["--teacher-codex", "--teacher-mailbox-config", "mailbox"],
    ["--teacher-token-file", "token", "--teacher-mailbox-config", "mailbox"],
])
def test_exactly_one_teacher_backend_required(tmp_path, flags):
    with pytest.raises(SystemExit) as error:
        parse_args(collection_cli(tmp_path) + flags)
    assert error.value.code == 2


@pytest.mark.parametrize("flags", [
    ["--teacher-codex", "--teacher-url", "http://127.0.0.1:8766"],
    ["--teacher-mailbox-config", "mailbox", "--teacher-url", "http://127.0.0.1:8766"],
    ["--teacher-token-file", "token", "--codex-binary", "/opt/bin/codex"],
    ["--teacher-mailbox-config", "mailbox", "--teacher-timeout", "120"],
])
def test_teacher_options_cannot_silently_select_another_transport(tmp_path, flags):
    with pytest.raises(SystemExit) as error:
        parse_args(collection_cli(tmp_path) + flags)
    assert error.value.code == 2


@pytest.mark.parametrize("seconds", ["0", "-1", "181", "nan", "inf"])
def test_direct_teacher_timeout_is_bounded_before_simulator_start(tmp_path, seconds):
    with pytest.raises(SystemExit) as error:
        parse_args(collection_cli(tmp_path) + ["--teacher-codex", "--teacher-timeout", seconds])
    assert error.value.code == 2


def test_direct_teacher_rejects_empty_binary(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(collection_cli(tmp_path) + ["--teacher-codex", "--codex-binary", "  "])


def test_direct_teacher_defaults_and_explicit_executable_are_not_shell_commands(tmp_path):
    from codex_collection_teacher import CodexTeacher
    default = build_teacher(parse_args(collection_cli(tmp_path) + ["--teacher-codex"]))
    assert isinstance(default, CodexTeacher)
    assert default.codex_binary == "codex" and default.timeout_seconds == 120
    args = parse_args(collection_cli(tmp_path) + ["--teacher-codex", "--codex-binary",
                                                "/opt/codex runtime/codex", "--teacher-timeout", "180"])
    teacher = build_teacher(args)
    assert args.teacher_backend == "codex_local"
    assert teacher.codex_binary == "/opt/codex runtime/codex" and teacher.timeout_seconds == 180


@pytest.mark.parametrize("flags,module_name,class_name,expected", [
    (["--teacher-token-file", "token"], "codex_teacher_bridge", "RemoteTeacher",
     ("http://127.0.0.1:8766", Path("token"))),
    (["--teacher-token-file", "token", "--teacher-url", "http://127.0.0.1:12345"],
     "codex_teacher_bridge", "RemoteTeacher", ("http://127.0.0.1:12345", Path("token"))),
    (["--teacher-mailbox-config", "mailbox"], "codex_teacher_mailbox", "WorkerAossTeacher", (Path("mailbox"),)),
])
def test_existing_remote_teacher_cli_selects_only_its_backend(tmp_path, monkeypatch, flags,
                                                            module_name, class_name, expected):
    import importlib
    module = importlib.import_module(module_name)
    calls = []
    sentinel = object()
    def constructor(*args):
        calls.append(args)
        return sentinel
    monkeypatch.setattr(module, class_name, constructor)
    assert build_teacher(parse_args(collection_cli(tmp_path) + flags)) is sentinel
    assert calls == [expected]


def test_unknown_teacher_backend_fails_closed():
    with pytest.raises(ValueError, match="Unknown teacher backend"):
        build_teacher(SimpleNamespace(teacher_backend="unrecognized"))


@pytest.fixture
def real_lazy_policy(monkeypatch):
    """Execute the real client/wrapper definitions; replace only socket transport."""
    import ast
    from copy import deepcopy
    import logging
    import sys
    from typing import Any,Dict,Optional,Tuple
    from urllib.parse import urlparse
    root=Path(__file__).resolve().parents[1]/"BEHAVIOR-1K/OmniGibson/omnigibson/eval"
    state={"metadata":{"server_instance_token":"fixture-private-token","subtask_generation":True,
                       "subtask_action_conditioning":"task_and_subtask"},"connections":[]}
    class Connection:
        def __init__(self):self.sent=[];self.closed=False
        def recv(self,**kwargs):return state["metadata"]
        def send(self,value):self.sent.append(value)
        def close(self):self.closed=True
    def connect(*args,**kwargs):
        connection=Connection();state["connections"].append(connection);return connection
    ws=SimpleNamespace(sync=SimpleNamespace(client=SimpleNamespace(ClientConnection=Connection,connect=connect)),
        exceptions=SimpleNamespace(InvalidMessage=type("InvalidMessage",(Exception,),{})))
    scope=dict(Any=Any,Dict=Dict,Optional=Optional,Tuple=Tuple,th=SimpleNamespace(Tensor=object),
        Packer=lambda:SimpleNamespace(pack=lambda value:value),unpackb=lambda value:value,
        deepcopy=deepcopy,logger=logging.getLogger(__name__),logging=logging,websockets=ws,
        requests=SimpleNamespace(get=lambda *a,**k:SimpleNamespace(ok=True)),time=time,urlparse=urlparse,
        ACTION_CHUNK_REQUEST_KEY="__action_chunk_size__")
    for relative,name in [("utils/network_utils.py","WebsocketClientPolicy"),("policies.py","WebsocketPolicy")]:
        tree=ast.parse((root/relative).read_text())
        nodes=[node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==name]
        assert len(nodes)==1
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/relative),"exec"),scope)
    monkeypatch.setitem(sys.modules,"omnigibson.eval.policies",SimpleNamespace(WebsocketPolicy=scope["WebsocketPolicy"]))
    return state,scope["WebsocketPolicy"]


def test_actual_lazy_client_reset_handshake_precedes_metadata_read(real_lazy_policy):
    from collect_b1k_codex import connect_collection_policy
    state,policy_class=real_lazy_policy
    unused=policy_class(host="127.0.0.1",port=8920,expected_server_token="fixture-private-token")
    assert unused.policy.get_server_metadata() is None and state["connections"]==[]
    args=SimpleNamespace(host="127.0.0.1",port=8920,expected_server_token="fixture-private-token",
                         step_timeout=120,max_seconds=7200)
    policy,metadata=connect_collection_policy(args)
    assert metadata==state["metadata"] and len(state["connections"])==1
    assert state["connections"][0].sent==[{"reset":True}]
    policy.reset()
    assert len(state["connections"])==1 and state["connections"][0].sent==[{"reset":True}]*2


@pytest.mark.parametrize("metadata",[None,{},
    dict(server_instance_token="wrong",subtask_generation=True,subtask_action_conditioning="task_and_subtask"),
    dict(server_instance_token="fixture-private-token",subtask_generation=False,subtask_action_conditioning="task_and_subtask"),
    dict(server_instance_token="fixture-private-token",subtask_generation=True,subtask_action_conditioning="subtask_only")])
def test_actual_handshake_rejects_missing_identity_or_conditioning(real_lazy_policy,metadata):
    from collect_b1k_codex import connect_collection_policy
    state,_=real_lazy_policy;state["metadata"]=metadata
    args=SimpleNamespace(host="127.0.0.1",port=8920,expected_server_token="fixture-private-token",
                         step_timeout=120,max_seconds=7200)
    with pytest.raises((ValueError,RuntimeError)):connect_collection_policy(args)


@pytest.mark.parametrize("bad_server,rollout_error",[(True,False),(False,True),(False,False)])
def test_main_checks_real_handshake_before_scene_and_preserves_shutdown_exit_status(
        tmp_path,monkeypatch,real_lazy_policy,bad_server,rollout_error,capsys):
    import sys
    import collect_b1k_codex as collector
    state,_=real_lazy_policy;created=[];shutdown=[]
    args=parse_args(collection_cli(tmp_path)+["--teacher-codex"])
    args.expected_server_token="fixture-private-token"
    if bad_server:state["metadata"]["subtask_generation"]=False
    class SceneEvaluator:
        def __init__(self,cfg):
            assert len(state["connections"])==1 and state["connections"][0].sent==[{"reset":True}]
            created.append(True);self.policy=self.load_policy()
            self.env=SimpleNamespace(env_config={"rendering_frequency":30})
            assert self.policy.policy.get_server_metadata()==state["metadata"]
        def __enter__(self):return self
        def __exit__(self,*args):
            shutdown.append(True)
            if rollout_error:assert (tmp_path/"collector_failure.json").is_file()
            raise SystemExit(0)
    base=SimpleNamespace(parse_args=lambda argv:object(),gm=SimpleNamespace(),
                         seed_everything=lambda seed:seed,DEFAULT_EVAL_SEED=42,Evaluator=SceneEvaluator)
    monkeypatch.setattr(collector,"parse_args",lambda argv:args)
    monkeypatch.setattr(collector,"build_teacher",lambda args:object())
    monkeypatch.setitem(sys.modules,"eval_b1k_queue",SimpleNamespace(_load_base_module=lambda:base,
                        _build_evaluator_cfg=lambda *args:object()))
    def collect(evaluator,*args,**kwargs):
        assert evaluator.policy.policy.get_server_metadata() is not None
        if rollout_error:raise RuntimeError("never expose fixture-private-token")
        return {"steps":1,"success":False}
    monkeypatch.setattr(collector,"collect_attempt",collect)
    if bad_server or rollout_error:
        with pytest.raises(SystemExit) as stopped:collector.main([])
        assert stopped.value.code==1
        failure=json.loads((tmp_path/"collector_failure.json").read_text())
        assert failure["stage"]==("policy_handshake" if bad_server else "collection")
        assert failure["error_type"]==("ValueError" if bad_server else "RuntimeError")
        assert "fixture-private-token" not in json.dumps(failure)+capsys.readouterr().err
    else:assert collector.main([])=={"steps":1,"success":False}
    assert len(created)==len(shutdown)==(0 if bad_server else 1)


@pytest.mark.parametrize("completed,failure",[(False,None),(True,None),(True,"collector_failure.json"),(True,"attempt/failed.json")])
def test_real_shell_completion_guard_rejects_native_success_exit_after_failure(tmp_path,completed,failure):
    import os
    import subprocess
    source=(Path(__file__).parent/"run_codex_collection.sh").read_text()
    # Execute the real post-collector shell path after a simulated native exit(0).
    suffix=source[source.index("collector_status=$?"):]
    attempt=tmp_path/"attempt";attempt.mkdir()
    if completed:(attempt/"completed.json").write_text("{}")
    if failure:(tmp_path/failure).write_text("{}")
    result=subprocess.run(["bash","-c","true\n"+suffix],capture_output=True,text=True,
                          env={**os.environ,"CODEX_COLLECTION_OUTPUT":str(attempt),"SERVER_PID":str(os.getpid())})
    assert result.returncode==(0 if completed and failure is None else 1)


@pytest.mark.parametrize("invalid_response", [False, True])
def test_direct_teacher_uses_account_pinned_model_and_strict_evidence_flow(tmp_path, monkeypatch,
                                                                         invalid_response):
    from codex_collection_teacher import CodexTeacherError, request_identity
    monkeypatch.setenv("CODEX_API_KEY", "fixture-never-forwarded")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-never-forwarded")
    monkeypatch.setenv("CODEX_HOME", "/fixture/private-cluster-account")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:17890")
    teacher = build_teacher(parse_args(collection_cli(tmp_path) + ["--teacher-codex", "--codex-binary",
                                                                 "/fixture/codex", "--teacher-timeout", "120"]))
    calls = []
    def runner(argv, **kwargs):
        calls.append((argv, kwargs))
        assert not {"CODEX_API_KEY", "OPENAI_API_KEY"} & kwargs["env"].keys()
        assert kwargs["env"]["CODEX_HOME"] == "/fixture/private-cluster-account"
        assert kwargs["env"]["HTTPS_PROXY"] == "http://127.0.0.1:17890"
        assert argv[0] == "/fixture/codex"
        if argv[1:] == ["login", "status"]:
            return SimpleNamespace(returncode=0, stdout="Logged in using ChatGPT\n", stderr="")
        if argv[1:4] == ["mcp", "list", "--json"]:
            return SimpleNamespace(returncode=0, stdout="[]", stderr="")
        assert argv[1] == "exec" and kwargs.get("shell", False) is False
        assert argv[argv.index("--model") + 1] == "gpt-6.1-sol"
        assert 'model_reasoning_effort="medium"' in argv
        schema = json.loads(Path(argv[argv.index("--output-schema") + 1]).read_text())
        request_id = schema["properties"]["response_id"]["enum"][0]
        evidence = Path(kwargs["cwd"])
        saved = json.loads((evidence / "request.json").read_text())
        image_paths = [evidence / image["file"] for image in saved["images"]]
        assert request_identity(saved["request"], image_paths) == request_id
        response = {"response_id": "incorrect" if invalid_response else request_id,
                    "action": "continue", "subtask": "turn on the radio", "correction": None,
                    "reason": "fixture only"}
        Path(argv[argv.index("--output-last-message") + 1]).write_text(json.dumps(response))
        kwargs["stdout"].write(json.dumps({"type": "turn.completed", "usage": {
            "input_tokens": 20, "cached_input_tokens": 0, "output_tokens": 10}}) + "\n")
        return SimpleNamespace(returncode=0)
    teacher.runner = runner
    evaluator = Evaluator(terminal_step=1)
    if invalid_response:
        with pytest.raises(CodexTeacherError, match="response_validation"):
            run(tmp_path, evaluator, teacher)
        assert not evaluator.executed
        assert not (tmp_path / "attempt/completed.json").exists()
    else:
        result = run(tmp_path, evaluator, teacher)
        assert result["success"] and len(evaluator.executed) == 1
    assert len(calls) == 3  # Authentication, read-only MCP inventory, one generation; no retry.
    receipts = list((tmp_path / "attempt/teacher/step-00000000-decision-0000/decision").glob("*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text())
    assert receipt["status"] == ("failed" if invalid_response else "validated")
    assert receipt["authentication"] == "verified_chatgpt"
    assert receipt["requested_model"] == "gpt-6.1-sol"
    assert receipt["requested_reasoning_effort"] == "medium"


def test_native_correction_records_teacher_label_then_resumes_native_once(tmp_path):
    evaluator = Evaluator(terminal_step=7)
    evaluator.policy = original = NativePolicy()
    evaluator.robot = SimpleNamespace(evaluator=evaluator)
    teacher = NativeTeacher()
    choose = teacher.choose
    def decision(request, images, output):
        answer = choose(request, images, output)
        if request["step"] == 2:
            correction = correction_decision(2, first=2, label="teacher grip adjustment")
            answer.update({key:correction[key] for key in ("action", "subtask", "correction")})
        return answer
    teacher.choose = decision
    result = run(tmp_path, evaluator, teacher, collection_mode="native_with_corrections",
                 correction_factory=CorrectionFixture)
    assert result["success"] and [row["step"] for row in teacher.requests] == [2, 4, 6]
    assert teacher.requests[1]["previous_subtask"] == "teacher grip adjustment"
    assert teacher.requests[1]["current_native_subtask"] == "native turn dial"
    assert teacher.requests[1]["current_native_subtask_source"] == "generated"
    assert original.reset_count == 3 and original.forward_steps == [0, 1, 4, 5, 6]
    info = inspect_codex_collection_trajectory(tmp_path / "attempt/trajectory.hdf5")
    assert info["segments"] == [dict(start=0,end=1,subtask="native reach radio"),
        dict(start=1,end=2,subtask="native turn dial"),dict(start=2,end=4,subtask="teacher grip adjustment"),
        dict(start=4,end=7,subtask="native turn dial")]
    rows = [json.loads(line) for line in (tmp_path / "attempt/action_sources.jsonl").read_text().splitlines()]
    assert [row["source"] for row in rows] == ["policy_native_subtask"]*2 + ["teacher_cartesian_correction"]*2 + ["policy_native_subtask"]*3
    assert all(row["subtask_source"] == "provided" and row["decision_index"] == 0 for row in rows[2:4])
    with h5py.File(tmp_path / "attempt/trajectory.hdf5") as stream:
        assert stream["action"][:,0].tolist() == [0,1,102,103,4,5,6]


def test_native_rejected_correction_replans_same_step_without_reset_or_fake_frame(tmp_path):
    from codex_collection_actions import RecoverableCorrectionError
    evaluator = Evaluator(terminal_step=3);evaluator.policy = original = NativePolicy()
    evaluator.robot = SimpleNamespace(evaluator=evaluator)
    class RejectCorrection(CorrectionFixture):
        def next_action(self):
            raise RecoverableCorrectionError("fixture rejection", reason_code="ik_residual", diagnostics={"arm":"left", "residual":0.01})
    teacher = NativeTeacher();choose = teacher.choose
    def decision(request, images, output):
        answer = choose(request, images, output)
        if request["decision_index"] == 0:
            answer.update(action="correct", subtask="teacher adjustment", correction=correction_decision(2,first=2)["correction"])
        return answer
    teacher.choose = decision
    result = run(tmp_path, evaluator, teacher, collection_mode="native_with_corrections",
                 correction_factory=RejectCorrection)
    assert result["success"] and result["teacher_decisions"] == 2
    assert [(r["step"],r["decision_index"]) for r in teacher.requests] == [(2,0),(2,1)]
    assert teacher.requests[1]["last_correction_outcome"]["executed"] is False
    assert original.reset_count == 2 and original.forward_steps == [0,1,2]
    assert evaluator.steps == 3 and len(evaluator.executed) == 3
    assert sorted(path.name for path in (tmp_path / "attempt/teacher").iterdir()) == ["step-00000002-decision-0000","step-00000002-decision-0001"]
    rows = [json.loads(line) for line in (tmp_path / "attempt/action_sources.jsonl").read_text().splitlines()]
    assert rows[2]["decision_index"] == 1 and all(row["source"] == "policy_native_subtask" for row in rows)


def test_collection_mode_cli_default_and_opt_in(tmp_path):
    flags = collection_cli(tmp_path) + ["--teacher-codex"]
    assert parse_args(flags).collection_mode == "teacher_conditioned"
    assert parse_args(flags + ["--collection-mode", "native_with_corrections"]).collection_mode == "native_with_corrections"
    with pytest.raises(SystemExit):parse_args(flags + ["--collection-mode", "invalid"])


@pytest.mark.parametrize("hard_error", [False, True])
def test_native_rejection_budget_keeps_only_initial_executed_frames_and_hard_errors_fail(tmp_path, hard_error):
    from codex_collection_actions import CorrectionError, RecoverableCorrectionError
    evaluator = Evaluator(terminal_step=100);evaluator.policy = original = NativePolicy()
    evaluator.robot = SimpleNamespace(evaluator=evaluator)
    class RejectCorrection(CorrectionFixture):
        def next_action(self):
            if hard_error:raise CorrectionError("invalid robot state")
            raise RecoverableCorrectionError("fixture rejection",reason_code="ik_residual",
                                             diagnostics={"arm":"left","residual":.01})
    teacher=NativeTeacher();choose=teacher.choose
    def decision(request, images, output):
        answer=choose(request,images,output)
        answer.update(action="correct",subtask="teacher adjustment",correction=correction_decision(2,first=2)["correction"])
        return answer
    teacher.choose=decision
    if hard_error:
        with pytest.raises(CorrectionError,match="invalid robot state"):
            run(tmp_path,evaluator,teacher,collection_mode="native_with_corrections",max_decisions=32,
                correction_factory=RejectCorrection)
        assert not (tmp_path / "attempt/completed.json").exists()
        assert len(teacher.requests)==1
    else:
        result=run(tmp_path,evaluator,teacher,collection_mode="native_with_corrections",max_decisions=32,
                   correction_factory=RejectCorrection)
        assert result["steps"]==2 and result["teacher_decisions"]==32 and not result["success"]
        assert result["stop_reason"]=="teacher_decision_budget"
        assert [row["decision_index"] for row in teacher.requests]==list(range(32))
        assert all(row["step"]==2 for row in teacher.requests)
        assert len(list((tmp_path / "attempt/teacher").iterdir()))==32
        info=inspect_codex_collection_trajectory(tmp_path / "attempt/trajectory.hdf5",require_success=False)
        assert info["steps"]==2
    assert evaluator.steps==2 and len(evaluator.executed)==2 and original.reset_count==2
    with h5py.File(tmp_path / "attempt/trajectory.hdf5") as stream:
        assert stream["action"][:,0].tolist()==[0,1]


def test_native_direct_teacher_request_evidence_and_actual_frame_labels(tmp_path):
    from codex_collection_teacher import CodexTeacher,request_identity
    from test_codex_collection_teacher import FakeRunner
    runner=FakeRunner(response=lambda answer:{**answer,"subtask":"","correction":None})
    evaluator=Evaluator(terminal_step=3);evaluator.policy=NativePolicy()
    result=run(tmp_path,evaluator,CodexTeacher(runner=runner),collection_mode="native_with_corrections")
    assert result["success"] and result["teacher_decisions"]==1 and len(runner.calls)==3
    snapshot=tmp_path / "attempt/teacher/step-00000002-decision-0000"
    request=json.loads((snapshot / "request.json").read_text())
    identity=request_identity(request,[snapshot / f"camera-{i}.png" for i in range(3)])
    evidence=snapshot / "decision" / identity
    assert json.loads((evidence / "response.json").read_text())["subtask"]==""
    assert json.loads((evidence / "receipt.json").read_text())["status"]=="validated"
    rows=[json.loads(line) for line in (tmp_path / "attempt/action_sources.jsonl").read_text().splitlines()]
    assert rows[2]["request_id"]==identity and rows[2]["subtask"]=="native turn dial"
