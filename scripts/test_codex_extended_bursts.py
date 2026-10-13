"""CPU protocol and executed-loop tests; these never certify robot success."""
import copy
import json

import h5py
import numpy as np
import pytest

from codex_collection_actions import BoundedCartesianCorrection, EXTENDED_BURST_PROTOCOL
from codex_collection_teacher import RejectedTeacherCorrection
from codex_subtask_teacher import CodexSubtaskTeacher, request_identity
from test_codex_collection_actions import Robot, Backend, correction
from test_codex_collection_teacher import FakeRunner
from test_codex_subtask_teacher import inputs
from test_codex_subtask_collection import fixture_plan, FakeEvaluator, FakePolicy, FakeMonitor
import codex_subtask_collection as contract
from collect_b1k_codex_subtask import collect_attempt


def extended_request(request):
    robot, backend = Robot(), Backend()
    request.update(task=contract.CAMERA_TASK, fixed_subtask=contract.CAMERA_SUBTASK,
        previous_subtask=contract.CAMERA_SUBTASK, teacher_execution_protocol=EXTENDED_BURST_PROTOCOL,
        robot_control_state=BoundedCartesianCorrection(robot, backend=backend,
            execution_protocol=EXTENDED_BURST_PROTOCOL).snapshot(),
        gripper_execution={'contract':'robot_gripper_execution_v1'})
    return request


def wire_correction(**values):
    result=dict(arm='left',translation_direction=[0,0,-1],translation_distance_m=.003,
        rotation_axis=[0,0,0],rotation_angle_rad=0,gripper='open',steps=20)
    result.update(values)
    return result


def test_teacher_transports_twenty_step_request_without_shortening_actual_decision(inputs):
    request,images,output=inputs;extended_request(request)
    runner=FakeRunner(response=lambda wire:{**wire,'action':'correct','subtask':contract.CAMERA_SUBTASK,
        'correction':wire_correction()})
    decision=CodexSubtaskTeacher(runner=runner).choose(request,images,output)
    assert decision['correction']['steps']==20
    assert decision['correction']['delta_position_m']==[0.,0.,-.003]
    evidence=output/decision['request_id']
    assert json.loads((evidence/'response.raw.json').read_text())['correction']['steps']==20
    assert json.loads((evidence/'response.json').read_text())['correction']['steps']==20
    schema=json.loads((evidence/'response.schema.json').read_text())
    assert schema['properties']['correction']['anyOf'][0]['properties']['steps']['maximum']==20
    prompt=(evidence/'prompt.txt').read_text()
    assert 'Prefer 10--20 control steps' in prompt and 'one arm, 1--20 steps' in prompt
    assert '15--20 actual zero-motion close controls' in prompt
    legacy=copy.deepcopy(request);legacy.pop('teacher_execution_protocol')
    legacy['robot_control_state']={'contract':'r1pro_cartesian_correction_v1'}
    assert request_identity(legacy,images)!=decision['request_id']


@pytest.mark.parametrize('values', [
    {'translation_distance_m':.006}, {'gripper':'close'},
    {'rotation_axis':[1,0,0],'rotation_angle_rad':.01}, {'steps':21},
])
def test_teacher_audits_and_rejects_long_motion_outside_exact_extended_bounds(inputs,values):
    request,images,output=inputs;extended_request(request)
    runner=FakeRunner(response=lambda wire:{**wire,'action':'correct','subtask':contract.CAMERA_SUBTASK,
        'correction':wire_correction(**values)})
    with pytest.raises(RejectedTeacherCorrection):
        CodexSubtaskTeacher(runner=runner).choose(request,images,output)


def test_legacy_camera_does_not_silently_accept_long_burst(inputs):
    request,images,output=inputs
    request.update(task=contract.CAMERA_TASK,fixed_subtask=contract.CAMERA_SUBTASK,
        previous_subtask=contract.CAMERA_SUBTASK)
    runner=FakeRunner(response=lambda wire:{**wire,'action':'correct','subtask':contract.CAMERA_SUBTASK,
        'correction':wire_correction()})
    with pytest.raises(RejectedTeacherCorrection):CodexSubtaskTeacher(runner=runner).choose(request,images,output)


@pytest.mark.parametrize('change',['missing_request_flag','missing_feedback','changed_limit'])
def test_extended_request_requires_matching_execution_and_tactile_contracts(inputs,change):
    request,images,_=inputs;extended_request(request)
    if change=='missing_request_flag':request.pop('teacher_execution_protocol')
    elif change=='missing_feedback':request.pop('gripper_execution')
    else:request['robot_control_state']['limits']['max_steps']=25
    with pytest.raises(ValueError):request_identity(request,images)


def test_extended_plan_is_frozen_camera_only_and_requires_robot_feedback():
    old=fixture_plan(camera=True,teacher_feedback='robot_gripper_execution_v1')
    new=fixture_plan(camera=True,teacher_feedback='robot_gripper_execution_v1',
        teacher_execution=EXTENDED_BURST_PROTOCOL)
    assert old['fingerprint']!=new['fingerprint'] and new['budgets']==old['budgets']
    with pytest.raises(ValueError):fixture_plan(camera=True,teacher_execution=EXTENDED_BURST_PROTOCOL)
    with pytest.raises(ValueError):fixture_plan(teacher_execution=EXTENDED_BURST_PROTOCOL)


@pytest.fixture
def extended_loop(tmp_path,monkeypatch):
    import codex_collection_demonstrations
    import codex_subtask_qualification
    monkeypatch.setattr(codex_collection_demonstrations,'demonstration_context',lambda *a,**k:({},[]))
    # CPU fixture only, never evidence of actual paired qualification.
    monkeypatch.setattr(codex_subtask_qualification,'verify_qualification',lambda *a,**k:{'sha256':'0'*64})

    def run(*,touch_at=3,gripper='open',success_at=999):
        plan=fixture_plan(camera=True,teacher_feedback='robot_gripper_execution_v1',
            teacher_execution=EXTENDED_BURST_PROTOCOL,max_steps=32,max_decisions=2)
        class CameraPolicy(FakePolicy):
            def forward(self,obs):
                assert obs['subtask']==contract.CAMERA_SUBTASK
                self.model_calls+=1;self.current_subtask=contract.CAMERA_SUBTASK
                return np.zeros(23,dtype=np.float32)
        class Evaluator(FakeEvaluator):
            def step(self):
                result=super().step()
                self.robot.q[10:17]=self.robot_action[7:14]
                return result
        ev=Evaluator();ev.policy=CameraPolicy();ev.robot=Robot()
        class Monitor(FakeMonitor):
            def result(self):return {**super().result(),'predicate':plan['sample']['predicate']}
        monitor=Monitor(success_at=success_at)
        class Teacher:
            def __init__(self):self.requests=[]
            def choose(self,request,images,output):
                self.requests.append(copy.deepcopy(request));output.mkdir(parents=True)
                return dict(request_id=str(len(self.requests)),subtask=contract.CAMERA_SUBTASK,
                    action='correct' if len(self.requests)==1 else 'abort',reason='CPU fixture only',
                    correction=correction(steps=20,gripper=gripper,
                        delta_position_m=[.003,0,0] if gripper=='open' else [0,0,0])
                        if len(self.requests)==1 else None)
        teacher=Teacher()
        class Feedback:
            count=0
            def observe_executed_action(self,action):self.count+=1
            def snapshot(self,robot):
                return dict(contract='robot_gripper_execution_v1',executed_control_steps=self.count,
                    contact_object_identity_disclosed=False,target_grasp_truth_disclosed=False,
                    arms={'left':{'finger_environment_contact':[self.count>=touch_at,False]}})
        feedback=Feedback()
        def restore(*args):
            sid=plan['sample']['sample_id']
            monitor.setup_evidence['source_role_binding']=dict(sample_id=sid,raw_sha256='d'*64,
                source_state_sha256='e'*64,desired_predicate_truth_used=False)
            return dict(recorded_native_source=dict(contract=contract.NATIVE,sample_id=sid,
                source_start=520,raw_sha256='d'*64,selected_state_sha256='e'*64,
                recorded_native_fields_verified_before_policy=True,full_robot_joints_match=True,
                unrecorded_physics_caches_or_timers_verified=False,recorded_grasps=[],actual_restored_grasps=[]),
                initialization_replay_is_training_data=False)
        output=tmp_path/'attempt'
        receipt=collect_attempt(ev,teacher,plan,{}, {},output,demonstration_manifest='CPU fixture',
            correction_factory=lambda robot,**kw:BoundedCartesianCorrection(robot,backend=Backend(),**kw),
            restore_fn=restore,monitor_factory=lambda *a:monitor,feedback_factory=lambda ev:feedback)
        return receipt,teacher,ev,output
    return run


def test_contact_interrupt_records_only_executed_steps_and_immediately_reobserves(extended_loop):
    receipt,teacher,ev,out=extended_loop()
    assert receipt['result']['actual_teacher_control_steps']==ev.steps==3
    assert [q['step'] for q in teacher.requests]==[0,3]
    assert teacher.requests[1]['last_correction_outcome']['burst_interrupted'] is True
    events=[json.loads(line) for line in (out/'correction_interruptions.jsonl').read_text().splitlines()]
    assert len(events)==1 and events[0]['step']==2 and events[0]['next_teacher_step']==3
    assert events[0]['post_action_feedback']['executed_control_steps']==3
    contract.inspect_attempt(out,require_success=False)


@pytest.mark.parametrize('damage',['no_contact','wrong_request','wrong_counter'])
def test_interruption_audit_rejects_semantic_corruption_even_with_updated_digest(extended_loop,damage):
    _,_,_,out=extended_loop()
    path=out/'correction_interruptions.jsonl';event=json.loads(path.read_text())
    if damage=='no_contact':event['post_action_feedback']['arms']['left']['finger_environment_contact']=[False,False]
    elif damage=='wrong_request':event['request_id']='not-the-executed-request'
    else:event['post_action_feedback']['executed_control_steps']=20
    path.write_text(json.dumps(event)+'\n')
    with h5py.File(out/'trajectory.hdf5','r+') as f:
        result=json.loads(f.attrs['result']);result['correction_interruptions_sha256']=contract.file_sha256(path)
        f.attrs['result']=json.dumps(result)
    with pytest.raises(ValueError,match='Contact interruption'):
        contract.inspect_attempt(out,require_success=False)


def test_no_contact_executes_twenty_controls_without_extra_teacher_calls(extended_loop):
    receipt,teacher,ev,out=extended_loop(touch_at=999)
    assert receipt['result']['actual_teacher_control_steps']==ev.steps==20
    assert [q['step'] for q in teacher.requests]==[0,20]
    assert not (out/'correction_interruptions.jsonl').read_text()


def test_close_settling_is_not_interrupted_by_finger_contact(extended_loop):
    receipt,teacher,ev,out=extended_loop(touch_at=1,gripper='close')
    assert receipt['result']['actual_teacher_control_steps']==ev.steps==20
    assert [q['step'] for q in teacher.requests]==[0,20]
    with h5py.File(out/'trajectory.hdf5') as f:assert np.all(f['action'][:,14]==-1)
    assert not (out/'correction_interruptions.jsonl').read_text()


def test_long_close_stops_when_existing_monitor_confirms_three_steps(extended_loop):
    receipt,teacher,ev,out=extended_loop(touch_at=1,gripper='close',success_at=18)
    assert receipt['result']['success'] and ev.steps==18 and len(teacher.requests)==1
    assert receipt['result']['monitor']['dwell']==3
    contract.inspect_attempt(out,require_success=True)
