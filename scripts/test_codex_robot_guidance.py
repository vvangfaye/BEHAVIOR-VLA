"""CPU frame/provenance tests; synthetic poses never prove robot success."""
import copy
import math
from types import SimpleNamespace

import numpy as np
import pytest

import codex_robot_guidance as guidance
from codex_subtask_collection import validate_plan, HISTORY
from test_codex_subtask_collection import fixture_plan, rehash


def plan():
    p = fixture_plan(camera=True, teacher_feedback='robot_gripper_execution_v1')
    sid = p['sample']['sample_id']
    p['source_plan']['source_protocol_by_sample'][sid] = HISTORY
    p['source_plan']['prefix_actions_sha256'] = {sid:'a'*64}
    p['teacher_robot_guidance'] = dict(contract=guidance.CONTRACT, frames=[1427,1442,1457],
                                      demonstration_fingerprint='a'*64)
    rehash(p)
    validate_plan(p)
    return p


class Robot:
    def __init__(self):
        self.base = ([1,2,0], [0,0,math.sqrt(.5),math.sqrt(.5)])
        self.world = ([1,3,.3], self.base[1])
        self.eef_links = {arm:SimpleNamespace(get_position_orientation=lambda:self.world)
                          for arm in ('left','right')}
    def get_position_orientation(self):
        return self.base


def state():
    arm = dict(eef_link='eef', eef_position_m=[1,0,.3], eef_quaternion_xyzw=[0,0,0,1],
               joint_position_rad=[0]*7, finger_joint_position_m=[.05,.05])
    return dict(frame='robot_base_footprint', arms={a:copy.deepcopy(arm) for a in ('left','right')})


def receipt(p=None):
    p = p or plan()
    anchors = {}
    robot = Robot()
    for f in p['teacher_robot_guidance']['frames']:
        anchor = guidance.capture_robot_anchor(robot, f)
        anchor['recorded_state_sha256'] = 'b'*64
        for arm in anchor['arms'].values():
            arm.update(position_m=[0,3,.5], quaternion_xyzw=[0,0,0,1])
        anchors[f] = anchor
    return guidance.build_reference_receipt(dict(sha256='d'*64,robot_reference_anchors=anchors), p)


def request():
    control = state()
    return dict(task='attach_a_camera_to_a_tripod', robot_guidance_protocol=guidance.CONTRACT,
        robot_control_state=control, gripper_execution={'contract':'robot_gripper_execution_v1'},
        robot_reference_guidance=guidance.guidance_from_anchors(receipt(),Robot(),control,
                                                               receipt_sha256='c'*64),
        previous_correction_motion=None,recent_decisions=[],step=10)


def test_original_world_reference_uses_current_rotated_and_translated_base():
    q = request()
    left = q['robot_reference_guidance']['references'][0]['arms']['left']
    assert np.allclose(left['reference_position_m'], [1,1,.5])
    assert np.allclose(left['position_error_m'], [0,1,.2])
    assert np.allclose(left['rotation_error_axis_angle_rad'], [0,0,-math.pi/2])
    guidance.validate_teacher_guidance(q)


def test_live_world_link_and_adapter_frame_must_agree():
    control = state(); control['arms']['left']['eef_position_m'][0] += .02
    with pytest.raises(ValueError, match='disagrees'):
        guidance.guidance_from_anchors(receipt(),Robot(),control,receipt_sha256='c'*64)


@pytest.mark.parametrize('angle',[0, .001, math.pi/2, math.pi-1e-7])
def test_quaternion_sign_does_not_change_shortest_rotation_error(angle):
    q = [0,math.sin(angle/2),0,math.cos(angle/2)]
    assert np.allclose(guidance.rotation_error(q,[0,0,0,1]),[0,angle,0])
    assert np.allclose(guidance.rotation_error([-v for v in q],[0,0,0,1]),[0,angle,0])


def correction():
    return dict(arm='left',delta_position_m=[.002,0,0],delta_rotation_axis_angle_rad=[0,0,0],
                gripper='open',steps=20)


def test_motion_audit_accepts_one_ulp_cross_platform_derived_roundoff():
    expected=guidance.burst_motion(state(),state(),correction(),start_step=10,end_step=13,
                                  decision_index=2,request_id='f'*64)
    expected['translation_projection_ratio']=0.7289134271326679
    actual=copy.deepcopy(expected);actual['translation_projection_ratio']=0.728913427132668
    assert actual!=expected and guidance.motion_matches(actual,expected)


@pytest.mark.parametrize('damage',['ratio','vector','nan','count_type','count','pose','request_id'])
def test_motion_roundoff_tolerance_never_weakens_controls_identity_or_pose_binding(damage):
    expected=guidance.burst_motion(state(),state(),correction(),start_step=10,end_step=13,
                                  decision_index=2,request_id='f'*64)
    actual=copy.deepcopy(expected)
    if damage=='ratio':actual['translation_projection_ratio']+=1e-7
    elif damage=='vector':actual['achieved_translation_m'][0]+=1e-7
    elif damage=='nan':actual['translation_projection_ratio']=float('nan')
    elif damage=='count_type':actual['executed_controls']=3.0
    elif damage=='count':actual['executed_controls']=4
    elif damage=='pose':actual['end_eef_pose']['eef_position_m'][0]+=1e-15
    else:actual['request_id']='a'*64
    assert not guidance.motion_matches(actual,expected)


@pytest.mark.parametrize('executed',[0,3,20])
def test_feedback_counts_emitted_controls_and_measures_underachieved_motion(executed):
    before,after = state(),state()
    after['arms']['left']['eef_position_m'][0] += .0005*executed
    result = guidance.burst_motion(before,after,correction(),start_step=10,end_step=10+executed,
                                    decision_index=2,request_id='f'*64)
    assert result['requested_controls']==20 and result['executed_controls']==executed
    assert np.allclose(result['requested_translation_m'],[.002*executed,0,0])
    assert np.allclose(result['achieved_translation_m'],[.0005*executed,0,0])
    assert result['translation_projection_ratio'] == pytest.approx(.25) if executed else result['translation_projection_ratio'] is None


def test_motion_is_bound_to_previous_correction_and_current_pose():
    q=request(); before=state();q['robot_control_state']['arms']['left']['eef_position_m'][0]+=.001
    previous=dict(action='correct',step=7,decision_index=2,request_id='f'*64,correction=correction())
    q['recent_decisions']=[previous]
    q['robot_reference_guidance']=guidance.guidance_from_anchors(receipt(),Robot(),state(),receipt_sha256='c'*64)
    # Keep the actual control pose unchanged; represent an earlier pose 1 mm behind.
    q['robot_control_state']=state();before['arms']['left']['eef_position_m'][0]-=.001
    q['previous_correction_motion']=guidance.burst_motion(before,q['robot_control_state'],correction(),
        start_step=7,end_step=10,decision_index=2,request_id='f'*64)
    guidance.validate_teacher_guidance(q)
    q['previous_correction_motion']['achieved_translation_m'][0]+=.01
    with pytest.raises(ValueError,match='differs'):
        guidance.validate_teacher_guidance(q)


@pytest.mark.parametrize('damage',['wrong_frame','wrong_error','object_pose','missing_flag','target_truth'])
def test_teacher_rejects_frame_error_or_extra_truth_fields(damage):
    q=request();g=q['robot_reference_guidance']
    if damage=='wrong_frame':g['frame']='world'
    elif damage=='wrong_error':g['references'][0]['arms']['left']['position_error_m'][0]=5
    elif damage=='object_pose':g['target_object_pose']=[0,0,0]
    elif damage=='missing_flag':q.pop('robot_guidance_protocol')
    else:g['object_ground_truth_disclosed']=True
    with pytest.raises(ValueError):guidance.validate_teacher_guidance(q)


@pytest.mark.parametrize('damage',['raw','plan','frame','object','nan'])
def test_anchor_receipt_cannot_be_relabelled_by_rehashing(damage):
    p=plan();r=receipt(p)
    if damage=='raw':r['source_raw_sha256']='e'*64
    elif damage=='plan':r['plan_fingerprint']='e'*64
    elif damage=='frame':r['anchors'][0]['frame']=1600
    elif damage=='object':r['anchors'][0]['object_pose']=[0,0,0]
    else:r['anchors'][0]['arms']['left']['position_m'][0]=float('nan')
    with pytest.raises(ValueError):
        r['fingerprint']=guidance.digest({k:v for k,v in r.items() if k!='fingerprint'})
        guidance.validate_reference_receipt(r,p)


@pytest.mark.parametrize('damage',['out_of_interval','duplicate','wrong_contract','native_source','no_feedback'])
def test_plan_requires_bound_historical_training_reference(damage):
    p=plan();s=p['teacher_robot_guidance']
    if damage=='out_of_interval':s['frames']=[1679]
    elif damage=='duplicate':s['frames']=[1427,1427]
    elif damage=='wrong_contract':s['contract']='unverified'
    elif damage=='native_source':p['source_plan']['source_protocol_by_sample'][p['sample']['sample_id']]='recorded_native_state_v1'
    else:p.pop('teacher_feedback')
    rehash(p)
    with pytest.raises(ValueError):validate_plan(p)


@pytest.fixture
def guided_loop(tmp_path, monkeypatch):
    import json
    import collect_b1k_codex_subtask as collector
    import codex_collection_demonstrations
    import codex_subtask_qualification
    from codex_collection_actions import BoundedCartesianCorrection, EXTENDED_BURST_PROTOCOL
    from test_codex_collection_actions import Robot as ArmRobot, Backend
    from test_codex_subtask_collection import FakeEvaluator, FakeMonitor
    # Only a loop fixture: no simulator success or real qualification is claimed.
    monkeypatch.setattr(codex_subtask_qualification, 'verify_qualification', lambda *a,**k:{'sha256':'0'*64})
    monkeypatch.setattr(collector, 'inspect_attempt', lambda *a,**k:{'content_sha256':'0'*64})

    def run(*, reject=False, touch_at=3, privileged=False, continue_policy=False):
        p=plan();p['teacher_execution']=EXTENDED_BURST_PROTOCOL
        if privileged:
            from codex_scene_ground_truth import CONTRACT as GT_CONTRACT
            p['teacher_scene_ground_truth'] = dict(contract=GT_CONTRACT, targets=['digital_camera_87'])
        p['budgets'].update(max_steps=32,max_decisions=2);rehash(p)
        source=receipt(p);qual=tmp_path/'qualification';qual.mkdir()
        (qual/'robot_reference.json').write_text(json.dumps(source))
        monkeypatch.setattr(codex_collection_demonstrations,'demonstration_context',lambda *a,**k:(
            {'demonstration_fingerprint':p['teacher_robot_guidance']['demonstration_fingerprint']},[]))
        backend=Backend()
        if reject:backend.solve=lambda **kwargs:kwargs['q']+.2
        class Evaluator(FakeEvaluator):
            def step(self):
                result=super().step()
                self.robot.q[10:17]=self.robot_action[7:14]
                return result
        ev=Evaluator();ev.robot=ArmRobot()
        policy_observations=[]
        if continue_policy:
            class CameraPolicy:
                current_subtask_source='provided'
                current_subtask=p['sample']['subtask']
                def reset(self):pass
                def forward(self,obs):
                    policy_observations.append(copy.deepcopy(obs))
                    return np.zeros(23,dtype=np.float32)
            ev.policy=CameraPolicy()
        ev.robot.get_position_orientation=lambda:([0,0,0],[0,0,0,1])
        ev.robot.eef_links={arm:SimpleNamespace(get_position_orientation=lambda arm=arm:
            (ev.robot.q[ev.robot.arm_control_idx[arm]][:3],[0,0,0,1])) for arm in ('left','right')}
        if privileged:
            from test_codex_scene_ground_truth import install_scene
            install_scene(ev, monkeypatch)
        class Feedback:
            count=0
            def observe_executed_action(self,action):self.count+=1
            def snapshot(self,robot):return dict(contract='robot_gripper_execution_v1',
                executed_control_steps=self.count,arms={'left':{'finger_environment_contact':[self.count>=touch_at,False]}})
        class Teacher:
            def __init__(self):self.requests=[]
            def choose(self,request,images,output):
                guidance.validate_teacher_guidance(request)
                self.requests.append(copy.deepcopy(request));rid=str(len(self.requests))
                first_action='continue' if continue_policy else 'correct'
                d=dict(request_id=rid,subtask=p['sample']['subtask'],action=first_action if rid=='1' else 'abort',
                       correction=correction() if rid=='1' and not continue_policy else None,reason='CPU fixture')
                folder=output/rid;folder.mkdir(parents=True)
                (folder/'response.json').write_text(json.dumps(d))
                (folder/'receipt.json').write_text(json.dumps({'usage':None}))
                return d
        teacher=Teacher();teacher.policy_observations=policy_observations
        monitor=FakeMonitor(success_at=999);feedback=Feedback()
        result=collector.collect_attempt(ev,teacher,p,{}, {},tmp_path/'attempt',
            demonstration_manifest='CPU fixture',
            correction_factory=lambda robot,**kw:BoundedCartesianCorrection(robot,backend=backend,**kw),
            restore_fn=lambda *a:{'initialization_replay_is_training_data':False},
            monitor_factory=lambda *a:monitor,feedback_factory=lambda *a:feedback)
        actions=[json.loads(line) for line in (tmp_path/'attempt/action_sources.jsonl').read_text().splitlines()]
        guidance.audit_attempt_guidance(tmp_path/'attempt',p,actions)
        return result,teacher,p,actions,tmp_path/'attempt'
    return run


@pytest.mark.parametrize('reject',[False,True])
def test_live_loop_feedback_handles_contact_interrupt_and_zero_step_ik_rejection(guided_loop,reject):
    result,teacher,p,actions,_=guided_loop(reject=reject)
    count=0 if reject else 3
    motion=teacher.requests[1]['previous_correction_motion']
    assert [q['step'] for q in teacher.requests]==[0,count]
    assert motion['executed_controls']==count and motion['requested_controls']==20
    assert np.allclose(motion['requested_translation_m'],[count*.002,0,0])
    assert len(actions)==count==result['result']['actual_teacher_control_steps']
    assert result['training_eligible'] is False and result['result']['success'] is False


@pytest.mark.parametrize('damage',['anchor_transform','motion_start_pose','motion_controls','source_hash'])
def test_export_audit_rejects_forged_motion_or_reference_even_if_locally_plausible(guided_loop,damage):
    import json
    _,teacher,p,actions,directory=guided_loop()
    f=sorted((directory/'teacher').glob('step-*'))[1]/'request.json'
    q=json.loads(f.read_text())
    if damage=='anchor_transform':
        arm=q['robot_reference_guidance']['references'][0]['arms']['left']
        arm['reference_position_m'][0]+=.01;arm['position_error_m'][0]+=.01
    elif damage=='source_hash':q['robot_reference_guidance']['source_receipt_sha256']='f'*64
    elif damage=='motion_controls':q['previous_correction_motion']['executed_controls']=20
    else:
        q['previous_correction_motion']['start_eef_pose']['eef_position_m'][0]-=.01
        m=q['previous_correction_motion'];previous=q['recent_decisions'][-1]
        q['previous_correction_motion']=guidance.burst_motion(
            dict(frame=m['frame'],arms={m['arm']:m['start_eef_pose']}),q['robot_control_state'],
            previous['correction'],start_step=previous['step'],end_step=q['step'],
            decision_index=previous['decision_index'],request_id=previous['request_id'])
    f.write_text(json.dumps(q))
    with pytest.raises(ValueError):guidance.audit_attempt_guidance(directory,p,actions)
