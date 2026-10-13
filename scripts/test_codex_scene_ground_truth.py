"""CPU privileged-teacher tests. Fixtures cannot certify simulator success."""
import copy
from enum import IntEnum
import json
from types import SimpleNamespace
import sys

import h5py
import numpy as np
import pytest

import codex_scene_ground_truth as gt
from codex_subtask_collection import validate_plan, file_sha256
from test_codex_robot_guidance import Robot, state, plan, rehash, guided_loop
from test_codex_subtask_teacher import inputs


class Grasp(IntEnum):
    UNKNOWN=0
    TRUE=1
    FALSE=-1


def install_scene(evaluator, monkeypatch):
    monkeypatch.setitem(sys.modules, 'omnigibson.controllers', SimpleNamespace(IsGraspingState=Grasp))
    target=SimpleNamespace(name='digital_camera_87', link_prim_paths=['/camera/body'],
        get_position_orientation=lambda:([1,3,.3],[0,0,0,1]),
        aabb=([.9,2.8,.2],[1.1,3.2,.4]))
    evaluator.env.scene=SimpleNamespace(object_registry=lambda key,name:target if name==target.name else None)
    robot=evaluator.robot
    robot.finger_links={arm:[SimpleNamespace(prim_path=f'/{arm}/{i}',
        get_position_orientation=lambda i=i:([1,2.95+i*.1,.3],[0,0,0,1])) for i in range(2)]
        for arm in ('left','right')}
    # Touching another surface is deliberately not a target contact.
    robot._find_gripper_contacts=lambda arm:(None, {'/floor/body':{f'/{arm}/0'}})
    robot._find_gripper_raycast_collisions=lambda arm:{'/floor/body'}
    robot.is_grasping=lambda **kw:Grasp.FALSE
    return target


def setup(monkeypatch):
    p=plan();p['teacher_scene_ground_truth']=dict(contract=gt.CONTRACT,targets=['digital_camera_87']);rehash(p)
    ev=SimpleNamespace(robot=Robot(),env=SimpleNamespace())
    target=install_scene(ev,monkeypatch)
    request=dict(task=p['sample']['task'],sample_id=p['sample']['sample_id'],attempt_id=p['attempt_id'],
        step=10,decision_index=2,robot_control_state=state(),
        gripper_execution=dict(arms={a:dict(finger_environment_contact=[True,False]) for a in ('left','right')}))
    request['scene_ground_truth_protocol']=gt.CONTRACT
    request['scene_ground_truth']=gt.capture(ev,p,request)
    return p,ev,target,request


def test_live_target_position_and_aabb_use_rotated_translated_control_frame(monkeypatch):
    p,_,_,r=setup(monkeypatch);o=r['scene_ground_truth']['objects'][0]
    assert np.allclose(o['pose_in_robot_base']['position_m'],[1,0,.3])
    assert np.allclose(o['aabb_center_in_robot_base_m'],[1,0,.3])
    assert np.allclose(o['arms']['left']['eef_to_aabb_center_m'],[0,0,0])
    assert np.allclose(o['world_aabb_corners_in_robot_base_m'][0],[.8,.1,.2])
    assert o['arms']['left']['target_finger_contact']==[False,False]
    assert o['arms']['left']['native_target_grasped'] is False
    gt.validate_request(r,selection=p['teacher_scene_ground_truth'],plan_fingerprint=p['fingerprint'])


def test_native_target_contact_does_not_confuse_floor_contact(monkeypatch):
    p,ev,_,r=setup(monkeypatch)
    ev.robot._find_gripper_contacts=lambda arm:(None,{'/camera/body':{f'/{arm}/0'},'/floor/body':{f'/{arm}/1'}})
    ev.robot._find_gripper_raycast_collisions=lambda arm:{'/camera/body'}
    ev.robot.is_grasping=lambda **kw:Grasp.TRUE
    o=gt.capture(ev,p,r)['objects'][0]
    assert o['arms']['left']['target_finger_contact']==[True,False]
    assert o['arms']['left']['target_raycast_hit'] is True
    assert o['arms']['left']['native_target_grasped'] is True


@pytest.mark.parametrize('damage',['missing','unknown_grasp','nan_pose','bad_box'])
def test_live_gt_fails_closed_for_missing_or_unobservable_state(monkeypatch,damage):
    p,ev,target,r=setup(monkeypatch)
    if damage=='missing':ev.env.scene.object_registry=lambda *a:None
    elif damage=='unknown_grasp':ev.robot.is_grasping=lambda **kw:Grasp.UNKNOWN
    elif damage=='nan_pose':target.get_position_orientation=lambda:([float('nan'),0,0],[0,0,0,1])
    else:target.aabb=([1,1,1],[0,0,0])
    with pytest.raises(ValueError):gt.capture(ev,p,r)


@pytest.mark.parametrize('damage',['target','step','plan','base_frame','object_transform','box_transform',
                                  'eef_error','finger_transform','contact','native_type','policy_input'])
def test_frozen_gt_rejects_wrong_identity_frame_and_derived_geometry(monkeypatch,damage):
    p,_,_,r=setup(monkeypatch);v=r['scene_ground_truth'];o=v['objects'][0];a=o['arms']['left']
    if damage=='target':o['name']='wrong_camera'
    elif damage=='step':v['step']+=1
    elif damage=='plan':v['plan_fingerprint']='f'*64
    elif damage=='base_frame':v['frame']='world'
    elif damage=='object_transform':o['pose_in_robot_base']['position_m'][0]+=.01
    elif damage=='box_transform':o['world_aabb_corners_in_robot_base_m'][0][0]+=.01
    elif damage=='eef_error':a['eef_to_aabb_center_m'][0]+=.01
    elif damage=='finger_transform':a['finger_link_positions_in_robot_base_m'][0][0]+=.01
    elif damage=='contact':a['target_finger_contact'][1]=True
    elif damage=='native_type':a['native_target_grasped']=1
    else:v['policy_input']=True
    with pytest.raises(ValueError):
        gt.validate_request(r,selection=p['teacher_scene_ground_truth'],plan_fingerprint=p['fingerprint'])


def test_plan_binds_privilege_to_the_exact_frozen_training_target():
    p=plan();p['teacher_scene_ground_truth']=dict(contract=gt.CONTRACT,targets=['digital_camera_87']);rehash(p)
    validate_plan(p)
    p['teacher_scene_ground_truth']['targets']=['another_camera'];rehash(p)
    with pytest.raises(ValueError):validate_plan(p)


def test_gt_checks_live_controller_frame_before_disclosing_target_geometry(monkeypatch):
    p,ev,_,r=setup(monkeypatch)
    r['robot_control_state']['arms']['left']['eef_position_m'][0]+=.02
    with pytest.raises(ValueError,match='actual correction EEF frame'):gt.capture(ev,p,r)


def test_gt_never_enters_the_fixed_policy_observations(guided_loop):
    receipt,teacher,p,_,directory=guided_loop(privileged=True,continue_policy=True)
    assert receipt['result']['actual_model_control_steps']==32
    assert receipt['result']['actual_teacher_control_steps']==0
    assert len(teacher.policy_observations)==32
    assert all(obs=={'subtask':p['sample']['subtask']} for obs in teacher.policy_observations)
    with h5py.File(directory/'trajectory.hdf5') as f:
        metadata=json.loads(f.attrs['metadata'])
        assert 'scene_ground_truth' not in f
    gt.audit_attempt(directory,p,metadata,receipt['result'])


def test_teacher_prompt_and_request_identity_explicitly_include_privileged_truth(monkeypatch,inputs):
    from codex_subtask_teacher import CodexSubtaskTeacher, request_identity
    from test_codex_collection_teacher import FakeRunner
    from codex_subtask_collection import CAMERA_TASK,CAMERA_SUBTASK
    p,_,_,observed=setup(monkeypatch);r,images,out=inputs
    r.update(observed,task=CAMERA_TASK,fixed_subtask=CAMERA_SUBTASK,previous_subtask=CAMERA_SUBTASK)
    r['gripper_execution']['contract']='robot_gripper_execution_v1'
    legacy=copy.deepcopy(r);legacy.pop('scene_ground_truth');legacy.pop('scene_ground_truth_protocol')
    runner=FakeRunner(response=lambda wire:{**wire,'action':'continue','subtask':CAMERA_SUBTASK,'correction':None})
    decision=CodexSubtaskTeacher(runner=runner).choose(r,images,out)
    evidence=out/decision['request_id'];prompt=(evidence/'prompt.txt').read_text()
    assert 'CURRENT simulator truth' in prompt and 'The policy receives no GT.' in prompt
    assert 'or assume access to object ground truth' not in prompt
    assert request_identity(legacy,images)!=decision['request_id']
    frozen=json.loads((evidence/'request.json').read_text())
    assert frozen['request']['scene_ground_truth']==r['scene_ground_truth']


@pytest.mark.parametrize('reject',[False,True])
def test_actual_collection_loop_records_gt_even_for_zero_step_rejection(guided_loop,reject):
    receipt,teacher,p,actions,directory=guided_loop(privileged=True,reject=reject)
    with h5py.File(directory/'trajectory.hdf5') as f:
        metadata=json.loads(f.attrs['metadata'])
    gt.audit_attempt(directory,p,metadata,receipt['result'])
    assert [q['step'] for q in teacher.requests]==[0,0 if reject else 3]
    assert all(q['scene_ground_truth']['policy_input'] is False for q in teacher.requests)
    assert receipt['result']['scene_ground_truth_sha256']==file_sha256(directory/'scene_ground_truth.jsonl')
    assert receipt['training_eligible'] is False
    # A request can be locally plausible yet differ from its separately frozen snapshot.
    folder=sorted((directory/'teacher').glob('step-*'))[0]
    q=json.loads((folder/'request.json').read_text());q['scene_ground_truth']['objects'][0]['arms']['left']['native_target_grasped']=True
    (folder/'request.json').write_text(json.dumps(q))
    with pytest.raises(ValueError,match='snapshot differs'):
        gt.audit_attempt(directory,p,metadata,receipt['result'])


def test_legacy_export_cannot_silently_gain_privileged_gt(guided_loop):
    receipt,_,p,_,directory=guided_loop()
    with h5py.File(directory/'trajectory.hdf5') as f:metadata=json.loads(f.attrs['metadata'])
    gt.audit_attempt(directory,p,metadata,receipt['result'])
    (directory/'scene_ground_truth.jsonl').write_text('{}\n')
    with pytest.raises(ValueError,match='Unplanned'):gt.audit_attempt(directory,p,metadata,receipt['result'])
