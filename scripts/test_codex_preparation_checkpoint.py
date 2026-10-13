import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from codex_preparation_checkpoint import save_preparation, load_preparation, reuse_qualification, sha


def plan():
    return dict(fingerprint='f'*64, checkpoint_identity={'checkpoint_step':1688977},
        policy_seed=9, sample={'sample_id':'camera-start'}, budgets={'max_steps':2048})


def state():
    return {0:{'registry':{'object_registry':{'robot':{
        'joint_pos':np.arange(28,dtype=np.float32),
        'ag_obj_constraint_params':{'left':None, 'right':None},
        'root_link':{'pos':np.array([1,2,3],dtype=np.float32)}}}}}}


def make_snapshot(tmp_path):
    states={'camera-start':{'state':state(), 'state_format':'named_simulator_state_v1',
        'history_counter_state':{'_ag_grasp_counter':{'left':None,'right':np.int64(2)}},
        'transition_steps':(4,9),'optional_deadline':float('inf')}}
    cache={'sha256':'a'*64, 'history_identity_grasps':{520:[]},
        'history_recovery_protocol':'history', 'robot_reference_anchors':{1427:{'x':[1.,2.]}},
        'preparation_runtime_signature':{'physics':'b'*64}}
    result=save_preparation(tmp_path/'checkpoint',plan(),states,cache)
    return states,cache,result


def test_complete_nested_scene_and_native_counters_survive_durable_reload(tmp_path):
    original,cache,(states,restored,proof)=make_snapshot(tmp_path)
    del states,restored
    states,restored,loaded=load_preparation(tmp_path/'checkpoint',plan(),cache['sha256'],runtime_signature=cache['preparation_runtime_signature'])
    robot=states['camera-start']['state'][0]['registry']['object_registry']['robot']
    np.testing.assert_array_equal(robot['joint_pos'],np.arange(28,dtype=np.float32))
    assert robot['joint_pos'].dtype==np.float32
    assert states['camera-start']['transition_steps']==(4,9)
    assert states['camera-start']['optional_deadline']==float('inf')
    assert isinstance(states['camera-start']['history_counter_state']['_ag_grasp_counter']['right'],np.int64)
    assert restored['history_identity_grasps']=={520:[]}
    assert restored['robot_reference_anchors']==cache['robot_reference_anchors']
    assert proof==loaded and loaded['readback_verified']
    with pytest.raises(FileExistsError):save_preparation(tmp_path/'checkpoint',plan(),original,cache)


@pytest.mark.parametrize('change',['seed','checkpoint','plan','raw','runtime','archive','ready'])
def test_wrong_experiment_and_corrupt_snapshot_fail_closed(tmp_path,change):
    _,cache,_=make_snapshot(tmp_path)
    p=copy.deepcopy(plan());raw=cache['sha256'];signature=cache['preparation_runtime_signature']
    if change=='seed':p['policy_seed']=10
    if change=='checkpoint':p['checkpoint_identity']['checkpoint_step']=1
    if change=='plan':p['fingerprint']='0'*64
    if change=='raw':raw='0'*64
    if change=='runtime':signature={}
    if change=='archive':
        f=tmp_path/'checkpoint/full-scene-state.npz';b=bytearray(f.read_bytes());b[50]^=1;f.write_bytes(b)
    if change=='ready':(tmp_path/'checkpoint/READY.json').write_text('{}')
    with pytest.raises(ValueError):load_preparation(tmp_path/'checkpoint',p,raw,runtime_signature=signature)


def qualification_source(tmp_path):
    source=tmp_path/'old';(source/'qualification').mkdir(parents=True);(source/'attempt').mkdir()
    code=tmp_path/'code';code.mkdir();(code/'physics.py').write_text('original physics')
    status=dict(state='failed',plan_fingerprint=plan()['fingerprint'],checkpoint_identity=plan()['checkpoint_identity'],source_files_sha256={'physics.py':sha(code/'physics.py')})
    (source/'status.json').write_text(json.dumps(status));(source/'attempt/failed.json').write_text('{"steps":0}')
    q=dict(kind='codex_subtask_qualification_v1',plan_fingerprint=plan()['fingerprint'],checkpoint_identity=plan()['checkpoint_identity'],policy_seed=9,baseline_teacher_calls=0,qualified=True,calibration={'success':True},baseline=dict(metric_eligible=True,success=False,terminated=False,truncated=False,failure_reason='sample_timeout',steps=2048))
    (source/'qualification/qualification.json').write_text(json.dumps(q))
    (source/'qualification/calibration.json').write_text('{"success":true}')
    return source,code


def test_qualification_reuse_keeps_original_bytes_and_performs_no_controls(tmp_path):
    source,code=qualification_source(tmp_path);before=(source/'qualification/qualification.json').read_bytes()
    proof=reuse_qualification(source,tmp_path/'new-qualification',plan(),code)
    assert (tmp_path/'new-qualification/qualification.json').read_bytes()==before
    assert (source/'qualification/qualification.json').read_bytes()==before
    assert proof['baseline_controls_executed']==proof['calibration_controls_executed']==0
    assert proof['source_baseline_steps']==2048
    with pytest.raises(FileExistsError):reuse_qualification(source,tmp_path/'new-qualification',plan(),code)


@pytest.mark.parametrize('change',['physics','running','partial','successful','seed','symlink','steps'])
def test_reuse_rejects_incompatible_or_unfinished_evidence(tmp_path,change):
    source,code=qualification_source(tmp_path)
    s=source/'status.json';q=source/'qualification/qualification.json';p=plan()
    if change=='physics':(code/'physics.py').write_text('changed')
    if change=='running':v=json.loads(s.read_text());v['state']='running';s.write_text(json.dumps(v))
    if change=='partial':v=json.loads(q.read_text());v['qualified']=False;q.write_text(json.dumps(v))
    if change=='successful':v=json.loads(q.read_text());v['baseline']['success']=True;q.write_text(json.dumps(v))
    if change=='seed':p['policy_seed']=10
    if change=='symlink':(source/'qualification/bad').symlink_to(q)
    if change=='steps':(source/'attempt/failed.json').write_text('{"steps":1}')
    with pytest.raises(ValueError):reuse_qualification(source,tmp_path/'new-qualification',p,code)


def test_recovery_main_skips_baseline_and_calibration_and_uses_readback(tmp_path, monkeypatch):
    import contextlib
    from types import SimpleNamespace
    import collect_b1k_codex_subtask as collect
    import collect_b1k_codex as bridge
    import codex_subtask_teacher as teachers
    import codex_subtask_qualification as qualify
    import codex_jacobian_validation as jacobian
    import eval_b1k_subtask_samples as runner
    import legacy_history_recovery as history
    import codex_preparation_checkpoint as checkpoint
    sample=dict(sample_id='camera-start', task='camera', start=0, end=1,
        task_index=35, raw_episode_id=350040, subtask='pick up')
    p={**plan(), 'sample':sample, 'schema_version':2,'raw_data_root':'/raw',
        'source_plan':{'raw_source_sha256':{'task-0035/episode_00350040.hdf5':'a'*64},
            'selected_state_sha256':{'camera-start':'b'*64},
            'prefix_actions_sha256':{'camera-start':'c'*64},
            'source_protocol_by_sample':{'camera-start':collect.HISTORY}},
        'budgets':{'max_steps':2048,'max_seconds':120,'step_timeout':120},
        'teacher_motion_validation':jacobian.RUNTIME_CONTRACT}
    original={'camera-start':{'state':state(),'history_counter_state':{'left':None},
        'restoration':{'prefix_actions_sha256':'c'*64}}}
    cache={'sha256':'a'*64,'grasp_serialization_audit':{'legacy_missing_assisted_grasp_state':True},
        'sample_evidence':{'camera-start':{'selected_state_sha256':'b'*64}},
        'history_identity_grasps':{0:[]}}
    calls=[]
    args=SimpleNamespace(plan=tmp_path/'plan',output_dir=tmp_path/'new/attempt',
        codex_binary='fake',codex_timeout=120,codex_capacity_retries=2,policy_host='localhost',
        policy_port=8000,policy_server_token='test',robot_config=tmp_path/'robot',
        demonstration_manifest=tmp_path/'demo',reuse_qualification_root=tmp_path/'old',
        persist_preparation=True,resume_preparation=None)
    monkeypatch.setattr(collect,'parse_args',lambda _:args)
    monkeypatch.setattr(collect,'load_plan',lambda _:p)
    monkeypatch.setattr(teachers,'CodexSubtaskTeacher',lambda **_:object())
    base=SimpleNamespace(parse_args=lambda _:SimpleNamespace(),gm=SimpleNamespace(HEADLESS=False),
        seed_everything=lambda _:None,Evaluator=object)
    monkeypatch.setattr(runner,'_load_base_module',lambda:base)
    monkeypatch.setattr(runner,'raw_hdf5_path',lambda *_:None)
    monkeypatch.setattr(runner,'load_episode_cache',lambda *_ ,**__:cache)
    monkeypatch.setattr(runner,'_build_evaluator_cfg',lambda *_:None)
    ev=SimpleNamespace(policy=object())
    monkeypatch.setattr(runner,'create_sample_evaluator',lambda *_:ev)
    monkeypatch.setattr(bridge,'connect_collection_policy',lambda _:(object(),{'collection_seeding':True}))
    monkeypatch.setattr(bridge,'collection_evaluator',lambda factory,_:contextlib.nullcontext(factory()))
    monkeypatch.setattr(bridge,'time_limit',lambda _:contextlib.nullcontext())
    monkeypatch.setattr(checkpoint,'reuse_qualification',lambda *_: {'verified_source_files_sha256':{'physics':'b'*64}})
    monkeypatch.setattr(history,'enabled',lambda:True)
    def materialize(*_):
        calls.append('rebuild_once')
        return original
    monkeypatch.setattr(history,'materialize',materialize)
    def forbidden(*_,**__):raise AssertionError('Reused qualification must not execute again')
    monkeypatch.setattr(qualify,'qualify',forbidden)
    monkeypatch.setattr(qualify,'calibrate',forbidden)
    monkeypatch.setattr(jacobian,'run_initial',forbidden)
    def collect_attempt(_,teacher,actual,actual_cache,loaded,output,**kwargs):
        assert actual is p and loaded is not original['camera-start']
        assert (output.parent/'preparation/READY.json').exists()
        assert actual_cache['preparation_checkpoint']['readback_verified']
        np.testing.assert_array_equal(loaded['state'][0]['registry']['object_registry']['robot']['joint_pos'],np.arange(28,dtype=np.float32))
        calls.append('collect_after_readback')
        return {'test_only':True}
    monkeypatch.setattr(collect,'collect_attempt',collect_attempt)
    assert collect.main([])==0
    assert calls==['rebuild_once','collect_after_readback']
