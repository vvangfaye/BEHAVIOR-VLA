"""CPU integrity checks; fixtures are not proof of GPU grasp restoration."""
import copy
import json
from pathlib import Path
import pytest

import validate_b1k_grasp_reconstruction as rec
import diagnose_b1k_grasp_restore as diag
import eval_b1k_subtask_probe as probe
from test_subtask_probe import inputs


def history_fixture():
    rows=[]
    for i in range(65):
        rows.append({'source_frame':1000+i,'physics_steps':0,'arms':{'left':{
            'eef_position':[i*.001,0,0],'eef_orientation':[0,0,0,1],
            'gripper_joint_positions':[.01,.02],'gripper_joint_upper_limits':[.05,.05],
            'gripper_joint_lower_limits':[0,0],'held_object':None,'constraint_joint_type':None,
            'release_counter':None,'contacts_observed_after_physics':False,
            'objects':{'source_carried':{'world_position':[i*.001+.05,0,0],'world_orientation':[0,0,0,1],
                'relative_position':[.05,0,0],'relative_orientation':[0,0,0,1]},
                'other':{'world_position':[1,0,0],'world_orientation':[0,0,0,1],
                    'relative_position':[1-i*.001,0,0],'relative_orientation':[0,0,0,1]}}}}})
    return rows


def test_selection_uses_source_motion_without_task_or_completion_targets():
    proposals=rec.propose_grasps(history_fixture())
    assert proposals['left']['object']=='source_carried'
    assert proposals['left']['original_grasp_annotation_available'] is False


@pytest.mark.parametrize('bad',['short','gap','ambiguous','no_motion','fully_open','dual_object'])
def test_selection_rejects_weak_ambiguous_or_open_hand_evidence(bad):
    h=history_fixture()
    if bad=='short':h.pop()
    elif bad=='gap':h[-1]['source_frame']+=1
    elif bad=='fully_open':h[-1]['arms']['left']['gripper_joint_positions']=[.05,.05]
    else:
        for row in h:
            arm=row['arms']['left']
            if bad=='ambiguous':arm['objects']['second']=copy.deepcopy(arm['objects']['source_carried'])
            elif bad=='dual_object':row['arms']['right']=copy.deepcopy(arm)
            else:arm['eef_position']=[0,0,0]
    with pytest.raises(ValueError):rec.propose_grasps(h)


@pytest.mark.parametrize('bad',['early','seed','too_many','steps'])
def test_reconstruction_plan_cannot_expand_diagnostic_scope(tmp_path,bad):
    panel,plan,samples=inputs();plan.update(kind=rec.KIND,max_steps=32,paired_seed=1)
    for s in samples:s['start']=1064
    if bad=='early':samples[0]['start']=0
    elif bad=='seed':plan['paired_seed']=42
    elif bad=='steps':plan['max_steps']=33
    else:
        s={**samples[0],'sample_id':'third'};panel['samples'].append(s);plan['sample_ids'].append('third')
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):probe.load_plan(path,panel)


def result_fixture(root,sample,panel):
    h=history_fixture();proposals=rec.propose_grasps(h);source=copy.deepcopy(h[-1])
    before=copy.deepcopy(source);before['arms']['left'].update(held_object='source_carried',constraint_joint_type='FixedJoint')
    released=copy.deepcopy(source)
    released['arms']['left']['objects']['source_carried']['world_position'][0]+=.2
    out=root/'pi05'/sample['task']/'grasp_reconstruction';out.mkdir(parents=True,exist_ok=True)
    (out/'videos').mkdir(exist_ok=True);(out/'native_snapshots').mkdir(exist_ok=True)
    native=out/'native_snapshots'/(sample['sample_id']+'.pt');native.write_bytes(b'CPU structural fixture only')
    branches={}
    for name in rec.BRANCHES:
        obs=[]
        for i in range(3):
            v=copy.deepcopy(before if name==rec.BRANCHES[1] else source)
            v.update(source_frame=1064+i,physics_steps=i)
            v['arms']['left']['contacts_observed_after_physics']=bool(i);obs.append(v)
        video=out/'videos'/(sample['sample_id']+'-'+name+'.mp4');video.write_bytes(b'CPU video hash fixture')
        action=[0.0]*23;action[14]=action[22]=-1.0
        branches[name]={'physics_steps':2,'observations':obs,'recorded_actions':[action.copy(),action.copy()],
                        'video_sha256':diag.runner.file_sha256(video),'paired_seed':1,
                        'retention':rec.retention(obs,proposals)}
    contact_obs=copy.deepcopy(source);contact_obs.update(source_frame=1065,physics_steps=1)
    contact_obs['arms']['left']['native_grasp_candidate']={'object':'source_carried','link':'base_link'}
    data={'kind':rec.KIND,'official_benchmark':False,'metric_eligible':False,
        'sample_id':sample['sample_id'],'task':sample['task'],'instance_id':sample['task_instance_id'],
        'episode_index':sample['episode_index'],'raw_episode_id':sample['raw_episode_id'],'source_start':1064,
        'panel_fingerprint':panel['fingerprint'],'raw_sha256':'a'*64,'reconstruction_source_sha256':'b'*64,
        'native_robot_source_sha256':'c'*64,'method':rec.METHOD,'advancement_source':'recorded_demonstration_actions',
        'model_action_requests':0,'exact_legacy_grasp_restoration_verified':False,'proposals':proposals,
        'source_history':h,'source_observation':source,'initial_pose_unchanged':rec.pose_errors(source,before),
        'initial_robot_joint_positions':[.01,.02],
        'physical_contact_probe':{'physics_steps':1,'recorded_action':[0.0]*23,'observation':contact_obs,
                                'contacts':{'left':{'object':'source_carried','link':'base_link',
                                                   'joint_type':'FixedJoint','world_contact_position':[.1,0,0]}}},
        'known_native_snapshot':{'sha256':diag.runner.file_sha256(native),'before':before,'after_release':released,
                                'after_reload':copy.deepcopy(before),'pose_errors':rec.pose_errors(before,before),
                                'after_reload_joint_positions':[.01,.02],'recreated_known_constraints':True,
                                'original_legacy_grasp_verified':False},'branches':branches}
    path=out/(sample['sample_id']+'.json');path.write_text(json.dumps(data));return path,data


@pytest.mark.parametrize('bad',['scoring','original_claim','foreign','no_contact','different_link','no_release',
                                'noop','joint','pose','missing_branch','different_actions','retention','snapshot','video'])
def test_reconstruction_rejects_missing_or_changed_physical_evidence(tmp_path,bad):
    panel,plan,samples=inputs();plan.update(kind=rec.KIND,max_steps=2,paired_seed=1)
    for s in samples:s['start']=1064
    path,d=result_fixture(tmp_path,samples[0],panel);result_fixture(tmp_path,samples[1],panel)
    assert probe.validate_results(tmp_path,panel,plan,samples)['exact_legacy_grasp_restoration_verified'] is False
    if bad=='scoring':d['metric_eligible']=True
    elif bad=='original_claim':d['exact_legacy_grasp_restoration_verified']=True
    elif bad=='foreign':d['raw_episode_id']=999
    elif bad=='no_contact':d['physical_contact_probe']['observation']['arms']['left']['native_grasp_candidate']=None
    elif bad=='different_link':d['physical_contact_probe']['contacts']['left']['link']='other_link'
    elif bad=='no_release':d['known_native_snapshot']['after_release']['arms']['left']['held_object']='source_carried'
    elif bad=='noop':d['known_native_snapshot']['after_release']['arms']['left']['objects']['source_carried']['world_position']=d['source_observation']['arms']['left']['objects']['source_carried']['world_position']
    elif bad=='joint':d['known_native_snapshot']['after_reload_joint_positions']=[.2,.2]
    elif bad=='pose':d['known_native_snapshot']['after_reload']['arms']['left']['eef_position'][0]+=.1
    elif bad=='missing_branch':d['branches'].pop(rec.BRANCHES[0])
    elif bad=='different_actions':d['branches'][rec.BRANCHES[1]]['recorded_actions'][0][0]=1
    elif bad=='retention':d['branches'][rec.BRANCHES[1]]['retention']['left']['held_physical_steps']=0
    elif bad=='snapshot':(path.parent/'native_snapshots'/(path.stem+'.pt')).write_bytes(b'changed')
    else:(path.parent/'videos'/(path.stem+'-unmodified.mp4')).write_bytes(b'changed')
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):probe.validate_results(tmp_path,panel,plan,samples)


def test_completed_paired_diagnostic_is_not_a_model_validation_gate(tmp_path):
    panel,plan,samples=inputs();plan.update(kind=rec.KIND,max_steps=2,paired_seed=1)
    for s in samples:s['start']=1064;result_fixture(tmp_path,s,panel)
    result=probe.validate_results(tmp_path,panel,plan,samples)
    assert result['recorded_action_physics_steps']==10 and result['source_inferred_retention_passed']==2
    assert result['model_action_requests']==0 and result['metric_eligible'] is False
    plan['kind']=probe.KIND
    with pytest.raises(ValueError):probe.validate_results(tmp_path,panel,plan,samples)
