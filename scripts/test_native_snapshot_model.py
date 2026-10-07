"""CPU integrity tests; fixture states/videos are not GPU or model evidence."""
import copy
import json
from pathlib import Path
import pytest
import validate_b1k_native_snapshot_model as native
import eval_b1k_subtask_probe as probe
import validate_b1k_grasp_reconstruction as rec
from test_subtask_probe import inputs
from test_grasp_reconstruction import result_fixture


def frozen_inputs(root):
    panel,source_plan,samples=inputs();source_plan.update(kind=rec.KIND,max_steps=32,paired_seed=1)
    for s in samples:s['start']=1064
    for sample in samples:
        path,d=result_fixture(root,sample,panel)
        for branch in d['branches'].values():
            branch['physics_steps']=32
            branch['recorded_actions']=[branch['recorded_actions'][0].copy() for _ in range(32)]
            obs=[]
            for i in range(33):
                row=copy.deepcopy(branch['observations'][0]);row.update(source_frame=1064+i,physics_steps=i)
                row['arms']['left']['contacts_observed_after_physics']=bool(i);obs.append(row)
            branch['observations']=obs;branch['retention']=rec.retention(obs,d['proposals'])
        path.write_text(json.dumps(d))
    (root/'validation_plan.json').write_text(json.dumps(source_plan));(root/'parent_panel.json').write_text(json.dumps(panel))
    (root/'validation_contract.json').write_text(json.dumps({'run_id':'source','checkpoint_step':1375000}))
    (root/'_SUCCESS').write_text(json.dumps({'kind':rec.KIND,'complete':True}))
    (root/'validation_summary.json').write_text(json.dumps(rec.validate_results(root,panel,source_plan,samples)))
    files=[{'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':native._sha(p)} for p in sorted(root.rglob('*')) if p.is_file()]
    manifest={'kind':native.MANIFEST_KIND,'state_protocol':native.PROTOCOL,'panel_fingerprint':panel['fingerprint'],
              'sample_ids':source_plan['sample_ids'],'source_iid':'217810','source_run_id':'source',
              'exact_legacy_grasp_restoration_verified':False,'files':files}
    (root/'manifest.json').write_text(json.dumps(manifest))
    plan={'schema_version':1,'kind':native.KIND,'panel_fingerprint':panel['fingerprint'],'sample_ids':source_plan['sample_ids'],
          'state_protocol':native.PROTOCOL,'snapshot_input_directory':native.INPUT_DIRECTORY,'paired_seed':1,
          'max_steps':2048,'checkpoint_step':1375000,'checkpoint_manifest_sha256':'f'*64,
          'snapshot_manifest_sha256':native._sha(root/'manifest.json'),'source_iid':'217810','source_run_id':'source',
          'metric_eligible':False,'official_benchmark':False,'exact_legacy_grasp_restoration_verified':False,
          'native_snapshot_sha256_by_sample':{json.loads(p.read_text())['sample_id']:json.loads(p.read_text())['known_native_snapshot']['sha256'] for p in root.glob('pi05/*/grasp_reconstruction/*.json')}}
    return panel,plan,samples


def test_frozen_inputs_require_completed_paired_source_and_copy_only_whitelist(tmp_path):
    source=tmp_path/'input';source.mkdir();panel,plan,samples=frozen_inputs(source)
    manifest,proofs=native.validate_input_artifacts(source,panel,plan)
    assert set(proofs)==set(plan['sample_ids'])
    native.pack_input_artifacts(source,tmp_path/'packed',panel,plan)
    assert native.validate_input_artifacts(tmp_path/'packed',panel,plan)[0]==manifest
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    assert probe.load_plan(path,panel)==(plan,samples)


@pytest.mark.parametrize('change',['scoring','official','historical','step_limit','checkpoint','directory','extra_start','missing_hash'])
def test_native_plan_preserves_boundaries_and_false_historical_fidelity(tmp_path,change):
    panel,plan,samples=frozen_inputs(tmp_path)
    if change=='scoring':plan['metric_eligible']=True
    elif change=='official':plan['official_benchmark']=True
    elif change=='historical':plan['exact_legacy_grasp_restoration_verified']=True
    elif change=='step_limit':plan['max_steps']=2049
    elif change=='checkpoint':plan['checkpoint_step']=1500000
    elif change=='directory':plan['snapshot_input_directory']='../source'
    elif change=='missing_hash':plan['native_snapshot_sha256_by_sample'].pop(samples[0]['sample_id'])
    else:plan['sample_ids'].append('extra')
    with pytest.raises(ValueError):native.validate_plan(plan)


@pytest.mark.parametrize('change',['extra_file','missing_file','altered_state','altered_manifest','traversal','symlink','retention'])
def test_native_source_rejects_integrity_and_source_failures(tmp_path,change):
    panel,plan,samples=frozen_inputs(tmp_path)
    if change=='extra_file':(tmp_path/'unexpected.json').write_text('{}')
    elif change=='missing_file':(tmp_path/'_SUCCESS').unlink()
    elif change=='altered_state':next(tmp_path.glob('pi05/*/grasp_reconstruction/native_snapshots/*.pt')).write_bytes(b'changed native state')
    elif change=='altered_manifest':(tmp_path/'manifest.json').write_text('{}')
    elif change=='symlink':
        p=next(tmp_path.glob('pi05/*/grasp_reconstruction/native_snapshots/*.pt'));body=p.read_bytes();p.unlink()
        target=tmp_path.parent/'external-native';target.write_bytes(body);p.symlink_to(target)
    else:
        m=json.loads((tmp_path/'manifest.json').read_text())
        if change=='traversal':m['files'][0]['path']='../outside.json'
        else:
            p=next(tmp_path.glob('pi05/*/grasp_reconstruction/*.json'));d=json.loads(p.read_text())
            d['branches']['source_contact_reconstruction']['retention']['left']['passed']=False;p.write_text(json.dumps(d))
            for f in m['files']:
                if f['path']==str(p.relative_to(tmp_path)):f.update(bytes=p.stat().st_size,sha256=native._sha(p))
        (tmp_path/'manifest.json').write_text(json.dumps(m));plan['snapshot_manifest_sha256']=native._sha(tmp_path/'manifest.json')
    with pytest.raises((ValueError,FileNotFoundError)):native.validate_input_artifacts(tmp_path,panel,plan)


@pytest.mark.parametrize('change',['joint','object','link','pose','implementation','nan'])
def test_pre_model_native_restart_requires_actual_joint_pose_and_grasp_match(tmp_path,change):
    panel,plan,samples=frozen_inputs(tmp_path);_,proofs=native.validate_input_artifacts(tmp_path,panel,plan)
    proof=proofs[samples[0]['sample_id']][1];obs=copy.deepcopy(proof['known_native_snapshot']['before'])
    q=proof['initial_robot_joint_positions'].copy();links={'left':'base_link'};sha=proof['native_robot_source_sha256']
    assert native.validate_restored_observation(proof,obs,q,links,sha)['known_native_restart_verified']
    if change=='joint':q[0]+=.001
    elif change=='object':obs['arms']['left']['held_object']=None
    elif change=='link':links['left']='wrong_link'
    elif change=='pose':obs['arms']['left']['eef_position'][0]+=.001
    elif change=='implementation':sha='d'*64
    else:obs['arms']['left']['eef_position'][0]=float('nan')
    with pytest.raises(ValueError):native.validate_restored_observation(proof,obs,q,links,sha)


def put_model_result(root,sample,plan,panel):
    path=root/'pi05'/sample['task']/'native_snapshot_validation'/'json'/(sample['sample_id']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    video=path.parent.parent/'videos'/(path.stem+'.mp4');video.parent.mkdir(exist_ok=True);video.write_bytes(b'CPU structural video fixture')
    restart={'known_native_restart_verified':True,'full_robot_joints_match':True,'state_protocol':native.PROTOCOL,
             'pose_errors':{'max_position_difference_m':0.0,'max_orientation_difference_rad':0.0},'exact_legacy_grasp_restoration_verified':False}
    sha=plan['native_snapshot_sha256_by_sample'][sample['sample_id']]
    rollout={**sample,'instance_id':sample['task_instance_id'],'segment_start':sample['start'],
             'steps':2,'success':False,'metric_eligible':True,'q_score':0.0,'result_state':'evaluated',
             'restoration':{'native_snapshot_restart':restart,'native_snapshot_sha256':sha},
             'observed_subtasks':[sample['subtask']]*2,'observed_subtask_sources':['provided']*2}
    rows=[{'step':i,'preceding_observation_state':[0.0]*61,'executed_action':[0.0]*23} for i in range(2)]
    d={'kind':native.KIND,'sample_id':sample['sample_id'],'panel_fingerprint':panel['fingerprint'],
       'state_protocol':native.PROTOCOL,'source_iid':plan['source_iid'],'source_run_id':plan['source_run_id'],
       'snapshot_manifest_sha256':plan['snapshot_manifest_sha256'],'native_snapshot_sha256':sha,'paired_seed':1,
       'official_benchmark':False,'metric_eligible':False,'exact_legacy_grasp_restoration_verified':False,
       'model_rollout':rollout,'actual_model_control_steps':2,'observed_policy_subtask_reports':2,
       'action_audit':rows,'video_sha256':native._sha(video)}
    path.write_text(json.dumps(d));return path,d


@pytest.mark.parametrize('change',['scoring','historic','snapshot','zero_model_actions','wrong_subtask','recorded_action_source','nonfinite_action','wrong_dimensions','no_restart','joint_claim','video','incomplete'])
def test_model_validation_requires_complete_real_controls_and_keeps_scoring_disabled(tmp_path,change):
    source=tmp_path/'input';source.mkdir();panel,plan,samples=frozen_inputs(source)
    out=tmp_path/'output';path,d=put_model_result(out,samples[0],plan,panel);second,_=put_model_result(out,samples[1],plan,panel)
    summary=probe.validate_results(out,panel,plan,samples)
    assert summary['actual_model_control_steps']==4 and summary['metric_eligible'] is False
    if change=='scoring':d['metric_eligible']=True
    elif change=='historic':d['exact_legacy_grasp_restoration_verified']=True
    elif change=='snapshot':d['native_snapshot_sha256']='f'*64
    elif change=='zero_model_actions':d['model_rollout']['steps']=0
    elif change=='wrong_subtask':d['model_rollout']['observed_subtasks'][0]='other'
    elif change=='recorded_action_source':d['model_rollout']['observed_subtask_sources'][0]='recorded'
    elif change=='nonfinite_action':d['action_audit'][0]['executed_action'][0]=float('nan')
    elif change=='wrong_dimensions':d['action_audit'][0]['executed_action'].pop()
    elif change=='no_restart':d['model_rollout']['restoration'].pop('native_snapshot_restart')
    elif change=='joint_claim':d['model_rollout']['restoration']['native_snapshot_restart']['full_robot_joints_match']=False
    elif change=='video':(path.parent.parent/'videos'/(path.stem+'.mp4')).unlink()
    else:second.unlink()
    path.write_text(json.dumps(d))
    with pytest.raises(ValueError):probe.validate_results(out,panel,plan,samples)


def test_failed_restart_validation_aborts_before_monitor_or_model(monkeypatch,tmp_path):
    from types import SimpleNamespace
    import eval_b1k_subtask_samples as runner
    monkeypatch.delenv('PI05_PAIRED_RETEST_SEED',raising=False)
    calls=[]
    monkeypatch.setattr(runner,'restore_materialized_sample_state',lambda *a:calls.append('restore') or {})
    monkeypatch.setattr(runner,'CompletionMonitor',lambda *a:pytest.fail('Monitor must not run after invalid restart'))
    def reject(evaluator):
        calls.append('verify');raise ValueError('Actual native restart invalid')
    with pytest.raises(ValueError,match='restart invalid'):
        runner.run_sample(SimpleNamespace(),SimpleNamespace(),SimpleNamespace(),{'subtask':'fixed'}, {},{},tmp_path,'panel',restoration_validator=reject)
    assert calls==['restore','verify']
