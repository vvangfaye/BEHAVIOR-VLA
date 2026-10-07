"""Synthetic strict-result fixtures; never simulator or model performance evidence."""
from pathlib import Path
import copy
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import subtask_paired_retest as paired
import subtask_improvement as collection
import eval_b1k_subtask_samples as runner
from test_subtask_sample_eval import _valid_result


def fixture_inputs():
    samples=[]
    for task_index,task in enumerate(['target','control']):
        for instance in range(3):
            samples.append(dict(sample_id=f'{task}-{instance}',task=task,task_index=task_index,
                task_instance_id=instance,episode_index=len(samples),raw_episode_id=len(samples),rollout_id=0,
                subtask='pick up',skill_description='pick up',predicate={'kind':'fixture'},
                source_segment_index=0,source_interval_index=0,start=10,end=40,segment_frames=30))
    panel=dict(fingerprint='synthetic-panel',samples=samples,exclusions=[],episodes=[
        dict(task=s['task'],task_instance_id=s['task_instance_id'],sample_ids=[s['sample_id']]) for s in samples])
    summary=dict(complete=True,evaluation_kind='training_subtask_closed_loop',panel_fingerprint=panel['fingerprint'],
        checkpoint='1375000',per_prompt_label={'target':{'pick up':dict(metric_eligible=20,successes=4,completion_rate=.2)}})
    summary.update(sample_max_steps=2048, overall={'metric_eligible':20},
        state_fidelity={'contract':'recorded_native_fields_v1', 'required':True,
                        'verified':True, 'eligible_results_verified':20})
    identity=dict(checkpoint_step=1375000,checkpoint_manifest_sha256='a'*64,checkpoint_source='oss://fixture/base')
    collect=collection.freeze_plan(summary,panel,max_starts=1,trials=1,
        held_out_instances=[(t,i) for t in ('target','control') for i in (1,2)],checkpoint_identity=identity)
    plan=paired.freeze_retest(panel,collect,['control'],max_steps=4,per_group=2)
    return panel,collect,plan


def write_run(root,panel,plan,role):
    root.mkdir();checkpoint='/'+role
    identity=dict(checkpoint_step=1375000 if role=='base' else 2000,
        checkpoint_manifest_sha256=('a' if role=='base' else 'b')*64,checkpoint_source='oss://fixture/'+role)
    receipt=dict(run_id=role,evaluation_kind='training_subtask_closed_loop',exit_code=0,panel_fingerprint=panel['fingerprint'],**identity)
    manifest=dict(profile='training_subtask_closed_loop_episode_segments',evaluation_kind='training_subtask_closed_loop',
        run_id=role,checkpoint=checkpoint,panel_fingerprint=panel['fingerprint'],expected_episode_jobs=len(panel['episodes']),
        expected_segment_rollouts=len(panel['samples']),expected_annotated_segments=len(panel['samples']),expected_exclusions=0,
        sample_max_steps=plan['max_steps'],child_environment=dict(PI05_PAIRED_RETEST_SEED=str(plan['seed']),
            SUBTASK_PREDICTION='1',PROVIDED_SUBTASK_AS_PROMPT='0',MAX_TOKEN_LEN='512'),
        jobs=[dict(task=e['task'],instance_id=e['task_instance_id'],sample_ids=e['sample_ids']) for e in panel['episodes']],
        raw_preflight=dict(schema_version=2,valid=True,episodes=[dict(sample_ids=e['sample_ids'],sha256='c'*64) for e in panel['episodes']],
            samples=[dict(sample_id=s['sample_id'],episode_index=s['episode_index'],rollout_id=s['rollout_id'],
                selected_frame=s['start'],selected_state_size=6,transition_count_before_frame=1) for s in panel['samples']]))
    manifest['fingerprint']=paired.digest(manifest)
    runtime={k:'e'*64 for k in ['policy_environment','evaluation_environment','assets_manifest','norm_stats','runtime_requirements']}
    contract=dict(run_id=role,**identity,sample_max_steps=4,conditioning='task_and_subtask',
        evaluator_code_sha256='d'*64,runtime_identity=runtime,paired_retest_seed=str(plan['seed']))
    for name,value in [('manifest.json',manifest),('submission_contract.json',contract),('_SUCCESS',dict(
        expected_segment_rollouts=len(panel['samples']),expected_episode_jobs=len(panel['episodes'])))]:
        (root/name).write_text(json.dumps(value))
    for s in panel['samples']:
        result=_valid_result(s,panel['fingerprint']);result['restoration'].update(sha256='c'*64,full_snapshot_format='named_simulator_state_v1')
        result.update(policy_seed=paired.policy_seed(plan['seed'],panel['fingerprint'],s['sample_id']),seed_protocol=paired.SEED_PROTOCOL)
        success=s['task']=='control' or role=='fine'
        if not success:
            result.update(success=False,steps=4,failure_reason='sample_timeout',observed_subtasks=[s['subtask']]*4,observed_subtask_sources=['provided']*4)
            result['monitor'].update(dwell=0,satisfied=False)
        path=paired._result_path(root/'results',s);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result))
        video=path.parent.parent/'videos'/(path.stem+'.mp4');video.parent.mkdir(exist_ok=True);video.write_bytes(b'synthetic nonempty diagnostic artifact')
    summary=paired.summarize(panel,root/'results',checkpoint=checkpoint,require_videos=True,run_manifest=manifest)
    marker=json.loads((root/'_SUCCESS').read_text());marker.update(run_id=role,panel_fingerprint=panel['fingerprint'],summary_sha256=paired.digest(summary))
    (root/'_SUCCESS').write_text(json.dumps(marker))
    return receipt


@pytest.fixture
def experiment(tmp_path):
    panel,collect,plan=fixture_inputs()
    base=write_run(tmp_path/'base',panel,plan,'base');fine=write_run(tmp_path/'fine',panel,plan,'fine')
    provenance=[dict(task_name=t,task_instance_id=0) for t in ('target','control')]
    training=dict(training_run_id='synthetic-training',complete=True,conditioning='task_and_subtask',
        collection_plan_fingerprint=collect['fingerprint'],export_provenance_sha256=paired.digest(provenance),
        init_checkpoint_manifest_sha256=base['checkpoint_manifest_sha256'],output_checkpoint_manifest_sha256=fine['checkpoint_manifest_sha256'])
    return dict(panel=panel,plan=plan,collection_plan=collect,base_root=tmp_path/'base',fine_root=tmp_path/'fine',
                base_receipt=base,fine_receipt=fine,provenance=provenance,training_receipt=training)


def test_frozen_plan_prefers_reserved_instances_and_requires_control_tasks():
    panel,collect,plan=fixture_inputs()
    assert paired.validate_plan(plan,panel,collect)==plan
    assert plan==paired.freeze_retest(panel,collect,['control'],max_steps=4,per_group=2)
    assert all(s['sample_id'].endswith(('1','2')) for s in plan['samples'])
    with pytest.raises(ValueError,match='distinct'):paired.freeze_retest(panel,collect,['target'])
    collect['held_out_instances']=[]
    with pytest.raises(ValueError,match='Reserve'):paired.freeze_retest(panel,collect,['control'])


def test_strict_comparison_reports_underpowered_target_gain_and_control_counts(experiment):
    report=paired.compare_runs(**experiment)
    assert report['target']['base_success_rate']==0 and report['target']['finetuned_success_rate']==1
    assert report['target']['gains']==2 and report['control']['pairs']==2
    assert not report['policy_improvement_confirmed'] and report['complete']
    assert report['held_out_scope']=='this_fine_tuning_experiment_only'


@pytest.mark.parametrize('damage,pattern',[
    ('missing_video','Strict'),('invalid_result','Strict'),('wrong_seed','seed'),
    ('changed_raw','start-state'),('changed_runtime','runtime'),('data_leak','leaks'),
    ('training_identity','training provenance'),('incomplete','Incomplete'),('changed_plan','Modified')])
def test_rejects_unmatched_invalid_or_leaked_comparisons(experiment,damage,pattern):
    e=experiment;root=e['fine_root'];sample=e['panel']['samples'][1];path=paired._result_path(root/'results',sample)
    if damage=='missing_video':next((root/'results').glob('pi05/*/videos/*.mp4')).unlink()
    elif damage in ('invalid_result','wrong_seed','changed_raw'):
        row=json.loads(path.read_text())
        if damage=='invalid_result':row['monitor']['dwell']=0
        elif damage=='wrong_seed':row['policy_seed']+=1
        else:
            row['restoration']['sha256']='f'*64
            m=json.loads((root/'manifest.json').read_text())
            m['raw_preflight']['episodes'][1]['sha256']='f'*64
            m['fingerprint']=paired.digest({k:v for k,v in m.items() if k!='fingerprint'})
            (root/'manifest.json').write_text(json.dumps(m))
        path.write_text(json.dumps(row))
    elif damage=='changed_runtime':
        p=root/'submission_contract.json';m=json.loads(p.read_text());m['runtime_identity']['policy_environment']='f'*64;p.write_text(json.dumps(m))
    elif damage=='data_leak':
        e['provenance'].append(dict(task_name='target',task_instance_id=1))
        e['training_receipt']['export_provenance_sha256']=paired.digest(e['provenance'])
    elif damage=='training_identity':e['training_receipt']['init_checkpoint_manifest_sha256']='z'*64
    elif damage=='incomplete':(root/'_SUCCESS').unlink()
    else:e['plan']['samples'][0]['policy_seed']+=1
    with pytest.raises(ValueError,match=pattern):paired.compare_runs(**e)


def test_cluster_uncertainty_keeps_correlated_samples_inside_instances():
    rows=[dict(task='target',task_instance_id=0,base_success=False,finetuned_success=True) for _ in range(100)]
    one=paired.paired_statistics(rows)
    assert one['instance_groups']==1 and one['delta_ci95'] is None and not one['evidence_sufficient']
    rows.extend(dict(task='target',task_instance_id=1,base_success=True,finetuned_success=False) for _ in range(100))
    two=paired.paired_statistics(rows)
    assert two['delta']==0 and two['delta_ci95']==[-1,1]
    eight=paired.paired_statistics([dict(task='target',task_instance_id=i,base_success=False,finetuned_success=True) for i in range(8)])
    assert eight['supported_improvement'] and eight['delta_ci95']==[1,1]
    assert 0 < eight['conservative_delta_ci95'][0] < .1 and eight['conservative_delta_ci95'][1]==1
    tied=paired.paired_statistics([dict(task='control',task_instance_id=i,base_success=True,finetuned_success=True) for i in range(8)])
    assert tied['delta_ci95']==[0,0] and tied['conservative_delta_ci95'][0]<0<tied['conservative_delta_ci95'][1]


@pytest.mark.parametrize('value',[-1,2**32,True,1.5])
def test_invalid_uint32_seed_is_rejected(value):
    with pytest.raises(ValueError):paired.policy_seed(value,'panel','sample')


def test_run_sample_requests_its_seed_once_and_persists_it(tmp_path,monkeypatch):
    panel,_,_=fixture_inputs();sample=panel['samples'][1];calls=[];seeds=[]
    monkeypatch.setenv('PI05_PAIRED_RETEST_SEED','8')
    monkeypatch.setattr(runner,'restore_materialized_sample_state',lambda *a:{'full_snapshot_format':'named_simulator_state_v1'})
    class Monitor:
        metric_eligible=True;setup_exclusion_reason=None;setup_evidence={};satisfied=False
        def __init__(self,*a):pass
        def refresh(self,steps):return steps==2
        def result(self):return {}
    monkeypatch.setattr(runner,'CompletionMonitor',Monitor)
    class Policy:
        policy=SimpleNamespace(get_server_metadata=lambda:{'collection_seeding':True})
        def reset(self):pass
        def forward(self,obs):calls.append(copy.deepcopy(obs))
    class Evaluator:
        def __init__(self):
            self.policy=Policy();self.env=SimpleNamespace(task=SimpleNamespace(success=False));self.metrics=[]
            self.conditioning_metadata={};self.subtask_events=[];self.observed_subtasks=[];self.observed_subtask_sources=[]
        def step(self):self.policy.forward(obs={'frame':1});return False,False
    base=SimpleNamespace(gm=SimpleNamespace(unlocked=nullcontext),seed_everything=seeds.append,verify_external_subtask_rollout=lambda *a,**kw:None)
    result=runner.run_sample(base,Evaluator(),SimpleNamespace(write_video=False,sample_max_steps=4),sample,{}, {},tmp_path,panel['fingerprint'])
    expected=paired.policy_seed(8,panel['fingerprint'],sample['sample_id'])
    assert seeds==[expected] and calls[0]['_collection_policy_seed']==expected and '_collection_policy_seed' not in calls[1]
    assert result['policy_seed']==expected and result['seed_protocol']==paired.SEED_PROTOCOL


def test_zero_eligible_target_is_reported_without_a_success_denominator(experiment):
    e=experiment
    for role in ('base','fine'):
        root=e[role+'_root']
        for sample in e['panel']['samples']:
            if sample['task']!='target':continue
            path=paired._result_path(root/'results',sample);row=json.loads(path.read_text())
            evidence={'initial_predicate_value':True}
            row.update(success=False,steps=0,metric_eligible=False,result_state='setup_excluded',
                setup_exclusion_reason='predicate_already_true',setup_evidence=evidence,
                failure_reason='predicate_already_true',observed_subtasks=[],observed_subtask_sources=[])
            row['monitor'].update(metric_eligible=False,setup_exclusion_reason='predicate_already_true',
                setup_evidence=evidence,dwell=0,satisfied=True)
            path.write_text(json.dumps(row))
        manifest=json.loads((root/'manifest.json').read_text())
        summary=paired.summarize(e['panel'],root/'results',checkpoint=manifest['checkpoint'],require_videos=True,run_manifest=manifest)
        p=root/'_SUCCESS';marker=json.loads(p.read_text());marker['summary_sha256']=paired.digest(summary);p.write_text(json.dumps(marker))
    report=paired.compare_runs(**e)
    assert report['target']['pairs']==0 and report['target']['delta'] is None and not report['policy_improvement_confirmed']
    target=next(g for g in report['per_group'] if g['role']=='target')
    assert target['scheduled_pairs']==2 and target['setup_excluded']==2 and len(report['setup_excluded'])==2
