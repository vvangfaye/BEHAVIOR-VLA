"""CPU checks for collection provenance, action pairing and export orchestration."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import nullcontext
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import subtask_improvement as plans
from subtask_trajectory import CAMERAS, TrajectoryWriter, capture_observation, inspect_trajectory
from export_subtask_collection import accepted_trials, export_collection
import eval_b1k_subtask_samples as runner


def inputs():
    samples=[dict(sample_id=f's{i}',task='task',task_instance_id=i//2,
                  episode_index=i,raw_episode_id=i,task_index=3,rollout_id=i,
                  subtask='pick up',predicate={},source_segment_index=0,source_interval_index=0,
                  start=0,end=10,segment_frames=10) for i in range(8)]
    panel=dict(fingerprint='frozen',samples=samples)
    summary=dict(complete=True,evaluation_kind='training_subtask_closed_loop',panel_fingerprint='frozen',
                 checkpoint='1375000',per_prompt_label={'task':{'pick up':dict(metric_eligible=20,successes=4,completion_rate=.2)}})
    summary.update(sample_max_steps=2048, overall={'metric_eligible':20},
                   state_fidelity={'contract':'recorded_native_fields_v1', 'required':True,
                                   'verified':True, 'eligible_results_verified':20})
    return panel,summary


def plan_for(panel,summary,**kw):
    return plans.freeze_plan(summary,panel,max_starts=2,trials=2,**kw)


def obs(value=0):
    return {'robot_r1::proprio':np.full(57,value,dtype=np.float32),
            **{source:np.full((2,3,4),value,dtype=np.uint8) for source in CAMERAS.values()}}


def write_trial(root,plan,panel,trial,value=1,success=True,video=True,q_score=0.0):
    sample=next(s for s in panel['samples'] if s['sample_id']==trial['sample_id'])
    out=root/'pi05'/'task'/'trials'/trial['trial_id'];out.mkdir(parents=True)
    metadata=dict(plan_fingerprint=plan['fingerprint'],trial=trial,sample=sample,fps=30,conditioning='task_and_subtask')
    result=dict(sample_id=sample['sample_id'],panel_fingerprint=panel['fingerprint'],q_score=q_score,steps=2,success=success,metric_eligible=True,subtask=sample['subtask'],conditioning={'source':'provided'})
    writer=TrajectoryWriter(out/'trajectory.hdf5',metadata)
    for step in range(2):writer.record(capture_observation(obs(value+step)),np.full(23,value,dtype=np.float32),False,False)
    writer.finish(result)
    inspected=inspect_trajectory(out/'trajectory.hdf5',require_success=False)
    (out/'completed.json').write_text(json.dumps(dict(plan_fingerprint=plan['fingerprint'],trial_id=trial['trial_id'],**result)))
    if success:
        (out/'accepted.json').write_text(json.dumps(dict(plan_fingerprint=plan['fingerprint'],content_sha256=inspected['content_sha256'])))
    if video:
        path=out/'videos'/f"task_{sample['task_instance_id']}_{sample['rollout_id']}.mp4"
        path.parent.mkdir();path.write_bytes(b'placeholder for CPU export contract test')
    return out


def test_plan_prefers_distinct_starts_excludes_holdout_and_is_reproducible(tmp_path):
    panel,summary=inputs();plan=plan_for(panel,summary,held_out_instances=[('task',0)])
    assert plan==plan_for(panel,summary,held_out_instances=[('task',0)])
    assert plan['maximum_control_steps']==4*2048
    chosen={t['sample_id'] for t in plan['trials']}
    instances={s['task_instance_id'] for s in panel['samples'] if s['sample_id'] in chosen}
    assert len(instances)==2 and 0 not in instances
    assert plan['selected_groups'][0]['distinct_instances']==2
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    assert plans.load_plan(path,panel)[0]==plan
    plan['trials'][0]['policy_seed']+=1
    plan['fingerprint']=plans.digest({k:v for k,v in plan.items() if k!='fingerprint'})
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='seed'):plans.load_plan(path,panel)


@pytest.mark.parametrize('change',['partial','probe','wrong_panel','invalid_results','underpowered','invalid_denominator'])
def test_selection_does_not_mislabel_infrastructure_or_underpowered_results(change):
    panel,summary=inputs()
    if change=='partial':summary['complete']=False
    elif change=='probe':summary['evaluation_kind']='subtask_state_validation'
    elif change=='wrong_panel':summary['panel_fingerprint']='other'
    elif change=='invalid_results':summary['invalid_results']=['bad']
    elif change=='underpowered':summary['per_prompt_label']['task']['pick up']=dict(metric_eligible=5,successes=1,completion_rate=.2)
    else:summary['per_prompt_label']['task']['pick up']['completion_rate']=.9
    if change=='underpowered':
        plan=plan_for(panel,summary);assert not plan['trials'] and len(plan['underpowered_groups'])==1
    else:
        with pytest.raises(ValueError):plan_for(panel,summary)


@pytest.mark.parametrize('budgets',[dict(max_steps=4097),dict(max_steps=True),dict(min_eligible=0),dict(threshold=float('nan'))])
def test_collection_budget_is_bounded(budgets):
    panel,summary=inputs()
    with pytest.raises(ValueError):plan_for(panel,summary,**budgets)


def test_recorder_copies_before_action_and_detects_mutated_arrays(tmp_path):
    panel,summary=inputs();plan=plan_for(panel,summary)
    out=write_trial(tmp_path,plan,panel,plan['trials'][0])
    assert inspect_trajectory(out/'trajectory.hdf5')['steps']==2
    with h5py.File(out/'trajectory.hdf5','r+') as f:f['action'][0,0]=5
    with pytest.raises(ValueError,match='changed'):inspect_trajectory(out/'trajectory.hdf5')
    observation=obs(1);frame=capture_observation(observation)
    observation['robot_r1::proprio'][:]=9
    assert np.all(frame['observation.state']==1)


def test_failed_duplicate_and_incomplete_trials_are_not_bc_data(tmp_path):
    panel,summary=inputs();plan=plan_for(panel,summary)
    write_trial(tmp_path,plan,panel,plan['trials'][0],value=1)
    with pytest.raises(ValueError,match='incomplete'):accepted_trials(tmp_path,plan,{s['sample_id']:s for s in panel['samples']})
    for trial in plan['trials'][1:]:write_trial(tmp_path,plan,panel,trial,value=1,success=trial==plan['trials'][1])
    kept,rejected=accepted_trials(tmp_path,plan,{s['sample_id']:s for s in panel['samples']})
    assert len(kept)==1 and {r['reason'] for r in rejected}=={'duplicate_trajectory','failed_or_setup_excluded'}


def test_export_requires_video_and_matching_provenance(tmp_path):
    panel,summary=inputs();plan=plan_for(panel,summary)
    outputs=[write_trial(tmp_path,plan,panel,t,value=i+1) for i,t in enumerate(plan['trials'])]
    next((outputs[0]/'videos').glob('*.mp4')).unlink()
    with pytest.raises(ValueError,match='video'):accepted_trials(tmp_path,plan,{s['sample_id']:s for s in panel['samples']})


class FakeLeRobot:
    """Only tests API orchestration and metadata; does not encode real videos."""
    finalized=False
    @classmethod
    def create(cls,**kw):
        obj=cls();obj.root=kw['root'];obj.frames=[];obj.episodes=[];obj.current=[]
        (obj.root/'meta/episodes/chunk-000').mkdir(parents=True)
        return obj
    def add_frame(self,frame):self.current.append(frame)
    def save_episode(self):
        self.episodes.append(dict(episode_index=len(self.episodes),length=len(self.current)))
        self.frames.extend(self.current);self.current=[]
    def finalize(self):
        type(self).finalized=True
        pq.write_table(pa.Table.from_pylist(self.episodes),self.root/'meta/episodes/chunk-000/file-000.parquet')


def test_single_root_export_has_dense_ids_full_labels_and_instance_groups(tmp_path):
    panel,summary=inputs();plan=plan_for(panel,summary)
    for i,t in enumerate(plan['trials']):write_trial(tmp_path/'collection',plan,panel,t,value=i+1)
    target=tmp_path/'dataset'
    report=export_collection(tmp_path/'collection',target,'pilot/dataset',panel,plan,dataset_class=FakeLeRobot,task_prompts={'task':'Do task'})
    assert FakeLeRobot.finalized and report['episodes']==4 and report['new_frame_fraction']==1
    episodes=pq.read_table(target/'meta/episodes/chunk-000/file-000.parquet').to_pylist()
    assert [r['episode_index'] for r in episodes]==[0,1,2,3]
    assert all(r['source_kind']=='collected' and r['source_task_index']==3 for r in episodes)
    labels=pq.read_table(target/'meta/subtask_segments.parquet').to_pylist()
    assert all(r['start']==0 and r['end']==2 and r['subtask']=='pick up' for r in labels)
    assert (target/'_EXPORTED').exists() and not report['gpu_training_verified']


def test_seed_restarts_wrapper_once_and_is_removed_from_policy_observation():
    path=Path(__file__).resolve().parents[1]/'openpi/src/openpi/shared/collection_seeding.py'
    spec=importlib.util.spec_from_file_location('collection_seeding_test',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    class Wrapper:
        def __init__(self):self.policy=SimpleNamespace(_is_pytorch_model=False,_rng=0);self.resets=0
        def reset(self):self.resets+=1
        def act(self,obs):
            assert '_collection_policy_seed' not in obs
            result=self.policy._rng;self.policy._rng+=1;return result
    wrapper=Wrapper();policy=module.SeededCollectionPolicy(wrapper,key_factory=lambda s:s)
    assert policy.act({'_collection_policy_seed':7})==7
    assert policy.act({})==8
    assert policy.act({'_collection_policy_seed':7})==7 and wrapper.resets==2
    with pytest.raises(ValueError):policy.act({'_collection_policy_seed':True})


def test_real_run_sample_callback_pairs_pre_step_state_with_executed_action(tmp_path,monkeypatch):
    panel,_=inputs();sample=panel['samples'][0]
    monkeypatch.setattr(runner,'restore_materialized_sample_state',lambda *a:{'full_snapshot_format':'named_simulator_state_v1'})
    class Monitor:
        metric_eligible=True;setup_exclusion_reason=None;setup_evidence={};satisfied=False
        def __init__(self,*a):pass
        def refresh(self,steps):return steps==2
        def result(self):return {'dwell':3}
    monkeypatch.setattr(runner,'CompletionMonitor',Monitor)
    class Evaluator:
        def __init__(self):
            self.obs=obs(1);self.env=SimpleNamespace(task=SimpleNamespace(success=False));self.metrics=[]
            self.conditioning_metadata={};self.subtask_events=[];self.observed_subtasks=[];self.observed_subtask_sources=[];self.n=0
        def step(self):
            self.n+=1;self.robot_action=np.full(23,self.n,dtype=np.float32)
            self.obs['robot_r1::proprio'][:]+=1
            return False,False
    base=SimpleNamespace(gm=SimpleNamespace(unlocked=nullcontext),verify_external_subtask_rollout=lambda *a,**kw:None)
    writer=TrajectoryWriter(tmp_path/'trajectory.hdf5',dict(sample=sample,conditioning='task_and_subtask',fps=30))
    result=runner.run_sample(base,Evaluator(),SimpleNamespace(write_video=False,sample_max_steps=3),sample,{}, {},tmp_path,'frozen',trajectory_writer=writer)
    writer.finish(result)
    with h5py.File(tmp_path/'trajectory.hdf5','r') as f:
        assert f['observation.state'][:,0].tolist()==[1,2]
        assert f['action'][:,0].tolist()==[1,2]


def test_nebula_collection_reuses_existing_launcher_and_checks_checkpoint(tmp_path,monkeypatch):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'nebula'))
    import subtask_eval
    import collect_b1k_subtasks as collector
    panel,summary=inputs()
    identity=dict(checkpoint_step=1375000,checkpoint_manifest_sha256='model-hash',checkpoint_source='oss://model')
    plan=plan_for(panel,summary,checkpoint_identity=identity)
    panel_path=tmp_path/'panel.json';panel_path.write_text(json.dumps(panel))
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    checkpoint=tmp_path/'checkpoint';checkpoint.mkdir();(checkpoint/'_READY').write_text(json.dumps(dict(step=1375000,manifest_sha256='model-hash')))
    args=SimpleNamespace(collection_plan=str(path),panel=str(panel_path),workers=1,sample_max_steps=2048,
        run_root=str(tmp_path/'run'),persistent_output=str(tmp_path/'published'),raw_root=str(tmp_path/'raw'),raw_source='/raw',
        checkpoint_source=str(checkpoint),checkpoint='/local/ckpt',run_id='collect',repo_id='joint',pi05_env='/env/policy',eval_env='/env/eval',asset_path='/assets',base_port=8330)
    inventory={'episodes':[{'relative_path':f'task-0003/episode_{s["raw_episode_id"]:08d}.hdf5'} for s in panel['samples']]}
    staged=[]
    def stage(selected,*a):staged.extend(selected['episodes']);return tmp_path/'raw'
    def supervise(command,local,persistent,*,env):
        assert command[:2]==['/bin/bash',str(subtask_eval.REPO_ROOT/'scripts/eval_b1k.sh')]
        assert command[2:]==['task'] and env['SERVER_GPU']==env['EVAL_GPU']=='0'
        assert env['PI05_SUBTASK_COLLECTION_PLAN']==str(path) and 'PI05_SUBTASK_PROBE_PLAN' not in env
    monkeypatch.setattr(subtask_eval,'stage_raw',stage);monkeypatch.setattr(subtask_eval,'supervise_results',supervise)
    monkeypatch.setattr(collector,'validate_collection',lambda *a:dict(complete=True,ready_for_export=False))
    subtask_eval.run_collection(args,panel,inventory)
    assert len(staged)==2
    marker=json.loads((tmp_path/'published/_SUCCESS').read_text())
    assert marker['kind']=='subtask_success_collection' and not marker['ready_for_export']
    (checkpoint/'_READY').write_text(json.dumps(dict(step=999,manifest_sha256='model-hash')))
    args.run_root=str(tmp_path/'other')
    with pytest.raises(ValueError,match='checkpoint'):subtask_eval.run_collection(args,panel,inventory)
    assert not (tmp_path/'other').exists()


@pytest.mark.parametrize('reserved',[False,True])
def test_original_replay_uses_same_root_and_cannot_leak_reserved_instances(tmp_path,reserved):
    panel,summary=inputs();plan=plan_for(panel,summary,held_out_instances=[('task',99)] if reserved else [])
    for i,t in enumerate(plan['trials']):write_trial(tmp_path/'collection',plan,panel,t,value=i+1,success=i==0)
    source=tmp_path/'base';(source/'meta/episodes/chunk-000').mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(episode_index=42,task_instance_id=99,task_index=3,tasks=['Do task'])]),source/'meta/episodes/chunk-000/file-000.parquet')
    labels=source/'labels.parquet';pq.write_table(pa.Table.from_pylist([dict(episode_index=42,start=0,end=3,subtask='move')]),labels)
    class WithReplay(FakeLeRobot):
        def __init__(self,**kwargs):self.meta=SimpleNamespace(fps=30)
        def __len__(self):return 3
        def __getitem__(self,i):
            frame=capture_observation(obs(2))
            return {**frame,'action':np.full(23,2,dtype=np.float32)}
    target=tmp_path/'output'
    kwargs=dict(base_dataset=source,base_segments=labels,base_episodes=[42],dataset_class=WithReplay,task_prompts={'task':'Do task'})
    if reserved:
        with pytest.raises(ValueError,match='reserved'):export_collection(tmp_path/'collection',target,'pilot',panel,plan,**kwargs)
        assert not (target/'_EXPORTED').exists()
    else:
        report=export_collection(tmp_path/'collection',target,'pilot',panel,plan,**kwargs)
        provenance=json.loads((target/'meta/collection_provenance.json').read_text())
        assert report['episodes']==2 and report['new_frame_fraction']==2/5
        assert provenance[1]['episode_index']==1 and provenance[1]['source_episode_index']==42
        assert provenance[1]['task_instance_id']==99 and provenance[1]['source_kind']=='original'


def test_collection_summary_requires_all_real_actions_and_acceptance_receipts(tmp_path):
    import collect_b1k_subtasks as collector
    panel,summary=inputs();plan=plan_for(panel,summary)
    outputs=[write_trial(tmp_path,plan,panel,t,value=i+1,success=i==0) for i,t in enumerate(plan['trials'])]
    report=collector.validate_collection(tmp_path,panel,plan)
    assert report['complete'] and report['eligible_trials']==4 and report['accepted_trials']==1 and report['ready_for_export']
    (outputs[0]/'accepted.json').unlink()
    with pytest.raises(ValueError,match='receipt'):collector.validate_collection(tmp_path,panel,plan)


@pytest.mark.parametrize('score,valid',[(0.2,True),({'final':0.2},True),
    ({'final':float('nan')},False),({'final':float('inf')},False),
    ({'other':0.2},False),({'final':{}},False)])
def test_collection_accepts_actual_omnigibson_score_shape_and_rejects_invalid_scores(tmp_path,score,valid):
    import collect_b1k_subtasks as collector
    panel,summary=inputs();plan=plan_for(panel,summary)
    for trial in plan['trials']:
        write_trial(tmp_path,plan,panel,trial,q_score=score)
    if valid:
        report=collector.validate_collection(tmp_path,panel,plan)
        assert report['complete'] and report['eligible_trials']==4
    else:
        with pytest.raises(ValueError,match='finite simulator score'):
            collector.validate_collection(tmp_path,panel,plan)


def test_replay_image_nonfinite_values_are_rejected_before_uint8_conversion():
    from export_subtask_collection import image_hwc
    with pytest.raises(ValueError,match='RGB'):image_hwc(np.full((8,8,3),np.nan))
    with pytest.raises(ValueError,match='Replay float RGB'):image_hwc(np.full((8,8,3),255.0))


def test_upstream_video_metadata_mutation_does_not_change_frozen_feature_contract(tmp_path):
    panel,summary=inputs();plan=plan_for(panel,summary)
    for i,t in enumerate(plan['trials']):write_trial(tmp_path/'collection',plan,panel,t,value=i+1)
    class MutableWriter(FakeLeRobot):
        @classmethod
        def create(cls,**kwargs):
            obj=super().create(**kwargs);obj.features=kwargs['features'];return obj
        def save_episode(self):
            super().save_episode()
            for key in CAMERAS:self.features[key]['info']={'video.fps':30}
    report=export_collection(tmp_path/'collection',tmp_path/'export','pilot',panel,plan,dataset_class=MutableWriter,task_prompts={'task':'Do task'})
    assert report['episodes']==4


@pytest.mark.parametrize('damage',[None,'stale_summary','missing_run','wrong_counts','model_identity'])
def test_baseline_marker_binds_completed_summary_and_verified_model(damage):
    panel,summary=inputs();panel['episodes']=[dict(task='task')];summary['run_id']='completed-baseline'
    receipt=dict(run_id=summary['run_id'],exit_code=0,evaluation_kind='training_subtask_closed_loop',
        panel_fingerprint=panel['fingerprint'],checkpoint_step=1375000,checkpoint_manifest_sha256='a'*64,checkpoint_source='oss://fixture/model')
    marker=dict(run_id=summary['run_id'],panel_fingerprint=panel['fingerprint'],summary_sha256=plans.digest(summary),
        expected_episode_jobs=1,expected_segment_rollouts=len(panel['samples']),expected_annotated_segments=len(panel['samples']),expected_exclusions=0)
    if damage=='stale_summary':marker['summary_sha256']='b'*64
    elif damage=='missing_run':summary.pop('run_id')
    elif damage=='wrong_counts':marker['expected_segment_rollouts']-=1
    elif damage=='model_identity':receipt['checkpoint_manifest_sha256']='model-name'
    if damage is None:assert plans.verified_baseline_identity(summary,panel,receipt,marker)['checkpoint_step']==1375000
    else:
        with pytest.raises(ValueError):plans.verified_baseline_identity(summary,panel,receipt,marker)


def test_collection_starts_exclude_baseline_setup_failures_and_remain_excluded_after_reload(tmp_path):
    panel,summary=inputs()
    excluded=panel['samples'][:2]
    summary['setup_exclusions']=[dict(sample_id=s['sample_id'],task=s['task'],subtask=s['subtask'],reason='already_satisfied') for s in excluded]
    plan=plan_for(panel,summary)
    assert plan['baseline_setup_excluded_sample_ids']==['s0','s1']
    assert not {t['sample_id'] for t in plan['trials']} & {'s0','s1'}
    # A plan with a correct digest/seed still cannot schedule an excluded start.
    original=plan['trials'][0]['sample_id']
    for trial in plan['trials']:
        if trial['sample_id']!=original:continue
        trial['sample_id']='s0'
        trial['policy_seed']=int(plans.digest([plan['seed'],'s0',trial['repeat']])[:8],16)
        trial['trial_id']=plans.digest(['s0',trial['repeat'],trial['policy_seed']])[:24]
    plan['fingerprint']=plans.digest({k:v for k,v in plan.items() if k!='fingerprint'})
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='setup-excluded'):plans.load_plan(path,panel)


@pytest.mark.parametrize('exclusions',[
    [{'sample_id':'unknown','task':'task','subtask':'pick up'}],
    [{'sample_id':'s0','task':'other','subtask':'pick up'}],
    [{'sample_id':'s0','task':'task','subtask':'other'}],
    [{'sample_id':'s0','task':'task','subtask':'pick up'}]*2,
    {'s0':True},
])
def test_collection_rejects_ambiguous_baseline_setup_exclusions(exclusions):
    panel,summary=inputs();summary['setup_exclusions']=exclusions
    with pytest.raises(ValueError,match='exclusion'):plan_for(panel,summary)
