"""Source-loss regressions: fixtures never establish simulator restoration."""
import copy
import json
from types import SimpleNamespace

import h5py
import pytest

import eval_b1k_subtask_samples as runner
from pi05_eval_queue import EvaluationQueue, ArtifactValidationError
from subtask_sample_panel import preflight_panel
from subtask_source_exclusion import make_source_exclusion, validate_source_exclusion
from summarize_subtask_closed_loop import summarize
from test_subtask_sample_eval import _sample, _write_hdf5, _summary_panel, _summary_manifest, _valid_result, _write_summary_result
import subtask_improvement as plans
from test_subtask_improvement import inputs


def legacy_source(tmp_path):
    sample = _sample()
    path = runner.raw_hdf5_path(tmp_path / 'raw', sample)
    _write_hdf5(path)
    with h5py.File(path, 'r+') as f:
        f['data'].attrs['config'] = json.dumps({'robots':[{'name':'robot','grasping_mode':'assisted'}]})
        f['data'].attrs['scene_file'] = json.dumps({'state':{'registry':{'object_registry':{'robot':{'controllers':{}}}}}})
    cache = runner.load_episode_cache(path, [sample], allow_unverified_grasp_state=True)
    panel = _summary_panel([sample])
    panel['episodes'][0].update(episode_index=sample['episode_index'], raw_episode_id=sample['raw_episode_id'])
    manifest = _summary_manifest(panel)
    manifest.update(run_id='source-loss-fixture', state_protocol='recorded_native_fields_v1')
    manifest['raw_preflight'] = preflight_panel(panel, tmp_path / 'raw')
    assert manifest['raw_preflight']['valid']
    return sample, panel, cache, manifest


def test_source_excluded_production_runner_publishes_without_evaluator_model_or_video(tmp_path, monkeypatch):
    sample, panel, cache, manifest = legacy_source(tmp_path)
    queue = EvaluationQueue(tmp_path / 'queue.sqlite3')
    args = SimpleNamespace(subtask_panel=tmp_path/'panel.json', task_name=sample['task'],
        queue_db=tmp_path / 'queue.sqlite3', queue_run_id=manifest['run_id'], queue_worker='slot-0', queue_generation=0,
        queue_attempt_root=str(tmp_path/'attempts'), mode='train', write_video=True,
        raw_data_root=tmp_path/'raw', output_dir=tmp_path/'results/pi05'/sample['task'])
    queue.initialize([(sample['task'], sample['task_instance_id'])], max_attempts=2, metadata={
        'run_id':args.queue_run_id, 'mode':'train', 'require_videos':True, 'panel_fingerprint':panel['fingerprint'],
        'profile':runner.QUEUE_PROFILE, 'evaluation_kind':runner.EVALUATION_KIND,
        'expected_episode_jobs':1, 'expected_segment_rollouts':1})
    monkeypatch.setattr(runner, 'load_panel', lambda _: panel)
    def forbidden(*a, **kw): raise AssertionError('Unrestorable sources must not create evaluator or control the model')
    monkeypatch.setattr(runner,'create_sample_evaluator',forbidden)
    monkeypatch.setattr(runner,'materialize_episode_start_states',forbidden)
    monkeypatch.setattr(runner,'run_sample',forbidden)
    results = runner.run(args, base_module=SimpleNamespace())
    assert len(results)==1 and results[0]['success'] is None and results[0]['steps']==0
    assert queue.counts()['counts']['succeeded']==1
    assert not list((tmp_path/'results').rglob('*.mp4'))
    summary = summarize(panel, tmp_path/'results', checkpoint='/checkpoint', require_videos=True, run_manifest=manifest)
    assert summary['complete'] and summary['state_fidelity']['verified']
    assert summary['overall']['source_excluded']==1 and summary['overall']['metric_eligible']==0
    assert summary['success_rate'] is None and not summary['setup_exclusions']
    assert summary['source_exclusions'][0]['sample_id']==sample['sample_id']


@pytest.mark.parametrize('damage',['success','steps','snapshot','monitor','responses','model_calls','hash','row_hash','inspected_prefix','native_schema','magic','runtime_verified','source_identity'])
def test_source_loss_cannot_be_used_to_mask_model_failure_or_fabricate_restore(tmp_path, damage):
    sample,panel,cache,manifest = legacy_source(tmp_path)
    result = make_source_exclusion(cache,sample,panel['fingerprint'])
    provenance = {**manifest['raw_preflight']['samples'][0], 'sha256':cache['sha256']}
    if damage=='success': result['success']=False
    elif damage=='steps': result['steps']=1
    elif damage=='snapshot': result['restoration']['full_snapshot']=True
    elif damage=='monitor': result['monitor']={}
    elif damage=='responses': result['observed_subtasks']=[sample['subtask']]
    elif damage=='model_calls': result['conditioning']['model_action_requests']=1
    elif damage=='hash': result['source_evidence']['raw_sha256']='b'*64
    elif damage=='row_hash': result['source_evidence']['selected_state_sha256']='b'*64
    elif damage=='inspected_prefix': result['source_evidence']['grasp_serialization_audit']['inspected_prefix_frames']=sample['start']
    elif damage=='native_schema': result['source_evidence']['grasp_serialization_audit']['robots'][0]['legacy_controller_layout']=False
    elif damage=='magic': result['source_evidence']['grasp_serialization_audit']['frames_with_ag_magic']=1
    elif damage=='runtime_verified': result['source_evidence']['grasp_serialization_audit']['runtime_grasp_restoration_verified']=True
    else: result['raw_episode_id']+=1
    with pytest.raises(ValueError):validate_source_exclusion(result,sample,panel['fingerprint'],provenance)
    if damage not in ('hash','row_hash','source_identity'):
        with pytest.raises(ArtifactValidationError):EvaluationQueue._validate_result(result,sample['task'],sample['task_instance_id'],sample['rollout_id'])


def test_modern_source_cannot_be_declared_legacy_missing(tmp_path):
    _,panel,cache,_=legacy_source(tmp_path)
    cache['grasp_serialization_audit']['legacy_missing_assisted_grasp_state']=False
    with pytest.raises(ValueError,match='Native'):make_source_exclusion(cache,panel['samples'][0],panel['fingerprint'])


def test_complete_older_baseline_with_no_native_proof_is_rejected(tmp_path):
    sample=_sample();panel=_summary_panel([sample]);manifest=_summary_manifest(panel)
    manifest['state_protocol']='recorded_native_fields_v1'
    _write_summary_result(tmp_path/'results',sample,_valid_result(sample,panel['fingerprint']))
    summary=summarize(panel,tmp_path/'results',checkpoint='/checkpoint',require_videos=False,run_manifest=manifest)
    assert not summary['complete'] and summary['invalid_results']
    assert summary['state_fidelity']['verified'] is False


@pytest.mark.parametrize('damage',['missing','unverified','short_cap','wrong_count'])
def test_hard_group_selection_rejects_unverified_or_short_baseline(damage):
    panel,summary=inputs()
    if damage=='missing':summary.pop('state_fidelity')
    elif damage=='unverified':summary['state_fidelity']['verified']=False
    elif damage=='short_cap':summary['sample_max_steps']=256
    else:summary['state_fidelity']['eligible_results_verified']=19
    with pytest.raises(ValueError,match='recorded-native'):plans.freeze_plan(summary,panel)


def test_source_excluded_starts_remain_blacklisted_after_collection_plan_reload(tmp_path):
    panel,summary=inputs()
    summary['source_exclusions']=[{'sample_id':'s0','task':'task','subtask':'pick up','reason':'legacy_missing_assisted_grasp_state'}]
    plan=plans.freeze_plan(summary,panel,max_starts=2,trials=2)
    assert plan['baseline_source_excluded_sample_ids']==['s0']
    assert not plan['baseline_setup_excluded_sample_ids']
    assert 's0' not in {t['sample_id'] for t in plan['trials']}
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan))
    assert plans.load_plan(path,panel)[0]==plan
    plan['baseline_source_excluded_sample_ids'].append('s0')
    plan['fingerprint']=plans.digest({k:v for k,v in plan.items() if k!='fingerprint'})
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError,match='excluded'):plans.load_plan(path,panel)


def test_cpu_prepublication_preserves_earlier_modern_job_and_excludes_exact_legacy_instance(tmp_path):
    from subtask_source_exclusion import prepublish_source_exclusions
    sample,panel,cache,manifest=legacy_source(tmp_path)
    modern=copy.deepcopy(sample)
    modern.update(sample_id='modern-first', task_instance_id=6, episode_index=2, raw_episode_id=140020)
    modern_path=runner.raw_hdf5_path(tmp_path/'raw',modern)
    __import__('shutil').copyfile(runner.raw_hdf5_path(tmp_path/'raw',sample),modern_path)
    with h5py.File(modern_path, 'r+') as f:
        f['data'].attrs['scene_file']=json.dumps({'state':{'registry':{'object_registry':{'robot':{'controller_groups':{}}}}}})
    panel['samples'].insert(0,modern)
    panel['episodes'].insert(0,dict(task=modern['task'],task_instance_id=6,episode_index=2,raw_episode_id=140020,sample_ids=['modern-first']))
    manifest.update(expected_episode_jobs=2,expected_segment_rollouts=2,expected_annotated_segments=2,
                    raw_data_root=str(tmp_path/'raw'),attempt_root=str(tmp_path/'attempts'),
                    queue_db=str(tmp_path/'queue.sqlite3'),results_root=str(tmp_path/'results'),
                    jobs=[{'task':s['task'],'instance_id':s['task_instance_id'],'sample_ids':[s['sample_id']]} for s in panel['samples']])
    manifest['raw_preflight']=preflight_panel(panel,tmp_path/'raw')
    queue=EvaluationQueue(manifest['queue_db'])
    queue.initialize([(s['task'],s['task_instance_id']) for s in panel['samples']],max_attempts=2)
    report=prepublish_source_exclusions(manifest,queue,panel)
    assert report['source_excluded_episode_jobs']==1 and report['source_excluded_samples']==1
    assert report['episodes'][0]['instance_id']==7
    assert queue.counts()['counts']['succeeded']==1 and queue.counts()['counts']['pending']==1
    lease=queue.claim(modern['task'],'gpu-worker',manifest['attempt_root'])
    assert lease.instance_id==6
    assert not list((tmp_path/'results').rglob('*.mp4'))


def test_cpu_prepublication_rejects_changed_source_before_consuming_lease(tmp_path):
    from subtask_source_exclusion import prepublish_source_exclusions
    sample,panel,cache,manifest=legacy_source(tmp_path)
    manifest.update(raw_data_root=str(tmp_path/'raw'),attempt_root=str(tmp_path/'attempts'),
                    queue_db=str(tmp_path/'queue.sqlite3'),results_root=str(tmp_path/'results'))
    queue=EvaluationQueue(manifest['queue_db']);queue.initialize([(sample['task'],7)],max_attempts=2)
    manifest['raw_preflight']['episodes'][0]['sha256']='b'*64
    with pytest.raises(ValueError,match='preflight'):prepublish_source_exclusions(manifest,queue,panel)
    assert queue.counts()['counts']['pending']==1
