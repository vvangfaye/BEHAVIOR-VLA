"""Measured replay yield, complete labels, control coverage and holdout isolation."""
import json
from pathlib import Path
import subprocess
import sys
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parent))
import select_subtask_replay as replay
import test_subtask_improvement as fixture
from collect_b1k_subtasks import validate_collection
from stage_subtask_replay import freeze as freeze_staging
from test_stage_subtask_replay import Bucket, CAMERAS


@pytest.fixture
def source(tmp_path):
    root = tmp_path/'source'; (root/'meta/episodes/chunk-000').mkdir(parents=True)
    info = dict(codebase_version='v3.0', data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
        features={k:dict(dtype='video') for k in CAMERAS})
    (root/'meta/info.json').write_text(json.dumps(info))
    (root/'meta/tasks.jsonl').write_text(''.join(json.dumps(dict(task_index=i,task_name=t,task='Human '+t))+'\n' for i,t in enumerate(('task','control'))))
    rows = [dict(episode_index=i,length=10*(i+1),task_index=0 if i<3 else 1,task_instance_id=i,
        tasks=['Human task' if i<3 else 'Human control'],**{'data/chunk_index':0,'data/file_index':i//2}) for i in range(5)]
    for row in rows:
        for k in CAMERAS: row.update({f'videos/{k}/chunk_index':0,f'videos/{k}/file_index':row['episode_index']//2,f'videos/{k}/from_timestamp':float(row['episode_index']),f'videos/{k}/to_timestamp':float(row['episode_index'])+row['length']/30})
    path=root/'meta/episodes/chunk-000/file-000.parquet';pq.write_table(pa.Table.from_pylist(rows),path)
    labels=tmp_path/'labels.parquet'
    pq.write_table(pa.Table.from_pylist([dict(episode_index=r['episode_index'],start=0,end=r['length'],subtask='pick up' if r['task_index']==0 else 'open') for r in rows]),labels)
    retained=tmp_path/'retained.txt';retained.write_text('0\n1\n2\n3\n4\n')
    return root,labels,retained,[['task',0]]


def select(candidates, **kwargs):
    args=dict(target_groups=[('task','pick up')],control_tasks=['control'],new_frames=20,target_fraction=.2)
    args.update(kwargs)
    return replay.choose(candidates,**args)


def test_catalog_checks_all_retained_labels_and_excludes_whole_instance(source):
    root,labels,retained,reserved=source
    candidates,audit=replay.source_catalog(*source)
    assert [r['episode_index'] for r in candidates]==[1,2,3,4]
    assert {r['task_name'] for r in candidates}=={'task','control'}
    assert audit['source']['excluded_held_out_episodes']==1
    assert audit['source']['label_sha256']==replay.file_sha256(labels)
    assert audit['labels_tile_all_retained_episodes']
    assert not audit['actual_collection_selected'] and not audit['original_video_reader_verified']
    assert len(audit['candidates_sha256'])==64


@pytest.mark.parametrize('damage',['duplicate_retained','unknown_retained','label_gap','missing_label','duplicate_episode','bad_task_mapping','negative_instance'])
def test_catalog_rejects_incompatible_source(source,damage):
    root,labels,retained,reserved=source
    if damage=='duplicate_retained':retained.write_text('1\n1\n')
    elif damage=='unknown_retained':retained.write_text('123\n')
    elif damage in ('label_gap','missing_label'):
        rows=pq.read_table(labels).to_pylist()
        if damage=='label_gap':rows[0]['start']=1
        else:rows=rows[1:]
        pq.write_table(pa.Table.from_pylist(rows),labels)
    elif damage=='bad_task_mapping':(root/'meta/tasks.jsonl').write_text('{"task_index":0,"task":"missing canonical name"}\n')
    else:
        path=root/'meta/episodes/chunk-000/file-000.parquet';rows=pq.read_table(path).to_pylist()
        if damage=='duplicate_episode':rows.append(rows[0])
        else:rows[0]['task_instance_id']=-1
        pq.write_table(pa.Table.from_pylist(rows),path)
    with pytest.raises(ValueError):replay.source_catalog(*source)


def test_selection_is_input_order_independent_and_includes_exact_prompt_and_controls(source):
    candidates,_=replay.source_catalog(*source)
    chosen=select(candidates)
    assert chosen==select(list(reversed(candidates)))
    rows=chosen['episodes'];assert {r['task_name'] for r in rows}=={'task','control'}
    assert all(r['episode_index']!=0 for r in rows)
    assert chosen['new_frames']==20 and chosen['original_replay_frames']==90
    assert chosen['realized_new_frame_fraction']==pytest.approx(20/110)
    assert chosen['approximate_fraction_met']
    assert not any(k.startswith('gpu') for k in chosen)
    with pytest.raises(ValueError,match='No non-held-out'):
        select(candidates,target_groups=[('task','unseen exact prompt')])


def test_controls_and_required_groups_cannot_be_silently_dropped_to_meet_budget(source):
    candidates,_=replay.source_catalog(*source)
    for limits in (dict(max_frames=59),dict(max_episodes=1)):
        with pytest.raises(ValueError,match='Mandatory'):select(candidates,**limits)
    with pytest.raises(ValueError,match='distinct control'):select(candidates,control_tasks=['task'])


def test_small_successful_yield_reports_actual_ratio_without_truncating_original(source):
    candidates,_=replay.source_catalog(*source)
    chosen=select(candidates,new_frames=2)
    assert chosen['original_replay_frames']==60 and len(chosen['episodes'])==2
    assert chosen['realized_new_frame_fraction']==pytest.approx(2/62)
    assert chosen['whole_episode_constraints_limited_ratio'] and not chosen['approximate_fraction_met']


@pytest.mark.parametrize('kw',[dict(new_frames=0),dict(new_frames=True),dict(target_fraction=float('nan')),dict(target_fraction=True),dict(target_fraction=1),dict(max_episodes=65),dict(max_frames=1000001),dict(seed=True),dict(target_groups=[('task','')]),dict(control_tasks=[False])])
def test_invalid_or_unbounded_selection_rejected(source,kw):
    candidates,_=replay.source_catalog(*source)
    with pytest.raises(ValueError):select(candidates,**kw)


@pytest.fixture
def collection(tmp_path):
    panel,summary=fixture.inputs();plan=fixture.plan_for(panel,summary,held_out_instances=[('task',0)])
    root=tmp_path/'collection'
    for i,t in enumerate(plan['trials']):fixture.write_trial(root,plan,panel,t,value=i+1)
    (root/'collection_summary.json').write_text(json.dumps(validate_collection(root,panel,plan)))
    (root/'_SUCCESS').write_text(json.dumps(dict(kind='subtask_success_collection',complete=True,ready_for_export=True,plan_fingerprint=plan['fingerprint'])))
    (root/'collection_contract.json').write_text(json.dumps(dict(kind='subtask_success_collection',run_id='synthetic-test',plan_fingerprint=plan['fingerprint'],parent_panel_fingerprint=panel['fingerprint'],checkpoint_identity=plan['checkpoint_identity'])))
    return root,panel,plan


def test_freezes_actual_yield_and_stages_selected_whole_episodes(source,collection):
    root,labels,retained,reserved=source;col,panel,plan=collection
    selected=replay.freeze_selection(root,labels,retained,col,panel,plan,['control'])
    assert selected['new_frames']==8 and len(selected['accepted_sources'])==4
    assert selected['collection_run_id']=='synthetic-test'
    assert selected['fingerprint']==replay.digest({k:v for k,v in selected.items() if k!='fingerprint'})
    assert not selected['training_verified'] and not selected['success_rate_improvement_verified']
    from stage_subtask_replay import required_files
    ids=[r['episode_index'] for r in selected['episodes']]
    paths,_=required_files(root,ids,reserved)
    bucket=Bucket({'source/'+p:b'synthetic bytes' for p in paths})
    staged=freeze_staging(root,'oss://fixture/source',ids,reserved,bucket)
    assert staged['original_replay_frames']==selected['original_replay_frames']
    assert [r['episode_index'] for r in staged['episodes']]==ids
    assert staged.get('reader_verified') is not True


@pytest.mark.parametrize('damage',['running','failed','summary','wrong_panel','wrong_kind','empty_run_id','video','trajectory','zero_success'])
def test_selection_cannot_infer_training_data_from_intent_or_damaged_collection(collection,damage):
    root,panel,plan=collection
    if damage in ('running','failed'):(root/('_'+damage.upper())).touch()
    elif damage=='summary':(root/'collection_summary.json').write_text('{}')
    elif damage in ('wrong_panel','wrong_kind','empty_run_id'):
        path=root/'collection_contract.json';contract=json.loads(path.read_text())
        key,value={'wrong_panel':('parent_panel_fingerprint','other'),'wrong_kind':('kind','other'),'empty_run_id':('run_id','')}[damage]
        contract[key]=value;path.write_text(json.dumps(contract))
    elif damage=='video':next(root.rglob('*.mp4')).unlink()
    elif damage=='trajectory':
        import h5py
        with h5py.File(next(root.rglob('trajectory.hdf5')),'r+') as f:f['action'][0,0]+=1
    else:
        (root/'_SUCCESS').write_text(json.dumps(dict(kind='subtask_success_collection',complete=True,ready_for_export=False,plan_fingerprint=plan['fingerprint'])))
    with pytest.raises(ValueError):replay.measured_collection(root,panel,plan)


def test_cli_does_not_write_audit_into_frozen_input_tree(source,tmp_path):
    root,labels,retained,reserved=source;holdout=tmp_path/'heldout.json';holdout.write_text(json.dumps(reserved))
    target=root/'nested-output'
    result=subprocess.run([sys.executable,str(Path(replay.__file__)),'audit','--metadata-root',str(root),'--labels',str(labels),'--retained-episodes',str(retained),'--held-out-instances',str(holdout),'--output',str(target)],capture_output=True,text=True)
    assert result.returncode!=0 and 'immutable input roots' in result.stderr and not target.exists()
