#!/usr/bin/env python3
"""Choose bounded, fully labelled original replay from measured successful yield."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
import pyarrow.parquet as pq
from subtask_improvement import digest, load_plan
from subtask_sample_panel import file_sha256, load_panel
from stage_subtask_replay import metadata_inventory

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'openpi/src'))
from openpi.training.subtask_data import SubtaskFromSegments

KIND = 'subtask_original_replay_selection'


def read(path): return json.loads(Path(path).read_text())


def source_catalog(metadata_root, label_path, retained_path, held_out):
    metadata_root = Path(metadata_root)
    ids = [int(line) for line in Path(retained_path).read_text().splitlines()]
    if not ids or len(ids) != len(set(ids)) or any(i < 0 for i in ids):
        raise ValueError('Retained original IDs must be unique and nonnegative')
    if not isinstance(held_out, list) or any(not isinstance(g, list) or len(g) != 2 or not isinstance(g[0], str) or not g[0].strip() or type(g[1]) is not int or g[1] < 0 for g in held_out):
        raise ValueError('Held-out instance groups must have canonical task names and integer IDs')
    reserved = {tuple(g) for g in held_out}
    tasks = {}
    for line in (metadata_root / 'meta/tasks.jsonl').read_text().splitlines():
        row = json.loads(line)
        if type(row['task_index']) is not int or row['task_index'] < 0 or row['task_index'] in tasks or not isinstance(row.get('task_name'), str) or not row['task_name']:
            raise ValueError('Original task mapping is missing or ambiguous')
        tasks[row['task_index']] = row['task_name']
    records = {}
    for path in sorted((metadata_root / 'meta/episodes').glob('chunk-*/*.parquet')):
        for row in pq.read_table(path, columns=['episode_index','length','task_index','task_instance_id']).to_pylist():
            i = row['episode_index']
            if type(i) is not int or i < 0 or i in records or type(row['length']) is not int or row['length'] < 1 or type(row['task_instance_id']) is not int or row['task_instance_id'] < 0 or type(row['task_index']) is not int or row['task_index'] not in tasks:
                raise ValueError('Original episode metadata is invalid or duplicated')
            records[i] = dict(episode_index=i, length=row['length'], task_name=tasks[row['task_index']], task_instance_id=row['task_instance_id'])
    if any(i not in records for i in ids): raise ValueError('Retained original IDs contain unknown episodes')
    labels = SubtaskFromSegments.from_parquet(label_path, episodes_index=ids)
    labels.validate_coverage({i:records[i]['length'] for i in ids})
    candidates = [{**records[i], 'subtasks':sorted(set(labels.segments[i][2]))} for i in sorted(ids)
                  if (records[i]['task_name'], records[i]['task_instance_id']) not in reserved]
    if not candidates: raise ValueError('No fully labelled original episodes remain outside holdout')
    files = metadata_inventory(metadata_root)
    source = dict(metadata_files=files, label_sha256=file_sha256(label_path), retained_ids_sha256=file_sha256(retained_path),
                  held_out_instances=sorted(map(list, reserved)), retained_episodes=len(ids), excluded_held_out_episodes=len(ids)-len(candidates))
    by_task = {}
    for row in candidates: by_task.setdefault(row['task_name'],[]).append(row)
    audit = dict(kind='original_replay_candidate_audit', complete=True, source=source,
        eligible_episodes=len(candidates), source_tasks=len(by_task), labels_tile_all_retained_episodes=True,
        candidates_sha256=digest(candidates), actual_collection_selected=False, original_video_reader_verified=False,
        per_task={task:dict(eligible_episodes=len(rows), instances=len({r['task_instance_id'] for r in rows}),
            min_frames=min(r['length'] for r in rows), max_frames=max(r['length'] for r in rows),
            total_frames=sum(r['length'] for r in rows)) for task,rows in sorted(by_task.items())})
    return candidates, audit


def measured_collection(root, panel, plan):
    from collect_b1k_subtasks import validate_collection
    from export_subtask_collection import accepted_trials
    root = Path(root)
    if any((root / name).exists() for name in ('_RUNNING', '_FAILED')):
        raise ValueError('Replay selection cannot use active or failed collection')
    marker = read(root / '_SUCCESS'); contract = read(root / 'collection_contract.json')
    if marker.get('kind') != 'subtask_success_collection' or marker.get('complete') is not True or marker.get('ready_for_export') is not True or marker.get('plan_fingerprint') != plan['fingerprint']:
        raise ValueError('Replay selection requires completed successful new collection')
    if contract.get('kind') != 'subtask_success_collection' or not isinstance(contract.get('run_id'), str) or not contract['run_id'].strip() or contract.get('plan_fingerprint') != plan['fingerprint'] or contract.get('parent_panel_fingerprint') != panel['fingerprint'] or contract.get('checkpoint_identity') != plan['checkpoint_identity']:
        raise ValueError('Collection runtime identity differs from the baseline')
    actual = validate_collection(root,panel,plan)
    if actual != read(root / 'collection_summary.json') or not actual['ready_for_export']:
        raise ValueError('Actual collection evidence does not match its summary')
    accepted,rejected = accepted_trials(root,plan,{s['sample_id']:s for s in panel['samples']})
    if not accepted: raise ValueError('No unique successful new trajectories')
    frames = sum(item['steps'] for _,item in accepted)
    sources = [dict(trial_id=item['metadata']['trial']['trial_id'], frames=item['steps'], content_sha256=item['content_sha256']) for _,item in accepted]
    return frames, sources, rejected


def choose(candidates, *, target_groups, control_tasks, new_frames, target_fraction=.2, seed=42, max_episodes=24, max_frames=200000):
    if type(new_frames) is not int or new_frames <= 0 or not isinstance(target_fraction,(int,float)) or isinstance(target_fraction,bool) or not math.isfinite(target_fraction) or not 0 < target_fraction < 1:
        raise ValueError('Replay ratio requires a positive measured new-frame count and fraction in (0,1)')
    if type(seed) is not int or type(max_episodes) is not int or not 1 <= max_episodes <= 64 or type(max_frames) is not int or not 1 <= max_frames <= 1000000:
        raise ValueError('Original replay resource limits must be positive and bounded')
    if any(not isinstance(g, (tuple,list)) or len(g) != 2 or any(not isinstance(x,str) or not x.strip() for x in g) for g in target_groups):
        raise ValueError('Target groups require canonical task and nonempty subtask strings')
    if any(not isinstance(t,str) or not t.strip() for t in control_tasks):
        raise ValueError('Control tasks require nonempty canonical task strings')
    if any(type(r.get('episode_index')) is not int or r['episode_index'] < 0 or type(r.get('length')) is not int or r['length'] <= 0 or type(r.get('task_instance_id')) is not int or r['task_instance_id'] < 0 or not isinstance(r.get('task_name'),str) or not r['task_name'].strip() or not isinstance(r.get('subtasks'),list) or not r['subtasks'] or any(not isinstance(x,str) or not x.strip() for x in r['subtasks']) for r in candidates):
        raise ValueError('Original replay candidates require valid episode/frame/instance IDs and labels')
    targets = sorted(set(tuple(g) for g in target_groups)); controls = sorted(set(control_tasks))
    target_tasks = {t for t,_ in targets}
    if not targets or not controls or target_tasks & set(controls): raise ValueError('Replay requires target groups and distinct control tasks')
    if len({r['episode_index'] for r in candidates}) != len(candidates): raise ValueError('Duplicate original replay candidate')
    requirements = [('target',task,prompt) for task,prompt in targets] + [('control',task,None) for task in controls]
    def covers(row, requirement):
        _,task,prompt = requirement
        return row['task_name'] == task and (prompt is None or prompt in row['subtasks'])
    relevant = [r for r in candidates if any(covers(r,g) for g in requirements)]
    missing = [g for g in requirements if not any(covers(r,g) for r in relevant)]
    if missing: raise ValueError('No non-held-out fully labelled replay for required task/subtask: ' + str(missing))
    selected = []; selected_ids = set(); uncovered = set(requirements); total = 0
    while uncovered:
        choices = [r for r in relevant if r['episode_index'] not in selected_ids and any(covers(r,g) for g in uncovered)]
        choices.sort(key=lambda r:(r['length']/sum(covers(r,g) for g in uncovered), r['length'], digest([seed,r['episode_index']])))
        row = choices[0]
        if len(selected) >= max_episodes or total + row['length'] > max_frames:
            raise ValueError('Mandatory target/control replay exceeds the explicit pilot budget')
        selected.append(row); selected_ids.add(row['episode_index']); total += row['length']
        uncovered = {g for g in uncovered if not covers(row,g)}
    target_old = new_frames * (1-target_fraction) / target_fraction
    # Full episodes preserve existing source indices and label tiling. Stop when
    # adding an episode no longer moves the measured frame ratio toward target.
    while len(selected) < max_episodes:
        distance = abs(total-target_old)
        choices = [r for r in relevant if r['episode_index'] not in selected_ids and total+r['length'] <= max_frames and abs(total+r['length']-target_old) < distance]
        if not choices: break
        seen = {(r['task_name'],r['task_instance_id']) for r in selected}
        row = min(choices,key=lambda r:(abs(total+r['length']-target_old),(r['task_name'],r['task_instance_id']) in seen,digest([seed,r['episode_index']])))
        selected.append(row); selected_ids.add(row['episode_index']); total += row['length']
    realized = new_frames/(new_frames+total)
    return dict(episodes=sorted(selected,key=lambda r:r['episode_index']), original_replay_frames=total, new_frames=new_frames,
        requested_new_frame_fraction=target_fraction, realized_new_frame_fraction=realized,
        approximate_fraction_met=abs(realized-target_fraction) <= .05,
        whole_episode_constraints_limited_ratio=total>target_old,
        target_groups=[list(g) for g in targets], control_tasks=controls,
        max_episodes=max_episodes,max_original_frames=max_frames,seed=seed)


def freeze_selection(metadata_root,label_path,retained_path,collection_root,panel,collection_plan,control_tasks,**kwargs):
    new_frames,sources,rejected = measured_collection(collection_root,panel,collection_plan)
    candidates,audit = source_catalog(metadata_root,label_path,retained_path,collection_plan['held_out_instances'])
    selected = choose(candidates,target_groups=[(g['task'],g['subtask']) for g in collection_plan['selected_groups']],
                      control_tasks=control_tasks,new_frames=new_frames,**kwargs)
    payload = dict(schema_version=1,kind=KIND,collection_plan_fingerprint=collection_plan['fingerprint'],
        checkpoint_identity=collection_plan['checkpoint_identity'],source_audit=audit,accepted_sources=sources,
        rejected_duplicate_or_failed_sources=rejected,collection_run_id=read(Path(collection_root)/'collection_contract.json')['run_id'],**selected,training_verified=False,success_rate_improvement_verified=False)
    return {**payload,'fingerprint':digest(payload)}


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands=parser.add_subparsers(dest='command',required=True)
    for command in ('audit','select'):
        p=commands.add_parser(command)
        for name in ('metadata-root','labels','retained-episodes','output'):p.add_argument('--'+name,type=Path,required=True)
        if command=='audit':p.add_argument('--held-out-instances',type=Path,required=True)
        else:
            for name in ('panel','collection-plan','collection-root','retest-plan'):p.add_argument('--'+name,type=Path,required=True)
            p.add_argument('--target-new-fraction',type=float,default=.2);p.add_argument('--seed',type=int,default=42)
            p.add_argument('--max-episodes',type=int,default=24);p.add_argument('--max-original-frames',type=int,default=200000)
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Replay audit/selection requires a fresh output root')
    input_roots = [a.metadata_root]
    if a.command == 'select':input_roots.append(a.collection_root)
    if any(a.output.resolve().is_relative_to(p.resolve()) for p in input_roots):
        raise ValueError('Replay selection output must be outside immutable input roots')
    if a.command=='audit':
        _,result=source_catalog(a.metadata_root,a.labels,a.retained_episodes,read(a.held_out_instances))
    else:
        from subtask_paired_retest import validate_plan as validate_retest
        panel=load_panel(a.panel);collection,_=load_plan(a.collection_plan,panel);retest=read(a.retest_plan)
        validate_retest(retest,panel,collection)
        result=freeze_selection(a.metadata_root,a.labels,a.retained_episodes,a.collection_root,panel,collection,
            retest['control_tasks'],target_fraction=a.target_new_fraction,seed=a.seed,max_episodes=a.max_episodes,max_frames=a.max_original_frames)
    a.output.mkdir(parents=True)
    (a.output/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.command=='select':(a.output/'original_episode_ids.txt').write_text(''.join(str(r['episode_index'])+'\n' for r in result['episodes']))
    print(json.dumps({k:result[k] for k in ('kind','eligible_episodes','source_tasks','new_frames','original_replay_frames','realized_new_frame_fraction','approximate_fraction_met') if k in result}))


if __name__=='__main__':main()
