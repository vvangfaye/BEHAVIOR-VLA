#!/usr/bin/env python3
"""Export accepted collection trials and optional original replay to one LeRobot root."""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from subtask_improvement import digest, load_plan
from subtask_sample_panel import load_panel
from subtask_trajectory import CAMERAS, array, inspect_trajectory


def accepted_trials(root,plan,samples):
    allowed={t['trial_id']:t for t in plan['trials']}
    kept,rejected,hashes=[],[],set()
    found=set()
    for completed in sorted(Path(root).glob('pi05/*/trials/*/completed.json')):
        out=completed.parent;trial_id=out.name
        if trial_id not in allowed or trial_id in found:raise ValueError('Unknown or duplicate collected trial')
        found.add(trial_id)
        done=json.loads(completed.read_text())
        if done.get('trial_id') != trial_id:raise ValueError('Completed trial identity differs from its directory')
        if done.get('plan_fingerprint')!=plan['fingerprint']:raise ValueError('Trial belongs to another collection plan')
        if not (out/'accepted.json').exists():
            if done.get('success') is True:raise ValueError('Successful trial lacks an acceptance receipt')
            rejected.append({'trial_id':trial_id,'reason':'failed_or_setup_excluded'});continue
        inspected=inspect_trajectory(out/'trajectory.hdf5')
        if done.get('success') is not True or done.get('metric_eligible') is not True or done.get('steps') != inspected['steps']:
            raise ValueError('Accepted trajectory differs from completion record')
        metadata=inspected['metadata'];trial=allowed[trial_id];sample=samples[trial['sample_id']]
        if metadata['trial']!=trial or metadata['sample']!=sample or metadata['plan_fingerprint']!=plan['fingerprint']:
            raise ValueError('Accepted trajectory provenance differs from frozen plan')
        receipt=json.loads((out/'accepted.json').read_text())
        if receipt.get('content_sha256')!=inspected['content_sha256'] or receipt.get('plan_fingerprint')!=plan['fingerprint']:
            raise ValueError('Accepted receipt does not verify trajectory content')
        video=out/'videos'/f"{sample['task']}_{sample['task_instance_id']}_{sample['rollout_id']}.mp4"
        if not video.is_file() or video.stat().st_size==0:raise ValueError('Accepted trial lacks nonempty rollout video')
        if inspected['content_sha256'] in hashes:
            rejected.append({'trial_id':trial_id,'reason':'duplicate_trajectory'});continue
        hashes.add(inspected['content_sha256'])
        kept.append((out/'trajectory.hdf5',inspected))
    if found!=set(allowed):raise ValueError('Collection is incomplete; do not export partial attempts as a complete dataset')
    if not kept:raise ValueError('Collection produced no successful, unique trajectories; teacher demonstrations are needed')
    return kept,rejected


def image_hwc(image):
    image=array(image)
    if image.ndim==3 and image.shape[0]==3 and image.shape[-1]!=3:image=image.transpose(1,2,0)
    if image.ndim != 3 or image.shape[-1] != 3 or not np.isfinite(image).all():
        raise ValueError('Invalid replay RGB image')
    if np.issubdtype(image.dtype,np.floating):
        if image.min() < 0 or image.max() > 1:raise ValueError('Replay float RGB must be in [0, 1]')
        image=np.rint(image*255).astype(np.uint8)
    if image.dtype != np.uint8:raise ValueError('Replay RGB must decode to uint8')
    return image


def feature_contract(inspected):
    features={'action':{'dtype':'float32','shape':(23,),'names':['action']},
              'observation.state':{'dtype':'float32','shape':tuple(inspected['state_shape']),'names':['proprio']}}
    for name,shape in inspected['camera_shapes'].items():
        features[name]={'dtype':'video','shape':tuple(shape),'names':['height','width','channels']}
    return features


def export_collection(root,output,repo_id,panel,plan,*,base_dataset=None,base_segments=None,base_episodes=(),
                      dataset_class=None,task_prompts=None):
    output=Path(output)
    if output.exists():raise ValueError('Export requires a fresh dataset root')
    samples={s['sample_id']:s for s in panel['samples']}
    accepted,rejected=accepted_trials(root,plan,samples)
    real_writer = dataset_class is None
    if dataset_class is None:
        from lerobot.datasets import LeRobotDataset
        dataset_class=LeRobotDataset
    if task_prompts is None:
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'openpi/src'))
        from openpi.configs.tasks.b1k import TASKS
        task_prompts=TASKS
    first=accepted[0][1];fps=first['metadata']['fps'];features=feature_contract(first)
    ds=dataset_class.create(repo_id=repo_id,root=output,fps=fps,features=copy.deepcopy(features),robot_type='r1pro',use_videos=True)
    provenance,segments,task_ids=[],[],{}
    def commit_episode(task,instance,source_task_index,source_episode,label_rows,length,source_kind,**extra):
        episode=len(provenance)
        ds.save_episode()
        task_ids.setdefault(task,len(task_ids))
        for row in label_rows:
            segments.append({'episode_index':episode,'start':int(row['start']),'end':int(row['end']),'subtask':row['subtask']})
        provenance.append(dict(episode_index=episode,task_name=task,task_index=task_ids[task],
            source_task_index=source_task_index,task_instance_id=instance,source_episode_index=source_episode,
            source_kind=source_kind,length=length,**extra))
    try:
        for path,info in accepted:
            if feature_contract(info)!=features or info['metadata']['fps']!=fps:raise ValueError('Collection feature/frequency contracts differ')
            sample=info['metadata']['sample'];task=sample['task']
            with h5py.File(path,'r') as f:
                for step in range(info['steps']):
                    ds.add_frame({**{k:f[k][step] for k in features},'task':task_prompts[task]})
            commit_episode(task,sample['task_instance_id'],sample['task_index'],sample['episode_index'],
                           [{'start':0,'end':info['steps'],'subtask':sample['subtask']}],info['steps'],'collected',
                           trial_id=info['metadata']['trial']['trial_id'],content_sha256=info['content_sha256'])
        if base_dataset is not None:
            if not base_segments or not base_episodes:raise ValueError('Original replay requires explicit episodes and segment labels')
            base_labels=pq.read_table(base_segments).to_pylist()
            rows=[r for p in sorted((Path(base_dataset)/'meta/episodes').glob('chunk-*/*.parquet')) for r in pq.read_table(p).to_pylist()]
            meta={r['episode_index']:r for r in rows}
            held_out={tuple(k) for k in plan['held_out_instances']}
            for old in base_episodes:
                info=meta[old];task=info['tasks'][0] if 'tasks' in info else info['task_name']
                if task not in task_prompts:
                    reverse={v:k for k,v in task_prompts.items()};task=reverse[task]
                if (task,info['task_instance_id']) in held_out:raise ValueError('Original replay includes reserved evaluation instance')
                base=dataset_class(repo_id='original/replay',root=Path(base_dataset),episodes=[old],download_videos=False,video_backend='pyav')
                if base.meta.fps!=fps:raise ValueError('Original replay frequency differs from collection')
                labels=[r for r in base_labels if r['episode_index']==old]
                labels.sort(key=lambda r:r['start'])
                if any(not r['subtask'] or r['end'] <= r['start'] for r in labels):raise ValueError('Empty or reversed original subtask labels')
                if not labels or labels[0]['start']!=0 or labels[-1]['end']!=len(base) or any(a['end']!=b['start'] for a,b in zip(labels,labels[1:])):
                    raise ValueError('Original replay labels do not tile the episode')
                for step in range(len(base)):
                    row=base[step];frame={}
                    for key,feature in features.items():
                        value=image_hwc(row[key]) if key in CAMERAS else array(row[key]).astype(np.float32)
                        if key in CAMERAS and list(value.shape)!=list(feature['shape']):
                            from PIL import Image
                            h,w,_=feature['shape'];value=np.asarray(Image.fromarray(value).resize((w,h)))
                        if tuple(value.shape)!=tuple(feature['shape']) or not np.isfinite(value).all():
                            raise ValueError('Original replay differs from collection state/action dimensions')
                        frame[key]=value
                    ds.add_frame({**frame,'task':task_prompts[task]})
                commit_episode(task,info['task_instance_id'],info['task_index'],old,labels,len(base),'original')
    finally:
        ds.finalize()
    # Attach source grouping to LeRobot's own episode rows. All descendants of
    # one task instance retain the same group for later independent splitting.
    by_episode={r['episode_index']:r for r in provenance}
    episode_files = sorted((output/'meta/episodes').glob('chunk-*/*.parquet'))
    if not episode_files:raise ValueError('LeRobot did not finalize episode metadata')
    seen_episodes = set()
    for path in episode_files:
        rows=pq.read_table(path).to_pylist()
        for row in rows:
            episode_id=row['episode_index']
            if episode_id in seen_episodes:raise ValueError('Duplicate exported episode metadata')
            seen_episodes.add(episode_id)
            source=by_episode[episode_id]
            if row.get('length') != source['length']:raise ValueError('LeRobot episode length differs from executed frames')
            for key in ['task_instance_id','source_task_index','source_episode_index','source_kind']:
                row[key]=source[key]
            row['task_index']=source['task_index']
        pq.write_table(pa.Table.from_pylist(rows),path)
    if seen_episodes != set(by_episode):raise ValueError('Exported episode metadata is incomplete')
    pq.write_table(pa.Table.from_pylist(segments),output/'meta/subtask_segments.parquet')
    (output/'meta/tasks.jsonl').write_text(''.join(json.dumps({'task_index':i,'task_name':task,'task':task_prompts[task]})+'\n' for task,i in task_ids.items()))
    (output/'meta/collection_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (output/'retained_episodes.txt').write_text('\n'.join(str(r['episode_index']) for r in provenance)+'\n')
    new_steps=sum(r['length'] for r in provenance if r['source_kind']=='collected')
    total=sum(r['length'] for r in provenance)
    report=dict(schema_version=1,repo_id=repo_id,plan_fingerprint=plan['fingerprint'],
                episodes=len(provenance),frames=total,new_frames=new_steps,
                new_frame_fraction=new_steps/total,rejected_trials=rejected,
                conditioning=plan['conditioning'],source_groups=len({(r['task_name'],r['task_instance_id']) for r in provenance}),
                label_sha256=digest(segments),gpu_training_verified=False)
    (output/'export_report.json').write_text(json.dumps(report,indent=2)+'\n')
    # This marker verifies export, not training or evaluation improvement.
    (output/'_EXPORTED').write_text(json.dumps({'plan_fingerprint':plan['fingerprint'],'label_sha256':report['label_sha256']})+'\n')
    if real_writer:
        from verify_subtask_collection_export import verify_export
        verification = verify_export(output, repo_id, collection_root=root)
        (output/'format_verification.json').write_text(json.dumps(verification,indent=2)+'\n')
        (output/'_VERIFIED_EXPORT').write_text(json.dumps(dict(plan_fingerprint=plan['fingerprint'],
            label_sha256=report['label_sha256'],kind=verification['kind']))+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['collection-root','output','panel','collection-plan']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--base-dataset',type=Path)
    p.add_argument('--base-segments',type=Path)
    p.add_argument('--base-episodes',type=Path)
    a=p.parse_args();panel=load_panel(a.panel);plan,_=load_plan(a.collection_plan,panel)
    marker=json.loads((a.collection_root/'_SUCCESS').read_text())
    if marker.get('kind') != plan['kind'] or marker.get('plan_fingerprint') != plan['fingerprint'] or marker.get('complete') is not True or marker.get('ready_for_export') is not True:
        p.error('Verified, exportable collection completion marker is required')
    ids=[int(x) for x in a.base_episodes.read_text().split()] if a.base_episodes else []
    print(json.dumps(export_collection(a.collection_root,a.output,a.repo_id,panel,plan,
                    base_dataset=a.base_dataset,base_segments=a.base_segments,base_episodes=ids)))


if __name__=='__main__':main()
