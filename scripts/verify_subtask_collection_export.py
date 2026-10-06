#!/usr/bin/env python3
"""Verify finalized collection data with real LeRobot readers and training labels."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import pyarrow.parquet as pq
from subtask_improvement import digest
from subtask_trajectory import CAMERAS, array, inspect_trajectory


def verify_export(root, repo_id, *, collection_root=None, horizon=32, dataset_class=None):
    root=Path(root)
    if type(horizon) is not int or horizon <= 0:raise ValueError('Invalid action horizon')
    for name in ('_EXPORTED','export_report.json','meta/info.json','meta/collection_provenance.json','meta/tasks.jsonl','meta/subtask_segments.parquet'):
        if not (root/name).is_file():raise ValueError(f'Export artifact missing: {name}')
    marker=json.loads((root/'_EXPORTED').read_text());report=json.loads((root/'export_report.json').read_text())
    provenance=json.loads((root/'meta/collection_provenance.json').read_text())
    segments=pq.read_table(root/'meta/subtask_segments.parquet').to_pylist()
    if marker.get('label_sha256') != digest(segments) or report.get('label_sha256') != marker['label_sha256'] or marker.get('plan_fingerprint') != report.get('plan_fingerprint'):
        raise ValueError('Export identity or label table changed')
    tasks={row['task_index']:row for row in (json.loads(line) for line in (root/'meta/tasks.jsonl').read_text().splitlines())}
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'openpi/src'))
    from openpi.training.subtask_data import SubtaskFromSegments, episode_lengths_in_dataset_order
    lengths=episode_lengths_in_dataset_order(root)
    labels=SubtaskFromSegments.from_parquet(root/'meta/subtask_segments.parquet');labels.validate_coverage(lengths)
    if sorted(lengths) != list(range(len(lengths))) or [p['episode_index'] for p in provenance] != list(range(len(lengths))):
        raise ValueError('Export episode IDs or provenance are not dense and complete')
    if report.get('frames') != sum(lengths.values()) or report.get('episodes') != len(lengths):
        raise ValueError('Export counts differ from actual metadata')
    if dataset_class is None:
        from lerobot.datasets import LeRobotDataset
        dataset_class=LeRobotDataset
    reader=dataset_class(repo_id=repo_id,root=root,download_videos=False,video_backend='pyav',
                         delta_timestamps={'action':[i/json.loads((root/'meta/info.json').read_text())['fps'] for i in range(horizon)]})
    if len(reader) != report['frames']:raise ValueError('LeRobot frame count differs from export')
    offset=0;decoded_samples=0;source_frames=0
    for source in provenance:
        ep=source['episode_index'];n=lengths[ep]
        if source['length'] != n:raise ValueError('Source episode length differs from metadata')
        row_data=reader.hf_dataset.select_columns(['action','observation.state','episode_index','frame_index','task_index'])[offset:offset+n]
        actions=array(row_data['action']).astype(np.float32);states=array(row_data['observation.state']).astype(np.float32)
        if actions.shape != (n,23) or states.ndim != 2 or not 57 <= states.shape[1] <= 512 or not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError('Invalid executed state/action arrays in finalized data')
        if array(row_data['episode_index']).reshape(-1).tolist() != [ep]*n or array(row_data['frame_index']).reshape(-1).tolist() != list(range(n)):
            raise ValueError('Frame indices do not align with exported episodes')
        if array(row_data['task_index']).reshape(-1).tolist() != [source['task_index']]*n or source['task_index'] not in tasks:
            raise ValueError('Data task indices differ from source task mapping')
        if collection_root is not None and source['source_kind']=='collected':
            paths=list(Path(collection_root).glob(f"pi05/*/trials/{source['trial_id']}/trajectory.hdf5"))
            if len(paths)!=1:raise ValueError('Cannot uniquely locate source collected trajectory')
            checked=inspect_trajectory(paths[0])
            if checked['content_sha256'] != source['content_sha256'] or checked['metadata']['plan_fingerprint'] != report['plan_fingerprint']:
                raise ValueError('Export source differs from its accepted trajectory')
            with h5py.File(paths[0],'r') as f:
                if not np.array_equal(actions,f['action'][:]) or not np.array_equal(states,f['observation.state'][:]):
                    raise ValueError('Export changed observation/action pairing or values')
            source_frames+=n
        for frame in sorted({0,n//2,n-1}):
            row=reader[offset+frame];labelled=labels(row)
            expected_actions=actions[np.minimum(frame+np.arange(horizon),n-1)]
            if not np.array_equal(array(row['action']),expected_actions):raise ValueError('Action horizon crosses an episode boundary or changes executed controls')
            if not np.array_equal(array(row['observation.state']),states[frame]):raise ValueError('Video reader shifted the paired state')
            if labelled['subtask'] != labels.lookup(ep,frame)[0] or row['task'] != tasks[source['task_index']]['task']:
                raise ValueError('Training prompt or subtask labels differ from source mapping')
            for key in CAMERAS:
                pixels=array(row[key]);expected_shape=tuple(reader.meta.features[key]['shape'])
                h,w,c=expected_shape
                if pixels.shape != (c,h,w) or not np.isfinite(pixels).all() or pixels.min()<0 or pixels.max()>1:
                    raise ValueError('Video cannot decode to valid CHW RGB training observations')
            decoded_samples+=1
        offset+=n
    result=dict(schema_version=1,complete=True,kind='collection_export_format_verification',
                plan_fingerprint=report['plan_fingerprint'],episodes=len(lengths),frames=offset,
                action_horizon=horizon,decoded_frame_samples=decoded_samples,
                aligned_source_frames_verified=source_frames,real_lerobot_reader=True,
                label_coverage=True,gpu_training_verified=False,success_rate_improvement_verified=False)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-root',required=True,type=Path);p.add_argument('--repo-id',required=True)
    p.add_argument('--collection-root',type=Path);p.add_argument('--action-horizon',type=int,default=32)
    p.add_argument('--report',required=True,type=Path)
    a=p.parse_args();result=verify_export(a.dataset_root,a.repo_id,collection_root=a.collection_root,horizon=a.action_horizon)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    with a.report.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
