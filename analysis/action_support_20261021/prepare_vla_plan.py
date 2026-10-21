"""Freeze VLA pilot anchors using only existing local expert columns/metadata."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow.parquet as pq

from train_activity import data

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'analysis/training_boundary_audit_20261019'))
from prepare_probe_cache import (DEFAULT_ARCHIVE, DEFAULT_ARCHIVE_SHA256,
                                 canonical, metadata_plan, sha_file)


def main():
    output = HERE / 'vla_pilot'
    output.mkdir(exist_ok=True)
    rows = data()
    meta = ROOT / 'analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1/dataset/meta'
    tasks = {r['task_index']: r for r in map(json.loads, (meta / 'tasks.jsonl').read_text().splitlines())}
    index = ROOT / 'analysis/retraining_20261016_quality_taskbalanced_v1/formal_stage/bundle/inputs/compiled/indexes/0000_replay.json'
    episodes = {e['episode_index']: e for e in json.loads(index.read_text())['episodes']}
    segments = pq.read_table(meta / 'original_subtask_segments.parquet',
                             filters=[('episode_index', 'in', sorted({r['episode'] for r in rows}))]).to_pylist()
    by_episode = {}
    for segment in segments:
        by_episode.setdefault(segment['episode_index'], []).append(segment)
    fold = {}
    for task in sorted({r['task'] for r in rows}):
        eps = sorted({r['episode'] for r in rows if r['task'] == task},
                     key=lambda e: hashlib.sha256(str(e).encode()).hexdigest())
        fold.update({e: i % 5 for i, e in enumerate(eps)})
    keys = []
    for row in rows:
        ep, frame = row['episode'], row['frame']
        matching = [s for s in by_episode[ep] if s['start'] <= frame < s['end']]
        if len(matching) != 1:
            raise ValueError(f'Expected one existing subtask annotation for {(ep, frame)}')
        task = tasks[row['task']]
        keys.append({'sample_id': hashlib.sha256(canonical({'episode': ep, 'frame': frame})).hexdigest(),
                     'episode_index': ep, 'frame_index': frame,
                     'task_name': task['task_name'], 'task_instance_id': episodes[ep]['task_instance_id'],
                     'subtask': matching[0]['subtask'], 'fold': fold[ep]})
    plan, blobs = metadata_plan(DEFAULT_ARCHIVE, {'sample_keys': keys}, DEFAULT_ARCHIVE_SHA256)
    info = json.loads(blobs['expert_original/meta/info.json'])
    plan.update({'kind': 'support_flow_vla_pilot_data', 'fps': info['fps'],
                 'selection': 'all 5448 previously frozen stride128 anchors; no outcome-based selection',
                 'split': {'train_folds': [2, 3, 4], 'validation_folds': [1], 'test_folds': [0]},
                 'raw_columns_receipt_sha256': sha_file(HERE / 'training_column_receipt.json'),
                 'subtask_metadata_sha256': sha_file(meta / 'original_subtask_segments.parquet'),
                 'image_transform': 'existing CheckpointIO PIL bilinear resize with black padding to224; uint8 RGB',
                 'image_tolerance_seconds': 1e-4, 'new_demonstrations_collected': False})
    raw = output / 'raw_targets.npz'
    if not raw.exists():
        np.savez_compressed(raw,
                            state=np.stack([r['x'][:61] for r in rows]),
                            actions=np.stack([r['future'] for r in rows]),
                            active=np.stack([r['y'] for r in rows]),
                            episode=np.array([r['episode'] for r in rows]),
                            frame=np.array([r['frame'] for r in rows]),
                            fold=np.array([fold[r['episode']] for r in rows]))
    else:
        with np.load(raw) as f:
            np.testing.assert_array_equal(f['episode'], [r['episode'] for r in rows])
            np.testing.assert_array_equal(f['frame'], [r['frame'] for r in rows])
            np.testing.assert_array_equal(f['actions'], np.stack([r['future'] for r in rows]))
    plan['raw_targets'] = {'path': raw.name, 'bytes': raw.stat().st_size, 'sha256': sha_file(raw)}
    target = output / 'data_plan.json'
    if target.exists():
        assert json.loads(target.read_text()) == plan
    else:
        target.write_bytes(canonical(plan) + b'\n')
    groups = {name: sum(fold[r['episode']] in group for r in rows)
              for name, group in [('train', [2, 3, 4]), ('validation', [1]), ('test', [0])]}
    video = [o for o in plan['objects'] if o['relative_path'].endswith('.mp4')]
    summary = {'samples': len(keys), 'episodes': len(fold), 'tasks': sorted({k['task_name'] for k in keys}),
               'splits': groups, 'video_objects': len(video),
               'whole_video_bytes': sum(o['bytes'] for o in video),
               'decoded_uint8_bytes': len(keys) * 3 * 224 * 224 * 3,
               'plan_sha256': sha_file(target), 'scope': 'local plan only; no source downloads or cloud submission'}
    (output / 'data_plan_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
