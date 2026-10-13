"""Prepare a cloud RGB metadata overlay and frozen indexes; no training."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile

from prepare_retraining_data import Preparation, save, sha


def prepare(cache, output):
    prep = Preparation(cache, output)
    root = output / 'original_expert'
    if (root / 'transfer_receipt.json').exists():
        return json.loads((root / 'transfer_receipt.json').read_text())
    root.mkdir(parents=True, exist_ok=True)
    source_prefix = 'wangfaye/embodied/datasets/behavior-1k/behavior-1k_2026-challenge-demos_20260728235542/'
    from subtask_trajectory import CAMERAS
    import oss2
    objects = []
    for number, obj in enumerate(oss2.ObjectIterator(prep.bucket, prefix=source_prefix, max_keys=1000), 1):
        rel = obj.key[len(source_prefix):]
        if (rel.startswith(('meta/', 'data/')) or
                any(rel.startswith('videos/' + camera + '/') for camera in CAMERAS)):
            objects.append({'key': obj.key, 'relative_path': rel, 'bytes': obj.size,
                            'etag': obj.etag, 'last_modified': obj.last_modified})
        if number % 1000 == 0:
            print(json.dumps({'phase': 'listing_original', 'objects_seen': number}), flush=True)
    if not objects or not any(r['relative_path'].startswith('data/') for r in objects):
        raise ValueError('Original expert source is incomplete')
    save(root / 'source_objects.json', {'bucket': 'traffic-model', 'prefix': source_prefix,
                                       'objects': objects, 'data_reused_not_copied': True})
    overlay = root / 'overlay'
    if overlay.exists():
        shutil.rmtree(overlay)
    overlay.mkdir()
    def fetch(record):
        path = overlay / record['relative_path']
        path.parent.mkdir(parents=True, exist_ok=True)
        response = prep.bucket.get_object(record['key'], headers={'If-Match': record['etag']})
        try:
            content = response.read()
        finally:
            response.resp.response.close()
        assert len(content) == record['bytes']
        path.write_bytes(content)
        return {'path': record['relative_path'], 'source_sha256': sha(path)}
    metas = [r for r in objects if r['relative_path'].startswith('meta/')]
    with ThreadPoolExecutor(max_workers=8) as pool:
        original_metadata = list(pool.map(fetch, metas))
    info_path = overlay / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features'] = {k: v for k, v in info['features'].items()
                        if v.get('dtype') != 'video' or k in CAMERAS}
    assert {k for k, v in info['features'].items() if v.get('dtype') == 'video'} == set(CAMERAS)
    save(info_path, info)
    # Preserve canonical task names from the already verified source metadata.
    known = Path.home() / 'code/Behavior-1k Challenge/BEHAVIOR-VLAs/eval_results/subtask_improvement_goal/original_dataset_metadata'
    canonical = {r['task_index']: r for r in map(json.loads, (known / 'meta/tasks.jsonl').read_text().splitlines())}
    tasks_path = overlay / 'meta/tasks.jsonl'
    tasks = list(map(json.loads, tasks_path.read_text().splitlines()))
    for task in tasks:
        existing = canonical[task['task_index']]
        assert task['task'] == existing['task']
        task['task_name'] = existing['task_name']
    tasks_path.write_text(''.join(json.dumps(r) + '\n' for r in tasks))
    inputs = Path.home() / 'code/Behavior-1k Challenge/BEHAVIOR-VLAs/eval_results/subtask_improvement_goal/original_training_inputs'
    labels = inputs / 'subtask_segments.v1.parquet'
    assert sha(labels) == '870086137b5fac4ff8a701e6e539cdc99ee43d3efacb81028a2ae0b4f78fff2d'
    shutil.copyfile(labels, overlay / 'meta/original_subtask_segments.parquet')
    filled = Path.home() / 'code/Behavior-1k Challenge/analysis/retraining_20261013/leading_fill_v1/subtask_segments.leading_fill_v1.parquet'
    shutil.copyfile(filled, overlay / 'meta/subtask_segments.parquet')
    save(overlay / 'rgb_view_manifest.json', {'kind': 'b1k_cloud_expert_rgb_metadata_overlay',
        'source': 'oss://traffic-model/' + source_prefix, 'source_metadata': original_metadata,
        'source_data_unchanged': True, 'source_videos_unchanged': True,
        'original_labels_sha256': sha(labels), 'filled_labels_sha256': sha(filled),
        'metadata_only_overlay': True, 'training_executed': False})
    import pyarrow.parquet as pq
    all_ids = sorted(r['episode_index'] for p in (overlay / 'meta/episodes').glob('chunk-*/*.parquet')
                     for r in pq.read_table(p, columns=['episode_index']).to_pylist())
    recovered_ids = list(map(int, (filled.parent / 'dropped_episode_indices.txt').read_text().splitlines()))
    from openpi.training.recovery_index import prepare_index, canonical_digest, validate_index
    indexes = []
    for kind, ids in [('recovered', recovered_ids), ('replay', sorted(set(all_ids)-set(recovered_ids)))]:
        index = prepare_index(overlay, labels, ids, repo_id='behavior-1k/2026-challenge-demos',
                              root_id='expert_original', source_kind=kind,
                              held_out_instances=prep.inputs['held_out_task_instances'],
                              fill_leading_with_first_subtask=(kind == 'recovered'))
        index['dataset_root'] = '/prepared/expert_original'
        for item in index['provenance']['source_files']:
            if item['role'] == 'labels':
                item['path'] = 'meta/original_subtask_segments.parquet'
        index['provenance']['labels_path'] = 'meta/original_subtask_segments.parquet'
        index['fingerprint'] = canonical_digest({k: v for k, v in index.items() if k != 'fingerprint'})
        validate_index(index)
        save(overlay / (kind + '_index.json'), index)
        indexes.append(index)
    assert sum(len(i['episodes']) for i in indexes) == 19898
    save(root / 'metadata_validation.json', {'source_metadata_files_verified': len(metas),
        'source_objects': len(objects), 'source_bytes': sum(r['bytes'] for r in objects),
        'original_episodes': len(all_ids), 'training_episodes': 19898,
        'indexes': {i['source_kind']: i['provenance']['statistics'] for i in indexes},
        'real_video_reader_pending': True, 'training_executed': False})
    package = root / 'expert_metadata.tar.gz'
    with tarfile.open(package, 'w:gz') as tar:
        tar.add(overlay, arcname='expert_original')
        tar.add(root / 'source_objects.json', arcname='source_objects.json')
        tar.add(root / 'metadata_validation.json', arcname='metadata_validation.json')
    uploaded = prep.upload(package, prep.prefix + 'expert/expert_metadata.tar.gz')
    result = {'metadata_prepared_and_transferred': True, 'target': uploaded,
              'source_data_reused': True, 'replay_episodes': 15949, 'recovered_episodes': 3949,
              'actual_video_reader_pending': True, 'data_ready': False, 'training_executed': False}
    save(root / 'transfer_receipt.json', result)
    return result


if __name__ == '__main__':
    print(json.dumps(prepare(Path(sys.argv[1]), Path(sys.argv[2]))), flush=True)
