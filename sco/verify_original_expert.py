"""CPU source readback: real LeRobot reader, original parquet and OSS RGB.

Only selected original parquet rows are materialized. Videos are decoded from
their real OSS objects through PyAV; no synthetic frames or copied video data
are used. Full-library metadata and frozen object identities are checked first.
"""
from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path
import re
import shutil
import time

from prepare_retraining_data import Preparation, save, sha


class ObjectReader(io.RawIOBase):
    def __init__(self, bucket, record):
        self.bucket, self.record, self.position = bucket, record, 0
        self.bytes_read = 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.position

    def seek(self, offset, whence=0):
        value = offset if whence == 0 else self.position + offset if whence == 1 else self.record['bytes'] + offset
        if value < 0: raise ValueError('Negative object offset')
        self.position = value
        return value

    def read(self, size=-1):
        length = self.record['bytes'] - self.position
        if size >= 0: length = min(length, size)
        if length <= 0: return b''
        result = self.bucket.get_object(self.record['key'], byte_range=(self.position, self.position + length - 1),
                                       headers={'If-Match': self.record['etag']})
        try: content = result.read()
        finally: result.resp.response.close()
        if len(content) != length: raise ValueError('Incomplete original parquet range')
        self.position += length
        self.bytes_read += length
        return content

    def readinto(self, buffer):
        content = self.read(len(buffer)); buffer[:len(content)] = content
        return len(content)


def sample_cases(indexes, *, task=None, limit=None):
    cases=[]
    for kind,index in indexes.items():
        for task_name in sorted({r['task_name'] for r in index['episodes']}):
            episode=min((r for r in index['episodes'] if r['task_name']==task_name),key=lambda r:r['length'])
            cases.append({'kind':kind,'task':task_name,'episode':episode['episode_index']})
    if task:cases=[c for c in cases if c['task']==task]
    if limit:cases=cases[:limit]
    if not cases:raise ValueError('No original-data readback cases matched')
    return cases


def run(cache, output, limit=None, task=None):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
    from lerobot.datasets import dataset_reader as reader_module
    from openpi.training.recovery_index import verify_source_files
    from subtask_trajectory import CAMERAS, array

    prep = Preparation(cache, output / 'reader_lock')
    original = output / 'original_expert'
    overlay = original / 'overlay'
    records = json.loads((original / 'source_objects.json').read_text())['objects']
    by_path = {r['relative_path']: r for r in records}
    indexes = {kind: json.loads((overlay / (kind + '_index.json')).read_text()) for kind in ('replay', 'recovered')}
    for index in indexes.values(): verify_source_files(index, dataset_root=overlay)
    metadata = [r for path in sorted((overlay / 'meta/episodes').glob('chunk-*/*.parquet'))
                for r in pq.read_table(path).to_pylist()]
    episodes = {r['episode_index']: r for r in metadata}
    info = json.loads((overlay / 'meta/info.json').read_text())
    assert len(metadata) == 20000 and sum(r['length'] for r in metadata) == 210916774
    assert sum(len(i['episodes']) for i in indexes.values()) == 19898
    assert len(set(i['task_name'] for i in indexes['replay']['episodes'])) == 100
    # Check every metadata reference against the complete live object listing.
    referenced = set()
    for ep in metadata:
        referenced.add(info['data_path'].format(chunk_index=ep['data/chunk_index'], file_index=ep['data/file_index']))
        for camera in CAMERAS:
            key = 'videos/' + camera
            referenced.add(info['video_path'].format(video_key=camera, chunk_index=ep[key+'/chunk_index'],
                                                       file_index=ep[key+'/file_index']))
    assert referenced <= by_path.keys()
    assert all(by_path[p]['bytes'] > 0 for p in referenced)
    cases=sample_cases(indexes,task=task,limit=limit)
    work = cache / 'original_reader_work'
    if work.exists(): shutil.rmtree(work)
    work.mkdir()
    shutil.copytree(overlay / 'meta', work / 'meta')
    original_decode = reader_module.decode_video_frames
    checked_video_keys = set()

    def decode_remote(path, *args, **kwargs):
        relative = Path(path).relative_to(work).as_posix()
        record = by_path[relative]
        head = prep.bucket.head_object(record['key'])
        assert head.etag.strip('"').lower() == record['etag'].strip('"').lower()
        assert head.content_length == record['bytes']
        url = prep.bucket.sign_url('GET', record['key'], 1200)
        # Keep the library's real PyAV implementation and timestamp matching.
        frames = original_decode(url, *args, **kwargs)
        checked_video_keys.add(relative)
        return frames

    reader_module.decode_video_frames = decode_remote
    accepted = output / 'reader_receipts_v2'
    accepted.mkdir(exist_ok=True)
    try:
        for number, case in enumerate(cases):
            receipt_path = accepted / (case['kind'] + '_' + case['task'] + '.json')
            if receipt_path.exists() and json.loads(receipt_path.read_text()).get('accepted'): continue
            ep = episodes[case['episode']]
            save(output/'reader_progress.json', {'current':case,'completed':len(list(accepted.glob('*.json'))),
                'expected':151,'phase':'source_readback','updated_at':time.time(),'training_executed':False})
            data_path = info['data_path'].format(chunk_index=ep['data/chunk_index'], file_index=ep['data/file_index'])
            record = by_path[data_path]
            source = ObjectReader(prep.bucket, record)
            table = pq.read_table(source, filters=[('episode_index', '=', ep['episode_index'])])
            assert len(table) == ep['length']
            local = work / data_path
            local.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(table, local)
            assert table.equals(pq.read_table(local))
            meta = LeRobotDatasetMetadata('behavior-1k/2026-challenge-demos', work)
            reader = reader_module.DatasetReader(meta, work, [ep['episode_index']], tolerance_s=1e-4,
                video_backend='pyav', delta_timestamps={'action': [i/30 for i in range(32)]}, image_transforms=None)
            reader.load_and_activate()
            assert reader.num_frames == ep['length']
            raw = reader.hf_dataset.select_columns(['action','observation.state','episode_index','frame_index','task_index'])[:]
            actions, states = array(raw['action']), array(raw['observation.state'])
            assert actions.shape == (ep['length'],23) and states.shape == (ep['length'],61)
            assert np.isfinite(actions).all() and np.isfinite(states).all()
            assert array(raw['episode_index']).reshape(-1).tolist() == [ep['episode_index']] * ep['length']
            assert array(raw['frame_index']).reshape(-1).tolist() == list(range(ep['length']))
            assert array(raw['task_index']).reshape(-1).tolist() == [ep['task_index']] * ep['length']
            intervals = [s for s in indexes[case['kind']]['intervals'] if s['episode_index']==ep['episode_index']]
            sample_frames = sorted({intervals[0]['start'], intervals[-1]['end']-32})
            for frame in sample_frames:
                row = reader.get_item(frame)
                assert np.array_equal(array(row['action']), actions[frame:frame+32])
                assert np.array_equal(array(row['observation.state']), states[frame])
                assert not array(row['action_is_pad']).any()
                for camera in CAMERAS:
                    image = array(row[camera]); shape = info['features'][camera]['shape']
                    assert image.shape == (3,shape[0],shape[1]) and np.isfinite(image).all()
                    assert image.min()>=0 and image.max()<=1
            save(receipt_path, {**case, 'accepted': True, 'acceptance_version':2,
                'source_parquet_etag':record['etag'],'sampled_table_sha256':sha(local),
                'task_identity_verified':True,
                'parquet_range_bytes':source.bytes_read,'all_selected_rows_checked':ep['length'],
                'decoded_samples':len(sample_frames),'cameras':list(CAMERAS),
                'real_lerobot_reader':True,'real_pyav_remote_video':True,
                'full_library_video_scan':False,'training_executed':False,'verified_at':time.time()})
            print(json.dumps({'phase':'original_reader', 'completed':number+1,'total':len(cases),**case}),flush=True)
            del reader, table, raw
            local.unlink()
    finally:
        reader_module.decode_video_frames = original_decode
    receipts = [json.loads(p.read_text()) for p in accepted.glob('*.json')]
    complete = len(receipts) == 151 and limit is None and task is None
    report = {'accepted':complete, 'metadata_episodes':len(metadata),'metadata_frames':210916774,
        'replay_episodes':15949,'recovered_episodes':3949,'holdout_episodes':102,
        'referenced_nonempty_source_objects':len(referenced),'per_task_source_readbacks':len(receipts),
        'expected_readbacks':151,'real_lerobot_reader':True,'full_library_video_scan':False,
        'original_cloud_data_reused':True,'training_executed':False,'verified_at':time.time()}
    save(original / 'reader_validation.json',report)
    if complete:
        target = prep.upload(original/'reader_validation.json',prep.prefix+'expert/reader_validation.json')
        report['target'] = target
        save(original/'reader_validation.json',report)
        transfer=json.loads((original/'transfer_receipt.json').read_text())
        transfer.update(actual_video_reader_pending=False,data_ready=True,reader_validation=target)
        save(original/'transfer_receipt.json',transfer)
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int);p.add_argument('--task')
    a=p.parse_args()
    try: print(json.dumps(run(a.cache,a.output,a.limit,a.task)),flush=True)
    except Exception as error:
        message=re.sub(r'https?://[^\s\"\']+','[redacted URL]',str(error))
        save(a.output/'reader_failure.json',{'type':type(error).__name__,'message':message[:2000],'training_executed':False})
        print(type(error).__name__,message[:2000],flush=True)
        raise SystemExit(1) from None
