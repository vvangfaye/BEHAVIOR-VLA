"""Replay isolation, shared file indexing and download integrity checks."""
from pathlib import Path
import hashlib
import io
import json
import sys
from types import SimpleNamespace
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parent))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from stage_subtask_replay import CAMERAS,freeze,required_files,stage,validate


class Bucket:
    bucket_name='fixture'
    def __init__(self,objects):self.objects=objects;self.reads=[];self.truncate=False
    def head_object(self,key):
        data=self.objects[key]
        return SimpleNamespace(content_length=len(data),etag=hashlib.sha256(data).hexdigest())
    def get_object(self,key,headers):
        data=self.objects[key];etag=self.head_object(key).etag
        assert headers['If-Match']==etag
        self.reads.append(key);stream=io.BytesIO(data[:-1] if self.truncate else data)
        stream.etag=etag;stream.resp=SimpleNamespace(response=stream)
        return stream


@pytest.fixture
def inputs(tmp_path):
    root=tmp_path/'metadata';(root/'meta/episodes/chunk-000').mkdir(parents=True)
    info=dict(codebase_version='v3.0',data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
        features={**{key:dict(dtype='video') for key in CAMERAS},'depth':dict(dtype='video',info=dict(is_depth_map=True))})
    (root/'meta/info.json').write_text(json.dumps(info))
    (root/'meta/tasks.jsonl').write_text(json.dumps(dict(task_index=0,task_name='task',task='Full task prompt'))+'\n')
    rows=[]
    for i in range(3):
        row=dict(episode_index=i,length=6,task_instance_id=10+i,task_index=0,tasks=['Full task prompt'],
                 **{'data/chunk_index':0,'data/file_index':2})
        for key in CAMERAS:
            row.update({f'videos/{key}/chunk_index':4,f'videos/{key}/file_index':7,
                        f'videos/{key}/from_timestamp':float(i),f'videos/{key}/to_timestamp':float(i+1)})
        rows.append(row)
    pq.write_table(pa.Table.from_pylist(rows),root/'meta/episodes/chunk-000/file-000.parquet')
    paths,_=required_files(root,[0,1],[])
    bucket=Bucket({'base/'+p:b'data for '+p.encode() for p in paths})
    plan=freeze(root,'oss://fixture/base',[0,1],[['task',12]],bucket)
    return root,bucket,plan


def test_stages_union_of_shared_files_and_preserves_episode_offsets(inputs,tmp_path):
    root,bucket,plan=inputs
    assert len(plan['objects'])==4 and plan['original_replay_frames']==12
    assert any(o['path']=='data/chunk-000/file-002.parquet' for o in plan['objects'])
    report=stage(plan,root,tmp_path/'local',bucket,reserve_bytes=0)
    assert len(bucket.reads)==4 and report['bytes_downloaded']==plan['download_bytes']
    assert all(len(o['sha256'])==64 for o in report['objects'])
    assert not report['reader_verified']
    working=json.loads((tmp_path/'local/meta/info.json').read_text())
    assert 'depth' not in working['features']
    assert 'depth' in json.loads((root/'meta/info.json').read_text())['features']
    assert (tmp_path/'local/selected_episodes.txt').read_text()=='0\n1\n'
    assert pq.read_table(tmp_path/'local/meta/episodes/chunk-000/file-000.parquet').to_pylist()==pq.read_table(root/'meta/episodes/chunk-000/file-000.parquet').to_pylist()


@pytest.mark.parametrize('ids,reserved', [([2],[['task',12]]),([0,0],[]),([9],[]),([True],[])])
def test_rejects_reserved_unknown_duplicate_or_boolean_ids(inputs,ids,reserved):
    root,bucket,plan=inputs
    with pytest.raises(ValueError):freeze(root,'oss://fixture/base',ids,reserved,bucket)


def test_metadata_change_prevents_staging_before_target_created(inputs,tmp_path):
    root,bucket,plan=inputs
    (root/'meta/info.json').write_text('{}')
    with pytest.raises(ValueError,match='metadata changed'):stage(plan,root,tmp_path/'local',bucket,reserve_bytes=0)
    assert not (tmp_path/'local').exists()


def test_changed_object_and_truncated_download_never_get_complete_receipt(inputs,tmp_path):
    root,bucket,plan=inputs
    key='base/'+plan['objects'][0]['path'];original=bucket.objects[key]
    bucket.objects[key]=b'changed'
    with pytest.raises(ValueError,match='changed after selection'):stage(plan,root,tmp_path/'changed',bucket,reserve_bytes=0)
    assert not (tmp_path/'changed/replay_staging_receipt.json').exists()
    bucket.objects[key]=original;bucket.truncate=True
    with pytest.raises(ValueError,match='Incomplete'):stage(plan,root,tmp_path/'truncated',bucket,reserve_bytes=0)
    assert not (tmp_path/'truncated/replay_staging_receipt.json').exists()


def test_replay_plan_fingerprint_and_fresh_target_are_required(inputs,tmp_path):
    root,bucket,plan=inputs
    other=dict(plan,download_bytes=plan['download_bytes']+1)
    with pytest.raises(ValueError,match='fingerprint'):validate(other,root)
    target=tmp_path/'existing';target.mkdir()
    with pytest.raises(ValueError,match='fresh'):stage(plan,root,target,bucket,reserve_bytes=0)
