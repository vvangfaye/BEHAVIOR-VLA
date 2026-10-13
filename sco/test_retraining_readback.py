import io
import json
from pathlib import Path
import sys
import struct
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'nebula'))
from stage_retraining_data import mounted_object, stage, verified_copy
from verify_original_expert import ObjectReader, sample_cases
from install_reader_timestamp_fix import apply as apply_timestamp_fix


def test_object_reader_seeks_and_keeps_etag_on_every_range():
    raw=b'0123456789'
    class Bucket:
        def get_object(self,key,byte_range,headers):
            assert key=='known' and headers=={'If-Match':'frozen'}
            start,end=byte_range
            result=io.BytesIO(raw[start:end+1])
            result.resp=SimpleNamespace(response=SimpleNamespace(close=lambda:None))
            return result
    source=ObjectReader(Bucket(),{'key':'known','etag':'frozen','bytes':len(raw)})
    assert source.read(3)==b'012'
    assert source.seek(-2,2)==8 and source.read()==b'89'
    assert source.seek(1)==1
    buffer=bytearray(2);assert source.readinto(buffer)==2 and buffer==b'12'


def test_unaccepted_bundle_cannot_stage_or_create_destination(tmp_path):
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'data_ready':False}))
    with pytest.raises(ValueError,match='acceptance'):stage(manifest,tmp_path,tmp_path/'output')
    assert not (tmp_path/'output').exists()


@pytest.mark.parametrize('key',['../outside','/absolute','allowed/../../outside'])
def test_mounted_object_rejects_path_escape(tmp_path,key):
    with pytest.raises(ValueError):mounted_object(tmp_path,{'bucket':'traffic-model','key':key})


def test_transfer_mismatch_never_creates_final_file(tmp_path):
    source=tmp_path/'source';source.write_bytes(b'wrong')
    dest=tmp_path/'destination'
    with pytest.raises(ValueError,match='SHA256'):
        verified_copy(source,dest,{'bytes':5,'sha256':'0'*64})
    assert not dest.exists()


def test_long_video_float32_comparison_can_falsely_reject_correct_frame():
    f32=lambda value:struct.unpack('f',struct.pack('f',value))[0]
    query=2575.7333333333336+f32(8334/30)
    actual=85606/30
    assert abs(f32(query)-f32(actual))>1e-4
    assert abs(query-actual)<1e-4


def test_readback_task_filter_does_not_reduce_the_unfiltered_campaign():
    indexes={'replay':{'episodes':[{'task_name':name,'episode_index':i,'length':10} for i,name in enumerate(['a','z'])]},
             'recovered':{'episodes':[{'task_name':'a','episode_index':2,'length':10}]}}
    assert len(sample_cases(indexes))==3
    assert [c['episode'] for c in sample_cases(indexes,task='a')]==[0,2]
    assert sample_cases(indexes,task='a',limit=1)[0]['episode']==0


def test_timestamp_patch_is_narrow_and_idempotent(tmp_path):
    p=tmp_path/'src/lerobot/datasets/video_utils.py';p.parent.mkdir(parents=True)
    p.write_text('def unrelated():\n    return torch.tensor(timestamps)\n\ndef decode_video_frames_pyav():\n    query_ts = torch.tensor(timestamps)\n    loaded_ts_t = torch.tensor(loaded_ts)\n\ndef after():\n    pass\n')
    first=apply_timestamp_fix(tmp_path);after=p.read_text()
    second=apply_timestamp_fix(tmp_path)
    assert first['tolerance_changed'] is False and after==p.read_text()
    assert after.count('dtype=torch.float64')==2
    assert 'return torch.tensor(timestamps)' in after
    assert second['source_sha256_before']==second['source_sha256_after']
