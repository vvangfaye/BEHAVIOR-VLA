import hashlib
import io
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent))
import transfer_pipeline as pipeline

class Client:
    def __init__(self,body):self.body=body
    def get_object(self,**kwargs):return {'Body':io.BytesIO(self.body)}
    def head_object(self,**kwargs):return {'ContentLength':100,'ETag':'actual-etag'}
    def generate_presigned_url(self,operation,Params,ExpiresIn):
        return operation+':'+Params['Key']

def sample(language=False):
    index=dict(root_id='full_episode_id',supervision_policy='action_only; policy_subtask_text_is_weak_not_human',
               action_only_episodes=[dict(language_loss_enabled=language,valid_anchors=100,
                task_name='turning_on_radio',task_instance_id=301,content_sha256='content')],
               data_target={'sha256':'data'})
    body=json.dumps(index).encode()
    proof=dict(source_key='source',source_sha256='raw',kind='full_episode',accepted=True,
               root_id='full_episode_id',target=dict(key='data',bytes=100,sha256='data',full_destination_readback=True),
               index_target=dict(key='index',sha256=hashlib.sha256(body).hexdigest(),full_destination_readback=True),
               acceptance=dict(real_lerobot_reader=True,episodes=1,frames=131))
    row=dict(key='source',sha256='raw',episodes=1,frames=131)
    return index,body,proof,row

def test_direct_transfer_uses_prepared_cloud_object():
    index,body,proof,_=sample();client=Client(body)
    item=pipeline.unsigned(proof,client,client)
    assert item['source']['key']=='data'
    assert item['source']['etag']=='actual-etag'
    assert item['index']==index
    assert item['receipt_put'].startswith('put_object:'+pipeline.PREFIX)

def test_full_episode_language_target_is_rejected():
    _,body,proof,_=sample(language=True);client=Client(body)
    with pytest.raises(AssertionError):pipeline.unsigned(proof,client,client)

def test_changed_index_bytes_are_rejected():
    _,body,proof,_=sample();client=Client(body+b' ')
    with pytest.raises(AssertionError):pipeline.unsigned(proof,client,client)

def test_incomplete_sources_never_publish_ready(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    _,_,_,row=sample()
    with pytest.raises(ValueError,match='Missing source receipts'):
        pipeline.finalize({'source_archives':[row]}, {}, {})
    assert not (tmp_path/'_DATA_READY.json').exists()

def test_duplicate_with_any_old_collected_source_stops_publication(tmp_path,monkeypatch):
    monkeypatch.setattr(pipeline,'ROOT',tmp_path)
    index,body,proof,row=sample()
    (tmp_path/'old_collected_content.json').write_text(json.dumps(dict(parent_sha256=pipeline.PARENT_SHA,
        indexes=[dict(episodes=[dict(content_sha256='content')])])))
    identity=hashlib.sha256(b'source').hexdigest()[:20]
    with pytest.raises(AssertionError,match='Duplicate selected content'):
        pipeline.finalize({'source_archives':[row]}, {'source':proof}, {identity:{'index_body':body.decode()}})
    assert not (tmp_path/'_DATA_READY.json').exists()
