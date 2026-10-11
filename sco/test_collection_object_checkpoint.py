"""Exercise complete attempt preservation and multipart content receipts."""
import hashlib
import io
import json
from pathlib import Path
import sys
import urllib.error

import pytest
sys.path[:0]=[str(Path(__file__).parents[1]/'scripts'),str(Path(__file__).parent)]
import collection_checkpoint as checkpoint
from run_full_collection_shard import MultipartArchive


class Response(io.BytesIO):
    def __init__(self,body=b'',headers=None):super().__init__(body);self.headers=headers or {}


def store(monkeypatch,lowercase):
    objects={}
    def open_request(request,timeout):
        if isinstance(request,str):url,method,body,headers=request,'GET',None,{}
        else:url,method,body,headers=request.full_url,request.method,request.data,request.headers
        if method=='PUT':
            data=body.read() if hasattr(body,'read') else body
            assert int(headers['Content-length'])==len(data)
            import base64
            assert headers['Content-md5']==base64.b64encode(hashlib.md5(data).digest()).decode()
            objects[url]=data
        elif url not in objects:raise urllib.error.HTTPError(url,404,'missing',{},None)
        data=objects[url]
        headers={'Content-Length':str(len(data)),'ETag':hashlib.md5(data).hexdigest()}
        if lowercase:headers={k.lower():v for k,v in headers.items()}
        return Response(data if method=='GET' else b'',headers)
    monkeypatch.setattr(checkpoint.urllib.request,'urlopen',open_request)
    return objects


@pytest.mark.parametrize('lowercase',[False,True])
def test_finished_attempt_roundtrip_preserves_raw_and_completion(tmp_path,monkeypatch,lowercase):
    objects=store(monkeypatch,lowercase);trial=tmp_path/'original'/'trial-a';trial.mkdir(parents=True)
    (trial/'completed.json').write_text(json.dumps(dict(success=True,steps=3)))
    (trial/'accepted.json').write_text('{}');(trial/'rollout.hdf5').write_bytes(b'original recorded controls and RGB')
    plan={'fingerprint':'f'*64};urls=lambda name:dict(put=name,get=name,head=name,key=name)
    storage=dict(plan_fingerprint=plan['fingerprint'],source_manifest_sha256='e'*64,
                 trials={'trial-a':dict(archive=urls('https://archive'),receipt=urls('https://receipt'))})
    manifest=tmp_path/'storage.json';manifest.write_text(json.dumps(storage));monkeypatch.setenv('B1K_COLLECTION_OBJECT_MANIFEST',str(manifest))
    checkpoint.save_trial(trial,plan)
    assert (trial/'rollout.hdf5').exists() and not (trial.parent/'trial-a.checkpoint.tar').exists()
    root=tmp_path/'restore';assert checkpoint.restore_trials(storage,root,'task')==['trial-a']
    assert (root/'pi05/task/trials/trial-a/rollout.hdf5').read_bytes()==(trial/'rollout.hdf5').read_bytes()
    objects['https://archive']+=b'corrupt'
    with pytest.raises(ValueError,match='hash changed'):checkpoint.restore_trials(storage,tmp_path/'corrupt','task')


@pytest.mark.parametrize('lowercase',[False,True])
def test_multipart_digest_and_length_checked(lowercase):
    parts={};finished=False
    def request(url,*,method='GET',body=None,headers=None):
        nonlocal finished
        if method=='PUT':
            import base64
            assert headers['Content-MD5']==base64.b64encode(hashlib.md5(body).digest()).decode()
            parts[url]=body;return b'',{('etag' if lowercase else 'ETag'):'"'+hashlib.md5(body).hexdigest()+'"'}
        if method=='POST':finished=True;return b'<CompleteMultipartUploadResult/>',{}
        assert finished
        etag=hashlib.md5(b''.join(hashlib.md5(v).digest() for v in parts.values())).hexdigest()+'-'+str(len(parts))
        return b'',{('content-length' if lowercase else 'Content-Length'):str(sum(map(len,parts.values()))),
                   ('etag' if lowercase else 'ETag'):'"'+etag+'"'}
    urls=dict(parts=['p1','p2','p3'],complete='finish',head='head',key='dataset.tar')
    sink=MultipartArchive(urls,request,part_bytes=4);sink.write(b'0123456789');receipt=sink.finish()
    assert b''.join(parts.values())==b'0123456789' and receipt['parts']==3
    assert receipt['sha256']==hashlib.sha256(b'0123456789').hexdigest()


def test_missing_last_object_receipt_blocks_complete_collection(tmp_path):
    from run_full_collection_shard import verify_trial_objects
    plan=dict(fingerprint='f'*64,trials=[dict(trial_id='a')]);directory=tmp_path/'pi05/task/trials/a';directory.mkdir(parents=True)
    completed=dict(success=False,steps=2048);(directory/'completed.json').write_text(json.dumps(completed))
    storage=dict(source_manifest_sha256='e'*64,trials={'a':dict(receipt=dict(get='receipt'),archive=dict(key='archive',head='head'))})
    def missing(*args,**kwargs):raise FileNotFoundError('missing receipt')
    with pytest.raises(FileNotFoundError):verify_trial_objects(storage,plan,tmp_path,'task',missing)
    receipt=dict(trial_id='a',plan_fingerprint=plan['fingerprint'],source_manifest_sha256='e'*64,
        object_key='archive',completed=completed,accepted=False,archive=dict(bytes=3,etag='digest'))
    def present(url,**kwargs):
        return (json.dumps(receipt).encode(),{}) if url=='receipt' else (b'',{'content-length':'3','etag':'digest'})
    assert verify_trial_objects(storage,plan,tmp_path,'task',present)==['a']
    receipt['completed']['steps']=1
    with pytest.raises(ValueError):verify_trial_objects(storage,plan,tmp_path,'task',present)
