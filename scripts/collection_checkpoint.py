"""Persist each finished attempt in private object storage, with verified content hashes."""
import base64
import hashlib
import json
import os
from pathlib import Path
import tarfile
import urllib.error
import urllib.request


def request(url, *, method='GET', body=None, headers=None):
    """Keep signed URLs out of exceptions and retry only the same object operation."""
    last = None
    for _ in range(3):
        try:
            if hasattr(body,'seek'): body.seek(0)
            with urllib.request.urlopen(urllib.request.Request(url,data=body,method=method,
                    headers=headers or {}),timeout=180) as response:
                return response.read(), {k.lower():v for k,v in response.headers.items()}
        except urllib.error.HTTPError as e:
            if e.code == 404: raise FileNotFoundError('Collection object not found') from None
            last = ('HTTPError',e.code)
        except Exception as e:
            last = (type(e).__name__,None)
    raise RuntimeError('Collection object operation failed: '+repr(last)) from None


def file_hash(path, algorithm='sha256'):
    h=hashlib.new(algorithm)
    with Path(path).open('rb') as source:
        for block in iter(lambda:source.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def upload_file(path, urls):
    path=Path(path)
    size=path.stat().st_size
    if size>=5*1024**3:raise ValueError('Single checkpoint exceeds the object PUT limit')
    md5=base64.b64encode(bytes.fromhex(file_hash(path,'md5'))).decode()
    with path.open('rb') as body:
        _,response=request(urls['put'],method='PUT',body=body,
            headers={'Content-Length':str(size),'Content-MD5':md5})
    _,head=request(urls['head'],method='HEAD')
    if int(head.get('content-length','-1'))!=size:
        raise ValueError('Stored checkpoint has wrong length')
    return dict(bytes=size,sha256=file_hash(path),etag=head.get('etag',response.get('etag')))


def upload_json(value, urls):
    body=(json.dumps(value,sort_keys=True)+'\n').encode()
    request(urls['put'],method='PUT',body=body,
        headers={'Content-Length':str(len(body)),'Content-MD5':base64.b64encode(hashlib.md5(body).digest()).decode()})
    actual,_=request(urls['get'])
    if actual!=body:raise ValueError('Object receipt differs after read-back')


def save_trial(directory, plan):
    manifest=os.environ.get('B1K_COLLECTION_OBJECT_MANIFEST')
    if not manifest:return
    directory=Path(directory)
    storage=json.loads(Path(manifest).read_text())
    assert storage['plan_fingerprint']==plan['fingerprint']
    urls=storage['trials'][directory.name]
    archive=directory.parent/(directory.name+'.checkpoint.tar')
    with tarfile.open(archive,'w') as stream:stream.add(directory,arcname=directory.name,recursive=True)
    receipt=dict(trial_id=directory.name,plan_fingerprint=plan['fingerprint'],
        source_manifest_sha256=storage['source_manifest_sha256'],
        completed=json.loads((directory/'completed.json').read_text()),
        accepted=(directory/'accepted.json').is_file(),object_key=urls['archive']['key'],
        archive=upload_file(archive,urls['archive']))
    upload_json(receipt,urls['receipt'])
    archive.unlink()  # Only this temporary duplicate; the original trajectory stays intact.
    print('OBJECT_CHECKPOINT='+json.dumps({k:receipt[k] for k in ('trial_id','accepted','object_key')}),flush=True)


def restore_trials(storage, results, task):
    results=Path(results);trials=results/'pi05'/task/'trials'
    restored=[]
    for trial_id,urls in storage['trials'].items():
        try:body,_=request(urls['receipt']['get'])
        except FileNotFoundError:continue
        receipt=json.loads(body)
        if (receipt['trial_id']!=trial_id or receipt['plan_fingerprint']!=storage['plan_fingerprint']
                or receipt['source_manifest_sha256']!=storage['source_manifest_sha256']):
            raise ValueError('Stored checkpoint belongs to another frozen collection')
        trials.mkdir(parents=True,exist_ok=True)
        target=trials/trial_id
        if target.exists():raise ValueError('Restore requires a fresh local checkpoint directory')
        archive=trials/(trial_id+'.restore.tar')
        # Stream large checkpoints; do not load RGB histories into Python memory.
        try:
            with urllib.request.urlopen(urls['archive']['get'],timeout=180) as response,archive.open('xb') as out:
                while True:
                    block=response.read(8*1024**2)
                    if not block:break
                    out.write(block)
        except Exception as e:
            raise RuntimeError('Checkpoint download failed: '+type(e).__name__) from None
        if archive.stat().st_size!=receipt['archive']['bytes'] or file_hash(archive)!=receipt['archive']['sha256']:
            raise ValueError('Stored checkpoint hash changed')
        with tarfile.open(archive,'r:') as stream:
            for member in stream.getmembers():
                p=Path(member.name)
                if (p.is_absolute() or '..' in p.parts or not p.parts or p.parts[0]!=trial_id
                        or not (member.isfile() or member.isdir())):
                    raise ValueError('Unsafe checkpoint archive')
            stream.extractall(trials)
        archive.unlink();restored.append(trial_id)
    return restored
