"""Cloud-side, resumable downloads using scoped URLs; never log URLs."""
import concurrent.futures, hashlib, json, os, shutil, time, urllib.request
from pathlib import Path

def run(payload):
    root = Path(payload['root']); root.mkdir(parents=True, exist_ok=True)
    items = payload['items']; records = []
    remaining = [x for x in items if not (
        x.get('expected_sha256') and (root/x['relative_path']).is_file()
        and (root/x['relative_path']).stat().st_size == x['bytes'])]
    assert shutil.disk_usage(root).free > sum(x['bytes'] for x in remaining) + max(
        (x['bytes'] for x in items), default=0)
    def publish(status):
        report = {'status':status,'root':str(root),'files':records,'updated_at':time.time(),'gpu_count':0}
        (root/'download_status.json').write_text(json.dumps(report,indent=2)+'\n')
        try:
            request=urllib.request.Request(payload['report_put'],data=json.dumps(report).encode(),method='PUT')
            with urllib.request.urlopen(request,timeout=30):pass
        except Exception: print('report_upload_failed',flush=True)
    publish('running')
    for item in items:
        relative=Path(item['relative_path']);assert not relative.is_absolute() and '..' not in relative.parts
        target=root/relative;target.parent.mkdir(parents=True,exist_ok=True)
        if item.get('expected_sha256') and target.is_file() and target.stat().st_size==item['bytes']:
            readback=hashlib.sha256()
            with target.open('rb') as source:
                for data in iter(lambda:source.read(8*1024**2),b''):readback.update(data)
            if readback.hexdigest()==item['expected_sha256']:
                records.append({'relative_path':item['relative_path'],'bytes':item['bytes'],
                    'source_etag':item['etag'],'sha256':readback.hexdigest(),'full_local_readback':True,
                    'reused_verified_file':True})
                publish('running')
                print('verified_existing',item['relative_path'],item['bytes'],flush=True)
                continue
        if item.get('crc64'):
            request=urllib.request.Request(item['url'],headers={'Range':'bytes=0-0','If-Match':item['etag']})
            with urllib.request.urlopen(request,timeout=90) as response:
                assert response.status==206
                assert response.headers.get('Content-Range')==f'bytes 0-0/{item["bytes"]}'
                assert response.headers.get('ETag','').strip('"').upper()==item['etag'].strip('"').upper()
                assert response.headers.get('x-oss-hash-crc64ecma')==item['crc64']
                assert len(response.read(2))==1
        chunk_size=16*1024**2;parts=target.parent/(target.name+'.chunks');parts.mkdir(exist_ok=True)
        count=(item['bytes']+chunk_size-1)//chunk_size
        def fetch(index):
            start=index*chunk_size;end=min(item['bytes'],start+chunk_size)-1
            part=parts/str(index);proof=parts/(str(index)+'.sha256')
            if part.exists() and proof.exists() and part.stat().st_size==end-start+1:
                if hashlib.sha256(part.read_bytes()).hexdigest()==proof.read_text():return
            for attempt in range(4):
                try:
                    request=urllib.request.Request(item['url'],headers={'Range':f'bytes={start}-{end}','If-Match':item['etag']})
                    with urllib.request.urlopen(request,timeout=90) as response:
                        assert response.status==206
                        assert response.headers.get('Content-Range')==f'bytes {start}-{end}/{item["bytes"]}'
                        data=response.read(chunk_size+1)
                    assert len(data)==end-start+1
                    temp=parts/(str(index)+'.partial');temp.write_bytes(data);os.replace(temp,part)
                    proof.write_text(hashlib.sha256(data).hexdigest());return
                except Exception:
                    if attempt==3:raise RuntimeError('object_chunk_download_failed') from None
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fetch,range(count)))
        digest=hashlib.sha256();temp=target.with_name(target.name+'.partial')
        with temp.open('wb') as output:
            for index in range(count):
                data=(parts/str(index)).read_bytes();digest.update(data);output.write(data)
        assert temp.stat().st_size==item['bytes']
        if item.get('expected_sha256'):assert digest.hexdigest()==item['expected_sha256']
        os.replace(temp,target)
        readback=hashlib.sha256()
        with target.open('rb') as source:
            for data in iter(lambda:source.read(8*1024**2),b''):readback.update(data)
        assert readback.hexdigest()==digest.hexdigest()
        records.append({'relative_path':item['relative_path'],'bytes':item['bytes'],'source_etag':item['etag'],'sha256':digest.hexdigest(),'full_local_readback':True})
        publish('running');shutil.rmtree(parts)
        print('verified',item['relative_path'],item['bytes'],flush=True)
    publish('complete');(root/'_DOWNLOAD_COMPLETE').write_text(json.dumps(records,indent=2)+'\n')
