"""Archive an explicit cold-file manifest; release only verified, leased files."""
import base64
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import io
import json
import os
from pathlib import Path
import ssl
import tarfile
import threading
import time
import urllib.parse


def read_small(url, limit=16*1024**2):
    parsed=urllib.parse.urlsplit(url)
    assert parsed.scheme=='https' and parsed.hostname in {
        'aoss.cn-sh-01b.sensecoreapi-oss.cn','aoss-internal.cn-sh-01b.sensecoreapi-oss.cn'}
    conn=http.client.HTTPSConnection(parsed.hostname,timeout=30,context=ssl.create_default_context())
    try:
        conn.request('GET',parsed.path+'?'+parsed.query,headers={'Cache-Control':'no-cache'})
        response=conn.getresponse()
        assert response.status==200 and not response.getheader('Via')
        assert 'HIT' not in (response.getheader('X-Cache') or '').upper()
        assert response.getheader('Content-Encoding') in (None,'identity')
        raw=response.read(limit+1);assert len(raw)<=limit
        assert len(raw)==int(response.getheader('Content-Length'))
        return raw
    finally:conn.close()


def run(config):
    exec(config['helpers_source'],globals())
    root=Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    account_root=Path(config.get('account_root','/mnt/afs/260010165'))
    started=time.time();lock=threading.RLock();released=[];retained=[];active={}
    metadata={}

    def publish(stage):
        with lock:
            report={'stage':stage,'observed_at':time.time(),'elapsed_seconds':time.time()-started,
                'released_files':len(released),'released_allocated_bytes':sum(x['allocated_bytes'] for x in released),
                'released':list(released),'retained':list(retained),'active':dict(active),
                'plan_sha256':config['plan_sha256'],'metadata':metadata}
            save(root/'status.json',report)
            request(config['status_put'],'PUT',payload=encode(report))

    def valid_source(item):
        p=Path(item['path'])
        assert p.is_absolute() and p.resolve()==p
        assert p.is_relative_to(account_root)
        assert any(p.is_relative_to(r) for r in config['allowed_roots'])
        assert not any(p.is_relative_to(r) for r in config['protected_roots'])
        state=identity(p)
        assert state['nlink']==1
        assert all(state[k]==item[k] for k in ('size','mtime_ns','nlink'))
        return p,state

    def release_lease(index):
        until=min(started+config['deadline_seconds'],time.time()+600)
        while time.time()<until:
            try:
                lease=json.loads(read_small(config['lease_get']))
                if (lease.get('plan_sha256')==config['plan_sha256'] and lease.get('expires_at',0)>time.time()
                        and index in lease.get('allowed_indices',[])):
                    return lease
            except Exception:pass
            threading.Event().wait(3)
        raise RuntimeError('No fresh dependency-checked release lease; source retained')

    def archive_metadata():
        contents={};links={};total=0
        for source in config['metadata_roots']:
            for folder,dirs,files in os.walk(source,followlinks=False):
                for name in list(dirs):
                    p=Path(folder)/name
                    if p.is_symlink():
                        links[str(p.relative_to(account_root))]=os.readlink(p)
                        dirs.remove(name)
                for name in files:
                    p=Path(folder)/name
                    arc=str(p.relative_to(account_root))
                    if p.is_symlink():links[arc]=os.readlink(p);continue
                    state=identity(p)
                    if state['size']>=32*1024**2:continue
                    raw=p.read_bytes();assert identity(p)==state
                    total+=len(raw);assert total<128*1024**2,'Metadata bundle limit exceeded'
                    contents[arc]=(raw,state)
        manifest={'files':{p:{'size':len(v[0]),'sha256':hashlib.sha256(v[0]).hexdigest(),'mode':v[1]['mode']}
                           for p,v in contents.items()},'symlinks':links,'plan_sha256':config['plan_sha256']}
        buffer=io.BytesIO()
        with tarfile.open(fileobj=buffer,mode='w:gz',compresslevel=1) as archive:
            for name,(raw,state) in contents.items():
                info=tarfile.TarInfo(name);info.size=len(raw);info.mode=state['mode']
                archive.addfile(info,io.BytesIO(raw))
            raw=encode(manifest);info=tarfile.TarInfo('ARCHIVE_METADATA_MANIFEST.json');info.size=len(raw)
            archive.addfile(info,io.BytesIO(raw))
        payload=buffer.getvalue();assert len(payload)<128*1024**2
        expected={'size':len(payload),'sha256':hashlib.sha256(payload).hexdigest()}
        request(config['metadata_put'],'PUT',payload=payload)
        recovered=read_small(config['metadata_get'],128*1024**2);assert hashlib.sha256(recovered).hexdigest()==expected['sha256']
        with tarfile.open(fileobj=io.BytesIO(recovered),mode='r:gz') as archive:
            seen=set()
            for member in archive:
                assert member.isfile() and not Path(member.name).is_absolute() and '..' not in Path(member.name).parts
                raw=archive.extractfile(member).read()
                if member.name=='ARCHIVE_METADATA_MANIFEST.json':assert json.loads(raw)==manifest
                else:
                    assert member.name in manifest['files'];meta=manifest['files'][member.name]
                    assert len(raw)==meta['size'] and hashlib.sha256(raw).hexdigest()==meta['sha256'];seen.add(member.name)
            assert seen==set(manifest['files'])
        marker=encode({'verified':True,'archive':expected,'members_verified':len(contents),'plan_sha256':config['plan_sha256']})
        request(config['metadata_complete_put'],'PUT',payload=marker)
        assert read_small(config['metadata_complete_get'])==marker
        metadata.update(expected,members=len(contents),symlinks=len(links),complete=True)
        save(root/'metadata_manifest.json',manifest)

    def one(item):
        index=item['index']
        try:
            if time.time()-started>config['deadline_seconds']:raise TimeoutError('Archive deadline')
            p,before=valid_source(item)
            with lock:active[str(index)]={'path':str(p),'stage':'hashing' if item.get('existing_object') else 'uploading'}
            publish('archiving')
            part_records=[]
            if item.get('sharded'):
                def part(spec):
                    with p.open('rb') as stream:
                        stream.seek(spec['offset']);raw=stream.read(spec['size'])
                    assert len(raw)==spec['size']
                    expected={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
                    result=request(spec['put'],'PUT',payload=raw)
                    assert result==expected
                    del raw
                    for attempt in range(8):
                        try:
                            recovered=read_small(spec['get'],spec['size']);break
                        except (OSError,http.client.HTTPException):
                            if attempt==7:raise
                            threading.Event().wait(min(2**attempt,4))
                    assert len(recovered)==expected['size']
                    assert hashlib.sha256(recovered).hexdigest()==expected['sha256']
                    record={k:spec[k] for k in ['number','offset','size','key']}
                    record['sha256']=expected['sha256']
                    return record,recovered
                digest=hashlib.sha256();count=0;specs=iter(item['parts']);pending=deque()
                with ThreadPoolExecutor(max_workers=4) as pool:
                    for _ in range(4):
                        spec=next(specs,None)
                        if spec is not None:pending.append(pool.submit(part,spec))
                    while pending:
                        record,recovered=pending.popleft().result()
                        assert record['offset']==count and record['number']==len(part_records)
                        digest.update(recovered);count+=len(recovered);part_records.append(record)
                        del recovered
                        with lock:active[str(index)].update(stage='uploading_and_verifying_parts',read_bytes=count,total_bytes=before['size'],parts_verified=len(part_records))
                        publish('archiving')
                        spec=next(specs,None)
                        if spec is not None:pending.append(pool.submit(part,spec))
                assert count==before['size']
                uploaded={'size':count,'sha256':digest.hexdigest()}
            elif item.get('existing_object'):
                uploaded={'size':before['size'],'sha256':sha_file(p)}
            else:
                uploaded=request(item['put'],'PUT',path=p)
            assert uploaded['size']==before['size'] and identity(p)==before
            with lock:active[str(index)]['stage']='verifying'
            publish('archiving')
            last_report=[0.]
            def progress(done,total):
                with lock:active[str(index)].update(read_bytes=done,total_bytes=total)
                if time.time()-last_report[0]>=15 or done==total:
                    publish('archiving');last_report[0]=time.time()
            if item.get('sharded'):
                pass  # Every part was fully read back and hashed in file order above.
            elif config.get('range_readback', False):
                # Hash every byte in order; range requests only bound retry cost.
                ranged_get(item['get'],uploaded,fallback=item.get('get_fallback'),progress=progress)
            else:
                request(item['get'],'GET',expected=uploaded,progress=progress)
            assert identity(p)==before and sha_file(p)==uploaded['sha256']
            manifest={'source':str(p),'identity':before,'bucket':'260010165','key':item['key'],
                'sha256':uploaded['sha256'],'full_origin_readback':True,'plan_sha256':config['plan_sha256'],
                'metadata_key':config['metadata_key'],
                'restore':'Restore metadata.tar.gz into a new root, download each listed key to its source-relative path, verify size and SHA256; restore symlinks listed in the metadata manifest.'}
            if part_records:
                manifest.update(storage_kind='ordered_parts',parts=part_records,
                    restore='Download every listed part in number order; verify each size and SHA256; concatenate and verify the full file size and SHA256 before restoring metadata and symlinks.')
            raw=encode(manifest)
            request(item['manifest_put'],'PUT',payload=raw)
            assert read_small(item['manifest_get'])==raw
            complete=encode({'verified':True,'manifest_sha256':hashlib.sha256(raw).hexdigest(),'source_sha256':uploaded['sha256']})
            request(item['complete_put'],'PUT',payload=complete)
            assert read_small(item['complete_get'])==complete
            with lock:active[str(index)]['stage']='waiting_for_release_lease'
            publish('archiving')
            lease=release_lease(index)
            assert identity(p)==before
            save(p.with_name(p.name+'.aoss.json'),manifest)
            p.unlink()
            with lock:
                released.append({'index':index,'path':str(p),'allocated_bytes':before['blocks']*512,
                                 **uploaded,'key':item['key'],'dependency_audit_at':lease['observed_at']})
                active.pop(str(index),None)
            publish('archiving')
        except BaseException as error:
            with lock:
                retained.append({'index':index,'path':item['path'],'type':type(error).__name__,
                    'reason':str(error)[:300] if '://' not in str(error) else 'Object request failed; source retained'})
                active.pop(str(index),None)
            publish('archiving')

    publish('metadata')
    archive_metadata()
    with ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(one,config['files']))
    publish('complete' if not retained else 'partial')
    if retained:raise RuntimeError('Some source files retained; inspect archive status')
