"""Four bounded CPU transfer streams; verify entire objects before source release."""
import hashlib,json,os,pathlib,time,threading
from concurrent.futures import ThreadPoolExecutor

def run(config):
    exec(config['helpers_source'],globals())
    root=pathlib.Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    started=time.time();released=[];retained=[];active={};lock=threading.RLock()
    def publish(stage,**extra):
        with lock:
            out={'stage':stage,'observed_at':time.time(),'released_files':len(released),
                 'released_allocated_bytes':sum(x['allocated_bytes'] for x in released),
                 'released':list(released),'retained':list(retained),'active':dict(active),**extra}
            save(root/'status.json',out);request(config['status_put'],'PUT',payload=encode(out));return out
    def one(item):
        p=pathlib.Path(item['path'])
        if time.time()-started>config['deadline_seconds']:
            with lock:retained.append({'path':str(p),'reason':'bounded deadline'})
            return
        try:
            assert p.is_relative_to('/mnt/afs/260010165/behavior_1k')
            assert any(p.is_relative_to(x) for x in config['allowed_roots'])
            assert p.resolve()==p and p.suffix in ('.pt','.safetensors')
            before=identity(p)
            if not item.get('identity_at_start'):
                assert all(before[k]==item[k] for k in ('size','mtime_ns','nlink')), 'Source changed since inventory'
            assert before['nlink']==1
            with lock:active[str(p)]='uploading'
            publish('archiving')
            uploaded=request(item['put'],'PUT',path=p)
            assert uploaded['size']==before['size']
            with lock:active[str(p)]='verifying'
            publish('archiving')
            request(item['get'],'GET',expected=uploaded)
            assert identity(p)==before and sha_file(p)==uploaded['sha256']
            manifest={'source':str(p),'identity':before,'bucket':'260010165','key':item['key'],
                      'sha256':uploaded['sha256'],'full_origin_readback':True,
                      'restore':'Download key into original path after creating parent; verify size and SHA256.',
                      'verification_at':time.time()}
            raw=encode(manifest);digest=hashlib.sha256(raw).hexdigest()
            request(item['manifest_put'],'PUT',payload=raw)
            request(item['manifest_get'],'GET',expected={'size':len(raw),'sha256':digest})
            complete=encode({'manifest_sha256':digest,'source_sha256':uploaded['sha256'],'verified':True})
            request(item['complete_put'],'PUT',payload=complete)
            request(item['complete_get'],'GET',expected={'size':len(complete),'sha256':hashlib.sha256(complete).hexdigest()})
            assert identity(p)==before
            save(p.with_name(p.name+'.aoss.json'),manifest);p.unlink()
            with lock:
                released.append({'path':str(p),'allocated_bytes':before['blocks']*512,
                                 'size':before['size'],'key':item['key'],'sha256':uploaded['sha256']})
                active.pop(str(p),None)
            publish('archiving')
        except BaseException as e:
            with lock:
                retained.append({'path':str(p),'error_type':type(e).__name__,
                    'reason':str(e)[:300] if '://' not in str(e) else 'Object transfer failed; source retained'})
                active.pop(str(p),None)
            publish('archiving')
    publish('starting',total_files=len(config['files']))
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one,config['files']))
    publish('complete' if not retained else 'partial',elapsed_seconds=time.time()-started)
    if retained:raise RuntimeError('Some archive sources were retained; inspect audit status')
