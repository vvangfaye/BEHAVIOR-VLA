"""Archive exact inactive files, verify every byte from AOSS, then release source."""
import hashlib,json,os,pathlib,time

def run(config):
    exec(config['helpers_source'],globals())
    root=pathlib.Path(config['output']);root.mkdir(parents=True,exist_ok=True)
    started=time.time();released=[];retained=[]
    def publish(stage,**extra):
        out={'stage':stage,'observed_at':time.time(),'released_files':len(released),
             'released_allocated_bytes':sum(x['allocated_bytes'] for x in released),
             'released':released,'retained':retained,**extra}
        save(root/'status.json',out);request(config['status_put'],'PUT',payload=encode(out));return out
    publish('starting',total_files=len(config['files']))
    for item in config['files']:
        p=pathlib.Path(item['path'])
        if time.time()-started>config['deadline_seconds']:
            retained.append({'path':str(p),'reason':'bounded deadline'});break
        try:
            assert p.is_relative_to('/mnt/afs/260010165/behavior_1k')
            assert any(p.is_relative_to(x) for x in config['allowed_roots'])
            assert p.resolve()==p and p.suffix in ('.pt','.safetensors')
            before=identity(p)
            assert all(before[k]==item[k] for k in ('size','mtime_ns','nlink')), 'Source size/mtime/link count changed since inventory'
            assert before['nlink']==1
            publish('uploading',current_file=str(p),current_bytes=before['size'])
            uploaded=request(item['put'],'PUT',path=p)
            assert uploaded['size']==before['size']
            publish('verifying',current_file=str(p))
            request(item['get'],'GET',expected=uploaded)
            assert identity(p)==before and sha_file(p)==uploaded['sha256']
            manifest={'source':str(p),'identity':before,'bucket':'260010165','key':item['key'],
                      'sha256':uploaded['sha256'],'full_origin_readback':True,
                      'restore':'Download key into original path after creating parent; verify size and SHA256 before using.',
                      'verification_at':time.time()}
            raw=encode(manifest);digest=hashlib.sha256(raw).hexdigest()
            request(item['manifest_put'],'PUT',payload=raw)
            request(item['manifest_get'],'GET',expected={'size':len(raw),'sha256':digest})
            complete=encode({'manifest_sha256':digest,'source_sha256':uploaded['sha256'],'verified':True})
            request(item['complete_put'],'PUT',payload=complete)
            request(item['complete_get'],'GET',expected={'size':len(complete),'sha256':hashlib.sha256(complete).hexdigest()})
            assert identity(p)==before
            save(p.with_name(p.name+'.aoss.json'),manifest)
            p.unlink()
            released.append({'path':str(p),'allocated_bytes':before['blocks']*512,
                             'size':before['size'],'key':item['key'],'sha256':uploaded['sha256']})
            publish('released')
        except BaseException as e:
            retained.append({'path':str(p),'error_type':type(e).__name__,
                             'reason':str(e)[:300] if '://' not in str(e) else 'Object transfer failed; source retained'})
            publish('source_retained_on_error');raise
    publish('complete' if not retained else 'partial',elapsed_seconds=time.time()-started)
