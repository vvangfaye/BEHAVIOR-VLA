"""Archive each selected derived file, fully verify AOSS bytes, then unlink it."""
import hashlib,json,os,pathlib,stat,time,traceback
from archive_helpers import encode,identity,request,sha_file


def run(config):
    started=time.time();released=[];failed=[]
    def publish(stage,**extra):
        report={'at':time.time(),'stage':stage,'archive_id':config['archive_id'],
                'released_files':len(released),'released_allocated_bytes':sum(x['allocated'] for x in released),
                'released':released,'failed':failed,**extra}
        request(config['status_put'],'PUT',payload=encode(report))
        return report
    before={str(root):int(__import__('subprocess').check_output(['du','-s','-B1',root],text=True).split()[0]) for root in config['roots']}
    for item in config['files']:
        if time.time()-started>config['deadline_seconds']:
            failed.append({'path':item['path'],'error':'deadline; source retained'});break
        path=pathlib.Path(item['path'])
        try:
            assert any(path.is_relative_to(pathlib.Path(r)) for r in config['roots'])
            assert path.resolve()==path and path.is_file() and not path.is_symlink()
            current=identity(path)
            assert current['nlink']==1
            assert all(current[k]==item[k] for k in ['size','mtime_ns','nlink'])
            assert 'core'==path.name or path.name.startswith('core.') or any(x in ['cache','warp','appdata'] for x in path.parts)
            publish('archiving',current_file=str(path))
            full=hashlib.sha256();parts=[]
            with path.open('rb') as source:
                for index,link in enumerate(item['parts']):
                    data=source.read(config['part_bytes'])
                    assert len(data)==link['bytes']
                    full.update(data)
                    uploaded=request(link['put'],'PUT',payload=data)
                    assert uploaded=={'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}
                    request(link['get'],'GET',expected=uploaded)
                    parts.append({'key':link['key'],**uploaded})
                    del data
                assert source.read(1)==b''
            assert identity(path)==current,'Source changed during archival'
            manifest={'archive_id':config['archive_id'],'bucket':'260010165','source':str(path),
                      'identity':current,'sha256':full.hexdigest(),'parts':parts,
                      'restore':'Concatenate parts in listed order into a NEW file; verify whole-file size and SHA256.',
                      'reason':item['reason']}
            raw=encode(manifest);mh=hashlib.sha256(raw).hexdigest()
            request(item['manifest_put'],'PUT',payload=raw)
            request(item['manifest_get'],'GET',expected={'size':len(raw),'sha256':mh})
            verified={'manifest_sha256':mh,'full_origin_readback':True,'part_count':len(parts),'bytes':current['size'],'verified_at':time.time()}
            raw=encode(verified);vh=hashlib.sha256(raw).hexdigest()
            request(item['verified_put'],'PUT',payload=raw)
            request(item['verified_get'],'GET',expected={'size':len(raw),'sha256':vh})
            complete={'archive_id':config['archive_id'],'manifest_sha256':mh,'verification_sha256':vh}
            raw=encode(complete)
            request(item['complete_put'],'PUT',payload=raw)
            request(item['complete_get'],'GET',expected={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
            # A final full source read prevents deleting content changed since inventory.
            assert sha_file(path)==manifest['sha256'] and identity(path)==current
            path.unlink()
            assert not path.exists()
            released.append({'path':str(path),'allocated':current['blocks']*512,'logical':current['size'],
                             'manifest_key':item['manifest_key'],'sha256':manifest['sha256']})
            publish('source_released')
        except BaseException as error:
            failed.append({'path':str(path),'error_type':type(error).__name__,
                           'error':str(error) if '://' not in str(error) else 'Network error; signed URL omitted'})
            publish('file_retained_on_error')
            break
    after={str(root):int(__import__('subprocess').check_output(['du','-s','-B1',root],text=True).split()[0]) for root in config['roots']}
    result=publish('complete' if not failed else 'partial',directory_before=before,directory_after=after,
                   directory_reduction_bytes=sum(before.values())-sum(after.values()))
    return result
