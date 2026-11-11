"""Restore an exact checkpoint into a new experiment-owned path."""
import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main(config):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
    report={'operation':'restore_original_checkpoint','passed':False,'started_at':time.time()}
    root=Path(config['destination']); partial=root.with_name(root.name+'.partial-restore01')
    try:
        if root.exists():
            raise FileExistsError('Destination already exists; inspect it instead of overwriting')
        partial.mkdir(parents=True,exist_ok=False)
        def restore(item):
            relative=Path(item['relative_path'])
            assert not relative.is_absolute() and '..' not in relative.parts
            path=partial/relative;path.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256();total=item['size'];block=8*1024**2
            with path.open('xb') as target:
                for start in range(0,total,block):
                    end=min(total-1,start+block-1)
                    for attempt in range(3):
                        try:
                            req=urllib.request.Request(item['url'],headers={'Range':f'bytes={start}-{end}'})
                            with opener.open(req,timeout=120) as response:
                                data=response.read(end-start+2)
                                if response.status==206:
                                    assert response.headers['Content-Range']==f'bytes {start}-{end}/{total}'
                                else:
                                    assert response.status==200 and start==0 and end+1==total
                            assert len(data)==end-start+1
                            break
                        except Exception:
                            if attempt==2:raise
                    target.write(data);digest.update(data)
            assert digest.hexdigest()==item['sha256'],item['relative_path']
            return {'relative_path':item['relative_path'],'size':total,'sha256':digest.hexdigest()}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            files=list(pool.map(restore,config['files']))
        report.update(passed=True,files=files,destination=str(root),finished_at=time.time())
        (partial/'_RESTORE_VERIFIED.json').write_text(json.dumps(report,indent=2)+'\n')
        partial.rename(root)
    except BaseException as error:
        report.update(error_type=type(error).__name__,error=str(error) if 'http' not in str(error).lower() else 'HTTP transport error; URL omitted')
    raw=json.dumps(report).encode()
    with opener.open(urllib.request.Request(config['put'],data=raw,method='PUT'),timeout=120) as response:
        assert response.status in (200,201,204)
    print(json.dumps({'passed':report['passed'],'error_type':report.get('error_type'),'files':len(report.get('files',[]))}),flush=True)
    if not report['passed']:raise SystemExit(1)
