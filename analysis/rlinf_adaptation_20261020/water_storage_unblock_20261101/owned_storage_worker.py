"""CPU-only measurement of immutable completed outputs; never edits task data."""
import json,os,time,http.client,ssl,urllib.parse
from pathlib import Path


def run(config):
    output=Path(config['output']);deadline=time.time()+config['max_seconds']
    u=urllib.parse.urlsplit(config['put'])
    assert u.scheme=='https' and u.hostname=='aoss-internal.cn-sh-01b.sensecoreapi-oss.cn'
    while time.time()<deadline:
        seen=set();files=[]
        for receipt in (output/'training_receipts').glob('round_*/episode_*.json'):
            try:
                data=json.loads(receipt.read_text());path=Path(data['path'])
                if not path.is_relative_to(output/'training') or path.resolve()!=path:continue
                state=path.stat()
                if not path.is_file() or state.st_nlink!=1:continue
                ident=(state.st_dev,state.st_ino)
                if ident not in seen:
                    seen.add(ident);files.append({'path':str(path),'bytes':state.st_blocks*512})
            except (OSError,ValueError):continue
        for ready in (output/'checkpoints').glob('round_*/ready.json'):
            for name in ['trainable.pt','optimizer.pt']:
                path=ready.parent/name
                try:
                    state=path.stat()
                    if path.resolve()!=path or not path.is_file() or state.st_nlink!=1:continue
                    ident=(state.st_dev,state.st_ino)
                    if ident not in seen:
                        seen.add(ident);files.append({'path':str(path),'bytes':state.st_blocks*512})
                except OSError:continue
        result={'experiment':config['experiment'],'output':str(output),'observed_at':time.time(),
                'method':'committed_regular_files_no_external_hardlinks',
                'committed_bytes':sum(f['bytes'] for f in files),'files':files}
        body=(json.dumps(result,sort_keys=True)+'\n').encode()
        for attempt in range(8):
            connection=http.client.HTTPSConnection(u.hostname,timeout=30,context=ssl.create_default_context())
            try:
                connection.request('PUT',u.path+'?'+u.query,body=body)
                response=connection.getresponse();assert response.status==200;response.read()
                break
            except Exception as error:
                print('measurement upload: '+type(error).__name__,flush=True)
                time.sleep(min(2**attempt,4))
            finally:connection.close()
        print(json.dumps({k:v for k,v in result.items() if k!='files'}),flush=True)
        time.sleep(30)
