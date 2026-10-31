"""Read-only bounded observer for the single authorized benchmark."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
TERMINAL={'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED','DELETED'}


def main():
    receipt=json.loads((HERE/('active_submission.json' if (HERE/'active_submission.json').exists() else 'submission.json')).read_text())
    env={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}
    client=aoss_client()
    deadline=time.monotonic()+14*3600
    while time.monotonic()<deadline:
        try:
            p=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','describe',receipt['job'],
                '--workspace-name=share-space','-o','json'],env=env,capture_output=True,text=True,check=True,timeout=45)
            d=json.loads(p.stdout)
            assert d['uid']==receipt['uid'] and d['ownership']['user_name']=='260010165'
            row={'observed_at':time.time(),'job':receipt['job'],'uid':d['uid'],'state':d['state']}
            try:
                o=client.get_object(Bucket=receipt['bucket'],Key=receipt['aoss_prefix']+'status.json')
                assert o['ContentLength']<8*1024**2
                row['progress']=json.loads(o['Body'].read())
            except client.exceptions.ClientError as error:
                if error.response['Error']['Code'] not in ('NoSuchKey','404'):
                    raise
            (HERE/'latest_status.json').write_text(json.dumps(row,indent=2)+'\n')
            print(json.dumps(row),flush=True)
            if d['state'] in TERMINAL:
                try:
                    o=client.get_object(Bucket=receipt['bucket'],Key=receipt['aoss_prefix']+'result.tar.gz')
                    assert o['ContentLength']<64*1024**2
                    raw=o['Body'].read()
                    attempt=receipt.get('attempt',1)
                    (HERE/('result.tar.gz' if attempt==1 else f'result_a{attempt}.tar.gz')).write_bytes(raw)
                    out=HERE/('retrieved' if attempt==1 else f'retrieved_a{attempt}');out.mkdir(exist_ok=True)
                    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
                        for item in archive:
                            path=Path(item.name)
                            assert item.isfile() and not path.is_absolute() and '..' not in path.parts
                            target=out/path;target.parent.mkdir(parents=True,exist_ok=True)
                            target.write_bytes(archive.extractfile(item).read())
                except client.exceptions.ClientError as error:
                    row['collection_error']=error.response['Error']['Code']
                (HERE/f"final_status_a{receipt.get('attempt',1)}.json").write_text(json.dumps(row,indent=2)+'\n')
                (HERE/'final_status.json').write_text(json.dumps(row,indent=2)+'\n')
                return
        except Exception as error:
            print(json.dumps({'observed_at':time.time(),'observation_error':type(error).__name__}),flush=True)
        time.sleep(45)
    (HERE/'monitor_expired.json').write_text(json.dumps({'observed_at':time.time(),'job_unchanged':True})+'\n')


if __name__=='__main__':
    main()
