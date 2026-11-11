"""Pipeline verified public checkpoint files to the existing private bucket."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

manifest=json.loads((HERE/'download_manifest.json').read_text())
items=[x for x in manifest['items'] if x['destination'].startswith('checkpoints/')]
client=aoss_client()
prefix='behavior1k/serf-reproduction-20261025/released-assets/'


def upload(item):
    path=HERE/'assets'/item['destination']
    # Downloaders promote verified files with os.replace, never write here.
    assert path.is_file() and path.stat().st_size==item['size']
    sha=hashlib.sha256(path.read_bytes()).hexdigest()
    if item['sha256']:assert sha==item['sha256']
    key=prefix+item['destination']
    try:
        old=client.head_object(Bucket='260010165',Key=key)
        if old['ContentLength']==item['size'] and old.get('Metadata',{}).get('sha256')==sha:
            return {'key':key,'size':item['size'],'sha256':sha,'state':'existing'}
    except client.exceptions.ClientError as e:
        if e.response['Error']['Code'] not in {'404','NoSuchKey','NotFound'}:raise
    client.upload_file(str(path),'260010165',key,ExtraArgs={'Metadata':{'sha256':sha}})
    head=client.head_object(Bucket='260010165',Key=key)
    assert head['ContentLength']==item['size'] and head.get('Metadata',{}).get('sha256')==sha
    return {'key':key,'size':item['size'],'sha256':sha,'state':'uploaded'}


try:
    submitted=set();futures={};done=[]
    deadline=time.monotonic()+7200
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        while len(done)<len(items):
            assert time.monotonic()<deadline,'Checkpoint staging deadline exceeded'
            for item in items:
                name=item['destination'];path=HERE/'assets'/name
                if name not in submitted and path.is_file() and path.stat().st_size==item['size']:
                    futures[pool.submit(upload,item)]=item;submitted.add(name)
            ready=[f for f in futures if f.done()]
            for future in ready:
                result=future.result();done.append(result);del futures[future]
                print(json.dumps(result),flush=True)
                (HERE/'checkpoint_upload_progress.json').write_text(json.dumps({'complete':len(done)==len(items),'uploaded':done},indent=2)+'\n')
            if len(done)<len(items):time.sleep(3)
    print(json.dumps({'complete':True,'files':len(done)}),flush=True)
finally:client.close()
