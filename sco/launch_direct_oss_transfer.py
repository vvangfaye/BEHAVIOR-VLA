"""One zero-GPU CPU transfer job; reuse verified cloud reader runtime read-only."""
import base64
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import zlib

from launch_retraining_cloud import frozen_tar,sha,range_hash,write
from launch_retraining_parallel import environment,reserved_admission
from submit_retraining_cloud_probe import client,CLI,PREFIX


def sign_items(items):
    from relay_retraining_stream import REMOTE_CACHE,SOCKET
    result=subprocess.run(['ssh','-S',SOCKET,'macpro',shlex.join([REMOTE_CACHE+'/env/bin/python',REMOTE_CACHE+'/code/sco/sign_direct_oss_transfers.py'])],
        input=json.dumps(items),capture_output=True,text=True,timeout=60,check=True)
    return json.loads(result.stdout)


def enrich(proof,public,internal,prefix):
    identity=hashlib.sha256(proof['source_key'].encode()).hexdigest()[:20]
    with public.get_object(Bucket='260010165',Key=proof['index_target']['key'])['Body'] as body:index=json.loads(body.read())
    target=proof['target'];head=public.head_object(Bucket='260010165',Key=target['key']);assert head['ContentLength']==target['bytes']
    source={**target,'etag':head['ETag'],'signed_get_url':internal.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':target['key']},ExpiresIn=21600)}
    return {'identity':identity,'proof':proof,'index':index,'source':source,'receipt_put':internal.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':prefix+'receipts/'+identity+'.json'},ExpiresIn=21600)}


def launch(out,pilot):
    path=out/'direct_oss_launch.json'
    previous=json.loads(path.read_text()) if path.exists() else None
    if previous:
        assert previous['state']=='submission_unknown' and not previous.get('attempt_history')
        result=subprocess.run([CLI,'acp','jobs','describe',previous['name'],'--workspace-name=share-space','-o','json'],env=environment(),capture_output=True,text=True,timeout=40)
        assert result.returncode and '404 Not Found' in result.stderr
        write(out/'direct_oss_first_create_not_found.json',{'name':previous['name'],'code':404,'observed_at':time.time(),'evidence':result.stderr})
    public,internal=client(),client(internal=True)
    stamp=previous['name'].removeprefix('behavior-direct-oss-') if previous else datetime.now(timezone.utc).strftime('%m%d-%H%M%S');name='behavior-direct-oss-'+stamp
    prefix=PREFIX+'direct-oss-'+stamp+'/'
    def sign(op,key):return internal.generate_presigned_url(op,Params={'Bucket':'260010165','Key':key},ExpiresIn=43200)
    record={'name':name,'state':'prepared','gpu_count':0,'cpu_count':8,'memory_gib':32,'workspace':'share-space',
            'quota_type':'reserved','resource_pool':'share-cluster','prefix':prefix,'queue_key':prefix+'queue.json','status_key':prefix+'status.json',
            'root':'/mnt/afs/260010165/behavior_1k/retraining_20261013/direct-oss-'+stamp,'training_allowed':False,
            'deadline_epoch':time.time()+12*3600}
    if previous:
        record['attempt_history']=[previous]
        record['quota_type']='spot'
        record['deadline_epoch']=previous['deadline_epoch']
    items=sign_items([enrich(pilot,public,internal,prefix)])
    frozen=json.loads((out/'source_archives.json').read_text())
    queue={'fingerprint':frozen['fingerprint'],'training_allowed':False,'all_sources_prepared':False,'items':items}
    public.put_object(Bucket='260010165',Key=record['queue_key'],Body=json.dumps(queue).encode())
    private=Path('/private/tmp/behavior-direct-oss-private');private.mkdir(mode=0o700,exist_ok=True)
    repo=Path(__file__).resolve().parents[1];code=private/'direct_code.tar.gz'
    frozen_tar(code,[(n,repo/'sco'/n) for n in ['direct_oss_worker.py','prepare_retraining_data.py']])
    key=prefix+'inputs/'+sha(code)+'/code.tar.gz'
    try:public.head_object(Bucket='260010165',Key=key)
    except public.exceptions.ClientError as e:
        if e.response['ResponseMetadata']['HTTPStatusCode']!=404:raise
        public.put_object(Bucket='260010165',Key=key,Body=code.read_bytes(),IfNoneMatch='*')
    head=public.head_object(Bucket='260010165',Key=key);assert range_hash(public,key,code.stat().st_size,head['ETag'])==sha(code)
    control={'queue_get':sign('get_object',record['queue_key']),'status_put':sign('put_object',record['status_key']),
             'deadline_epoch':record['deadline_epoch'],'fingerprint':frozen['fingerprint'],'training_allowed':False}
    payload={'root':record['root'],'code_get':sign('get_object',key),'code_sha256':sha(code),'control':control}
    bootstrap=Path(__file__).with_name('bootstrap_direct_oss_transfer.py').read_text()+'\nrun('+repr(payload)+')\n'
    encoded=base64.b64encode(zlib.compress(bootstrap.encode())).decode()
    command='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))'
    record['admission']=reserved_admission([],additional_gpus=0);record['code_sha256']=sha(code)
    record['state']='submission_intent';write(path,record)
    args=[CLI,'acp','jobs','create','--workspace-name=share-space','--aec2-name=share-cluster','--name='+name,'--job-name='+name,
          '--training-framework=pytorch','--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
          '--worker-spec=n6ls.iu.i40.8c32g','--worker-nodes=1','--quota-type='+record['quota_type'],'--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
          '--command=timeout --signal=TERM --kill-after=30 43200 python3 -c '+shlex.quote(command)]
    result=subprocess.run(args,env=environment(),capture_output=True,text=True,timeout=90)
    if result.returncode:
        import re
        record['create_error']=re.sub(r'https?://[^\s]+','<URL>',result.stderr)[-3000:]
    record.update(state='created_needs_describe' if result.returncode==0 else 'submission_unknown',cli_exit_code=result.returncode);write(path,record)
    if result.returncode:raise RuntimeError('Reconcile exact direct transfer job name; do not retry')
    result=subprocess.run([CLI,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],env=environment(),capture_output=True,text=True,timeout=40,check=True)
    j=json.loads(result.stdout);assert j['ownership']['user_name']=='260010165'
    from bounded_collection_admission import requested_gpus
    assert requested_gpus(j)==0
    record.update(uid=j['uid'],state='submitted',platform_state=j['state']);write(path,record)
    write(private/'queue.json',queue);(private/'queue.json').chmod(0o600)
    print(json.dumps({k:record[k] for k in ['name','uid','platform_state','gpu_count','cpu_count']}))
    return record


if __name__=='__main__':
    out=Path('/Users/faye/code/Behavior-1k Challenge/analysis/retraining_20261013/preparation')
    d=json.loads((out/'parallel/progress_latest.json').read_text())
    pilot=min([r for s in d['shards'] for r in s['receipts']],key=lambda r:r['target']['bytes'])
    launch(out,pilot)
