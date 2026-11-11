"""Four exact zero-GPU Spot builds for one immutable snapshot of successful data."""
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[1]
REPO=WORKSPACE/'BEHAVIOR-VLAs'
sys.path.insert(0,str(REPO/'sco'))
from video_eval_adapter import aoss_client,cli_json,environment,CLI
from build_retraining_parallel import balanced_parts
from launch_retraining_cloud import frozen_tar


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');temp.chmod(0o600);temp.replace(path)


def internal_client(public):
    import boto3
    from botocore.config import Config
    creds=public._request_signer._credentials.get_frozen_credentials()
    return boto3.client('s3',endpoint_url='https://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn',region_name='cn-sh-01b',
        aws_access_key_id=creds.access_key,aws_secret_access_key=creds.secret_key,aws_session_token=creds.token,
        config=Config(signature_version='s3v4',s3={'addressing_style':'path'},proxies={}))


def verify_job(job,record):
    assert job['name']==record['name'] and job['ownership']['user_name']=='260010165'
    assert job['workspace_name']=='share-space' and job['resource_pool']['name']=='share-cluster'
    assert job['scheduling']['quota_type']=='SPOT'
    if record.get('uid'):assert job['uid']==record['uid']
    specs=[s for role in job['roles'] for s in role['resource_spec']]
    assert len(specs)==1 and specs[0]['name']=='n6ls.iu.i40.8c32g'
    assert int(specs[0]['limits'].get('nvidia.com/gpu',0))==0
    return dict(name=job['name'],uid=job['uid'],state=job['state'],observed_at=time.time(),gpu_count=0)


def describe(record):
    return verify_job(cli_json(['acp','jobs','describe',record['name'],'--workspace-name=share-space','-o','json']),record)


def main():
    frozen=json.loads((ROOT/'source_archives.json').read_text());rows=frozen['source_archives']
    assert rows and all(r['kind']=='full_episode' for r in rows)
    assert frozen['authorization']['cpu_only'] and frozen['authorization']['training_allowed'] is False
    public=aoss_client();internal=internal_client(public)
    prefix='behavior1k/retraining_20261015_round3_v2/batch-0/'
    private=ROOT/'private';private.mkdir(exist_ok=True);private.chmod(0o700)
    def sign(op,key,exclusive=False):
        params={'Bucket':'260010165','Key':key}
        if exclusive:params['IfNoneMatch']='*'
        return internal.generate_presigned_url(op,Params=params,ExpiresIn=172800)
    def asset(path,name):
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();key=prefix+'inputs/'+digest+'/'+name
        try:head=public.head_object(Bucket='260010165',Key=key)
        except public.exceptions.ClientError as error:
            if error.response['ResponseMetadata']['HTTPStatusCode']!=404:raise
            public.put_object(Bucket='260010165',Key=key,Body=raw,IfNoneMatch='*',Metadata={'sha256':digest})
            head=public.head_object(Bucket='260010165',Key=key)
        assert head['ContentLength']==len(raw) and head['Metadata']['sha256']==digest
        return dict(name=name,key=key,get=sign('get_object',key),sha256=digest,bytes=len(raw))
    names=['sco/prepare_retraining_data.py','sco/build_retraining_cloud.py','sco/build_retraining_parallel.py',
           'scripts/subtask_trajectory.py','scripts/full_episode_trajectory.py','scripts/export_full_episode_collection.py',
           'openpi/src/openpi/__init__.py','openpi/src/openpi/training/recovery_index.py','openpi/src/openpi/configs/tasks/b1k.py']
    archive=private/'code.tar.gz'
    frozen_tar(archive,[(name,REPO/name) for name in names]);code=asset(archive,'code.tar.gz')
    path=ROOT/'build_launch.json'
    if path.exists():
        plan=json.loads(path.read_text())
        assert plan['source_fingerprint']==frozen['fingerprint']
        for row in plan['shards']:
            if row.get('creation_attempted'):
                row['latest_observation']=describe(row)
        save(path,plan);print(json.dumps({'existing_jobs_reconciled':True,'new_creations':0}));return
    plan=dict(kind='round3_successful_full_episode_build',source_fingerprint=frozen['fingerprint'],created_at=time.time(),
              cpu_jobs=4,gpu_count=0,cpu_processes_per_job=1,maximum_job_seconds=86400,
              training_allowed=False,raw_sources_retained=True,prefix=prefix,shards=[])
    payloads=[]
    for i,assigned in enumerate(balanced_parts(rows,4)):
        name=f'b1k-r3-{frozen["fingerprint"][:8]}-s{i}-g0';sub=prefix+f'shard-{i}/'
        scope=dict(shard=i,shard_count=4,cpu_processes=1,assigned_keys=[r['key'] for r in assigned])
        inputs=dict(frozen,source_archives=[dict(r,signed_get_url=sign('get_object',r['key'])) for r in assigned],parallel_execution=scope)
        storage=dict(prefix=sub,status_put=sign('put_object',sub+'status.json'),outputs={},already_transferred={},
                     child_status_put=[sign('put_object',sub+'process-0.json')])
        keys=[sub+'cloud_prepared_manifest.json']
        for row in assigned:
            identity=hashlib.sha256(row['key'].encode()).hexdigest()[:20]
            keys.extend([sub+'receipts/'+identity+'.json',sub+'indexes/full_episode_'+identity+'.json',
                         sub+'full_episode/exports/'+row['sha256']+'.tar'])
        storage['outputs']={key:{'put':sign('put_object',key,True),'get':sign('get_object',key)} for key in keys}
        folder=private/f'shard-{i}';folder.mkdir(exist_ok=True)
        save(folder/'private_sources.json',inputs);save(folder/'private_storage.json',storage)
        assets=[code]+[asset(folder/n,n) for n in ['private_sources.json','private_storage.json']]
        root='/mnt/afs/260010165/behavior_1k/retraining_20261015_round3_v2/batch-0/shard-'+str(i)
        payload=dict(name=name,root=root,assets=assets,status_put=storage['status_put'],source_count=len(assigned))
        record=dict(name=name,uid=None,root=root,shard=i,source_count=len(assigned),source_keys=scope['assigned_keys'],
                    source_frames=sum(r['frames'] for r in assigned),source_bytes=sum(r['bytes'] for r in assigned),
                    status_key=sub+'status.json',manifest_key=sub+'cloud_prepared_manifest.json',prefix=sub,
                    quota_type='SPOT',resource_pool='share-cluster',gpu_count=0,cpu_count=8,memory_gib=32,
                    state='prepared_not_submitted',creation_attempted=False)
        plan['shards'].append(record);payloads.append(payload)
    save(path,plan)
    source=(ROOT/'cloud_bootstrap.py').read_text()
    for record,payload in zip(plan['shards'],payloads):
        program=source+'\nrun('+repr(payload)+')\n'
        packed=base64.b64encode(zlib.compress(program.encode(),9)).decode()
        startup='import base64,zlib;exec(compile(zlib.decompress(base64.b64decode('+repr(packed)+')),"round3-data-bootstrap","exec"))'
        save(private/(record['name']+'.payload.json'),payload)
        command='timeout --signal=TERM --kill-after=30 86400 python3 -c '+shlex.quote(startup)
        args=[CLI,'acp','jobs','create','--workspace-name=share-space','--aec2-name=share-cluster',
              '--name='+record['name'],'--job-name='+record['name'],'--training-framework=pytorch',
              '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
              '--worker-spec=n6ls.iu.i40.8c32g','--worker-nodes=1','--quota-type=spot','--retry-times=0',
              '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--command='+command]
        record.update(creation_attempted=True,creation_started_at=time.time(),state='creation_unknown');save(path,plan)
        try:
            cp=subprocess.run(args,env=environment(),capture_output=True,text=True,timeout=90);record['cli_exit_code']=cp.returncode
        except subprocess.TimeoutExpired:record['cli_outcome']='timeout_unknown'
        save(path,plan)
        obs=describe(record);record.update(uid=obs['uid'],state=obs['state'],verified_at=obs['observed_at'])
        save(path,plan);print(json.dumps({'name':record['name'],'uid':record['uid'],'state':record['state'],
                                          'gpu_count':0,'source_count':record['source_count']}),flush=True)


if __name__=='__main__':main()
