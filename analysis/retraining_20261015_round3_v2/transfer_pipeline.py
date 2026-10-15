"""One finite metadata controller, four conversion jobs and one direct CPU transfer.

Only the frozen 82-source snapshot is admitted. No raw TAR passes through a Mac,
no collection is changed, and this controller never starts model training.
"""
from concurrent.futures import ThreadPoolExecutor
import base64
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import traceback
import zlib

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]/'BEHAVIOR-VLAs'
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(REPO/'sco'))
from launch_build import aoss_client,internal_client,save,describe,CLI,environment
from launch_retraining_cloud import frozen_tar
from build_retraining_parallel import checked_receipts

PREFIX='behavior1k/retraining_20261015_round3_v2/batch-0/direct-transfer/'
OSS_PREFIX='wangfaye/embodied/datasets/behavior-1k/retrain_20261015_v2/'
PARENT=ROOT.parent/'retraining_20261013/preparation/data_preparation_manifest.json'
PARENT_SHA='bcfef52fd6e8be6892c409161e01e2e773aa2bcee48702544124042dc86b6357'

def remote(request):
    program=(ROOT/'oss_metadata_remote.py').read_text()
    argv=['ssh','-S','none','-o','BatchMode=yes','-o','ConnectTimeout=8',
          '-o','ServerAliveInterval=5','-o','ServerAliveCountMax=2',os.environ.get('BEHAVIOR_PREP_SIGNING_SSH','faye@10.204.55.178'),
          shlex.join(['/Users/faye/Library/Caches/behavior-data-prep-20261013-v1/env/bin/python','-c',program])]
    cp=subprocess.run(argv,input=json.dumps(request),capture_output=True,text=True,timeout=180)
    if cp.returncode:
        raise ConnectionError('Scoped OSS metadata request failed: '+re.sub(r'https?://[^\s]+','<URL>',cp.stderr[-1000:]))
    return json.loads(cp.stdout)

def cloud_json(client,key):
    with client.get_object(Bucket='260010165',Key=key)['Body'] as body:
        data=body.read(16*1024**2+1)
    assert len(data)<=16*1024**2
    return json.loads(data)

def optional_json(client,key):
    try:return cloud_json(client,key)
    except client.exceptions.ClientError as error:
        if error.response['ResponseMetadata']['HTTPStatusCode']==404:return None
        raise

def listed(client,prefix):
    return [o['Key'] for p in client.get_paginator('list_objects_v2').paginate(Bucket='260010165',Prefix=prefix)
            for o in p.get('Contents',[]) if o['Key'].endswith('.json')]

def unsigned(proof,public,internal):
    identity=hashlib.sha256(proof['source_key'].encode()).hexdigest()[:20]
    index=cloud_json(public,proof['index_target']['key'])
    # Hash original index bytes, not a reserialization.
    with public.get_object(Bucket='260010165',Key=proof['index_target']['key'])['Body'] as body:raw=body.read()
    assert hashlib.sha256(raw).hexdigest()==proof['index_target']['sha256']
    assert index['root_id']==proof['root_id']
    assert index['supervision_policy']=='action_only; policy_subtask_text_is_weak_not_human'
    assert all(e['language_loss_enabled'] is False for e in index['action_only_episodes'])
    target=proof['target'];head=public.head_object(Bucket='260010165',Key=target['key'])
    assert head['ContentLength']==target['bytes']
    sign=lambda op,key:internal.generate_presigned_url(op,Params={'Bucket':'260010165','Key':key},ExpiresIn=129600)
    return dict(identity=identity,proof=proof,index=index,
                source=dict(target,etag=head['ETag'],signed_get_url=sign('get_object',target['key'])),
                receipt_put=sign('put_object',PREFIX+'receipts/'+identity+'.json'))

def launch_transfer(public,internal,deadline,fingerprint):
    path=ROOT/'transfer_launch.json'
    if path.exists():
        record=json.loads(path.read_text());obs=describe(record)
        record.update(uid=obs['uid'],latest_observation=obs);save(path,record);return record
    name='b1k-r3-fdf69560-direct-g0'
    record=dict(name=name,uid=None,state='prepared',root='/mnt/afs/260010165/behavior_1k/retraining_20261015_round3_v2/batch-0/direct-transfer',
                prefix=PREFIX,queue_key=PREFIX+'queue.json',status_key=PREFIX+'status.json',deadline_epoch=deadline,
                gpu_count=0,cpu_count=8,memory_gib=32,quota_type='SPOT',resource_pool='share-cluster',training_allowed=False)
    private=ROOT/'private';code=private/'direct_code.tar.gz'
    frozen_tar(code,[(n,REPO/'sco'/n) for n in ['direct_oss_worker.py','prepare_retraining_data.py']])
    raw=code.read_bytes();digest=hashlib.sha256(raw).hexdigest();key=PREFIX+'inputs/'+digest+'/code.tar.gz'
    try:public.put_object(Bucket='260010165',Key=key,Body=raw,IfNoneMatch='*')
    except public.exceptions.ClientError as error:
        if error.response['ResponseMetadata']['HTTPStatusCode'] not in (409,412):raise
    with public.get_object(Bucket='260010165',Key=key)['Body'] as body:assert body.read()==raw
    sign=lambda op,key:internal.generate_presigned_url(op,Params={'Bucket':'260010165','Key':key},ExpiresIn=129600)
    control=dict(queue_get=sign('get_object',record['queue_key']),status_put=sign('put_object',record['status_key']),
                 deadline_epoch=deadline,fingerprint=fingerprint,training_allowed=False)
    payload=dict(root=record['root'],code_get=sign('get_object',key),code_sha256=digest,control=control)
    source=(REPO/'sco/bootstrap_direct_oss_transfer.py').read_text()+'\nrun('+repr(payload)+')\n'
    encoded=base64.b64encode(zlib.compress(source.encode(),9)).decode()
    command='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))'
    save(private/'transfer_payload.json',payload)
    record.update(state='creation_unknown',creation_attempted=True,creation_started_at=time.time(),code_sha256=digest)
    save(path,record) # Persist intent before the only create call.
    argv=[CLI,'acp','jobs','create','--workspace-name=share-space','--aec2-name=share-cluster',
          '--name='+name,'--job-name='+name,'--training-framework=pytorch',
          '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
          '--worker-spec=n6ls.iu.i40.8c32g','--worker-nodes=1','--quota-type=spot','--retry-times=0',
          '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
          '--command=timeout --signal=TERM --kill-after=30 86400 python3 -c '+shlex.quote(command)]
    try:
        cp=subprocess.run(argv,env=environment(),capture_output=True,text=True,timeout=90)
        record['cli_exit_code']=cp.returncode
    except subprocess.TimeoutExpired:record['cli_outcome']='timeout_unknown'
    save(path,record)
    obs=describe(record);record.update(uid=obs['uid'],state=obs['state'],latest_observation=obs)
    save(path,record);return record

def finalize(frozen,proofs,items):
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest()==PARENT_SHA
    old=json.loads(PARENT.read_text())
    checked_receipts(frozen['source_archives'],list(proofs.values()))
    old_content_path=ROOT/'old_collected_content.json'
    if not old_content_path.exists():save(old_content_path,remote(dict(operation='old_content')))
    old_content=json.loads(old_content_path.read_text());assert old_content['parent_sha256']==PARENT_SHA
    seen={e['content_sha256'] for row in old_content['indexes'] for e in row['episodes']}
    reserved=set(map(tuple,old['held_out_task_instances']))
    descriptors=ROOT/'destination_indexes';descriptors.mkdir(exist_ok=True)
    eligible=0
    for row in frozen['source_archives']:
        proof=proofs[row['key']];identity=hashlib.sha256(row['key'].encode()).hexdigest()[:20]
        item=items[identity];raw=item['index_body'].encode()
        assert hashlib.sha256(raw).hexdigest()==proof['index_target']['sha256']
        index=json.loads(raw);assert index['data_target']['sha256']==proof['target']['sha256']
        assert index['supervision_policy']=='action_only; policy_subtask_text_is_weak_not_human'
        assert all(e['language_loss_enabled'] is False for e in index['action_only_episodes'])
        episodes=index['action_only_episodes'];assert len(episodes)==1
        for episode in episodes:
            if episode['valid_anchors']<=0:continue
            assert (episode['task_name'],episode['task_instance_id']) not in reserved
            assert episode['content_sha256'] not in seen,'Duplicate selected content'
            seen.add(episode['content_sha256']);eligible+=1
        target=descriptors/(proof['root_id']+'.json')
        target.write_bytes(raw)
    manifest=copy.deepcopy(old)
    manifest.update(version=2,created_at=time.time(),source_fingerprint=frozen['fingerprint'],
        parent_manifest=dict(bucket='traffic-model',key='wangfaye/embodied/datasets/behavior-1k/retrain_20261013_v1/data_preparation_manifest.json',sha256=PARENT_SHA),
        data_ready=True,training_allowed=False,training_executed=False,
        archives=old['archives']+[proofs[r['key']] for r in frozen['source_archives']],
        source_cutoff=json.loads((ROOT/'training_data_cutoff_authorization.json').read_text()),
        increment=dict(successful_full_episodes=82,frames=825801,training_eligible_episodes=eligible,
                       late_arrivals_deferred=True,raw_sources_retained=True),
        gpu_training_smoke_test='v2_not_run; v1_proof_does_not_accept_changed_mixture')
    counts=manifest['collected_data']['full_episode']
    counts.update(accepted_archives=270,expected_archives=270,accepted_episodes=270,accepted_frames=2529538,
                  training_eligible_episodes=old['collected_data']['full_episode']['training_eligible_episodes']+eligible)
    assert eligible==82
    path=ROOT/'data_preparation_manifest.json'
    if path.exists():manifest=json.loads(path.read_text()) # Immutable publication retry.
    else:path.write_text(json.dumps(manifest,indent=2,sort_keys=True,allow_nan=False)+'\n')
    result=remote(dict(operation='publish',manifest=manifest))
    save(ROOT/'_DATA_READY.json',result)
    return result

def run():
    frozen=json.loads((ROOT/'source_archives.json').read_text())
    assert len(frozen['source_archives'])==82
    plan=json.loads((ROOT/'build_launch.json').read_text())
    assert plan['source_fingerprint']==frozen['fingerprint']
    public=aoss_client();internal=internal_client(public)
    config_path=ROOT/'pipeline_launch.json'
    if config_path.exists():config=json.loads(config_path.read_text())
    else:
        config=dict(created_at=time.time(),deadline_epoch=time.time()+86400,source_fingerprint=frozen['fingerprint'],
                    training_allowed=False,late_arrivals_deferred=True)
    config.update(pid=os.getpid(),started_at=time.time());save(config_path,config)
    queue_path=ROOT/'private/transfer_queue.json'
    queue=json.loads(queue_path.read_text()) if queue_path.exists() else dict(fingerprint=frozen['fingerprint'],training_allowed=False,all_sources_prepared=False,items=[])
    items={item['identity']:item for item in queue['items']};accepted={};cloud={}
    cached_receipts=set();cached_direct=set();builds_done=set();iteration=0
    while time.time()<config['deadline_epoch']:
        if (ROOT/'STOP').exists():raise RuntimeError('User STOP present')
        observations=[]
        failures=[]
        for shard in plan['shards']:
            status=optional_json(public,shard['status_key']) or dict(phase='starting')
            observations.append(dict(name=shard['name'],**status))
            if status.get('phase') in ('needs_diagnostic','bootstrap_needs_diagnostic'):failures.append(dict(name=shard['name'],status=status))
            if status.get('phase')=='all_cloud_exports_verified' and shard['name'] not in builds_done:
                with public.get_object(Bucket='260010165',Key=shard['manifest_key'])['Body'] as body:raw=body.read()
                assert hashlib.sha256(raw).hexdigest()==status['manifest']['sha256']
                manifest=json.loads(raw)
                assert manifest['source_fingerprint']==frozen['fingerprint']
                expected=[r for r in frozen['source_archives'] if r['key'] in set(shard['source_keys'])]
                checked_receipts(expected,manifest['sources'])
                builds_done.add(shard['name'])
            for key in listed(public,shard['prefix']+'receipts/'):
                if key in cached_receipts:continue
                receipt=cloud_json(public,key)
                checked_receipts(frozen['source_archives'],[receipt],require_complete=False)
                cloud[receipt['source_key']]=receipt;cached_receipts.add(key)
                save(ROOT/'cloud_receipts'/(hashlib.sha256(receipt['source_key'].encode()).hexdigest()[:20]+'.json'),receipt)
        new=[proof for key,proof in cloud.items() if hashlib.sha256(key.encode()).hexdigest()[:20] not in items]
        if new:
            with ThreadPoolExecutor(max_workers=4) as pool:unsigned_items=list(pool.map(lambda p:unsigned(p,public,internal),new))
            for item in remote(dict(operation='sign',items=unsigned_items)):items[item['identity']]=item
        queue.update(items=list(items.values()),all_sources_prepared=len(cloud)==82 and len(builds_done)==4)
        if items:
            public.put_object(Bucket='260010165',Key=PREFIX+'queue.json',Body=json.dumps(queue).encode())
            save(queue_path,queue)
            if not (ROOT/'transfer_launch.json').exists():launch_transfer(public,internal,config['deadline_epoch'],frozen['fingerprint'])
            elif not json.loads((ROOT/'transfer_launch.json').read_text()).get('uid'):
                launch_transfer(public,internal,config['deadline_epoch'],frozen['fingerprint'])
        for key in listed(public,PREFIX+'receipts/'):
            if key in cached_direct:continue
            receipt=cloud_json(public,key)
            checked_receipts(frozen['source_archives'],[receipt],require_complete=False)
            assert receipt['transfer_route']=='source_cloud_direct_to_OSS' and receipt['destination_reader_acceptance']['real_lerobot_reader']
            accepted[receipt['source_key']]=receipt;cached_direct.add(key)
            save(ROOT/'destination_receipts'/(hashlib.sha256(receipt['source_key'].encode()).hexdigest()[:20]+'.json'),receipt)
        worker=optional_json(public,PREFIX+'status.json') or dict(phase='not_started')
        if worker.get('error_type') or worker.get('phase')=='bootstrap_needs_diagnostic':failures.append(dict(transfer=worker))
        status=dict(pid=os.getpid(),updated_at=time.time(),phase='conversion_and_direct_upload',
                    frozen_successes=82,cloud_prepared=len(cloud),destination_accepted=len(accepted),
                    completed_build_shards=len(builds_done),builds=observations,transfer=worker,
                    diagnostics=failures,data_ready=False,training_executed=False,raw_tar_bytes_relayed_through_mac=0)
        save(ROOT/'pipeline_status.json',status)
        if len(accepted)==82 and len(builds_done)==4:
            result=finalize(frozen,accepted,items)
            save(ROOT/'pipeline_status.json',dict(status,phase='complete',data_ready=True,publication=result,updated_at=time.time()))
            return
        if failures:
            save(ROOT/'pipeline_diagnostics.json',dict(observed_at=time.time(),failures=failures))
            raise RuntimeError('Data preparation needs diagnosis; preserving exact jobs and outputs')
        if iteration%20==0:
            for record in plan['shards']:
                observation=describe(record)
                record['latest_observation']=observation
                if observation['state'] in ('FAILED','CANCELLED','CANCELED','DELETED','STOPPED'):
                    save(ROOT/'platform_failure.json',observation)
                    raise RuntimeError('Exact data job entered a non-success terminal state')
            save(ROOT/'build_launch.json',plan)
            if (ROOT/'transfer_launch.json').exists():
                record=json.loads((ROOT/'transfer_launch.json').read_text())
                observation=describe(record);record['latest_observation']=observation
                save(ROOT/'transfer_launch.json',record)
                if observation['state'] in ('FAILED','CANCELLED','CANCELED','DELETED','STOPPED'):
                    save(ROOT/'platform_failure.json',observation)
                    raise RuntimeError('Exact transfer job entered a non-success terminal state')
        iteration+=1
        time.sleep(30)
    raise TimeoutError('Finite preparation deadline reached')

if __name__=='__main__':
    with (ROOT/'pipeline.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            # Network retries resume metadata reconciliation only, never re-create jobs.
            from botocore.exceptions import ConnectionClosedError,EndpointConnectionError,ReadTimeoutError,SSLError
            for attempt in range(20):
                try:
                    run();break
                except (ConnectionError,ConnectionClosedError,EndpointConnectionError,ReadTimeoutError,SSLError) as error:
                    save(ROOT/'pipeline_network_retry.json',dict(attempt=attempt+1,error_type=type(error).__name__,observed_at=time.time()))
                    if attempt==19:raise
                    time.sleep(30)
        except BaseException as error:
            text=re.sub(r'https?://[^\s]+','<URL>',''.join(traceback.format_exception(type(error),error,error.__traceback__)))
            save(ROOT/'pipeline_failure.json',dict(error_type=type(error).__name__,message=text[-6000:],observed_at=time.time(),training_executed=False))
            raise SystemExit(1)
