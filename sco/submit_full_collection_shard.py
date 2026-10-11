"""Submit one immutable full-coverage shard with private object checkpoints."""
import argparse
import base64
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import zlib

from public_test_admission import query_standard_capacity

BUCKET='260010165'
RUNTIME='/mnt/afs/260010165/behavior_1k/runtime_20261007_1306'


def environment():
    return dict(os.environ,SCO_CONFIG='/private/tmp/behavior-sco-config',SCO_HOME='/private/tmp/behavior-sco-runtime',
                SCO_DATA_HOME='/private/tmp/behavior-sco-data')


def describe(cli,env,name):
    for attempt in range(3):
        try:
            value=json.loads(subprocess.check_output([cli,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],env=env,text=True,timeout=30))
            break
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, json.JSONDecodeError):
            if attempt==2:raise
            time.sleep(2)
    assert value['ownership']['user_name']==BUCKET
    return value


def terminal_worker_evidence(workers,uid):
    """Accept the ACP Deleted phase only for the exact previously stopped worker."""
    expected='pt-'+uid+'-worker-0'
    rows=[line for line in workers.splitlines()
          if len(line.split('|'))>2 and line.split('|')[1].strip()==expected]
    if len(rows)!=1 or not re.search(r'\|\s*(Failed|Succeeded|Deleted)\s*\|\s*$',rows[0]):
        raise ValueError('Previous worker is not demonstrably terminal')
    return rows[0]


def verify_gates(client,cli,env,path,ready,identities):
    gates=json.loads(Path(path).read_text())
    assert gates['source_manifest_sha256']==ready['source_manifest_sha256']
    assert gates['state']=='complete' and gates['protocols_verified']==['history','native']
    assert gates['additional_start_verified'] is True
    assert gates['driver_identities']==identities
    positive=0
    for gate in gates['gates']:
        live=describe(cli,env,gate['name'])
        assert live['uid']==gate['uid'] and live['state']=='SUCCEEDED'
        assert live['scheduling']['quota_type']=='RESERVED'
        report=json.loads(client.get_object(Bucket=BUCKET,Key=gate['final_report_key'])['Body'].read())
        assert report['source_manifest_sha256']==ready['source_manifest_sha256']
        assert report['plan_fingerprint']==gate['plan_fingerprint']
        assert report['artifact_acceptance_complete'] and report['object_checkpoints_complete']
        assert report['checkpoint_restore_verified'] and report['collection_summary']['eligible_trials']>0
        assert report['collection_summary']['complete']
        if report['collection_summary']['accepted_trials']:
            assert report['actual_training_reader']['actual_openpi_training_reader']
            positive+=report['collection_summary']['accepted_trials']
    assert positive>0,'Standard tests have not yet produced verified trainable data'
    return gates


def submit(args):
    receipt=Path(args.receipt)
    if receipt.exists():raise ValueError('Existing intent: reconcile its exact name before any submission')
    ready=json.loads(Path(args.prepared).read_text());plan=ready['plan']
    assert ready['state']=='complete' and 1<=plan['expected_trials']<=30
    assert 1<=args.timeout_hours<=24
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from full_subtask_collection import BASIS
    assert plan['selection_basis']==BASIS and len(plan['selected_groups'])==1
    spec=importlib.util.spec_from_file_location('private_aoss',args.aoss_client_module)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);client=module.client()
    root=Path(__file__).resolve().parent
    outer=(root/'run_full_collection_shard.py').read_bytes();inner=(root/'run_subtask_collection.py').read_bytes()
    reader=(root/'verify_collection_training_reader.py').read_text()
    identities=dict(outer=hashlib.sha256(outer).hexdigest(),inner=hashlib.sha256(inner).hexdigest(),reader=hashlib.sha256(reader.encode()).hexdigest())
    env=environment();capacity=None
    if args.quota_type=='reserved':
        capacity=query_standard_capacity(args.cli,env)
        assert capacity['used_gpus']<=3,'No owner-wide standard GPU slot'
    else:
        assert args.gates
        verify_gates(client,args.cli,env,args.gates,ready,identities)
    template=describe(args.cli,env,'pt-y91th8k9')
    assert template['uid']=='c14d0892eaa148d381c7f4424684c9d3'
    assert template['resource_pool']['name']=='computing-cluster-5090-01g'
    quota=subprocess.check_output([args.cli,'afs','dir-quota','list','--volumeID=01a04263-91e5-7603-bc01-c67e503da6b5','--dirPath=/260010165'],env=env,text=True,timeout=30)
    numbers=re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*(TB|GB|MB)',quota)
    assert len(numbers)==2
    def conservative(v,u,high=False):
        decimals=len(v.split('.')[1]) if '.' in v else 0
        return (float(v)+(1 if high else -1)*.5*10**(-decimals))*{'TB':10**12,'GB':10**9,'MB':10**6}[u]
    free=int(conservative(*numbers[0])-conservative(*numbers[1],high=True))
    assert free>=16*1024**3,'Insufficient personal AFS room for logs and receipts'
    generation=0;recovery=None;previous=None
    if args.recover_receipt:
        previous=json.loads(Path(args.recover_receipt).read_text())
        assert previous['plan_fingerprint']==plan['fingerprint'] and previous['source_manifest_sha256']==ready['source_manifest_sha256']
        assert previous['quota_type']==args.quota_type and previous['driver_identities']==identities
        assert previous['name']!=args.name
        generation=previous['generation']+1;assert generation<=2,'At most three executions per group'
        live=describe(args.cli,env,previous['name']);assert live['uid']==previous['uid'] and live['state'] in ('FAILED','SUSPENDED')
        workers=subprocess.check_output([args.cli,'acp','jobs','get-workers',previous['name'],'--workspace-name=share-space'],env=env,text=True,timeout=30)
        worker_evidence=terminal_worker_evidence(workers,previous['uid'])
        recovery=dict(old_name=previous['name'],old_uid=previous['uid'],old_platform_state=live['state'],old_writer_stopped=True,
            worker_terminal_evidence=worker_evidence,observed_at=datetime.now(timezone.utc).isoformat(),parent_receipt=str(args.recover_receipt))
    prefix=args.object_prefix.rstrip('/')
    if previous:assert prefix==previous['object_prefix']
    record=dict(name=args.name,state='preparing_objects',quota_type=args.quota_type,gpu_count=1,generation=generation,
        prepared_root=ready['output_root'],source_manifest_sha256=ready['source_manifest_sha256'],code_root=ready['code_root'],
        plan_fingerprint=plan['fingerprint'],group=plan['selected_groups'][0],expected_trials=plan['expected_trials'],
        object_prefix=prefix,driver_identities=identities,submission_client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timeout_hours=args.timeout_hours,capacity=capacity,
        quota_observed_at=datetime.now(timezone.utc).isoformat(),personal_quota_free_lower_bound_bytes=free,recovery=recovery)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x') as f:f.write(json.dumps(record,indent=2)+'\n')
    def signed(operation,key,**extra):
        return client.generate_presigned_url(operation,Params=dict(Bucket=BUCKET,Key=key,**extra),ExpiresIn=604800)
    def urls(key,head=False):
        result=dict(key=key,put=signed('put_object',key),get=signed('get_object',key))
        if head:result['head']=signed('head_object',key)
        return result
    archive_key=prefix+'/executions/g%d/dataset.tar'%generation
    upload=client.create_multipart_upload(Bucket=BUCKET,Key=archive_key,ContentType='application/x-tar')
    upload_id=upload['UploadId']
    record.update(final_report_key=prefix+'/executions/g%d/acceptance.json'%generation,
                  archive_key=archive_key,multipart_upload_id=upload_id)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    parts=max(1,math.ceil(plan['maximum_output_bytes']/(128*1024**2))+8)
    storage=dict(plan_fingerprint=plan['fingerprint'],source_manifest_sha256=ready['source_manifest_sha256'],trials={},
        final_archive=dict(key=archive_key,parts=[signed('upload_part',archive_key,UploadId=upload_id,PartNumber=i) for i in range(1,parts+1)],
            complete=signed('complete_multipart_upload',archive_key,UploadId=upload_id),head=signed('head_object',archive_key)),
        final_report=urls(record['final_report_key']))
    for trial in plan['trials']:
        key=prefix+'/trials/'+trial['trial_id']
        storage['trials'][trial['trial_id']]=dict(archive=urls(key+'.tar',head=True),receipt=urls(key+'.json'))
    private=(json.dumps(storage)+'\n').encode();storage_key=prefix+'/executions/g%d/storage.private.json'%generation
    client.put_object(Bucket=BUCKET,Key=storage_key,Body=private)
    payload=dict(runtime_root=RUNTIME,code_root=ready['code_root'],prepared_root=ready['output_root'],
        execution_name=args.name,source_manifest_sha256=ready['source_manifest_sha256'],plan_fingerprint=plan['fingerprint'],
        driver_sha256=identities['inner'],raw_data_root=ready['raw_data_root'],generation=generation,
        persistent_root='/mnt/afs/260010165/behavior_1k/eval_results/subtask-collection-full299/'+args.name,
        storage_manifest_url=signed('get_object',storage_key),storage_manifest_sha256=hashlib.sha256(private).hexdigest(),reader_source=reader)
    payload['verify_checkpoint_restore']=args.quota_type=='reserved'
    if recovery:payload['recovery']=recovery
    encoded=base64.b64encode(zlib.compress(json.dumps(dict(source=outer.decode(),payload=payload)).encode())).decode()
    startup='import base64,hashlib,json,zlib;d=json.loads(zlib.decompress(base64.b64decode('+repr(encoded)+')));assert hashlib.sha256(d["source"].encode()).hexdigest()=='+repr(identities['outer'])+';exec(compile(d["source"],"full_collection_driver","exec"));run(d["payload"])'
    compile(startup,'startup','exec')
    record.update(state='submitting',persistent_root=payload['persistent_root'],storage_key=storage_key)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    command=[args.cli,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=pt-y91th8k9','--name='+args.name,
        '--job-name='+args.name,'--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type='+args.quota_type,
        '--env=NVIDIA_DRIVER_CAPABILITIES:all','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
        '--command=timeout --signal=TERM --kill-after=30 '+str(args.timeout_hours*3600)+' '+RUNTIME+'/envs/b1k-eval/bin/python -c '+shlex.quote(startup)]
    try:
        result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
        record.update(state='submitted' if result.returncode==0 else 'submission_unknown',exit_code=result.returncode)
    except subprocess.TimeoutExpired:
        record.update(state='submission_unknown');receipt.write_text(json.dumps(record,indent=2)+'\n');raise RuntimeError('Submission unknown; reconcile exact saved name') from None
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    # An RPC error may follow a successful creation; resolve the saved name once.
    live=describe(args.cli,env,args.name)
    assert live['scheduling']['quota_type']==args.quota_type.upper()
    record.update(uid=live['uid'],platform_state=live['state']);receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ['name','uid','platform_state','quota_type','expected_trials']}),flush=True)
    return record


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('prepared','receipt','name','object-prefix'):p.add_argument('--'+field,required=True)
    p.add_argument('--quota-type',required=True,choices=['reserved','spot']);p.add_argument('--gates')
    p.add_argument('--recover-receipt');p.add_argument('--timeout-hours',type=int,default=24)
    p.add_argument('--aoss-client-module',default='/private/tmp/behavior_aoss_client.py')
    p.add_argument('--cli',default='/private/tmp/behavior-sco-runtime/bin/sco')
    args=p.parse_args();submit(args)


if __name__=='__main__':main()
