"""Submit one frozen collection shard; keep unique intent before API mutation."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
import os
import re
from pathlib import Path
import shlex
import subprocess
import zlib

from public_test_admission import query_standard_capacity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,required=True)
    p.add_argument('--receipt',type=Path,required=True)
    p.add_argument('--name',required=True)
    p.add_argument('--quota-type',choices=('reserved','spot'),required=True)
    p.add_argument('--cli',default='/private/tmp/behavior-sco-runtime/bin/sco')
    p.add_argument('--timeout-hours',type=int,default=8)
    p.add_argument('--personal-quota-free-lower-bound-bytes',type=int,required=True)
    p.add_argument('--quota-observed-at',required=True)
    p.add_argument('--task-name')
    p.add_argument('--shared-results-root')
    p.add_argument('--recover-receipt',type=Path)
    args=p.parse_args()
    if args.receipt.exists():raise ValueError('Existing intent: query its exact name; do not resubmit')
    ready=json.loads(args.prepared.read_text())
    assert ready['state']=='complete'
    plan=ready['plan']
    assert bool(args.task_name)==bool(args.shared_results_root)
    assert 1 <= args.timeout_hours <= 24 and plan['expected_trials'] <= 300
    observed=datetime.fromisoformat(args.quota_observed_at)
    assert observed.tzinfo and 0 <= (datetime.now(timezone.utc)-observed).total_seconds() <= 120
    # Keep room for both the retained raw attempts and the exported dataset.
    storage_budget = 2*plan['maximum_output_bytes'] + 16*1024**3
    assert args.personal_quota_free_lower_bound_bytes >= storage_budget
    env=os.environ.copy()
    env.update(SCO_CONFIG='/private/tmp/behavior-sco-config',SCO_HOME='/private/tmp/behavior-sco-runtime',
               SCO_DATA_HOME='/private/tmp/behavior-sco-data')
    if args.quota_type=='reserved':
        capacity=query_standard_capacity(args.cli,env)
        assert capacity['used_gpus']<=3, 'No owner-wide standard GPU slot'
    else:
        capacity=None
        pilot=ready.get('accepted_standard_pilot')
        assert pilot and pilot['platform_state']=='SUCCEEDED' and pilot['trainable_dataset_verified'] is True
        assert pilot['source_manifest_sha256']==ready['source_manifest_sha256']
        live_pilot=json.loads(subprocess.check_output([args.cli,'acp','jobs','describe',pilot['name'],
            '--workspace-name=share-space','-o','json'],env=env,text=True))
        assert live_pilot['uid']==pilot['uid'] and live_pilot['state']=='SUCCEEDED'
        assert live_pilot['ownership']['user_name']=='260010165'
        assert live_pilot['scheduling']['quota_type']=='RESERVED'
        assert pilot['export']['episodes']>0 and pilot['export']['new_frames']>0
    template=json.loads(subprocess.check_output([args.cli,'acp','jobs','describe','pt-y91th8k9',
        '--workspace-name=share-space','-o','json'],env=env,text=True))
    assert template['ownership']['user_name']=='260010165'
    assert template['resource_pool']['name']=='computing-cluster-5090-01g'
    driver=(Path(__file__).parent/'run_subtask_collection.py').read_bytes()
    payload=dict(runtime_root=plan['checkpoint_identity']['checkpoint_source'].rsplit('/',1)[0],
        code_root=ready['code_root'],prepared_root=ready['output_root'],
        output_root='/mnt/afs/260010165/behavior_1k/eval_results/subtask-collection/'+args.name,
        execution_name=args.name,source_manifest_sha256=ready['source_manifest_sha256'],
        plan_fingerprint=plan['fingerprint'],driver_sha256=hashlib.sha256(driver).hexdigest(),
        raw_data_root=ready['raw_data_root'])
    if args.task_name:
        shard=next(s for s in ready['shards'] if s['task']==args.task_name)
        assert shard['expected_trials']>0
        payload.update(task_name=args.task_name,shared_results_root=args.shared_results_root)
    else:
        shard=None
    if args.recover_receipt:
        old=json.loads(args.recover_receipt.read_text())
        assert args.name!=old['name'] and old['quota_type']==args.quota_type
        for key in ('code_root','source_manifest_sha256','plan_fingerprint','raw_data_root'):
            assert old[key]==payload[key]
        assert old.get('task_name')==payload.get('task_name')
        assert old.get('shared_results_root')==payload.get('shared_results_root')
        generation=old.get('generation',0)+1
        assert generation<=2, 'Collection permits at most three executions per shard'
        previous=json.loads(subprocess.check_output([args.cli,'acp','jobs','describe',old['name'],
            '--workspace-name=share-space','-o','json'],env=env,text=True))
        assert previous['uid']==old['uid'] and previous['ownership']['user_name']=='260010165'
        assert previous['state'] in ('FAILED','SUSPENDED')
        workers=subprocess.check_output([args.cli,'acp','jobs','get-workers',old['name'],
            '--workspace-name=share-space'],env=env,text=True)
        worker='pt-'+old['uid']+'-worker-0'
        rows=[line for line in workers.splitlines() if worker in line]
        assert len(rows)==1 and re.search(r'\|\s*(Failed|Succeeded)\s*\|',rows[0])
        payload.update(output_root=old['output_root'],generation=generation,recovery=dict(
            old_name=old['name'],old_uid=old['uid'],old_platform_state=previous['state'],
            old_writer_stopped=True,worker_terminal_evidence=rows[0],
            observed_at=datetime.now(timezone.utc).isoformat(),parent_receipt=str(args.recover_receipt)))
    encoded=base64.b64encode(zlib.compress(json.dumps(dict(source=driver.decode(),payload=payload)).encode())).decode()
    startup='import base64,json,zlib,hashlib;d=json.loads(zlib.decompress(base64.b64decode('+repr(encoded)+')));assert hashlib.sha256(d["source"].encode()).hexdigest()==d["payload"]["driver_sha256"];exec(compile(d["source"],"collection_driver","exec"));run(d["payload"])'
    record=dict(payload,name=args.name,state='submitting',quota_type=args.quota_type,gpu_count=1,
        expected_trials=shard['expected_trials'] if shard else plan['expected_trials'],
        max_steps=plan['max_steps'],timeout_hours=args.timeout_hours,
        capacity=capacity,personal_quota_free_lower_bound_bytes=args.personal_quota_free_lower_bound_bytes,
        quota_observed_at=args.quota_observed_at,storage_budget_bytes=storage_budget)
    args.receipt.parent.mkdir(parents=True,exist_ok=True)
    with args.receipt.open('x') as stream:stream.write(json.dumps(record,indent=2)+'\n')
    (args.receipt.parent/(args.name+'.driver.py')).write_bytes(driver)
    command=[args.cli,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=pt-y91th8k9',
        '--name='+args.name,'--job-name='+args.name,'--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1',
        '--quota-type='+args.quota_type,'--env=NVIDIA_DRIVER_CAPABILITIES:all',
        '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
        '--command=timeout --signal=TERM --kill-after=30 '+str(args.timeout_hours*3600)+' '+
            payload['runtime_root']+'/envs/b1k-eval/bin/python -c '+shlex.quote(startup)]
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
    record.update(state='submitted' if result.returncode==0 else 'submission_unknown',exit_code=result.returncode)
    args.receipt.write_text(json.dumps(record,indent=2)+'\n')
    assert result.returncode==0, 'Query exact saved name before another submission'
    job=json.loads(subprocess.check_output([args.cli,'acp','jobs','describe',args.name,
        '--workspace-name=share-space','-o','json'],env=env,text=True))
    assert job['ownership']['user_name']=='260010165' and job['scheduling']['quota_type']==args.quota_type.upper()
    record.update(uid=job['uid'],platform_state=job['state'])
    args.receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:record[k] for k in ('name','uid','platform_state','quota_type','gpu_count','expected_trials')}))


if __name__=='__main__':main()
