"""Submit a disjoint shard using fresh Spot admission or the one authorized standard GPU."""
import argparse
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import zlib
from public_test_admission import query_standard_capacity, validate_standard_capacity


POOL = 'computing-cluster-5090-01g'


def build_payload(ready, plan, pilot, index, execution_name, driver_sha256):
    assert ready['state'] == 'ready' and ready['checkpoint_step'] == 1688977
    assert ready['preparation_job']['state'] == 'SUCCEEDED'
    assert pilot['state'] == 'accepted' and pilot['checkpoint_step'] == 1688977
    assert pilot['platform_state'] == 'SUCCEEDED' and pilot['full_artifact_sha256_verified'] is True
    assert pilot['source_manifest_sha256'] == ready['source_manifest_sha256']
    assert pilot['driver_sha256'] == driver_sha256
    assert pilot['task'] == 'turning_on_radio' and pilot['instance_ids'] == [301]
    assert pilot['whole_task_native_generated_subtasks'] is True
    assert pilot['video_decoded_and_reviewed'] is True
    assert pilot['canonical_result_kept_regardless_of_score'] is True
    assert plan['checkpoint_step'] == 1688977 and plan['total_rollouts'] == 2000
    assert plan['production_rollouts'] == 1999 and plan['reuse_pilot_regardless_of_model_score'] is True
    shards = plan['shards']
    keys = [(shard['task'], i) for shard in shards for i in shard['instance_ids']]
    assert len(keys) == 1999 and len(set(keys)) == 1999
    assert ('turning_on_radio', 301) not in keys
    assert set(keys) | {('turning_on_radio',301)} == {
        (task,i) for task in ready['scope']['tasks'] for i in range(301,321)}
    assert isinstance(index,int) and not isinstance(index,bool) and 0 <= index < len(shards)
    shard = shards[index]
    assert shard['index'] == index and shard['expected_rollouts'] == len(shard['instance_ids'])
    assert shard['instance_ids'] and all(301 <= i <= 320 for i in shard['instance_ids'])
    run_id = 'public-test2000-1688977-20261010-a'+str(index).zfill(3)
    return dict(checkpoint=ready['checkpoint'], code=ready['code'],
        runtime='/mnt/afs/260010165/behavior_1k/runtime_20261007_1306',
        source_manifest_sha256=ready['source_manifest_sha256'],
        task=shard['task'], instance_ids=shard['instance_ids'], expected_rollouts=shard['expected_rollouts'],
        pilot=False, write_video=False, generation=0, run_id=run_id,
        assignment_root='/mnt/afs/260010165/behavior_1k/eval_results/public-test2000-1688977-20261010/assignments/a'+str(index).zfill(3),
        execution_name=execution_name, driver_sha256=driver_sha256)


def validate_capacity(capacity, *, now):
    assert capacity['resource_pool'] == POOL and capacity['quota_type'] == 'spot'
    assert capacity['fresh_query'] is True
    observed = datetime.fromisoformat(capacity['observed_at'])
    assert observed.tzinfo is not None and 0 <= (now-observed).total_seconds() <= 300
    assert isinstance(capacity['idle_gpus'],int) and not isinstance(capacity['idle_gpus'],bool)
    assert capacity['idle_gpus'] >= 0
    assert capacity['idle_gpus'] >= 1, 'No compatible Spot5090 card available for expansion'
    for key in ('idle_vcpu', 'idle_memory_gib'):
        assert isinstance(capacity[key], (int, float)) and not isinstance(capacity[key], bool)
        assert math.isfinite(capacity[key]) and capacity[key] >= 0
    slots = min(capacity['idle_gpus'], math.floor(capacity['idle_vcpu'] / 14),
        math.floor(capacity['idle_memory_gib'] / 60))
    assert slots >= 1, 'No complete 1GPU/14CPU/60GiB Spot slot available'
    return slots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--quota-type', choices=('spot','reserved'), default='spot')
    parser.add_argument('--capacity',type=Path)
    parser.add_argument('--queue-spot', action='store_true', help='User-authorized finite Spot queue; scheduler allocates resources without a free-capacity prerequisite.')
    parser.add_argument('--admission-slot',type=int,default=1,
        help='One-based ordinal within this exact fresh capacity snapshot; never reuse a slot.')
    parser.add_argument('--cli',default='/private/tmp/behavior-sco-runtime/bin/sco')
    args = parser.parse_args()
    root = args.evidence.resolve()
    receipts = root / 'production_submissions'
    receipts.mkdir(exist_ok=True)
    receipt = receipts / ('a'+str(args.index).zfill(3)+'.json')
    assert not receipt.exists(), 'Inspect exact existing job; never submit duplicate or blind retry'
    ready=json.loads((root/'preparation_ready.json').read_text())
    plan=json.loads((root/'production_plan.json').read_text())
    pilot=json.loads((root/'pilot_accepted.json').read_text())
    now=datetime.now(timezone.utc)
    assert not args.queue_spot or args.quota_type == 'spot'
    if args.quota_type == 'spot' and args.queue_spot:
        capacity_bytes=json.dumps({'mode':'scheduler_queue','resource_pool':POOL,'quota_type':'spot','scope':'public_test_2000','authorized_at':'2026-10-10 user:一直提交，然后监控每个提交的状态，不用管是否有余量'},sort_keys=True).encode()
        capacity_hash=hashlib.sha256(capacity_bytes).hexdigest()
    elif args.quota_type == 'spot':
        assert args.capacity is not None, 'Spot requires a fresh observed capacity file'
        capacity_bytes=args.capacity.read_bytes()
        capacity=json.loads(capacity_bytes)
        available_slots = validate_capacity(capacity,now=now)
        assert 1 <= args.admission_slot <= available_slots
        capacity_hash=hashlib.sha256(capacity_bytes).hexdigest()
        for existing in receipts.glob('a*.json'):
            prior=json.loads(existing.read_text())
            assert (prior.get('capacity_sha256'),prior.get('admission_slot')) != (capacity_hash,args.admission_slot)
    else:
        assert args.admission_slot == 1
    source=(root/'pilot_driver_source.py').read_bytes()
    source_hash=hashlib.sha256(source).hexdigest()
    compile(source,'public_test_driver','exec')
    name='behavior-public1688977-a'+str(args.index).zfill(3)+'-'+now.astimezone(timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
    payload=build_payload(ready,plan,pilot,args.index,name,source_hash)
    env=os.environ.copy()
    env.update(SCO_CONFIG='/private/tmp/behavior-sco-config',SCO_HOME='/private/tmp/behavior-sco-runtime',SCO_DATA_HOME='/private/tmp/behavior-sco-data')
    template=json.loads(subprocess.run([args.cli,'acp','jobs','describe','pt-y91th8k9',
        '--workspace-name=share-space','-o','json'],env=env,capture_output=True,text=True,check=True,timeout=30).stdout)
    assert template['resource_pool']['name']==POOL
    if args.quota_type == 'spot' and not args.queue_spot:
        assert args.admission_slot <= validate_capacity(capacity,now=datetime.now(timezone.utc))
    if args.quota_type == 'reserved':
        # Reuse only the one authorized latest-model standard card, after its prior job finishes.
        capacity=query_standard_capacity(args.cli,env)
        validate_standard_capacity(capacity,now=datetime.now(timezone.utc))
        capacity_bytes=json.dumps(capacity,sort_keys=True).encode()
        capacity_hash=hashlib.sha256(capacity_bytes).hexdigest()
        (root/'standard_production_admission_current.json').write_bytes(capacity_bytes+b'\n')
    from behavior_aoss_client import client
    report_key='behavior1k/test2000-1688977-20261010/production/a'+str(args.index).zfill(3)+'_status.json'
    payload['report_put']=client().generate_presigned_url('put_object',
        Params={'Bucket':'260010165','Key':report_key},ExpiresIn=432000)
    encoded=base64.b64encode(zlib.compress(json.dumps(payload).encode())).decode()
    encoded_source=base64.b64encode(zlib.compress(source)).decode()
    startup='import base64,zlib,json,hashlib;p=json.loads(zlib.decompress(base64.b64decode('+repr(encoded)+')));s=zlib.decompress(base64.b64decode('+repr(encoded_source)+'));assert hashlib.sha256(s).hexdigest()==p["driver_sha256"];ns={};exec(compile(s,"public_test_shard","exec"),ns);ns["run"](p)'
    record={key:value for key,value in payload.items() if key!='report_put'}
    record.update(name=name,state='submitting',index=args.index,resource_pool=POOL,gpu_count=1,
        worker_spec='n12lp.nn.i10a.1',quota_type=args.quota_type,timeout_seconds=345600,max_gpu_hours=96,
        report_key=report_key,capacity_sha256=capacity_hash,admission_slot=args.admission_slot,
        admission_mode='scheduler_queue' if args.queue_spot else 'capacity',user_authorized_spot_queue=args.queue_spot)
    # Preserve the exact intended name before contacting the API so ambiguous outcomes cannot duplicate a writer.
    with receipt.open('x') as output:output.write(json.dumps(record,indent=2)+'\n')
    command=[args.cli,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=pt-y91th8k9',
        '--name='+name,'--job-name='+name,'--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1',
        '--quota-type='+args.quota_type,'--env=NVIDIA_DRIVER_CAPABILITIES:all',
        '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
        '--command=timeout --signal=TERM --kill-after=30 345600 python3 -c '+shlex.quote(startup)]
    result=subprocess.run(command,env=env,capture_output=True,text=True,timeout=90)
    record.update(state='submitted' if result.returncode==0 else 'submission_unknown',exit_code=result.returncode)
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    assert result.returncode==0, 'Inspect exact saved name; do not retry blindly'
    job=json.loads(subprocess.run([args.cli,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],
        env=env,capture_output=True,text=True,check=True,timeout=30).stdout)
    assert job['ownership']['user_name']=='260010165' and job['resource_pool']['name']==POOL
    assert job['scheduling']['quota_type']==args.quota_type.upper()
    record.update(uid=job['uid'],platform_state=job['state'])
    receipt.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({key:record[key] for key in ('index','name','uid','platform_state','quota_type','gpu_count','expected_rollouts')}))


if __name__=='__main__':main()
