"""Feed verified packages and replicate only small manifests; never relay data TARs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

from build_retraining_parallel import checked_receipts
from launch_direct_oss_transfer import enrich,sign_items
from launch_retraining_parallel import environment
from prepare_retraining_data import save,error_detail
from relay_retraining_stream import accepted_keys,collect_new,command_remote,REMOTE_CACHE,REMOTE_OUTPUT,SOCKET
from relay_cloud_prepared import merge_parallel
from submit_retraining_cloud_probe import client,CLI


def sync_metadata(out,receipts,items):
    mapping={}
    for receipt in receipts:
        identity=hashlib.sha256(receipt['source_key'].encode()).hexdigest()[:20]
        mapping['receipts/'+identity+'.json']=json.dumps(receipt,indent=2)+'\n'
        mapping['indexes/'+receipt['root_id']+'.json']=items[identity]['index_body']
    if not mapping:return
    code="""import json,sys,pathlib
root=pathlib.Path(ROOT)
for name,text in json.load(sys.stdin).items():
 p=pathlib.PurePosixPath(name)
 assert len(p.parts)==2 and p.parts[0] in ('receipts','indexes') and '..' not in p.parts
 target=root/name;target.parent.mkdir(parents=True,exist_ok=True)
 if target.exists():assert target.read_text()==text,'Existing metadata differs'
 else:
  tmp=target.with_suffix('.tmp');tmp.write_text(text);tmp.replace(target)
print('metadata accepted')
""".replace('ROOT',repr(REMOTE_OUTPUT))
    r=subprocess.run(['ssh','-S',SOCKET,'macpro',shlex.join(['python3','-c',code])],input=json.dumps(mapping),capture_output=True,text=True,timeout=60)
    if r.returncode:raise RuntimeError('Metadata replication failed: '+r.stderr[-1000:])
    for name,text in mapping.items():
        p=out/'direct_metadata'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(text)


def _run(out):
    record=json.loads((out/'direct_oss_launch.json').read_text());launch=json.loads((out/'parallel/parallel_launch.json').read_text())
    frozen=json.loads((out/'source_archives.json').read_text());rows=frozen['source_archives'];bykey={r['key']:r for r in rows}
    baseline=json.loads(Path(launch['baseline_file']).read_text())['sources']
    assert hashlib.sha256(Path(launch['baseline_file']).read_bytes()).hexdigest()==launch['baseline_sha256']
    checked_receipts(rows,baseline,require_complete=False)
    proofs={'by_key':{r['source_key']:r for r in baseline},'seen_objects':set()}
    private=Path('/private/tmp/behavior-direct-oss-private');queue=json.loads((private/'queue.json').read_text())
    items={r['identity']:r for r in queue['items']};queued={r['proof']['source_key'] for r in items.values()}
    current=accepted_keys(out);checked_receipts(rows,current,require_complete=False)
    accepted={r['source_key'] for r in current};initial=len(accepted);copied=set();completed={}
    public,internal=client(),client(internal=True);pilot_passed=False
    def read(key):
        with public.get_object(Bucket='260010165',Key=key)['Body'] as b:return json.loads(b.read())
    while time.time()<record['deadline_epoch']:
        if (out/'CLOUD_STOP').exists():raise RuntimeError('Direct OSS controller STOP is present')
        fresh=[]
        for page in public.get_paginator('list_objects_v2').paginate(Bucket='260010165',Prefix=record['prefix']+'receipts/'):
            for obj in page.get('Contents',[]):
                if obj['Key'] in copied:continue
                r=read(obj['Key']);checked_receipts([bykey[r['source_key']]],[r])
                assert r['transfer_route']=='source_cloud_direct_to_OSS' and r['destination_reader_acceptance']['real_lerobot_reader']
                fresh.append(r);copied.add(obj['Key']);accepted.add(r['source_key'])
        if fresh:
            sync_metadata(out,fresh,items);pilot_passed=True
        collect_new(public,launch,frozen,proofs)
        observations=[]
        for shard in launch['shards']:
            if shard['name'] in completed:continue
            status=read(shard['status_key']);observations.append({'name':shard['name'],**status})
            if status.get('phase') in ('needs_diagnostic','bootstrap_needs_diagnostic'):
                raise RuntimeError('Cloud build needs diagnosis: '+shard['name'])
            if status.get('phase')=='all_cloud_exports_verified':
                with public.get_object(Bucket='260010165',Key=shard['manifest_key'])['Body'] as b:data=b.read()
                assert hashlib.sha256(data).hexdigest()==status['manifest']['sha256']
                completed[shard['name']]=json.loads(data)
        if pilot_passed:
            new=[p for k,p in proofs['by_key'].items() if k not in accepted|queued]
            if new:
                with ThreadPoolExecutor(max_workers=8) as pool:
                    unsigned=list(pool.map(lambda p:enrich(p,public,internal,record['prefix']),new))
                signed=sign_items(unsigned)
                for item in signed:items[item['identity']]=item;queued.add(item['proof']['source_key'])
            queue['items']=list(items.values());queue['all_sources_prepared']=len(completed)==4
            if queue['all_sources_prepared']:
                manifest=merge_parallel(frozen,baseline,completed,launch['shards']);save(out/'cloud_prepared_manifest.json',manifest)
            public.put_object(Bucket='260010165',Key=record['queue_key'],Body=json.dumps(queue).encode())
            save(private/'queue.json',queue)
        try:worker=read(record['status_key'])
        except public.exceptions.NoSuchKey:worker={'phase':'starting'}
        if worker.get('error_type'):raise RuntimeError('Direct transfer worker needs diagnosis: '+worker['error_type'])
        save(out/'cloud_relay_status.json',{'pid':os.getpid(),'phase':'direct_cloud_to_oss' if pilot_passed else 'direct_oss_real_pilot',
            'updated_at':time.time(),'source_accepted_archives':len(proofs['by_key']), 'destination_accepted_archives':len(accepted),
            'initial_destination_archives':initial,'cloud_complete_shards':len(completed),'direct_worker':worker,'cloud_shards':observations,
            'training_executed':False,'data_tar_bytes_relayed_through_macpro':0})
        if len(accepted)==len(rows) and len(completed)==4:
            result=command_remote('finalize_retraining_data.py',[REMOTE_CACHE,REMOTE_OUTPUT])
            save(out/'cloud_relay_status.json',{'pid':os.getpid(),'phase':'complete' if result['data_ready'] else 'waiting_original_reader',
                'updated_at':time.time(),'acceptance':result,'training_executed':False})
            if result['data_ready']:return
            if result['missing_archives'] or result['duplicates']:raise RuntimeError('Final audit needs diagnosis')
        time.sleep(15)
    raise RuntimeError('Direct OSS controller deadline reached')


def run(out):
    import fcntl
    with (out/'cloud_relay.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return _run(out)


def run_with_network_retries(out):
    import fcntl
    from botocore.exceptions import SSLError,EndpointConnectionError,ReadTimeoutError,ConnectionClosedError
    deadline=json.loads((out/'direct_oss_launch.json').read_text())['deadline_epoch']
    attempt=0
    with (out/'cloud_relay.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while time.time()<deadline:
            try:return _run(out)
            except (SSLError,EndpointConnectionError,ReadTimeoutError,ConnectionClosedError) as error:
                detail=error_detail(error)
            except subprocess.TimeoutExpired as error:
                if not error.cmd or error.cmd[0]!='ssh':raise
                detail=error_detail(error)
            except subprocess.CalledProcessError as error:
                if error.returncode!=255 or not error.cmd or error.cmd[0]!='ssh':raise
                detail=error_detail(error)
            attempt+=1
            save(out/'direct_controller_network_retry.json',{'attempt':attempt,**detail,'deadline_epoch':deadline})
            old=json.loads((out/'cloud_relay_status.json').read_text()) if (out/'cloud_relay_status.json').exists() else {}
            save(out/'cloud_relay_status.json',{**old,'pid':os.getpid(),'phase':'waiting_for_control_connection',
                'updated_at':time.time(),'last_connection_error':detail['error_type'],'training_executed':False})
            time.sleep(30)
    raise RuntimeError('Authorized direct transfer deadline reached')


if __name__=='__main__':
    out=Path('/Users/faye/code/Behavior-1k Challenge/analysis/retraining_20261013/preparation')
    try:run_with_network_retries(out)
    except BaseException as error:save(out/'cloud_relay_failure.json',error_detail(error));raise
