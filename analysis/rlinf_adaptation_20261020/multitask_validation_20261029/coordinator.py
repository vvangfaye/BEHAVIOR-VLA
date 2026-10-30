"""Run fixed task replications sequentially after the existing modem campaign."""
from datetime import datetime,timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
MODEM=HERE.parent/'modem_parallel_20261029'


def prepare_storage(task,status):
    """Inventory on a small CPU job, never while occupying a rollout GPU."""
    sys.path.insert(0,str(HERE.parent.parent/'serf_multitask_20261029'))
    from cluster_helpers import submit,aoss_client,describe
    folder=Path(task['directory']);plan=json.loads((folder/'plan.json').read_text())
    receipt_path=folder/'capacity_submission.json'
    client=aoss_client()
    try:
        if not receipt_path.exists():
            prefix=f"behavior1k/rlinf-multitask-20261029/{task['slug']}/capacity_20261030_a1/"
            key=prefix+'status.json'
            config={'output':plan['remote_root']+'/capacity_20261030_a1',
                    'result_key':key,'result_put':client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=86400),
                    'account_root':'/mnt/afs/260010165','experiment':plan['experiment'],
                    'reserve_bytes':plan['runtime_storage']['max_persistent_bytes'],
                    'proof_path':plan['output']+'/storage_inventory.json',
                    'quota_lower_bound_bytes':2150000000000,
                    'quota_source':{'volume':'01a04263-91e5-7603-bc01-c67e503da6b5','path':'/260010165',
                        'observed_display':'2.2 TB','display_unit_bytes':100000000000,
                        'method':'conservative lower endpoint of one-decimal TB rounding interval',
                        'observation_file':'quota_uuid_query_20261030.json'}}
            submit(f"b1k-mt-storage-1030-{task['slug']}-a1",HERE/'capacity_worker.py',config,receipt_path,7200,False)
        receipt=json.loads(receipt_path.read_text())
        while True:
            actual=describe(receipt['name']);assert actual['uid']==receipt['uid']
            try:
                result=json.loads(client.get_object(Bucket='260010165',Key=receipt['result_key'])['Body'].read())
            except client.exceptions.NoSuchKey:result={}
            write(folder/'capacity_status.json',{'state':actual['state'],'result':result})
            status(state='storage_inventory',task=task['task'],cpu_job=receipt['name'],gpu_count=0)
            if result.get('complete'):
                assert result['passed'],'Insufficient conservatively estimated account headroom'
                write(folder/'training_storage_clearance.json',result)
                return
            if actual['state'] in {'FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}:
                raise RuntimeError('CPU inventory incomplete; preserve its evidence')
            threading.Event().wait(30)
    finally:client.close()

def write(p,value):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)

def main():
    lock=(HERE/'coordinator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((HERE/'campaign.json').read_text())
    write(HERE/'coordinator_process.json',{'pid':os.getpid(),'started_at':time.time()})
    def status(**kw):write(HERE/'status.json',{'observed_at':datetime.now(timezone.utc).isoformat(),**kw})
    # The modem experiment retains its authorized 8-GPU concurrency. Never
    # allocate another RL campaign in parallel with it.
    while True:
        if (MODEM/'outcome.json').is_file():break
        if (MODEM/'stopped.json').is_file():
            status(state='waiting_for_modem_automatic_recovery',preserve_existing_baselines=True)
        else:status(state='queued_after_modem',new_task_jobs_submitted=0)
        threading.Event().wait(30)
    for task in plan['preregistered_tasks']:
        folder=Path(task['directory'])
        if (folder/'outcome.json').is_file():continue
        if (folder/'stopped.json').is_file():
            status(state='needs_attention',task=task['task'],reason='Previous task attempt stopped');return
        status(state='running',task=task['task'])
        with (folder/'controller.log').open('ab',buffering=0) as log:
            child=subprocess.Popen([sys.executable,str(folder/'controller.py')],cwd=folder,
                stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                env={**os.environ,'PYTHONUNBUFFERED':'1','PYTHONDONTWRITEBYTECODE':'1'})
        try:
            prepare_storage(task,status)
        except Exception as error:
            write(folder/'capacity_pending.json',{'at':time.time(),'error_type':type(error).__name__,
                  'baseline_may_continue':True,'training_remains_gated':True})
        while child.poll() is None:threading.Event().wait(30)
        if child.returncode or not (folder/'outcome.json').is_file():
            status(state='needs_attention',task=task['task'],returncode=child.returncode);return
    outcomes={x['slug']:json.loads((Path(x['directory'])/'outcome.json').read_text()) for x in plan['preregistered_tasks']}
    status(state='complete',finished=True,outcomes=outcomes)


if __name__=='__main__':main()
