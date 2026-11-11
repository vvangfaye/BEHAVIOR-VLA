"""Bounded eight-GPU queue for eight independent five-instance shards."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from cluster_helpers import HERE,aoss_client,describe,write
from checkpoint_inventory import verified_checkpoints


def main():
    lock=(HERE/'monitor.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={'phase':'preparing_assets','complete':False,'pid':os.getpid(),'started_at':time.time(),'completed_episodes':0,'required_episodes':80};deadline=time.time()+7*86400;c=aoss_client()
    try:
        while time.time()<deadline:
            plan=json.loads((HERE/'plan.json').read_text());state['updated_at']=time.time()
            ready=all((HERE/n).exists() and json.loads((HERE/n).read_text()).get('complete',n in ['preflight.json','source_upload.json']) for n in ['small_upload.json','data_bundle_upload.json','preflight.json','source_upload.json'])
            if not ready:write(HERE/'status.json',state);time.sleep(20);continue
            inventory=verified_checkpoints(c,HERE)
            state['checkpoint_tasks_ready']=inventory['tasks_ready']
            active=[];pending=[];reports={};receipts={};attention={}
            for shard in plan['shards']:
                i=shard['shard'];path=HERE/f'shard{i}_submission.json'
                if not path.exists():pending.append(i);continue
                r=json.loads(path.read_text());d=describe(r['name']);assert d['uid']==r['uid'];receipts[i]=r
                try:s=json.loads(c.get_object(Bucket='260010165',Key=r['result_key'])['Body'].read())
                except c.exceptions.ClientError as e:
                    if e.response['Error']['Code'] in {'NoSuchKey','404'}:s={}
                    else:raise
                write(HERE/f'shard{i}_result.json',s);reports[i]=s
                if d['state']=='SUCCEEDED':assert s.get('complete'),f'Shard {i} exited without results'
                elif d['state'] in {'FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}:
                    gate=plan.get('recovery_gates',{}).get(str(i))
                    gate_passed=False
                    if gate:
                        probe=reports.get(gate['shard'],{})
                        progress=(probe.get('heartbeat') or {}).get('progress.json','{}')
                        try:progress=json.loads(progress)
                        except (TypeError,ValueError):progress={}
                        gate_passed=not probe.get('error') and (bool(probe.get('conditions')) or progress.get('steps',0)>=gate['min_steps'])
                    if r.get('generation',1)>=plan['max_generation_per_shard']:
                        attention[str(i)]='Recovery limit reached';continue
                    if s.get('error') and not s.get('termination_signal') and not gate_passed:
                        attention[str(i)]='Waiting for startup diagnosis' if gate else s['error'][-2000:]
                        continue
                    args=[sys.executable,str(HERE/'launch.py'),'--shard',str(i),'--recover']
                    with (HERE/f'shard{i}_recovery.log').open('a') as log:subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
                    active.append(i)
                else:active.append(i)
            counts={i:max(len(s.get('conditions',{})),len(receipts[i].get('preserved_episodes',[]))) for i,s in reports.items()}
            state.update(phase=('evaluating_with_errors' if active and attention else 'needs_attention' if attention else 'evaluating' if active else ('preparing_jobs' if inventory['tasks_ready'] else 'waiting_for_task_checkpoints')),attention=attention,completed_episodes=sum(counts.values()),shards={str(i):{'phase':s.get('phase'),'completed':counts[i],'heartbeat':s.get('heartbeat')} for i,s in reports.items()});write(HERE/'status.json',state)
            for i in pending:
                if len(active)>=plan['max_concurrent_gpus']:break
                shard=next(x for x in plan['shards'] if x['shard']==i)
                if shard['task_id'] not in inventory['tasks_ready']:continue
                with (HERE/f'shard{i}_launch.log').open('w') as log:subprocess.run([sys.executable,str(HERE/'launch.py'),'--shard',str(i)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=1200)
                active.append(i)
            if state['completed_episodes']==80 and all(s.get('complete') for s in reports.values()) and len(reports)==len(plan['shards']):
                if active:time.sleep(20);continue
                state['phase']='collecting';write(HERE/'status.json',state)
                with (HERE/'collection.log').open('w') as log:subprocess.run([sys.executable,str(HERE/'collect.py'),'--artifacts'],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=7200)
                result=json.loads((HERE/'comparison.json').read_text());assert result['complete'] and result['artifact_checks_passed'];state.update(phase='complete',complete=True);write(HERE/'status.json',state);return
            time.sleep(30)
        raise TimeoutError('Campaign deadline reached')
    except BaseException:
        state.update(phase='needs_attention',error=traceback.format_exc(),updated_at=time.time());write(HERE/'status.json',state);raise
    finally:c.close()


if __name__=='__main__':main()
