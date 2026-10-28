"""Bounded local phase controller. Never holds a GPU while waiting for another phase."""
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import threading
import time

from deploy import HERE,describe,submit,write,aoss_client
from eval_utils import paired_report

TERMINAL={'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}


def inspect(client):
    rows=[]
    for path in sorted((HERE/'jobs').glob('*/submission.json')):
        receipt=json.loads(path.read_text())
        actual=describe(receipt['job'])
        assert actual['ownership']['user_name']=='260010165'
        if 'uid' not in receipt:
            assert actual['scheduling']['quota_type']==receipt['quota'].upper()
            receipt.update(uid=actual['uid'],state=actual['state']);write(path,receipt)
        assert actual['uid']==receipt['uid']
        row={**receipt,'state':actual['state']}
        try:
            result=client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'status.json')
            assert result['ContentLength']<4*1024**2
            row['progress']=json.loads(result['Body'].read())
        except client.exceptions.ClientError as error:
            if error.response['Error']['Code'] not in ('NoSuchKey','404'):raise
        rows.append(row)
    return rows


def latest(rows,phase,rank):
    values=[r for r in rows if r['phase']==phase and r['rank']==rank]
    return max(values,key=lambda r:r['attempt']) if values else None


def retryable(row):
    progress=row.get('progress',{})
    if progress.get('state')=='bootstrap_failed':return False
    kind=progress.get('error_type')
    if kind in ('AssertionError','ValueError','KeyError','AttributeError'):return False
    return kind in (None,'ExecutionStopped','SimulatorInitializationError','EOFError','BrokenPipeError')


def collect(client,row):
    folder=HERE/'jobs'/f"{row['phase']}_r{row['rank']}_a{row['attempt']}"
    marker=folder/'collection.json'
    if marker.exists():return
    response=client.get_object(Bucket='260010165',Key=row['aoss_prefix']+'result.tar.gz')
    assert response['ContentLength']<512*1024**2
    body=response['Body'].read();path=folder/'result.tar.gz';path.write_bytes(body)
    target=HERE/'retrieved'/folder.name
    with tarfile.open(path,'r:gz') as archive:
        for item in archive:
            relative=Path(item.name)
            assert item.isfile() and not relative.is_absolute() and '..' not in relative.parts
            assert relative.suffix=='.json'
            output=target/relative;output.parent.mkdir(parents=True,exist_ok=True)
            data=archive.extractfile(item).read()
            if output.exists():assert output.read_bytes()==data
            else:output.write_bytes(data)
    write(marker,{'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest(),'retrieved':str(target)})


def finalize():
    plan=json.loads((HERE/'plan.json').read_text())
    target=HERE/'final_evidence';target.mkdir(exist_ok=True)
    sources=sorted((HERE/'retrieved').iterdir())
    for phase in ('baseline','final'):
        for i,s in plan['evaluation']['cases']:
            case=f'i{i}_s{s}'
            candidates=[r/phase/case for r in sources if all((r/phase/case/n).is_file() for n in ('result.json','initial.json','spec.json'))]
            assert candidates,f'Missing {phase}/{case}'
            for name in ('result.json','initial.json','spec.json'):
                data=(candidates[-1]/name).read_bytes()
                assert all((r/name).read_bytes()==data for r in candidates),'Conflicting completed results'
                out=target/phase/case/name;out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(data)
    report=paired_report(target,plan['evaluation']['cases'],plan['evaluation']['paired_initial_pose_tolerance'])
    assert report['comparison_complete'] and report['comparison_valid']
    training=[json.loads(p.read_text()) for p in (HERE/'retrieved').glob('*/training_complete.json')]
    assert training and all(r['optimizer_steps']==17 and r['episodes']==32 for r in training)
    report.update(training=training[-1],sparse_ppo_reference=json.loads((HERE.parent/'eval17_baseline_first_20261027/final_audit.json').read_text())['ppo17'],
                  completed_at=datetime.now(timezone.utc).isoformat(),only_intended_change='physical milestone rewards')
    write(HERE/'outcome.json',report)
    return report


def stop_active(rows, reason):
    stopped=[]
    for row in rows:
        if row['state'] in TERMINAL:
            continue
        actual=describe(row['job'])
        assert actual['uid']==row['uid'] and actual['ownership']['user_name']=='260010165'
        if actual['state'] in TERMINAL:
            continue
        result=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','stop',row['job'],
                               '--workspace-name=share-space'],capture_output=True,text=True,timeout=40)
        stopped.append({'job':row['job'],'uid':row['uid'],'returncode':result.returncode})
    write(HERE/'phase_stop_receipt.json',{'reason':reason,'jobs':stopped})


def main():
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(HERE/'controller_process.json',{'pid':os.getpid(),'started_at':time.time(),'maximum_wall_hours':48,'maximum_live_gpus':2})
    client=aoss_client();end=time.time()+48*3600;previous=None
    try:
        while time.time()<end:
            if (HERE/'CANCELLED_BY_USER').exists():
                stop_active(inspect(client),'user_cancelled');return
            try:
                rows=inspect(client)
                for row in rows:
                    if row['state'] in TERMINAL:
                        try:collect(client,row)
                        except client.exceptions.ClientError as error:
                            if error.response['Error']['Code'] not in ('NoSuchKey','404'):raise
                now=datetime.now(timezone.utc).isoformat()
                compact=[{'phase':r['phase'],'rank':r['rank'],'attempt':r['attempt'],'job':r['job'],'uid':r['uid'],
                          'state':r['state'],'progress':r.get('progress',{})} for r in rows]
                status={'observed_at':now,'workers':compact,'finished':False}
                write(HERE/'latest_status.json',status)
                active_path=HERE.parent/'active_run.json'
                active=json.loads(active_path.read_text()) if active_path.exists() else {}
                if active.get('experiment')=='subtask_reward_ppo_20261028':
                    active.update(status='running',observed_at=now,workers=compact)
                    write(active_path,active)
                display=[{k:r[k] for k in ('phase','rank','attempt','state')}|{'stage':r.get('progress',{}).get('stage'),'collected':r.get('progress',{}).get('collected')} for r in rows]
                encoded=json.dumps(display,sort_keys=True)
                if encoded!=previous:print(json.dumps({'at':now,'workers':display}),flush=True);previous=encoded
                active_count=sum(r['state'] not in TERMINAL for r in rows)
                submitted=False
                # Recover only interrupted phases, with identical data and a bounded attempt count.
                for phase,rank in [('sample',0),('sample',1),('train_eval',0),('eval',1)]:
                    row=latest(rows,phase,rank)
                    if row and row['state'] in TERMINAL-{'SUCCEEDED'}:
                        if row['attempt']>=2 or not retryable(row):
                            write(HERE/'stopped.json',{'reason':'phase_failed','worker':row,'no_scope_change':True})
                            stop_active(rows,'unrecoverable_phase_failure');return
                        if active_count<2:
                            submit(phase,rank,row['attempt']+1);submitted=True;break
                if submitted:continue
                samples=[latest(rows,'sample',r) for r in (0,1)]
                if all(r and r['state']=='SUCCEEDED' for r in samples):
                    train=latest(rows,'train_eval',0)
                    if train is None:
                        submit('train_eval',0);continue
                    evaluation=latest(rows,'eval',1)
                    progress=train.get('progress',{})
                    if evaluation is None and progress.get('checkpoint') and progress.get('baseline_ready'):
                        submit('eval',1);continue
                    if train['state']=='SUCCEEDED' and evaluation and evaluation['state']=='SUCCEEDED':
                        outcome=finalize();status.update(finished=True,outcome=outcome);write(HERE/'latest_status.json',status)
                        active.update(status='complete',completed_at=now,outcome_path=str(HERE/'outcome.json'));write(active_path,active)
                        print(json.dumps({'complete':True,'baseline':outcome['baseline'],'shaped_ppo':outcome['final']}),flush=True);return
            except Exception as error:
                print(json.dumps({'controller_error':type(error).__name__,'message':str(error)[:300]}),flush=True)
                write(HERE/'controller_last_error.json',{'at':time.time(),'type':type(error).__name__,'message':str(error)[:500]})
            threading.Event().wait(45)
        stop_active(inspect(client),'controller_wall_deadline')
        write(HERE/'controller_deadline.json',{'wall_hours':48,'no_unbounded_submissions':True})
    finally:client.close()


if __name__=='__main__':main()
