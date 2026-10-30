"""One bounded local scheduler; no allocated GPU waits between phases."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import threading
import time

from deploy import HERE, describe, submit, write, aoss_client
from eval_utils import paired_report

TERMINAL = {'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}


def phase_ranks(phase):
    plan=json.loads((HERE/'plan.json').read_text())
    return (0,) if phase.startswith('train') else tuple(range(plan['parallel_workers']))


def inspect(client):
    rows = []
    for path in sorted((HERE/'jobs').glob('*/submission.json')):
        receipt = json.loads(path.read_text())
        actual = describe(receipt['job'])
        assert actual['ownership']['user_name']=='260010165'
        if 'uid' not in receipt:
            assert actual['scheduling']['quota_type']==receipt['quota'].upper()
            receipt.update(uid=actual['uid'],state=actual['state']);write(path,receipt)
        assert actual['uid']==receipt['uid']
        row = {**receipt,'state':actual['state']}
        try:
            value = client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'status.json')
            assert value['ContentLength']<8*1024**2
            row['progress'] = json.loads(value['Body'].read())
        except client.exceptions.ClientError as error:
            if error.response['Error']['Code'] not in ('NoSuchKey','404'):
                raise
        rows.append(row)
    return rows


def latest(rows, phase, rank):
    candidates = [r for r in rows if r['phase']==phase and r['rank']==rank]
    return max(candidates,key=lambda r:r['attempt']) if candidates else None


def retryable(row):
    runtime_recovery = HERE/'recovery_20261029/retry_authorization.json'
    if runtime_recovery.is_file():
        recovery = json.loads(runtime_recovery.read_text())
        if row.get('uid') in recovery['verified_stopped_uids']:
            assert row['phase']=='sample2' and row['attempt']==1
            for name,digest in recovery['fixed_source_sha256'].items():
                assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
            return True
    recovery_file = HERE/'startup_recovery_20261029/retry_authorization.json'
    if recovery_file.is_file():
        recovery = json.loads(recovery_file.read_text())
        if row.get('uid') in recovery['verified_stopped_uids']:
            assert row['phase']=='baseline' and row['attempt']==1
            for name,digest in recovery['fixed_source_sha256'].items():
                assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest
            return True
    progress = row.get('progress',{})
    if progress.get('state')=='bootstrap_failed':
        return False
    return progress.get('error_type') in (None,'ExecutionStopped','SimulatorInitializationError','EOFError','BrokenPipeError')


def collect(client,row):
    folder = HERE/'jobs'/f"{row['phase']}_r{row['rank']}_a{row['attempt']}"
    if (folder/'collection.json').exists():
        return
    value = client.get_object(Bucket='260010165',Key=row['aoss_prefix']+'result.tar.gz')
    assert value['ContentLength']<512*1024**2
    body = value['Body'].read(); (folder/'result.tar.gz').write_bytes(body)
    target = HERE/'retrieved'/folder.name
    with tarfile.open(folder/'result.tar.gz','r:gz') as archive:
        for member in archive:
            path = Path(member.name)
            assert member.isfile() and not path.is_absolute() and '..' not in path.parts and path.suffix=='.json'
            data = archive.extractfile(member).read()
            output = target/path; output.parent.mkdir(parents=True,exist_ok=True)
            if output.exists():
                assert output.read_bytes()==data
            else:
                output.write_bytes(data)
    write(folder/'collection.json',{'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})


def stop_active(rows,reason):
    results = []
    for row in rows:
        current = describe(row['job'])
        assert current['uid']==row['uid'] and current['ownership']['user_name']=='260010165'
        if current['state'] in TERMINAL:
            continue
        value = subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','stop',row['job'],
                                '--workspace-name=share-space'],capture_output=True,text=True,timeout=40)
        results.append({'job':row['job'],'uid':row['uid'],'returncode':value.returncode})
    write(HERE/'stop_receipt.json',{'reason':reason,'jobs':results})


def finalize(plan):
    evidence = HERE/'final_evidence'
    roots = sorted((HERE/'retrieved').iterdir())
    for phase in ('baseline','final'):
        for instance,seed in plan['evaluation']['cases']:
            case = f'i{instance}_s{seed}'
            completed=[r/phase/case for r in roots if all((r/phase/case/name).is_file()
                for name in ('result.json','initial.json','spec.json','actions.json','goal_summary.json'))]
            assert completed, (phase,case,'No complete case evidence')
            for name in ('result.json','initial.json','spec.json','goal_summary.json'):
                candidates = [directory/name for directory in completed]
                assert candidates, (phase,case,name)
                data = candidates[-1].read_bytes()
                assert all(p.read_bytes()==data for p in candidates), 'Conflicting completed evidence'
                output = evidence/phase/case/name;output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(data)
    report = {'training_instances':paired_report(evidence,plan['evaluation']['training_cases']),
              'heldout_instances':paired_report(evidence,plan['evaluation']['heldout_cases'])}
    assert all(v['comparison_complete'] and v['comparison_valid'] for v in report.values())
    for panel,key in [('training_instances','training_cases'),('heldout_instances','heldout_cases')]:
        cases=plan['evaluation'][key]
        report[panel]['physical_goal_counts']={}
        for phase in ('baseline','final'):
            values=[json.loads((evidence/phase/f'i{i}_s{s}'/'goal_summary.json').read_text()) for i,s in cases]
            report[panel]['physical_goal_counts'][phase]={
                goal:{'finally_satisfied':sum(v['final'][goal] for v in values),
                      'ever_satisfied':sum(v['ever_achieved'][goal] for v in values)}
                for goal in ('toggled_on','nextto_tv','shared_tv_cabinet')}
    for index in (1,2):
        candidates = [r/'checkpoints'/f'round_{index:02d}'/'ready.json' for r in roots
                      if (r/'checkpoints'/f'round_{index:02d}'/'ready.json').is_file()]
        assert candidates
        ready = json.loads(candidates[-1].read_text())
        assert ready['optimizer_steps']==17 and ready['cumulative_optimizer_steps']==index*17
    report.update(completed_at=datetime.now(timezone.utc).isoformat(),rounds=2,total_optimizer_steps=34,
                  scope=plan['comparisons'],checkpoint_selection=False)
    write(HERE/'outcome.json',report)
    return report


def next_pending(plan,rows):
    for phase in plan['phase_order']:
        current = [latest(rows,phase,rank) for rank in phase_ranks(phase)]
        if all(r and r['state']=='SUCCEEDED' for r in current):
            continue
        return phase,current
    return None,[]


def prepare_migration(client,plan):
    ready=HERE/'migration_ready.json'
    if ready.is_file():
        return True
    old=Path(plan['previous_generation_directory'])
    dep=plan['first_case_dependency']
    actual=describe(dep['job'])
    assert actual['uid']==dep['uid'] and actual['ownership']['user_name']=='260010165'
    receipt=json.loads((old/'jobs/baseline_r0_a3/submission.json').read_text())
    progress={}
    try:
        value=client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'status.json')
        progress=json.loads(value['Body'].read())
    except client.exceptions.ClientError as error:
        if error.response['Error']['Code'] not in ('NoSuchKey','404'):
            raise
    complete=progress.get('completed_evaluations',0)>=1
    write(HERE/'latest_status.json',{'experiment':plan['experiment'],'generation':plan['generation'],
          'state':'waiting_for_first_complete_case','observed_at':datetime.now(timezone.utc).isoformat(),
          'first_case_job':{'job':dep['job'],'uid':dep['uid'],'state':actual['state'],'progress':progress},
          'max_live_gpus':8,'finished':False})
    if not complete:
        if actual['state'] in TERMINAL:
            write(HERE/'stopped.json',{'reason':'first_case_failed_no_parallel_expansion','job':dep,'progress':progress})
            raise SystemExit(1)
        return False
    if actual['state'] not in TERMINAL:
        intent=HERE/'migration_stop_requested.json'
        if not intent.exists():
            write(intent,{'job':dep,'first_complete_case':progress.get('last_result'),'at':time.time()})
            value=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','stop',dep['job'],
                                  '--workspace-name=share-space'],capture_output=True,text=True,timeout=40)
            assert value.returncode==0, 'Could not stop the exact prior writer'
        return False
    workers=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','get-workers',dep['job'],
                            '--workspace-name=share-space'],capture_output=True,text=True,check=True,timeout=40)
    if 'Running' in workers.stdout:
        return False
    total=0.
    audit=[]
    for path in (old/'jobs').glob('*/submission.json'):
        previous=json.loads(path.read_text());state=describe(previous['job'])
        assert state['uid']==previous['uid'] and state['state'] in TERMINAL
        if state.get('start_time'):
            end=state.get('complete_time') or state['update_time']
            seconds=max(0.,(datetime.fromisoformat(end.replace('Z','+00:00'))-
                           datetime.fromisoformat(state['start_time'].replace('Z','+00:00'))).total_seconds())
        else:
            seconds=0.
        total+=seconds
        audit.append({'job':previous['job'],'uid':state['uid'],'gpu_seconds_upper_bound':seconds})
    write(ready,{'first_case_complete':True,'old_worker_stopped':True,'first_case':progress.get('last_result'),
                 'previous_generation_gpu_seconds_upper_bound':total,'audit':audit,
                 'fixed_cases_reused':True,'at':time.time()})
    return True


def main():
    lock = (HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan = json.loads((HERE/'plan.json').read_text());started = time.time()
    process_file = HERE/'controller_process.json'
    if process_file.exists():
        started = json.loads(process_file.read_text())['first_started_at']
    write(process_file,{'pid':os.getpid(),'first_started_at':started,'latest_started_at':time.time(),
                        'max_live_gpus':8,'max_gpu_hours':80,'max_wall_hours':72})
    client = aoss_client();previous = None
    try:
        while time.time()<started+plan['max_wall_hours']*3600:
            try:
                if not prepare_migration(client,plan):
                    threading.Event().wait(20);continue
                rows = inspect(client)
                if (HERE/'CANCELLED_BY_USER').exists():
                    stop_active(rows,'user_cancelled');return
                dependencies = []
                dependency_receipts = [json.loads(path.read_text()) for path in
                    (Path(plan['dependency_campaign_directory'])/'jobs').glob('*/submission.json')]
                for item in dependency_receipts:
                    if item['phase']=='sample':
                        continue
                    actual = describe(item['job'])
                    assert actual['uid']==item['uid'] and actual['ownership']['user_name']=='260010165'
                    dependencies.append({'job':item['job'],'uid':item['uid'],'state':actual['state']})
                for row in rows:
                    if row['state'] in TERMINAL:
                        try:
                            collect(client,row)
                        except client.exceptions.ClientError as error:
                            if error.response['Error']['Code'] not in ('NoSuchKey','404'):
                                raise
                # Platform completion alone is insufficient. The worker's
                # completion state and collected integrity check must agree.
                for row in rows:
                    if row['state']=='SUCCEEDED':
                        task=f"{plan['worker_prefix']}{row['phase']}_r{row['rank']}_a{row['attempt']}"
                        folder=f"{row['phase']}_r{row['rank']}_a{row['attempt']}"
                        check=HERE/'retrieved'/folder/'checks'/f'{task}.json'
                        assert row.get('progress',{}).get('state')=='complete', 'Missing completed worker status'
                        assert check.is_file(), 'Completed job has no collected integrity check'
                        proof=json.loads(check.read_text())
                        assert proof['completed'] and proof['frozen_unchanged']
                phase,current = next_pending(plan,rows)
                predecessor = json.loads(Path(plan['dependency_status_file']).read_text())
                waiting = not predecessor.get('finished') or any(d['state'] not in TERMINAL for d in dependencies)
                status = {'observed_at':datetime.now(timezone.utc).isoformat(),'experiment':plan['experiment'],
                          'state':'waiting_for_radio' if waiting else 'running',
                          'dependencies':dependencies,'next_phase':phase,'workers':rows,'finished':False}
                write(HERE/'latest_status.json',status)
                pointer={'experiment':plan['experiment'],'status':status['state'],
                         'directory':str(HERE),'observed_at':status['observed_at'],
                         'maximum_live_gpus':8,'maximum_gpu_hours':80}
                if waiting:
                    write(HERE.parent/'queued_experiment.json',pointer)
                else:
                    active_file=HERE.parent/'active_run.json'
                    if active_file.exists() and not (HERE/'previous_active_run.json').exists():
                        write(HERE/'previous_active_run.json',json.loads(active_file.read_text()))
                    write(active_file,pointer)
                compact = {**status,'workers':[{k:r.get(k) for k in ('phase','rank','attempt','job','uid','state')}
                                               |{'stage':r.get('progress',{}).get('stage')} for r in rows]}
                compact.pop('observed_at')
                encoded = json.dumps(compact,sort_keys=True)
                if encoded!=previous:
                    print(json.dumps(compact),flush=True);previous=encoded
                if status['state']=='waiting_for_radio':
                    threading.Event().wait(45);continue
                if phase is None:
                    result=finalize(plan);status.update(state='complete',finished=True,outcome=result)
                    write(HERE/'latest_status.json',status)
                    pointer.update(status='complete',outcome_path=str(HERE/'outcome.json'))
                    write(HERE.parent/'active_run.json',pointer);return
                active = sum(r['state'] not in TERMINAL for r in rows)
                made_submission = False
                for rank,row in zip(phase_ranks(phase),current):
                    if row and row['state']=='SUCCEEDED':
                        continue
                    if row and row['state'] in TERMINAL:
                        repaired = row['uid'] in plan.get('startup_repair',{}).get('recover_previous_uids',[])
                        cap=plan.get('attempt_caps',{}).get(phase,plan['max_attempts_per_phase'])
                        if row['attempt']>=cap or not (retryable(row) or repaired):
                            write(HERE/'stopped.json',{'reason':'phase_failed','worker':row})
                            stop_active(rows,'unrecoverable_phase_failure');return
                    if active<plan['max_live_gpus'] and (row is None or row['state'] in TERMINAL):
                        if phase=='baseline' and rank==1 and plan.get('first_case_gate'):
                            first=latest(rows,'baseline',0)
                            done=first.get('progress',{}).get('completed_evaluations',0) if first else 0
                            if not (first and (first['state']=='SUCCEEDED' or done>=1)):
                                continue
                        submit(phase,rank,1 if row is None else row['attempt']+1)
                        made_submission=True;break
                if made_submission:
                    continue
            except Exception as error:
                # A failed query is unknown state, never permission to resubmit.
                message = str(error)[:400]
                write(HERE/'controller_last_error.json',{'at':time.time(),'type':type(error).__name__,'message':message})
                print(json.dumps({'controller_error':type(error).__name__,'message':message}),flush=True)
                if isinstance(error,AssertionError) and 'budget' in message.lower():
                    stop_active(inspect(client),'fixed_compute_budget');return
            threading.Event().wait(45)
        stop_active(inspect(client),'fixed_wall_deadline')
        write(HERE/'deadline.json',{'max_wall_hours':72})
    finally:
        client.close()


if __name__=='__main__':
    main()
