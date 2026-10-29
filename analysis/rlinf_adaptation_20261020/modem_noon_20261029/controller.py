from datetime import datetime, timezone
import fcntl,hashlib,json,os,subprocess,tarfile,threading,time
from pathlib import Path
from deploy import HERE,describe,submit,write,aoss_client
from eval_utils import paired_report
TERMINAL={'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}

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


def original_baseline(client,plan):
    dep=plan['first_case_dependency'];actual=describe(dep['job'])
    assert actual['uid']==dep['uid'] and actual['ownership']['user_name']=='260010165'
    source=Path(plan['previous_generation_directory'])/'jobs/baseline_r0_a3/submission.json'
    receipt=json.loads(source.read_text());progress={}
    try:
        value=client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'status.json')
        progress=json.loads(value['Body'].read())
    except client.exceptions.ClientError as error:
        if error.response['Error']['Code'] not in ('NoSuchKey','404'):raise
    complete=progress.get('completed_evaluations',0)>=1
    if complete and actual['state'] not in TERMINAL:
        if not (HERE/'predecessor_stop_requested.json').exists():
            write(HERE/'predecessor_stop_requested.json',{'job':dep,'at':time.time(),'first_complete_result':progress.get('last_result')})
            result=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','stop',dep['job'],'--workspace-name=share-space'],
                                  capture_output=True,text=True,timeout=40)
            assert result.returncode==0
    stopped=actual['state'] in TERMINAL
    if stopped:
        workers=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','get-workers',dep['job'],'--workspace-name=share-space'],
                               capture_output=True,text=True,check=True,timeout=40)
        stopped='Running' not in workers.stdout
    state={'job':dep,'state':actual['state'],'progress':progress,
           'first_case_complete':complete,'old_worker_stopped':stopped,'observed_at':time.time()}
    write(HERE/'predecessor_state.json',state)
    if stopped and not complete:
        raise RuntimeError('Original baseline failed before completing a case; inspect before further training')
    return state


def finish(plan):
    root=HERE/'final_evidence';sources=list((HERE/'retrieved').glob('*'))
    for phase in ('baseline','final'):
        for instance,seed in plan['evaluation']['cases']:
            case=f'i{instance}_s{seed}'
            candidates=[s/phase/case for s in sources if all((s/phase/case/n).is_file()
                        for n in ('result.json','initial.json','spec.json','actions.json','goal_summary.json'))]
            assert candidates,(phase,case)
            for name in ('result.json','initial.json','spec.json','goal_summary.json'):
                body=(candidates[-1]/name).read_bytes();assert all((p/name).read_bytes()==body for p in candidates)
                output=root/phase/case/name;output.parent.mkdir(parents=True,exist_ok=True);output.write_bytes(body)
    report={'training_instances':paired_report(root,plan['evaluation']['training_cases']),
            'heldout_instances':paired_report(root,plan['evaluation']['heldout_cases']),
            'all_four_pairs':paired_report(root,plan['evaluation']['cases'])}
    assert all(value['comparison_complete'] and value['comparison_valid'] for value in report.values())
    ready=[json.loads(p.read_text()) for s in sources for p in (s/'checkpoints').glob('round_01/ready.json')]
    assert ready and ready[-1]['optimizer_steps']==2
    report.update(optimizer_steps=2,training_episodes=4,checkpoint=ready[-1],
                  completed_at=datetime.now(timezone.utc).isoformat(),deadline_shanghai=plan['deadline_shanghai'],
                  scope='Four-pair timing/trainability pilot; cannot establish stable improvement')
    write(HERE/'outcome.json',report);return report


def main():
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((HERE/'plan.json').read_text());start=time.time();client=aoss_client();last=None
    write(HERE/'controller_process.json',{'pid':os.getpid(),'started_at':start,'max_live_gpus':8,'max_gpu_hours':12})
    intake=[('baseline',rank) for rank in (1,2,3)]+[('sample1',rank) for rank in range(4)]
    try:
        while time.time()<start+plan['max_wall_hours']*3600:
            rows=[]
            try:
                old=original_baseline(client,plan);rows=inspect(client)
                if (HERE/'CANCELLED_BY_USER').exists():stop_active(rows,'user_cancelled');return
                for row in rows:
                    if row['state'] in TERMINAL:
                        try:collect(client,row)
                        except client.exceptions.ClientError as error:
                            if error.response['Error']['Code'] not in ('NoSuchKey','404'):raise
                latest_rows=[latest(rows,p,r) for p,r in intake]
                intake_done=old['first_case_complete'] and old['old_worker_stopped'] and all(r and r['state']=='SUCCEEDED' for r in latest_rows)
                train=latest(rows,'train1',0)
                if not intake_done:
                    targets=intake;stage='parallel_baseline_and_sampling'
                elif not train or train['state']!='SUCCEEDED':
                    targets=[('train1',0)];stage='ppo_update'
                else:
                    targets=[('final',rank) for rank in range(4)];stage='paired_evaluation'
                status={'state':'running','stage':stage,'experiment':plan['experiment'],
                        'observed_at':datetime.now(timezone.utc).isoformat(),'workers':rows,'original_baseline':old,
                        'finished':False,'deadline_shanghai':plan['deadline_shanghai']}
                write(HERE/'latest_status.json',status)
                write(HERE.parent/'active_run.json',{'experiment':plan['experiment'],'directory':str(HERE),
                      'status':'running','stage':stage,'observed_at':status['observed_at']})
                compact=json.dumps({'stage':stage,'old_state':old['state'],
                    'workers':[{k:r.get(k) for k in ('job','phase','rank','attempt','state')} |
                               {'stage':r.get('progress',{}).get('stage')} for r in rows]},sort_keys=True)
                if compact!=last:print(compact,flush=True);last=compact
                if stage=='paired_evaluation' and all(latest(rows,'final',r) and latest(rows,'final',r)['state']=='SUCCEEDED' for r in range(4)):
                    report=finish(plan);status.update(state='complete',finished=True,outcome=report)
                    write(HERE/'latest_status.json',status);return
                active=sum(r['state'] not in TERMINAL for r in rows)+int(old['state'] not in TERMINAL)
                submitted=False
                for phase,rank in targets:
                    row=latest(rows,phase,rank)
                    if row and row['state']=='SUCCEEDED':continue
                    if row and row['state'] in TERMINAL:
                        if row['attempt']>=plan['max_attempts_per_phase'] or not retryable(row):
                            write(HERE/'stopped.json',{'reason':'phase_failed','worker':row})
                            stop_active(rows,'phase_failure');return
                    if active<8 and (row is None or row['state'] in TERMINAL):
                        submit(phase,rank,1 if row is None else row['attempt']+1);submitted=True;break
                if submitted:continue
            except Exception as error:
                write(HERE/'controller_last_error.json',{'at':time.time(),'error_type':type(error).__name__,'error':str(error)[:500]})
                print(json.dumps({'error':type(error).__name__,'message':str(error)[:400]}),flush=True)
                if isinstance(error,RuntimeError) and 'Original baseline failed' in str(error):
                    stop_active(rows,'original_baseline_failed');return
            threading.Event().wait(20)
        stop_active(inspect(client),'three_hour_pilot_budget')
        write(HERE/'deadline.json',{'reason':'pilot maximum wall reached; incomplete results remain incomplete'})
    finally:client.close()


if __name__=='__main__':main()


