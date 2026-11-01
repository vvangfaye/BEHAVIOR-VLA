"""One bounded scheduler for the predeclared comparison; no best-case retries."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import fcntl
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import time
from deployment import HERE,ADAPT,CLI,ENV,TERMINAL,aoss_client,describe,save,submit


def job_id(role,variant,case=None,attempt=1):
    tag=variant.replace('_','-')
    return f't-{tag}-a{attempt}' if role=='train' else f'e-{tag}-{case[0]}s{case[1]}-a{attempt}'


def seconds(job,receipt):
    start=job.get('start_time')
    if job['state'] not in TERMINAL:return float(receipt['max_seconds'])*receipt.get('gpu_count',1)
    if not start:return 0.
    end=job.get('complete_time') or job.get('update_time')
    if not end:return float(receipt['max_seconds'])*receipt.get('gpu_count',1)
    parse=lambda s:datetime.fromisoformat(s.replace('Z','+00:00'))
    return max(0.,(parse(end)-parse(start)).total_seconds())*receipt.get('gpu_count',1)


def inspect(client):
    receipts=[(p,json.loads(p.read_text())) for p in sorted((HERE/'jobs').glob('*/submission.json'))]
    def one(pair):
        path,receipt=pair;cached=path.with_name('latest.json')
        if cached.exists():
            previous=json.loads(cached.read_text())
            if previous['state'] in TERMINAL:return previous
        job=describe(receipt['job'])
        if 'uid' not in receipt:
            receipt['uid']=job['uid'];save(path,receipt)
        assert job['uid']==receipt['uid']
        row={**receipt,'state':job['state'],'gpu_seconds_bound':seconds(job,receipt),'checked_at':time.time()}
        try:
            response=client.get_object(Bucket='260010165',Key=receipt['aoss_prefix']+'status.json')
            assert response['ContentLength']<1024**2
            row['progress']=json.loads(response['Body'].read())
        except client.exceptions.ClientError as error:
            if error.response['Error']['Code'] not in ('NoSuchKey','404'):raise
        save(cached,row);return row
    with ThreadPoolExecutor(max_workers=4) as pool:return list(pool.map(one,receipts))


def collect(client,row):
    folder=HERE/'jobs'/row['id'];done=folder/'collected.json'
    if done.exists():return
    response=client.get_object(Bucket='260010165',Key=row['aoss_prefix']+'result.tar.gz')
    assert response['ContentLength']<256*1024**2
    body=response['Body'].read();root=folder/'retrieved'
    with tarfile.open(fileobj=io.BytesIO(body),mode='r:gz') as archive:
        for item in archive:
            name=Path(item.name)
            assert item.isfile() and not name.is_absolute() and '..' not in name.parts and name.suffix=='.json'
            data=archive.extractfile(item).read();json.loads(data)
            dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():assert dest.read_bytes()==data
            else:dest.write_bytes(data)
    if row['role']=='train':
        ready=json.loads((root/'checkpoints/round_01/ready.json').read_text())
        assert ready['round_trip'] and ready['variant']==row['variant'] and 0<=ready['optimizer_steps']<=17
        save(HERE/'checkpoints'/f"{row['variant']}.json",ready)
    else:
        phase='baseline' if row['variant']=='original' else 'final';i,s=row['case']
        source=root/phase/f'i{i}_s{s}';dest=HERE/'evidence'/row['variant']/f'i{i}_s{s}'
        for name in ['result.json','initial.json','spec.json','actions.json','goal_summary.json']:
            data=(source/name).read_bytes();target=dest/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==data
            else:target.write_bytes(data)
        result=json.loads((dest/'result.json').read_text());assert (result['instance'],result['seed'])==(i,s)
        if row['variant']=='original' and (i,s)==(301,1):
            plan=json.loads((folder/'worker_plan.json').read_text())
            assert plan['seed']==plan['model_seed']==0
            identity=json.loads(next((root/'identities').glob('*.json')).read_text())
            assert identity['loaded_policy_hash']==plan['source_identity']['cold_trainable_hash']
            assert json.loads((dest/'spec.json').read_text())['seed']==1
            assert result['steps']>0
            save(HERE/'seed_fix_canary_verified.json',{'at':time.time(),'job':row['job'],'uid':row['uid'],
                'complete_episode':True,'model_seed':0,'evaluation_seed':1,'steps':result['steps'],
                'loaded_policy_hash':identity['loaded_policy_hash']})
    save(done,{'complete':True,'bytes':len(body),'at':time.time()})


def comparison(variant,cases):
    pairs=[]
    for instance,seed in cases:
        key=f'i{instance}_s{seed}';a=HERE/'evidence/original'/key;b=HERE/'evidence'/variant/key
        if not (a/'result.json').exists() or not (b/'result.json').exists():continue
        before=json.loads((a/'initial.json').read_text());after=json.loads((b/'initial.json').read_text())
        assert before['physical_state'].keys()==after['physical_state'].keys()
        error=max(abs(x-y) for k in before['physical_state'] for part in (0,1)
                  for x,y in zip(before['physical_state'][k][part],after['physical_state'][k][part]))
        pe=max(abs(x-y) for x,y in zip(before['proprio'],after['proprio']))
        valid=error<=1e-4 and pe<=1e-4 and all(before[k]==after[k] for k in ['physics_dt','rendering_dt','observation_dt'])
        old=json.loads((a/'result.json').read_text());new=json.loads((b/'result.json').read_text())
        pairs.append({'instance':instance,'seed':seed,'valid_initial_state':valid,
            'baseline_score':old['score'],'score':new['score'],
            'baseline_success':old['success'],'success':new['success']})
    n=len(pairs)
    return {'expected_pairs':len(cases),'completed_pairs':n,'complete':n==len(cases),
        'valid':bool(pairs) and all(x['valid_initial_state'] for x in pairs),
        'baseline_success_rate':sum(x['baseline_success'] for x in pairs)/n if n else None,
        'success_rate':sum(x['success'] for x in pairs)/n if n else None,
        'baseline_mean_score':sum(x['baseline_score'] for x in pairs)/n if n else None,
        'mean_score':sum(x['score'] for x in pairs)/n if n else None,'pairs':pairs}


def copy_control_results():
    source=ADAPT/'multitask_validation_20261029/water/retrieved'
    for instance in range(301,321):
        case=f'i{instance}_s0';names=['result.json','initial.json','spec.json','actions.json','goal_summary.json']
        choices=[p for p in source.glob('*/final/'+case) if all((p/n).exists() for n in names)]
        if not choices:continue
        for name in names:
            bodies=[(p/name).read_bytes() for p in choices];assert all(x==bodies[0] for x in bodies)
            target=HERE/'evidence/control'/case/name;target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert target.read_bytes()==bodies[0]
            else:target.write_bytes(bodies[0])


def candidates(plan,rows):
    ready=(HERE/'implementation_verified.json').exists()
    if ready and not (HERE/'TRAINING_HANDOFF.json').exists():
        for variant in plan['variants']:
            yield {'role':'train','variant':variant}
    for case in plan['new_baseline_cases']:
        if case!=[301,1] and not (HERE/'seed_fix_canary_verified.json').exists():continue
        yield {'role':'eval','variant':'original','case':case}
    if ready:
        for variant in plan['variants']:
            if (HERE/'checkpoints'/f'{variant}.json').exists():
                for case in plan['eval_cases']:yield {'role':'eval','variant':variant,'case':case}
        control=ADAPT/'multitask_validation_20261029/water/retrieved'
        if any(control.glob('train1*/checkpoints/round_01/ready.json')):
            for case in plan['control_extra_cases']:yield {'role':'eval','variant':'control','case':case}


def main():
    lock=(HERE/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=json.loads((HERE/'plan.json').read_text());deadline=datetime.fromisoformat(plan['created_at']).timestamp()+plan['max_wall_hours']*3600
    client=aoss_client();(HERE/'jobs').mkdir(exist_ok=True)
    try:
        while time.time()<deadline:
            try:
                rows=inspect(client)
                for row in rows:
                    if row['state']=='SUCCEEDED':collect(client,row)
                copy_control_results()
                report={v:comparison(v,plan['eval_cases']) for v in [*plan['variants'],'control']}
                save(HERE/'comparisons.json',report)
                active=[r for r in rows if r['state'] not in TERMINAL]
                used=sum(r['gpu_seconds_bound'] for r in rows)
                original_done=(ADAPT/'multitask_validation_20261029/water/outcome.json').exists()
                reserved_original=0 if original_done else plan.get('reserve_original_until_complete',0)
                allowed_new=min(plan['max_new_gpu_jobs'],plan.get('max_total_comparison_and_original_gpus',plan['max_new_gpu_jobs'])-reserved_original)
                save(HERE/'status.json',{'at':time.time(),'active_jobs':len(active),'active_gpus':sum(r.get('gpu_count',1) for r in active),'gpu_hours_bound':used/3600,
                    'allowed_new_jobs':allowed_new,'reserved_original_slots':reserved_original,
                    'total_gpu_cap':plan.get('max_total_comparison_and_original_gpus',plan['max_new_gpu_jobs']),
                    'workers':rows,'comparisons':{v:{k:x for k,x in p.items() if k!='pairs'} for v,p in report.items()}})
                if all(report[v]['complete'] for v in plan['variants']):
                    save(HERE/'outcome.json',{'at':time.time(),'comparisons':report,
                        'scope':'One training seed, fixed shared 32-trajectory batch, 20 held-out instances with two evaluation seeds.'})
                    # The original control may still be running; its extra seed is a separate pending comparison.
                    control_done=report['control']['complete']
                    if control_done:return
                if sum(r.get('gpu_count',1) for r in active)>=allowed_new:
                    time.sleep(20);continue
                launched=False
                for spec in candidates(plan,rows):
                    relevant=[r for r in rows if r['role']==spec['role'] and r['variant']==spec['variant'] and r.get('case')==spec.get('case')]
                    last=max(relevant,key=lambda r:r['attempt']) if relevant else None
                    if last and last['state']=='SUCCEEDED':continue
                    if last and last['state'] not in TERMINAL:continue
                    attempt=1 if last is None else last['attempt']+1
                    if attempt>plan['attempt_cap']:continue
                    if last:
                        error=last.get('progress',{}).get('error_type')
                        seed_fix=False
                        if error=='AssertionError' and last['variant']=='original' and last['attempt']==1:
                            response=client.get_object(Bucket='260010165',Key=last['aoss_prefix']+'result.tar.gz')
                            assert response['ContentLength']<256*1024**2
                            with tarfile.open(fileobj=io.BytesIO(response['Body'].read()),mode='r:gz') as archive:
                                for item in archive:
                                    if item.isfile() and item.name.startswith('failures/') and item.name.endswith('.json'):
                                        failure=json.loads(archive.extractfile(item).read())
                                        seed_fix=('assert policy_hash(model) == plan[\'source_identity\'][\'cold_trainable_hash\']' in failure.get('traceback',''))
                                        if seed_fix:save(HERE/'jobs'/last['id']/'seed_initialization_failure.json',failure);break
                        if not seed_fix and error not in (None,'ExecutionStopped','SimulatorInitializationError','TimeoutError','EOFError','BrokenPipeError'):continue
                        check=subprocess.run([CLI,'acp','jobs','get-workers',last['job'],'--workspace-name=share-space'],
                            env=ENV,capture_output=True,text=True,timeout=40,check=True)
                        if 'Running' in check.stdout:continue
                    seconds_requested=plan['train_timeout_seconds'] if spec['role']=='train' else plan['eval_timeout_seconds']
                    if used+seconds_requested>plan['max_total_gpu_hours']*3600:
                        save(HERE/'budget_stop.json',{'at':time.time(),'gpu_hours_bound':used/3600});return
                    spec.update(attempt=attempt,id=job_id(**spec,attempt=attempt))
                    submit(spec);launched=True;break
                if not launched:time.sleep(20)
            except Exception as error:
                message=str(error).split('--command')[0]
                message=re.sub(r'https?://\S+','[URL]',message)
                save(HERE/'controller_error.json',{'at':time.time(),'type':type(error).__name__,'message':message[:600]})
                time.sleep(20)
        for row in inspect(client):
            if row['state'] not in TERMINAL:
                subprocess.run([CLI,'acp','jobs','stop',row['job'],'--workspace-name=share-space'],
                               env=ENV,capture_output=True,timeout=45,check=True)
        save(HERE/'deadline_reached.json',{'at':time.time(),'new_jobs_stopped':True,'original_control_untouched':True})
    finally:client.close()


if __name__=='__main__':main()
