"""Submit finite experiment phases with unique receipts and owner-wide admission."""
import argparse
import base64
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=HERE.parent
CLI='/home/faye/.local/bin/sco'
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from public_test_admission import query_standard_capacity
from video_eval_adapter import aoss_client
from recovery_policy import attempt_cap, gpu_seconds_upper, other_rl_jobs


def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def describe(name):
    p=subprocess.run([CLI,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=40)
    return json.loads(p.stdout)


def submit(phase,rank,attempt=1):
    assert phase in ('baseline','sample1','train1','sample2','train2','final')
    plan=json.loads((HERE/'plan.json').read_text())
    assert rank in ((0,) if phase.startswith('train') else range(plan['parallel_workers']))
    task=f'{phase}_r{rank}_a{attempt}'
    target=HERE/'jobs'/task;intent=target/'submission.json'
    assert not intent.exists(), 'Reconcile the existing intent; never submit twice'
    assert not (HERE/'CANCELLED_BY_USER').exists()
    plan=json.loads((HERE/'plan.json').read_text())
    migration=json.loads((HERE/'migration_ready.json').read_text())
    assert migration['old_worker_stopped'] and migration['first_case_complete']
    cap=attempt_cap(plan,phase,HERE)
    assert 1 <= attempt <= cap
    for name in ('test_result.json','preflight_result.json'):assert json.loads((HERE/name).read_text())['passed']
    manifest=json.loads((HERE/'execution_manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==digest, 'Frozen modem source changed: '+name
    dependency_status=json.loads(Path(plan['dependency_status_file']).read_text())
    assert dependency_status.get('finished'), 'Wait for the radio campaign to finish without interrupting its controller'
    for dependency in plan['dependencies']:
        prior=describe(dependency['job'])
        assert prior['uid']==dependency['uid'] and prior['ownership']['user_name']=='260010165'
        assert prior['state'] in {'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}, 'Finish radio evaluation before allocating modem GPUs'
    receipts=[json.loads(p.read_text()) for p in (HERE/'jobs').glob('*/submission.json')]
    active=[];upper=migration['previous_generation_gpu_seconds_upper_bound']
    terminal={'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}
    assert all('uid' in previous for previous in receipts),'Unresolved submission intent exists'
    with ThreadPoolExecutor(max_workers=6) as pool:
        previous_states=list(pool.map(lambda r:describe(r['job']),receipts))
    for previous,old in zip(receipts,previous_states):
        if 'uid' not in previous:raise RuntimeError('Unresolved submission intent exists')
        assert old['uid']==previous['uid'] and old['ownership']['user_name']=='260010165'
        if old['state'] not in terminal:
            active.append(old);upper+=gpu_seconds_upper(old,previous)
        elif old['uid'] in plan.get('prior_terminal_gpu_seconds_upper_bound',{}):
            upper+=plan['prior_terminal_gpu_seconds_upper_bound'][old['uid']]
        else:upper+=gpu_seconds_upper(old,previous)
    assert len(active)<plan['max_live_gpus'],'At most eight active experiment GPUs'
    other=other_rl_jobs(HERE,describe)
    if len(active)+len(other)>=plan['max_live_gpus']:
        raise RuntimeError('Waiting for shared RL slot; existing multi-task baseline is still active')
    limit=plan['phase_limits_seconds'][phase]
    assert upper+limit<=plan['max_total_gpu_hours']*3600,'Fixed campaign compute budget exhausted'
    if attempt>1:
        prior=json.loads((HERE/'jobs'/f'{phase}_r{rank}_a{attempt-1}'/'submission.json').read_text())
        old=describe(prior['job']);assert old['state'] in terminal
        p=subprocess.run([CLI,'acp','jobs','get-workers',prior['job'],'--workspace-name=share-space'],capture_output=True,text=True,check=True,timeout=40)
        assert 'Running' not in p.stdout,'Prior writer must stop before reuse'
    capacity=query_standard_capacity(CLI,dict(os.environ))
    target.mkdir(parents=True,exist_ok=True);write(target/'capacity.json',capacity)
    quota='reserved' if capacity['used_gpus']<capacity['standard_limit'] else 'spot'
    template=describe('b1k-repeat1688-1019-batch-00')
    assert template['ownership']['user_name']=='260010165'
    assert template['resource_pool']['name']=='computing-cluster-5090-01g'
    assert template['roles'][0]['resource_spec'][0]['name']=='n12lp.nn.i10a.1'
    label={'baseline':'b','sample1':'s1','train1':'t1','sample2':'s2','train2':'t2','final':'f'}[phase]
    job=f'b1k-mdmp-1029-{label}r{rank}a{attempt}'
    prefix=f'behavior1k/rlinf-modem-20261028/parallel8_20261029/{task}/'
    remote=plan['remote_root']+'/jobs/'+plan['worker_prefix']+task
    names=['common.py','training.py','experiment.py','simulator.py','helpers.py','driver.py','goals.py','eval_utils.py','resource_watch.py','plan.json']
    names += ['b1k_rlinf/'+n+'.py' for n in ('__init__','action_interface','factory','io','model','precision','tokenizer')]
    buffer=io.BytesIO();hashes={}
    with tarfile.open(fileobj=buffer,mode='w:gz',compresslevel=9) as archive:
        for name in names:
            data=(HERE/name).read_bytes();item=tarfile.TarInfo('run/'+name);item.size=len(data);item.mtime=0
            archive.addfile(item,io.BytesIO(data));hashes[name]=hashlib.sha256(data).hexdigest()
    bundle=buffer.getvalue();(target/'bundle.tar.gz').write_bytes(bundle)
    before=json.loads((BASE/'cloud_smoke/local_preflight.json').read_text());artifacts=json.loads((BASE/'cloud_smoke/artifacts.json').read_text())
    reuse={'checkpoint/'+k:v['sha256'] for k,v in artifacts['checkpoint'].items()};reuse['bundle.tar.gz']=artifacts['bundle']['sha256']
    reuse.update({k:v for k,v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
    for name in ['rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py','rlinf/models/embodiment/openpi/tasks/rl.py']:
        reuse['upstream/'+name]=hashlib.sha256((BASE/'upstream'/name).read_bytes()).hexdigest()
    receipt={'job':job,'phase':phase,'rank':rank,'attempt':attempt,'state':'preparing','quota':quota,'gpu_count':1,
             'max_seconds':limit,'source_hashes':hashes,'aoss_prefix':prefix,'output':plan['output'],'remote_job_root':remote,
             'bundle_bytes':len(bundle),'submitted_at':datetime.now(timezone.utc).isoformat()}
    with intent.open('x') as file:json.dump(receipt,file,indent=2)
    client=aoss_client()
    try:
        client.put_object(Bucket='260010165',Key=prefix+'bundle.tar.gz',Body=bundle)
        assert client.get_object(Bucket='260010165',Key=prefix+'bundle.tar.gz')['Body'].read()==bundle
        def signed(name):return client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':prefix+name},ExpiresIn=432000)
        config={'root':remote,'base':'/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
                'task_phase':phase,'rank':rank,'attempt':attempt,'job_name':job,'bundle_inline':base64.b64encode(bundle).decode(),
                'bundle':{'bytes':len(bundle),'sha256':hashlib.sha256(bundle).hexdigest()},'reuse_sha256':reuse,
                'status_put':signed('status.json'),'result_put':signed('result.tar.gz'),'artifact_puts':{}}
        source=(HERE/'bootstrap.py').read_text()+'\nguarded_run('+repr(config)+')\n'
        packed=base64.b64encode(zlib.compress(source.encode())).decode()
        command=f'timeout --signal=TERM --kill-after=60s {limit-60}s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        assert len(command)<64000
        receipt.update(state='submitting',startup_sha256=hashlib.sha256(command.encode()).hexdigest());write(intent,receipt)
        assert not (HERE/'CANCELLED_BY_USER').exists()
        try:
            p=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=b1k-repeat1688-1019-batch-00',
              '--name='+job,'--job-name='+job,'--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type='+quota,
              '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
              '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
              '--container-image-url='+template['roles'][0]['image_path'],'--command='+command],capture_output=True,text=True,timeout=90)
            receipt['submit_returncode']=p.returncode
            if p.returncode:receipt['submit_error']=re.sub(r'https?://\S+','[URL omitted]',p.stderr)[-700:]
        except subprocess.TimeoutExpired:receipt['submit_returncode']='unknown_timeout'
        write(intent,receipt)
        actual=describe(job);assert actual['ownership']['user_name']=='260010165'
        assert actual['scheduling']['quota_type']==quota.upper()
        assert hashlib.sha256(actual['roles'][0]['startup_script'].encode()).hexdigest()==receipt['startup_sha256']
        receipt.update(uid=actual['uid'],state=actual['state']);write(intent,receipt)
        print(json.dumps({k:receipt[k] for k in ['job','uid','state','phase','rank','attempt','quota','bundle_bytes']}),flush=True)
        return receipt
    finally:client.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--attempt',type=int,default=1)
    a=p.parse_args();submit(a.phase,a.rank,a.attempt)
