"""Atomically request two standard GPUs for the preserved PPO experiment."""
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
BASE = HERE.parent.parent
CLI = '/home/faye/.local/bin/sco'
JOB = 'b1k-s32-stdpair-1026-01'
PREFIX = 'behavior1k/rlinf-sampling32-20261025/resume3-reserved-20261026/pilot01/'
REMOTE = '/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/resume3_reserved_20261026/pilot01'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temporary.replace(path)


def describe(name):
    p = subprocess.run([CLI,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],
                       capture_output=True,text=True,check=True,timeout=40)
    return json.loads(p.stdout)


def main():
    assert not (HERE/'CANCELLED_BY_USER').exists(), 'Cancelled by user; do not submit'
    intent = HERE/'submission_pair.json'
    assert not intent.exists(), 'Submission intent already exists: inspect its exact job, never resubmit blindly'
    for name in ('resume_test_result.json','configuration_equivalence.json','preflight_result.json'):
        assert json.loads((HERE/name).read_text())['passed']
    plan = json.loads((HERE/'plan.json').read_text())
    assert plan['gpu_count']==2 and plan['max_job_seconds']==48600
    assert plan['optimizer']['updates_per_round']==[17,13]
    for previous in plan['resume']['previous_jobs']:
        actual=describe(previous['name'])
        assert actual['uid']==previous['uid'] and actual['ownership']['user_name']=='260010165'
        assert actual['state'] in ('FAILED','SUSPENDED')
        p=subprocess.run([CLI,'acp','jobs','get-workers',previous['name'],'--workspace-name=share-space'],
                         capture_output=True,text=True,check=True,timeout=40)
        assert 'Running' not in p.stdout and ('Deleted' in p.stdout or 'Failed' in p.stdout)
    template=describe('b1k-repeat1688-1019-batch-00')
    assert template['ownership']['user_name']=='260010165'
    assert template['resource_pool']['name']=='computing-cluster-5090-01g'
    assert template['roles'][0]['resource_spec'][0]['name']=='n12lp.nn.i10a.1'
    sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
    from public_test_admission import query_standard_capacity
    from video_eval_adapter import aoss_client
    capacity=query_standard_capacity(CLI,dict(os.environ))
    write(HERE/'submission_capacity.json',capacity)
    if capacity['used_gpus'] > capacity['standard_limit']-2:
        print('WAIT_STANDARD_CAPACITY',flush=True)
        return 75
    files=['common.py','training.py','driver.py','experiment.py','parallel.py','helpers.py','simulator.py','plan.json']
    files += [str(p.relative_to(HERE)) for p in sorted((HERE/'b1k_rlinf').glob('*.py'))]
    buffer=io.BytesIO();hashes={}
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name in files:
            raw=(HERE/name).read_bytes();hashes[name]=hashlib.sha256(raw).hexdigest()
            item=tarfile.TarInfo('run/'+name);item.size=len(raw);item.mtime=0
            archive.addfile(item,io.BytesIO(raw))
    bundle=buffer.getvalue();(HERE/'bundle.tar.gz').write_bytes(bundle)
    before=json.loads((BASE/'cloud_smoke/local_preflight.json').read_text())
    artifacts=json.loads((BASE/'cloud_smoke/artifacts.json').read_text())
    reuse={'checkpoint/'+k:v['sha256'] for k,v in artifacts['checkpoint'].items()}
    reuse['bundle.tar.gz']=artifacts['bundle']['sha256']
    reuse.update({k:v for k,v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
    for name in ['rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py','rlinf/models/embodiment/openpi/tasks/rl.py']:
        reuse['upstream/'+name]=hashlib.sha256((BASE/'upstream'/name).read_bytes()).hexdigest()
    receipt={'job':JOB,'state':'preparing','gpu_count':2,'quota':'reserved',
        'max_seconds':48600,'source_hashes':hashes,'aoss_prefix':PREFIX,'output':plan['output'],
        'submitted_at':datetime.now(timezone.utc).isoformat(),'atomic_two_worker_request':True}
    # Persist the unique request before any external mutation.
    with intent.open('x') as stream:json.dump(receipt,stream,indent=2)
    client=aoss_client()
    try:
        client.put_object(Bucket='260010165',Key=PREFIX+'bundle.tar.gz',Body=bundle)
        assert client.get_object(Bucket='260010165',Key=PREFIX+'bundle.tar.gz')['Body'].read()==bundle
        def signed(name):
            return client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':PREFIX+name},ExpiresIn=604800)
        common={'base':'/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
            'job_name':JOB,'bundle_inline':base64.b64encode(bundle).decode(),
            'bundle':{'bytes':len(bundle),'sha256':hashlib.sha256(bundle).hexdigest()},'reuse_sha256':reuse}
        per_rank={}
        for rank in (0,1):
            outputs={}
            if rank==0:
                for relative in ('checkpoints/round_02/trainable.pt','checkpoints/round_02/optimizer.pt'):
                    outputs[relative]={'url':signed(relative),'key':PREFIX+relative}
            per_rank[rank]={'rank':rank,'root':REMOTE+f'/worker_{rank}',
                'status_put':signed(f'status_rank{rank}.json'),'result_put':signed(f'result_rank{rank}.tar.gz'),
                'artifact_puts':outputs}
        source=(HERE/'bootstrap.py').read_text()+'\n'+(HERE/'rank_resolver.py').read_text()
        source+='\nrank=resolve_rank(os.environ)\nguarded_run({**'+repr(common)+',**'+repr(per_rank)+'[rank]})\n'
        packed=base64.b64encode(zlib.compress(source.encode())).decode()
        command='timeout --signal=TERM --kill-after=60s 48540s python3 -u -c '+shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        assert len(command)<64000
        receipt.update(state='submitting',startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        write(intent,receipt)
        assert not (HERE/'CANCELLED_BY_USER').exists(), 'Cancelled by user; do not submit'
        try:
            p=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space',
                '--copy-job-name=b1k-repeat1688-1019-batch-00','--name='+JOB,'--job-name='+JOB,
                '--worker-spec=n12lp.nn.i10a.1','--worker-nodes=2','--quota-type=reserved',
                '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
                '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--container-image-url='+template['roles'][0]['image_path'],'--command='+command],
                capture_output=True,text=True,timeout=90)
            receipt['submit_returncode']=p.returncode
        except subprocess.TimeoutExpired:
            receipt['submit_returncode']='unknown_timeout'
        job=describe(JOB)
        assert job['ownership']['user_name']=='260010165' and job['scheduling']['quota_type']=='RESERVED'
        assert job['roles'][0]['total_replicas']==2
        assert hashlib.sha256(job['roles'][0]['startup_script'].encode()).hexdigest()==receipt['startup_sha256']
        receipt.update(uid=job['uid'],state=job['state']);write(intent,receipt)
        for rank in (0,1):write(HERE/f'r{rank}/submission.json',{**receipt,'rank':rank,'gpu_count':1})
        print(json.dumps({k:receipt[k] for k in ['job','uid','state','quota','gpu_count']}),flush=True)
        return 0
    finally:client.close()


if __name__=='__main__':raise SystemExit(main())
