"""Submit one bounded evaluation shard with fresh owner-wide standard admission."""
import argparse
import base64
from datetime import datetime, timezone
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = HERE.parent
CLI = '/home/faye/.local/bin/sco'


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def describe(name):
    p = subprocess.run([CLI,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],
                       capture_output=True,text=True,check=True,timeout=45)
    return json.loads(p.stdout)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', type=int, choices=(1,), required=True)
    rank = parser.parse_args().rank
    intent = HERE/f'r{rank}/submission.json'
    assert not intent.exists(), 'Existing submission intent must be reconciled, never resubmitted'
    assert not (HERE/'CANCELLED_BY_USER').exists()
    for name in ('preflight_result.json','test_result.json'):
        assert json.loads((HERE/name).read_text())['passed']
    plan = json.loads((HERE/'plan.json').read_text())
    assert plan['no_training'] and plan['gpu_count']==2
    previous = describe(plan['resume_evaluation']['previous_job'])
    assert previous['uid'] == plan['resume_evaluation']['previous_uid']
    assert previous['state'] in ('SUSPENDED','FAILED','SUCCEEDED','STOPPED','TERMINATED')
    check = subprocess.run([CLI,'acp','jobs','get-workers',previous['name'],'--workspace-name=share-space'],
                           capture_output=True,text=True,check=True,timeout=40)
    assert 'Running' not in check.stdout, 'Previous worker has not stopped'
    template = describe('b1k-repeat1688-1019-batch-00')
    assert template['ownership']['user_name']=='260010165'
    assert template['resource_pool']['name']=='computing-cluster-5090-01g'
    assert template['roles'][0]['resource_spec'][0]['name']=='n12lp.nn.i10a.1'
    sys.path.insert(0, str(ROOT/'BEHAVIOR-VLAs/sco'))
    from public_test_admission import query_standard_capacity
    from video_eval_adapter import aoss_client
    capacity = query_standard_capacity(CLI, dict(os.environ))
    write(HERE/f'r{rank}/capacity.json', capacity)
    assert capacity['used_gpus'] < capacity['standard_limit'], 'No free standard quota; do not queue or use Spot automatically'
    job = f'b1k-e17-base-1027-r{rank}'
    prefix = f'behavior1k/rlinf-eval17-20261027/baseline-first-attempt01/r{rank}/'
    remote = plan['remote_root']+f'/worker_{rank}'
    names = ['common.py','training.py','experiment.py','simulator.py','helpers.py','driver.py','eval_utils.py','plan.json']
    names += [str(p.relative_to(HERE)) for p in sorted((HERE/'b1k_rlinf').glob('*.py'))]
    buffer = io.BytesIO()
    hashes = {}
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for name in names:
            data = (HERE/name).read_bytes()
            item = tarfile.TarInfo('run/'+name); item.size=len(data); item.mtime=0
            tar.addfile(item, io.BytesIO(data))
            hashes[name] = hashlib.sha256(data).hexdigest()
    bundle = buffer.getvalue()
    (HERE/f'r{rank}/bundle.tar.gz').write_bytes(bundle)
    before = json.loads((BASE/'cloud_smoke/local_preflight.json').read_text())
    artifacts = json.loads((BASE/'cloud_smoke/artifacts.json').read_text())
    reuse = {'checkpoint/'+k:v['sha256'] for k,v in artifacts['checkpoint'].items()}
    reuse['bundle.tar.gz'] = artifacts['bundle']['sha256']
    reuse.update({k:v for k,v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
    for name in ['rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py','rlinf/models/embodiment/openpi/tasks/rl.py']:
        reuse['upstream/'+name] = hashlib.sha256((BASE/'upstream'/name).read_bytes()).hexdigest()
    receipt = {'job':job,'rank':rank,'state':'preparing','quota':'reserved','gpu_count':1,
               'max_seconds':plan['max_job_seconds'],'source_hashes':hashes,'aoss_prefix':prefix,
               'output':remote+'/results','bundle_bytes':len(bundle),'no_training':True,
               'submitted_at':datetime.now(timezone.utc).isoformat()}
    intent.parent.mkdir(parents=True, exist_ok=True)
    with intent.open('x') as out:
        json.dump(receipt,out,indent=2)
    client = aoss_client()
    try:
        client.put_object(Bucket='260010165',Key=prefix+'bundle.tar.gz',Body=bundle)
        assert client.get_object(Bucket='260010165',Key=prefix+'bundle.tar.gz')['Body'].read()==bundle
        def signed(name):
            return client.generate_presigned_url('put_object', Params={'Bucket':'260010165','Key':prefix+name},ExpiresIn=172800)
        config = {'root':remote,'base':'/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
                  'rank':rank,'job_name':job,'bundle_inline':base64.b64encode(bundle).decode(),
                  'bundle':{'bytes':len(bundle),'sha256':hashlib.sha256(bundle).hexdigest()},
                  'reuse_sha256':reuse,'status_put':signed('status.json'),'result_put':signed('result.tar.gz'),
                  'artifact_puts':{}}
        source = (HERE/'bootstrap.py').read_text()+'\nguarded_run('+repr(config)+')\n'
        packed = base64.b64encode(zlib.compress(source.encode())).decode()
        command = f"timeout --signal=TERM --kill-after=60s {plan['max_job_seconds']-60}s python3 -u -c "+shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        assert len(command)<64000
        receipt.update(state='submitting',startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        write(intent,receipt)
        assert not (HERE/'CANCELLED_BY_USER').exists()
        try:
            p = subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space',
                '--copy-job-name=b1k-repeat1688-1019-batch-00','--name='+job,'--job-name='+job,
                '--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type=reserved',
                '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
                '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--container-image-url='+template['roles'][0]['image_path'],'--command='+command],
                capture_output=True,text=True,timeout=90)
            receipt['submit_returncode']=p.returncode
            if p.returncode:
                receipt['submit_error']=re.sub(r'https?://\S+','[URL omitted]',p.stderr)[-1000:]
        except subprocess.TimeoutExpired:
            receipt['submit_returncode']='unknown_timeout'
        write(intent,receipt)
        actual=describe(job)
        assert actual['ownership']['user_name']=='260010165'
        assert actual['scheduling']['quota_type']=='RESERVED'
        assert actual['roles'][0]['total_replicas']==1
        assert hashlib.sha256(actual['roles'][0]['startup_script'].encode()).hexdigest()==receipt['startup_sha256']
        receipt.update(uid=actual['uid'],state=actual['state'])
        write(intent,receipt)
        print(json.dumps({k:receipt[k] for k in ['job','uid','state','rank','quota','gpu_count','bundle_bytes']}),flush=True)
    finally:
        client.close()


if __name__=='__main__':
    main()
