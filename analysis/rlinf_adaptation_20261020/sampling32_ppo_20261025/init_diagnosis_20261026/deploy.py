"""Submit one bounded, non-training initialization diagnostic."""
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
PARENT = HERE.parent
ROOT = PARENT.parents[2]
CLI = '/home/faye/.local/bin/sco'
NAME = 'b1k-s32-init-1026-probe01'
PREFIX = 'behavior1k/rlinf-sampling32-20261025/init-diagnosis-20261026/pilot01/'
REMOTE = '/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/init_diagnosis_20261026/pilot01'


def describe(name):
    p = subprocess.run([CLI,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],
                       capture_output=True,text=True,check=True,timeout=40)
    return json.loads(p.stdout)


def main():
    target = HERE/'submission.json'
    assert not target.exists(), 'Never submit the same diagnostic twice'
    for rank in (0,1):
        old = json.loads((PARENT/'attempt02'/f'r{rank}/submission.json').read_text())
        job = describe(old['job'])
        assert job['uid'] == old['uid'] and job['ownership']['user_name'] == '260010165'
        assert job['state'] == 'FAILED'
        workers = subprocess.run([CLI,'acp','jobs','get-workers',old['job'],'--workspace-name=share-space'],
                                 capture_output=True,text=True,check=True,timeout=40).stdout
        assert 'Failed' in workers and 'Running' not in workers
    template = describe('b1k-repeat1688-1019-batch-00')
    assert template['ownership']['user_name'] == '260010165'
    assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
    sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
    from public_test_admission import query_standard_capacity
    from video_eval_adapter import aoss_client
    buffer = io.BytesIO(); hashes = {}
    with tarfile.open(fileobj=buffer,mode='w:gz') as t:
        for p in sorted(HERE.rglob('*.py')) + [HERE/'plan.json']:
            if p.name == 'deploy.py': continue
            name = str(p.relative_to(HERE)); raw = p.read_bytes()
            hashes[name] = hashlib.sha256(raw).hexdigest()
            item = tarfile.TarInfo('run/'+name); item.size=len(raw); item.mtime=0
            t.addfile(item,io.BytesIO(raw))
    bundle = buffer.getvalue(); assert len(bundle)<200000
    (HERE/'bundle.tar.gz').write_bytes(bundle)
    old_artifacts = json.loads((PARENT.parent/'cloud_smoke/artifacts.json').read_text())
    reuse = {'checkpoint/'+k:v['sha256'] for k,v in old_artifacts['checkpoint'].items()}
    reuse['bundle.tar.gz'] = old_artifacts['bundle']['sha256']
    before = json.loads((PARENT.parent/'cloud_smoke/local_preflight.json').read_text())
    reuse.update({k:v for k,v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
    for name in ['rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py','rlinf/models/embodiment/openpi/tasks/rl.py']:
        reuse['upstream/'+name] = hashlib.sha256((PARENT.parent/'upstream'/name).read_bytes()).hexdigest()
    capacity = query_standard_capacity(CLI,dict(os.environ))
    (HERE/'capacity.json').write_text(json.dumps(capacity,indent=2)+'\n')
    quota = 'reserved' if capacity['used_gpus']<capacity['standard_limit'] else 'spot'
    receipt = {'job':NAME,'state':'preparing','quota':quota,'gpu_count':1,'max_seconds':3600,
               'source_hashes':hashes,'bundle_bytes':len(bundle),'aoss_prefix':PREFIX,
               'output':REMOTE+'/results','submitted_at':datetime.now(timezone.utc).isoformat()}
    target.write_text(json.dumps(receipt,indent=2)+'\n')
    client = aoss_client()
    try:
        client.put_object(Bucket='260010165',Key=PREFIX+'bundle.tar.gz',Body=bundle)
        assert client.get_object(Bucket='260010165',Key=PREFIX+'bundle.tar.gz')['Body'].read()==bundle
        def signed(name):
            return client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':PREFIX+name},ExpiresIn=86400)
        config={'root':REMOTE+'/worker_0','base':'/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
                'rank':0,'job_name':NAME,'bundle_inline':base64.b64encode(bundle).decode(),
                'bundle':{'bytes':len(bundle),'sha256':hashlib.sha256(bundle).hexdigest()},
                'reuse_sha256':reuse,'status_put':signed('status.json'),
                'result_put':signed('report.tar.gz'),'artifact_puts':{}}
        source=(PARENT/'bootstrap.py').read_text()+'\nguarded_run('+repr(config)+')\n'
        packed=base64.b64encode(zlib.compress(source.encode())).decode()
        command='timeout --signal=TERM --kill-after=60s 3540s python3 -u -c '+shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        assert len(command)<64000
        receipt.update(state='submitting',startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        target.write_text(json.dumps(receipt,indent=2)+'\n')
        try:
            r=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space',
                '--copy-job-name=b1k-repeat1688-1019-batch-00','--name='+NAME,'--job-name='+NAME,
                '--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type='+quota,
                '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
                '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--container-image-url='+template['roles'][0]['image_path'],'--command='+command],
                capture_output=True,text=True,timeout=90)
            receipt['submit_returncode']=r.returncode
        except subprocess.TimeoutExpired:
            receipt['submit_returncode']='unknown_timeout'
        actual=describe(NAME)
        assert actual['ownership']['user_name']=='260010165'
        assert hashlib.sha256(actual['roles'][0]['startup_script'].encode()).hexdigest()==receipt['startup_sha256']
        receipt.update(uid=actual['uid'],state=actual['state'])
        target.write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:receipt[k] for k in ['job','uid','state','quota','bundle_bytes']}))
    finally:
        client.close()


if __name__ == '__main__':main()
