"""One bounded read-only GPU audit; no training or simulator episodes."""
import base64
from datetime import datetime,timezone
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

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
CLI=str(Path.home()/'.local/bin/sco')
JOB='b1k-ppo-ctrl-1025-a01'
PREFIX='behavior1k/ppo-degradation-control-20261025/a01/'
REMOTE='/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/control_20261025a'
IMAGE='registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634'

def write(name,data):(HERE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def describe(name):return json.loads(subprocess.run([CLI,'acp','jobs','describe',name,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=45).stdout)

def main():
    assert not (HERE/'submission.json').exists()
    plan=json.loads((HERE/'plan.json').read_text())
    assert plan['gpu_count']==1 and plan['diagnostic_control']['no_training'] and plan['diagnostic_control']['max_episodes']==2
    sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
    from public_test_admission import query_standard_capacity
    from video_eval_adapter import aoss_client
    capacity=query_standard_capacity(CLI,dict(os.environ));write('capacity.json',capacity)
    quota='reserved' if capacity['used_gpus']<capacity['standard_limit'] else 'spot'
    plan['quota_type']=quota;write('plan.json',plan)
    files=['common.py','training.py','driver.py','experiment.py','helpers.py','simulator.py','plan.json']+[str(p.relative_to(HERE)) for p in sorted((HERE/'b1k_rlinf').glob('*.py'))]
    buffer=io.BytesIO();hashes={}
    with tarfile.open(fileobj=buffer,mode='w:gz') as archive:
        for name in files:
            raw=(HERE/name).read_bytes();item=tarfile.TarInfo('run/'+name);item.size=len(raw);item.mtime=0
            archive.addfile(item,io.BytesIO(raw));hashes[name]=hashlib.sha256(raw).hexdigest()
    bundle=buffer.getvalue();(HERE/'bundle.tar.gz').write_bytes(bundle)
    previous=json.loads((HERE.parent/'cloud_smoke/artifacts.json').read_text())
    reuse={'checkpoint/'+k:v['sha256'] for k,v in previous['checkpoint'].items()};reuse['bundle.tar.gz']=previous['bundle']['sha256']
    before=json.loads((HERE.parent/'cloud_smoke/local_preflight.json').read_text())
    reuse.update({k:v for k,v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
    for rel in ['rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py','rlinf/models/embodiment/openpi/tasks/rl.py']:
        reuse['upstream/'+rel]=hashlib.sha256((HERE.parent/'upstream'/rel).read_bytes()).hexdigest()
    receipt={'job':JOB,'state':'preparing','gpu_count':1,'max_seconds':7200,'quota':quota,'source_hashes':hashes,
             'bundle_bytes':len(bundle),'bundle_sha256':hashlib.sha256(bundle).hexdigest(),'submitted_at':datetime.now(timezone.utc).isoformat()}
    write('submission.json',receipt)
    client=aoss_client()
    try:
        key=PREFIX+'bundle.tar.gz';client.put_object(Bucket='260010165',Key=key,Body=bundle)
        assert client.get_object(Bucket='260010165',Key=key)['Body'].read()==bundle
        def signed(key):return client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':PREFIX+key},ExpiresIn=86400)
        config={'root':REMOTE,'base':'/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01','rank':0,'job_name':JOB,
                'bundle_inline':base64.b64encode(bundle).decode(),'bundle':{'bytes':len(bundle),'sha256':receipt['bundle_sha256']},
                'reuse_sha256':reuse,'status_put':signed('status.json'),'result_put':signed('audit.json'),'artifact_puts':{}}
        source=(HERE/'bootstrap.py').read_text()+'\nguarded_run('+repr(config)+')\n'
        packed=base64.b64encode(zlib.compress(source.encode())).decode()
        cmd='timeout --signal=TERM --kill-after=30s 7170s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        receipt['startup_sha256']=hashlib.sha256(cmd.encode()).hexdigest();receipt['state']='submitting';write('submission.json',receipt)
        reply=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=b1k-repeat1688-1019-batch-00',
            '--name='+JOB,'--job-name='+JOB,'--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type='+quota,
            '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--container-image-url='+IMAGE,'--command='+cmd],
            capture_output=True,text=True,timeout=90)
        receipt['submit_returncode']=reply.returncode
        actual=describe(JOB)
        assert actual['ownership']['user_name']=='260010165' and actual['scheduling']['quota_type']==quota.upper()
        assert actual['roles'][0]['total_replicas']==1
        assert hashlib.sha256(actual['roles'][0]['startup_script'].encode()).hexdigest()==receipt['startup_sha256']
        receipt.update(uid=actual['uid'],state=actual['state']);write('submission.json',receipt)
        print(json.dumps({k:receipt[k] for k in ['job','uid','state','quota','bundle_bytes']}))
    finally:client.close()

if __name__=='__main__':main()
