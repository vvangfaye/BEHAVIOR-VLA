import base64
import hashlib
import json
import shlex
import subprocess
import sys
import zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
JOB='b1k-rlinf-forensic-1023-01'
PREFIX='behavior1k/rlinf-adapter-20261023/forensic-01/'
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client


def main():
    previous=HERE.parent/'cloud_smoke'
    artifacts=json.loads((previous/'artifacts.json').read_text())
    hashes={'checkpoint/'+name:entry['sha256'] for name,entry in artifacts['checkpoint'].items()}
    hashes['bundle.tar.gz']=artifacts['bundle']['sha256']
    manifest=json.loads((previous/'local_preflight.json').read_text())
    hashes.update({k:v for k,v in manifest['execution_source_sha256'].items() if k.startswith('adapter/')})
    for rel in ['rlinf/algorithms/advantages.py','rlinf/algorithms/losses.py','rlinf/models/embodiment/openpi/tasks/rl.py']:
        hashes['upstream/'+rel]=hashlib.sha256((HERE.parent/'upstream'/rel).read_bytes()).hexdigest()
    plan={'job_name':JOB,'cpu':8,'memory_gib':32,'gpu':0,'max_seconds':900,'read_only':True,'hashes':hashes}
    (HERE/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (HERE/'submission_intent.json').open('x') as f:json.dump({'job_name':JOB,'submission_attempted':False},f)
    template='pt-o8xl7f79'
    r=subprocess.run(['sco','acp','jobs','describe',template,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=40)
    x=json.loads(r.stdout)
    assert x['ownership']['user_name']=='260010165' and x['resource_pool']['name']=='share-cluster'
    client=aoss_client()
    try:
        put=client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':PREFIX+'report.json'},ExpiresIn=86400)
        config={'base':'/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01','retry':'/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/retry01','hashes':hashes,'put':put}
        code=(HERE/'worker_probe.py').read_text()+'\nmain('+repr(config)+')\n'
        encoded=base64.b64encode(zlib.compress(code.encode())).decode()
        command='timeout --signal=TERM --kill-after=30s 870s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))')
        state={'job_name':JOB,'submission_attempted':True,'command_sha256':hashlib.sha256(command.encode()).hexdigest()}
        (HERE/'submission_intent.json').write_text(json.dumps(state,indent=2)+'\n')
        r=subprocess.run(['sco','acp','jobs','copy','--workspace-name=share-space','--copy-job-name='+template,'--name='+JOB,'--job-name='+JOB,'--worker-spec=n6ls.iu.i40.8c32g','--worker-nodes=1','--quota-type=spot','--retry-times=0','--enable-fault-tolerance=false','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--command='+command],capture_output=True,text=True,timeout=100)
        state['submit_returncode']=r.returncode
        r=subprocess.run(['sco','acp','jobs','describe',JOB,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,timeout=40)
        if r.returncode==0:
            x=json.loads(r.stdout);state.update(uid=x['uid'],state=x['state'])
        else:state['state']='needs_exact_name_query'
        (HERE/'submission_intent.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps(state))
    finally:client.close()
if __name__=='__main__':main()
