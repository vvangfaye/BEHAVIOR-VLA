import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis/rl_scheme_comparison_20261101'))
from deployment import CLI,ENV,describe,aoss_client,save

def main():
    attempt=int(sys.argv[1]) if len(sys.argv)>1 else 1
    assert attempt in (1,2,3)
    receipt=HERE/('communication_probe_submission.json' if attempt==1 else f'communication_probe_submission_a{attempt}.json')
    assert not receipt.exists(),'Reconcile existing probe; do not duplicate'
    template=describe('b1k-repeat1688-1019-batch-00')
    assert template['resource_pool']['name']=='computing-cluster-5090-01g'
    client=aoss_client()
    try:
        key=f'behavior1k/rl-multigpu-20261101/communication_probe/a{attempt}/result.json' if attempt>1 else 'behavior1k/rl-multigpu-20261101/communication_probe/result.json'
        url=client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=86400)
        code=HERE.joinpath('communication_probe.py' if attempt<3 else 'communication_probe_gloo.py').read_text()
        runner="""import base64,json,os,pathlib,subprocess,tempfile,urllib.request
root=pathlib.Path(tempfile.mkdtemp(prefix='b1k-comm-'))
script=root/'probe.py';script.write_bytes(base64.b64decode(PAYLOAD))
env=dict(os.environ);env['OMP_NUM_THREADS']='1';env['PYTHONNOUSERSITE']='1';env['GLOO_SOCKET_IFNAME']='lo'
for key in ['MASTER_ADDR','MASTER_PORT','RANK','WORLD_SIZE','LOCAL_RANK','LOCAL_WORLD_SIZE','TORCHELASTIC_RUN_ID']:env.pop(key,None)
runtime='/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python'
p=subprocess.run([runtime,'-m','torch.distributed.run','--rdzv_backend=static','--master_addr=127.0.0.1','--master_port=29578','--nnodes=1','--node_rank=0','--nproc_per_node=4',str(script)],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=180)
result={'passed':False,'returncode':p.returncode,'log_tail':p.stdout[-12000:]}
for line in p.stdout.splitlines():
 if line.startswith('B1K_COMM_RESULT '):result=json.loads(line.split(' ',1)[1])
request=urllib.request.Request(RESULT_URL,data=json.dumps(result).encode(),method='PUT')
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=60) as r:assert r.status in (200,201,204)
print(json.dumps(result),flush=True)
raise SystemExit(0 if result['passed'] else 1)
""".replace('PAYLOAD',repr(base64.b64encode(code.encode()).decode())).replace('RESULT_URL',repr(url))
        encoded=base64.b64encode(zlib.compress(runner.encode())).decode()
        command='timeout --signal=TERM --kill-after=60s 300s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))')
        name=f'b1k-4gpu-comm-1101-a{attempt}'
        row={'job':name,'created_at':time.time(),'gpu_count':4,'quota':'spot','max_seconds':300,'result_key':key,
             'source_sha256':hashlib.sha256(code.encode()).hexdigest(),'startup_sha256':hashlib.sha256(command.encode()).hexdigest()}
        with receipt.open('x') as f:json.dump(row,f,indent=2)
        p=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=b1k-repeat1688-1019-batch-00',
            '--name='+name,'--job-name='+name,'--worker-spec=n12lp.nn.i10a.4','--worker-nodes=1','--quota-type=spot',
            '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--command='+command],
            env=ENV,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90)
        row['submit_returncode']=p.returncode;save(receipt,row)
        job=describe(name);assert job['scheduling']['quota_type']=='SPOT'
        assert job['roles'][0]['resource_spec'][0]['name']=='n12lp.nn.i10a.4'
        assert hashlib.sha256(job['roles'][0]['startup_script'].encode()).hexdigest()==row['startup_sha256']
        row.update(uid=job['uid'],state=job['state']);save(receipt,row);print(json.dumps(row))
    finally:client.close()

if __name__=='__main__':main()
