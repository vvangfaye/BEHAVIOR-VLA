import base64
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
from public_test_admission import query_standard_capacity
cli='/home/faye/.local/bin/sco'
receipt=HERE/'init01_submission.json'
assert not receipt.exists(),'Resolve the existing initialization job before another submission'
stage=json.loads((HERE/'dev01_result.json').read_text())
assert stage.get('probes',{}).get('sim',{}).get('passed') is True
tracker=json.loads((HERE/'cotracker_upload.json').read_text())
client=aoss_client()
prefix='behavior1k/serf-reproduction-20261025/init01/'
config={'stage':stage['stage'],'checks_file':stage['root']+f'/iteration_{stage["iteration"]}.json',
        'site_path':stage['root']+'/site',
        'output':'/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/init01'}
try:
    source=HERE/'runtime51_source.tar.gz';sha=hashlib.sha256(source.read_bytes()).hexdigest();key=prefix+'source-'+sha+'.tar.gz'
    client.upload_file(str(source),'260010165',key,ExtraArgs={'Metadata':{'sha256':sha}})
    assert client.get_object(Bucket='260010165',Key=key)['Body'].read()==source.read_bytes()
    config.update(source_sha256=sha,source_get=client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=21600),
        result_put=client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':prefix+'status.json'},ExpiresIn=21600),
        artifacts_put=client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':prefix+'artifacts.tar.gz'},ExpiresIn=21600),
        cotracker_get=client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':tracker['key']},ExpiresIn=21600),
        cotracker_sha256=tracker['sha256'])
finally:client.close()
code=(HERE/'gpu_init_worker.py').read_text()+'\nrun('+repr(config)+')\n'
encoded=base64.b64encode(zlib.compress(code.encode())).decode()
command='timeout --signal=TERM --kill-after=30s 3550s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))')
capacity=query_standard_capacity(cli,dict(os.environ))
(HERE/'init01_capacity.json').write_text(json.dumps(capacity,indent=2)+'\n')
quota='reserved' if capacity['used_gpus']<capacity['standard_limit'] else 'spot'
intent={'job':'b1k-serf-init-1025-a01','gpus':1,'max_seconds':3550,'quota_type':quota,'root':config['output'],
        'result_key':prefix+'status.json','submission_attempted':True,'source_sha256':sha}
receipt.write_text(json.dumps(intent,indent=2)+'\n')
process=subprocess.run([cli,'acp','jobs','copy','--workspace-name=share-space',
    '--copy-job-name=b1k-repeat1688-1019-batch-00','--name='+intent['job'],'--job-name='+intent['job'],
    '--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type='+quota,
    '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--command='+command],
    capture_output=True,text=True,timeout=90)
intent['returncode']=process.returncode
d=json.loads(subprocess.check_output([cli,'acp','jobs','describe',intent['job'],'--workspace-name=share-space','-o','json'],text=True))
assert d['ownership']['user_name']=='260010165' and d['resource_pool']['name']=='computing-cluster-5090-01g'
intent.update(uid=d['uid'],state=d['state']);receipt.write_text(json.dumps(intent,indent=2)+'\n');print(json.dumps(intent))
