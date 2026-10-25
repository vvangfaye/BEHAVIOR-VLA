import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zlib

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
receipt=HERE/'stage02_submission.json'
assert not receipt.exists()
cli='/home/faye/.local/bin/sco'
old=json.loads(subprocess.check_output([cli,'acp','jobs','describe','b1k-serf-stage-1025-a01','--workspace-name=share-space','-o','json'],text=True))
assert old['uid']=='fbf5f62c9a844f98a0e678a9af64ac73' and old['state'] in {'SUSPENDED','SUCCEEDED','FAILED','TERMINATED','STOPPED','CANCELED'}
client=aoss_client()
prefix='behavior1k/serf-reproduction-20261025/stage02/'
config={'root':'/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/prepare02','archives':{}}
try:
    path=HERE/'runtime51_source.tar.gz';digest=hashlib.sha256(path.read_bytes()).hexdigest();key=prefix+'source-'+digest+'.tar.gz'
    client.upload_file(str(path),'260010165',key,ExtraArgs={'Metadata':{'sha256':digest}})
    assert client.get_object(Bucket='260010165',Key=key)['Body'].read()==path.read_bytes()
    config['archives']['source']={'size':path.stat().st_size,'sha256':digest,'url':client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=7200)}
    path=HERE/'bootstrap_assets.tar.gz';digest=hashlib.sha256(path.read_bytes()).hexdigest()
    key='behavior1k/serf-reproduction-20261025/stage01/assets-'+digest+'.tar.gz'
    assert client.head_object(Bucket='260010165',Key=key)['ContentLength']==path.stat().st_size
    config['archives']['assets']={'size':path.stat().st_size,'sha256':digest,'url':client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=7200)}
    result_key=prefix+'result.json'
    config['result_put']=client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':result_key},ExpiresIn=7200)
finally:client.close()
source=(HERE/'stage_worker.py').read_text()+'\nrun('+repr(config)+')\n'
packed=base64.b64encode(zlib.compress(source.encode())).decode()
command='timeout --signal=TERM --kill-after=30s 900s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
intent={'job':'b1k-serf-stage-1025-a02','gpus':0,'max_seconds':900,'root':config['root'],'result_key':result_key,'submission_attempted':True}
receipt.write_text(json.dumps(intent,indent=2)+'\n')
process=subprocess.run([cli,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=b1k-storage-inv-1024',
    '--name='+intent['job'],'--job-name='+intent['job'],'--worker-spec=n6ls.iu.i40.2c8g','--worker-nodes=1','--quota-type=reserved',
    '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--command='+command],capture_output=True,text=True,timeout=90)
intent['returncode']=process.returncode
d=json.loads(subprocess.check_output([cli,'acp','jobs','describe',intent['job'],'--workspace-name=share-space','-o','json'],text=True))
intent.update(uid=d['uid'],state=d['state']);receipt.write_text(json.dumps(intent,indent=2)+'\n');print(json.dumps(intent))
