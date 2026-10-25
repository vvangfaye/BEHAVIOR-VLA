import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

job='b1k-serf-pre-1025-a01'
receipt=HERE/'preflight_submission.json'
assert not receipt.exists(), 'Resolve existing job before resubmitting'
client=aoss_client()
key='behavior1k/serf-reproduction-20261025/preflight01.json'
try:
    url=client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=3600)
finally:client.close()
source=(HERE/'preflight_worker.py').read_text()+'\nrun('+repr({'put_url':url})+')\n'
packed=base64.b64encode(zlib.compress(source.encode())).decode()
command='timeout --signal=TERM --kill-after=20s 600s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
intent={'job':job,'key':key,'gpus':0,'cpu':2,'memory_gib':8,'max_seconds':600,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'submission_attempted':True}
receipt.write_text(json.dumps(intent,indent=2)+'\n')
argv=['/home/faye/.local/bin/sco','acp','jobs','copy','--workspace-name=share-space',
      '--copy-job-name=b1k-storage-inv-1024','--name='+job,'--job-name='+job,
      '--worker-spec=n6ls.iu.i40.2c8g','--worker-nodes=1','--quota-type=reserved',
      '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
      '--command='+command]
try:
    reply=subprocess.run(argv,capture_output=True,text=True,timeout=90)
    intent['returncode']=reply.returncode
except subprocess.TimeoutExpired:intent['submit_error']='timeout_unknown_outcome'
reply=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','describe',job,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=45)
data=json.loads(reply.stdout)
assert data['ownership']['user_name']=='260010165'
intent.update(uid=data['uid'],state=data['state'])
receipt.write_text(json.dumps(intent,indent=2)+'\n')
print(json.dumps(intent))
