import base64
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import zlib

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

receipt=HERE/'stage01_submission.json'
assert not receipt.exists(),'Resolve the existing staging submission first'
path=HERE/'bootstrap_assets.tar.gz'
with tarfile.open(path,'w:gz') as tar:
    for item in sorted((HERE/'bootstrap_assets').rglob('*')):
        if not item.is_file() or item.name=='download_receipt.jsonl':continue
        body=item.read_bytes();member=tarfile.TarInfo(str(item.relative_to(HERE/'bootstrap_assets')))
        member.size=len(body);member.mtime=0;member.mode=0o644
        tar.addfile(member,io.BytesIO(body))
client=aoss_client()
config={'root':'/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/prepare01','archives':{}}
intent={'job':'b1k-serf-stage-1025-a01','gpus':0,'max_seconds':1200,'root':config['root']}
prefix='behavior1k/serf-reproduction-20261025/stage01/'
try:
    for name,path in [('source',HERE/'runtime51_source.tar.gz'),('assets',HERE/'bootstrap_assets.tar.gz')]:
        digest=hashlib.sha256(path.read_bytes()).hexdigest();key=prefix+name+'-'+digest+'.tar.gz'
        client.upload_file(str(path),'260010165',key,ExtraArgs={'Metadata':{'sha256':digest}})
        h=hashlib.sha256();body=client.get_object(Bucket='260010165',Key=key)['Body']
        for block in iter(lambda:body.read(8*1024**2),b''):h.update(block)
        assert h.hexdigest()==digest
        config['archives'][name]={'size':path.stat().st_size,'sha256':digest,
            'url':client.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=7200)}
    result_key=prefix+'result.json'
    config['result_put']=client.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':result_key},ExpiresIn=7200)
finally:client.close()
intent['result_key']=result_key
source=(HERE/'stage_worker.py').read_text()+'\nrun('+repr(config)+')\n'
packed=base64.b64encode(zlib.compress(source.encode())).decode()
command='timeout --signal=TERM --kill-after=30s 1200s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
intent.update(submission_attempted=True,startup_sha256=hashlib.sha256(command.encode()).hexdigest())
receipt.write_text(json.dumps(intent,indent=2)+'\n')
r=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','copy','--workspace-name=share-space',
    '--copy-job-name=b1k-storage-inv-1024','--name='+intent['job'],'--job-name='+intent['job'],
    '--worker-spec=n6ls.iu.i40.2c8g','--worker-nodes=1','--quota-type=reserved',
    '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--command='+command],capture_output=True,text=True,timeout=90)
intent['returncode']=r.returncode
r=subprocess.run(['/home/faye/.local/bin/sco','acp','jobs','describe',intent['job'],'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True,timeout=45)
d=json.loads(r.stdout);assert d['ownership']['user_name']=='260010165'
intent.update(uid=d['uid'],state=d['state'])
receipt.write_text(json.dumps(intent,indent=2)+'\n');print(json.dumps(intent))
