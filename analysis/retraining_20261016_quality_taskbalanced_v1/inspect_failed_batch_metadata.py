"""Bounded CPU diagnostic: export only fixed metadata and enumerated error flags, never log text."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import CLI,aoss_client,environment,cli_json
FOLDER=ROOT/'failure_metadata_audit';NAME='b1k-qbal5k-failed-metadata-1016';KEY='behavior1k/qbal5k-failure-audit-20261016/metadata.json'
SOURCE=r'''
import hashlib,json,re,time,urllib.request
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/quality_taskbalanced_fixed20_20261016_5x4')
result={'observed_at':time.time(),'log_contents_exported':False,'files':{}}
allowed_errors=['FileExistsError','ModuleNotFoundError','PermissionError','NameError','SyntaxError','SSLError','HTTPError','MemoryError','CUDA out of memory','No such file or directory','ExpiredToken','SignatureDoesNotMatch']
for batch,ids in [('batch-03',[8,14,6,4]),('batch-04',[15,19,11,5])]:
 paths=[root/'launch'/batch/'bootstrap.py',root/'batches'/batch/'status.json']
 for i in ids:
  id=f'candidate-{i:02d}'
  paths += [root/'batches'/batch/(id+'.supervisor.log'),root/'workers'/id/'worker.log',root/'workers'/id/'report.json',root/'episodes'/id/'eval.log',root/'episodes'/id/'policy.log',root/'episodes'/id/'seed_ack.json',root/'supervision'/id/'heartbeat.json']
 for p in paths:
  row={'exists':p.is_file()}
  if p.is_file():
   size=p.stat().st_size;row['bytes']=size
   with p.open('rb') as f:f.seek(max(0,size-65536));text=f.read(65536).decode(errors='replace')
   row['error_flags']={label:label in text for label in allowed_errors}
   if p.name=='bootstrap.py':row['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
   if p.suffix=='.json' and size<=65536:
    try:
     data=json.loads(text)
     if type(data.get('child_returncode')) is int:row['child_returncode']=data['child_returncode']
     if data.get('state') in ['running','accepted','failed','exited']:row['state']=data['state']
    except ValueError:pass
  result['files'][str(p.relative_to(root))]=row
raw=json.dumps(result).encode()
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(urllib.request.Request(REPORT_URL,data=raw,method='PUT'),timeout=40):pass
print(json.dumps({'metadata_uploaded':True,'log_contents_exported':False,'files':len(result['files'])}))
'''
def main():
 FOLDER.mkdir(exist_ok=True);path=FOLDER/'intent.json'
 if path.exists():raise RuntimeError('Prior intent exists: query, do not resubmit')
 c=aoss_client();url=c.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':KEY},ExpiresIn=86400)
 source=SOURCE.replace('REPORT_URL',repr(url));intent={'name':NAME,'key':KEY,'uid':None,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'log_contents_exported':False,'time':time.time()}
 path.write_text(json.dumps(intent,indent=2));path.chmod(0o600)
 image=json.loads((ROOT/'paired_fixed20_step5000_5x4/plan.json').read_text())['image']
 cmd=['timeout','240','/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-eval/bin/python','-c',source]
 args=[CLI,'acp','jobs','create','--workspace-name=share-space','--name='+NAME,'--job-name='+NAME,'--aec2-name=share-cluster','--training-framework=pytorch','--worker-spec=n6ls.iu.i40.8c32g','--worker-nodes=1','--quota-type=spot','--retry-times=0','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--container-image-url='+image,'--command='+shlex.join(cmd)]
 try:p=subprocess.run(args,env=environment(),capture_output=True,text=True,timeout=90);intent['cli_exit_code']=p.returncode
 except subprocess.TimeoutExpired:intent['cli_outcome']='timeout_unknown'
 path.write_text(json.dumps(intent,indent=2));j=cli_json(['acp','jobs','describe',NAME,'--workspace-name=share-space','-o','json']);intent.update(uid=j['uid'],state=j['state']);path.write_text(json.dumps(intent,indent=2));print(json.dumps(intent))
if __name__=='__main__':main()
