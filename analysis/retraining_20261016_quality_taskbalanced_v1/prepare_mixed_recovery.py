"""Stage exact missing mixed-model batches on AFS; no evaluation is rerun here."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RECOVERY=ROOT/'paired_fixed20_step5000_recovery_g1'
ORIGINAL=ROOT/'paired_fixed20_step5000_5x4'
sys.path.insert(0,str(ROOT))
from run_dual_evaluation import module,read,save

STAGER=r'''
import hashlib,json,os,urllib.request
from pathlib import Path
config=CONFIG
root=Path(config['afs_root']);root.mkdir(parents=True,exist_ok=True)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}));verified=[]
for item in config['sources']:
 path=root/'bootstraps'/(item['id']+'.py');path.parent.mkdir(parents=True,exist_ok=True)
 with opener.open(item['url'],timeout=60) as response:data=response.read(1024*1024)
 assert hashlib.sha256(data).hexdigest()==item['sha256']
 if path.exists():assert path.read_bytes()==data
 else:
  with path.open('xb') as out:out.write(data)
  path.chmod(0o600)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
 verified.append({'id':item['id'],'bytes':len(data),'sha256':item['sha256']})
result={'state':'accepted','sources':verified,'evaluation_executed':False,'log_contents_exported':False}
raw=json.dumps(result).encode()
(root/'_RECOVERY_BOOTSTRAP_READY.json').write_bytes(raw)
with opener.open(urllib.request.Request(config['report_url'],data=raw,method='PUT'),timeout=40):pass
print(json.dumps({'state':'accepted','staged_count':len(verified)}))
'''

def main():
 meta=read(ROOT/'failure_metadata_audit/result.json')
 assert meta['log_contents_exported'] is False
 # No old worker was entered, no seed was acknowledged, no result can be overwritten.
 for row in meta['files'].values():assert row['exists'] is False
 if (RECOVERY/'stage_intent.json').exists():raise RuntimeError('Staging intent already exists; reconcile it')
 for name in ['frozen','private','intents','observations','reports']:(RECOVERY/name).mkdir(parents=True,exist_ok=True)
 import shutil
 for p in (ORIGINAL/'frozen').glob('*.py'):shutil.copy2(p,RECOVERY/'frozen'/p.name)
 shutil.copy2(ORIGINAL/'frozen_sha256.json',RECOVERY/'frozen_sha256.json')
 ctl=module('recovery_original_controller',ORIGINAL/'controller.py');ctl.ROOT=RECOVERY;ctl.base.ROOT=RECOVERY
 original=read(ORIGINAL/'plan.json');plan=dict(original)
 plan.update(afs_root=original['afs_root']+'/recovery-g1',object_prefix=original['object_prefix']+'/recovery-g1',recovery_generation=1)
 plan['batch_groups']=[g for g in original['batch_groups'] if g['id'] in ['batch-03','batch-04']]
 save(RECOVERY/'plan.json',plan)
 c=ctl.aoss_client();sources=[];history=[]
 for group in plan['batch_groups']:
  old=read(ORIGINAL/'intents'/(group['id']+'.json'));native=ctl.observe(old)
  assert native['state']=='FAILED' and native['uid']==old['uid']
  for index in group['indices']:
   assert not (ORIGINAL/'reports'/f'candidate-{index:02d}.json').exists()
  name='b1k-qbal5k-1016-5x4-'+group['id']+'-g1'
  source=ctl.bootstrap(c,plan,group,name).encode();digest=hashlib.sha256(source).hexdigest()
  path=RECOVERY/'private'/(group['id']+'.startup.py');path.write_bytes(source);path.chmod(0o600)
  key=plan['object_prefix']+'/bootstrap-source/'+group['id']+'.py'
  c.put_object(Bucket='260010165',Key=key,Body=source,Metadata={'sha256':digest})
  sources.append({'id':group['id'],'sha256':digest,'url':c.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=86400)})
  history.append({'id':group['id'],'indices':group['indices'],'old_uid':old['uid'],'old_state':'FAILED','generation':1,'attempt':2,'source_sha256':digest,'name':name})
 save(RECOVERY/'recovery_manifest.json',{'reason':'failed before bootstrap file existed on AFS','diagnostic_sha256':hashlib.sha256((ROOT/'failure_metadata_audit/result.json').read_bytes()).hexdigest(),'groups':history,'frozen_worker_code_unchanged':True,'protocol_unchanged':True,'only_missing_samples':True})
 key=plan['object_prefix']+'/staging.json';report_url=c.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=86400)
 source=STAGER.replace('CONFIG',repr({'afs_root':plan['afs_root'],'sources':sources,'report_url':report_url}))
 name='b1k-qbal5k-recovery-stage-1016';intent={'name':name,'uid':None,'report_key':key,'stage_only':True,'created_at':time.time()}
 save(RECOVERY/'stage_intent.json',intent)
 command=shlex.join(['timeout','240',plan['runtime']+'/envs/b1k-eval/bin/python','-c',source])
 args=[ctl.CLI,'acp','jobs','create','--workspace-name=share-space','--name='+name,'--job-name='+name,'--aec2-name=share-cluster','--training-framework=pytorch','--worker-spec=n6ls.iu.i40.8c32g','--worker-nodes=1','--quota-type=spot','--retry-times=0','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--container-image-url='+plan['image'],'--command='+command]
 try:p=subprocess.run(args,env=ctl.environment(),capture_output=True,text=True,timeout=90);intent['cli_exit_code']=p.returncode
 except subprocess.TimeoutExpired:intent['outcome']='unknown'
 save(RECOVERY/'stage_intent.json',intent)
 job=ctl.base.cli_json(['acp','jobs','describe',name,'--workspace-name=share-space','-o','json']);intent.update(uid=job['uid'],state=job['state']);save(RECOVERY/'stage_intent.json',intent);print(json.dumps(intent))

if __name__=='__main__':main()
