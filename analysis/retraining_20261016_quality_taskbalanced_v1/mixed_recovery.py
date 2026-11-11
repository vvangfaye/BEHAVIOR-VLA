"""Resume only proved-unstarted mixed-model samples; preserve accepted results."""
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from run_dual_evaluation import ROOT,module,read,save,native_rows

FOLDER=ROOT/'paired_fixed20_step5000_recovery_g1'
ORIGINAL=ROOT/'paired_fixed20_step5000_5x4'


def fetch(client,key):
 from botocore.exceptions import ClientError
 try:
  response=client.get_object(Bucket='260010165',Key=key)
  try:return json.loads(response['Body'].read(2*1024**2))
  finally:response['Body'].close()
 except ClientError as error:
  if error.response['Error']['Code'] in ('NoSuchKey','404'):return None
  raise


def tick(original_active):
 if not (FOLDER/'stage_intent.json').exists():return 0
 ctl=module('mixed_recovery_controller',ORIGINAL/'controller.py');ctl.ROOT=FOLDER;ctl.base.ROOT=FOLDER
 plan=read(FOLDER/'plan.json');manifest=read(FOLDER/'recovery_manifest.json');client=ctl.aoss_client()
 intents,states=native_rows(ctl,FOLDER)
 active=sum(s['state'] not in ctl.TERMINAL for s in states)
 stage_intent=read(FOLDER/'stage_intent.json')
 staged=fetch(client,stage_intent['report_key'])
 expected={g['id']:g['source_sha256'] for g in manifest['groups']}
 if staged:
  if staged.get('state')!='accepted' or {s['id']:s['sha256'] for s in staged['sources']}!=expected:raise ValueError('Recovery bootstrap readback failed')
  save(FOLDER/'staging_accepted.json',staged)
 else:
  job=ctl.base.cli_json(['acp','jobs','describe',stage_intent['name'],'--workspace-name=share-space','-o','json'])
  if job['uid']!=stage_intent['uid']:raise ValueError('Staging job UID changed')
  if job['state'] in ctl.TERMINAL:raise RuntimeError('Recovery staging terminal without accepted proof')
 if staged and original_active+active<5:
  existing={i['id'] for i in intents}
  for group in manifest['groups']:
   if group['id'] in existing or original_active+active>=5:continue
   old=read(ORIGINAL/'intents'/(group['id']+'.json'));state=ctl.observe(old)
   if state['uid']!=group['old_uid'] or state['state']!='FAILED':raise ValueError('Original failed worker is no longer safely closed')
   for i in group['indices']:
    if (ORIGINAL/'reports'/f'candidate-{i:02d}.json').exists():raise ValueError('Recovery would rerun an existing result')
   # Read the already-verified AFS source: no network bootstrap on the GPU worker.
   script=plan['afs_root']+'/bootstraps/'+group['id']+'.py'
   loader='from pathlib import Path;import hashlib;p=Path('+repr(script)+');b=p.read_bytes();assert hashlib.sha256(b).hexdigest()=='+repr(group['source_sha256'])+';exec(compile(b,str(p),"exec"))'
   command=shlex.join(['timeout','--signal=TERM','--kill-after=60',str(plan['job_walltime_seconds']),plan['runtime']+'/envs/b1k-eval/bin/python','-c',loader])
   intent={'id':group['id'],'name':group['name'],'uid':None,'gpu':1,'image':plan['image'],'indices':group['indices'],'generation':1,'attempts':2,'original_uid':group['old_uid'],'source_sha256':group['source_sha256'],'created_at':time.time(),'creation_attempted':False}
   path=FOLDER/'intents'/(group['id']+'.json');ctl.new(path,intent)
   args=[ctl.CLI,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name='+plan['gpu_copy_template'],'--name='+intent['name'],'--job-name='+intent['name'],'--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1','--quota-type=spot','--retry-times=0','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--container-image-url='+plan['image'],'--env=NVIDIA_DRIVER_CAPABILITIES:all','--command='+command]
   intent['creation_attempted']=True;save(path,intent)
   try:result=subprocess.run(args,env=ctl.environment(),capture_output=True,text=True,timeout=90);intent['cli_exit_code']=result.returncode
   except subprocess.TimeoutExpired:intent['outcome']='unknown'
   save(path,intent);state=ctl.observe(intent);intent.update(uid=state['uid'],state=state['state']);save(path,intent);active+=int(state['state'] not in ctl.TERMINAL)
 # Copy only newly accepted records into empty canonical slots, retaining provenance.
 copied=[];bundle=hashlib.sha256((FOLDER/'frozen_sha256.json').read_bytes()).hexdigest()
 for group in manifest['groups']:
  for index in group['indices']:
   id=f'candidate-{index:02d}';report=fetch(client,plan['object_prefix']+'/reports/'+id+'.json')
   if not report or report.get('state')!='accepted':continue
   if report['id']!=id or report['worker_bundle_sha256']!=bundle or report['protocol_sha256']!=plan['protocol_sha256'] or report['checkpoint_manifest_sha256']!=plan['expected_checkpoint_manifest_sha256']:raise ValueError('Recovery result identity differs')
   if [report['record']['task'],report['record']['instance_id']]!=plan['pairs'][index]:raise ValueError('Recovery result assigned to wrong instance')
   save(FOLDER/'reports'/(id+'.json'),report)
   canonical=ORIGINAL/'reports'/(id+'.json')
   if canonical.exists():
    existing=read(canonical)
    if existing.get('state')=='accepted' and existing.get('record')==report.get('record'):continue
    raise ValueError('Never overwrite an existing canonical result')
   value=dict(report,recovery_provenance={'generation':1,'attempt':2,'old_uid':group['old_uid'],'source_report_key':plan['object_prefix']+'/reports/'+id+'.json'})
   ctl.new(canonical,value);copied.append(id)
 save(FOLDER/'status.json',{'observed_at':time.time(),'staging_accepted':bool(staged),'active_jobs':active,'copied_this_tick':copied})
 return active
