import pathlib,json,sys,time,zlib,base64,re,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
PROJECT=ROOT.parents[3]
sys.path.insert(0,str(PROJECT/'analysis/serf_multitask_20261029'))
from cluster_helpers import submit,aoss_client,write
files=json.loads((ROOT/'cold_inventory.json').read_text())['files']
assert len(files)>0 and sum(x['allocated'] for x in files)>50*1024**3
roots=sorted({x['path'].split('/behavior_1k/')[0]+'/behavior_1k/'+x['path'].split('/behavior_1k/')[1].split('/')[0] for x in files})
# Check every active owner job, including queued work, without saving its command.
cli='/home/faye/.local/bin/sco';references=[];checked=[]
for state in ['WAITING','INIT','QUEUEING','CREATING','STARTING','RESTARTING','RECOVERING','RUNNING','SUSPENDING','DELETING']:
 for page in range(1,201):
  p=subprocess.run([cli,'acp','jobs','list','--workspace-name=share-space','--user-name=260010165','--state='+state,'--page-size=10','--page-token='+str(page),'-o','json'],capture_output=True,text=True,check=True,timeout=40)
  batch=[] if p.stdout.strip()=='No jobs found' else json.loads(p.stdout)
  for job in batch:
   texts=[r.get('startup_script','') for r in job.get('roles',[])]
   for s in list(texts):
    for match in re.finditer(r"b64decode\((?:'|\")([A-Za-z0-9+/=]+)(?:'|\")\)",s):
     try:texts.append(zlib.decompress(base64.b64decode(match.group(1))).decode())
     except Exception:pass
   refs=[r for r in roots if any(r in s for s in texts)]
   checked.append({'job':job['name'],'uid':job['uid'],'state':job['state']})
   if refs:references.append({'job':job['name'],'roots':refs})
  if len(batch)<10:break
 else:raise RuntimeError('Incomplete owner inventory')
write(ROOT/'archive_dependency_audit.json',{'observed_at':time.time(),'checked':checked,'references':references,'roots':roots})
assert not references,'Active jobs reference archive candidates'
client=aoss_client()
try:
 prefix='behavior1k/storage-reclaim-20261030/rl-inactive-a2/'
 def url(method,key):return client.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=86400)
 prepared=[]
 for i,x in enumerate(sorted(files,key=lambda x:-x['allocated'])):
  key=prefix+f'files/{i:04d}/'+pathlib.Path(x['path']).name
  y={**x,'key':key,'put':url('put_object',key),'get':url('get_object',key)}
  for typ in ['manifest','complete']:
   k=prefix+f'files/{i:04d}/'+typ+'.json'
   y[typ+'_put']=url('put_object',k);y[typ+'_get']=url('get_object',k)
  prepared.append(y)
 public={'prefix':prefix,'files':[{k:v for k,v in x.items() if not k.endswith(('_put','_get')) and k not in ['put','get']} for x in prepared], 'allowed_roots':roots}
 write(ROOT/'archive_plan.json',public)
 helpers=(PROJECT/'analysis/rlinf_storage_20261024/archive_helpers.py').read_text()
 config={'output':'/mnt/afs/260010165/behavior_1k/operational_cleanup/20261030_rl_capacity_a2',
         'result_key':prefix+'status.json','status_put':url('put_object',prefix+'status.json'),
         'helpers_source':helpers,'files':prepared,'allowed_roots':roots,'deadline_seconds':7000}
 receipt=ROOT/'archive_submission_a2.json'
 assert not receipt.exists(),'Do not duplicate a submitted archive job'
 submit('b1k-rl-archive-1030-a2',ROOT/'archive_worker.py',config,receipt,7200,False)
finally:client.close()
