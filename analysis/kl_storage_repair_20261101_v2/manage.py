"""Submit one CPU archive worker and maintain durable restore indices."""
import base64,hashlib,importlib.util,json,os
from pathlib import Path
import re,subprocess,sys,time,zlib
HERE=Path(__file__).absolute().parent
PROJECT=HERE.parents[1]
sys.path.insert(0,str(PROJECT/'analysis/rl_multigpu_20261101'))
import deployment as d
sys.path.insert(0,str(PROJECT/'analysis/rlinf_adaptation_20261020/multitask_validation_20261029/capacity_repair_20261030'))
from internal_storage import internal_aoss_client

STATES=['WAITING','INIT','QUEUEING','CREATING','STARTING','RESTARTING','RECOVERING','RUNNING','SUSPENDING','DELETING']

def audit():
 jobs=[]
 for state in STATES:
  seen=set()
  for page in range(1,101):
   p=subprocess.run([d.CLI,'acp','jobs','list','--workspace-name=share-space','--user-name=260010165','--state='+state,'--page-size=10','--page-token='+str(page),'-o','json'],env=d.ENV,capture_output=True,text=True,check=True,timeout=45)
   batch=[] if p.stdout.strip()=='No jobs found' else json.loads(p.stdout)
   for j in batch:
    assert j['uid'] not in seen;seen.add(j['uid'])
    scripts=[r.get('startup_script','') for r in j.get('roles',[])]
    for s in list(scripts):
     for encoded in re.findall(r'b64decode\([\'"]([A-Za-z0-9+/=]+)[\'"]\)',s):
      try:scripts.append(zlib.decompress(base64.b64decode(encoded)).decode())
      except Exception:pass
    paths=sorted(set(p for s in scripts for p in re.findall(r'/mnt/afs/260010165/[A-Za-z0-9_./-]+',s)))
    jobs.append({'name':j['name'],'uid':j['uid'],'state':j['state'],'paths':paths})
   if len(batch)<10:break
  else:raise RuntimeError('Incomplete job dependency inventory')
 return {'at':time.time(),'jobs':jobs}

def references(root,observed):
 return [j['name'] for j in observed['jobs'] if any(p==root or p.startswith(root+'/') or (root.startswith(p+'/') and p!='/mnt/afs/260010165/behavior_1k') for p in j['paths'])]

def make_plan():
 base='/mnt/afs/260010165/behavior_1k/'
 previous=json.loads((PROJECT/'analysis/rl_multigpu_20261101/code_snapshot_index.json').read_text())
 roots=[base+'update_composition_20261022/trial_h100/meta/'+x for x in ['r02','r03','r05','r06','renderer','trainer']]
 # Completed RL comparisons; model checkpoints and any expert-data directories are excluded by the worker.
 roots += [base+x for x in ['rlinf_conservative_20261024','rlinf_multiround_20261023','rlinf_modem_20261028','rlinf_midreward_20261028','rlinf_sampling32_20261025','rlinf_eval17_20261027','rlinf_interface_fix_20261025']]
 roots += [base+'serf_reproduction_20261025/'+x for x in ['dev01','eval01','eval02','eval03','eval04','eval05','init01','init02','legacy01','prepare01','prepare02','recovery01']]
 roots += [base+x for x in ['confirmation_selected_vs_baseline_20261017','heldout20_baseline_expert1k_20261017','huber50_heldout20_20261017','quality_taskbalanced_fixed20_20261016_5x4','paired_quick20_reused1688977_step10000_20261014','paired_quick20_reused1688977_step5000_20261014','paired_quick20_reused1688977_step20000_20261015','repeatability_1688977_seed0_30_20261019']]
 roots += [base+'code_snapshots/'+x['name'] for x in previous if x['name']!='b1k-video1688977-20261013']
 protected=[base+x for x in ['rlinf_adapter_20261020','rlinf_adapter_20261023','runtime_20261007_1306','rlinf_multitask_20261029','rl_scheme_comparison_20261101','rl_multigpu_20261101','raw_full100_20261007_1308','checkpoints','code_snapshots/b1k-video1688977-20261013']]
 for name in ['control','gamma1_ppo','balanced_ppo','group_outcome']:
  p=json.loads((PROJECT/'analysis/rl_multigpu_20261101/jobs'/('t-'+name.replace('_','-')+'-a3')/'worker_plan.json').read_text())
  def collect(v):
   if isinstance(v,str) and v.startswith(base):protected.append(v)
   elif isinstance(v,list):
    for x in v:collect(x)
   elif isinstance(v,dict):
    for x in v.values():collect(x)
  collect(p)
 observed=audit();d.save(HERE/'dependency_audit.json',observed)
 selected=[];held=[]
 for root in roots:
  refs=references(root,observed)
  conflict=any(root==p or root.startswith(p+'/') or p.startswith(root+'/') for p in protected)
  if refs or conflict:held.append({'root':root,'references':refs,'protected':conflict})
  else:selected.append(root)
 assert len(selected)==len(set(selected))
 plan={'created_at':time.time(),'bucket':'260010165','prefix':'behavior1k/storage-reclaim-20261101/kl-unblock-v2/',
  'roots':selected,'protected':sorted(set(protected)),'held':held,'target_free_bytes':70*1024**3,
  'deadline_seconds':4*3600,'quota':'reserved','gpus':0,'cpus':2,
  'protected_policy':'Keep all checkpoint directories, weights, expert videos/datasets, current rollouts, active and scheduled dependencies. Archive only explicit inactive experiment roots.',
  'source_deletion':'Only exact manifest files after full origin tar and member-hash verification, durable local+AOSS recovery index, and fresh dependency audit.'}
 d.save(HERE/'plan.json',plan);return plan

def submit():
 plan=make_plan();receipt=HERE/'submission.json';assert not receipt.exists()
 client=internal_aoss_client();public=d.aoss_client()
 try:
  prefix=plan['prefix'];existing=public.list_objects_v2(Bucket='260010165',Prefix=prefix,MaxKeys=1)
  assert not existing.get('Contents'),'Archive version already exists; reconcile instead of overwriting'
  def url(method,key):return client.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=86400)
  units={}
  for i in range(len(plan['roots'])):
   for part in range(16):
    unit=f'{i:03d}-{part:03d}';up=prefix+'units/'+unit+'/'
    record={'archive_key':up+'archive.tar.gz'}
    for typ,file in [('archive','archive.tar.gz'),('manifest','manifest.json'),('verification','verification.json'),('complete','COMPLETE.json'),('released','RELEASED.json')]:
     record[typ+'_put']=url('put_object',up+file);record[typ+'_get']=url('get_object',up+file)
    record['release_get']=url('get_object',up+'release.json');units[unit]=record
  config={**plan,'scratch':'/tmp/b1k-cold-archive-1101-v2','status_put':url('put_object',prefix+'status.json'),
   'stop_get':url('get_object',prefix+'STOP.json'),'units':units}
  source=(HERE/'archive_worker.py').read_text()
  payload=json.dumps({'source':source,'config':config}).encode();key=prefix+'worker_payload.json'
  public.put_object(Bucket='260010165',Key=key,Body=payload)
  assert public.get_object(Bucket='260010165',Key=key)['Body'].read()==payload
  wrapper="""import hashlib,json,urllib.request
request=urllib.request.Request(URL,headers={'Cache-Control':'no-cache'})
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request,timeout=120) as response:body=response.read(32*1024**2)
assert hashlib.sha256(body).hexdigest()==SHA
payload=json.loads(body);ns={};exec(compile(payload['source'],'archive_worker','exec'),ns);ns['run'](payload['config'])
""".replace('SHA',repr(hashlib.sha256(payload).hexdigest())).replace('URL',repr(url('get_object',key)))
  compile(wrapper,'archive-bootstrap','exec')
  packed=base64.b64encode(zlib.compress(wrapper.encode())).decode()
  command='timeout --signal=TERM --kill-after=60s 14400s python3 -u -c '+__import__('shlex').quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
  template=d.describe('b1k-storage-inv-1024');assert template['resource_pool']['name']=='share-cluster'
  row={'job':'b1k-kl-storage-1101-v2','gpu_count':0,'cpus':2,'max_seconds':14400,'state':'submitting','prefix':prefix,'worker_sha256':hashlib.sha256(source.encode()).hexdigest(),'payload_sha256':hashlib.sha256(payload).hexdigest(),'startup_sha256':hashlib.sha256(command.encode()).hexdigest(),'at':time.time()}
  d.save(receipt,row)
  p=subprocess.run([d.CLI,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=b1k-storage-inv-1024','--name='+row['job'],'--job-name='+row['job'],'--worker-spec=n6ls.iu.i40.2c8g','--worker-nodes=1','--quota-type=reserved','--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--command='+command],env=d.ENV,capture_output=True,text=True,timeout=90)
  j=d.describe(row['job']);assert j['ownership']['user_name']=='260010165'
  row.update(uid=j['uid'],state=j['state'],returncode=p.returncode);d.save(receipt,row);print(json.dumps(row))
 finally:client.close();public.close()


def supervise():
 plan=json.loads((HERE/'plan.json').read_text());row=json.loads((HERE/'submission.json').read_text());client=d.aoss_client();last_quota=0;last_release=-1
 try:
  while time.time()<row['at']+plan['deadline_seconds']+900:
   j=d.describe(row['job']);assert j['uid']==row['uid']
   try:state=json.loads(client.get_object(Bucket=plan['bucket'],Key=plan['prefix']+'status.json')['Body'].read())
   except client.exceptions.ClientError as e:
    if e.response['Error']['Code'] not in ('404','NoSuchKey'):raise
    state={}
   d.save(HERE/'status.json',{'job_state':j['state'],**state})
   if state.get('stage')=='awaiting_recovery_index':
    unit=state['unit'];up=plan['prefix']+'units/'+unit+'/'
    complete_body=client.get_object(Bucket=plan['bucket'],Key=up+'COMPLETE.json')['Body'].read();complete=json.loads(complete_body)
    manifest_body=client.get_object(Bucket=plan['bucket'],Key=up+'manifest.json')['Body'].read();verification_body=client.get_object(Bucket=plan['bucket'],Key=up+'verification.json')['Body'].read()
    assert hashlib.sha256(manifest_body).hexdigest()==complete['manifest_sha256']
    assert hashlib.sha256(verification_body).hexdigest()==complete['verification_sha256']
    manifest=json.loads(manifest_body);assert json.loads(verification_body)['full_origin_readback']
    observed=audit();assert not references(manifest['root'],observed),'New active dependency; preserve source'
    index={'archive':complete,'manifest':manifest,'verification':json.loads(verification_body),'bucket':plan['bucket'],'prefix':up,'dependency_audit':observed,'restore':'GET archive.tar.gz from this prefix, verify its SHA256, extract to a NEW directory, verify all files using manifest before reuse.'}
    directory=HERE/'recovery_indices';directory.mkdir(exist_ok=True)
    path=directory/(unit+'.json')
    if path.exists():
     index=json.loads(path.read_text());assert index['archive']==complete and index['manifest']==manifest
    else:d.save(path,index)
    saved=path.read_bytes();assert json.loads(saved)==index
    key=up+'RECOVERY_INDEX.json'
    try:existing=client.get_object(Bucket=plan['bucket'],Key=key)['Body'].read()
    except client.exceptions.ClientError as e:
     if e.response['Error']['Code'] not in ('404','NoSuchKey'):raise
     client.put_object(Bucket=plan['bucket'],Key=key,Body=saved);existing=client.get_object(Bucket=plan['bucket'],Key=key)['Body'].read()
    assert existing==saved
    ack={'unit':unit,'manifest_sha256':complete['manifest_sha256'],'external_index_verified':True,'external_index_path':str(path),'external_index_sha256':hashlib.sha256(saved).hexdigest(),'aoss_index':key}
    client.put_object(Bucket=plan['bucket'],Key=up+'release.json',Body=json.dumps(ack).encode())
   if state.get('released_bytes',0)!=last_release or time.time()-last_quota>180:
    quota=d.single.query_quota();last_quota=time.time();last_release=state.get('released_bytes',0)
    free=quota['rows'][0]['quota_bytes']-quota['rows'][0]['used_bytes'];d.save(HERE/'quota.json',quota)
    if free>=plan['target_free_bytes']:
     client.put_object(Bucket=plan['bucket'],Key=plan['prefix']+'STOP.json',Body=json.dumps({'reason':'Target free quota attained','free_bytes':free,'at':time.time()}).encode())
   if j['state'] in ['SUCCEEDED','FAILED','SUSPENDED','STOPPED','CANCELED']:return
   time.sleep(5)
 finally:client.close()

if __name__=='__main__':
 if sys.argv[1]=='submit':submit()
 else:
  import fcntl
  with (HERE/'supervise.lock').open('a') as lock:
   fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
   until=time.time()+5*3600
   while time.time()<until:
    try:supervise();break
    except AssertionError:raise
    except Exception as e:
     d.save(HERE/'supervise_error.json',{'at':time.time(),'type':type(e).__name__})
     time.sleep(20)
