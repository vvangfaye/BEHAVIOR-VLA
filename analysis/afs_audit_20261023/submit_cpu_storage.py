"""Submit a bounded CPU-only CCI phase using the existing user's verified resource pool."""
import argparse,hashlib,json,os,shlex,subprocess,sys,time
from pathlib import Path
import boto3
from botocore.config import Config
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
ap=argparse.ArgumentParser();ap.add_argument('--action',choices=['archive','release'],required=True);args=ap.parse_args()
plan=json.loads((HERE/'storage_plan.json').read_text());phase=args.action+'-cpu-'+str(int(time.time()));name='b1k-afs-'+args.action+'-1023-'+str(int(time.time()))[-5:]
receipt_path=HERE/(args.action+'_cpu_submission.json');assert not receipt_path.exists(),'Submission intent exists; inspect exact app instead of duplicating'
c=aoss_client();credentials=c._request_signer._credentials.get_frozen_credentials();internal=boto3.client('s3',endpoint_url='https://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn',region_name='cn-sh-01b',aws_access_key_id=credentials.access_key,aws_secret_access_key=credentials.secret_key,aws_session_token=credentials.token,config=Config(signature_version='s3v4',s3={'addressing_style':'path'},proxies={}))
root='behavior1k/afs-cleanup/'+plan['archive_id']+'/';config={**plan,'action':args.action,'max_seconds':13800,'archive_workers':4,'allow_mount_rebind':True}
if args.action=='release':
 retained=[];prior=[]
 for u in config['units']:
  try:
   response=c.get_object(Bucket='260010165',Key=u['prefix']+'/RELEASED.json')
   try:r=json.loads(response['Body'].read())
   finally:response['Body'].close()
  except c.exceptions.NoSuchKey:retained.append(u);continue
  assert r['source']==u['source'] and r['archive_id']==plan['archive_id'] and r['released_files']==len(u['files'])
  prior.append(r)
 config['units']=retained;config['prior_releases']=prior;config['all_source_roots']={u['id']:u['source'] for u in plan['units']}
def signed(method,key):return internal.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=21600)
for unit in config['units']:
 for f in unit['files']:
  f['put_url']=signed('put_object',unit['prefix']+'/files/'+f['path']);f['get_url']=signed('get_object',f.get('existing_object_key',unit['prefix']+'/files/'+f['path']));f['get_url_public']=c.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':f.get('existing_object_key',unit['prefix']+'/files/'+f['path'])},ExpiresIn=21600)
 for label,object_name in [('manifest','MANIFEST.json'),('receipt','VERIFIED.json'),('complete','COMPLETE.json'),('release','RELEASED.json'),('error','ERROR.json')]:
  unit[label+'_put']=signed('put_object',unit['prefix']+'/'+object_name);unit[label+'_get']=signed('get_object',unit['prefix']+'/'+object_name)
result_key=root+phase+'/RESULT.json';config['result_put']=signed('put_object',result_key)
code=(HERE/'storage_worker.py').read_text()+'''\ndef execute(config):\n import subprocess\n final_path=pathlib.Path(config['evidence_root'])/(config['action']+'_FINAL.json')\n if final_path.exists():\n  raw=final_path.read_bytes();request(config['result_put'],'PUT',payload=raw);return\n def allocated():\n  return {u['id']:int(subprocess.check_output(['du','-s','-B1',u['source']],text=True).split()[0]) for u in config['units']}\n try:\n  before=allocated() if config['action']=='release' else None\n  done_name='RELEASED.json' if config['action']=='release' else 'COMPLETE.json'\n  done=[pathlib.Path(config['evidence_root'])/u['id']/done_name for u in config['units']]\n  if all(p.exists() for p in done):\n   result={'action':config['action'],'archive_id':config['archive_id'],'units':[json.loads(p.read_text()) for p in done],'already_complete':True}\n  else:result=run_all(config)\n  if before is not None:\n   result['prior_releases']=config.get('prior_releases',[])\n   after=allocated();result['directory_usage_before']=before;result['directory_usage_after']=after;result['directory_bytes_reduced']=sum(before.values())-sum(after.values())\n   result['remaining_files']={name:[str(p.relative_to(source)) for p in pathlib.Path(source).rglob('*') if p.is_file()] for name,source in config.get('all_source_roots',{u['id']:u['source'] for u in config['units']}).items()}\n   result['all_released_allocated_bytes']=sum(u['unlinked_allocated_bytes'] for u in result['units'])+sum(u['unlinked_allocated_bytes'] for u in result.get('prior_releases',[]))\n   probe=pathlib.Path(config['evidence_root'])/('.write_probe_'+str(time.time_ns()))\n   with probe.open('xb') as stream:stream.write(b'0'*4096);stream.flush();os.fsync(stream.fileno())\n   probe.unlink();result['write_probe_succeeded']=True\n except Exception as error:\n  result={'error_type':type(error).__name__,'error':str(error) if 'http' not in str(error) else 'HTTP transport error; URL omitted'}\n raw=encode(result);save(final_path,result);request(config['result_put'],'PUT',payload=raw);print(json.dumps(result),flush=True)\n'''
code_bytes=code.encode();code_key=root+'execution/'+hashlib.sha256(code_bytes).hexdigest()+'.py';config_key=root+'private/'+phase+'.json'
c.put_object(Bucket='260010165',Key=code_key,Body=code_bytes);c.put_object(Bucket='260010165',Key=config_key,Body=json.dumps(config).encode())
bootstrap='import json,urllib.request;ns={};exec(compile(urllib.request.urlopen('+repr(signed('get_object',code_key))+').read(),"storage_phase","exec"),ns);ns["execute"](json.load(urllib.request.urlopen('+repr(signed('get_object',config_key))+')))'
command='timeout --signal=TERM --kill-after=30s 14400s python3 -u -c '+shlex.quote(bootstrap)+'; rc=$?; if [ -x /mnt/afs/share/stop-cci ]; then /mnt/afs/share/stop-cci --workspace share-space; fi; exit "$rc"'
original=json.loads((HERE/'cpu_app_description.txt').read_text());container=original['template']['containers'][0]
app={'display_name':name,'resource_pool':{k:original['resource_pool'][k] for k in ['name','available_zone','vpc_id']},'replicas':1,'template':{'resource_spec':{'name':original['template']['resource_spec']['name']},'containers':[{'name':'storage','image_path':container['image_path'],'command':['bash','-lc',command],'resource_request':{'cpu':'2','memory':'8GiB','nvidia.com/mig-3g.40gb':'0'},'volume_mounts':[m for m in container['volume_mounts'] if m['type']=='PV_AFS']}]},'scheduling':{'priority':'NORMAL','quota_type':'RESERVED'}}
private_path=Path('/tmp')/(name+'.json');fd=os.open(private_path,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
with os.fdopen(fd,'w') as stream:json.dump(app,stream)
receipt={'name':name,'action':args.action,'archive_id':plan['archive_id'],'created_at':time.time(),'result_key':result_key,'private_config_key':config_key,'code_key':code_key,'cpu':2,'memory_gib':8,'gpus':0,'max_seconds':14400,'submitted':False}
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
receipt['submitted']=True;receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
try:
 result=subprocess.run(['sco','cci','apps','create',name,'--workspace-name=share-space','--config='+str(private_path)],capture_output=True,text=True,timeout=90)
 receipt['returncode']=result.returncode;receipt['stdout']=result.stdout[:4000];receipt['stderr']=result.stderr[:2000]
finally:private_path.unlink();internal.close();c.close()
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['name','action','returncode','cpu','memory_gib','gpus']}))
