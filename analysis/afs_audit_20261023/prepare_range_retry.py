import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
plan_path=HERE/'storage_plan.json';p=json.loads(plan_path.read_text());stop=json.loads((HERE/'worker_current.json').read_text());assert stop['state']=='FAILED' and stop['uid']==p['worker']['uid']
c=aoss_client();root='behavior1k/afs-cleanup/'+p['archive_id']+'/';objects=[];token=None
while True:
 kw=dict(Bucket='260010165',Prefix=root,MaxKeys=1000)
 if token:kw['ContinuationToken']=token
 r=c.list_objects_v2(**kw);objects.extend(r.get('Contents',[]))
 if not r.get('IsTruncated'):break
 token=r['NextContinuationToken']
lookup={o['Key']:o['Size'] for o in objects};reused=0;reused_bytes=0
plan_path.with_name('storage_plan_before_retries.json').write_text(plan_path.read_text())
for u in p['units']:
 if u['prefix']+'/COMPLETE.json' in lookup:continue
 for f in u['files']:
  k=u['prefix']+'/files/'+f['path']
  if not f.get('existing_object_key') and k in lookup and lookup[k]==f['size']:
   f['existing_object_key']=k;reused+=1;reused_bytes+=f['size']
 u['id']+='r5';u['prefix']+='r5'
p['units'].sort(key=lambda u:(0 if u['id'].startswith('installer_cache') else 1 if u['id'].startswith('baseline_aoss_copy') else 2,u['id']))
p['dependencies']['executor_sha256']=hashlib.sha256((HERE/'storage_worker.py').read_bytes()).hexdigest();p['dependencies']['readback_method']='Complete ordered SHA256 from bounded 8 MiB HTTPS ranges, each checked against exact Content-Range, length, TLS origin and non-cache response.'
plan_path.write_text(json.dumps(p,indent=2)+'\n');(HERE/'range_retry_reuse.json').write_text(json.dumps({'files_reused':reused,'bytes_reused':reused_bytes},indent=2));print('Reusing',reused,'objects',reused_bytes/2**30,'GiB');c.close()
