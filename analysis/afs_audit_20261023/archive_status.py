import sys,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
plan=json.loads((HERE/'storage_plan.json').read_text());c=aoss_client();prefix='behavior1k/afs-cleanup/'+plan['archive_id']+'/'
rows=[];token=None
while True:
 kw={'Bucket':'260010165','Prefix':prefix,'MaxKeys':1000}
 if token:kw['ContinuationToken']=token
 r=c.list_objects_v2(**kw);rows.extend(r.get('Contents',[]))
 if not r.get('IsTruncated'):break
 token=r['NextContinuationToken']
keys={o['Key']:o for o in rows};units=[]
for u in plan['units']:
 files=[o for o in rows if o['Key'].startswith(u['prefix']+'/files/')]
 units.append({'id':u['id'],'objects_uploaded':len(files),'bytes_uploaded':sum(o['Size'] for o in files),'complete':u['prefix']+'/COMPLETE.json' in keys,'released':u['prefix']+'/RELEASED.json' in keys})
result={'time':time.time(),'units':units};(HERE/'archive_status.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));c.close()
