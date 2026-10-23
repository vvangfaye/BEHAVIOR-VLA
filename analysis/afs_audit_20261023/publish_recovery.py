import sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
p=json.loads((HERE/'storage_plan.json').read_text());c=aoss_client();records=[]
for name in ['storage_plan.json','storage_worker.py','restore_archive.py','RESTORE.md','cleanup_progress.md','final_verification.json','released_summary.json','release_cpu_result.json']:
 raw=(HERE/name).read_bytes();sha=hashlib.sha256(raw).hexdigest();key='behavior1k/afs-cleanup/'+p['archive_id']+'/recovery/'+sha+'/'+name
 c.put_object(Bucket='260010165',Key=key,Body=raw)
 r=c.get_object(Bucket='260010165',Key=key)
 try:assert hashlib.sha256(r['Body'].read()).hexdigest()==sha
 finally:r['Body'].close()
 records.append({'key':key,'sha256':sha,'size':len(raw)})
(HERE/'recovery_objects.json').write_text(json.dumps(records,indent=2)+'\n');print(json.dumps(records));c.close()
