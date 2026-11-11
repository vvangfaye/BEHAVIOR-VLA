import json,pathlib,sys,time
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
c=aoss_client();deadline=time.monotonic()+2100;previous=None
try:
 while time.monotonic()<deadline:
  try:
   raw=c.get_object(Bucket='260010165',Key='behavior1k/storage-reclaim-20261024/cache-clear03/STATUS.json')['Body'].read();x=json.loads(raw)
   (HERE/'cleanup_status.json').write_bytes(raw)
   current=(x.get('state'),x.get('removed_files'))
   if current!=previous:
    print(json.dumps({k:x.get(k) for k in ['state','removed_files','removed_allocated_bytes','error_count']}),flush=True);previous=current
   if x.get('state') in ['complete','partial','failed']:
    if x.get('manifest_key'):
     data=c.get_object(Bucket='260010165',Key=x['manifest_key'])['Body'].read();(HERE/'cache_manifest.json.gz').write_bytes(data)
    break
  except c.exceptions.NoSuchKey:pass
  time.sleep(5)
finally:c.close()
