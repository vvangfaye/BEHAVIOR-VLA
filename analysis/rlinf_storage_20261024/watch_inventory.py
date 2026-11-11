import json,pathlib,subprocess,sys,time
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
c=aoss_client();deadline=time.monotonic()+1300
try:
 while time.monotonic()<deadline:
  try:
   obj=c.get_object(Bucket='260010165',Key='behavior1k/storage-reclaim-20261024/inventory01.json')
   data=obj['Body'].read();json.loads(data);(HERE/'inventory.json').write_bytes(data);break
  except c.exceptions.NoSuchKey:time.sleep(5)
finally:
 c.close()
 p=subprocess.run(['/home/faye/.local/bin/sco','cci','apps','describe','b1k-storage-cpu-1024','--workspace-name=share-space','-o','json'],capture_output=True,text=True,timeout=45)
 if p.returncode==0:
  app=json.loads(p.stdout)
  assert app['uid']=='131be07700064971ab3eeffeffed2009'
  if app['state']!='SUSPENDED':
   r=subprocess.run(['/home/faye/.local/bin/sco','cci','apps','stop','b1k-storage-cpu-1024','--workspace-name=share-space'],capture_output=True,text=True,timeout=45)
   (HERE/'inventory_stop.json').write_text(json.dumps({'returncode':r.returncode,'uid':app['uid'],'at':time.time()})+'\n')
