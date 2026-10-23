"""Finish archive -> fresh dependency check -> release -> stop, with bounded CPU resources."""
import json,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
active_name=None
phase_errors=[]
plan=json.loads((HERE/'storage_plan.json').read_text());c=aoss_client();status=HERE/'cleanup_controller_status.json';started=time.time()
def report(stage,**extra):
 value={'stage':stage,'time':time.time(),'elapsed_seconds':time.time()-started,**extra};status.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value),flush=True)
def cli(*args):
 p=subprocess.run(['sco',*args],capture_output=True,text=True,timeout=60)
 if p.returncode:raise RuntimeError('CLI failed: '+' '.join(args[:3]))
 return p.stdout
def object_json(key):
 try:r=c.get_object(Bucket='260010165',Key=key)
 except c.exceptions.NoSuchKey:return None
 try:return json.loads(r['Body'].read())
 finally:r['Body'].close()
def states():
 global phase_errors
 r=c.list_objects_v2(Bucket='260010165',Prefix='behavior1k/afs-cleanup/'+plan['archive_id']+'/',MaxKeys=1000);assert not r.get('IsTruncated')
 keys={x['Key'] for x in r.get('Contents',[])}
 phase_errors=[object_json(u['prefix']+'/ERROR.json') for u in plan['units'] if u['prefix']+'/ERROR.json' in keys]
 return [u['id'] for u in plan['units'] if u['prefix']+'/COMPLETE.json' in keys],[u['id'] for u in plan['units'] if u['prefix']+'/RELEASED.json' in keys]
def stop_and_confirm(name):
 data=json.loads(cli('cci','apps','describe',name,'--workspace-name=share-space'))
 if data['state']!='SUSPENDED':cli('cci','apps','stop',name,'--workspace-name=share-space')
 deadline=time.time()+600
 while time.time()<deadline:
  data=json.loads(cli('cci','apps','describe',name,'--workspace-name=share-space'))
  if data['state']=='SUSPENDED' and data.get('ready_replicas',0)==0:return
  time.sleep(10)
 raise RuntimeError('CPU app did not stop within deadline')
try:
 receipt=json.loads((HERE/'archive_cpu_submission.json').read_text());active_name=receipt['name'];deadline=time.time()+5*3600
 while time.time()<deadline:
  complete,released=states();result=object_json(receipt['result_key']);report('archiving',complete_units=complete,total_units=len(plan['units']))
  if phase_errors:raise RuntimeError('Archive unit error: '+json.dumps(phase_errors))
  if len(complete)==len(plan['units']):break
  if result and 'error_type' in result:raise RuntimeError('Archive phase failed: '+result['error_type']+' '+result['error'])
  time.sleep(25)
 else:raise RuntimeError('Archive deadline exceeded')
 report('stopping_archive_cpu');stop_and_confirm(receipt['name'])
 report('fresh_dependency_check')
 subprocess.run([sys.executable,str(HERE/'release_guard.py'),'--prepare-only','--units',*[u['id'] for u in plan['units']]],check=True)
 report('submitting_release_cpu')
 subprocess.run([sys.executable,str(HERE/'submit_cpu_storage.py'),'--action','release'],check=True)
 release=json.loads((HERE/'release_cpu_submission.json').read_text());active_name=release['name'];assert release['returncode']==0,'Release app creation failed'
 while time.time()<deadline:
  complete,released=states();result=object_json(release['result_key']);report('releasing',released_units=released,total_units=len(plan['units']))
  if result and 'error_type' in result:raise RuntimeError('Release phase failed: '+result['error_type']+' '+result['error'])
  if len(released)==len(plan['units']) and result:
   (HERE/'release_cpu_result.json').write_text(json.dumps(result,indent=2)+'\n');break
  time.sleep(25)
 else:raise RuntimeError('Release deadline exceeded')
 report('stopping_release_cpu');stop_and_confirm(release['name'])
 report('complete',released_units=released,directory_bytes_reduced=result.get('directory_bytes_reduced'))
except Exception as error:
 if active_name:
  report('stopping_failed_phase',app=active_name,error=str(error))
  try:stop_and_confirm(active_name)
  except Exception:pass
 report('needs_attention',error_type=type(error).__name__,error=str(error));raise
finally:c.close()
