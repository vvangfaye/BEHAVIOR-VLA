"""Recheck all owned active jobs and exact cloud completion before releasing chosen units."""
import argparse,base64,hashlib,json,re,subprocess,sys,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
ap=argparse.ArgumentParser();ap.add_argument('--units',nargs='+',required=True);ap.add_argument('--prepare-only',action='store_true');args=ap.parse_args()
plan=json.loads((HERE/'storage_plan.json').read_text());units=[u for u in plan['units'] if u['id'] in args.units];assert len(units)==len(args.units)
subprocess.run([sys.executable,str(HERE/'active_jobs.py')],check=True,stdout=subprocess.DEVNULL)
jobs=json.loads((HERE/'protected_active_jobs.json').read_text());texts=[]
for job in jobs:
 text=json.dumps(job)
 for role in job.get('roles',[]):
  for data in re.findall(r'[A-Za-z0-9+/=]{100,}',role.get('startup_script','')):
   try:text+='\n'+zlib.decompress(base64.b64decode(data)).decode()
   except Exception:pass
 texts.append(text)
if not args.prepare_only:
 assert any(j['name']==plan['worker']['name'] and j['uid']==plan['worker']['uid'] and j['state']=='RUNNING' for j in jobs),'Worker identity changed'
c=aoss_client();proof=[]
def get(prefix,name):
 r=c.get_object(Bucket='260010165',Key=prefix+'/'+name)
 try:return r['Body'].read()
 finally:r['Body'].close()
for unit in units:
 assert not any(unit['source'] in t for t in texts),'Active job references source; release stopped'
 cb=get(unit['prefix'],'COMPLETE.json');complete=json.loads(cb);raw=get(unit['prefix'],'MANIFEST.json');receipt=get(unit['prefix'],'VERIFIED.json')
 assert hashlib.sha256(raw).hexdigest()==complete['manifest_sha256'] and hashlib.sha256(receipt).hexdigest()==complete['receipt_sha256']
 m=json.loads(raw);assert m['source']==unit['source'] and m['archive_id']==plan['archive_id']
 assert {(f['path'],f['size']) for f in m['files']}=={(f['path'],f['size']) for f in unit['files']}
 proof.append({'unit':unit['id'],'source':unit['source'],'manifest_sha256':complete['manifest_sha256'],'active_job_count':len(jobs)})
c.close();(HERE/('release_guard_'+'_'.join(args.units)+'.json')).write_text(json.dumps(proof,indent=2)+'\n')
print(json.dumps({'release_guard_passed':args.units,'active_jobs_checked':len(jobs)}),flush=True)
if not args.prepare_only:subprocess.run([sys.executable,str(HERE/'run_storage.py'),'--action','release','--units',*args.units],check=True)
