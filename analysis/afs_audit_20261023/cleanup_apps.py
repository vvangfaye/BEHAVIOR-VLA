import json,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent;names=[]
for file in ['archive_cpu_submission_attempt2.json','archive_cpu_submission.json','release_cpu_submission.json']:
 r=json.loads((HERE/file).read_text())
 if r.get('returncode')==0:names.append(r['name'])
for name in names:
 assert name.startswith('b1k-afs-')
 p=subprocess.run(['sco','cci','apps','describe',name,'--workspace-name=share-space'],capture_output=True,text=True,check=True,timeout=40)
 x=json.loads(p.stdout);assert x['ownership']['user_name']=='260010165' and x['state']=='SUSPENDED' and x.get('ready_replicas',0)==0
p=subprocess.run(['sco','cci','apps','delete',*names,'--workspace-name=share-space'],capture_output=True,text=True,timeout=90)
(HERE/'temporary_apps_cleanup.json').write_text(json.dumps({'names':names,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr},indent=2)+'\n');print('temporary CPU apps',names,'cleanup_returncode',p.returncode)
if p.returncode:raise SystemExit(p.returncode)
