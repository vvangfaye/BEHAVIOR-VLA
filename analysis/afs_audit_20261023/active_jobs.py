import subprocess,json,concurrent.futures
from pathlib import Path
states=['WAITING','QUEUEING','STARTING','INIT','CREATING','RUNNING','RESTARTING','RECOVERING']
def query(state):
 rows=[]
 for page in range(1,30):
  p=subprocess.run(['sco','acp','jobs','list','--workspace-name=share-space','--user-name=260010165','--state='+state,'--page-size=10','--page-token='+str(page),'-o','json'],capture_output=True,text=True,timeout=40)
  if p.returncode:raise RuntimeError(state+' query failed')
  batch=[] if p.stdout.strip()=='No jobs found' else json.loads(p.stdout);rows.extend(batch)
  if len(batch)<10:return rows
 raise RuntimeError('pagination exceeds bound')
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:all_rows=sum(list(pool.map(query,states)),[])
Path('analysis/afs_audit_20261023/protected_active_jobs.json').write_text(json.dumps(all_rows,indent=2))
print(json.dumps([{'name':j['name'],'state':j['state']} for j in all_rows]))
