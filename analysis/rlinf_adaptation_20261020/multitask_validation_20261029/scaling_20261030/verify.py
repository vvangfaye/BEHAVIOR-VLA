from pathlib import Path
import sys,json,time
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT.parents[1]/'serf_multitask_20261029'))
from cluster_helpers import describe
receipts={}
for slug in ['trash','water']:
 for p in (ROOT/slug/'jobs').glob('*/submission.json'):
  r=json.loads(p.read_text());key=(slug,r['phase'],r['rank'])
  if key not in receipts or receipts[key]['attempt']<r['attempt']:receipts[key]=r
sample=[r for (s,p,k),r in receipts.items() if s=='trash' and p=='sample1']
if len(sample)!=8 or not all(r.get('uid') for r in sample):
 print(json.dumps({'ready':False,'sample_shards':len(sample),'resolved_uids':sum(bool(r.get('uid')) for r in sample)}));raise SystemExit(0)
terminal={'SUCCEEDED','FAILED','SUSPENDED','STOPPED','TERMINATED','CANCELED'}
with ThreadPoolExecutor(max_workers=6) as pool:actuals=list(pool.map(lambda r:describe(r['job']),receipts.values()))
rows=[]
for ((slug,phase,rank),r),a in zip(receipts.items(),actuals):
 assert r['uid']==a['uid'] and a['ownership']['user_name']=='260010165'
 rows.append({'task':slug,'phase':phase,'rank':rank,'job':r['job'],'uid':a['uid'],'state':a['state'],'quota':a['scheduling']['quota_type']})
active=[r for r in rows if r['state'] not in terminal]
assert len(active)<=16
assert sum(r['task']=='trash' for r in active)<=8
assert sum(r['task']=='water' for r in active)<=8
report={'ready':True,'observed_at':time.time(),'active_count':len(active),'active':active,'all_latest':rows,
        'total_limit':16,'per_task_limit':8,'no_duplicate_sample_ranks':len({r['rank'] for r in sample})==8}
(ROOT/'scaling_20261030/verified_jobs.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'ready':True,'active_count':len(active),'active':active},ensure_ascii=False))
