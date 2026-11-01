import json,base64,time
from pathlib import Path
r={'observed_at':time.time(),'evals':[],'trainers':{}}
root=Path('/mnt/afs/260010165/behavior_1k/rl_scheme_comparison_20261101')
for i in range(301,321):
 d=root/'evaluations/original'/f'i{i}_s1/results/baseline'/f'i{i}_s1'
 v={'instance':i}
 for n in ['progress.json','result.json','process_failure.json','initialization.json']:
  f=d/n
  if f.exists():
   v[n]={'mtime':f.stat().st_mtime,'data':json.loads(f.read_text())}
 r['evals'].append(v)
for name in ['gamma1_ppo','balanced_ppo','group_outcome']:
 d=root/'variants'/name/'results'
 r['trainers'][name]={}
 for sub in ['worker_status','training_updates/train1']:
  for f in (d/sub).glob('*.json'):
   r['trainers'][name][str(f.relative_to(d))]={'mtime':f.stat().st_mtime,'data':json.loads(f.read_text())}
old=Path('/mnt/afs/260010165/behavior_1k/rlinf_multitask_20261029/water/results')
r['original']={str(f.relative_to(old)):json.loads(f.read_text()) for f in (old/'worker_status').glob('*train*.json')}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
