import json,base64,time
from pathlib import Path
r={'at':time.time(),'learners':{}}
base=Path('/mnt/afs/260010165/behavior_1k')
for variant in ['control','gamma1_ppo','balanced_ppo','group_outcome']:
 output=(base/'rlinf_multitask_20261029/water/results' if variant=='control' else base/'rl_scheme_comparison_20261101/variants'/variant/'results')
 item={}
 for name in ['status.json','resume/latest.json','checkpoints/round_01/ready.json']:
  f=output/name
  if f.exists():
   d=json.loads(f.read_text())
   if name=='resume/latest.json':d={k:d[k] for k in ['saved_at','relative_path','state']};d['state']={k:v for k,v in d['state'].items() if k!='steps'}
   item[name]=d
 f=base/'rl_multigpu_20261101/jobs'/('t-'+variant.replace('_','-')+'-a3')/'driver.log'
 if f.exists():
  with f.open('rb') as h:h.seek(max(0,f.stat().st_size-2*1024**2));body=h.read().decode(errors='replace')
  checkpoints=[]
  for line in body.splitlines():
   if not line.startswith('{'):continue
   try:d=json.loads(line)
   except ValueError:continue
   if d.get('stage')=='update_checkpoint_saved':checkpoints.append({k:d.get(k) for k in ['updated_at','accepted_optimizer_steps','candidate','candidate_kl']})
  item['committed_history']=checkpoints
 r['learners'][variant]=item
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
