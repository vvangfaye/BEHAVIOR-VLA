import base64,json,time
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_midreward_20261028/pilot01')
out={'observed_at':time.time(),'files':{}}
for folder in [root/'results/checkpoints/round_01',root/'results/training_updates/train_eval_r0_a1',root/'results/final']:
 for name in ('ready.json','result.json','i301_s0/initial.json','i301_s0/result.json'):
  file=folder/name
  if file.is_file():out['files'][str(file)]={'mtime':file.stat().st_mtime}
file=root/'jobs/train_eval_r0_a1/driver.log'
if file.is_file():
 with file.open() as h:
  for line in h:
   try:d=json.loads(line)
   except Exception:continue
   if d.get('stage') in ('training_complete','training_update','critic_features','critic_warmup','loading','evaluation'):
    out.setdefault('training_progress',[]).append({k:d[k] for k in ('stage','started_at','updated_at','accepted_optimizer_steps','candidate','instance','seed') if k in d})
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
