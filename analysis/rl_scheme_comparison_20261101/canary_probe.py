import json,base64,time
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rl_scheme_comparison_20261101/evaluations/original/i301_s1/results')
names=['worker_status/cmp_baseline_r0_a2.json','identities/cmp_baseline_r0_a2.json','baseline/i301_s1/spec.json','baseline/i301_s1/progress.json','baseline/i301_s1/initial.json','baseline/i301_s1/result.json','baseline/i301_s1/initialization.json','baseline/i301_s1/process_failure.json']
r={'at':time.time(),'files':{}}
for name in names:
 p=root/name
 if p.exists():
  d=json.loads(p.read_text())
  if name.endswith('initial.json'):d={k:v for k,v in d.items() if k not in ['physical_state','proprio']}
  r['files'][name]=d
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
