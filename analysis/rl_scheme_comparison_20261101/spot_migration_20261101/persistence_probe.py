import json,base64,time
from pathlib import Path
base=Path('/mnt/afs/260010165/behavior_1k')
roots={'gamma1_ppo':base/'rl_scheme_comparison_20261101/variants/gamma1_ppo/results','balanced_ppo':base/'rl_scheme_comparison_20261101/variants/balanced_ppo/results','group_outcome':base/'rl_scheme_comparison_20261101/variants/group_outcome/results','original_ppo':base/'rlinf_multitask_20261029/water/results'}
r={'at':time.time(),'roots':{}}
for k,p in roots.items():
 d={'checkpoint_files':[{'path':str(x.relative_to(p)),'bytes':x.stat().st_size} for x in (p/'checkpoints').rglob('*') if x.is_file()],'updates':{}}
 for x in (p/'training_updates').rglob('result.json'):
  y=json.loads(x.read_text());d['updates'][str(x.relative_to(p))]={a:y.get(a) for a in ['optimizer_steps','attempted_optimizer_steps','passed']}
 if (p/'failure.json').exists():d['failure']=json.loads((p/'failure.json').read_text())
 if k=='original_ppo':
  d['episodes']=sum(x.is_file() for x in (p/'training_receipts/round_01').glob('episode_*.json'))
 r['roots'][k]=d
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
