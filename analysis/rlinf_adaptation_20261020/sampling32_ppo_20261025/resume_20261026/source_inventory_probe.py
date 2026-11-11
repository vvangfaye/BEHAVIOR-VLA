import base64,json,os
from pathlib import Path
source=Path('/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/pilot01/attempt02/results')
report={'root_exists':source.is_dir(),'rounds':{},'attempts':{},'source_failure_times':{}}
for r in (1,2):
 files=sorted((source/'coordination').glob(f'round_{r:02d}_episode_*.json'))
 report['rounds'][str(r)]=[{'filename':p.name,'mtime':p.stat().st_mtime,**json.loads(p.read_text())} for p in files]
 result=source/'training'/f'round_{r:02d}'/'result.json'
 if result.is_file():
  x=json.loads(result.read_text());report['rounds'][str(r)+'_result']={'optimizer_steps':x['update']['optimizer_steps']}
for p in (source/'training/round_02').glob('episode_*/progress.json'):
 report['attempts'][str(p.parent.name)]=json.loads(p.read_text())
for p in source.glob('status_rank*.json'):
 x=json.loads(p.read_text());report['source_failure_times'][p.name]={k:x[k] for k in ('updated_at','stage','error_type','error') if k in x}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(report).encode()).decode()+'B1KPROBE_END')
