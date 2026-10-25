import base64,json,subprocess
from pathlib import Path
base=Path('/mnt/afs/260010165/behavior_1k/rlinf_interface_fix_20261025/pilot01/attempt02')
report={}
for rank in (0,1):
 root=base/f'r{rank}'
 row={}
 for name in ['runtime_receipt.json','bootstrap_error.json','evaluation_instances_verified.json','results/status.json','results/action_interface.json','results/policy_density_checks.json','results/storage_usage.json']:
  p=root/name
  if p.is_file():row[name]=json.loads(p.read_text())
 for phase in ('baseline','ppo'):
  out=root/'results'/phase/f'i{301+rank}_s0'
  for name in ('initialization.json','progress.json','result.json'):
   p=out/name
   if p.is_file():row[f'{phase}/{name}']=json.loads(p.read_text())
  log=out/'simulator.log'
  if log.exists():
   with log.open('rb') as stream:
    stream.seek(max(0,log.stat().st_size-9000));text=stream.read().decode(errors='replace').replace(chr(0),'');row[f'{phase}/log_tail']=text.splitlines()[-3:];row[f'{phase}/last_stack_head']=text.rsplit('(most recent call first):',1)[-1].splitlines()[:4] if '(most recent call first):' in text else []
 report[str(rank)]=row
report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(report).encode()).decode()+'B1KPROBE_END')
