import base64
import json
import pathlib
import re
import subprocess
import time

root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/retry07/gpu_continuation')
result={'observed_at':time.time(),'root_exists':root.exists(),'files':{}}
result['runtime_processes']=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  targets={str(root/'run'/name).encode():name for name in ('driver.py','rollout.py','precision_gate.py','ppo.py')}
  names=[name for arg,name in targets.items() if arg in args]
  if names:
   st=(p/'stat').read_text().split(); info=dict(line.split(':',1) for line in (p/'status').read_text().splitlines() if ':' in line)
   result['runtime_processes'].append({'pid':int(p.name),'program':names[0],'state':st[2],'cpu_ticks':int(st[13])+int(st[14]),'rss':info.get('VmRSS','').strip(),'threads':info.get('Threads','').strip(),'wchan':(p/'wchan').read_text()})
 except (FileNotFoundError,PermissionError):pass
for name in ('launch.json','handoff_receipt.json','takeover_state.json'):
 p=root/name
 if p.exists():result['files'][name]=json.loads(p.read_text())
device=root/'results/jax/device.json'
if device.exists():result['files']['results/jax/device.json']=json.loads(device.read_text())
for name in ('takeover.log','takeover_repaired.log','driver.log'):
 p=root/name
 if p.exists():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-1500));result['files'][name]={'tail':f.read().decode(errors='replace')}
for key in ('memory.current','memory.max'):
 q=pathlib.Path('/sys/fs/cgroup')/key
 if q.is_file():result[key]=q.read_text().strip()
stat=pathlib.Path('/sys/fs/cgroup/memory.stat')
if stat.is_file():
 vals=dict(line.split() for line in stat.read_text().splitlines())
 result['memory_breakdown']={k:int(vals[k]) for k in ('anon','file','shmem','kernel') if k in vals}
try:result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=10).strip()
except Exception as e:result['gpu_error']=type(e).__name__
for rel in ['runtime_receipt.json','prompt_hotfix_receipt.json','bootstrap_error.json','results/status.json','results/jax/301/initialization.json','results/jax/301/inference.json','results/jax/301/failure.json','results/jax/301/progress.json','results/jax/301/result.json','results/jax/302/initialization.json','results/jax/302/inference.json','results/jax/302/failure.json','results/jax/302/progress.json','results/jax/302/result.json','results/torch/301/initialization.json','results/torch/301/inference.json','results/torch/301/failure.json','results/torch/301/progress.json','results/torch/301/result.json','results/torch/302/initialization.json','results/torch/302/inference.json','results/torch/302/failure.json','results/torch/302/progress.json','results/torch/302/result.json','results/precision_gate.json','results/reference_progress.json','results/gate.json','results/ppo/progress.json','results/ppo/result.json']:
 p=root/rel
 if p.is_file():
  try:
   val=json.loads(p.read_text())
   if rel.endswith('failure.json') and 'traceback' in val:val['traceback']=val['traceback'][-1500:]
   if rel.endswith('gate.json'):
    ps=val.get('probes',[])
    val={**{k:val[k] for k in ['passed','checks','thresholds']},'probe_count':len(ps),'max_mean_error':max((p['mean_error'] for p in ps),default=None),'max_action_error':max((p['max_error'] for p in ps),default=None)}
   if rel.endswith('ppo/result.json'):val={k:v for k,v in val.items() if k not in ['metrics','max_parameter_changes']}
   result['files'][rel]=val
  except Exception as e:result['files'][rel]={'read_error':str(e)}
for rel in ['results/reference_inputs.log','results/precision_gate.log','results/baseline.log','results/jax/301/simulator.log','results/jax/302/simulator.log','results/torch/301/simulator.log','results/torch/302/simulator.log','results/adapted.log','results/ppo.log']:
 p=root/rel
 if p.is_file():
  with p.open('rb') as f:
   f.seek(max(0,p.stat().st_size-600));s=f.read().decode(errors='replace')
  result['files'][rel]={'size':p.stat().st_size,'tail':re.sub(r'https?://[^\s\"\']+','[URL omitted]',s)}
result['completed_pair_initial_errors'] = {}
for instance in (301, 302):
 a = root / 'results/jax' / str(instance) / 'initial.json'
 b = root / 'results/torch' / str(instance) / 'initial.json'
 if a.is_file() and b.is_file():
  left = json.loads(a.read_text())['physical_state']
  right = json.loads(b.read_text())['physical_state']
  if left.keys() == right.keys():
   result['completed_pair_initial_errors'][str(instance)] = max(abs(x-y) for k in left for u,v in zip(left[k],right[k]) for x,y in zip(u,v))
result['saved_probes'] = len(list((root / 'results/jax').glob('*/probe_*.npz')))
import zipfile, ast, struct
result['latest_probe_subtasks']={}
for instance in (301,302):
 paths=sorted((root/'results/jax'/str(instance)).glob('probe_*.npz'))
 if paths:
  with zipfile.ZipFile(paths[-1]) as z:
   b=z.read('subtask.npy'); nsize=2 if b[6]==1 else 4
   n=int.from_bytes(b[8:8+nsize],'little'); header=ast.literal_eval(b[8+nsize:8+nsize+n].decode('latin1'))
   if header['descr'].startswith('<U'):
    result['latest_probe_subtasks'][str(instance)]={'probe':paths[-1].name,'subtask':b[8+nsize+n:].decode('utf-32-le').rstrip('\x00')}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KPROBE_END',flush=True)
