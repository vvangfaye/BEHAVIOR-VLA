import base64
import json
import pathlib
import re
import subprocess
import time

root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/retry04')
result={'observed_at':time.time(),'root_exists':root.exists(),'files':{}}
try:result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True,timeout=10).strip()
except Exception as e:result['gpu_error']=type(e).__name__
for rel in ['runtime_receipt.json','prompt_hotfix_receipt.json','bootstrap_error.json','results/status.json','results/jax/301/initialization.json','results/jax/301/inference.json','results/jax/301/failure.json','results/jax/301/progress.json','results/jax/301/result.json','results/jax/302/initialization.json','results/jax/302/inference.json','results/jax/302/failure.json','results/jax/302/progress.json','results/jax/302/result.json','results/torch/301/initialization.json','results/torch/301/inference.json','results/torch/301/failure.json','results/torch/301/progress.json','results/torch/301/result.json','results/torch/302/initialization.json','results/torch/302/inference.json','results/torch/302/failure.json','results/torch/302/progress.json','results/torch/302/result.json','results/gate.json','results/ppo/progress.json','results/ppo/result.json']:
 p=root/rel
 if p.is_file():
  try:
   val=json.loads(p.read_text())
   if rel.endswith('failure.json') and 'traceback' in val:val['traceback']=val['traceback'][-1500:]
   if rel.endswith('gate.json'):val={k:val[k] for k in ['passed','checks','thresholds']}
   if rel.endswith('ppo/result.json'):val={k:v for k,v in val.items() if k not in ['metrics','max_parameter_changes']}
   result['files'][rel]=val
  except Exception as e:result['files'][rel]={'read_error':str(e)}
for rel in ['results/baseline.log','results/jax/301/simulator.log','results/jax/302/simulator.log','results/torch/301/simulator.log','results/torch/302/simulator.log','results/adapted.log','results/ppo.log']:
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
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KPROBE_END',flush=True)
