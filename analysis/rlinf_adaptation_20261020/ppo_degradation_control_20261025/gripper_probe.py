import json,pathlib,base64,subprocess
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/control_20261025a/results')
r={}
for rel in ['status.json','failure.json','baseline/i301_s0/progress.json','baseline/i301_s0/result.json','baseline/i301_s0/initialization.json','final/i301_s0/progress.json','final/i301_s0/result.json','final/i301_s0/initialization.json']:
 q=p/rel
 if q.exists():r[rel]=json.loads(q.read_text())
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
import subprocess
code = """import numpy as np,json
p='/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/diagnostic_20261025a/results/actions.npz'
d=np.load(p,allow_pickle=False);r={}
for dim in [14,22]:
 r[str(dim)]={}
 for t in [-1.,0.,1.]:
  flips=[];total=0
  for k in d.files:
   if not k.startswith('base_') or not k.endswith('_ode'):continue
   b=d[k][:,dim];a=d[k.replace('base_','round_2_',1)][:,dim]
   changed=np.logical_xor(b>=t,a>=t);total+=len(b)
   if changed.any():flips.append({'sample':k,'indices':np.where(changed)[0].tolist(),'before':b[changed].tolist(),'after':a[changed].tolist()})
  r[str(dim)][str(t)]={'flips':flips,'total_commands':total}
print(json.dumps(r))
"""
try:
 out=subprocess.check_output(['/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python','-c',code],text=True,timeout=30)
 r['gripper_threshold_audit']=json.loads(out)
except Exception as error:r['gripper_threshold_error']=type(error).__name__
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
