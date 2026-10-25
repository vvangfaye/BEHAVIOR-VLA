import json,pathlib,base64,subprocess
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/control_20261025a/results')
r={}
for rel in ['status.json','failure.json','baseline/i301_s0/progress.json','baseline/i301_s0/result.json','baseline/i301_s0/initialization.json','final/i301_s0/progress.json','final/i301_s0/result.json','final/i301_s0/initialization.json']:
 q=p/rel
 if q.exists():r[rel]=json.loads(q.read_text())
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
code = """import numpy as np,json
p='/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/diagnostic_20261025a/results/actions.npz'
d=np.load(p,allow_pickle=False)
ids=[k[len('base_'):-len('_ode')] for k in d.files if k.startswith('base_') and k.endswith('_ode')]
a={}
for model in ['base','round_1','round_2']:
 for mode in ['ode','sde']:a[model+'_'+mode]=np.concatenate([d[model+'_'+k+'_'+mode] for k in ids])
r={}
for dim in [14,22]:
 mask=a['base_ode'][:,dim]>.9
 r[str(dim)]={'reference_near_open_commands':int(mask.sum()),'counts_ge_1':{k:int((v[mask,dim]>=1.).sum()) for k,v in a.items()},'mean_command':{k:float(v[mask,dim].mean()) for k,v in a.items()}}
print(json.dumps(r))
"""
try:r['gripper_cap_statistics']=json.loads(subprocess.check_output(['/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python','-c',code],text=True,timeout=30))
except Exception as error:r['cap_error']=type(error).__name__
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
