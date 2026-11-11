import json,pathlib,base64,subprocess
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/control_20261025a/results')
r={}
for rel in ['status.json','failure.json','baseline/i301_s0/progress.json','baseline/i301_s0/result.json','baseline/i301_s0/initialization.json','final/i301_s0/progress.json','final/i301_s0/result.json','final/i301_s0/initialization.json']:
 q=p/rel
 if q.exists():r[rel]=json.loads(q.read_text())
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
a=p/'baseline/i301_s0/initial.json'
b=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_multiround_20261023/pilot01_reserved/results/baseline/i301_s0/initial.json')
if a.exists() and b.exists():
 x=json.loads(a.read_text());y=json.loads(b.read_text());r['baseline_vs_historical_initial']={'physical_exact':x['physical_state']==y['physical_state'],'proprio_exact':x['proprio']==y['proprio'],'physical_sha256':x['physical_sha256'],'time_steps_equal':all(x[k]==y[k] for k in ['physics_dt','rendering_dt','observation_dt'])}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
