import json,pathlib,base64,subprocess
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/control_20261025a/results')
r={}
for rel in ['status.json','failure.json','baseline/i301_s0/progress.json','baseline/i301_s0/result.json','baseline/i301_s0/initialization.json','final/i301_s0/progress.json','final/i301_s0/result.json','final/i301_s0/initialization.json']:
 q=p/rel
 if q.exists():r[rel]=json.loads(q.read_text())
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
