import pathlib,json,base64,subprocess
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/gripper_probe_20261025a/results')
r={}
for name in ['status.json','intervention/i301_s0/initialization.json','intervention/i301_s0/progress.json','intervention/i301_s0/result.json','failure.json']:
 q=p/name
 if q.exists():r[name]=json.loads(q.read_text())
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
