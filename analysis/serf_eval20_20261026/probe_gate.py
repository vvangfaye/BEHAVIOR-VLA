import base64,json,subprocess
from pathlib import Path
r={}
p=Path('/mnt/afs/260010165/behavior_1k/serf_eval20_20261026/gate01')
for name in ['gate.log','gate/progress.json','gate/tracking_audit.jsonl','gate/gate_result.json']:
 f=p/name
 if f.exists():
  with f.open('rb') as s:s.seek(max(0,f.stat().st_size-4500));r[name]=s.read().decode(errors='replace')
  r[name+'_bytes']=f.stat().st_size
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
