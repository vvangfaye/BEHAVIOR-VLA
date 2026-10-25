import base64,json,subprocess
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/eval05/worker0')
r={}
if p.exists():
 for pattern in ['status.json','events.jsonl','serving_imports.log','*/progress.json','*/initial.json','*/result.json','*/server.log','*/evaluation.log']:
  for f in sorted(p.glob(pattern)):
   if f.is_file():r[str(f.relative_to(p))]=f.read_text()[-5000:]
r['checkpoint_downloads']=[{'directory':str(d),'files':len(fs),'bytes':sum(f.stat().st_size for f in fs),'latest_mtime':max((f.stat().st_mtime for f in fs),default=None)} for d in Path('/tmp').glob('serf-init-*') for fs in [[f for f in (d/'checkpoints').rglob('*') if f.is_file()]]]
r['processes']=subprocess.check_output(['ps','-eo','pid,ppid,comm,etime,pcpu,pmem,wchan'],text=True)[-6500:]
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True)
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
