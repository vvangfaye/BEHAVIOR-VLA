import json,base64,subprocess
from pathlib import Path
root=Path('/tmp/b1k-serf20-eval03-worker0')
r={'root_exists':root.exists()}
if root.exists():
 r['files']=[p.name for p in root.iterdir()]
 for name in ['status.json','serving_imports.log','client_transport.log']:
  f=root/name
  if f.exists():r[name]=f.read_text(errors='replace')[-5000:]
 for dest in root.iterdir():
  if not dest.is_dir() or not (dest/'server.log').exists():continue
  item={}
  for name in ['progress.json','server.log','evaluation.log']:
   f=dest/name
   if f.exists():item[name]=f.read_text(errors='replace')[-2500:]
  r[dest.name]=item
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
