import json,base64
from pathlib import Path
roots=list(Path('/tmp').glob('serf-init-*'));r=[]
for root in roots:
 p=root/'checkpoints'
 if p.is_dir():
  files=[x for x in p.rglob('*') if x.is_file()];r.append({'files':len(files),'bytes':sum(f.stat().st_size for f in files),'largest':sorted([f.stat().st_size for f in files],reverse=True)[:5]})
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
