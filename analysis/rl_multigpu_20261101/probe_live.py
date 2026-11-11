import json,base64,subprocess
from pathlib import Path
r={'gpu':subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used','--format=csv,noheader'],capture_output=True,text=True).stdout,'processes':[]}
for f in Path('/proc').glob('[0-9]*/status'):
 try:
  d={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in f.read_text().splitlines() if ':' in line}
  if int(d.get('VmRSS','0 kB').split()[0])>200000:r['processes'].append({k:d.get(k) for k in ['Name','Pid','PPid','State','VmRSS']})
 except (OSError,ValueError):pass
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
