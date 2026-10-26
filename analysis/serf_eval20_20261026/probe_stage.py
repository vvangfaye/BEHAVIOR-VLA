import json,base64,subprocess
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_eval20_20261026/assets03');r={'exists':p.exists(),'scans':[]}
if p.exists():
 for f in p.glob('scan_*.json'):
  x=json.loads(f.read_text());r['scans'].append(x)
 for name in ['status.json']:
  if (p/name).exists():r[name]=json.loads((p/name).read_text())
r['processes']=[s for s in subprocess.check_output(['ps','-eo','comm,etime,stat'],text=True).splitlines() if s.strip().startswith('du ')]
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
