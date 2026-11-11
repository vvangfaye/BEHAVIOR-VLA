import base64,json
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/init02')
r={}
for condition in ['image','serf']:
 for f in sorted((p/condition).glob('*_initial.png')):
  assert f.stat().st_size<2000000
  r[condition+'/'+f.name]=base64.b64encode(f.read_bytes()).decode()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
