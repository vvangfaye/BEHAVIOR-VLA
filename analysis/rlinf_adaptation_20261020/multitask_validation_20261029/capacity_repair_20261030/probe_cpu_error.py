import base64,json
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/rlinf_multitask_20261029/trash/capacity_20261030_a1')
out={}
for name in ['du.stdout','du.stderr','status.json']:
 f=p/name
 if f.exists():
  with f.open('rb') as stream:
   stream.seek(max(0,f.stat().st_size-16000));out[name]=stream.read().decode(errors='replace')
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
