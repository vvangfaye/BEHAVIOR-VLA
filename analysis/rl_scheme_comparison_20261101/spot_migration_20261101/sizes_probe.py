import json,base64,time
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rl_scheme_comparison_20261101')
r={'at':time.time(),'cases':[]}
for d in (root/'evaluations').glob('*/*/results'):
 files=[x for x in d.rglob('*') if x.is_file()]
 r['cases'].append({'path':str(d.relative_to(root)),'allocated_bytes':sum(x.stat().st_blocks*512 for x in files),'max_file':max(((x.stat().st_size,str(x.relative_to(d))) for x in files),default=None)})
p=Path('/mnt/afs/260010165/behavior_1k/rlinf_multitask_20261029/water/results')
r['original_output_allocated_bytes']=sum(x.stat().st_blocks*512 for x in p.rglob('*') if x.is_file())
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
