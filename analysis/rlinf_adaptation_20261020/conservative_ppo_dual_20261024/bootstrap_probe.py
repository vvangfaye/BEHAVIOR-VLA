import base64,json,pathlib
base=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/dual01')
r={}
for rank in (0,1):
 root=base/f'worker_{rank}'
 d={'exists':root.exists(),'files':[p.name for p in root.iterdir()] if root.exists() else []}
 for name in ('runtime_receipt.json','resume_preflight.json','bootstrap_error.json'):
  p=root/name
  if p.exists():d[name]=json.loads(p.read_text())
 r[str(rank)]=d
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
