import pathlib,json,base64,time
r=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/pilot01')
out={'at':time.time(),'exists':r.exists(),'files':[p.name for p in r.iterdir()] if r.exists() else []}
for name in ['bootstrap_error.json','runtime_receipt.json','training_instances_verified.json','results/status.json']:
 p=r/name
 if p.is_file():out[name]=json.loads(p.read_text())
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END',flush=True)
