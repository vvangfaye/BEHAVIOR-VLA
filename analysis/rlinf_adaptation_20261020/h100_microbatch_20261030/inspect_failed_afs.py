import pathlib,json,base64
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/h100_microbatch_20261030/a1')
r={'exists':root.exists()}
if root.exists():
 r['files']=[str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()][:60]
 for name in ['bootstrap_error.json','bootstrap_status.json','results/status.json','results/error.json','driver.log']:
  p=root/name
  if p.exists():r[name]=p.read_text(errors='replace')[-12000:]
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
