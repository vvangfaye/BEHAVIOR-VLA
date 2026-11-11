import pathlib,json,base64,os
r={}
for name in ['b1k_afs_usage_20261023.json','b1k_afs_parallel_usage.json']:
 p=pathlib.Path('/tmp')/name
 if p.exists():r[name]=json.loads(p.read_text())
s=os.statvfs('/mnt/afs/260010165');r['shared_available_bytes']=s.f_bavail*s.f_frsize;r['shared_total_bytes']=s.f_blocks*s.f_frsize
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KRUNTIMEEND',flush=True)
