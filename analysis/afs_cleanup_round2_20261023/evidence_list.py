import pathlib,json,base64
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct')
r={'files':[{'name':f.name,'size':f.stat().st_size} for f in p.iterdir()]}
for name in ['duplicate_progress.json','cache_delete_error.json','cache_release.json','zip_progress.json']:
 if (p/name).exists():r[name]=json.loads((p/name).read_text())
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KRUNTIMEEND',flush=True)
