import pathlib,json,base64
p=pathlib.Path('/tmp/b1k_afs_usage_20261023.json');r={'done':p.exists()}
if p.exists():r['result']=json.loads(p.read_text())
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KRUNTIMEEND',flush=True)
