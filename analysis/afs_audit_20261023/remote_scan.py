import base64,json,pathlib,hashlib
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/serving_bundles');rows=[]
for n in ['b1k-video1688977-20261014-source','b1k-video1688977-20261014-source-g1','b1k-video1688977-20261014-source-g2']:
 f=p/n/'source.tar.gz';h=hashlib.sha256()
 with f.open('rb') as r:
  for chunk in iter(lambda:r.read(8*1024*1024),b''):h.update(chunk)
 rows.append({'path':str(f),'sha256':h.hexdigest(),'size':f.stat().st_size})
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps({'hashes':rows}).encode()).decode()+'B1KRUNTIMEEND',flush=True)
