import base64,hashlib,json,urllib.request,os
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/legacy01/data/omnigibson-robot-assets/models/background');p.mkdir(exist_ok=True,parents=True)
records=[]
items=[{'file': 'checkerboard.png', 'size': 384, 'sha256': '363ec8294ba36ed94fa8f746e329abbc83c4170bc562b4f0222a6885bc7878f4', 'key': 'behavior1k/serf-reproduction-20261025/legacy-assets/background/363ec8294ba36ed94fa8f746e329abbc83c4170bc562b4f0222a6885bc7878f4/checkerboard.png', 'url': 'https://aoss.cn-sh-01b.sensecoreapi-oss.cn/260010165/behavior1k/serf-reproduction-20261025/legacy-assets/background/363ec8294ba36ed94fa8f746e329abbc83c4170bc562b4f0222a6885bc7878f4/checkerboard.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=01A06AB4C7317A258E4BB929E817FD4E%2F20261025%2Fcn-sh-01b%2Fs3%2Faws4_request&X-Amz-Date=20261025T044656Z&X-Amz-Expires=7200&X-Amz-SignedHeaders=host&X-Amz-Signature=9e476791e7c6057723336346aabb61dccd5ace3452433aa84f4e60083c1bdcac'}, {'file': 'sky.jpg', 'size': 889717, 'sha256': '857cf073297ed69d7981d6f3a61b1e6c6118c1837f9620b7f24d725b20014b92', 'key': 'behavior1k/serf-reproduction-20261025/legacy-assets/background/857cf073297ed69d7981d6f3a61b1e6c6118c1837f9620b7f24d725b20014b92/sky.jpg', 'url': 'https://aoss.cn-sh-01b.sensecoreapi-oss.cn/260010165/behavior1k/serf-reproduction-20261025/legacy-assets/background/857cf073297ed69d7981d6f3a61b1e6c6118c1837f9620b7f24d725b20014b92/sky.jpg?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=01A06AB4C7317A258E4BB929E817FD4E%2F20261025%2Fcn-sh-01b%2Fs3%2Faws4_request&X-Amz-Date=20261025T044656Z&X-Amz-Expires=7200&X-Amz-SignedHeaders=host&X-Amz-Signature=4efde14cb7d45af1432de8d6e1b31f57e739941e500ff00276fa97c3cbd97795'}]
for x in items:
 with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(x['url'],timeout=90) as r:data=r.read()
 assert len(data)==x['size'] and hashlib.sha256(data).hexdigest()==x['sha256']
 dest=p/x['file']
 if dest.exists():assert dest.read_bytes()==data
 else:
  tmp=dest.with_name(dest.name+'.part');tmp.write_bytes(data);tmp.replace(dest)
 records.append({k:v for k,v in x.items() if k!='url'})
(p/'verified_assets.json').write_text(json.dumps(records,indent=2))
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps({'verified_background':records}).encode()).decode()+'B1KPROBE_END')
