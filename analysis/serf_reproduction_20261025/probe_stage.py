import base64,json
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/prepare01')
out={'root':p.exists()}
if p.exists():
    out['entries']=[{'name':x.name,'is_dir':x.is_dir(),'bytes':x.stat().st_size} for x in p.iterdir()]
    for name in ['stage_result.json','import_policy.log','import_sim.log']:
        f=p/name
        if f.is_file():out[name]=f.read_text()[-7000:]
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
