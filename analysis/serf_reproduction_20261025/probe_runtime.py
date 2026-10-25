import base64
import json
from pathlib import Path
import subprocess

root = Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
out = {'processes': subprocess.check_output(['ps','-eo','pid,comm,etime'], text=True)[-7000:]}
out['asset_versions'] = {}
for name in ('behavior-1k-assets','omnigibson-robot-assets'):
    p=root/'omnigibson-data'/name/'VERSION'
    out['asset_versions'][name]=p.read_text()[:100] if p.exists() else None
for name in ('b1k-pi05','b1k-eval'):
    site=list((root/'envs'/name/'lib').glob('python*/site-packages'))
    out[name] = []
    for path in site:
        out[name] += [p.name for p in path.glob('*.dist-info') if p.name.split('-')[0].lower() in {'jax','jaxlib','torch','flax','isaacsim','numpy','open3d','yourdfpy','orbax_checkpoint'}]
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END',flush=True)
