import hashlib
import io
import json
from pathlib import Path
import tarfile

HERE=Path(__file__).resolve().parent
BASE=HERE/'runtime51'
paths=['repo/src','repo/BEHAVIOR-1K/OmniGibson/omnigibson','repo/BEHAVIOR-1K/bddl/bddl',
       'repo/BEHAVIOR-1K/joylo/gello','repo/BEHAVIOR-1K/joylo/sampled_task/available_tasks.yaml',
       'repo/external/openpi/src','repo/external/openpi/packages/openpi-client/src',
       'repo/external/co-tracker/cotracker','repo/external/co-tracker/hubconf.py',
       'repo/scripts/serve_b1k.py','repo/scripts/serve_serf_b1k.py','import_probe.py',
       'serve_policy.py','run_case.py']
files=[]
for rel in paths:
    p=BASE/rel
    assert p.exists(),str(p)
    files.extend([p] if p.is_file() else sorted(p.rglob('*')))
manifest={}
with tarfile.open(HERE/'runtime51_source.tar.gz','w:gz') as tar:
    for path in files:
        if not path.is_file() or path.suffix in {'.pyc','.orig','.rej'} or '__pycache__' in path.parts:
            continue
        rel=str(path.relative_to(BASE))
        data=path.read_bytes();manifest[rel]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        item=tarfile.TarInfo(rel);item.size=len(data);item.mode=0o644;item.mtime=0
        tar.addfile(item,io.BytesIO(data))
(HERE/'runtime51_source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
p=HERE/'runtime51_source.tar.gz'
print(json.dumps({'files':len(manifest),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}))
