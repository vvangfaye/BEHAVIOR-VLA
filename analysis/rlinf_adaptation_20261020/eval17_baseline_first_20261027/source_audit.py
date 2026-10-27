import base64,hashlib,json
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_eval17_20261027/pilot01/worker_1')
r=root/'results';ready=root/'CUTOVER_READY.json'
out={'paused':ready.exists()}
if ready.exists():
 out['cutover']=json.loads(ready.read_text());manifest={};cases=[]
 for phase in ['baseline','final']:
  for path in sorted((r/phase).glob('i*_s*/result.json')):
   result=json.loads(path.read_text());cases.append([phase,result['instance'],result['seed']])
   names=['result.json','initial.json','spec.json']+(['actions.json'] if phase=='final' else [])
   for name in names:
    f=path.parent/name;assert f.is_file(),str(f);manifest[str(f.relative_to(r))]=hashlib.sha256(f.read_bytes()).hexdigest()
 out.update(passed=True,complete_manifest=manifest,completed_cases=cases,source_root=str(r),status=json.loads((r/'status.json').read_text()))
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
