import base64,json
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_eval17_20261027/pilot01')
r=root/'worker_1/results'
s=json.loads((r/'status.json').read_text())
case=r/s['stage']/f"i{s['instance']}_s{s['seed']}"
result={'status':{k:v for k,v in s.items() if k!='comparison'},'case':str(case),'files':{n:json.loads((case/n).read_text()) for n in ['initialization.json','progress.json','result.json'] if (case/n).is_file()},'completed':[]}
for phase in ['baseline','final']:
 for p in sorted((r/phase).glob('i*_s*/result.json')):result['completed'].append({'phase':phase,**json.loads(p.read_text())})
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KPROBE_END')
