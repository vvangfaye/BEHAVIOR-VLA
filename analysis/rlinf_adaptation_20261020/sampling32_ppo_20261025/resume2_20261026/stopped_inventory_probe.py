import base64,json,os
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/resume2_20261026/pilot01/results')
report={'exists':root.exists(),'training':{},'failures':{},'workers':{},'evaluation':{}}
for rnd in (1,2):
 rows=[json.loads(p.read_text()) for p in sorted((root/'coordination').glob(f'round_{rnd:02d}_episode_*.json'))]
 report['training'][str(rnd)]={'complete':len(rows),'successes':sum(x['status']['success'] for x in rows),'episodes':[x['episode'] for x in rows],'rows':rows if rnd==2 else []}
 if (root/'training'/f'round_{rnd:02d}'/'result.json').exists():
  report['training'][str(rnd)]['optimizer_steps']=json.loads((root/'training'/f'round_{rnd:02d}'/'result.json').read_text())['update']['optimizer_steps']
for p in (root/'coordination').glob('failure*.json'):report['failures'][p.name]=json.loads(p.read_text())
for rank in (0,1):
 p=root/f'status_rank{rank}.json'
 if not p.is_file():continue
 status=json.loads(p.read_text());entry={'status':status}
 if 'episode' in status:
  ep=root/'training'/f"round_{status['round']:02d}"/f"episode_{status['episode']:02d}"
  for name in ('progress.json','rollout/initialization.json','rollout/failure.json','rollout/process_failure.json'):
   p=ep/name
   if p.is_file():entry[name]=json.loads(p.read_text())
  p=ep/'rollout/simulator.log'
  if p.is_file():
   with p.open('rb') as f:
    f.seek(max(0,p.stat().st_size-2400));entry['log_tail']=f.read().decode(errors='replace')
 report['workers'][str(rank)]=entry
for phase in ('baseline','final'):
 rows=[json.loads(p.read_text()) for p in (root/phase).glob('i*_s*/result.json')]
 report['evaluation'][phase]={'complete':len(rows),'successes':sum(x['success'] for x in rows)}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(report).encode()).decode()+'B1KPROBE_END')
