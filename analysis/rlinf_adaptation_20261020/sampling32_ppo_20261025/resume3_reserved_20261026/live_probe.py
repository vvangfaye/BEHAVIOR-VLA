import base64,json,subprocess
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_sampling32_20261025/resume3_reserved_20261026/pilot01/results')
report={'training':{},'workers':{},'evaluation':{},'recovery':{}}
for rank in (0,1):
 audit=root.parent/f'worker_{rank}'/'resume_preflight.json'
 if audit.is_file():report['recovery'][str(rank)]=json.loads(audit.read_text())
for r in (1,2):
 files=sorted((root/'coordination').glob(f'round_{r:02d}_episode_*.json'))
 rows=[json.loads(p.read_text()) for p in files]
 report['training'][str(r)]={'complete':len(rows),'reused':sum((x['reused'] or x.get('reused_by_resume',False)) for x in rows),'new':sum(not (x['reused'] or x.get('reused_by_resume',False)) for x in rows),'successes':sum(x['status']['success'] for x in rows),'policy_hashes':sorted(set(x['policy_hash'] for x in rows))}
 p=root/'training'/f'round_{r:02d}/result.json'
 if p.is_file():
  x=json.loads(p.read_text());report['training'][str(r)]['optimizer_steps']=x['update']['optimizer_steps']
for rank in (0,1):
 s=root/f'status_rank{rank}.json';row={}
 if s.is_file():
  x=json.loads(s.read_text());row['status']=x
  if x.get('stage')=='training_rollout':
   out=root/'training'/f"round_{x['round']:02d}"/f"episode_{x['episode']:02d}"
   for name in ('progress.json','rollout/initialization.json'):
    if (out/name).is_file():row[name]=json.loads((out/name).read_text())
 report['workers'][str(rank)]=row
for phase in ('baseline','final'):
 rows=[json.loads(p.read_text()) for p in (root/phase).glob('i*_s*/result.json')]
 report['evaluation'][phase]={'complete':len(rows),'successes':sum(x['success'] for x in rows)}

for name in ('optimizer_restore.json','initialization.json'):
 if (root/name).is_file():report[name]=json.loads((root/name).read_text())
report['gpu']=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip()
report['root_exists']=root.exists()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(report).encode()).decode()+'B1KPROBE_END')
