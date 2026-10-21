from pathlib import Path
import hashlib,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
CACHE=ROOT/'analysis/training_boundary_audit_20261019/pilot_three_tasks_cache_v1'
manifest=json.loads((CACHE/'cache_manifest.json').read_text())
rows=[]
for r in manifest['records']:
 p=CACHE/r['npz_path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==r['npz_sha256']
 with np.load(p) as f:
  a=f['action'][~f['action_is_pad'].astype(bool)]
  assert a.ndim==2 and a.shape[1]==23 and len(a)>1
  spread=np.ptp(a,axis=0)
  steps=np.diff(a,axis=0)
  rows.append({'episode':r['episode_index'],'frame':r['frame_index'],'n':len(a),'range':spread.tolist(),'constant_exact':(spread==0).tolist(),'constant_1e6':(spread<=1e-6).tolist(),'adjacent_exact_zero_rate':(steps==0).mean(0).tolist()})
fields=np.array([x['constant_exact'] for x in rows]);near=np.array([x['constant_1e6'] for x in rows]);adj=np.array([x['adjacent_exact_zero_rate'] for x in rows])
result={'source':'existing 36 expert anchors, three selected tasks, nine episodes; not a population estimate','samples':len(rows),'episodes':len(set(r['episode'] for r in rows)),'constant_chunk_fraction_by_coordinate':fields.mean(0).tolist(),'near_constant_chunk_fraction_by_coordinate':near.mean(0).tolist(),'exact_repeat_fraction_by_coordinate':adj.mean(0).tolist(),'rows':rows,'limits':['A constant command is not necessarily a stationary physical joint','Continuous controls can have small nonzero changes not atoms','A finite recorded plateau does not prove the population has singular support','No checkpoint performance or closed-loop causal conclusion']}
Path(__file__).with_name('support_result.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
