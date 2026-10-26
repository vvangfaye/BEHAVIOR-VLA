import json,base64,subprocess,statistics,math
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_eval20_20261026/gate01')
r={}
f=p/'gate/progress.json'
if f.exists():r['progress']=json.loads(f.read_text())
f=p/'gate/tracking_audit.jsonl'
if f.exists():
 rows=[json.loads(x) for x in f.read_text().splitlines()]
 r['tracking_records']=len(rows)
 r['geometry_changed_boundaries']=sum(x['geometry_changed'] for x in rows)
 r['max_intrinsics_relative_error']=max([v for x in rows for v in x['intrinsics_relative_error'].values()] or [0])
 moved=[v for x in rows for v in x['targets'].values() if v['movement_m']>=.2 and v['visible_pixels']>=32]
 r['moved_visible_samples']=len(moved)
 if moved:r['median_error_m']=statistics.median(x['centroid_error_m'] for x in moved);r['max_error_m']=max(x['centroid_error_m'] for x in moved)
 r['latest_targets']={k:{name:v[name] for name in ['movement_m','estimated_movement_m','centroid_error_m','visible_pixels']} for k,v in rows[-1]['targets'].items()}
f=p/'gate/gate_result.json'
if f.exists():r['gate_result']=json.loads(f.read_text())
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader'],text=True).strip()
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
