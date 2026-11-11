import pathlib,json,base64,subprocess
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/gripper_probe_20261025a/results')
r={}
for name in ['status.json','intervention/i301_s0/initialization.json','intervention/i301_s0/progress.json','intervention/i301_s0/result.json','failure.json']:
 q=p/name
 if q.exists():r[name]=json.loads(q.read_text())
q=p/'intervention/i301_s0/initial.json'
if q.exists():r['intervention_initial_hash']=json.loads(q.read_text())['physical_sha256']
r['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.used,utilization.gpu','--format=csv,noheader'],text=True).strip()
base=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/control_20261025a/results')
a=json.loads((base/'baseline/i301_s0/initial.json').read_text());b=json.loads((base/'final/i301_s0/initial.json').read_text())
g=p/'intervention/i301_s0/initial.json'
if g.exists():
 z=json.loads(g.read_text());r['intervention_matches_base']={'physical_exact':z['physical_state']==a['physical_state'],'proprio_exact':z['proprio']==a['proprio']}
r['fresh_pair_extra']={'proprio_exact':a['proprio']==b['proprio'],'physical_exact':a['physical_state']==b['physical_state'],'same_time_steps':all(a[k]==b[k] for k in ['physics_dt','rendering_dt','observation_dt']),'physical_sha256':a['physical_sha256']}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
