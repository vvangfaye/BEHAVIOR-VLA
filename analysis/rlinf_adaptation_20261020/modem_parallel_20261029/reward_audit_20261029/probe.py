"""Read scalar reward traces from existing immutable round-one rollouts only."""
import base64
import json
import os
import subprocess

inner = r'''
import gc, json, time
from pathlib import Path
import torch
torch.set_num_threads(1)
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_modem_20261028/pilot01/results')
keys=['toggled_on','nextto_tv','shared_tv_cabinet']
out={'columns':['step','phi','on','near','cabinet','score','success','task_reward','shaping_reward','reward','done','truncated'], 'episodes':[]}
start=time.monotonic()
for ep in range(1,33):
    if time.monotonic()-start > 42:
        out['timeout_after_episode']=ep-1
        break
    p=root/'training'/'round_01'/f'episode_{ep:02d}'
    trace=[json.loads(x) for x in (p/'rollout'/'goal_trace.jsonl').read_text().splitlines() if x.strip()]
    receipt=json.loads((root/'training_receipts'/'round_01'/f'episode_{ep:02d}.json').read_text())
    rows=[]
    item={'episode':ep,'receipt':receipt,'trace_count':len(trace),'pt_bytes':(p/'rollout.pt').stat().st_size}
    item['initial']=json.loads((p/'rollout'/'initial.json').read_text())
    item['videos']=[{'name':str(v.relative_to(p)), 'bytes':v.stat().st_size} for v in (p/'rollout').glob('*.mp4')]
    data=torch.load(p/'rollout.pt',map_location='cpu',weights_only=True,mmap=True)
    transitions=data['transitions']
    assert len(trace)==len(transitions),(ep,len(trace),len(transitions))
    for t,tr in zip(trace,transitions):
        rows.append([t['step'],t['potential'],*[int(t[k]) for k in keys],t['official']['score'],int(t['official']['success']),tr['task_reward'],tr['shaping_reward'],tr['reward'],int(tr['terminal']),int(tr['truncated'])])
        assert all(t[k]==tr['goals_after'][k] for k in keys)
    item['rows']=rows
    out['episodes'].append(item)
    del data,transitions,tr
    gc.collect()
out['elapsed_seconds']=time.monotonic()-start
print(json.dumps(out,separators=(',',':')))
'''
env=dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
r=subprocess.run(['/mnt/afs/260010165/behavior_1k/runtime_20261007_1306/envs/b1k-pi05/bin/python','-c',inner],env=env,text=True,capture_output=True,timeout=52)
if r.returncode:
    out={'error':r.stderr[-5000:], 'stdout_tail':r.stdout[-1000:], 'returncode':r.returncode}
else:
    out=json.loads(r.stdout)
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out,separators=(',',':')).encode()).decode()+'B1KPROBE_END',flush=True)
