import base64,json,re
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_modem_20261028/pilot01')
out={}
job=root/'jobs/p8_sample2_r6_a1'
for name in ['driver.log','driver_exit.json']:
    p=job/name
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-14000))
            out[name]=re.sub(r'https?://\S+','[URL omitted]',f.read().decode(errors='replace'))
p=root/'results/failures/p8_sample2_r6_a1.json'
if p.exists():out['failure']=json.loads(p.read_text())
out['training_receipts']=[json.loads(p.read_text()) for p in sorted((root/'results/training_receipts/round_02').glob('episode_*.json'))]
out['episodes']=[{'episode':p.name,'has_rollout':(p/'rollout.pt').exists(),'has_result':(p/'result.json').exists(),'progress':json.loads((p/'progress.json').read_text()) if (p/'progress.json').exists() else None} for p in sorted((root/'results/training/round_02').glob('episode_*'))]
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
