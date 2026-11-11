import base64, json, zlib
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_modem_20261028/pilot01/results')
out=[]
for ep in range(1,33):
    p=root/'training'/'round_01'/f'episode_{ep:02d}'
    out.append({'episode':ep,'trace':(p/'rollout'/'goal_trace.jsonl').read_text(),
                'receipt':json.loads((root/'training_receipts'/'round_01'/f'episode_{ep:02d}.json').read_text()),
                'initial':json.loads((p/'rollout'/'initial.json').read_text()),
                'videos':[{'name':v.name,'bytes':v.stat().st_size} for v in (p/'rollout').glob('*.mp4')]})
packed={'zlib_base64':base64.b64encode(zlib.compress(json.dumps(out).encode())).decode()}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(packed).encode()).decode()+'B1KPROBE_END',flush=True)
