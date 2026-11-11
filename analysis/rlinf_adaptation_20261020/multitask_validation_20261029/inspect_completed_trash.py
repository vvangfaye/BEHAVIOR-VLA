import json,base64,pathlib,hashlib
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_multitask_20261029/trash/results')
receipts=[]
for i in range(1,33):
 p=root/'training_receipts/round_01'/f'episode_{i:02d}.json'
 if p.exists():
  r=json.loads(p.read_text());q=pathlib.Path(r['path'])
  receipts.append({'episode':r['episode'],'success':r['status']['success'],'score':r['status']['score'],'rollout_exists':q.is_file(),'rollout_size':q.stat().st_size if q.is_file() else None,'policy_hash':r['policy_hash'],'source_task':r.get('source_task')})
checks={}
for task in ['m1_sample1_r0_a4']+[f'm1_sample1_r{i}_a2' for i in range(1,8)]:
 p=root/'checks'/f'{task}.json'
 checks[task]=json.loads(p.read_text()) if p.exists() else None
r={'receipts':receipts,'checks':checks}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
