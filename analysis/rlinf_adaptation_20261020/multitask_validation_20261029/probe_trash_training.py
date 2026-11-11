import pathlib,json,subprocess,base64
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_multitask_20261029/trash/results')
receipts=[json.loads(p.read_text()) for p in (root/'training_receipts/round_01').glob('episode_*.json')]
r={'episodes':len(receipts),'training_control_steps':sum(x['status']['steps'] for x in receipts),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],text=True).strip()}
p=root/'worker_status/m1_train1_r0_a1.json'
if p.exists():r['progress']=json.loads(p.read_text())
for name in ['critic_warmup.json','training_updates/train1/result.json']:
 p=root/name
 if p.exists():
  d=json.loads(p.read_text());r[name]={k:v for k,v in d.items() if k not in ['order','metrics','update_steps']}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
