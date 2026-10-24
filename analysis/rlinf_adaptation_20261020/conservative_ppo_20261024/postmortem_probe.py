import pathlib,json,base64,time
r=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_conservative_20261024/pilot01')
out={'at':time.time(),'files':{}}
for name in ['results/status.json','results/failure.json','results/critic_warmup.json','results/previous_comparison.json','results/latest_checkpoint.json','results/training_summary.json','bootstrap_error.json','report_upload_receipt.json']:
 p=r/name
 if p.exists():out['files'][name]=json.loads(p.read_text())
for p in sorted((r/'results/training').glob('round_*/episode_*/*')):
 if p.is_file() and p.name in ('result.json','progress.json'):out['files'][str(p.relative_to(r))]=json.loads(p.read_text())
for p in sorted((r/'results/training').glob('round_*/episode_*/rollout/*')):
 if p.is_file() and p.name in ('result.json','progress.json','initialization.json','failure.json'):out['files'][str(p.relative_to(r))]=json.loads(p.read_text())
 if p.is_file() and p.name=='simulator.log':out['files'][str(p.relative_to(r))]={'tail':p.read_text(errors='replace')[-3500:]}
p=r/'result.tar.gz'
if p.exists():out['archive_bytes']=p.stat().st_size
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END',flush=True)
