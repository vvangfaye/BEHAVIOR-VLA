import pathlib,json,base64
p=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct');r={}
for name in ['duplicate_progress.json','cache_summary.json','cache_release.json','duplicate_release.json','cache_delete_error.json','zip_progress.json','zip_summary.json','zip_release.json','tar_progress.json','tar_release.json','cache_verification.json']:
 if (p/name).exists():
  value=json.loads((p/name).read_text())
  if name=='tar_release.json':value={k:v for k,v in value.items() if k not in ['results','retained']}|{'retained_count':len(value.get('retained',[])),'retained_sample':value.get('retained',[])[:3]}
  r[name]=value
if (p/'duplicate_manifest.json').exists():
 x=json.loads((p/'duplicate_manifest.json').read_text());r['duplicate_samples']=x['pairs'][:5]
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KRUNTIMEEND',flush=True)
