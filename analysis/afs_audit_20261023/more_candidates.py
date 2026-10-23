import base64,json,pathlib,os,subprocess,concurrent.futures
root=pathlib.Path('/mnt/afs/260010165')
paths=[root/'cache/wheels',root/'cache/conda',root/'behavior_1k/runtime_20261007_1306',root/'behavior-train-local/meta',root/'behavior_1k/checkpoints']
rows=[]
for p in paths:
 entries=[]
 for x in p.iterdir():
  s=x.lstat();entries.append({'name':x.name,'size':s.st_size,'directory':x.is_dir(),'symlink':x.is_symlink()})
 rows.append({'path':str(p),'entries':entries[:200]})
result={'rows':rows}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
