import pathlib,os,json,base64
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/eval_results');out={}
for p in root.iterdir():
 if not p.is_dir():continue
 entries=[]
 for q in p.iterdir():
  if q.is_dir() and any(w in q.name.lower() for w in ['runtime','cache','env','prepar','repair','validation']):entries.append(str(q))
 if entries:out[str(p)]=entries
p=root/'b1k-video1688977-20261013'
for q in p.iterdir():
 if q.is_dir():out[str(q)]=[{'name':r.name,'dir':r.is_dir(),'link':r.is_symlink(),'size':r.lstat().st_size} for r in q.iterdir()][:100]
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KRUNTIMEEND',flush=True)
