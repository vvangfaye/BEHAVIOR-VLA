import pathlib,json,base64
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/eval_results/b1k-video1688977-20261013');out={}
for rel in ['episodes/0249-g004','episodes/0000','batches/b1k-video1688977-20261013-b0260-0264','batches/b1k-video1688977-20261013-b0000-0004']:
 p=root/rel
 if not p.exists():continue
 out[rel]=[]
 for x in p.iterdir():
  out[rel].append({'name':x.name,'dir':x.is_dir(),'size':x.lstat().st_size})
  if x.is_dir():out[rel].append({'children_of':x.name,'children':[{'name':y.name,'dir':y.is_dir(),'size':y.lstat().st_size} for y in list(x.iterdir())[:12]]})
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KRUNTIMEEND',flush=True)
