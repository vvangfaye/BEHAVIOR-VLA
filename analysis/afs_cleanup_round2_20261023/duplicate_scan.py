import os,pathlib,stat,json,hashlib,time,concurrent.futures
os.nice(10)
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/eval_results/b1k-video1688977-20261013')
out=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct');out.mkdir(parents=True,exist_ok=True)
def save(name,x):
 p=out/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x));t.replace(p)
def meta(p):
 s=p.lstat();return {'path':str(p),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,'nlink':s.st_nlink,'blocks':s.st_blocks}
def digest(item):
 p=pathlib.Path(item['path']);before=meta(p);assert before==item and p.resolve()==p
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assert meta(p)==before
 return {**item,'sha256':h.hexdigest()}
try:
 refs={};candidates=[]
 for name in ['episodes','final_submission','batches']:
  for d,ds,fs in os.walk(root/name,followlinks=False):
   ds[:]=[x for x in ds if not (pathlib.Path(d)/x).is_symlink()]
   for name2 in fs:
    p=pathlib.Path(d)/name2;s=p.lstat()
    if not stat.S_ISREG(s.st_mode) or s.st_size<1024**2 or s.st_nlink!=1:continue
    item=meta(p)
    if pathlib.Path(item['path']).is_relative_to(root/'batches'):candidates.append(item)
    else:refs.setdefault(item['size'],[]).append(item)
 sizes={c['size'] for c in candidates if c['size'] in refs}
 todo=[r for size in sizes for r in refs[size]]+[c for c in candidates if c['size'] in sizes]
 save('duplicate_progress.json',{'stage':'hashing','files':len(todo),'bytes':sum(x['size'] for x in todo),'done':0})
 hashed=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for r in pool.map(digest,todo):
   hashed.append(r)
   if len(hashed)%20==0:save('duplicate_progress.json',{'stage':'hashing','files':len(todo),'done':len(hashed),'time':time.time()})
 keep={}
 for r in hashed:
  if not pathlib.Path(r['path']).is_relative_to(root/'batches'):keep.setdefault((r['size'],r['sha256']),r)
 pairs=[]
 for r in hashed:
  if pathlib.Path(r['path']).is_relative_to(root/'batches') and (r['size'],r['sha256']) in keep:pairs.append({'remove':r,'retain':keep[(r['size'],r['sha256'])]})
 result={'state':'review_ready','root':str(root),'pairs':pairs,'files':len(pairs),'allocated_bytes':sum(x['remove']['blocks']*512 for x in pairs),'hashed_files':len(hashed),'hash_bytes':sum(x['size'] for x in hashed),'time':time.time()}
 save('duplicate_manifest.json',result);save('duplicate_progress.json',{k:v for k,v in result.items() if k!='pairs'})
except Exception as e:save('duplicate_progress.json',{'state':'error','type':type(e).__name__,'message':str(e)})
