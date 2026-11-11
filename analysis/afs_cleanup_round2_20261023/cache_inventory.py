import os,pathlib,stat,json,time,concurrent.futures,collections
os.nice(10)
base=pathlib.Path('/mnt/afs/260010165/behavior_1k/eval_results')
names=['behavior-native5090-20261007-150246','behavior-native5090-20261007-151849','behavior-state-schema-gate-20261007-185556','behavior-transition-gate-20261007-172224','behavior-transition-gate-20261007-175054']
roots=[base/n/'local-results-sim-runtime'/child for n in names for child in ['cache','warp','appdata']]
out=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct');out.mkdir(parents=True,exist_ok=True)
queue=collections.deque((i,r) for i,r in enumerate(roots) if r.exists());pending={};files=[];dirs=[];skipped=[]
def scan(i,p):
 rows=[];errors=[]
 try:
  with os.scandir(p) as it:
   for e in it:
    try:s=e.stat(follow_symlinks=False);rows.append((i,e.path,s.st_mode,s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_nlink,s.st_blocks))
    except OSError as err:errors.append((e.path,str(err)))
 except OSError as err:errors.append((str(p),str(err)))
 return rows,errors
def save(n,x):
 p=out/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x));t.replace(p)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
 while queue or pending:
  while queue and len(pending)<32:
   i,p=queue.popleft();pending[pool.submit(scan,i,p)]=p
  done,_=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
  for future in done:
   pending.pop(future);rows,errors=future.result();skipped.extend(errors)
   for i,p,mode,dev,ino,size,mtime,nlink,blocks in rows:
    if stat.S_ISDIR(mode):queue.append((i,pathlib.Path(p)));dirs.append([i,str(pathlib.Path(p).relative_to(roots[i]))])
    elif stat.S_ISREG(mode) and nlink==1 and mtime<1792684800*10**9:
     files.append({'root':i,'path':str(pathlib.Path(p).relative_to(roots[i])),'size':size,'mtime_ns':mtime,'inode':ino,'device':dev,'nlink':nlink,'blocks':blocks})
    else:skipped.append((p,'linked/special/recent file retained'))
result={'state':'review_ready','roots':[str(r) for r in roots],'files':files,'dirs':dirs,'skipped':skipped,'allocated_bytes':sum(f['blocks']*512 for f in files),'time':time.time()}
save('cache_manifest.json',result);save('cache_summary.json',{k:v for k,v in result.items() if k not in ['files','dirs'] }|{'file_count':len(files),'directory_count':len(dirs),'by_root':{str(r):sum(f['blocks']*512 for f in files if f['root']==i) for i,r in enumerate(roots)}})
