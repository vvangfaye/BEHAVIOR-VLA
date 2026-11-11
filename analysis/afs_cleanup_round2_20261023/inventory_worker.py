import os,pathlib,stat,time,json,concurrent.futures,collections
root=pathlib.Path('/mnt/afs/260010165');out=pathlib.Path('/tmp/b1k_afs_round2_inventory.json');started=time.monotonic();rs=root.lstat();device=rs.st_dev
queue=collections.deque([root]);pending={};seen=set();total=rs.st_blocks*512;by_top={};by_depth={};largest=[];errors=[];files=0;last_save=0

def scan(path):
 rows=[];fail=[]
 try:
  with os.scandir(path) as entries:
   for e in entries:
    try:s=e.stat(follow_symlinks=False);rows.append((e.path,s.st_dev,s.st_ino,s.st_nlink,s.st_blocks,stat.S_ISDIR(s.st_mode),s.st_size,s.st_mtime))
    except OSError as error:fail.append((e.path,str(error)))
 except OSError as error:fail.append((str(path),str(error)))
 return rows,fail

def save(complete):
 result={'time':time.time(),'complete':complete,'allocated_bytes':total,'files_counted':files,'by_top':by_top,'by_depth':{k:v for k,v in by_depth.items() if v>256*1024**2},'large_files':sorted(largest,reverse=True)[:400],'errors':errors[:40],'error_count':len(errors),'elapsed_seconds':time.monotonic()-started,'directories_pending':len(queue)+len(pending)}
 temp=out.with_suffix('.tmp');temp.write_text(json.dumps(result));temp.replace(out)
with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
 while queue or pending:
  while queue and len(pending)<64:
   path=queue.popleft();pending[pool.submit(scan,path)]=path
  done,_=concurrent.futures.wait(pending,timeout=1,return_when=concurrent.futures.FIRST_COMPLETED)
  for future in done:
   pending.pop(future);rows,fail=future.result();errors.extend(fail)
   for path,dev,ino,nlink,blocks,isdir,logical,mtime in rows:
    if dev!=device:continue
    if nlink>1:
     key=(dev,ino)
     if key in seen:continue
     seen.add(key)
    size=blocks*512;total+=size;files+=1
    parts=pathlib.Path(path).relative_to(root).parts;top=parts[0];by_top[top]=by_top.get(top,0)+size
    for depth in (2,3):
     key='/'.join(parts[:depth]);by_depth[key]=by_depth.get(key,0)+size
    if not isdir and logical>512*1024**2:largest.append((size,path,logical,mtime,nlink))
    if isdir:queue.append(pathlib.Path(path))
  if time.monotonic()-last_save>10:save(False);last_save=time.monotonic()
  if time.monotonic()-started>600:save(False);os._exit(2)
save(True)
