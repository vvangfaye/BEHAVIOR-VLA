import os,pathlib,stat,json,time,collections,urllib.request,traceback
ROOT=pathlib.Path('/mnt/afs/260010165/behavior_1k')
TARGETS=['rlinf_multiround_20261023','rlinf_adapter_20261023','rlinf_conservative_20261024']

def inventory():
 started=time.time();by_dir=collections.defaultdict(int);by_type=collections.defaultdict(int);largest=[];errors=[];seen=set();count=0
 for name in TARGETS:
  root=ROOT/name
  for directory,subdirs,files in os.walk(root,followlinks=False):
   subdirs[:]=[n for n in subdirs if not pathlib.Path(directory,n).is_symlink()]
   for n in files:
    p=pathlib.Path(directory,n)
    try:
     s=p.lstat();count+=1
     if not stat.S_ISREG(s.st_mode):continue
     key=(s.st_dev,s.st_ino)
     if key in seen:continue
     seen.add(key);size=s.st_blocks*512;rel=p.relative_to(ROOT);parts=rel.parts
     for depth in range(1,min(6,len(parts))+1):by_dir['/'.join(parts[:depth])]+=size
     group='cache' if any(x in ['cache','appdata','warp','config'] for x in parts) else ('core_dump' if n=='core' or n.startswith('core.') else p.suffix)
     by_type[group]+=size
     if size>=128*1024**2:largest.append({'path':str(p),'allocated':size,'size':s.st_size,'nlink':s.st_nlink,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns})
    except OSError as e:errors.append({'path':str(p),'error':str(e)})
    if time.time()-started>900:raise TimeoutError('Inventory budget 15 minutes')
 probe=ROOT/f'.storage_write_probe_{os.getpid()}'
 try:
  with probe.open('xb') as f:f.write(b'x'*4096);f.flush();os.fsync(f.fileno())
  probe.unlink();writable=True
 except OSError as e:writable=str(e)
 v=os.statvfs(ROOT)
 return {'at':time.time(),'complete':True,'elapsed':time.time()-started,'files':count,'by_directory':dict(by_dir),'by_type':dict(by_type),'large_files':sorted(largest,key=lambda x:x['allocated'],reverse=True)[:120],'errors':errors[:40],'error_count':len(errors),'write_probe':writable,'volume_free_bytes_not_user_quota':v.f_bavail*v.f_frsize}

def main(config):
 try:r=inventory()
 except BaseException:r={'complete':False,'traceback':traceback.format_exc()}
 data=json.dumps(r).encode();opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
 with opener.open(urllib.request.Request(config['put'],data=data,method='PUT'),timeout=120) as response:assert response.status in (200,201,204)
 print(json.dumps({'complete':r.get('complete'),'files':r.get('files')}),flush=True)
