import os,pathlib,stat,json,time,hashlib
out=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct')
raw=(out/'cache_manifest.json').read_bytes();m=json.loads(raw)
assert m['state']=='review_ready' and 0<m['allocated_bytes']<60*1024**3
allowed={'behavior-native5090-20261007-150246','behavior-native5090-20261007-151849','behavior-state-schema-gate-20261007-185556','behavior-transition-gate-20261007-172224','behavior-transition-gate-20261007-175054'}
roots=[pathlib.Path(p) for p in m['roots']]
for p in roots:
 assert p.resolve()==p and p.parent.name=='local-results-sim-runtime' and p.parent.parent.name in allowed and p.name in {'cache','warp','appdata'}
for proc in pathlib.Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  for fd in (proc/'fd').iterdir():
   try:t=os.readlink(fd)
   except OSError:continue
   if any(t.startswith(str(r)+'/') for r in roots):raise RuntimeError('Candidate cache still open by process '+proc.name)
 except PermissionError:pass

def snapshot(p):
 s=p.lstat();return {k:getattr(s,'st_'+k) for k in ['size','mtime_ns','ino','dev','nlink','blocks']}
def matches(item,p):
 s=p.lstat();return stat.S_ISREG(s.st_mode) and p.resolve()==p and all(getattr(s,'st_'+a)==item[b] for a,b in [('size','size'),('mtime_ns','mtime_ns'),('ino','inode'),('dev','device'),('nlink','nlink'),('blocks','blocks')])
# Full preflight finishes before the first unlink.
for f in m['files']:assert matches(f,roots[f['root']]/f['path'])
released=[];errors=[]
for f in m['files']:
 p=roots[f['root']]/f['path']
 if not matches(f,p):raise RuntimeError('Source changed during cleanup')
 try:p.unlink();released.append(f)
 except OSError as e:errors.append({'path':str(p),'error':str(e)})
for i,rel in sorted(m['dirs'],key=lambda x:len(pathlib.Path(x[1]).parts),reverse=True):
 try:(roots[i]/rel).rmdir()
 except OSError:pass
result={'state':'completed','manifest_sha256':hashlib.sha256(raw).hexdigest(),'files_removed':len(released),'allocated_bytes_removed':sum(f['blocks']*512 for f in released),'by_root':{str(r):sum(f['blocks']*512 for f in released if f['root']==i) for i,r in enumerate(roots)},'errors':errors,'skipped':m['skipped'],'time':time.time(),'backup_requested':False}
(out/'cache_release.json').write_text(json.dumps(result,indent=2))
