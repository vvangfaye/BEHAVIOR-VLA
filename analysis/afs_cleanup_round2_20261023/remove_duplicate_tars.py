import os,pathlib,tarfile,hashlib,json,time,stat,concurrent.futures,threading
os.nice(10)
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/eval_results/b1k-video1688977-20261013')
out=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct');proofroot=out/'tar_verification';proofroot.mkdir(exist_ok=True)
files=sorted((root/'episodes').glob('*/artifact.tar.gz'))+sorted((root/'artifact_repairs').glob('*/artifact.tar.gz'))
assert len(files)==1000 and sum(p.lstat().st_blocks*512 for p in files)<31*1024**3

def meta(p):
 s=p.lstat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,'nlink':s.st_nlink,'blocks':s.st_blocks,'mode':s.st_mode}
def save(p,obj):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(obj));t.replace(p)
def one(p):
 item={'source':str(p),'state':'retained'}
 try:
  source=meta(p);bundle=p.parent/'bundle';assert p.resolve()==p and bundle.resolve()==bundle and source['nlink']==1 and stat.S_ISREG(source['mode'])
  matched=[]
  with tarfile.open(p,'r:gz') as tf:
   for member in tf:
    rel=pathlib.PurePosixPath(member.name)
    assert not rel.is_absolute() and '..' not in rel.parts and rel.parts[0]=='episode'
    target=bundle.joinpath(*rel.parts[1:])
    if member.isdir():assert target.is_dir();continue
    if member.issym():assert target.is_symlink() and os.readlink(target)==member.linkname;continue
    assert member.isfile(),'Unsupported archive entry retained'
    assert target.resolve()==target
    before=meta(target);assert stat.S_ISREG(before['mode']) and before['size']==member.size
    h=hashlib.sha256()
    with tf.extractfile(member) as a,target.open('rb') as b:
     while True:
      left=a.read(1024*1024);right=b.read(1024*1024)
      assert left==right,'Unpacked bytes differ'
      if not left:break
      h.update(left)
    assert meta(target)==before
    matched.append({'member':member.name,'retained':str(target),'sha256':h.hexdigest(),'identity':before})
  assert meta(p)==source and matched
  # Recheck every retained file after the complete archive has been read.
  assert all(meta(pathlib.Path(f['retained']))==f['identity'] for f in matched)
  proof={'source':str(p),'source_identity':source,'verified_members':matched,'verified_at':time.time()}
  key=p.parent.name+'-'+p.parent.parent.name+'.json';save(proofroot/key,proof)
  assert meta(p)==source
  p.unlink()
  item.update(state='deleted_duplicate_tar',bytes=source['blocks']*512,verified_files=len(matched),proof=str(proofroot/key))
 except Exception as e:item.update(reason=type(e).__name__+': '+str(e))
 return item
results=[]
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
 for result in pool.map(one,files):
  results.append(result)
  save(out/'tar_progress.json',{'checked':len(results),'total':len(files),'removed':sum(x['state']=='deleted_duplicate_tar' for x in results),'allocated_bytes_removed':sum(x.get('bytes',0) for x in results),'last':result,'time':time.time()})
summary={'state':'completed','checked':len(results),'files_removed':sum(x['state']=='deleted_duplicate_tar' for x in results),'allocated_bytes_removed':sum(x.get('bytes',0) for x in results),'retained':[x for x in results if x['state']!='deleted_duplicate_tar'],'results':results,'time':time.time()}
save(out/'tar_release.json',summary)
