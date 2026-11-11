import os,pathlib,zipfile,hashlib,json,time,stat
os.nice(10)
root=pathlib.Path('/mnt/afs/260010165/data/urban_city_v2');out=pathlib.Path('/mnt/afs/260010165/behavior_1k/operational_cleanup/20261023-round2-direct')
def meta(p):
 s=p.lstat();return {'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,'nlink':s.st_nlink,'blocks':s.st_blocks}
def save(n,x):
 p=out/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x));t.replace(p)
results=[];archives=sorted((root/'archives').glob('*.zip'))
for zip_path in archives:
 item={'source':str(zip_path),'identity':meta(zip_path),'members':[]}
 try:
  assert zip_path.resolve()==zip_path and item['identity']['nlink']==1
  with zipfile.ZipFile(zip_path) as z:
   for info in z.infolist():
    if info.is_dir():continue
    rel=pathlib.PurePosixPath(info.filename);assert not rel.is_absolute() and '..' not in rel.parts
    target=root/'tiles'/zip_path.stem/'mesh'/str(rel)
    before=meta(target);assert target.resolve()==target and stat.S_ISREG(target.lstat().st_mode) and before['size']==info.file_size
    h=hashlib.sha256()
    with z.open(info) as a,target.open('rb') as b:
     while True:
      left=a.read(1024*1024);right=b.read(1024*1024)
      assert left==right,'Extracted bytes differ'
      if not left:break
      h.update(left)
    assert meta(target)==before
    item['members'].append({'path':str(target),'sha256':h.hexdigest(),'identity':before})
  assert meta(zip_path)==item['identity']
  item['state']='identical_extracted_copy'
 except Exception as e:item.update(state='retained',reason=type(e).__name__+': '+str(e))
 results.append(item)
 save('zip_progress.json',{'checked':len(results),'total':len(archives),'eligible_count':sum(r['state']=='identical_extracted_copy' for r in results),'eligible_bytes':sum(r['identity']['blocks']*512 for r in results if r['state']=='identical_extracted_copy'),'last':{k:v for k,v in item.items() if k!='members'}})
save('zip_manifest.json',{'results':results,'time':time.time()})
save('zip_summary.json',{'checked':len(results),'eligible_count':sum(r['state']=='identical_extracted_copy' for r in results),'eligible_bytes':sum(r['identity']['blocks']*512 for r in results if r['state']=='identical_extracted_copy'),'retained':[{'source':r['source'],'reason':r.get('reason')} for r in results if r['state']!='identical_extracted_copy']})
