import os,pathlib,json,base64,re,stat
root=pathlib.Path('/mnt/afs/260010165'); rows=[]
for rel in ['data','models','behavior-train-local','behavior_1k/raw_full100_20261007_1308','behavior_1k/runtime_20261007_1306','behavior_1k/codex_pilot_20261011','behavior_1k/codex_subtask_freeze_20261012','behavior_1k/codex_subtask_camera_20261013','behavior_1k/rlinf_adapter_20261020/smoke01','behavior_1k/rlinf_adapter_20261023']:
 p=root/rel;files=[]
 try:
  for f in p.iterdir():
   s=f.lstat();files.append({'name':f.name,'dir':stat.S_ISDIR(s.st_mode),'link':os.readlink(f) if f.is_symlink() else None,'size':s.st_size,'nlink':s.st_nlink})
  rows.append({'path':str(p),'entries':files})
 except OSError as e:rows.append({'path':str(p),'error':str(e)})
procs=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  text=(p/'cmdline').read_bytes().decode(errors='replace').replace('\0',' ')
  if '/mnt/afs/260010165' not in text:continue
  paths=sorted(set(re.findall(r'/mnt/afs/260010165/[A-Za-z0-9_./-]+',text)))
  refs=[]
  for f in (p/'fd').iterdir():
   try:
    v=os.readlink(f)
    if v.startswith('/mnt/afs/260010165'):refs.append(v)
   except OSError:pass
  procs.append({'pid':p.name,'paths':paths,'fd_paths':sorted(set(refs))[:200]})
 except OSError:pass
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps({'rows':rows,'processes':procs}).encode()).decode()+'B1KRUNTIMEEND',flush=True)
