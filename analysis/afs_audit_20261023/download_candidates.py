import base64,json,pathlib
root=pathlib.Path('/mnt/afs/260010165');rows=[]
for p in [root/'cache/wheels/isaacsim',root/'cache/conda/pkgs',root/'behavior_1k/runtime_20261007_1306/inputs']:
 files=[]
 for f in p.rglob('*'):
  if f.is_file() and not f.is_symlink():
   s=f.stat()
   if s.st_size>1000000:files.append({'path':str(f),'size':s.st_size,'nlink':s.st_nlink,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,'blocks':s.st_blocks})
 rows.append({'root':str(p),'files':sorted(files,key=lambda f:-f['size'])[:150]})
p=root/'behavior-train-local/meta/info.json';info=json.loads(p.read_text())
result={'rows':rows,'train_info':{k:v for k,v in info.items() if k in ['repo_id','total_episodes','total_frames','total_tasks','video_path','data_path','codebase_version']}}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
