import base64,json,pathlib,stat
root=pathlib.Path('/mnt/afs/260010165/cache/wheels/isaacsim');files=[]
for p in root.rglob('*'):
 if p.is_dir():continue
 s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
 files.append({'path':str(p.relative_to(root)),'size':s.st_size,'mtime_ns':s.st_mtime_ns,'inode':s.st_ino,'device':s.st_dev,'nlink':s.st_nlink,'blocks':s.st_blocks})
result={'source':str(root),'files':files}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
