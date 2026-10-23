import pathlib,json,base64,stat
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/checkpoints/pi05_b1k_full100_joint_1688977_aoss_20261010');files=[]
for p in root.rglob('*'):
 if p.is_dir():continue
 s=p.lstat();assert stat.S_ISREG(s.st_mode)
 files.append(dict(path=str(p.relative_to(root)),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,nlink=s.st_nlink,blocks=s.st_blocks))
result={'source':str(root),'files':files}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
