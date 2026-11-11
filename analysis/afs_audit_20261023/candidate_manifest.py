import base64,json,pathlib,os,stat,hashlib,time
root=pathlib.Path('/mnt/afs/260010165/behavior_1k')
names=['expert_fixed20_step1000_20261016/checkpoint1000','quality_taskbalanced_fixed20_20261016/checkpoint5000','quality_taskbalanced_fixed20_step1000_20261016/checkpoint1000','paired_quick20_reused1688977_step10000_20261014/checkpoint10000','paired_quick20_reused1688977_step20000_20261015/checkpoint20000','paired_quick20_reused1688977_step5000_20261014/checkpoint5000','rft60_40_fixed20_step5000_20261015/checkpoint5000','huber50_heldout20_20261017/checkpoint1000']
rows=[]
for name in names:
 p=root/name;files=[]
 for d,ds,fs in os.walk(p):
  for f in fs:
   q=pathlib.Path(d)/f;s=q.lstat()
   if not stat.S_ISREG(s.st_mode):raise RuntimeError('nonregular')
   files.append(dict(path=str(q.relative_to(p)),size=s.st_size,mtime_ns=s.st_mtime_ns,inode=s.st_ino,device=s.st_dev,nlink=s.st_nlink,blocks=s.st_blocks))
 rows.append(dict(source=str(p),files=files,ready=json.loads((p/'_READY').read_text()),sha256sums=(p/'SHA256SUMS.inference').read_text()))
result={'roots':rows,'time':time.time()}
print('B1KRUNTIMEBEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KRUNTIMEEND',flush=True)
