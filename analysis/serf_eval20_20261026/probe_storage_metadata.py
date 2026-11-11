import os,json,base64
p='/mnt/afs/260010165';r={}
for name in ['ceph.dir.rbytes','ceph.quota.max_bytes','ceph.quota.max_files']:
 try:r[name]=os.getxattr(p,name).decode()
 except OSError as e:r[name]={'errno':e.errno}
for line in open('/proc/self/mountinfo'):
 a=line.split()
 if len(a)>8 and a[4]=='/mnt/afs':r['filesystem_type']=line.split(' - ',1)[1].split()[0]
s=os.statvfs(p);r['filesystem_available_bytes']=s.f_bavail*s.f_frsize
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
