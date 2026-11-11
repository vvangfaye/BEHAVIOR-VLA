import os,json,base64,stat
paths=['/mnt/afs/260010165/envs/b1k-eval/lib/python3.11/site-packages/isaacsim/kit/data/documents/Kit/shared/screenshots','/mnt/afs/260010165/runtime/omnigibson/8000/appdata/local/data/documents/Kit/shared/screenshots']
r={'euid':os.geteuid(),'egid':os.getegid(),'paths':[]}
for p in paths:
 try:
  s=os.stat(p);row={'path':p,'uid':s.st_uid,'gid':s.st_gid,'mode':oct(stat.S_IMODE(s.st_mode)),'size':s.st_size}
  try:row['children']=os.listdir(p)[:10]
  except OSError as e:row['list_errno']=e.errno
  r['paths'].append(row)
 except OSError as e:r['paths'].append({'path':p,'errno':e.errno})
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
