import pathlib,os,json,base64,time
rows=[]
for d in pathlib.Path('/proc').iterdir():
 if not d.name.isdigit():continue
 try:
  comm=(d/'comm').read_text().strip()
  if 'python' not in comm:continue
  status=(d/'status').read_text().splitlines();row={'pid':int(d.name),'comm':comm,'status':[x for x in status if x.startswith(('VmRSS:','State:','Threads:'))],'wchan':(d/'wchan').read_text(),'io':(d/'io').read_text(),'fds':[]}
  for f in (d/'fd').iterdir():
   try:
    t=os.readlink(f)
    if t.startswith('/mnt/afs/260010165/behavior_1k') or t.startswith('socket:'):row['fds'].append(t)
   except OSError:pass
  rows.append(row)
 except OSError:pass
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps({'at':time.time(),'rows':rows}).encode()).decode()+'B1KPROBE_END',flush=True)
