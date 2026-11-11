import pathlib,json,base64
r={}
for name in ['memory.events','memory.current','memory.peak','memory.max','memory/memory.failcnt','memory/memory.max_usage_in_bytes','memory/memory.limit_in_bytes']:
 p=pathlib.Path('/sys/fs/cgroup')/name
 if p.exists():r[name]=p.read_text()[:4000]
r['processes']=[]
for p in pathlib.Path('/proc').glob('[0-9]*/status'):
 try:
  d={}
  for l in p.read_text().splitlines():
   k,_,v=l.partition(':')
   if k in ['Name','Pid','PPid','VmRSS','VmHWM']:d[k]=v.strip()
  if 'VmRSS' in d and int(d['VmRSS'].split()[0])>500000:r['processes'].append(d)
 except (FileNotFoundError,PermissionError):pass
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END',flush=True)
