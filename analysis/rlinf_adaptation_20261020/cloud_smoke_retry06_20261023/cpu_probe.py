import pathlib,json,os,time,base64
result={}
cg=pathlib.Path('/sys/fs/cgroup')
for name in ['cpu.max','cpuset.cpus.effective','cpu.stat']:
 p=cg/name
 if p.exists():result[name]=p.read_text()
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try:
  args=(path/'cmdline').read_bytes().split(b'\0')
  if not any(a.endswith(b'/fp32_reference.py') for a in args):continue
  pid=int(path.name);st=(path/'stat').read_text().split();start=int(st[13])+int(st[14]);t=time.monotonic();time.sleep(1);st=(path/'stat').read_text().split();stop=int(st[13])+int(st[14])
  status={line.split(':')[0]:line.split(':',1)[1].strip() for line in (path/'status').read_text().splitlines() if ':' in line}
  env=dict(e.split(b'=',1) for e in (path/'environ').read_bytes().split(b'\0') if b'=' in e)
  result['reference']={'pid':pid,'cpu_cores_used':(stop-start)/os.sysconf('SC_CLK_TCK')/(time.monotonic()-t),'threads':status.get('Threads'),'cpu_allowed':status.get('Cpus_allowed_list'),'rss':status.get('VmRSS'),'omp':env.get(b'OMP_NUM_THREADS',b'').decode(),'mkl':env.get(b'MKL_NUM_THREADS',b'').decode()}
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KPROBE_END',flush=True)
