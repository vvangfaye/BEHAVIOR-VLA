import os,json,time,pathlib,base64
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/retry06')
roles={}
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().split(b'\0')
  for role in ['driver','fp32_reference']:
   if str(root/'run'/f'{role}.py').encode() in args: roles[role]=int(p.name)
 except (FileNotFoundError,PermissionError):pass
assert 'driver' in roles and 'fp32_reference' in roles, roles
quota,period=map(int,pathlib.Path('/sys/fs/cgroup/cpu.max').read_text().split())
assert quota//period==14
allowed=sorted(os.sched_getaffinity(roles['driver']))
chosen=allowed[:14]
assert len(chosen)==14
record={'time':time.time(),'reason':'Bound CPU inference to the existing 14-core cgroup allocation; future children inherit affinity.','cpus':chosen,'targets':{}}
for role,pid in roles.items():
 before=sorted(os.sched_getaffinity(pid));tasks=list(pathlib.Path(f'/proc/{pid}/task').iterdir());count=0
 for task in tasks:
  try: os.sched_setaffinity(int(task.name),set(chosen));count+=1
  except ProcessLookupError:pass
 record['targets'][role]={'pid':pid,'before_cpus':before,'after_cpus':sorted(os.sched_getaffinity(pid)),'threads_bound':count}
(root/'cpu_affinity_receipt.json').write_text(json.dumps(record,indent=2)+'\n')
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(record).encode()).decode()+'B1KPROBE_END',flush=True)
