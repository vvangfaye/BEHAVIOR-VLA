import json,base64,socket,subprocess,os
from pathlib import Path
r={'gpu':subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout,'processes':[]}
for p in Path('/proc').glob('[0-9]*'):
 try:
  status=(p/'status').read_text()
  if 'python' not in status and 'elastic' not in status:continue
  fields={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in status.splitlines() if ':' in x}
  if int(fields.get('VmRSS','0 kB').split()[0])<200000:continue
  env={}
  for item in (p/'environ').read_bytes().split(b'\0'):
   key,_,value=item.partition(b'=')
   if key in [b'MASTER_ADDR',b'MASTER_PORT',b'RANK',b'WORLD_SIZE',b'LOCAL_RANK',b'GLOO_SOCKET_IFNAME',b'NCCL_SOCKET_IFNAME']:
    env[key.decode()]=value.decode()
  row={'pid':p.name,'state':fields.get('State'),'wait':(p/'wchan').read_text(),'env':env}
  if 'MASTER_ADDR' in env:
   try:
    conn=socket.create_connection((env['MASTER_ADDR'],int(env['MASTER_PORT'])),timeout=1);conn.close();row['master_tcp']='reachable'
   except Exception as e:row['master_tcp']=str(e)
  r['processes'].append(row)
 except (OSError,ValueError):pass
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
