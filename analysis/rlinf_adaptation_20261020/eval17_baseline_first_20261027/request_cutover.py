import base64,json,os,signal,subprocess,sys
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/rlinf_eval17_20261027/pilot01/worker_1')
results=root/'results'
expected=str(root/'run/driver.py').encode()
pids=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  if expected in p.read_bytes().split(b'\x00'):pids.append(int(p.parent.name))
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
assert len(pids)==1,pids
pid=pids[0]
os.kill(pid,signal.SIGSTOP)
state=json.loads((results/'status.json').read_text())
case=results/state['stage']/f"i{state['instance']}_s{state['seed']}"
init=json.loads((case/'initialization.json').read_text()) if (case/'initialization.json').exists() else {}
ready={'driver_pid':pid,'reason':'baseline_first_schedule_requested','status':{k:v for k,v in state.items() if k!='comparison'},'initialization':init}
receipt=root/'CUTOVER_READY.json'
if init.get('stage')!='ready' and not (case/'progress.json').exists():
 ready.update(paused=True,controls_in_current_case=0)
 receipt.write_text(json.dumps(ready,indent=2))
else:
 os.kill(pid,signal.SIGCONT)
 ready.update(paused=False,awaiting_completed_count=state['completed_new_episodes']+1)
 watcher="""import ctypes,json,os,select,signal,time
from pathlib import Path
root=Path(ROOT_TEXT);results=root/'results';pid=PID_NUMBER;target=TARGET_NUMBER
libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(0);assert fd>=0
assert libc.inotify_add_watch(fd,str(results).encode(),0x80|0x8)>=0
end=time.monotonic()+2400
while time.monotonic()<end:
 s=json.loads((results/'status.json').read_text())
 if s['completed_new_episodes']>=target:
  os.kill(pid,signal.SIGSTOP)
  out={'paused':True,'driver_pid':pid,'reason':'completed_episode_boundary','completed_new_episodes':s['completed_new_episodes'],'status':{k:v for k,v in s.items() if k!='comparison'}}
  tmp=root/'CUTOVER_READY.tmp';tmp.write_text(json.dumps(out,indent=2));tmp.replace(root/'CUTOVER_READY.json');break
 if select.select([fd],[],[],min(60,max(0,end-time.monotonic())))[0]:os.read(fd,65536)
os.close(fd)
""".replace('ROOT_TEXT',repr(str(root))).replace('PID_NUMBER',str(pid)).replace('TARGET_NUMBER',str(ready['awaiting_completed_count']))
 log=(root/'cutover_watch.log').open('ab')
 child=subprocess.Popen([sys.executable,'-c',watcher],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 ready['watcher_pid']=child.pid
 (root/'CUTOVER_REQUEST.json').write_text(json.dumps(ready,indent=2))
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(ready).encode()).decode()+'B1KPROBE_END')
