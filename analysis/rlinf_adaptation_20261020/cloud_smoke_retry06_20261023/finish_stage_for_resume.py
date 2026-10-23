"""End only this baseline process after its first complete episode is durable."""
import os,json,signal,time,pathlib,base64
root=pathlib.Path('/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/retry06')
result=json.loads((root/'results/jax/301/result.json').read_text())
assert result['instance']==301 and result['backend']=='jax' and 0<result['steps']<=3224
status=json.loads((root/'results/status.json').read_text())
assert status['stage']=='baseline' and status['state']=='running'
second=root/'results/jax/302/progress.json'
assert not second.exists() or json.loads(second.read_text())['steps']==0
assert not (root/'results/ppo').exists()
roles={}
for path in pathlib.Path('/proc').iterdir():
 if not path.name.isdigit():continue
 try: args=(path/'cmdline').read_bytes().split(b'\0')
 except (FileNotFoundError,PermissionError):continue
 if str(root/'run/driver.py').encode() in args:roles['driver']=int(path.name)
 if str(root/'run/rollout.py').encode() in args and b'jax' in args:roles['baseline']=int(path.name)
assert set(roles)=={'driver','baseline'},roles
pid=roles['baseline'];stat=pathlib.Path(f'/proc/{pid}/stat').read_text().split()
assert int(stat[3])==roles['driver'] and os.getpgid(pid)==pid and os.getpgrp()!=pid
receipt={'at':time.time(),'reason':'Resume the remaining CPU reference episode in a job with a sufficient stage deadline; preserve the finished episode without resampling.','job_uid':'19f023b244544fd787eb449c546b5ed4','roles':roles,'first_episode':result,'second_has_no_control_steps':True}
(root/'results/controlled_resume.json').write_text(json.dumps(receipt,indent=2)+'\n')
os.killpg(pid,signal.SIGTERM)
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(receipt).encode()).decode()+'B1KPROBE_END',flush=True)
