import base64,json,subprocess,time
from pathlib import Path
started=time.time()
try:
    result=subprocess.run(['nice','-n','15','du','-sx','--block-size=1','/mnt/afs/260010165'],capture_output=True,text=True,timeout=40)
    out={'elapsed_seconds':time.time()-started,'returncode':result.returncode}
    if result.returncode==0:
        out['account_allocated_bytes']=int(result.stdout.split()[0])
        out['policy_quota_bytes']=2*1024**4
        out['estimated_policy_headroom_bytes']=out['policy_quota_bytes']-out['account_allocated_bytes']
    else:out['error']=result.stderr[-1000:]
except subprocess.TimeoutExpired:out={'bounded_scan_timed_out':True,'elapsed_seconds':time.time()-started}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(out).encode()).decode()+'B1KPROBE_END')
