import base64,json,subprocess,time
result={'at':time.time(),'gpu':subprocess.check_output([
    'nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total',
    '--format=csv,noheader'],text=True,timeout=10).strip()}
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(result).encode()).decode()+'B1KPROBE_END',flush=True)
