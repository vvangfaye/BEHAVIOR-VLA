import os,json,subprocess,base64,urllib.request
from pathlib import Path
root=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/policydeps01')
runtime=Path('/mnt/afs/260010165/behavior_1k/runtime_20261007_1306')
ready=json.loads((runtime/'library_snapshots/behavior-libxml-repair3-20261008-011710/_READY').read_text())
env=dict(os.environ)
env.update(JAX_PLATFORMS='cpu',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',OMNIGIBSON_DATA_PATH='/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/legacy01/data',PYTHONPATH=str(root/'site')+':/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/dev01/site',LD_LIBRARY_PATH=ready['library_path']+':'+str(runtime/'envs/b1k-eval/lib')+':'+env.get('LD_LIBRARY_PATH',''))
p=subprocess.run([str(runtime/'envs/b1k-pi05/bin/python'),str(root/'probe.py')],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=120)
(root/'corrected_probe.log').write_text(p.stdout)
r=json.loads((root/'result.json').read_text());r['initial_probe_setup_error']=r.get('tail','');r.update(passed=p.returncode==0 and 'SERVING_IMPORTS_PASSED' in p.stdout,returncode=p.returncode,tail=p.stdout[-6000:],corrected_data_path=env['OMNIGIBSON_DATA_PATH'])
b=json.dumps(r,indent=2).encode();(root/'result.json').write_bytes(b)
url='https://aoss.cn-sh-01b.sensecoreapi-oss.cn/260010165/behavior1k/serf-reproduction-20261025/policydeps01/result.json?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=01A06AB4C7317A258E4BB929E817FD4E%2F20261025%2Fcn-sh-01b%2Fs3%2Faws4_request&X-Amz-Date=20261025T051210Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=c7cdf6b905ec25eca4d1cc479c65cf0c52fbc278081b32af539cf62d062c4dd7'
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(urllib.request.Request(url,data=b,method='PUT'),timeout=30):pass
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps(r).encode()).decode()+'B1KPROBE_END')
