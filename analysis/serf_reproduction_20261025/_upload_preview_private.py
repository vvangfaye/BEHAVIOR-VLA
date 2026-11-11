import base64,json,tarfile,io,hashlib,urllib.request
from pathlib import Path
p=Path('/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/init02')
b=io.BytesIO();files=[]
with tarfile.open(fileobj=b,mode='w:gz') as t:
 for condition in ['image','serf']:
  for f in sorted((p/condition).glob('*_initial.png')):
   assert f.stat().st_size<2000000
   t.add(f,arcname=condition+'/'+f.name,recursive=False);files.append(condition+'/'+f.name)
data=b.getvalue()
url='https://aoss.cn-sh-01b.sensecoreapi-oss.cn/260010165/behavior1k/serf-reproduction-20261025/init02/image_preview.tar.gz?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=01A06AB4C7317A258E4BB929E817FD4E%2F20261025%2Fcn-sh-01b%2Fs3%2Faws4_request&X-Amz-Date=20261025T044926Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=a3e59b36cf6c2ae1045698aef108daaba09eb148c466cb632c4495792257c21f'
with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(urllib.request.Request(url,data=data,method='PUT'),timeout=60):pass
print('B1KPROBE_BEGIN'+base64.b64encode(json.dumps({'files':files,'size':len(data),'sha256':hashlib.sha256(data).hexdigest()}).encode()).decode()+'B1KPROBE_END')
