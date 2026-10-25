"""Read only the failed SERF attempt's persisted logs in a short CPU job."""
import base64, json, shlex, subprocess, sys, zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
receipt=HERE/'recovery01_submission.json'
assert not receipt.exists(), 'Inspect the existing recovery job instead of resubmitting'
cli='/home/faye/.local/bin/sco'
old=json.loads((HERE/'eval02_submission.json').read_text())['workers'][0]
d=json.loads(subprocess.check_output([cli,'acp','jobs','describe',old['name'],'--workspace-name=share-space','-o','json'],text=True))
assert d['uid']==old['uid'] and d['state']=='FAILED' and d['ownership']['user_name']=='260010165'
root='/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025'
key='behavior1k/serf-reproduction-20261025/recovery01/result.json'
c=aoss_client()
try:url=c.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=7200)
finally:c.close()
source="""import json,urllib.request
from pathlib import Path
root=Path(ROOT)
source=root/'eval02/worker0'
report={'source':str(source),'files':{}}
if source.exists():
 for pattern in ['status.json','*/server.log','*/evaluation.log','*/result.json','*/progress.json','*/initial.json']:
  for path in sorted(source.glob(pattern)):
   if path.is_file():
    with path.open('rb') as f:
     f.seek(max(0,path.stat().st_size-22000));data=f.read(22000)
    report['files'][str(path.relative_to(source))]={'bytes':path.stat().st_size,'tail':data.decode(errors='replace')}
report['direct_children']=[{'path':str(p.relative_to(source)),'directory':p.is_dir()} for p in source.glob('*')]
dest=root/'recovery01';dest.mkdir(exist_ok=False)
data=json.dumps(report,indent=2).encode();(dest/'result.json').write_bytes(data)
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(urllib.request.Request(PUT,data=data,method='PUT'),timeout=45):pass
print('Persisted failure logs collected:',len(report['files']),flush=True)
"""
source='ROOT='+repr(root)+'\nPUT='+repr(url)+'\n'+source
packed=base64.b64encode(zlib.compress(source.encode())).decode()
command='timeout --signal=TERM --kill-after=10s 180s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
intent={'job':'b1k-serf-recover-1025-c01','source_job':old['name'],'source_uid':old['uid'],'gpus':0,'max_seconds':180,'result_key':key,'submission_attempted':True}
receipt.write_text(json.dumps(intent,indent=2)+'\n')
p=subprocess.run([cli,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=b1k-storage-inv-1024',
    '--name='+intent['job'],'--job-name='+intent['job'],'--worker-spec=n6ls.iu.i40.2c8g','--worker-nodes=1',
    '--quota-type=reserved','--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--command='+command],capture_output=True,text=True,timeout=90)
d=json.loads(subprocess.check_output([cli,'acp','jobs','describe',intent['job'],'--workspace-name=share-space','-o','json'],text=True))
assert d['ownership']['user_name']=='260010165'
intent.update(uid=d['uid'],state=d['state'],returncode=p.returncode);receipt.write_text(json.dumps(intent,indent=2)+'\n');print(json.dumps(intent))
