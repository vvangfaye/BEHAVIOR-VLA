import pathlib,sys,json,subprocess,base64,zlib,shlex,hashlib
HERE=pathlib.Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
CLI='/home/faye/.local/bin/sco';NAME='b1k-storage-inv-1024';KEY='behavior1k/storage-reclaim-20261024/inventory02.json'
app=json.loads(subprocess.run([CLI,'cci','apps','describe','b1k-storage-cpu-1024','--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True).stdout)
assert app['uid']=='131be07700064971ab3eeffeffed2009' and app['state']=='SUSPENDED'
c=aoss_client();source=(HERE/'inventory.py').read_text();config={'put':c.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':KEY},ExpiresIn=14400)}
packed=base64.b64encode(zlib.compress((source+'\nmain('+repr(config)+')\n').encode())).decode()
command='timeout --signal=TERM --kill-after=15s 1200s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
r={'name':NAME,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'key':KEY,'cpu':2,'memory_gib':8,'gpu':0,'max_seconds':1200,'state':'submitting'}
(HERE/'cpu_inventory_submission.json').write_text(json.dumps(r,indent=2)+'\n')
try:
 p=subprocess.run([CLI,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=pt-i8qy0bow','--name='+NAME,'--job-name='+NAME,'--worker-spec=n6ls.iu.i40.2c8g','--worker-nodes=1','--quota-type=reserved','--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--container-image-url=registry.cn-sh-01.sensecore.cn/ccr-zhicheng-04/wangfaye-base-20261010:wangfaye-debug-20261010154634','--command='+command],capture_output=True,text=True,timeout=90)
 r['returncode']=p.returncode
 d=json.loads(subprocess.run([CLI,'acp','jobs','describe',NAME,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,check=True).stdout)
 assert d['resource_pool']['name']=='share-cluster'
 r.update(uid=d['uid'],state=d['state'],resource_spec=d['roles'][0]['resource_spec'])
finally:c.close()
(HERE/'cpu_inventory_submission.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
