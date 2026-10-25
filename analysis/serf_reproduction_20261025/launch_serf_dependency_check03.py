import base64,hashlib,json,shlex,subprocess,sys,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
receipt=HERE/'policydeps03_submission.json';assert not receipt.exists()
last=json.loads((HERE/'eval04_submission.json').read_text());sha=last['source_sha256']
assert hashlib.sha256((HERE/'runtime51_source.tar.gz').read_bytes()).hexdigest()==sha
c=aoss_client();prefix='behavior1k/serf-reproduction-20261025/policydeps03/'
try:
    config={'root':'/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/policydeps03','source_sha256':sha,'base_dependency_site':'/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/policydeps02/site',
        'source_get':c.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':'behavior1k/serf-reproduction-20261025/eval04/source-'+sha+'.tar.gz'},ExpiresIn=7200),
        'result_put':c.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':prefix+'result.json'},ExpiresIn=7200)}
finally:c.close()
source=(HERE/'serf_dependency_worker.py').read_text()+'\nrun('+repr(config)+')\n'
packed=base64.b64encode(zlib.compress(source.encode())).decode()
command='timeout --signal=TERM --kill-after=30s 1200s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
intent={'job':'b1k-serf-deps-1025-c03','gpus':0,'max_seconds':1200,'root':config['root'],'source_sha256':sha,'base_dependency_site':'/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/policydeps02/site','result_key':prefix+'result.json','submission_attempted':True}
receipt.write_text(json.dumps(intent,indent=2)+'\n');cli='/home/faye/.local/bin/sco'
p=subprocess.run([cli,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=b1k-storage-inv-1024','--name='+intent['job'],'--job-name='+intent['job'],
    '--worker-spec=n6ls.iu.i40.2c8g','--worker-nodes=1','--quota-type=reserved','--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--command='+command],capture_output=True,text=True,timeout=90)
d=json.loads(subprocess.check_output([cli,'acp','jobs','describe',intent['job'],'--workspace-name=share-space','-o','json'],text=True))
assert d['ownership']['user_name']=='260010165'
intent.update(uid=d['uid'],state=d['state'],returncode=p.returncode);receipt.write_text(json.dumps(intent,indent=2)+'\n');print(json.dumps(intent))
