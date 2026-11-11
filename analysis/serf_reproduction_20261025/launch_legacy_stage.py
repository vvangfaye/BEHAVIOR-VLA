import base64,json,shlex,subprocess,sys,zlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parents[1]/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client
path=HERE/'legacy_stage01_submission.json';assert not path.exists()
asset=json.loads((HERE/'legacy_asset_upload.json').read_text())
c=aoss_client();key='behavior1k/serf-reproduction-20261025/legacy-stage01/result.json'
try:
    config={'root':'/mnt/afs/260010165/behavior_1k/serf_reproduction_20261025/legacy01','bytes':asset['bytes'],'sha256':asset['sha256'],
        'get':c.generate_presigned_url('get_object',Params={'Bucket':'260010165','Key':asset['key']},ExpiresIn=7200),
        'put':c.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':key},ExpiresIn=7200)}
finally:c.close()
source=(HERE/'legacy_stage_worker.py').read_text()+'\nrun('+repr(config)+')\n'
encoded=base64.b64encode(zlib.compress(source.encode())).decode()
command='timeout --signal=TERM --kill-after=30s 1800s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))')
intent={'job':'b1k-serf-legacy-1025-a01','gpus':0,'max_seconds':1800,'root':config['root'],'result_key':key,'submission_attempted':True}
path.write_text(json.dumps(intent,indent=2)+'\n')
cli='/home/faye/.local/bin/sco'
r=subprocess.run([cli,'acp','jobs','copy','--workspace-name=share-space','--copy-job-name=b1k-storage-inv-1024',
    '--name='+intent['job'],'--job-name='+intent['job'],'--worker-spec=n6ls.iu.i40.2c8g','--worker-nodes=1','--quota-type=reserved',
    '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false','--command='+command],capture_output=True,text=True,timeout=90)
d=json.loads(subprocess.check_output([cli,'acp','jobs','describe',intent['job'],'--workspace-name=share-space','-o','json'],text=True))
intent.update(uid=d['uid'],state=d['state'],returncode=r.returncode);path.write_text(json.dumps(intent,indent=2)+'\n');print(json.dumps(intent))
