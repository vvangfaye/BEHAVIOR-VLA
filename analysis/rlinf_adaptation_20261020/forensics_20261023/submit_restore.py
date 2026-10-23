import base64
import concurrent.futures
import hashlib
import json
import shlex
import subprocess
import sys
import zlib
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

JOB='b1k-rlinf-restore-1023-01'


def main():
    manifest=json.loads((ROOT/'analysis/afs_audit_20261023/baseline_aoss_manifest.json').read_text())
    assert manifest['file_count']==21 and manifest['checkpoint_step']==1688977
    assert sum(x['size'] for x in manifest['files'])==manifest['total_bytes']
    plan={'job_name':JOB,'destination':'/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261023/checkpoints/original1688977','source_bucket':manifest['bucket'],'source_prefix':manifest['prefix'],'expected_bytes':manifest['total_bytes'],'expected_files':21,'cpu':8,'memory_gib':32,'gpu':0,'max_seconds':1800}
    (HERE/'restore_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (HERE/'restore_submission.json').open('x') as f:json.dump({**plan,'submission_attempted':False},f)
    c=aoss_client()
    try:
        def prepare(x):
            key=manifest['prefix']+x['relative_path']
            head=c.head_object(Bucket=manifest['bucket'],Key=key)
            assert head['ContentLength']==x['size'],x['relative_path']
            return {**x,'url':c.generate_presigned_url('get_object',Params={'Bucket':manifest['bucket'],'Key':key},ExpiresIn=86400)}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:files=list(pool.map(prepare,manifest['files']))
        resultkey='behavior1k/rlinf-adapter-20261023/restore-01/report.json'
        config={'destination':plan['destination'],'files':files,'put':c.generate_presigned_url('put_object',Params={'Bucket':'260010165','Key':resultkey},ExpiresIn=86400)}
        source=(HERE/'restore_worker.py').read_text()+'\nmain('+repr(config)+')\n'
        encoded=base64.b64encode(zlib.compress(source.encode())).decode()
        command='timeout --signal=TERM --kill-after=30s 1770s python3 -u -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(encoded)+')))')
        state={**plan,'submission_attempted':True,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'result_key':resultkey}
        (HERE/'restore_submission.json').write_text(json.dumps(state,indent=2)+'\n')
        cmd=['sco','acp','jobs','copy','--workspace-name=share-space','--copy-job-name=pt-o8xl7f79','--name='+JOB,'--job-name='+JOB,'--worker-spec=n6ls.iu.i40.8c32g','--worker-nodes=1','--quota-type=spot','--retry-times=0','--enable-fault-tolerance=false','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs','--command='+command]
        r=subprocess.run(cmd,capture_output=True,text=True,timeout=100)
        state['submit_returncode']=r.returncode
        r=subprocess.run(['sco','acp','jobs','describe',JOB,'--workspace-name=share-space','-o','json'],capture_output=True,text=True,timeout=40)
        if r.returncode==0:
            x=json.loads(r.stdout);state.update(uid=x['uid'],state=x['state'])
        else:state['state']='needs_exact_name_query'
        (HERE/'restore_submission.json').write_text(json.dumps(state,indent=2)+'\n')
        print(json.dumps({k:state.get(k) for k in ['job_name','uid','state','expected_files','expected_bytes']}))
    finally:c.close()
if __name__=='__main__':main()
