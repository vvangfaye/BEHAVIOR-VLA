"""Prepare or submit exactly one preregistered arm after effective adaptation."""
import argparse
import base64
import fcntl
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import zlib

from submit import cli,write,sha
from admission import verified_configuration,validate_trial_configuration

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
AFS='/mnt/afs/260010165/behavior_1k'
METHODS=('meta','maml','fixed_crossplay','rl_bc','before_after')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--admission-attempt',choices=('05','06','06h100'),default='06h100')
    parser.add_argument('--validate-only',action='store_true')
    parser.add_argument('--submit',action='store_true');args=parser.parse_args()
    source_result=HERE/('full_checkpoint_'+args.admission_attempt)/'result.json'
    if not source_result.exists():
        print(json.dumps(dict(ready=False,reason='effective_adaptation_result_pending')));return
    admission=json.loads(source_result.read_text())
    if not (admission.get('passed') and admission.get('adaptation_effective') and admission.get('worker_returncode')==0):
        print(json.dumps(dict(ready=False,reason='effective_adaptation_not_passed')));return
    accepted_plan=source_result.with_name('plan.json')
    if sha(accepted_plan)!=admission['plan_sha256']:
        raise ValueError('Accepted result is not bound to the frozen optimizer configuration')
    accepted=json.loads(accepted_plan.read_text())
    plan=json.loads((HERE/'closed_loop_pilot_v2'/f'{args.method}.json').read_text())
    proof=dict(admission,verified_configuration=verified_configuration(accepted))
    validate_trial_configuration(plan,proof)
    if len(plan['iterations'])*2!=24:raise ValueError('Unexpected episode budget')
    if args.validate_only:
        print(json.dumps(dict(ready=True,configuration_verified=True,method=args.method,
                              admission_attempt=args.admission_attempt,submitted=False)))
        return
    job='b1k-composition-1022-'+args.method.replace('_','-')+'-s17'
    prefix='behavior1k/update-composition-20261022/trial-v2/'+args.method+'/'
    remote=AFS+'/update_composition_20261022/trial_v2/'+args.method
    work=HERE/'trial_submissions'/args.method;work.mkdir(parents=True,exist_ok=True)
    lock=(work/'submission.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    files={name:HERE/name for name in ('core.py','native.py','lora.py','closed_loop.py',
            'streaming_gradient.py','train.py','simulator_backend.py','admission.py')}
    for name in ('vla_data.py','vla_runtime.py'):
        files[name]=ROOT/'analysis/action_support_20261021'/name
    for name in ('common.py','simulator.py','helpers.py'):
        files[name]=ROOT/'analysis/rlinf_adaptation_20261020/cloud_smoke'/name
    for path in sorted((ROOT/'BEHAVIOR-VLAs/rlinf_adapter/b1k_rlinf').glob('*.py')):
        files['adapter/b1k_rlinf/'+path.name]=path
    for path in sorted((HERE/'closed_loop_pilot/data').iterdir()):
        files['data/'+path.name]=path
    compact={k:admission[k] for k in ('passed','adaptation_effective','worker_returncode','plan_sha256',
              'original_checkpoint','inner_optimizer','inner_loss_before','inner_loss_after',
              'inner_common_noise_action_max_change')}
    compact['source_result_sha256']=sha(source_result)
    compact['verified_configuration']=proof['verified_configuration']
    write(work/'admission.json',compact);files['admission.json']=work/'admission.json'
    old=json.loads((ROOT/'analysis/rlinf_adaptation_20261020/cloud_smoke/plan.json').read_text())
    plan.update(admission_result=remote+'/admission.json',data_directory=remote+'/data',
                trial_status='frozen',source_sha256={k:sha(p) for k,p in files.items()},
                original_source_sha256=old['original_source_sha256'],
                upstream_sha256=accepted['upstream_sha256'])
    plan['environment']['simulator_common']=remote+'/common.py'
    plan['max_seconds']=82000
    plan_path=work/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:
        raise ValueError('Frozen trial inputs changed; no silent replacement')
    write(plan_path,plan)
    archive=work/'bundle.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:
            for name,path in files.items():tar.add(path,arcname=name,recursive=False)
            tar.add(plan_path,arcname='plan.json',recursive=False)
    if not args.submit:
        print(json.dumps(dict(ready=True,method=args.method,job_name=job,bundle_bytes=archive.stat().st_size,
                              training_episodes=24,max_gpu_hours=24)));return
    intent_path=work/'submission.json'
    if intent_path.exists() and json.loads(intent_path.read_text()).get('submission_attempted'):
        print(json.dumps(dict(already_attempted=True,**json.loads(intent_path.read_text()))));return
    template='b1k-repeat1688-1019-batch-00'
    response=cli('acp','jobs','describe',template,'--workspace-name=share-space','-o','json')
    if response.returncode:raise RuntimeError('Template unavailable')
    template_info=json.loads(response.stdout)
    assert template_info['uid']=='168d5223752b415caf67bd27ed504a83'
    assert template_info['ownership']['user_name']=='260010165'
    assert template_info['resource_pool']['name']=='computing-cluster-5090-01g'
    intent=dict(job_name=job,method=args.method,submission_attempted=False,
                bundle_sha256=sha(archive),bundle_bytes=archive.stat().st_size,plan_sha256=sha(plan_path),
                gpu_count=1,max_seconds=86400,training_episodes=24,bucket='260010165',prefix=prefix)
    write(intent_path,intent)
    sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    client=aoss_client()
    try:
        key=prefix+'code-data-'+sha(archive)+'.tar.gz'
        client.upload_file(str(archive),'260010165',key,ExtraArgs={'Metadata':{'sha256':sha(archive)}})
        head=client.head_object(Bucket='260010165',Key=key)
        assert head['ContentLength']==archive.stat().st_size and head['Metadata']['sha256']==sha(archive)
        def signed(method,key):
            return client.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=259200)
        config=dict(root=remote,method=args.method,bundle_bytes=archive.stat().st_size,bundle_sha256=sha(archive),
            bundle_url=signed('get_object',key),progress_put=signed('put_object',prefix+'progress.json'),
            result_put=signed('put_object',prefix+'result.json'),checkpoint_put=signed('put_object',prefix+'adapter.pt'),
            upstream=AFS+'/rlinf_adapter_20261020/smoke01/upstream',site=AFS+'/rlinf_adapter_20261020/smoke01/site',
            python=AFS+'/runtime_20261007_1306/envs/b1k-pi05/bin/python')
        source=(HERE/'trial_bootstrap.py').read_text()+'\nrun('+repr(config)+')\n'
        packed=base64.b64encode(zlib.compress(source.encode())).decode()
        command='timeout --signal=TERM --kill-after=90s 86400s python3 -u -c '+shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        intent.update(submission_attempted=True,startup_sha256=hashlib.sha256(command.encode()).hexdigest())
        write(intent_path,intent)
        try:
            response=cli('acp','jobs','copy','--workspace-name=share-space','--copy-job-name='+template,
                '--name='+job,'--job-name='+job,'--worker-spec=n12lp.nn.i10a.1','--worker-nodes=1',
                '--quota-type=spot','--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
                '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--container-image-url='+template_info['roles'][0]['image_path'],'--command='+command,timeout=120)
            intent['submit_returncode']=response.returncode
        except subprocess.TimeoutExpired:intent['submit_outcome']='unknown_timeout'
        write(intent_path,intent)
        response=cli('acp','jobs','describe',job,'--workspace-name=share-space','-o','json')
        if response.returncode==0:
            value=json.loads(response.stdout);intent.update(uid=value['uid'],state=value['state'])
        else:intent['state']='requires_exact_name_query'
        write(intent_path,intent)
        print(json.dumps({k:intent.get(k) for k in ('job_name','uid','state','training_episodes','bundle_bytes')}))
    finally:client.close()


if __name__=='__main__':main()
