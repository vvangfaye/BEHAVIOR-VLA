"""Single bounded submission. No credentials or signed URLs in local receipts."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import zlib

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
TEMPLATE='b1k-repeat1688-1019-batch-00'
AFS='/mnt/afs/260010165/behavior_1k'


def write(path,value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def cli(*args,timeout=45):
    return subprocess.run(['/home/faye/.local/bin/sco',*args],capture_output=True,text=True,timeout=timeout)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--submit',action='store_true')
    parser.add_argument('--attempt',choices=['01','02','03','04','05','06','06h100'],default='01');args=parser.parse_args()
    migrated=args.attempt=='06h100'
    job_name='b1k-update-composition-1022-plumbing'+args.attempt
    prefix='behavior1k/update-composition-20261022/'+job_name+'/'
    work=HERE/('full_checkpoint_'+args.attempt);work.mkdir(exist_ok=True)
    files={name:HERE/name for name in ('core.py','native.py','lora.py','full_checkpoint.py')}
    for name in ('vla_runtime.py','vla_data.py'):
        files[name]=ROOT/'analysis/action_support_20261021'/name
    for source in sorted((ROOT/'BEHAVIOR-VLAs/rlinf_adapter/b1k_rlinf').glob('*.py')):
        files['adapter/b1k_rlinf/'+source.name]=source
    if args.attempt in ('02','03','04','05','06','06h100'):
        for source in sorted((HERE/'two_existing_observations').iterdir()):
            files['data/'+source.name]=source
    upstream=ROOT/'analysis/rlinf_adaptation_20261020/upstream'
    upstream_names=['rlinf/models/embodiment/openpi/'+p for p in
                    ('pi0.py','tasks/rl.py','modules/gemma.py','modules/model.py','sampling/rl_sampler.py')]
    plan=dict(checkpoint=AFS+'/rlinf_adapter_20261020/smoke01/checkpoint',
        tokenizer=AFS+'/runtime_20261007_1306/openpi-cache/big_vision/paligemma_tokenizer.model',
        data_directory=(AFS+'/action_support_20261021/data-v1' if args.attempt=='01' else
                        AFS+'/update_composition_20261022/plumbing'+args.attempt+'/data'),
        precision='bf16',rank=4,seed=17,inner_lr=.01,outer_lr=.001,
        source_sha256={k:sha(v) for k,v in files.items()},
        upstream_sha256={k:sha(upstream/k) for k in upstream_names},
        resource=dict(gpus=1,worker_spec='n12lp.nn.i10a.1',quota='spot',max_seconds=3600),
        scope='One existing checkpoint; two cached expert observations; synthetic outer weights; no simulator or new data',
        rules='2026-09-11 snapshot; live source inaccessible; one GPU-hour cap')
    if args.attempt in ('04','05','06','06h100'):
        plan.update(inner_optimizer='adam',inner_lr=.001 if args.attempt=='04' else .0001,inner_epsilon=1e-6,
                    purpose='Verify that the differentiable adaptation step is numerically effective in the actual BF16 model',
                    require_effective_adaptation=True)
    if args.attempt in ('06','06h100'):
        plan.update(precision='fp32',check_noop=True,
                    purpose='Check exact-value parameter replacement and effective adaptation under full FP32 computation')
    if migrated:
        original=json.loads((HERE/'full_checkpoint_06/plan.json').read_text())
        assert plan['source_sha256']==original['source_sha256'], 'Migration must use identical source and data'
        plan['resource'].update(worker_spec='n6ls.iu.i40.1',cluster='share-cluster')
        plan['migration']=dict(source_attempt='06',source_plan_sha256=sha(HERE/'full_checkpoint_06/plan.json'),
            reason='User requested H100; original job still pending without an assigned worker',
            unchanged='Code, data, optimizer, precision, seed, thresholds and one-hour cap')
    plan_path=work/'plan.json'
    if plan_path.exists():
        if json.loads(plan_path.read_text())!=plan:raise ValueError('Frozen plan differs')
    else:write(plan_path,plan)
    archive=work/'bundle.tar.gz'
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:
            for name,source in files.items():tar.add(source,arcname=name,recursive=False)
            tar.add(plan_path,arcname='plan.json',recursive=False)
    if not args.submit:
        print(json.dumps(dict(job=job_name,bundle_bytes=archive.stat().st_size,plan_sha256=sha(plan_path))))
        return
    if migrated:
        prior=json.loads((HERE/'full_checkpoint_06/submission.json').read_text())
        checked=cli('acp','jobs','describe',prior['job_name'],'--workspace-name=share-space','-o','json')
        if checked.returncode:raise RuntimeError('Cannot verify original job is stopped')
        prior_state=json.loads(checked.stdout)
        if prior_state['uid']!=prior['uid'] or prior_state['state'] not in ('SUSPENDED','STOPPED','CANCELLED'):
            raise RuntimeError('H100 migration requires the original job to be confirmed stopped')
    intent_path=work/'submission.json'
    existing=None
    if intent_path.exists():
        existing=json.loads(intent_path.read_text())
        if existing.get('submission_attempted'):
            print(json.dumps(dict(already_attempted=True,**existing)));return
        if existing['plan_sha256']!=sha(plan_path) or existing['bundle_sha256']!=sha(archive):
            raise ValueError('Cannot resume preflight with changed inputs')
    response=cli('acp','jobs','describe',TEMPLATE,'--workspace-name=share-space','-o','json')
    if response.returncode:raise RuntimeError('Template read failed')
    template=json.loads(response.stdout)
    assert template['uid']=='168d5223752b415caf67bd27ed504a83'
    assert template['ownership']['user_name']=='260010165'
    assert template['resource_pool']['name']=='computing-cluster-5090-01g'
    image=template['roles'][0]['image_path']
    intent=dict(job_name=job_name,created_at=datetime.now(timezone.utc).isoformat(),
                plan_sha256=sha(plan_path),bundle_sha256=sha(archive),bundle_bytes=archive.stat().st_size,
                gpu_count=1,max_seconds=3600,submission_attempted=False,
                bucket='260010165',prefix=prefix)
    if existing is None:
        with intent_path.open('x') as stream:json.dump(intent,stream,indent=2)
    else:
        intent['created_at']=existing['created_at']
        intent['preflight_resume_reason']='Previous interpreter lacked boto3; no upload or copy attempted'
        write(intent_path,intent)
    sys.path.insert(0,str(ROOT/'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    client=aoss_client()
    try:
        key=prefix+'code-'+sha(archive)+'.tar.gz'
        client.upload_file(str(archive),'260010165',key,ExtraArgs={'Metadata':{'sha256':sha(archive)}})
        head=client.head_object(Bucket='260010165',Key=key)
        assert head['ContentLength']==archive.stat().st_size and head['Metadata']['sha256']==sha(archive)
        def signed(method,key):
            return client.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=86400)
        config=dict(root=AFS+'/update_composition_20261022/plumbing'+args.attempt,
            bundle_sha256=sha(archive),bundle_url=signed('get_object',key),
            progress_put=signed('put_object',prefix+'progress.json'),
            result_put=signed('put_object',prefix+'result.json'),
            upstream=AFS+'/rlinf_adapter_20261020/smoke01/upstream',
            site=AFS+'/rlinf_adapter_20261020/smoke01/site',
            python=AFS+'/runtime_20261007_1306/envs/b1k-pi05/bin/python')
        if migrated:config['record_gpu']=True
        source=(HERE/'bootstrap.py').read_text()+'\nrun('+repr(config)+')\n'
        packed=base64.b64encode(zlib.compress(source.encode())).decode()
        script='import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
        command='timeout --signal=TERM --kill-after=30s 3540s python3 -u -c '+shlex.quote(script)
        intent.update(submission_attempted=True,command_sha256=hashlib.sha256(command.encode()).hexdigest())
        write(intent_path,intent)
        try:
            resource_flags=['--aec2-name=share-cluster'] if migrated else []
            response=cli('acp','jobs','copy','--workspace-name=share-space','--copy-job-name='+TEMPLATE,
                '--name='+job_name,'--job-name='+job_name,'--worker-spec='+plan['resource']['worker_spec'],'--worker-nodes=1',
                '--quota-type=spot','--retry-times=0','--enable-fault-tolerance=false',
                '--enable-anomaly-detection=false','--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--container-image-url='+image,'--command='+command,*resource_flags,timeout=120)
            intent['submit_returncode']=response.returncode
        except subprocess.TimeoutExpired:
            intent['submit_outcome']='unknown_timeout'
        write(intent_path,intent)
        response=cli('acp','jobs','describe',job_name,'--workspace-name=share-space','-o','json')
        if response.returncode==0:
            job=json.loads(response.stdout);intent.update(uid=job['uid'],state=job['state'])
        else:intent['state']='requires_exact_name_query'
        write(intent_path,intent)
        print(json.dumps({k:intent.get(k) for k in ('job_name','uid','state','submit_returncode','bundle_bytes')}))
    finally:client.close()


if __name__=='__main__':main()
