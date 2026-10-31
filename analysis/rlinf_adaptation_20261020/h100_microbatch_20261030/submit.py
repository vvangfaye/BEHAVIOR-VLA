"""Submit one uniquely named H100 benchmark; no automatic retries."""
import base64
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import zlib

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT/'BEHAVIOR-VLAs/sco'))
from public_test_admission import query_standard_capacity
from video_eval_adapter import aoss_client
from source_contract import original_training_source

CLI='/home/faye/.local/bin/sco'
ENV={k:v for k,v in os.environ.items() if k.lower() not in ('http_proxy','https_proxy','all_proxy')}


def call(*args, timeout=60):
    return subprocess.run([CLI,*args],env=ENV,capture_output=True,text=True,timeout=timeout)


def describe(name):
    response=call('acp','jobs','describe',name,'--workspace-name=share-space','-o','json')
    response.check_returncode()
    return json.loads(response.stdout)


def write(path, value):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def main():
    attempt=int(sys.argv[1]) if len(sys.argv)>1 else 1
    assert attempt in (1,2,3,4,5,6,7)
    path=HERE/('submission.json' if attempt==1 else f'submission_a{attempt}.json')
    previous_preparation=None
    if path.exists():
        previous_preparation=json.loads(path.read_text())
        if '--resume-preparation' not in sys.argv or previous_preparation.get('submission_attempted'):
            raise RuntimeError('Existing submission intent must be reconciled, never duplicated')
        assert previous_preparation['state']=='preparing'
        backup=HERE/f'submission_a{attempt}_preparation1.json'
        with backup.open('x') as handle:
            json.dump(previous_preparation,handle,indent=2)
    prior_runs=[]
    for prior_attempt in range(1, attempt):
        prior_path=HERE/('submission.json' if prior_attempt==1 else f'submission_a{prior_attempt}.json')
        old=json.loads(prior_path.read_text())
        actual=describe(old['job'])
        assert actual['uid']==old['uid'] and actual['ownership']['user_name']=='260010165'
        assert actual['state'] in ('FAILED','SUCCEEDED','STOPPED','TERMINATED','CANCELED','SUSPENDED')
        workers=call('acp','jobs','get-workers',old['job'],'--workspace-name=share-space')
        assert workers.returncode==0 and 'running' not in workers.stdout.lower()
        if actual['state']=='SUSPENDED':
            assert 'Deleted' in workers.stdout, 'Suspension alone does not prove writer termination'
        start=datetime.fromisoformat(actual['start_time'].replace('Z','+00:00'))
        end=datetime.fromisoformat(actual['complete_time'].replace('Z','+00:00')) if actual.get('complete_time') else datetime.now(timezone.utc)
        prior_runs.append({'job':old['job'],'uid':old['uid'],'state':actual['state'],
                          'allocated_seconds':math.ceil((end-start).total_seconds()),'old_writer_stopped':True,
                          'allocated_seconds_is_upper_bound':not bool(actual.get('complete_time'))})
    # The entire recovery chain remains within the original twelve GPU-hours.
    max_seconds=43200-sum(x['allocated_seconds'] for x in prior_runs)
    assert max_seconds>1800
    if attempt>1:
        write(HERE/f'recovery_a{attempt}.json',{'prior_runs':prior_runs,'total_budget_seconds':43200,
              'remaining_seconds':max_seconds,'max_live_gpus':1,
              'scope':'Existing data only; unchanged PPO loss/update equations, rewards, learning rates and final update-equivalence thresholds',
              'repair':'Validate cross-device replay in PPO probability-ratio units; preserve stored old logprobs; retain strict same-device batch-equivalence checks' if attempt>=4 else
                       'Restore original sampler cuDNN TF32 setting; audit before any update'})
    capacity=query_standard_capacity(CLI,ENV)
    write(HERE/'capacity.json',capacity)
    quota='reserved' if capacity['used_gpus'] < capacity['standard_limit'] else 'spot'
    template=describe('b1k-update-composition-1022-plumbing06h100')
    assert template['uid']=='14848bdf49374eab8bf6fc6737f8e3cf'
    assert template['ownership']['user_name']=='260010165'
    assert template['resource_pool']['name']=='share-cluster'
    assert template['roles'][0]['resource_spec'][0]['name']=='n6ls.iu.i40.1'
    source=HERE.parent/'modem_parallel_20261029'
    frozen=json.loads((source/'execution_manifest.json').read_text())['source_sha256']
    for name in ['training.py','common.py']+[f'b1k_rlinf/{p.name}' for p in (HERE/'run/b1k_rlinf').glob('*.py')]:
        content=(HERE/'run'/name).read_bytes()
        if name=='training.py':
            content=original_training_source(content.decode()).encode()
        assert hashlib.sha256(content).hexdigest()==frozen[name], name
    bundle=io.BytesIO()
    unused_adapter_files={'export.py','launch.py','prepare.py','reference_check.py','verify.py'}
    with tarfile.open(fileobj=bundle,mode='w:gz',compresslevel=9) as archive:
        for p in sorted((HERE/'run').rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                if p.parent.name=='b1k_rlinf' and p.name in unused_adapter_files:
                    continue
                data=p.read_bytes()
                if p.name=='input_manifest.json':
                    # Runtime only uses receipt identities here. Download sizes
                    # and checksums are already embedded in the bootstrap.
                    data=json.dumps({'receipts':json.loads(data)['receipts']},separators=(',',':')).encode()
                item=tarfile.TarInfo('run/'+str(p.relative_to(HERE/'run')))
                item.size=len(data);item.mtime=0
                archive.addfile(item,io.BytesIO(data))
    raw=bundle.getvalue()
    (HERE/('bundle.tar.gz' if attempt==1 else f'bundle_a{attempt}.tar.gz')).write_bytes(raw)
    manifest=json.loads((HERE/'inputs.json').read_text())
    job=f'b1k-h100-microbatch-1030-a{attempt}'
    prefix=f'behavior1k/h100-microbatch-20261030/a{attempt}/'
    root=f'/mnt/afs/260010165/behavior_1k/h100_microbatch_20261030/a{attempt}'
    receipt={'job':job,'attempt':attempt,'gpu_count':1,'max_seconds':max_seconds,'quota':quota,'resource_spec':'n6ls.iu.i40.1',
             'template_uid':template['uid'],'bucket':'260010165','aoss_prefix':prefix,'output':root,
             'bundle_bytes':len(raw),'bundle_sha256':hashlib.sha256(raw).hexdigest(),
             'input_bytes':sum(x['size'] for x in manifest['files']),
             'created_at':datetime.now(timezone.utc).isoformat(),'state':'preparing','submission_attempted':False}
    if previous_preparation is None:
        with path.open('x') as f:
            json.dump(receipt,f,indent=2)
    else:
        assert previous_preparation['job']==job
        receipt['preparation_recovery']='Inline startup exceeded 64KB before any create call; resume same unsubmitted intent using signed bundle download'
        write(path,receipt)
    client=aoss_client()
    try:
        for item in manifest['files']:
            head=client.head_object(Bucket='260010165',Key=item['key'])
            assert head['ContentLength']==item['size'],item['relative']
        client.put_object(Bucket='260010165',Key=prefix+'bundle.tar.gz',Body=raw)
        assert client.get_object(Bucket='260010165',Key=prefix+'bundle.tar.gz')['Body'].read()==raw
        # Sign against the internal endpoint itself, not by editing signed URLs.
        import boto3
        from botocore.config import Config
        credentials=client._request_signer._credentials.get_frozen_credentials()
        internal=boto3.client('s3',endpoint_url='https://aoss-internal.cn-sh-01b.sensecoreapi-oss.cn',
            region_name='cn-sh-01b',aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,aws_session_token=credentials.token,
            config=Config(signature_version='s3v4',s3={'addressing_style':'path'},proxies={}))
        def signed(method,key):
            return internal.generate_presigned_url(method,Params={'Bucket':'260010165','Key':key},ExpiresIn=432000)
        config={'root':root,'bundle_inline':base64.b64encode(raw).decode(),
                'bundle_sha256':receipt['bundle_sha256'],'max_seconds':max_seconds,
                'input_bytes':receipt['input_bytes'],
                'inputs':[{k:x[k] for k in ('relative','size','sha256')}|{'url':signed('get_object',x['key'])} for x in manifest['files']],
                'status_put':signed('put_object',prefix+'status.json'),
                'result_put':signed('put_object',prefix+'result.tar.gz')}
        bootstrap=(HERE/'bootstrap.py').read_text()+'\nraise SystemExit(run('+repr(config)+'))\n'
        packed=base64.b64encode(zlib.compress(bootstrap.encode(),9)).decode()
        command=f'timeout --signal=TERM --kill-after=60s {max_seconds-60}s python3 -u -c '+shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        assert len(command)<64000,len(command)
        receipt.update(state='submitting',submission_attempted=True,
                       command_sha256=hashlib.sha256(command.encode()).hexdigest())
        write(path,receipt)
        try:
            response=call('acp','jobs','copy','--workspace-name=share-space',
                '--copy-job-name=b1k-update-composition-1022-plumbing06h100',
                '--name='+job,'--job-name='+job,'--aec2-name=share-cluster',
                '--worker-spec=n6ls.iu.i40.1','--worker-nodes=1','--quota-type='+quota,
                '--retry-times=0','--enable-fault-tolerance=false','--enable-anomaly-detection=false',
                '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                '--container-image-url='+template['roles'][0]['image_path'],'--command='+command,timeout=120)
            receipt['returncode']=response.returncode
            if response.returncode:
                # Avoid exposing signed inline startup payloads in errors.
                receipt['error']='SCO returned nonzero; inspect exact job before any retry'
        except subprocess.TimeoutExpired:
            receipt['returncode']='unknown_timeout'
        write(path,receipt)
        actual=describe(job)
        assert actual['ownership']['user_name']=='260010165'
        assert actual['resource_pool']['name']=='share-cluster'
        assert actual['scheduling']['quota_type']==quota.upper()
        assert hashlib.sha256(actual['roles'][0]['startup_script'].encode()).hexdigest()==receipt['command_sha256']
        receipt.update(uid=actual['uid'],state=actual['state'])
        write(path,receipt)
        write(HERE/'active_submission.json',receipt)
        print(json.dumps({k:receipt[k] for k in ['job','uid','state','quota','gpu_count','max_seconds','input_bytes','bundle_bytes']},indent=2))
    finally:
        client.close()


if __name__=='__main__':
    main()
