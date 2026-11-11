"""Freeze one matched training arm, then submit one H100 and one RTX worker.

Neither role is retried here. Unknown submission outcomes must be resolved by
the exact job name before a later invocation can proceed.
"""
import argparse
import ast
import base64
import fcntl
import json
from pathlib import Path
import re
import shlex
import sys
import tarfile
import zlib

from admission import validate_trial_configuration, verified_configuration
from submit import cli, sha, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AFS = '/mnt/afs/260010165/behavior_1k'
METHODS = ('meta', 'maml', 'fixed_crossplay', 'rl_bc', 'before_after')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--run-id',default='')
    parser.add_argument('--role',choices=('both','renderer','trainer'),default='both')
    parser.add_argument('--prewarm',action='store_true')
    parser.add_argument('--ready-proof-run',default='')
    parser.add_argument('--quota-type',choices=('spot','reserved'),default='spot')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.run_id and not re.fullmatch(r'r[0-9]{2}',args.run_id):
        raise ValueError('Expected an explicit run revision such as r02')
    if args.ready_proof_run and not re.fullmatch(r'r[0-9]{2}',args.ready_proof_run):
        raise ValueError('Invalid readiness proof run')
    source = HERE/'full_checkpoint_06h100'
    admission = json.loads((source/'result.json').read_text())
    accepted = json.loads((source/'plan.json').read_text())
    if admission['plan_sha256'] != sha(source/'plan.json'):
        raise ValueError('Admission does not match the frozen verification')
    admission['verified_configuration'] = verified_configuration(accepted)
    plan = json.loads((HERE/'closed_loop_pilot_v2'/f'{args.method}.json').read_text())
    validate_trial_configuration(plan, admission)
    if len(plan['iterations'])*2 != 24 or plan['max_training_episodes'] != 24:
        raise ValueError('Unexpected training budget')
    work = HERE/'split_trial_submissions'/args.method
    if args.run_id:work=work/args.run_id
    work.mkdir(parents=True, exist_ok=True)
    lock = (work/'submission.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    remote = AFS+'/update_composition_20261022/trial_h100/'+args.method
    if args.run_id:remote+='/'+args.run_id
    files = {name: HERE/name for name in (
        'core.py', 'native.py', 'lora.py', 'closed_loop.py', 'streaming_gradient.py',
        'train.py', 'simulator_backend.py', 'admission.py', 'renderer_service.py',
        'renderer_mailbox.py')}
    for name in ('vla_data.py', 'vla_runtime.py'):
        files[name] = ROOT/'analysis/action_support_20261021'/name
    for name in ('common.py', 'simulator.py', 'helpers.py', 'observation_text.py', 'step_recording.py'):
        files[name] = HERE/'simulation_bridge'/name
    for path in sorted((ROOT/'BEHAVIOR-VLAs/rlinf_adapter/b1k_rlinf').glob('*.py')):
        files['adapter/b1k_rlinf/'+path.name] = path
    for path in sorted((HERE/'closed_loop_pilot/data').iterdir()):
        files['data/'+path.name] = path
    compact = {key: admission[key] for key in (
        'passed', 'adaptation_effective', 'worker_returncode', 'plan_sha256',
        'original_checkpoint', 'inner_optimizer', 'inner_loss_before',
        'inner_loss_after', 'inner_common_noise_action_max_change', 'verified_configuration')}
    compact['source_result_sha256'] = sha(source/'result.json')
    if (work/'admission.json').exists() and json.loads((work/'admission.json').read_text()) != compact:
        raise ValueError('Frozen admission changed')
    write(work/'admission.json', compact)
    files['admission.json'] = work/'admission.json'
    old = json.loads((ROOT/'analysis/rlinf_adaptation_20261020/cloud_smoke/plan.json').read_text())
    plan.update(admission_result=remote+'/trainer/admission.json',
        data_directory=remote+'/trainer/data', max_seconds=82000, trial_status='frozen',
        source_sha256={name: sha(path) for name, path in files.items()},
        original_source_sha256=old['original_source_sha256'],
        upstream_sha256=accepted['upstream_sha256'],
        resource=dict(trainer_gpu='H100 80GB', renderer_gpu='RTX 5090',
                      gpus_per_role=1, max_seconds_per_role=86400, quota=args.quota_type))
    plan['environment']['simulator_common'] = remote+'/trainer/common.py'
    registry=ROOT/'BEHAVIOR-VLAs/openpi/src/openpi/configs/tasks/b1k.py'
    tree=ast.parse(registry.read_text())
    prompts=ast.literal_eval(next(node.value for node in tree.body
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TASKS' for t in node.targets)))
    samples=json.loads((HERE/'closed_loop_pilot/data/data_plan.json').read_text())['samples']
    for sample in samples:
        if sample['prompt']!=prompts[sample['task_name']]:
            raise ValueError('Cached demonstration differs from the evaluation-server task prompt')
    plan['environment']['task_prompts']={name:prompts[name] for name in sorted({s['task_name'] for s in samples})}
    plan['environment']['task_prompt_source_sha256']=sha(registry)
    plan['environment']['simulator_common_sha256']=sha(files['common.py'])
    plan['environment']['renderer_mailbox'] = dict(directory=remote+'/messages',
        session='b1k-composition-'+args.method+'-s17-v1'+('-'+args.run_id if args.run_id else ''), idle_timeout_seconds=3600,
        episode_output_root=remote+('/episodes' if args.prewarm else '/trainer/output'))
    if args.prewarm:plan['environment']['renderer_mailbox']['prewarm_first']=True
    if args.run_id:plan['execution_revision']=args.run_id
    if args.ready_proof_run:
        proof_root=HERE/'split_trial_submissions'/args.method/args.ready_proof_run
        proof_plan=json.loads((proof_root/'plan.json').read_text())
        proof=json.loads((proof_root/'renderer/progress.json').read_text())
        if proof.get('stage')!='prewarmed' or proof.get('session')!=proof_plan['environment']['renderer_mailbox']['session']:
            raise ValueError('Earlier run did not return a verified initial observation')
        for name in ('common.py','simulator.py','helpers.py','observation_text.py'):
            if proof_plan['source_sha256'][name]!=plan['source_sha256'][name]:
                raise ValueError('Readiness proof uses a different simulator bridge')
        if proof_plan['original_source_sha256']!=plan['original_source_sha256'] or proof_plan['iterations'][0]!=plan['iterations'][0]:
            raise ValueError('Readiness proof does not match the first planned environment')
        if proof_plan['environment']['task_prompts']!=plan['environment']['task_prompts']:
            raise ValueError('Readiness proof uses different task instructions')
        plan['initial_observation_proof']=dict(run=args.ready_proof_run,progress_sha256=sha(proof_root/'renderer/progress.json'),
            plan_sha256=sha(proof_root/'plan.json'),stage='prewarmed')
    plan_path = work/'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Frozen trial changed; do not replace it silently')
    write(plan_path, plan)
    archive = work/'bundle.tar.gz'
    if not archive.exists():
        with tarfile.open(archive, 'w:gz') as tar:
            for name, path in files.items():
                tar.add(path, arcname=name, recursive=False)
            tar.add(plan_path, arcname='plan.json', recursive=False)
    with tarfile.open(archive) as tar:
        import hashlib
        for name, path in files.items():
            if hashlib.sha256(tar.extractfile(name).read()).hexdigest() != sha(path):
                raise ValueError('Frozen bundle no longer matches sources')
    if not args.submit:
        print(json.dumps(dict(ready=True, method=args.method, training_episodes=24,
                              workers=2, max_gpu_hours=48, bundle_bytes=archive.stat().st_size)))
        return
    template = 'b1k-repeat1688-1019-batch-00'
    response = cli('acp', 'jobs', 'describe', template, '--workspace-name=share-space', '-o', 'json')
    response.check_returncode()
    info = json.loads(response.stdout)
    assert info['uid'] == '168d5223752b415caf67bd27ed504a83'
    assert info['ownership']['user_name'] == '260010165'
    assert info['resource_pool']['name'] == 'computing-cluster-5090-01g'
    sys.path.insert(0, str(ROOT/'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    client = aoss_client()
    prefix = 'behavior1k/update-composition-20261022/trial-h100/'+args.method+'/'
    if args.run_id:prefix+=args.run_id+'/'
    try:
        key = prefix+'code-data-'+sha(archive)+'.tar.gz'
        client.upload_file(str(archive), '260010165', key,
                           ExtraArgs={'Metadata': {'sha256': sha(archive)}})
        head = client.head_object(Bucket='260010165', Key=key)
        assert head['ContentLength'] == archive.stat().st_size
        assert head['Metadata']['sha256'] == sha(archive)
        def signed(method, path):
            return client.generate_presigned_url(method,
                Params={'Bucket': '260010165', 'Key': path}, ExpiresIn=259200)
        roles=('renderer','trainer') if args.role=='both' else (args.role,)
        for role in roles:
            job = 'b1k-comp1022-'+args.method.replace('_', '-')+'-'+role+'-s17'+('-'+args.run_id if args.run_id else '')
            role_dir = work/role
            role_dir.mkdir(exist_ok=True)
            intent_path = role_dir/'submission.json'
            if intent_path.exists() and json.loads(intent_path.read_text()).get('submission_attempted'):
                previous = json.loads(intent_path.read_text())
                if not previous.get('uid'):
                    raise RuntimeError('Resolve the previous exact-name submission before proceeding: '+job)
                print(json.dumps(dict(already_submitted=True, role=role, uid=previous['uid'])))
                continue
            if args.quota_type=='reserved':
                from resource_admission import check_standard_capacity
                check_standard_capacity(role_dir/'standard_admission.json')
            if role=='trainer' and args.prewarm:
                try:
                    response=client.get_object(Bucket='260010165',Key=prefix+'renderer/progress.json')
                except client.exceptions.NoSuchKey:
                    print(json.dumps(dict(submitted=False,role='trainer',reason='renderer_not_ready')))
                    return
                try:progress=json.loads(response['Body'].read(128*1024))
                finally:response['Body'].close()
                if progress.get('stage')!='prewarmed' or progress.get('session')!=plan['environment']['renderer_mailbox']['session']:
                    print(json.dumps(dict(submitted=False,role='trainer',reason='renderer_not_ready',stage=progress.get('stage'))))
                    return
                prior=json.loads((work/'renderer/submission.json').read_text())
                check=cli('acp','jobs','describe',prior['job_name'],'--workspace-name=share-space','-o','json')
                check.check_returncode();running=json.loads(check.stdout)
                if running['uid']!=prior['uid'] or running['state']!='RUNNING':
                    raise RuntimeError('Prewarmed renderer is not confirmed running')
                write(work/'renderer_ready_gate.json',dict(uid=prior['uid'],stage=progress['stage'],
                    session=progress['session'],plan_sha256=sha(plan_path)))
            config = dict(root=remote+'/'+role, role=role, remote_renderer=True, method=args.method,
                bundle_bytes=archive.stat().st_size, bundle_sha256=sha(archive),
                bundle_url=signed('get_object', key), progress_put=signed('put_object', prefix+role+'/progress.json'),
                result_put=signed('put_object', prefix+role+'/result.json'),
                checkpoint_put=signed('put_object', prefix+role+'/adapter.pt'),
                upstream=AFS+'/rlinf_adapter_20261020/smoke01/upstream',
                site=AFS+'/rlinf_adapter_20261020/smoke01/site',
                python=AFS+'/runtime_20261007_1306/envs/b1k-pi05/bin/python')
            source_text = (HERE/'trial_bootstrap.py').read_text()+'\nrun('+repr(config)+')\n'
            packed = base64.b64encode(zlib.compress(source_text.encode())).decode()
            command = 'timeout --signal=TERM --kill-after=90s 86400s python3 -u -c '+shlex.quote(
                'import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
            intent = dict(job_name=job, role=role, method=args.method, submission_attempted=True,
                bundle_sha256=sha(archive), bundle_bytes=archive.stat().st_size,
                plan_sha256=sha(plan_path), gpu_count=1, max_seconds=86400,
                bucket='260010165', prefix=prefix+role+'/',quota_type=args.quota_type,
                startup_sha256=hashlib.sha256(command.encode()).hexdigest())
            write(intent_path, intent)
            resource = ['--worker-spec=n6ls.iu.i40.1', '--aec2-name=share-cluster'] if role=='trainer' else ['--worker-spec=n12lp.nn.i10a.1']
            try:
                response = cli('acp', 'jobs', 'copy', '--workspace-name=share-space', '--copy-job-name='+template,
                    '--name='+job, '--job-name='+job, '--worker-nodes=1', '--quota-type='+args.quota_type,
                    '--retry-times=0', '--enable-fault-tolerance=false', '--enable-anomaly-detection=false',
                    '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
                    '--container-image-url='+info['roles'][0]['image_path'], '--command='+command,
                    *resource, timeout=120)
                intent['submit_returncode'] = response.returncode
            except Exception as error:
                intent['submit_outcome'] = 'unknown_'+type(error).__name__
            write(intent_path, intent)
            observed = cli('acp', 'jobs', 'describe', job, '--workspace-name=share-space', '-o', 'json')
            if observed.returncode == 0:
                actual = json.loads(observed.stdout)
                intent.update(uid=actual['uid'], state=actual['state'])
            else:
                intent['state'] = 'requires_exact_name_query'
            write(intent_path, intent)
            print(json.dumps({k: intent.get(k) for k in ('job_name', 'uid', 'state', 'role', 'submit_returncode')}), flush=True)
            if 'uid' not in intent:
                raise RuntimeError('Submission outcome unresolved; do not submit another role')
    finally:
        client.close()


if __name__ == '__main__':
    main()
