"""Finite five-job replacement for the same fixed20 checkpoint5000 panel."""
from concurrent.futures import ThreadPoolExecutor
import base64
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import zlib

ROOT = Path(__file__).resolve().parent
OLD = ROOT.parent / 'paired_fixed20_step5000'
spec = importlib.util.spec_from_file_location('single_episode_controller', OLD / 'controller.py')
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
base.ROOT = ROOT
load, save, new, sha_bytes = base.load, base.save, base.new, base.sha_bytes
observe, CLI, environment, aoss_client = base.observe, base.CLI, base.environment, base.aoss_client
TERMINAL = base.TERMINAL


def validate_plan(plan):
    indices = [i for g in plan['batch_groups'] for i in g['indices']]
    if (len(plan['batch_groups']) != 5 or any(len(g['indices']) != 4 for g in plan['batch_groups'])
            or sorted(indices) != list(range(20)) or len({tuple(p) for p in plan['pairs']}) != 20
            or plan['max_active_gpu_jobs'] != 5 or plan['max_pending'] != 5
            or plan['candidate_step'] != 5000 or plan['training_target_unchanged'] != 5000
            or plan['modify_or_stop_training'] or plan['automatic_failure_retry']
            or plan['job_walltime_seconds'] > 128 * 3600
            or plan['job_walltime_seconds'] < 4 * plan['member_walltime_seconds']):
        raise ValueError('Plan differs from five jobs x four unique episodes')
    if sha_bytes((OLD / 'controller.py').read_bytes()) != plan['base_controller_sha256']:
        raise ValueError('Reused single-episode controller changed')
    expected = load(ROOT / 'frozen_sha256.json')
    if expected != {p.name: sha_bytes(p.read_bytes()) for p in (ROOT / 'frozen').glob('*.py')}:
        raise ValueError('Frozen source changed')
    predecessor=Path(plan['predecessor_root'])
    old_launch=load(predecessor/'controller_launch.json')
    proc=subprocess.run(['ps','-p',str(old_launch['pid']),'-o','command='],capture_output=True,text=True)
    if proc.returncode==0 and str(predecessor/'controller.py') in proc.stdout:
        raise ValueError('Prior five-job evaluator still active')
    evidence=load(ROOT/'predecessor_closed.json')
    if not evidence['all_terminal'] or evidence.get('job_count')!=5:
        raise ValueError('Verify all five predecessor jobs terminal before final5000')
    if not plan.get('final_checkpoint_restore_verified'):
        raise ValueError('Exact final5000 training-state restore proof is required')
    export = load(OLD / 'reports/export.json')
    if export['state'] != 'accepted' or export['manifest_sha256'] != plan['expected_checkpoint_manifest_sha256']:
        raise ValueError('Reusable export not accepted')


def get_report(client, plan, id):
    from botocore.exceptions import ClientError
    try:
        response = client.get_object(Bucket='260010165', Key=plan['object_prefix'] + '/reports/' + id + '.json')
        try:
            result = json.loads(response['Body'].read(16 * 1024**2))
        finally:
            response['Body'].close()
    except ClientError as exc:
        if exc.response['Error']['Code'] in ('NoSuchKey', '404'):
            return None
        raise
    if (result.get('id') != id or result.get('protocol_sha256') != plan['protocol_sha256']
            or result.get('worker_bundle_sha256') != sha_bytes((ROOT / 'frozen_sha256.json').read_bytes())):
        raise ValueError('Report identity changed: ' + id)
    save(ROOT / 'reports' / (id + '.json'), result)
    return result


def bootstrap(client, plan, group, name):
    bundle = sha_bytes((ROOT / 'frozen_sha256.json').read_bytes())
    members = []
    def url(key):
        return client.generate_presigned_url('put_object', Params={'Bucket':'260010165', 'Key':key}, ExpiresIn=172800)
    for index in group['indices']:
        id = f'candidate-{index:02d}'
        report_url = url(plan['object_prefix'] + '/reports/' + id + '.json')
        payload = dict(id=id, mode='evaluate', job_name=name + '-' + id, role='candidate',
                       pair=plan['pairs'][index], plan=plan, worker_bundle_sha256=bundle,
                       heartbeat_put_url=url(plan['object_prefix'] + '/telemetry/' + id + '/heartbeat.json'),
                       diagnostics_put_url=url(plan['object_prefix'] + '/telemetry/' + id + '/diagnostics.json'))
        source = base.bootstrap(plan, payload, report_url)
        members.append(dict(id=id, source=base64.b64encode(zlib.compress(source.encode(), 9)).decode(),
                            sha256=sha_bytes(source.encode()), report_url=report_url,
                            report_path=plan['afs_root'] + '/workers/' + id + '/report.json'))
    config = dict(id=group['id'], members=members, root=plan['afs_root'] + '/batches/' + group['id'],
                  python=plan['runtime'] + '/envs/b1k-eval/bin/python',
                  member_walltime_seconds=plan['member_walltime_seconds'],
                  protocol_sha256=plan['protocol_sha256'], worker_bundle_sha256=bundle,
                  report_url=url(plan['object_prefix'] + '/reports/' + group['id'] + '.json'))
    return (ROOT / 'frozen/batch_runner.py').read_text() + '\nraise SystemExit(run(' + repr(config) + '))\n'


def bootstrap_loader(url, digest, destination):
    # Four encoded episode payloads can exceed Linux's single-argument limit.
    # Stage the private signed bootstrap once and verify its exact bytes remotely.
    return ("import hashlib,urllib.request;from pathlib import Path;"
            "o=urllib.request.build_opener(urllib.request.ProxyHandler({}));"
            "b=o.open(" + repr(url) + ",timeout=90).read(4194304);"
            "assert hashlib.sha256(b).hexdigest()==" + repr(digest) + ";"
            "p=Path(" + repr(destination) + ");p.parent.mkdir(parents=True,exist_ok=True);"
            "p.open('xb').write(b);p.chmod(0o600);exec(compile(b,str(p),'exec'))")


def submit(client, plan, group):
    id = group['id']; path = ROOT / 'intents' / (id + '.json')
    if path.exists() or (ROOT / 'STOP').exists() or time.time() >= plan['deadline_epoch']:
        raise ValueError('Existing intent or admission stopped; never duplicate')
    name = 'b1k-qbal5k-1016-5x4-' + id
    startup = bootstrap(client, plan, group, name)
    private = ROOT / 'private' / (id + '.startup.py')
    private.write_text(startup); os.chmod(private, 0o600)
    intent = dict(id=id, name=name, uid=None, gpu=1, image=plan['image'], indices=group['indices'],
                  mode='four_sequential_episodes', creation_attempted=False, attempts=1,
                  created_intent_at=time.time(), startup_sha256=sha_bytes(startup.encode()),
                  worker_bundle_sha256=sha_bytes((ROOT / 'frozen_sha256.json').read_bytes()))
    new(path, intent)
    key = plan['object_prefix'] + '/private-startup/' + id + '.py'
    # Immutable job intent exists before staging or calling the create API.
    client.put_object(Bucket='260010165', Key=key, Body=startup.encode(),
                      ContentType='text/x-python', Metadata={'sha256':intent['startup_sha256']})
    url = client.generate_presigned_url('get_object', Params={'Bucket':'260010165', 'Key':key}, ExpiresIn=172800)
    loader = bootstrap_loader(url, intent['startup_sha256'], plan['afs_root'] + '/launch/' + id + '/bootstrap.py')
    intent.update(startup_object_key=key, startup_object_staged=True)
    save(path, intent)
    command = ('timeout --signal=TERM --kill-after=60 ' + str(plan['job_walltime_seconds']) + ' ' +
               shlex.quote(plan['runtime'] + '/envs/b1k-eval/bin/python') + ' -c ' + shlex.quote(loader))
    if len(command.encode()) >= 100000:
        raise ValueError('Bootstrap exceeds safe command argument size; no create call')
    args = [CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
            '--copy-job-name=' + plan['gpu_copy_template'], '--name=' + name, '--job-name=' + name,
            '--worker-spec=n12lp.nn.i10a.1', '--worker-nodes=1', '--quota-type=spot', '--retry-times=0',
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
            '--container-image-url=' + plan['image'], '--env=NVIDIA_DRIVER_CAPABILITIES:all', '--command=' + command]
    intent.update(creation_attempted=True, creation_started_at=time.time(), state='creation_unknown')
    save(path, intent)
    try:
        cp = subprocess.run(args, env=environment(), capture_output=True, text=True, timeout=90)
        intent['cli_exit_code'] = cp.returncode
    except subprocess.TimeoutExpired:
        intent['cli_outcome'] = 'timeout_unknown'
    save(path, intent)
    obs = observe(intent)
    intent.update(uid=obs['uid'], state=obs['state'], verified_at=obs['observed_at'])
    save(path, intent)


def tick(client, plan):
    intents = [load(p) for p in sorted((ROOT / 'intents').glob('*.json'))]
    with ThreadPoolExecutor(max_workers=5) as pool:
        observations = list(pool.map(observe, intents))
    for intent, obs in zip(intents, observations):
        save(ROOT / 'observations' / (intent['id'] + '.json'), obs)
        if not intent.get('uid'):
            intent.update(uid=obs['uid']); save(ROOT / 'intents' / (intent['id'] + '.json'), intent)
    ids = [f'candidate-{i:02d}' for i in range(20)]
    with ThreadPoolExecutor(max_workers=6) as pool:
        reports = dict(zip(ids, pool.map(lambda id: get_report(client, plan, id), ids)))
    batch_reports = {i['id']: get_report(client, plan, i['id']) for i in intents}
    state = dict(observed_at=time.time(), state='five_jobs_four_samples_each',
                 created_gpu_jobs=len(intents), total_gpu_jobs=5, episodes_per_job=4,
                 running_gpu_jobs=sum(o['state']=='RUNNING' for o in observations),
                 pending_gpu_jobs=sum(o['state'] not in TERMINAL | {'RUNNING'} for o in observations),
                 active_gpu_jobs=sum(o['state'] not in TERMINAL for o in observations),
                 accepted_episodes=sum(bool(r and r['state']=='accepted') for r in reports.values()),
                 failed_episodes=[id for id,r in reports.items() if r and r['state']=='failed'],
                 training_iid_unchanged=239707, training_target_unchanged=5000)
    save(ROOT / 'status.json', state)
    if (ROOT / 'STOP').exists():
        state['state']='stopped_admission'; save(ROOT/'status.json',state); return True
    if len(intents)<5 and time.time()<plan['deadline_epoch']:
        existing={i['id'] for i in intents}
        for group in plan['batch_groups']:
            if group['id'] not in existing:
                submit(client, plan, group)
        return False
    if all(r and r['state']=='accepted' for r in reports.values()):
        base.finalize(plan, {id:dict(report=r) for id,r in reports.items()})
        state['state']='comparison_complete'; save(ROOT/'status.json',state); return True
    if len(intents)==5 and all(o['state'] in TERMINAL for o in observations):
        # Native writers have ended; keep failed/missing episodes distinct from model failure.
        grace=ROOT/'terminal_grace.json'
        if not grace.exists(): save(grace,dict(started_at=time.time()))
        if time.time()-load(grace)['started_at']>=180:
            state['state']='finished_with_missing_or_failed_artifacts';save(ROOT/'status.json',state);return True
    return False


def main():
    with (ROOT / '.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = (ROOT / 'plan.json').read_bytes(); plan=json.loads(raw)
        validate_plan(plan)
        if not plan['execution_enabled'] or not plan['launch_preflight_complete']:
            raise ValueError('Preparation is not admission')
        save(ROOT / 'launch.json', dict(pid=os.getpid(), started_at=time.time(),
             entry=str(Path(__file__).resolve()), plan_sha256=sha_bytes(raw), recurring_automation=False))
        client=aoss_client()
        while True:
            if (ROOT/'plan.json').read_bytes()!=raw:
                raise ValueError('Plan changed')
            try:
                if tick(client,plan): return
            except Exception as exc:
                save(ROOT/'last_error.json',dict(observed_at=time.time(),error_type=type(exc).__name__,
                      message=str(exc)[:300] if not isinstance(exc,subprocess.SubprocessError) else 'subprocess error'))
            time.sleep(plan['poll_seconds'])


if __name__=='__main__':
    main()
