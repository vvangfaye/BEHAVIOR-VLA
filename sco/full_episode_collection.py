"""Freeze a finite public-2000 collection; submit only fresh-capacity Spot shards."""
import argparse
import ast
import base64
from contextlib import contextmanager
import fcntl
import hashlib
import io
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import time
import zlib
import urllib.error
import urllib.request

from free_capacity_collection import CapacityUnavailable, available_slots, pending_requests
from submit_full_collection_shard import describe, environment

REPO = Path(__file__).resolve().parents[1]
BUCKET = '260010165'
POOL = 'computing-cluster-5090-01g'
OVERLAYS = ('scripts/eval_b1k.sh', 'scripts/eval_b1k_queue.py', 'scripts/subtask_trajectory.py',
    'scripts/full_episode_trajectory.py', 'scripts/full_episode_checkpoint.py',
    'scripts/full_episode_multipart.py', 'scripts/collection_checkpoint.py',
    'scripts/export_full_episode_collection.py', 'scripts/verify_full_episode_codec.py',
    'sco/run_public_test_assignment.py', 'sco/run_full_episode_collection.py')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def fetch_startup_payload(url, expected_sha256):
    """Retry only bounded, read-only bootstrap GETs before starting any writer."""
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                body = response.read(32 * 1024**2 + 1)
            if len(body) > 32 * 1024**2 or hashlib.sha256(body).hexdigest() != expected_sha256:
                raise ValueError('Private startup payload size or SHA256 differs')
            return body
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            if isinstance(error, urllib.error.HTTPError) and error.code not in (429, 500, 502, 503, 504):
                raise
            if attempt == 5:
                raise RuntimeError('Startup payload GET failed after six bounded attempts') from None
            print('STARTUP_PAYLOAD_GET_RETRY', attempt + 1, type(error).__name__, flush=True)
            time.sleep(min(2**attempt, 16))


def build_startup(payload_url, payload_sha256, driver):
    encoded = base64.b64encode(zlib.compress(driver)).decode()
    return ('import base64,hashlib,json,time,urllib.error,urllib.request,zlib\n'
        + inspect.getsource(fetch_startup_payload) + '\n'
        + 'b=fetch_startup_payload(' + repr(payload_url) + ',' + repr(payload_sha256) + ')\n'
        + 's=zlib.decompress(base64.b64decode(' + repr(encoded) + '))\n'
        + 'assert hashlib.sha256(s).hexdigest()==' + repr(digest(driver)) + '\n'
        + 'exec(compile(s,"full_episode_driver","exec"));run(json.loads(b))')


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
    temporary.replace(path)


def require_capacity(observation, ledger, cli, env):
    now = time.time()
    if observation.get('admission_mode') == 'bounded_spot_queue':
        assert observation['queue_authorization'] == 'user_20261012_max_five_queued'
        assert observation['workspace'] == 'share-space'
        assert observation['resource_pool'] == POOL and observation['quota_type'] == 'spot'
        assert observation['max_pending_tasks'] == 5
        assert observation['observed_at'] <= now < observation['deadline_epoch']
        # Reserve this monitor cycle's slots once. A job that starts or finishes
        # during serial preparation still consumes its reservation until the
        # next real queue observation. Re-query before each create to catch
        # other owner requests and count unknown intents conservatively.
        campaign_only = observation.get('queue_scope') == 'authorized_campaign_only'
        charged = {name: value for name, value in observation['pending_requests'].items()
                   if not campaign_only or name in ledger}
        charged.update({name: value for name, value in pending_requests(cli, env).items()
                        if not campaign_only or name in ledger})
        for name, item in ledger.items():
            if (not item.get('uid') or item['reserved_at'] >= observation['observed_at']
                    or item.get('created_at', float('inf')) >= observation['observed_at']):
                charged[name] = max(1, charged.get(name, 0))
        if sum(max(1, value) for value in charged.values()) >= 5:
            raise CapacityUnavailable('Five pending Spot task slots are reserved')
        return
    if observation.get('admission_mode') == 'finite_observed_batch':
        # Explicit user authorization consumes this real observation once,
        # even when serial creates outlast the normal freshness window. Keep
        # its timestamp unchanged: the shared ledger charges every later
        # creation, including jobs that have already finished.
        assert observation['batch_authorization'] == 'user_20261012_every20min_fill_observed_capacity'
        assert observation['observed_at'] <= now < observation['deadline_epoch']
        assert observation['worker_budget'] == min(
            int(observation['free_gpus']), int(observation['free_vcpus'] // 14),
            int(observation['free_memory_gib'] // 60))
        now = observation['observed_at']
    if available_slots(observation, ledger, pending_requests(cli, env), now) < 1:
        raise CapacityUnavailable('No complete unreserved worker slot remains')


def build_plan(ready, prompts, validation_instances, *, run_id, episodes_per_shard=5):
    if not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,60}', run_id):
        raise ValueError('Use a safe unique run ID')
    tasks = sorted(ready['scope']['tasks'])
    if len(tasks) != 100 or len(set(tasks)) != 100 or set(tasks) - prompts.keys():
        raise ValueError('Public collection requires exactly 100 known tasks')
    if ready['checkpoint_step'] != 1688977 or ready['state'] != 'ready':
        raise ValueError('Use the verified current policy checkpoint')
    validation = sorted({(task, instance) for task, instance in validation_instances})
    if not validation or any(task not in tasks or type(i) is not int or not 0 <= i <= 300
                             for task, i in validation):
        raise ValueError('Freeze separate nonpublic task instances for validation')
    if type(episodes_per_shard) is not int or not 1 <= episodes_per_shard <= 5:
        raise ValueError('Bound each GPU shard to at most five full episodes')
    shards = [dict(index=0, task='turning_on_radio', instance_ids=[301], pilot=True)]
    for task in tasks:
        instances = [i for i in range(301, 321) if (task, i) != ('turning_on_radio', 301)]
        for start in range(0, len(instances), episodes_per_shard):
            shards.append(dict(index=len(shards), task=task,
                               instance_ids=instances[start:start + episodes_per_shard], pilot=False))
    pairs = {(s['task'], i) for s in shards for i in s['instance_ids']}
    assert len(pairs) == sum(len(s['instance_ids']) for s in shards) == 2000
    plan = dict(kind='native_full_episode_collection_plan', run_id=run_id, expected_episodes=2000,
        checkpoint=ready['checkpoint'], checkpoint_identity=dict(checkpoint_step=1688977,
            checkpoint_source=ready['checkpoint'], checkpoint_files=ready['scope']['checkpoint_files']),
        base_code=ready['code'], base_source_manifest_sha256=ready['source_manifest_sha256'],
        task_prompts={task: prompts[task] for task in tasks}, shards=shards,
        held_out_instances=[list(pair) for pair in validation], official_benchmark=False,
        all_public_instances_training_seen=True, training_selection='complete_environment_success_only',
        policy_labels='weak_native_generated_subtasks', max_complete_rollouts_per_instance=1,
        automatic_failure_recovery=False, max_gpu_hours_per_shard=96, quota_type='spot')
    plan['fingerprint'] = digest(json.dumps(plan, sort_keys=True).encode())
    return plan


def freeze(root, ready, validation_instances, *, run_id, codec_wheels=None):
    registry = ast.parse((REPO / 'openpi/src/openpi/configs/tasks/b1k.py').read_text())
    assignment = next(node for node in registry.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == 'TASKS' for target in node.targets))
    prompts = ast.literal_eval(assignment.value)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    plan = build_plan(ready, prompts, validation_instances, run_id=run_id)
    files = {name: digest((REPO / name).read_bytes()) for name in OVERLAYS}
    dependencies = [] if codec_wheels is None else sorted(Path(codec_wheels).glob('h5py-*.whl'))
    if codec_wheels is not None:
        assert dependencies, 'Provide pinned Linux h5py wheels for the isolated codec dependency directory'
    extra_files = {'vendor/full_episode_codec/' + p.name: p for p in dependencies}
    files.update({name: digest(p.read_bytes()) for name, p in extra_files.items()})
    with tarfile.open(root / 'overlay.tar.gz', 'w:gz') as archive:
        for name in OVERLAYS:
            archive.add(REPO / name, arcname=name, recursive=False)
        for name, path in extra_files.items():
            archive.add(path, arcname=name, recursive=False)
        body = json.dumps(files, sort_keys=True).encode()
        member = tarfile.TarInfo('_FULL_EPISODE_OVERLAY.json'); member.size = len(body)
        archive.addfile(member, io.BytesIO(body))
    plan['overlay_sha256'] = digest((root / 'overlay.tar.gz').read_bytes())
    plan['overlay_files_sha256'] = files
    # The final plan identity includes exact deployed source bytes.
    plan.pop('fingerprint')
    plan['fingerprint'] = digest(json.dumps(plan, sort_keys=True).encode())
    save(root / 'plan.json', plan)
    save(root / 'held_out_instances.json', plan['held_out_instances'])
    (root / 'submissions').mkdir()
    return plan


def submission_generation(root, record):
    generation = record.get('generation', 0)
    assert type(generation) is int and 0 <= generation <= 2
    if record.get('failure_recovery'):
        from full_episode_failure_recovery import validate_record
        validate_record(Path(root), record)
        return generation
    if generation:
        recovery = record['startup_recovery']
        history = Path(recovery['history'])
        assert not history.is_absolute() and '..' not in history.parts
        raw = (Path(root) / history).read_bytes()
        assert digest(raw) == recovery['history_sha256']
        previous = json.loads(raw)
        assert previous.get('generation', 0) == 0 and previous['uid'] == recovery['previous_uid']
        assert previous['name'] == record['name'].removesuffix('-g1') + '-g0'
        assert all(previous[key] == record[key] for key in ('index', 'task', 'instance_ids', 'plan_fingerprint'))
        evidence = Path(recovery['evidence'])
        assert not evidence.is_absolute() and '..' not in evidence.parts
        proof_raw = (Path(root) / evidence).read_bytes()
        assert digest(proof_raw) == recovery['evidence_sha256']
        proof = json.loads(proof_raw)
        assert proof['previous_uid'] == previous['uid'] and proof['previous_platform_state'] == 'FAILED'
        assert proof['old_writer_stopped'] is True and proof['assignment_root_absent'] is True
        assert proof['all_episode_objects_absent'] is True and proof['completion_absent'] is True
        assert proof['reason'] == 'bootstrap_dns_before_inference'
    return generation


def recover_startup(root, index, capacity_path, cli, client, afs_proof_path):
    """Explicit recovery of the two known pre-inference DNS failures only."""
    root = Path(root)
    assert index in (1, 4)
    previous = json.loads((root / f'submissions/{index:04d}.json').read_text())
    assert previous.get('uid') and previous.get('generation', 0) == 0
    job = describe(cli, environment(), previous['name'])
    assert job['uid'] == previous['uid'] and job['state'] == 'FAILED'
    workers = subprocess.check_output([cli, 'acp', 'jobs', 'get-workers', previous['name'],
        '--workspace-name=share-space'], env=environment(), text=True, timeout=30)
    assert previous['uid'] in workers and 'Failed' in workers and 'Running' not in workers
    afs = json.loads(Path(afs_proof_path).read_text())
    assert 0 <= time.time() - afs['observed_at'] <= 120
    assert afs['assignment_roots'][previous['assignment_root']] is False
    observer = describe(cli, environment(), afs['observer_job']['name'])
    assert observer['uid'] == afs['observer_job']['uid'] and observer['state'] == 'RUNNING'
    dns = json.loads((root / f'production_{index:04d}_bootstrap_dns_evidence.json').read_text())
    assert dns['uid'] == previous['uid'] and 'Temporary failure in name resolution' in dns['error']
    assert dns['inference_started'] is False
    plan = json.loads((root / 'plan.json').read_text())
    prefix = 'behavior1k/full-episode-collection/' + plan['run_id'] + '/' + plan['fingerprint'][:12]
    keys = [previous['completion_key']]
    for instance in previous['instance_ids']:
        base = prefix + f"/episodes/{previous['task']}/{instance}/attempt-1"
        keys.extend([base + '.json', base + '.tar'])
    for key in keys:
        try:
            client.head_object(Bucket=BUCKET, Key=key)
        except Exception as error:
            assert getattr(error, 'response', {}).get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound')
        else:
            raise ValueError('Existing episode/completion object prevents startup recovery')
    history = root / 'submission_history' / f'{index:04d}'
    history.mkdir(parents=True, exist_ok=True)
    old_raw = (root / f'submissions/{index:04d}.json').read_bytes()
    old_path = history / 'g0.json'
    if old_path.exists():
        assert old_path.read_bytes() == old_raw
    else:
        old_path.write_bytes(old_raw)
    proof = dict(reason='bootstrap_dns_before_inference', observed_at=time.time(),
        previous_uid=previous['uid'], previous_platform_state='FAILED', old_writer_stopped=True,
        assignment_root_absent=True, all_episode_objects_absent=True, completion_absent=True,
        afs_proof_sha256=digest(Path(afs_proof_path).read_bytes()), object_keys_checked=keys)
    proof_path = history / 'recovery_evidence.json'
    save(proof_path, proof)
    recovery = dict(history=str(old_path.relative_to(root)), history_sha256=digest(old_raw),
        evidence=str(proof_path.relative_to(root)), evidence_sha256=digest(proof_path.read_bytes()),
        previous_uid=previous['uid'])
    return submit(root, index, capacity_path, cli, client, previous=previous, recovery=recovery)


def submit(root, index, capacity_path, cli, client, *, previous=None, recovery=None):
    root = Path(root)
    plan = json.loads((root / 'plan.json').read_text())
    assert plan['fingerprint'] == digest(json.dumps({k: v for k, v in plan.items() if k != 'fingerprint'}, sort_keys=True).encode())
    if plan.get('expected_episodes') == 1000:
        from full_episode_scope import validate_final1000
        validate_final1000(plan)
    elif plan.get('paired_evaluation_only'):
        from paired_checkpoint_evaluation import validate_plan
        validate_plan(plan)
    assert digest((root / 'overlay.tar.gz').read_bytes()) == plan['overlay_sha256']
    transport = None
    if (root / 'transport-v1/revision.json').exists():
        from full_episode_transport_revision import validate
        transport = validate(root)
        assert previous is None or (recovery and recovery.get('kind') in ('user_failure78_recovery', 'user_final1000_failure5_recovery')), 'Unaccepted recovery revision'
    authorization_path = root / 'controller_authorization.json'
    run_timeout = 345600
    if authorization_path.exists():
        authorization = json.loads(authorization_path.read_text())
        assert authorization['plan_fingerprint'] == plan['fingerprint']
        run_timeout = min(run_timeout, int(authorization['deadline_epoch'] - time.time()))
        if run_timeout <= 0:
            raise TimeoutError('Finite collection deadline reached')
    shard = plan['shards'][index]
    assert shard['index'] == index
    receipt_path = root / 'submissions' / f'{index:04d}.json'
    failure_recovery = recovery and recovery.get('kind') in ('user_failure78_recovery', 'user_final1000_failure5_recovery')
    generation = 0 if previous is None else previous.get('generation', 0) + 1
    if failure_recovery:
        from full_episode_failure_recovery import validate_preparation
        validate_preparation(root, previous, recovery)
        shard = dict(shard, instance_ids=recovery['missing_instance_ids'])
        assert shard['instance_ids'], 'No missing episodes to recover'
    if previous is None:
        assert not receipt_path.exists(), 'Reconcile the exact saved UID; never blind resubmit'
    else:
        assert json.loads(receipt_path.read_text()) == previous and recovery is not None
    if not shard['pilot']:
        from full_episode_acceptance import validate_gate
        from full_episode_scope import pilot_gate_waived
        if not pilot_gate_waived(root, plan):
            validate_gate(root, plan)
    env = environment()
    template = describe(cli, env, 'pt-y91th8k9')
    assert template['uid'] == 'c14d0892eaa148d381c7f4424684c9d3' and template['resource_pool']['name'] == POOL
    quota = subprocess.check_output([cli, 'afs', 'dir-quota', 'list',
        '--volumeID=01a04263-91e5-7603-bc01-c67e503da6b5', '--dirPath=/260010165'], env=env, text=True, timeout=30)
    numbers = re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*(TB|GB|MB)', quota)
    assert len(numbers) == 2
    def estimate(v, u, upper=False):
        decimals = len(v.split('.')[1]) if '.' in v else 0
        return (float(v) + (.5 if upper else -.5) * 10**(-decimals)) * {'TB': 10**12, 'GB': 10**9, 'MB': 10**6}[u]
    assert estimate(*numbers[0]) - estimate(*numbers[1], upper=True) >= 16 * 1024**3
    observation = json.loads(Path(capacity_path).read_text())
    ledger = {v['name']: v for p in (root / 'submissions').glob('*.json')
              for v in [json.loads(p.read_text())]}
    require_capacity(observation, ledger, cli, env)
    name = f"behavior-full-ep-{plan['fingerprint'][:8]}-a{index:04d}-g{generation}"
    record = dict(name=name, index=index, state='preparing_objects', reserved_at=time.time(),
        plan_fingerprint=plan['fingerprint'], task=shard['task'], instance_ids=plan['shards'][index]['instance_ids'],
        quota_type='spot', gpu_count=1, official_benchmark=False, max_gpu_hours=96, generation=generation)
    if recovery is not None:
        record['failure_recovery' if failure_recovery else 'startup_recovery'] = recovery
        submission_generation(root, record)
        save(receipt_path, record)
    else:
        with receipt_path.open('x') as out:
            out.write(json.dumps(record, indent=2) + '\n')
    prefix = 'behavior1k/full-episode-collection/' + plan['run_id'] + '/' + plan['fingerprint'][:12]
    def signed(operation, key, **extra):
        return client.generate_presigned_url(operation, Params=dict(Bucket=BUCKET, Key=key, **extra), ExpiresIn=604800)
    def urls(key):
        return dict(key=key, put=signed('put_object', key), get=signed('get_object', key))
    bundle_key = prefix + ('/transport-v1/overlay.tar.gz' if transport else '/overlay.tar.gz')
    bundle_path = root / ('transport-v1/overlay.tar.gz' if transport else 'overlay.tar.gz')
    client.put_object(Bucket=BUCKET, Key=bundle_key, Body=bundle_path.read_bytes())
    storage = {}
    metrics_storage = {}
    for instance in shard['instance_ids']:
        object_prefix = prefix + f"/episodes/{shard['task']}/{instance}/attempt-1"
        if plan.get('paired_evaluation_only'):
            storage[f"{shard['task']}:{instance}:0"] = dict(receipt=urls(object_prefix + '.json'))
            metrics_storage[f"{shard['task']}:{instance}:0"] = urls(
                prefix + f"/metrics/{shard['task']}/{instance}/attempt-1.json")
            continue
        if failure_recovery:
            object_prefix += f'-g{generation}'
        key = object_prefix + '.tar'
        upload_id = client.create_multipart_upload(Bucket=BUCKET, Key=key, ContentType='application/x-tar')['UploadId']
        storage[f"{shard['task']}:{instance}:0"] = dict(receipt=urls(object_prefix + '.json'),
            archive=dict(key=key, parts=[signed('upload_part', key, UploadId=upload_id, PartNumber=i)
                for i in range(1, (transport['maximum_parts'] if transport else plan.get('maximum_archive_parts', 512)) + 1)], complete=signed('complete_multipart_upload', key, UploadId=upload_id),
                head=signed('head_object', key)))
        if plan.get('trained_model_export') or plan.get('publish_compact_metrics'):
            metrics_storage[f"{shard['task']}:{instance}:0"] = urls(
                prefix + f"/metrics/{shard['task']}/{instance}/attempt-1.json")
    assignment_root = f"/mnt/afs/260010165/behavior_1k/eval_results/{plan['run_id']}/assignments/a{index:04d}"
    if generation:
        assignment_root += f'-g{generation}'
    payload = dict(plan, **shard, code=f"/mnt/afs/260010165/behavior_1k/code_snapshots/{plan['run_id']}-{plan['overlay_sha256'][:12]}",
        runtime='/mnt/afs/260010165/behavior_1k/runtime_20261007_1306', execution_name=name,
        assignment_root=assignment_root, expected_rollouts=len(shard['instance_ids']), generation=generation,
        overlay_get=signed('get_object', bundle_key), object_storage=storage,
        completion_object=urls(prefix + f'/shards/{index:04d}.json'),
        failure_object=urls(prefix + f'/diagnostics/{index:04d}.json'))
    if failure_recovery:
        payload['completion_object'] = urls(prefix + f'/shards/{index:04d}-g{generation}.json')
        payload['failure_object'] = urls(prefix + f'/diagnostics/{index:04d}-g{generation}.json')
    payload['run_id'] = plan['run_id'] + f'-a{index:04d}'
    if plan.get('trained_model_export') or plan.get('publish_compact_metrics'):
        payload['metrics_object_storage'] = metrics_storage
    if transport:
        payload.update(overlay_sha256=transport['overlay_sha256'],
                       overlay_files_sha256=transport['overlay_files_sha256'],
                       code=payload['code'] + '-transport-' + transport['revision_sha256'][:12],
                       transport_revision_sha256=transport['revision_sha256'])
        record.update(transport_revision_sha256=transport['revision_sha256'],
                      deployed_source_sha256=transport['source_manifest_sha256'])
    payload_key = prefix + f'/private/{index:04d}.json'
    if generation:
        payload_key = prefix + f'/private/{index:04d}-g{generation}.json'
    body = json.dumps(payload, sort_keys=True).encode()
    assert len(body) <= 32 * 1024**2, 'Private startup payload exceeds its read bound'
    client.put_object(Bucket=BUCKET, Key=payload_key, Body=body)
    driver = (REPO / 'sco/run_full_episode_collection.py').read_bytes()
    assert digest(driver) == plan['overlay_files_sha256']['sco/run_full_episode_collection.py']
    startup = build_startup(signed('get_object', payload_key), digest(body), driver)
    observation = json.loads(Path(capacity_path).read_text())
    try:
        require_capacity(observation, ledger, cli, env)
    except (CapacityUnavailable, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # No create RPC has happened. Preserve the preparation and release only
        # this local intent; an uncertain cloud creation is never handled here.
        assert 'created_at' not in record and not record.get('uid')
        archive = root / 'unsubmitted_preparations'
        archive.mkdir(exist_ok=True)
        receipt_path.replace(archive / f'{index:04d}-g{generation}-{time.time_ns()}.json')
        if previous is not None:
            save(receipt_path, previous)
        raise
    record.update(state='submitting', created_at=time.time(), payload_key=payload_key,
                  assignment_root=assignment_root, completion_key=payload['completion_object']['key'],
                  failure_key=payload['failure_object']['key'],
                  run_timeout_seconds=run_timeout)
    save(receipt_path, record)
    command = [cli, 'acp', 'jobs', 'copy', '--workspace-name=share-space', '--copy-job-name=pt-y91th8k9',
        '--name=' + name, '--job-name=' + name, '--worker-spec=n12lp.nn.i10a.1', '--worker-nodes=1',
        '--quota-type=spot', '--env=NVIDIA_DRIVER_CAPABILITIES:all',
        '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
        '--command=timeout --signal=TERM --kill-after=30 ' + str(run_timeout) + ' python3 -c ' + shlex.quote(startup)]
    # Persist intent before creation. Exceptions leave it for exact-name reconciliation.
    subprocess.run(command, env=env, capture_output=True, text=True, check=True, timeout=90)
    job = describe(cli, env, name)
    assert job['scheduling']['quota_type'] == 'SPOT' and job['resource_pool']['name'] == POOL
    record.update(state='submitted', uid=job['uid'], platform_state=job['state'])
    save(receipt_path, record)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('freeze')
    prepare.add_argument('--root', type=Path, required=True)
    prepare.add_argument('--ready', type=Path, required=True)
    prepare.add_argument('--held-out-instances', type=Path, required=True)
    prepare.add_argument('--run-id', required=True)
    prepare.add_argument('--codec-wheels', type=Path)
    create = commands.add_parser('submit')
    create.add_argument('--root', type=Path, required=True)
    create.add_argument('--index', type=int, required=True)
    create.add_argument('--capacity', type=Path, required=True)
    create.add_argument('--cli', default='/private/tmp/behavior-sco-runtime/bin/sco')
    batch = commands.add_parser('submit-batch')
    batch.add_argument('--root', type=Path, required=True)
    batch.add_argument('--capacity', type=Path, required=True)
    batch.add_argument('--max-shards', type=int, default=400)
    batch.add_argument('--cli', default='/private/tmp/behavior-sco-runtime/bin/sco')
    accept = commands.add_parser('accept-pilot')
    accept.add_argument('--root', type=Path, required=True)
    accept.add_argument('--cli', default='/private/tmp/behavior-sco-runtime/bin/sco')
    args = parser.parse_args()
    if args.command == 'freeze':
        value = freeze(args.root, json.loads(args.ready.read_text()),
                       json.loads(args.held_out_instances.read_text()), run_id=args.run_id,
                       codec_wheels=args.codec_wheels)
        print(json.dumps(dict(fingerprint=value['fingerprint'], episodes=2000, shards=len(value['shards']))))
    else:
        from behavior_aoss_client import client
        with (args.root / '.submission.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            storage = client()
            if args.command == 'accept-pilot':
                from full_episode_acceptance import accept_pilot
                value = accept_pilot(args.root, args.cli, storage)
            elif args.command == 'submit-batch':
                assert 1 <= args.max_shards <= 400
                plan = json.loads((args.root / 'plan.json').read_text())
                submitted = []
                for shard in plan['shards'][1:]:
                    if (args.root / 'submissions' / f"{shard['index']:04d}.json").exists():
                        continue
                    if len(submitted) >= args.max_shards:
                        break
                    try:
                        item = submit(args.root, shard['index'], args.capacity, args.cli, storage)
                    except CapacityUnavailable:
                        break
                    submitted.append({k: item[k] for k in ('index', 'name', 'uid', 'platform_state')})
                    print(json.dumps(submitted[-1]), flush=True)
                value = dict(submitted=submitted, capacity_refresh_required=True)
            else:
                item = submit(args.root, args.index, args.capacity, args.cli, storage)
                value = {k: item[k] for k in ('index', 'name', 'uid', 'platform_state')}
        print(json.dumps(value))


if __name__ == '__main__':
    main()
