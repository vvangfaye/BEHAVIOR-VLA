"""Isolated submission for the one authorized combined branch."""
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time
import zlib

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
SOURCE = HERE.parent / 'rl_scheme_comparison_20261101'
OLD = HERE.parent / 'rl_multigpu_20261101'
ADAPT = HERE.parent / 'rlinf_adaptation_20261020'
spec = importlib.util.spec_from_file_location('frozen_comparison_deployment', SOURCE / 'deployment.py')
single = importlib.util.module_from_spec(spec)
spec.loader.exec_module(single)
CLI, ENV, describe, save, aoss_client = single.CLI, single.ENV, single.describe, single.save, single.aoss_client
TERMINAL = {'SUCCEEDED', 'FAILED', 'SUSPENDED', 'STOPPED', 'TERMINATED', 'CANCELED'}


def status_client():
    sys.path.insert(0, str(ADAPT / 'multitask_validation_20261029/capacity_repair_20261030'))
    from internal_storage import internal_aoss_client
    return internal_aoss_client()


def campaign():
    return json.loads((HERE / 'plan.json').read_text())


def worker_plan(role, case=None):
    config = campaign()
    args = {'role': role, 'variant': 'gamma1_ppo'}
    if case is not None:
        args['case'] = case
    plan = single.worker_plan(args)
    plan.update(experiment=config['experiment'], comparison_variant=config['variant'], worker_prefix='combo_')
    assert plan['optimizer']['gamma'] == plan['shaping']['gamma'] == 1.
    assert plan['optimizer']['updates_per_round'] == [17]
    assert plan['optimizer']['actor_lr'] == config['actor_lr']
    assert plan['optimizer']['value_lr'] == config['value_lr']
    assert plan['optimizer']['shuffle_seed'] == config['shuffle_seed']
    assert len(plan['source_rollout_receipts']) == 32
    assert hashlib.sha256((SOURCE / 'source_training_receipts.json').read_bytes()).hexdigest() == config['source_rollout_receipts_sha256']
    if role == 'train':
        plan['output'] = config['remote_root'] + '/training/results'
        plan['runtime_storage']['max_persistent_bytes'] = config['training_storage_limit_bytes']
        plan['phase_limits_seconds']['train1'] = config['train_timeout_seconds']
        plan['distributed'] = {'world_size': 4, 'preflight_only': False,
                               'backend': 'gloo_cpu_reduction', 'kl_batch_size': 8}
        # The logprob/KL implementation is unchanged from the accepted batch-8 gate.
        gate = json.loads((OLD / 'kl_batch_gate.json').read_text())
        assert gate['passed'] and gate['selected_batch_size'] == 8 and gate['training_state_unchanged']
        for name in ['kl_batch.py', 'b1k_rlinf/model.py', 'b1k_rlinf/precision.py']:
            assert (HERE / 'runtime' / name).read_bytes() == (OLD / 'runtime' / name).read_bytes()
        plan['distributed']['runtime_signature'] = hashlib.sha256(
            json.dumps(config['runtime_sha256'], sort_keys=True).encode()).hexdigest()
    else:
        assert case in config['eval_cases']
        plan['output'] = config['remote_root'] + f'/evaluations/i{case[0]}_s{case[1]}/results'
        plan['comparison_checkpoint'] = config['remote_root'] + '/training/results/checkpoints/round_01'
        plan['runtime_storage']['max_persistent_bytes'] = config['evaluation_storage_limit_bytes']
        plan['phase_limits_seconds']['final'] = config['eval_timeout_seconds']
        assert plan['model_seed'] == plan['seed'] == 0
        assert plan['evaluation']['cases'] == [case]
    plan['runtime_storage']['shared_budget_roots'] = [plan['output']]
    return plan


def storage_proof(plan):
    config = campaign()
    cache = HERE / 'quota_latest.json'
    if not cache.exists():
        cache = HERE / 'quota_preflight.json'
    quota = json.loads(cache.read_text()) if cache.exists() else None
    if quota is None or not 0 <= time.time() - quota['observed_at'] <= 180:
        # query_quota uses subprocess inheritance; do not send it through a proxy.
        for key in list(os.environ):
            if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
                os.environ.pop(key)
        quota = single.query_quota()
        save(HERE / 'quota_latest.json', quota)
    row = quota['rows'][0]
    reserve = config['campaign_storage_reserve_bytes']
    margin = config['storage_margin_bytes']
    proof = {
        'complete': True, 'passed': row['quota_bytes'] - row['used_bytes'] >= reserve + margin,
        'experiment': plan['experiment'], 'account_root': '/mnt/afs/260010165',
        'account_allocated_bytes': row['used_bytes'], 'quota_lower_bound_bytes': row['quota_bytes'],
        'new_task_reserve_bytes': reserve, 'other_remaining_commitments_bytes': 0,
        'headroom_margin_bytes': margin, 'required_remaining_bytes': reserve + margin,
        'observed_at': quota['observed_at'], 'quota_source': quota['method'],
        'scope': '36 GiB new-branch reservation: 16 GiB learner plus forty 512 MiB evaluation outputs; inputs already stored',
    }
    assert proof['passed'], 'Wait for conservative storage headroom; never remove protected inputs'
    return proof


def _submit_unlocked(role, case=None, attempt=1):
    config = campaign()
    ident = f'train-a{attempt}' if role == 'train' else f'eval-{case[0]}s{case[1]}-a{attempt}'
    folder = HERE / 'jobs' / ident
    receipt = folder / 'submission.json'
    assert not receipt.exists(), 'Reconcile the exact existing name; never duplicate a submission'
    verification = json.loads((HERE / 'verification.json').read_text())
    assert verification['passed']
    assert verification['plan_sha256'] == hashlib.sha256((HERE / 'plan.json').read_bytes()).hexdigest()
    for name, digest in config['runtime_sha256'].items():
        assert hashlib.sha256((HERE / 'runtime' / name).read_bytes()).hexdigest() == digest
    plan = worker_plan(role, case)
    if role == 'train' and attempt > 1:
        accepted = []
        for path in (HERE / 'jobs').glob('train-*/latest.json'):
            previous = json.loads(path.read_text())
            accepted.append(previous.get('progress', {}).get('accepted_optimizer_steps', 0))
        if max(accepted, default=0) > 0:
            plan['distributed']['require_resume'] = True
    proof = storage_proof(plan)
    template = describe('b1k-repeat1688-1019-batch-00')
    assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / 'worker_plan.json', plan)
    contents = {n: (HERE / 'runtime' / n).read_bytes() for n in config['runtime_sha256']}
    contents['driver.py'] = contents['distributed_driver.py' if role == 'train' else 'comparison_driver.py']
    bootstrap_source = (HERE / 'runtime' / ('bootstrap.py' if role == 'train' else 'single_bootstrap.py')).read_text()
    if role == 'eval':
        # The proven one-GPU bootstrap consumes an inline bundle. Exclude unused
        # four-GPU utilities so its immutable command remains below 64 KiB.
        names = ['common.py','training.py','experiment.py','simulator.py','helpers.py','eval_utils.py',
                 'resource_watch.py','goals.py','storage_guard.py','capacity_admission.py',
                 'variant_update.py','driver.py']
        names += ['b1k_rlinf/' + n + '.py' for n in ['__init__','action_interface','factory','io','model','precision','tokenizer']]
        contents = {name: contents[name] for name in names}
    else:
        names = ['common.py','training.py','variant_update.py','parallel_training.py','experiment.py','goals.py','helpers.py','eval_utils.py','storage_guard.py','capacity_admission.py','resource_watch.py','kl_batch.py','driver.py']
        names += ['b1k_rlinf/' + n + '.py' for n in ['__init__','action_interface','factory','io','model','precision','tokenizer']]
        contents = {name: contents[name] for name in names}

    wire_plan = dict(plan)
    # Historical coordinator notes are kept in worker_plan.json, not needed by the driver.
    for key in ['authorization','resume_authorization','dependencies','dependency_status_file','dependency_campaign_directory','previous_generation_directory','startup_repair','rules','comparisons','baseline_sources']:
        wire_plan.pop(key, None)
    contents['plan.json'] = json.dumps(wire_plan, sort_keys=True).encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=9) as archive:
        for name, body in contents.items():
            item = tarfile.TarInfo('run/' + name)
            item.size = len(body)
            item.mtime = 0
            archive.addfile(item, io.BytesIO(body))
    bundle = stream.getvalue()
    prefix = config['aoss_prefix'] + 'jobs/' + ident + '/'
    name = 'b1k-combo-1102-' + ident
    client = aoss_client()
    signing_client = status_client()
    try:
        client.put_object(Bucket='260010165', Key=prefix + 'bundle.tar.gz', Body=bundle)
        assert client.get_object(Bucket='260010165', Key=prefix + 'bundle.tar.gz')['Body'].read() == bundle
        def signed(method, item):
            return signing_client.generate_presigned_url(method, Params={'Bucket': '260010165', 'Key': prefix + item}, ExpiresIn=5 * 86400)
        assets = json.loads((ADAPT / 'cloud_smoke/artifacts.json').read_text())
        before = json.loads((ADAPT / 'cloud_smoke/local_preflight.json').read_text())
        reuse = {'checkpoint/' + k: v['sha256'] for k, v in assets['checkpoint'].items()}
        reuse['bundle.tar.gz'] = assets['bundle']['sha256']
        reuse.update({k: v for k, v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
        for path in ['rlinf/algorithms/advantages.py', 'rlinf/algorithms/losses.py', 'rlinf/models/embodiment/openpi/tasks/rl.py']:
            reuse['upstream/' + path] = hashlib.sha256((ADAPT / 'upstream' / path).read_bytes()).hexdigest()
        phase = 'train1' if role == 'train' else 'final'
        startup = {
            'root': config['remote_root'] + '/jobs/' + ident,
            'base': '/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
            'task_phase': phase, 'rank': 0, 'attempt': attempt, 'job_name': name,
            'bundle_get': signed('get_object', 'bundle.tar.gz'),
            'bundle': {'bytes': len(bundle), 'sha256': hashlib.sha256(bundle).hexdigest()},
            'reuse_sha256': reuse, 'status_put': signed('put_object', 'status.json'),
            'result_put': signed('put_object', 'result.tar.gz'), 'artifact_puts': {}, 'storage_proof': proof,
        }
        startup['bundle_inline'] = base64.b64encode(bundle).decode()
        startup.pop('bundle_get')
        bootstrap = bootstrap_source
        source = bootstrap + '\nguarded_run(' + repr(startup) + ')\n'
        packed = base64.b64encode(zlib.compress(source.encode(), level=9)).decode()
        limit = config['train_timeout_seconds' if role == 'train' else 'eval_timeout_seconds']
        command = f'timeout --signal=TERM --kill-after=60s {limit - 60}s python3 -u -c ' + shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
        assert len(command) < 64000
        row = {
            'id': ident, 'job': name, 'role': role, 'variant': config['variant'],
            'attempt': attempt, 'gpu_count': 4 if role == 'train' else 1,
            'quota': 'spot', 'aoss_prefix': prefix, 'output': plan['output'],
            'max_seconds': limit, 'state': 'submitting', 'submitted_at': time.time(),
            'bundle_bytes': len(bundle), 'startup_sha256': hashlib.sha256(command.encode()).hexdigest(),
            'runtime_sha256': config['runtime_sha256'],
        }
        if case is not None:
            row['case'] = case
        with receipt.open('x') as output:
            json.dump(row, output, ensure_ascii=False, indent=2)
        result = subprocess.run([
            CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
            '--copy-job-name=b1k-repeat1688-1019-batch-00', '--name=' + name, '--job-name=' + name,
            '--worker-spec=' + ('n12lp.nn.i10a.4' if role == 'train' else 'n12lp.nn.i10a.1'),
            '--worker-nodes=1', '--quota-type=spot', '--retry-times=0',
            '--enable-fault-tolerance=false', '--enable-anomaly-detection=false', '--command=' + command,
        ], env=ENV, capture_output=True, text=True, timeout=90)
        row['submit_returncode'] = result.returncode
        save(receipt, row)
        job = describe(name)
        assert job['scheduling']['quota_type'] == 'SPOT'
        assert hashlib.sha256(job['roles'][0]['startup_script'].encode()).hexdigest() == row['startup_sha256']
        row.update(uid=job['uid'], state=job['state'])
        save(receipt, row)
        print(json.dumps({k: row[k] for k in ['job', 'uid', 'state', 'gpu_count', 'bundle_bytes']}), flush=True)
        return row
    finally:
        client.close()
        signing_client.close()


def submit(role, case=None, attempt=1):
    import fcntl
    import controller as scheduling
    # Host-side lock only; never occupies a GPU or changes an existing writer.
    with (PROJECT / 'analysis/rl_gpu_submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        active = scheduling.active_gpu_count([])
        required = 4 if role == 'train' else 1
        if active + required > campaign()['max_total_live_gpus']:
            raise RuntimeError('Waiting for the shared 32-GPU admission limit')
        return _submit_unlocked(role, case, attempt)
