"""Job construction and submission for rl_eval_power_lr_20261102.

Adapted from rl_cross_task_20261102/modem/deployment.py (same template job, bootstrap, bundle
format and AOSS status/result channel). Differences: arms (training LR scan + five evaluation
arms), original-policy evaluation without an overlay, Reserved-first quota for 1-GPU evaluations.
"""
import base64
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time
import zlib

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
ANALYSIS = PROJECT / 'analysis'
SOURCE = ANALYSIS / 'rl_scheme_comparison_20261101'
OLD = ANALYSIS / 'rl_multigpu_20261101'
ADAPT = ANALYSIS / 'rlinf_adaptation_20261020'
CROSS = ANALYSIS / 'rl_cross_task_20261102/modem'
spec = importlib.util.spec_from_file_location('frozen_comparison_deployment', SOURCE / 'deployment.py')
single = importlib.util.module_from_spec(spec)
spec.loader.exec_module(single)
CLI, ENV, describe, save, aoss_client = single.CLI, single.ENV, single.describe, single.save, single.aoss_client
TERMINAL = {'SUCCEEDED', 'FAILED', 'SUSPENDED', 'STOPPED', 'TERMINATED', 'CANCELED'}
TEMPLATE = 'b1k-repeat1688-1019-batch-00'
EVAL_FILES = ['common.py', 'training.py', 'experiment.py', 'simulator.py', 'helpers.py', 'eval_utils.py',
              'resource_watch.py', 'goals.py', 'storage_guard.py', 'capacity_admission.py', 'variant_update.py', 'driver.py']
# helpers.py / eval_utils.py are evaluation-only (simulator path); the learner import graph is
# driver -> common, experiment, training, parallel_training, kl_batch, variant_update, goals, b1k_rlinf,
# plus storage_guard -> capacity_admission and resource_watch for the bootstrap. Dropping the two
# unused modules keeps the inline command below the 64 KiB limit.
TRAIN_FILES = ['common.py', 'training.py', 'variant_update.py', 'parallel_training.py', 'experiment.py', 'goals.py',
               'storage_guard.py', 'capacity_admission.py', 'resource_watch.py', 'kl_batch.py', 'driver.py']
PACKAGE = ['b1k_rlinf/' + n + '.py' for n in ['__init__', 'action_interface', 'factory', 'io', 'model', 'precision', 'tokenizer']]


def status_client():
    sys.path.insert(0, str(ADAPT / 'multitask_validation_20261029/capacity_repair_20261030'))
    from internal_storage import internal_aoss_client
    return internal_aoss_client()


def campaign():
    return json.loads((HERE / 'plan.json').read_text())


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_cases(arm):
    config = campaign()
    return config['repeat_cases'] if config['eval_arms'][arm]['cases'] == 'seeds_0_4' else config['eval_cases']


def checkpoint_dir(arm):
    config = campaign()
    target = config['eval_arms'][arm]['checkpoint']
    if target is None:
        return None
    if target.startswith('training:'):
        return config['remote_root'] + f'/training/{target.split(":", 1)[1]}/results/checkpoints/round_01'
    return target


def base_plan():
    """The verified modem source plan with the restored, hash-checked 32 rollouts."""
    config = campaign()
    assert sha_file(CROSS / 'source_plan.json') == config['source_plan_sha256']
    plan = json.loads((CROSS / 'source_plan.json').read_text())
    contract = hashlib.sha256(json.dumps({'goals': plan['goal_contract'], 'shaping': plan['shaping']},
                                         sort_keys=True).encode()).hexdigest()
    receipts = json.loads((CROSS / 'source_training_receipts.json').read_text())
    assert sha_file(CROSS / 'source_training_receipts.json') == config['source_rollout_receipts_sha256']
    assert len(receipts) == 32 and all(r['contract'] == contract for r in receipts)
    evidence = json.loads((CROSS / 'input_restore_verified.json').read_text())
    assert evidence['passed'] and evidence['all_sha256_verified']
    assert evidence['original_receipts_sha256'] == config['source_rollout_receipts_sha256']
    raw = (CROSS / 'restored_training_receipts.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == evidence['restored_receipts_sha256'] == config['restored_receipts_sha256']
    moved = json.loads(raw)
    assert len(moved) == 32
    for old, new in zip(receipts, moved):
        assert {k: v for k, v in old.items() if k != 'path'} == {k: v for k, v in new.items() if k != 'path'}
        assert Path(new['path']).is_relative_to(evidence['restored_root'])
    assert sum(r['chunks'] for r in moved) == config['source_chunks']
    plan['source_rollout_root'] = evidence['restored_root']
    plan['source_rollout_receipts'] = [{k: r[k] for k in ['episode', 'instance', 'seed', 'path', 'sha256', 'policy_hash',
                                                          'contract', 'chunks']} for r in moved]
    plan['source_reward_contract_sha256'] = contract
    if 'task_bddl_sha256' not in plan['source_identity']:
        plan['source_identity']['task_bddl_sha256'] = plan['goal_contract']['bddl_sha256']
    plan.update(worker_prefix='plr_', rounds=1, baseline_sources={})
    plan['optimizer']['updates_per_round'] = [17]
    plan['optimizer']['gamma'] = plan['shaping']['gamma'] = 0.99
    return plan


def worker_plan(role, arm, case=None):
    config = campaign()
    plan = base_plan()
    plan['comparison_role'] = role
    plan['comparison_arm'] = arm
    plan['comparison_variant'] = 'control'
    plan['experiment'] = config['experiment'] + '_' + arm
    common = config['training_common']
    assert plan['optimizer']['value_lr'] == common['value_lr'] and plan['optimizer']['clip_ratio'] == common['clip_ratio']
    assert plan['optimizer']['kl_target'] == common['kl_target'] and plan['optimizer']['gae_lambda'] == common['gae_lambda']
    if role == 'train':
        settings = config['training_arms'][arm]
        plan['optimizer']['actor_lr'] = settings['actor_lr']
        plan['optimizer']['allow_kl_early_stop'] = True
        plan['eval_instances'] = []
        plan['output'] = config['remote_root'] + f'/training/{arm}/results'
        plan['runtime_storage']['max_persistent_bytes'] = config['training_storage_limit_bytes']
        plan['phase_limits_seconds']['train1'] = config['train_timeout_seconds']
        plan['distributed'] = {'world_size': 4, 'preflight_only': False, 'backend': 'gloo_cpu_reduction',
                               'kl_batch_size': common['kl_batch_size'],
                               'runtime_signature': hashlib.sha256(json.dumps(config['runtime_sha256'], sort_keys=True).encode()).hexdigest()}
        gate = json.loads((OLD / 'kl_batch_gate.json').read_text())
        assert gate['passed'] and gate['selected_batch_size'] == 8
        assert (HERE / 'runtime/kl_batch.py').read_bytes() == (OLD / 'runtime/kl_batch.py').read_bytes()
    else:
        assert case in arm_cases(arm), (arm, case)
        plan['output'] = config['remote_root'] + f'/evaluations/{arm}/i{case[0]}_s{case[1]}/results'
        plan['comparison_checkpoint'] = checkpoint_dir(arm)
        plan['evaluate_original_policy'] = plan['comparison_checkpoint'] is None
        plan['evaluation']['cases'] = [case]
        plan['evaluation_runtime_order'] = [case]
        plan['eval_instances'] = [case[0]]
        plan['runtime_storage']['max_persistent_bytes'] = config['evaluation_storage_limit_bytes']
        plan['phase_limits_seconds']['final'] = config['eval_timeout_seconds']
        assert plan['model_seed'] == plan['seed'] == 0
    # Identical to the modem cross-task admission block; the bootstrap writes our local
    # quota proof as storage_inventory.json first, so no account-wide scan runs on a GPU.
    plan['storage_admission'] = json.loads((CROSS / 'plan.json').read_text())['storage_admission']
    plan['runtime_storage']['shared_budget_roots'] = [plan['output']]
    return plan


def storage_proof(plan):
    config = campaign()
    cache = HERE / 'quota_latest.json'
    quota = json.loads(cache.read_text()) if cache.exists() else None
    if quota is None or not 0 <= time.time() - quota['observed_at'] <= 180:
        for key in list(os.environ):
            if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
                os.environ.pop(key)
        quota = single.query_quota()
        save(cache, quota)
    row = quota['rows'][0]
    reserve, margin = config['campaign_storage_reserve_bytes'], config['storage_margin_bytes']
    proof = {'complete': True, 'passed': row['quota_bytes'] - row['used_bytes'] >= reserve + margin,
             'experiment': plan['experiment'], 'account_root': '/mnt/afs/260010165',
             'account_allocated_bytes': row['used_bytes'], 'quota_lower_bound_bytes': row['quota_bytes'],
             'new_task_reserve_bytes': reserve, 'other_remaining_commitments_bytes': 0,
             'headroom_margin_bytes': margin, 'required_remaining_bytes': reserve + margin,
             'observed_at': quota['observed_at'], 'quota_source': quota['method'],
             'scope': '110 GiB: three 16 GiB learners plus 900 evaluation outputs (<=512 MiB cap each, small in practice)'}
    assert proof['passed'], 'Wait for storage headroom; never remove protected inputs'
    return proof


def identity(role, arm, case, attempt):
    return f'train-{arm}-a{attempt}' if role == 'train' else f'eval-{arm}-{case[0]}s{case[1]}-a{attempt}'


def bundle_for(role, plan):
    config = campaign()
    for name, digest in config['runtime_sha256'].items():
        assert sha_file(HERE / 'runtime' / name) == digest, name
    contents = {n: (HERE / 'runtime' / n).read_bytes() for n in config['runtime_sha256']}
    contents['driver.py'] = contents['distributed_driver.py' if role == 'train' else 'comparison_driver.py']
    names = (TRAIN_FILES if role == 'train' else EVAL_FILES) + PACKAGE
    contents = {name: contents[name] for name in names}
    wire = dict(plan)
    for key in ['authorization', 'resume_authorization', 'dependencies', 'dependency_status_file',
                'dependency_campaign_directory', 'previous_generation_directory', 'startup_repair', 'rules',
                'comparisons', 'baseline_sources']:
        wire.pop(key, None)
    contents['plan.json'] = json.dumps(wire, sort_keys=True).encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz', compresslevel=9) as archive:
        for name, body in contents.items():
            item = tarfile.TarInfo('run/' + name)
            item.size = len(body)
            item.mtime = 0
            archive.addfile(item, io.BytesIO(body))
    return stream.getvalue()


def startup_command(role, ident, attempt, bundle, proof, signed):
    config = campaign()
    assets = json.loads((ADAPT / 'cloud_smoke/artifacts.json').read_text())
    before = json.loads((ADAPT / 'cloud_smoke/local_preflight.json').read_text())
    reuse = {'checkpoint/' + k: v['sha256'] for k, v in assets['checkpoint'].items()}
    reuse['bundle.tar.gz'] = assets['bundle']['sha256']
    reuse.update({k: v for k, v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
    for path in ['rlinf/algorithms/advantages.py', 'rlinf/algorithms/losses.py', 'rlinf/models/embodiment/openpi/tasks/rl.py']:
        reuse['upstream/' + path] = sha_file(ADAPT / 'upstream' / path)
    name = config['job_prefix'] + ident
    startup = {
        'root': config['remote_root'] + '/jobs/' + ident,
        'base': '/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
        'task_phase': 'train1' if role == 'train' else 'final', 'rank': 0, 'attempt': attempt, 'job_name': name,
        'bundle': {'bytes': len(bundle), 'sha256': hashlib.sha256(bundle).hexdigest()},
        'reuse_sha256': reuse, 'status_put': signed('put_object', 'status.json'),
        'result_put': signed('put_object', 'result.tar.gz'), 'artifact_puts': {}, 'storage_proof': proof,
        'bundle_inline': base64.b64encode(bundle).decode(),
    }
    bootstrap = (HERE / 'runtime' / ('bootstrap.py' if role == 'train' else 'single_bootstrap.py')).read_text()
    source = bootstrap + '\nguarded_run(' + repr(startup) + ')\n'
    packed = base64.b64encode(zlib.compress(source.encode(), level=9)).decode()
    limit = config['train_timeout_seconds' if role == 'train' else 'eval_timeout_seconds']
    command = f'timeout --signal=TERM --kill-after=60s {limit - 60}s python3 -u -c ' + shlex.quote(
        'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
    assert len(command) < 64000, len(command)
    return name, command, limit


def _submit_unlocked(role, arm, case=None, attempt=1, quota='spot'):
    config = campaign()
    assert quota in ('spot', 'reserved') and (role == 'eval' or quota == 'spot')
    ident = identity(role, arm, case, attempt)
    folder = HERE / 'jobs' / ident
    receipt = folder / 'submission.json'
    assert not receipt.exists(), 'Reconcile the exact existing name; never duplicate a submission'
    verification = json.loads((HERE / 'verification.json').read_text())
    assert verification['passed'] and verification['plan_sha256'] == sha_file(HERE / 'plan.json')
    plan = worker_plan(role, arm, case)
    if role == 'train' and attempt > 1:
        accepted = [json.loads(p.read_text()).get('progress', {}).get('accepted_optimizer_steps', 0)
                    for p in (HERE / 'jobs').glob(f'train-{arm}-a*/latest.json')]
        if max(accepted, default=0) > 0:
            plan['distributed']['require_resume'] = True
    proof = storage_proof(plan)
    template = describe(TEMPLATE)
    assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
    folder.mkdir(parents=True, exist_ok=True)
    save(folder / 'worker_plan.json', plan)
    bundle = bundle_for(role, plan)
    prefix = config['aoss_prefix'] + 'jobs/' + ident + '/'
    client, signing_client = aoss_client(), status_client()
    try:
        client.put_object(Bucket='260010165', Key=prefix + 'bundle.tar.gz', Body=bundle)
        assert client.get_object(Bucket='260010165', Key=prefix + 'bundle.tar.gz')['Body'].read() == bundle

        def signed(method, item):
            return signing_client.generate_presigned_url(method, Params={'Bucket': '260010165', 'Key': prefix + item},
                                                         ExpiresIn=5 * 86400)
        name, command, limit = startup_command(role, ident, attempt, bundle, proof, signed)
        row = {'id': ident, 'job': name, 'role': role, 'arm': arm, 'attempt': attempt,
               'gpu_count': 4 if role == 'train' else 1, 'quota': quota, 'aoss_prefix': prefix,
               'output': plan['output'], 'max_seconds': limit, 'state': 'submitting', 'submitted_at': time.time(),
               'bundle_bytes': len(bundle), 'startup_sha256': hashlib.sha256(command.encode()).hexdigest(),
               'runtime_sha256': config['runtime_sha256']}
        if case is not None:
            row['case'] = case
        if role == 'train':
            row['actor_lr'] = plan['optimizer']['actor_lr']
            row['require_resume'] = bool(plan['distributed'].get('require_resume'))
        with receipt.open('x') as output:
            json.dump(row, output, ensure_ascii=False, indent=2)
        result = subprocess.run([
            CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space', '--copy-job-name=' + TEMPLATE,
            '--name=' + name, '--job-name=' + name,
            '--worker-spec=' + ('n12lp.nn.i10a.4' if role == 'train' else 'n12lp.nn.i10a.1'),
            '--worker-nodes=1', '--quota-type=' + quota, '--retry-times=0',
            '--enable-fault-tolerance=false', '--enable-anomaly-detection=false', '--command=' + command,
        ], env=ENV, capture_output=True, text=True, timeout=90)
        row['submit_returncode'] = result.returncode
        if result.returncode:  # keep the API's reason (URLs redacted) for diagnosis
            row['submit_error'] = re.sub(r'https?://\S+', '<url>', result.stderr or result.stdout)[-600:]
        save(receipt, row)
        job = describe(name)
        assert job['scheduling']['quota_type'] == quota.upper(), job['scheduling']
        assert hashlib.sha256(job['roles'][0]['startup_script'].encode()).hexdigest() == row['startup_sha256']
        row.update(uid=job['uid'], state=job['state'])
        save(receipt, row)
        print(json.dumps({k: row[k] for k in ['job', 'uid', 'state', 'gpu_count', 'quota', 'bundle_bytes']}), flush=True)
        return row
    finally:
        client.close()
        signing_client.close()


def submit(role, arm, case=None, attempt=1, admission=None):
    """Shared host lock (same file as the other BEHAVIOR controllers) around admission + submit."""
    import fcntl
    with (ANALYSIS / 'rl_gpu_submit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        quota = admission(role)  # raises RuntimeError when not admissible
        return _submit_unlocked(role, arm, case, attempt, quota)
