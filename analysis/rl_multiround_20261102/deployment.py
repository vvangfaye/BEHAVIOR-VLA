"""Job construction and submission for rl_multiround_20261102 (multi-round online PPO).

Roles
  sample : 1 x RTX 5090 (Spot), sampler_driver.py collects assigned episodes with policy k-1.
  learn  : 4 GPUs on the configured learner pool (H100 'share-cluster' or 5090), distributed_driver.py
           with the fast learner, NCCL, tolerance replay and enforced on-policy receipts.
  eval   : 1 x RTX 5090, comparison_driver.py evaluates one case for one published policy.
Bundles are inline in the startup command (no download at start), exactly like the earlier campaigns.
"""
import base64
import hashlib
import importlib.util
import io
import json
import os
import re
from pathlib import Path
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
ADAPT = ANALYSIS / 'rlinf_adaptation_20261020'
CROSS = ANALYSIS / 'rl_cross_task_20261102/modem'
OLD = ANALYSIS / 'rl_multigpu_20261101'
spec = importlib.util.spec_from_file_location('frozen_comparison_deployment', SOURCE / 'deployment.py')
single = importlib.util.module_from_spec(spec)
spec.loader.exec_module(single)
CLI, ENV, describe, save, aoss_client = single.CLI, single.ENV, single.describe, single.save, single.aoss_client
TERMINAL = {'SUCCEEDED', 'FAILED', 'SUSPENDED', 'STOPPED', 'TERMINATED', 'CANCELED'}

POOLS = {
    '5090': {'template': 'b1k-repeat1688-1019-batch-00', 'pool': 'computing-cluster-5090-01g',
             'spec': {1: 'n12lp.nn.i10a.1', 4: 'n12lp.nn.i10a.4'}, 'gpu_models': ['5090', 'N12LP']},
    'h100': {'template': 'b1k-update-composition-1022-plumbing06h100', 'pool': 'share-cluster',
             'spec': {1: 'n6ls.iu.i40.1', 4: 'n6ls.iu.i40.4'}, 'gpu_models': ['H100', 'N6LS']},
}
EVAL_FILES = ['common.py', 'training.py', 'experiment.py', 'simulator.py', 'helpers.py', 'eval_utils.py',
              'resource_watch.py', 'goals.py', 'storage_guard.py', 'capacity_admission.py', 'variant_update.py', 'driver.py']
SAMPLE_FILES = ['common.py', 'training.py', 'experiment.py', 'simulator.py', 'helpers.py', 'resource_watch.py',
                'goals.py', 'storage_guard.py', 'capacity_admission.py', 'driver.py']
# experiment.py is not needed by the learner (its one helper, seed_all, is inlined in distributed_driver.py);
# dropping it makes room for prefix_cache.py under the 64 KiB inline-command limit.
TRAIN_FILES = ['common.py', 'training.py', 'variant_update.py', 'parallel_training.py', 'goals.py', 'prefix_cache.py',
               'storage_guard.py', 'capacity_admission.py', 'resource_watch.py', 'kl_batch.py', 'driver.py']
PACKAGE = ['b1k_rlinf/' + n + '.py' for n in ['__init__', 'action_interface', 'factory', 'io', 'model', 'precision', 'tokenizer']]
RECEIPT_KEYS = ['episode', 'instance', 'seed', 'path', 'sha256', 'policy_hash', 'contract', 'chunks']
DRIVERS = {'learn': 'distributed_driver.py', 'sample': 'sampler_driver.py', 'eval': 'comparison_driver.py'}


def campaign():
    return json.loads((HERE / 'plan.json').read_text())


def eval_cases_for(k):
    """Cases evaluated for policy k: eval_cases, or final_eval_cases for the last round (user 2026-11-03: 200)."""
    config = campaign()
    return config.get('final_eval_cases', config['eval_cases']) if k == config['rounds'] else config['eval_cases']


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status_client():
    sys.path.insert(0, str(ADAPT / 'multitask_validation_20261029/capacity_repair_20261030'))
    from internal_storage import internal_aoss_client
    return internal_aoss_client()


# ---------------------------------------------------------------- paths of the round protocol

def sampling_root(k):
    return campaign()['remote_root'] + f'/rounds/round_{k:02d}/sampling'


def learner_root(k):
    return campaign()['remote_root'] + f'/rounds/round_{k:02d}/learner'


def policy_record(k):
    """Published policy k (k >= 1) as recorded locally after the learner job was collected; None for k = 0."""
    if k == 0:
        return None
    path = HERE / 'policies' / f'round_{k:02d}.json'
    return json.loads(path.read_text()) if path.exists() else None


def episode_layout():
    config = campaign()
    instances = [i for i in config['train_instances'] for _ in range(config['rollouts_per_instance'])]
    seeds = [[config['seed_base'] + 1000 * k + j for j in range(len(instances))] for k in range(1, config['rounds'] + 1)]
    return instances, seeds


# ---------------------------------------------------------------- worker plans

def base_plan():
    config = campaign()
    assert sha_file(CROSS / 'source_plan.json') == config['source_plan_sha256']
    plan = json.loads((CROSS / 'source_plan.json').read_text())
    plan['source_reward_contract_sha256'] = hashlib.sha256(json.dumps(
        {'goals': plan['goal_contract'], 'shaping': plan['shaping']}, sort_keys=True).encode()).hexdigest()
    if 'task_bddl_sha256' not in plan['source_identity']:
        plan['source_identity']['task_bddl_sha256'] = plan['goal_contract']['bddl_sha256']
    plan['storage_admission'] = json.loads((CROSS / 'plan.json').read_text())['storage_admission']
    plan.update(worker_prefix='mr_', baseline_sources={}, comparison_variant='control')
    plan['optimizer'].update(actor_lr=config['actor_lr'], value_lr=config['value_lr'], gamma=config['gamma'],
                             gae_lambda=config['gae_lambda'], updates_per_round=[config['optimizer_steps']],
                             allow_kl_early_stop=True)
    plan['shaping']['gamma'] = config['gamma']
    instances, seeds = episode_layout()
    plan['train_instances'], plan['train_seeds'] = instances, seeds
    plan['rounds'] = config['rounds']
    return plan


def worker_plan(role, k, payload):
    """payload: sample -> {'sampler_id', 'episodes'}; learn -> {'receipts'}; eval -> {'case'}."""
    config = campaign()
    plan = base_plan()
    plan['experiment'] = f"{config['experiment']}_{role}_r{k:02d}"
    plan['round_index'] = k
    if role == 'sample':
        previous = policy_record(k - 1)
        assert k == 1 or previous is not None, 'Round k samples only with the published policy k-1'
        plan['sampling_policy'] = None if k == 1 else {'directory': previous['directory'],
                                                       'trainable_sha256': previous['trainable_sha256']}
        plan['sampler_id'] = payload['sampler_id']
        plan['assigned_episodes'] = payload['episodes']
        plan['eval_instances'] = []
        plan['output'] = sampling_root(k)
        plan['phase_limits_seconds']['sample1'] = config['sample_timeout_seconds']
        plan['runtime_storage']['max_persistent_bytes'] = config['round_sampling_storage_bytes']
    elif role == 'learn':
        receipts = payload['receipts']
        assert config['min_episodes_per_round'] <= len(receipts) <= config['episodes_per_round']
        previous = policy_record(k - 1)
        expected_policy = previous['trainable_sha256'] if previous else base_plan()['source_identity']['cold_trainable_hash']
        assert all(r['policy_hash'] == expected_policy and r['round'] == k for r in receipts), 'Receipts are not on-policy'
        plan['initial_policy'] = None if previous is None else {
            'directory': previous['directory'], 'trainable_sha256': previous['trainable_sha256'],
            'load_optimizer': config['carry_optimizer']}
        plan['previous_cumulative_optimizer_steps'] = 0 if previous is None else previous['cumulative_optimizer_steps']
        plan['expected_episodes'] = len(receipts)
        plan['source_rollout_root'] = sampling_root(k)
        listed = [{key: r[key] for key in RECEIPT_KEYS} for r in sorted(receipts, key=lambda r: r['episode'])]
        # Receipts stay on AFS (written by the samplers); the plan carries only their canonical digest.
        plan['source_rollout_manifest'] = {
            'receipt_directory': sampling_root(k) + f'/training_receipts/round_{k:02d}',
            'episodes': [r['episode'] for r in listed], 'count': len(listed),
            'sha256': hashlib.sha256(json.dumps(listed, sort_keys=True).encode()).hexdigest()}
        plan['source_rollout_receipts'] = []
        plan['eval_instances'] = []
        plan['output'] = learner_root(k)
        plan['phase_limits_seconds']['train1'] = config['learner_timeout_seconds']
        plan['runtime_storage']['max_persistent_bytes'] = config['learner_storage_bytes']
        pool = POOLS[config['learner_pool']]
        plan['distributed'] = {
            'world_size': 4, 'preflight_only': False, 'backend': config['collective_backend'][config['learner_pool']],
            'kl_batch_size': config['kl_batch_size'], 'gpu_models': pool['gpu_models'], 'learner_only': True,
            'enforce_on_policy': True, 'replay_tolerance': config['replay_tolerance'], 'fast': config['fast_learner'],
            'runtime_signature': hashlib.sha256(json.dumps(config['runtime_sha256'], sort_keys=True).encode()).hexdigest()}
        attempts = [json.loads(p.read_text()) for p in (HERE / 'jobs').glob(f'learn-r{k:02d}-a*/latest.json')]
        if any((a.get('progress') or {}).get('accepted_optimizer_steps', 0) > 0 for a in attempts):
            plan['distributed']['require_resume'] = True
    elif role == 'eval':
        case = payload['case']
        assert case in eval_cases_for(k)
        policy = policy_record(k)
        assert k == 0 or policy is not None
        plan['comparison_checkpoint'] = None if k == 0 else policy['directory']
        plan['evaluate_original_policy'] = k == 0
        plan['comparison_arm'] = f'policy_r{k:02d}'
        plan['evaluation']['cases'] = [case]
        plan['evaluation_runtime_order'] = [case]
        plan['eval_instances'] = [case[0]]
        plan['output'] = config['remote_root'] + f'/evaluations/policy_r{k:02d}/i{case[0]}_s{case[1]}/results'
        plan['phase_limits_seconds']['final'] = config['eval_timeout_seconds']
        plan['runtime_storage']['max_persistent_bytes'] = config['evaluation_storage_bytes']
        assert plan['model_seed'] == plan['seed'] == 0
    else:
        raise ValueError(role)
    plan['runtime_storage']['shared_budget_roots'] = [plan['output']]
    return plan


# ---------------------------------------------------------------- bundles and startup


def minify_source(source):
    """Deterministic, semantics-preserving size reduction for the learner bundle only: drop docstrings and
    comments via an AST round trip (Python 3.11 here and on the cluster). Tested by tests/run_minified_tests.py."""
    import ast
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                node.body = node.body[1:] or [ast.Pass()]
    return ast.unparse(tree) + '\n'


def bundle_for(role, plan):
    config = campaign()
    for name, digest in config['runtime_sha256'].items():
        assert sha_file(HERE / 'runtime' / name) == digest, name
    contents = {n: (HERE / 'runtime' / n).read_bytes() for n in config['runtime_sha256']}
    contents['driver.py'] = contents[DRIVERS[role]]
    names = {'learn': TRAIN_FILES, 'sample': SAMPLE_FILES, 'eval': EVAL_FILES}[role] + PACKAGE
    contents = {name: contents[name] for name in names}
    if role == 'learn':
        contents = {name: minify_source(body.decode()).encode() for name, body in contents.items()}
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
             'scope': 'multi-round campaign reserve (rollouts, learner checkpoints, evaluation outputs)'}
    assert proof['passed'], 'Wait for storage headroom; never remove protected inputs'
    return proof


def startup_command(role, ident, attempt, bundle, proof, signed, limit):
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
        'task_phase': {'learn': 'train1', 'sample': 'sample1', 'eval': 'final'}[role], 'rank': 0, 'attempt': attempt,
        'job_name': name, 'bundle': {'bytes': len(bundle), 'sha256': hashlib.sha256(bundle).hexdigest()},
        'reuse_sha256': reuse, 'status_put': signed('put_object', 'status.json'),
        'result_put': signed('put_object', 'result.tar.gz'), 'artifact_puts': {}, 'storage_proof': proof,
        'bundle_inline': base64.b64encode(bundle).decode(),
    }
    bootstrap = (HERE / 'runtime' / ('bootstrap.py' if role == 'learn' else 'single_bootstrap.py')).read_text()
    source = bootstrap + '\nguarded_run(' + repr(startup) + ')\n'
    packed = base64.b64encode(zlib.compress(source.encode(), level=9)).decode()
    command = f'timeout --signal=TERM --kill-after=60s {limit - 60}s python3 -u -c ' + shlex.quote(
        'import base64,zlib;exec(zlib.decompress(base64.b64decode(' + repr(packed) + ')))')
    assert len(command) < 64000, len(command)
    return name, command


def identity(role, k, payload, attempt):
    if role == 'sample':
        return f"sample-r{k:02d}-{payload['sampler_id']}-a{attempt}"
    if role == 'learn':
        return f'learn-r{k:02d}-a{attempt}'
    return f"eval-p{k:02d}-{payload['case'][0]}s{payload['case'][1]}-a{attempt}"


def submit_unlocked(role, k, payload, attempt, quota):
    """Caller holds analysis/rl_gpu_submit.lock and has passed admission."""
    config = campaign()
    pool_name = config['learner_pool'] if role == 'learn' else '5090'
    pool = POOLS[pool_name]
    gpus = 4 if role == 'learn' else 1
    assert quota in ('spot', 'reserved')
    ident = identity(role, k, payload, attempt)
    folder = HERE / 'jobs' / ident
    receipt = folder / 'submission.json'
    assert not receipt.exists(), 'Reconcile the exact existing name; never duplicate a submission'
    verification = json.loads((HERE / 'verification.json').read_text())
    assert verification['passed'] and verification['plan_sha256'] == sha_file(HERE / 'plan.json')
    plan = worker_plan(role, k, payload)
    proof = storage_proof(plan)
    template = describe(pool['template'])
    assert template['resource_pool']['name'] == pool['pool']
    limit = config[{'learn': 'learner_timeout_seconds', 'sample': 'sample_timeout_seconds', 'eval': 'eval_timeout_seconds'}[role]]
    assert gpus * limit / 3600 <= 128, 'Cluster rule: at most 128 GPU-hours per job'
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
        name, command = startup_command(role, ident, attempt, bundle, proof, signed, limit)
        row = {'id': ident, 'job': name, 'role': role, 'round': k, 'payload': payload, 'attempt': attempt,
               'pool': pool_name, 'gpu_count': gpus, 'quota': quota, 'aoss_prefix': prefix, 'output': plan['output'],
               'max_seconds': limit, 'state': 'submitting', 'submitted_at': time.time(), 'bundle_bytes': len(bundle),
               'startup_sha256': hashlib.sha256(command.encode()).hexdigest()}
        if role == 'learn':
            row['require_resume'] = bool(plan['distributed'].get('require_resume'))
        with receipt.open('x') as output:
            json.dump(row, output, ensure_ascii=False, indent=2)
        result = subprocess.run([
            CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space', '--copy-job-name=' + pool['template'],
            '--name=' + name, '--job-name=' + name, '--worker-spec=' + pool['spec'][gpus], '--worker-nodes=1',
            '--quota-type=' + quota, '--retry-times=0', '--enable-fault-tolerance=false',
            '--enable-anomaly-detection=false', '--command=' + command,
        ], env=ENV, capture_output=True, text=True, timeout=90)
        row['submit_returncode'] = result.returncode
        if result.returncode:  # keep the API's reason (URLs redacted) for diagnosis
            row['submit_error'] = re.sub(r'https?://\S+', '<url>', result.stderr or result.stdout)[-600:]
        save(receipt, row)
        job = describe(name)
        assert job['scheduling']['quota_type'] == quota.upper(), job['scheduling']
        assert job['resource_pool']['name'] == pool['pool']
        assert hashlib.sha256(job['roles'][0]['startup_script'].encode()).hexdigest() == row['startup_sha256']
        row.update(uid=job['uid'], state=job['state'])
        save(receipt, row)
        print(json.dumps({k2: row[k2] for k2 in ['job', 'uid', 'state', 'pool', 'gpu_count', 'quota']}), flush=True)
        return row
    finally:
        client.close()
        signing_client.close()
