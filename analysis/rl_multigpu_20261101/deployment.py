"""Immutable four-GPU job bundles; the existing controllers retain evaluation."""
import base64
import hashlib
import io
import importlib.util
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
COMPARISON = PROJECT/'analysis/rl_scheme_comparison_20261101'
WATER = PROJECT/'analysis/rlinf_adaptation_20261020/multitask_validation_20261029/water'
spec = importlib.util.spec_from_file_location('single_gpu_deployment', COMPARISON/'deployment.py')
single = importlib.util.module_from_spec(spec)
spec.loader.exec_module(single)
CLI, ENV, describe, save, aoss_client = single.CLI, single.ENV, single.describe, single.save, single.aoss_client
REMOTE = '/mnt/afs/260010165/behavior_1k/rl_multigpu_20261101'


def worker_plan(variant, attempt, preflight=False):
    plan = single.worker_plan({'role': 'train', 'variant': variant})
    plan['comparison_variant'] = variant
    plan['distributed'] = {'world_size': 4, 'preflight_only': preflight, 'backend': 'gloo_cpu_reduction'}
    if variant == 'control':
        original = json.loads((WATER/'plan.json').read_text())
        plan['output'] = original['output']
        plan['worker_prefix'] = original['worker_prefix']
    if preflight:
        plan['output'] = REMOTE+'/parity/results'
        records = json.loads((COMPARISON/'source_training_receipts.json').read_text())
        for instance in range(1, 17):
            pair = [r for r in records if r['instance'] == instance]
            if len({r['status']['success'] for r in pair}) == 2:
                pair.sort(key=lambda r: not r['status']['success'])
                selected = {r['episode'] for r in pair}
                plan['source_rollout_receipts'] = [r for episode in [r['episode'] for r in pair]
                    for r in plan['source_rollout_receipts'] if r['episode'] == episode]
                break
        assert len(plan['source_rollout_receipts']) == 2
    plan['runtime_storage']['max_persistent_bytes'] = (1 if preflight else 28 if variant == 'control' else 16)*1024**3
    plan['runtime_storage']['shared_budget_roots'] = [plan['output']]
    plan['phase_limits_seconds']['train1'] = 3600 if preflight else 43200
    sources = {str(p.relative_to(HERE/'runtime')): hashlib.sha256(p.read_bytes()).hexdigest()
               for p in (HERE/'runtime').rglob('*.py')}
    plan['distributed']['runtime_signature'] = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return plan


def storage_proof(plan):
    quota = single.query_quota()
    row = quota['rows'][0]
    committed = 0
    ready_files = list((COMPARISON/'checkpoints').glob('*.json'))
    control = WATER/'retrieved'
    ready_files += list(control.glob('train1*/checkpoints/round_01/ready.json'))
    identities = set()
    for path in ready_files:
        data = json.loads(path.read_text())
        key = data.get('trainable_sha256')
        if key and key not in identities:
            committed += data.get('model_bytes', 0)+data.get('optimizer_bytes', 0)
            identities.add(key)
    # Once the numerical gate has read and hashed all four saved snapshots,
    # their already allocated bytes must not be reserved a second time.
    # This credit applies only before final checkpoints enter ready_files.
    gate_path = HERE/'kl_batch_gate.json'
    if not ready_files and gate_path.exists():
        gate = json.loads(gate_path.read_text())
        assert gate['passed'] and gate['training_state_unchanged']
        sources = json.loads((HERE/'kl_batch_plan.json').read_text())['sources']
        committed = sum(s['metadata']['model_bytes']+s['metadata']['optimizer_bytes'] for s in sources.values())
    reserve = max(plan['runtime_storage']['max_persistent_bytes'], 52*1024**3-committed)
    margin = 20*1024**3
    result = {'complete': True, 'passed': row['quota_bytes']-row['used_bytes'] >= reserve+margin,
        'experiment': plan['experiment'], 'account_root': '/mnt/afs/260010165',
        'account_allocated_bytes': row['used_bytes'], 'quota_lower_bound_bytes': row['quota_bytes'],
        'new_task_reserve_bytes': reserve, 'other_remaining_commitments_bytes': 0,
        'headroom_margin_bytes': margin, 'required_remaining_bytes': reserve+margin,
        'observed_at': quota['observed_at'], 'quota_source': quota['method'],
        'scope': '52 GiB shared remaining-output allowance for four learners and fixed evaluations; input rollouts already stored'}
    assert result['passed'], 'Wait for shared training storage headroom'
    return result


def register_with_evaluator(row, plan):
    variant, attempt = row['variant'], row['attempt']
    destination = (WATER/'jobs'/f'train1_r0_a{attempt}' if variant == 'control' else
                   COMPARISON/'jobs'/f't-{variant.replace("_", "-")}-a{attempt}')
    destination.mkdir(parents=True, exist_ok=True)
    path=destination/'submission.json'
    if path.exists():
        assert json.loads(path.read_text())['uid']==row['uid']
    else:
        save(path,row)
        save(destination/'worker_plan.json',plan)


def submit(variant, attempt=3, preflight=False, benchmark=False):
    kind = 'kl-benchmark' if benchmark else 'parity' if preflight else variant.replace('_', '-')
    ident = f'{kind}-a{attempt}' if preflight or benchmark else f't-{kind}-a{attempt}'
    folder = HERE/'jobs'/ident
    receipt = folder/'submission.json'
    assert not receipt.exists(), 'Reconcile existing job; do not duplicate'
    assert json.loads((HERE/'cpu_tests_result.json').read_text())['passed']
    communication = json.loads((HERE/'selected_communication.json').read_text())
    assert communication['passed']
    if not preflight and not benchmark:
        assert json.loads((HERE/'parity_result.json').read_text())['passed']
    plan = worker_plan(variant, attempt, preflight)
    plan['distributed']['backend'] = communication['backend']
    if benchmark:
        plan['output'] = '/tmp/b1k-kl-benchmark-a'+str(attempt)+'/results'
        plan['phase_limits_seconds']['train1'] = 10800
        plan['runtime_storage']['max_persistent_bytes'] = 2*1024**3
        plan['runtime_storage']['shared_budget_roots'] = [plan['output']]
        plan['kl_benchmark'] = json.loads((HERE/'kl_batch_plan.json').read_text())
    elif not preflight:
        gate_path = HERE/'kl_batch_gate.json'
        if gate_path.exists():
            gate = json.loads(gate_path.read_text())
            assert gate['passed'] and gate['training_state_unchanged']
            assert plan['distributed']['runtime_signature'] == gate['runtime_signature']
            plan['distributed']['kl_batch_size'] = gate['selected_batch_size']
            plan['distributed']['require_resume'] = True
            source = json.loads((HERE/'kl_batch_plan.json').read_text())['sources'][variant]
            plan['distributed']['resume_predecessor'] = {
                'metadata': source['metadata'], 'files': gate['variants'][variant]['source_files']}
        else:
            assert plan['distributed']['runtime_signature'] == json.loads((HERE/'parity_result.json').read_text())['runtime_signature']
    proof = ({'scope': 'benchmark reads AFS only; outputs to node /tmp and AOSS',
              'passed': True, 'afs_growth_bytes': 0} if benchmark else storage_proof(plan))
    template = describe('b1k-repeat1688-1019-batch-00')
    assert template['resource_pool']['name'] == 'computing-cluster-5090-01g'
    folder.mkdir(parents=True, exist_ok=True)
    save(folder/'worker_plan.json', plan)
    names = ['common.py', 'training.py', 'variant_update.py', 'parallel_training.py', 'experiment.py',
             'goals.py', 'helpers.py', 'eval_utils.py', 'storage_guard.py', 'capacity_admission.py', 'resource_watch.py', 'kl_batch.py']
    names += ['b1k_rlinf/'+n+'.py' for n in ['__init__', 'action_interface', 'factory', 'io', 'model', 'precision', 'tokenizer']]
    content = {n: (HERE/'runtime'/n).read_bytes() for n in names}
    content['driver.py'] = (HERE/('benchmark_driver.py' if benchmark else 'runtime/distributed_driver.py')).read_bytes()
    content['plan.json'] = json.dumps(plan, sort_keys=True).encode()
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, body in content.items():
            item = tarfile.TarInfo('run/'+name)
            item.size = len(body)
            item.mtime = 0
            archive.addfile(item, io.BytesIO(body))
    bundle = stream.getvalue()
    prefix = 'behavior1k/rl-multigpu-20261101/jobs/'+ident+'/'
    job_name = 'b1k-4gpu-1101-'+ident
    client = aoss_client()
    try:
        client.put_object(Bucket='260010165', Key=prefix+'bundle.tar.gz', Body=bundle)
        assert client.get_object(Bucket='260010165', Key=prefix+'bundle.tar.gz')['Body'].read() == bundle
        def signed(method, name):
            return client.generate_presigned_url(method, Params={'Bucket': '260010165', 'Key': prefix+name}, ExpiresIn=5*86400)
        adapt = PROJECT/'analysis/rlinf_adaptation_20261020'
        assets = json.loads((adapt/'cloud_smoke/artifacts.json').read_text())
        before = json.loads((adapt/'cloud_smoke/local_preflight.json').read_text())
        reuse = {'checkpoint/'+k: v['sha256'] for k, v in assets['checkpoint'].items()}
        reuse['bundle.tar.gz'] = assets['bundle']['sha256']
        reuse.update({k: v for k, v in before['execution_source_sha256'].items() if k.startswith('adapter/')})
        for name in ['rlinf/algorithms/advantages.py', 'rlinf/algorithms/losses.py', 'rlinf/models/embodiment/openpi/tasks/rl.py']:
            reuse['upstream/'+name] = hashlib.sha256((adapt/'upstream'/name).read_bytes()).hexdigest()
        config = {'root': ('/tmp/b1k-kl-benchmark-a'+str(attempt) if benchmark else REMOTE+'/jobs/'+ident), 'base': '/mnt/afs/260010165/behavior_1k/rlinf_adapter_20261020/smoke01',
            'task_phase': 'train1', 'rank': 0, 'attempt': attempt, 'job_name': job_name,
            'bundle_get': signed('get_object', 'bundle.tar.gz'),
            'bundle': {'bytes': len(bundle), 'sha256': hashlib.sha256(bundle).hexdigest()},
            'reuse_sha256': reuse, 'status_put': signed('put_object', 'status.json'),
            'result_put': signed('put_object', 'result.tar.gz'), 'artifact_puts': {}, 'storage_proof': proof}
        source = (HERE/'runtime/bootstrap.py').read_text()+'\nguarded_run('+repr(config)+')\n'
        packed = base64.b64encode(zlib.compress(source.encode())).decode()
        limit = plan['phase_limits_seconds']['train1']
        command = f'timeout --signal=TERM --kill-after=60s {limit-60}s python3 -u -c '+shlex.quote(
            'import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
        assert len(command) < 64000
        row = {'id': ident, 'job': job_name, 'variant': variant, 'role': 'train', 'phase': 'train1', 'rank': 0,
            'attempt': attempt, 'gpu_count': 4, 'quota': 'spot', 'aoss_prefix': prefix, 'output': plan['output'],
            'max_seconds': limit, 'state': 'submitting', 'submitted_at': time.time(),
            'source_sha256': {k: hashlib.sha256(v).hexdigest() for k, v in content.items()},
            'startup_sha256': hashlib.sha256(command.encode()).hexdigest(), 'preflight': preflight,
            'benchmark': benchmark}
        if benchmark:
            row['role'] = 'benchmark'
        with receipt.open('x') as f:
            json.dump(row, f, indent=2)
        result = subprocess.run([CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
            '--copy-job-name=b1k-repeat1688-1019-batch-00', '--name='+job_name, '--job-name='+job_name,
            '--worker-spec=n12lp.nn.i10a.4', '--worker-nodes=1', '--quota-type=spot', '--retry-times=0',
            '--enable-fault-tolerance=false', '--enable-anomaly-detection=false', '--command='+command],
            env=ENV, capture_output=True, text=True, timeout=90)
        row['submit_returncode'] = result.returncode
        save(receipt, row)
        job = describe(job_name)
        assert job['scheduling']['quota_type'] == 'SPOT' and job['roles'][0]['resource_spec'][0]['name'] == 'n12lp.nn.i10a.4'
        assert hashlib.sha256(job['roles'][0]['startup_script'].encode()).hexdigest() == row['startup_sha256']
        row.update(uid=job['uid'], state=job['state'])
        save(receipt, row)
        if not preflight and not benchmark:
            register_with_evaluator(row, plan)
        return row
    finally:
        client.close()
