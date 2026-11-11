"""Cluster validation for rl_multiround_20261102 (user, 2026-11-03 ~00:05: "现在就用闲时卡跑这三项验证吧").

Jobs (all Spot, fixed scope, one attempt each unless reclaimed):
  A  val-learn-5090-full : fast learner on 4 x 5090, full KL check every step -> must reproduce tonight's lr1e6
                           (same 32 rollouts, LR 1e-6, shuffle seed, 17 updates); records per-phase timing.
  B  val-learn-h100      : production fast config on 4 x H100 -> network, NCCL, microbatch-4 memory, full-round
                           cross-device tolerance replay, H100 timing; compared with the same reference.
  C  val-sample-5090     : sampler_driver on 1 x 5090, one episode with the original policy.
Budget: <= 12 GPU-hours (A: 4 x 2 h, B: 4 x 2 h worst case are timeouts; C: 1 x 1.5 h).
  D  val-learn-h100-cache: (user 2026-11-03, "就按你说的办") the production learner with the per-round prefix
                           cache, full KL after every step and a checkpoint every 4 steps, on 4 x H100 (Spot).
                           Final state vs tonight's lr1e6; state after 8 steps vs H100 run B (r4, same device,
                           uncached). Budget: 4 x 2 h per attempt (expected ~15 min), reclaims resubmitted.

Usage: python validation.py submit|status|analyze
"""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import deployment as d

HERE = d.HERE / 'validation'
CROSS = d.CROSS
REFERENCE = '/mnt/afs/260010165/behavior_1k/rl_eval_power_lr_20261102/training/lr1e6/results/checkpoints/round_01'
REFERENCE_LOCAL = d.ANALYSIS / 'rl_eval_power_lr_20261102/checkpoints/lr1e6.json'
# Validation B (val-learn-h100-r4, uncached, sampled KL) stopped after 8 accepted steps; its round_01 is that state.
H100_UNCACHED_STEP8 = '/mnt/afs/260010165/behavior_1k/rl_multiround_20261102/validation/val-learn-h100-series/results/checkpoints/round_01'
LIMITS = {'learn': 2 * 3600, 'sample': 5400}
# v2 = after the two-part parity gate fix (v1 learners failed the strict AdamW-step comparison).
JOBS = ['val-learn-5090-full', 'val-learn-h100', 'val-sample-5090', 'val-learn-5090-full-v2', 'val-learn-h100-v2', 'val-learn-h100-v3']


def restored_receipts():
    evidence = json.loads((CROSS / 'input_restore_verified.json').read_text())
    assert evidence['passed'] and evidence['all_sha256_verified']
    raw = (CROSS / 'restored_training_receipts.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == evidence['restored_receipts_sha256']
    receipts = json.loads(raw)
    assert len(receipts) == 32 and sum(r['chunks'] for r in receipts) == 5677
    return evidence['restored_root'], [{k: r[k] for k in d.RECEIPT_KEYS} for r in receipts]


def series(tag):
    """Attempts of one logical validation share an output directory (-> automatic resume after reclaim)."""
    return tag.rsplit('-r', 1)[0] + '-series' if '-r' in tag.split('val-learn-')[-1] else tag


def learner_plan(tag, pool, fast):
    config = d.campaign()
    plan = d.base_plan()
    source = json.loads((CROSS / 'source_plan.json').read_text())
    root, receipts = restored_receipts()
    cold = plan['source_identity']['cold_trainable_hash']
    assert all(r['policy_hash'] == cold for r in receipts)
    plan.update(experiment=f'rl_multiround_validation_{tag}', round_index=1, initial_policy=None,
                previous_cumulative_optimizer_steps=0, expected_episodes=32,
                train_instances=source['train_instances'], train_seeds=source['train_seeds'],
                source_rollout_root=root, source_rollout_receipts=receipts, eval_instances=[],
                reference_checkpoint=REFERENCE, output=config['remote_root'] + f'/validation/{series(tag)}/results')
    # Identical optimisation contract to tonight's lr1e6 learner (rl_eval_power_lr_20261102).
    assert plan['optimizer']['shuffle_seed'] == source['optimizer']['shuffle_seed']
    plan['optimizer'].update(actor_lr=1e-6, updates_per_round=[17], allow_kl_early_stop=True, gamma=0.99)
    plan['phase_limits_seconds']['train1'] = LIMITS['learn']
    plan['runtime_storage']['max_persistent_bytes'] = 24 * 1024**3
    plan['runtime_storage']['shared_budget_roots'] = [plan['output']]
    plan['distributed'] = {
        'world_size': 4, 'preflight_only': False, 'backend': config['collective_backend'][pool], 'kl_batch_size': config['kl_batch_size'],
        'gpu_models': d.POOLS[pool]['gpu_models'], 'learner_only': True, 'enforce_on_policy': True,
        'replay_tolerance': config['replay_tolerance'], 'fast': fast,
        'runtime_signature': hashlib.sha256(json.dumps(config['runtime_sha256'], sort_keys=True).encode()).hexdigest()}
    return plan


def sampler_plan():
    config = d.campaign()
    plan = d.worker_plan('sample', 1, {'sampler_id': 'v000', 'episodes': [0]})
    plan['experiment'] = 'rl_multiround_validation_sampler'
    plan['output'] = config['remote_root'] + '/validation/sampler/results'
    plan['phase_limits_seconds']['sample1'] = LIMITS['sample']
    plan['runtime_storage']['max_persistent_bytes'] = 2 * 1024**3
    plan['runtime_storage']['shared_budget_roots'] = [plan['output']]
    return plan


def spec(tag):
    config = d.campaign()
    if tag.startswith('val-learn-h100-cache'):
        plan = learner_plan(tag, 'h100', config['fast_learner'])
        assert plan['distributed']['fast'].get('prefix_cache') and plan['distributed']['fast']['full_kl_every'] == 1
        plan['reference_checkpoints'] = [{'after_accepted': 8, 'label': 'h100_uncached_r4_step8',
                                          'directory': H100_UNCACHED_STEP8}]
        return 'learn', 'h100', plan
    if tag.startswith('val-learn-5090-full'):
        return 'learn', '5090', learner_plan(tag, '5090', {**config['fast_learner'], 'full_kl_every': 1})
    if tag.startswith('val-learn-h100'):
        return 'learn', 'h100', learner_plan(tag, 'h100', config['fast_learner'])
    return 'sample', '5090', sampler_plan()


def build(tag, signed):
    role, pool, plan = spec(tag)
    bundle = d.bundle_for(role, plan)
    proof = d.storage_proof(plan)
    name, command = d.startup_command(role, tag, 1, bundle, proof, signed, LIMITS[role])
    return role, pool, plan, bundle, name, command


def reserved_free(gpus):
    """User default (2026-10-24): use idle Reserved standard quota first. Account-wide Reserved GPUs in every
    active state (incl. queued/stopping) must leave room for `gpus` within the 4-GPU personal quota."""
    import controller as mc
    jobs = mc.account_jobs()
    used = sum(j['gpus'] for j in jobs.values() if j['quota'] == 'RESERVED')
    return not getattr(mc.account_jobs, 'approximate', True) and used + gpus <= 4


def quota_for(tag, gpus):
    # '-std-' series: Reserved when the personal quota is free at submission time, otherwise Spot.
    return 'reserved' if '-std-' in tag and reserved_free(gpus) else 'spot'


def submit():
    verification = json.loads((d.HERE / 'verification.json').read_text())
    assert verification['passed'] and verification['plan_sha256'] == d.sha_file(d.HERE / 'plan.json')
    signer, client = d.status_client(), d.aoss_client()
    try:
        for tag in JOBS:
            folder = HERE / 'jobs' / tag
            receipt = folder / 'submission.json'
            if receipt.exists():
                print('exists', tag)
                continue
            prefix = d.campaign()['aoss_prefix'] + 'validation/' + tag + '/'
            signed = lambda method, item: signer.generate_presigned_url(
                method, Params={'Bucket': '260010165', 'Key': prefix + item}, ExpiresIn=5 * 86400)
            role, pool, plan, bundle, name, command = build(tag, signed)
            gpus = 4 if role == 'learn' else 1
            assert gpus * LIMITS[role] / 3600 <= 128
            quota = quota_for(tag, gpus)
            client.put_object(Bucket='260010165', Key=prefix + 'bundle.tar.gz', Body=bundle)
            folder.mkdir(parents=True, exist_ok=True)
            d.save(folder / 'worker_plan.json', plan)
            row = {'id': tag, 'job': name, 'role': role, 'pool': pool, 'gpu_count': gpus, 'quota': quota,
                   'aoss_prefix': prefix, 'output': plan['output'], 'max_seconds': LIMITS[role], 'submitted_at': time.time(),
                   'command_chars': len(command), 'startup_sha256': hashlib.sha256(command.encode()).hexdigest()}
            with receipt.open('x') as handle:
                json.dump(row, handle, indent=2)
            template = d.POOLS[pool]['template']
            assert d.describe(template)['resource_pool']['name'] == d.POOLS[pool]['pool']
            result = subprocess.run([d.CLI, 'acp', 'jobs', 'copy', '--workspace-name=share-space',
                                     '--copy-job-name=' + template, '--name=' + name, '--job-name=' + name,
                                     '--worker-spec=' + d.POOLS[pool]['spec'][gpus], '--worker-nodes=1',
                                     '--quota-type=' + quota, '--retry-times=0', '--enable-fault-tolerance=false',
                                     '--enable-anomaly-detection=false', '--command=' + command],
                                    env=d.ENV, capture_output=True, text=True, timeout=90)
            row['submit_returncode'] = result.returncode
            if result.returncode:
                import re
                row['submit_error'] = re.sub(r'https?://\S+', '<url>', result.stderr or result.stdout)[-600:]
                d.save(receipt, row)
            job = d.describe(name)
            assert job['scheduling']['quota_type'] == quota.upper() and job['resource_pool']['name'] == d.POOLS[pool]['pool']
            assert hashlib.sha256(job['roles'][0]['startup_script'].encode()).hexdigest() == row['startup_sha256']
            row.update(uid=job['uid'], state=job['state'])
            d.save(receipt, row)
            print(json.dumps({k: row[k] for k in ['job', 'uid', 'state', 'pool', 'gpu_count', 'command_chars']}))
    finally:
        client.close()
        signer.close()


def status():
    client = d.aoss_client()
    rows = []
    for found in sorted(p.name for p in (HERE / 'jobs').glob('val-*') if p.is_dir()):
        if found not in JOBS:
            JOBS.append(found)
    try:
        for tag in JOBS:
            path = HERE / 'jobs' / tag / 'submission.json'
            if not path.exists():
                continue
            row = json.loads(path.read_text())
            job = d.describe(row['job'])
            assert job['uid'] == row['uid']
            row.update(state=job['state'], start_time=job.get('start_time'), complete_time=job.get('complete_time'))
            try:
                row['progress'] = json.loads(client.get_object(Bucket='260010165', Key=row['aoss_prefix'] + 'status.json')['Body'].read())
            except client.exceptions.ClientError:
                row['progress'] = None
            if row['state'] in d.TERMINAL:
                try:
                    body = client.get_object(Bucket='260010165', Key=row['aoss_prefix'] + 'result.tar.gz')['Body'].read()
                    with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as archive:
                        for item in archive:
                            name = Path(item.name)
                            assert item.isfile() and not name.is_absolute() and '..' not in name.parts and name.suffix == '.json'
                            target = HERE / 'jobs' / tag / 'retrieved' / name
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(archive.extractfile(item).read())
                    row['retrieved'] = True
                except client.exceptions.ClientError:
                    row['retrieved'] = False
            d.save(HERE / 'jobs' / tag / 'latest.json', row)
            rows.append(row)
            progress = row.get('progress') or {}
            print(f"{tag:22s} {row['state']:10s} stage={progress.get('stage')} cand={progress.get('candidate')} "
                  f"acc={progress.get('accepted_optimizer_steps')} err={progress.get('error_type')}")
    finally:
        client.close()
    return rows


def analyze():
    reference = json.loads(REFERENCE_LOCAL.read_text())['training_result']
    report = {}
    for tag in [t for t in JOBS if t.startswith('val-learn')]:
        root = HERE / 'jobs' / tag / 'retrieved'
        if not (root / 'training_updates/train1/result.json').exists():
            report[tag] = {'available': False}
            continue
        result = json.loads((root / 'training_updates/train1/result.json').read_text())
        steps = [s for s in result['update_steps'] if 'candidate' in s]
        pairs = [(a['candidate_kl'], b['candidate_kl'], b.get('kl_check')) for a, b in zip(reference['update_steps'], steps)]
        full_pairs = [(a, b) for a, b, kind in pairs if kind == 'full']
        report[tag] = {
            'accepted': result['optimizer_steps'], 'reference_accepted': reference['optimizer_steps'],
            'early_stopped': result['early_stopped'], 'final_rollback_to_verified': result.get('final_rollback_to_verified'),
            'final_kl': result['final_batch_kl']['mean'], 'reference_final_kl': reference['final_batch_kl']['mean'],
            'max_relative_kl_difference_on_full_checks': max((abs(b - a) / a for a, b in full_pairs if a), default=None),
            'timing_seconds': result.get('timing_seconds'),
            'phase_seconds': json.loads((root / 'timing.json').read_text())['phase_seconds'] if (root / 'timing.json').exists() else None,
            'replay_check': json.loads((root / 'replay_check.json').read_text()) if (root / 'replay_check.json').exists() else None,
            'parity_gate_passed': json.loads((root / 'parity_gate.json').read_text())['passed'] if (root / 'parity_gate.json').exists() else None,
            'reference_comparison': json.loads((root / 'reference_comparison.json').read_text()) if (root / 'reference_comparison.json').exists() else None,
            'reference_comparison_steps': json.loads((root / 'reference_comparison_steps.json').read_text()) if (root / 'reference_comparison_steps.json').exists() else None,
            'prefix_cache': json.loads((root / 'prefix_cache.json').read_text()) if (root / 'prefix_cache.json').exists() else None,
            'parity_gate': json.loads((root / 'parity_gate.json').read_text()) if (root / 'parity_gate.json').exists() else None,
            'hardware': json.loads((root / 'hardware.json').read_text()) if (root / 'hardware.json').exists() else None}
    sampler = HERE / 'jobs/val-sample-5090/retrieved/training_receipts/round_01/episode_001.json'
    report['val-sample-5090'] = json.loads(sampler.read_text()) if sampler.exists() else {'available': False}
    d.save(HERE / 'report.json', report)
    print(json.dumps(report, indent=1, ensure_ascii=False)[:6000])


# D moved to the idle Reserved quota at 09:44 (user default 2026-10-24): '-std-' series; the Spot attempt
# val-learn-h100-cache-r1 never left the queue (worker Pending, no node) and was stopped before it ran.
SERIES = {'val-learn-5090-full': 6, 'val-learn-h100': 6, 'val-learn-h100-cache-std': 2}  # logical -> max attempts
# Attempts up to this number are superseded by a code/config fix (5090 r1: NCCL failure -> gloo).
SUPERSEDED = {'val-learn-5090-full': 1}


def reclaimed(row):
    """Spot reclaim / startup loss (resubmit) vs a real failure (stop and report)."""
    if row['state'] == 'SUSPENDED':
        return True
    progress = row.get('progress') or {}
    if not progress or progress.get('state') == 'bootstrap_failed':
        return True
    if progress.get('error_type') in ('KeyboardInterrupt', 'ExecutionStopped') and row.get('start_time') and row.get('complete_time'):
        from datetime import datetime
        parse = lambda v: datetime.fromisoformat(v.replace('Z', '+00:00'))
        return (parse(row['complete_time']) - parse(row['start_time'])).total_seconds() < row['max_seconds'] - 900
    return False


def watch():
    """Keep each logical learner validation alive across Spot reclaims, then analyze. Exits when both end."""
    import fcntl
    while True:
        rows = {r['id']: r for r in status()}
        pending = False
        for logical, cap in SERIES.items():
            numbered = sorted([t for t in rows if t.startswith(logical + '-r')], key=lambda t: int(t.rsplit('-r', 1)[1]))
            attempts = [t for t in numbered if int(t.rsplit('-r', 1)[1]) > SUPERSEDED.get(logical, 0)]
            last = rows[attempts[-1]] if attempts else None
            if last and last['state'] == 'SUCCEEDED':
                continue
            if last and last['state'] not in d.TERMINAL:
                pending = True
                continue
            if last and not reclaimed(last):
                print('STOP', logical, 'real failure in', last['id'])
                continue
            if len(numbered) >= cap:
                print('STOP', logical, 'attempt cap reached')
                continue
            tag = f'{logical}-r{len(numbered) + 1}'
            if tag not in JOBS:
                JOBS.append(tag)
            with open(d.ANALYSIS / 'rl_gpu_submit.lock', 'a') as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                submit()
            pending = True
        if not pending:
            analyze()
            return
        time.sleep(120)


if __name__ == '__main__':
    {'submit': submit, 'status': status, 'analyze': analyze, 'watch': watch}[sys.argv[1]]()
