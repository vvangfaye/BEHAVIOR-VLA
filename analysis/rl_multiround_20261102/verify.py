"""Offline verification for rl_multiround_20261102 (no upload, no job creation). Writes verification.json.

Builds every job type for round 1 and round 2 using synthetic receipts / policy records held in memory,
checks plan contracts and command sizes, and exercises the controller's scheduling decisions.
"""
import io
import json
import tarfile
import time

import deployment as d
import controller as c

config = d.campaign()
checks = {}
signer = d.status_client()


def signed(method, item):
    return signer.generate_presigned_url(method, Params={'Bucket': '260010165',
        'Key': config['aoss_prefix'] + 'verification-only/' + item}, ExpiresIn=5 * 86400)


def names(bundle):
    with tarfile.open(fileobj=io.BytesIO(bundle), mode='r:gz') as archive:
        return {item.name: archive.extractfile(item).read() for item in archive}


cold = d.base_plan()['source_identity']['cold_trainable_hash']
fake_policy = {1: {'directory': d.learner_root(1) + '/checkpoints/round_01', 'trainable_sha256': 'f' * 64,
                   'cumulative_optimizer_steps': 17}}
d.policy_record = lambda k: None if k == 0 else fake_policy.get(k)


def fake_receipts(k, policy_hash, count):
    instances, seeds = d.episode_layout()
    return [{'round': k, 'episode': i + 1, 'instance': instances[i], 'seed': seeds[k - 1][i],
             'path': d.sampling_root(k) + f'/training/round_{k:02d}/episode_{i+1:02d}/rollout.pt', 'sha256': '0' * 64,
             'policy_hash': policy_hash, 'contract': d.base_plan()['source_reward_contract_sha256'], 'chunks': 180,
             'status': {'success': i % 3 == 0}} for i in range(count)]


proof = None
for k, sampling_hash in [(1, cold), (2, fake_policy[1]['trainable_sha256'])]:
    sample = d.worker_plan('sample', k, {'sampler_id': 's000', 'episodes': [0, 1, 2, 3]})
    assert sample['round_index'] == k and sample['assigned_episodes'] == [0, 1, 2, 3]
    assert (sample['sampling_policy'] is None) == (k == 1)
    assert sample['output'] == d.sampling_root(k) and len(sample['train_seeds']) == config['rounds']
    assert len(sample['train_instances']) == config['episodes_per_round'] == len(sample['train_seeds'][k - 1])
    learn = d.worker_plan('learn', k, {'receipts': fake_receipts(k, sampling_hash, config['episodes_per_round'])})
    dist = learn['distributed']
    assert dist['enforce_on_policy'] and dist['learner_only'] and dist['backend'] == config['collective_backend'][config['learner_pool']]
    assert dist['gpu_models'] == d.POOLS[config['learner_pool']]['gpu_models'] and dist['fast'] == config['fast_learner']
    assert dist['replay_tolerance'] == config['replay_tolerance'] and learn['expected_episodes'] == config['episodes_per_round']
    assert (learn['initial_policy'] is None) == (k == 1)
    assert learn['optimizer']['actor_lr'] == config['actor_lr'] and learn['optimizer']['updates_per_round'] == [config['optimizer_steps']]
    assert learn['previous_cumulative_optimizer_steps'] == (0 if k == 1 else 17)
    # The learner re-derives the digest from the AFS receipt files exactly like this (distributed_driver.py).
    manifest = learn['source_rollout_manifest']
    on_disk = fake_receipts(k, sampling_hash, config['episodes_per_round'])
    listed = [{key: r[key] for key in d.RECEIPT_KEYS} for r in on_disk]
    import hashlib
    assert hashlib.sha256(json.dumps(listed, sort_keys=True).encode()).hexdigest() == manifest['sha256']
    assert manifest['receipt_directory'] == d.sampling_root(k) + f'/training_receipts/round_{k:02d}'
    assert manifest['episodes'] == list(range(1, config['episodes_per_round'] + 1)) and learn['source_rollout_receipts'] == []
    try:
        d.worker_plan('learn', k, {'receipts': fake_receipts(k, 'e' * 64, config['episodes_per_round'])})
        raise SystemExit('off-policy receipts were accepted')
    except AssertionError:
        pass
    # While round k runs, the published policy k-1 (here: policy 1) is evaluated.
    evaluation = d.worker_plan('eval', 1, {'case': config['eval_cases'][-1]})
    assert evaluation['comparison_checkpoint'] == fake_policy[1]['directory'] and not evaluation['evaluate_original_policy']
    try:
        d.worker_plan('eval', 2, {'case': config['eval_cases'][-1]})
        raise SystemExit('evaluation of an unpublished policy was accepted')
    except AssertionError:
        pass
    for role, plan in [('sample', sample), ('learn', learn), ('eval', evaluation)]:
        bundle = d.bundle_for(role, plan)
        files = names(bundle)
        source = (d.HERE / 'runtime' / d.DRIVERS[role]).read_bytes()
        expected = d.minify_source(source.decode()).encode() if role == 'learn' else source
        assert files['run/driver.py'] == expected
        proof = proof or d.storage_proof(plan)
        limit = config[{'learn': 'learner_timeout_seconds', 'sample': 'sample_timeout_seconds', 'eval': 'eval_timeout_seconds'}[role]]
        payload = {'sampler_id': 's123', 'episodes': [0]} if role == 'sample' else ({'case': [320, 4]} if role == 'eval' else {})
        name, command = d.startup_command(role, d.identity(role, k, payload, 3), 3, bundle, proof, signed, limit)
        gpus = 4 if role == 'learn' else 1
        assert gpus * limit / 3600 <= 128
        checks[f'{role}_r{k}'] = {'bundle_bytes': len(bundle), 'command_chars': len(command), 'job_name': name,
                                  'gpu_hours_cap': gpus * limit / 3600}

# Controller scheduling decisions on synthetic state.
rows = []
d.policy_record = lambda k: None
c.receipts = lambda k: []
tasks, current = c.plan_tasks(rows)
samplers = [t for t in tasks if t[0] == 'sample']
assert current == 1 and len(samplers) == config['episodes_per_round'] // config['episodes_per_sampler']
assert sorted(i for t in samplers for i in t[2]['episodes']) == list(range(config['episodes_per_round']))
c.receipts = lambda k: fake_receipts(k, cold, config['episodes_per_round']) if k == 1 else []
d.policy_record = lambda k: None
tasks, current = c.plan_tasks(rows)
assert [t[0] for t in tasks] == ['learn'] and current == 1
# Two episodes abandoned after the attempt cap: learn on the rest (>= min_episodes_per_round).
partial = fake_receipts(1, cold, config['episodes_per_round'])[:-2]
c.receipts = lambda k: partial if k == 1 else []
dead = [{'role': 'sample', 'round': 1, 'state': 'FAILED', 'payload': {'episodes': [62, 63]}, 'attempt': 1,
         'workers_exited': True, 'max_seconds': 14400, 'progress': {'error_type': 'RuntimeError'}}] * config['attempt_cap']
tasks, _ = c.plan_tasks(dead)
assert [t[0] for t in tasks] == ['learn'] and len(tasks[0][2]['receipts']) == config['episodes_per_round'] - 2
# After policy 1 is published: round-2 samplers plus round-1 evaluations (evaluations after samplers).
d.policy_record = lambda k: None if k == 0 else fake_policy.get(k)
c.receipts = lambda k: fake_receipts(1, cold, config['episodes_per_round']) if k == 1 else []
c.policy_result = lambda k, case: None
tasks, current = c.plan_tasks([])
roles = [t[0] for t in tasks]
round2 = {'samplers': roles.count('sample'), 'evaluations': roles.count('eval')}
expected_evals = len(d.eval_cases_for(1)) if 1 in config['eval_rounds'] else 0
assert current == 2 and roles.count('eval') == expected_evals and (expected_evals == 0 or roles.index('eval') > max(i for i, r in enumerate(roles) if r == 'sample'))
# Spot reclaim never consumes the failure cap: 4 preemptions of episode 0 -> still re-assigned.
d.policy_record = lambda k: None
c.receipts = lambda k: []
reclaimed = [{'role': 'sample', 'round': 1, 'state': 'SUSPENDED', 'payload': {'episodes': [0]}, 'attempt': 1,
              'workers_exited': True, 'max_seconds': 14400}] * 4
tasks, _ = c.plan_tasks(reclaimed)
assert 0 in [i for t in tasks if t[0] == 'sample' for i in t[2]['episodes']]
# ... but the hard cap still ends it after preemption_attempt_cap attempts.
tasks, _ = c.plan_tasks(reclaimed * 3)
assert 0 not in [i for t in tasks if t[0] == 'sample' for i in t[2]['episodes']]
# A reclaimed learner (5 times, no real failure) may retry; the next attempt number continues.
c.receipts = lambda k: fake_receipts(k, cold, config['episodes_per_round']) if k == 1 else []
learner_rows = [{'role': 'learn', 'round': 1, 'state': 'FAILED', 'payload': {}, 'attempt': a, 'workers_exited': True,
                 'max_seconds': 21600, 'start_time': '2026-11-03T00:00:00Z', 'complete_time': '2026-11-03T00:30:00Z',
                 'progress': {'error_type': 'KeyboardInterrupt'}} for a in range(1, 6)]
tasks, _ = c.plan_tasks(learner_rows)
assert [(t[0], t[3]) for t in tasks] == [('learn', 6)]
# A SIGTERM at the job's own time limit is a timeout (real failure), not a reclaim.
timeout_row = {**learner_rows[0], 'start_time': '2026-11-03T00:00:00Z', 'complete_time': '2026-11-03T05:59:00Z'}
assert not c.preempted(timeout_row) and c.preempted(learner_rows[0])
checks['preemption'] = {'reclaim_does_not_consume_failure_cap': True, 'hard_cap_enforced': True, 'timeout_is_failure': True}
# A failed `jobs copy` (no uid; the API reports the job missing) becomes NOT_CREATED and is retried, instead of
# aborting every pass (the 2026-11-03 04:18 stall of rl_eval_power_lr_20261102). Runs in a temporary job folder.
import tempfile
from pathlib import Path
with tempfile.TemporaryDirectory() as tmp:
    home, missing = c.HERE, c.job_missing
    try:
        c.HERE, c.job_missing = Path(tmp), (lambda name: True)
        folder = Path(tmp) / 'jobs' / 'learn-r01-a1'
        folder.mkdir(parents=True)
        d.save(folder / 'submission.json', {
            'id': 'learn-r01-a1', 'job': 'b1k-mr-1102-learn-r01-a1', 'role': 'learn', 'round': 1, 'payload': {},
            'attempt': 1, 'gpu_count': 4, 'quota': 'reserved', 'max_seconds': 21600, 'state': 'submitting',
            'submitted_at': time.time() - 600, 'submit_returncode': 70})
        [never] = c.inspect(None)
        assert never['state'] == c.NOT_CREATED and c.workers_exited(never) and c.gpu_seconds_bound([never]) == 0
        allowed, attempt, _ = c.retry_state([never], 'learn', 1)
        assert allowed and attempt == 2
    finally:
        c.HERE, c.job_missing = home, missing
checks['never_created_submission'] = {'marked_not_created': True, 'retried_as_attempt_2': True}
checks['scheduler'] = {'round1_samplers': len(samplers), 'round2': round2, 'evaluations_after_samplers': True}
# Final round (user 2026-11-03): policy R is evaluated on 200 cases, earlier policies on 100; all pair with
# existing original-model evidence.
R = config['rounds']
d.policy_record = lambda k: None if k == 0 else fake_policy.get(1)
c.receipts = lambda k: fake_receipts(k, cold, config['episodes_per_round'])
c.policy_result = lambda k, case: None
tasks, current = c.plan_tasks([])
per_round = {k: sum(1 for t in tasks if t[0] == 'eval' and t[1] == k) for k in range(1, R + 1)}
assert current is None and per_round[R] == 200 and all(per_round[k] == (len(config['eval_cases']) if k in config['eval_rounds'] else 0) for k in range(1, R))
assert set(map(tuple, config['eval_cases'])) <= set(map(tuple, config['final_eval_cases']))
assert all(c.original_result(case) is not None for case in config['final_eval_cases'])
d.worker_plan('eval', R, {'case': config['final_eval_cases'][-1]})
checks['final_round_evaluation'] = {'cases_per_policy': per_round, 'original_evidence_for_all': True}
# Quota choice (user default 2026-10-24: idle Reserved first), with a synthetic account snapshot.
import tempfile as _tempfile
from pathlib import Path as _Path
listing, here = c.account_jobs, c.HERE
try:
    c.HERE = _Path(_tempfile.mkdtemp())
    def account(reserved_gpus):
        def jobs():
            jobs.approximate = False
            return {f'u{i}': {'state': 'RUNNING', 'gpus': 1, 'quota': 'RESERVED'} for i in range(reserved_gpus)}
        return jobs
    c.account_jobs = account(0)
    a = c.Admission([], current=1)
    assert [a('sample', 1) for _ in range(5)] == ['spot'] * 5 and a('eval', 0) == 'spot'  # Reserved: learner only
    assert c.Admission([], current=1)('learn', 1) == 'reserved'
    c.account_jobs = account(1)
    try:
        c.Admission([], current=1)('learn', 2)
        raise AssertionError('learner must wait for the Reserved quota within the grace period')
    except RuntimeError:
        pass
    c.save(c.HERE / 'learner_ready.json', {'2': time.time() - c.LEARNER_RESERVED_GRACE - 1})
    assert c.Admission([], current=1)('learn', 2) == 'spot'
    c.account_jobs = account(0)
    # Final evaluation (every policy published): 4 Reserved evaluations first (outside the 32-GPU Spot cap),
    # then Spot; up to final_max_pending_jobs queued. During training rounds evaluations stay Spot with 8 queued.
    f = c.Admission([], current=None)
    assert f.pending_cap == config['final_max_pending_jobs'] == 32
    assert [f('eval', 3) for _ in range(6)] == ['reserved'] * 4 + ['spot'] * 2
    full = [{'role': 'eval', 'state': 'RUNNING', 'gpu_count': 1, 'pool': '5090', 'quota': 'spot',
             'submitted_at': time.time()} for _ in range(32)]
    g = c.Admission(full, current=None)
    assert g('eval', 3) == 'reserved'          # Reserved is not counted in the Spot cap
    t2 = c.Admission(full + [{'role': 'eval', 'state': 'RUNNING', 'gpu_count': 1, 'pool': '5090', 'quota': 'reserved',
                              'submitted_at': time.time()}] * 4, current=None)
    t2.reserved = 4
    try:
        t2('eval', 3)
        raise AssertionError('Spot cap must bind once Reserved is used')
    except RuntimeError:
        pass
    m = c.Admission([], current=2)
    assert m.pending_cap == config['max_pending_jobs'] == 8 and m('eval', 1) == 'spot'
    # The 32-GPU cap counts 5090 jobs only: with 32 samplers live, an H100 learner is still admitted.
    c.account_jobs = account(0)
    busy = [{'role': 'sample', 'state': 'RUNNING', 'gpu_count': 1, 'pool': '5090', 'quota': 'spot',
             'submitted_at': time.time()} for _ in range(32)]
    b = c.Admission(busy, current=2)
    assert b('learn', 2) == 'reserved'
    try:
        b('sample', 2)
        raise AssertionError('5090 cap must still bind for samplers')
    except RuntimeError:
        pass
finally:
    c.account_jobs, c.HERE = listing, here
checks['quota_choice'] = {'final_evaluation': {'reserved_first': 4, 'spot_cap': 32, 'max_queued': 32}, 'reserved_for_learner_during_rounds': True, 'h100_learner_outside_5090_cap': True, 'learner_waits_then_spot': True, 'learner_grace_seconds': c.LEARNER_RESERVED_GRACE}

tests = json.loads((d.HERE / 'tests/cpu_tests_result.json').read_text())
assert tests['passed'] and tests['runtime_sha256'] == config['runtime_sha256']
checks['cpu_tests'] = {k: tests[k] for k in ['passed', 'tests', 'elapsed_seconds']}
minified = json.loads((d.HERE / 'tests/minified_tests_result.json').read_text())
assert minified['passed']
checks['minified_learner_tests'] = {'passed': True}
checks['storage'] = {k: proof[k] for k in ['account_allocated_bytes', 'quota_lower_bound_bytes', 'required_remaining_bytes', 'passed']}
result = {'passed': True, 'at': time.time(), 'plan_sha256': d.sha_file(d.HERE / 'plan.json'), 'checks': checks}
d.save(d.HERE / 'verification.json', result)
print(json.dumps(result, indent=1))
