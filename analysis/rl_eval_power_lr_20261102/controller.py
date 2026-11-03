"""Scheduler for rl_eval_power_lr_20261102.

Workstream 2: three plain-PPO learners on the fixed 32 modem rollouts (actor LR 1e-7 / 1e-6 / 5e-6).
Workstream 1: high-power paired evaluation of five arms on 301-320 x seeds 0-9.
Only this campaign's jobs are ever submitted or stopped.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import io
import json
from math import comb
import os
from pathlib import Path
import random
import subprocess
import tarfile
import time

from deployment import (ANALYSIS, CLI, ENV, HERE, TERMINAL, _submit_unlocked, aoss_client, arm_cases, campaign,
                        describe, save)

ACTIVE_STATES = ['WAITING', 'INIT', 'QUEUEING', 'CREATING', 'STARTING', 'RESTARTING', 'RECOVERING', 'RUNNING',
                 'SUSPENDING', 'DELETING']
# DriverProcessExit: the simulator process died (e.g. 'No module named omni.kit.usd' during setup on a bad node,
# observed 2026-11-03 01:01-01:45). Environment failures, retried within the per-case attempt cap.
RETRYABLE = {'KeyboardInterrupt', 'ExecutionStopped', 'SimulatorInitializationError', 'TimeoutError', 'EOFError',
             'BrokenPipeError', 'ConnectionError', 'ConnectionResetError', 'DriverProcessExit'}
CANARY = [301, 0]
MAX_PENDING_JOBS = 8
FAILURE_BREAKER = 4
# A receipt whose `jobs copy` failed and whose job the API positively reports as nonexistent never ran.
# Observed 2026-11-03 04:18: copy exit 70 (Reserved), then `describe` 404 on every pass for 3.5 h, which
# aborted each scheduling pass. Such an attempt is terminal and retryable (no progress -> failure_kind 'retry').
NOT_CREATED = 'NOT_CREATED'
TERMINAL = TERMINAL | {NOT_CREATED}


def job_missing(name):
    """True only when the API says the job does not exist (any other describe error stays an error)."""
    reply = subprocess.run([CLI, 'acp', 'jobs', 'describe', name, '--workspace-name=share-space', '-o', 'json'],
                           env=ENV, capture_output=True, text=True, timeout=40)
    text = reply.stdout + reply.stderr
    return reply.returncode != 0 and ('trainingJobNotFound' in text or 'does not exist' in text)


def key(case):
    return f'i{case[0]}_s{case[1]}'


# ---------------------------------------------------------------- job state

def inspect(client):
    paths = sorted((HERE / 'jobs').glob('*/submission.json'))

    def read(path):
        row = json.loads(path.read_text())
        cached = path.with_name('latest.json')
        previous = json.loads(cached.read_text()) if cached.exists() else None
        if previous and previous['state'] in TERMINAL:
            return previous
        if 'uid' not in row:  # submission never confirmed: a failed copy, or a timed-out one
            age = time.time() - row['submitted_at']
            if ((age > 120 and row.get('submit_returncode', 0) != 0) or age > 900) and job_missing(row['job']):
                row.update(state=NOT_CREATED, checked_at=time.time())
                save(cached, row)
                return row
        job = describe(row['job'])
        if 'uid' not in row:
            row['uid'] = job['uid']
            save(path, row)
        assert job['uid'] == row['uid']
        row.update(state=job['state'], checked_at=time.time())
        for name in ['start_time', 'complete_time', 'update_time']:
            row[name] = job.get(name)
        try:
            response = client.get_object(Bucket='260010165', Key=row['aoss_prefix'] + 'status.json')
            assert response['ContentLength'] < 1024**2
            row['progress'] = json.loads(response['Body'].read())
        except client.exceptions.ClientError as error:
            if error.response['Error']['Code'] not in ['NoSuchKey', '404']:
                raise
        save(cached, row)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        return list(pool.map(read, paths))


def collect(client, row):
    folder = HERE / 'jobs' / row['id']
    if (folder / 'collected.json').exists():
        return
    try:
        response = client.get_object(Bucket='260010165', Key=row['aoss_prefix'] + 'result.tar.gz')
    except client.exceptions.ClientError as error:
        if error.response['Error']['Code'] in ['NoSuchKey', '404'] and row['state'] != 'SUCCEEDED':
            save(folder / 'collected.json', {'at': time.time(), 'state': row['state'], 'bytes': 0, 'no_result': True})
            return
        raise
    assert response['ContentLength'] < 256 * 1024**2
    body = response['Body'].read()
    root = folder / 'retrieved'
    with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as archive:
        for item in archive:
            name = Path(item.name)
            assert item.isfile() and not name.is_absolute() and '..' not in name.parts and name.suffix == '.json'
            data = archive.extractfile(item).read()
            json.loads(data)
            destination = root / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert destination.read_bytes() == data
            else:
                destination.write_bytes(data)
    if row['state'] == 'SUCCEEDED':
        config = campaign()
        if row['role'] == 'train':
            ready = json.loads((root / 'checkpoints/round_01/ready.json').read_text())
            update = json.loads((root / 'training_updates/train1/result.json').read_text())
            assert ready['round_trip'] and ready['actor_lr'] == config['training_arms'][row['arm']]['actor_lr']
            assert ready['optimizer_steps'] == update['optimizer_steps'] and update['episodes'] == 32
            assert update['chunks'] == config['source_chunks']
            assert json.loads((root / 'parity_gate.json').read_text())['passed']
            save(HERE / 'checkpoints' / f"{row['arm']}.json", {**ready, 'training_result': update})
        else:
            source = root / 'final' / key(row['case'])
            destination = HERE / 'evidence' / row['arm'] / key(row['case'])
            for name in ['result.json', 'initial.json', 'spec.json', 'actions.json', 'goal_summary.json']:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                data = (source / name).read_bytes()
                if target.exists():
                    assert target.read_bytes() == data
                else:
                    target.write_bytes(data)
            identity = json.loads((root / 'policy_identity.json').read_text())
            expected = expected_policy_hash(row['arm'])
            assert expected is None or identity['trainable_sha256'] == expected, 'Evaluated policy is not the declared arm'
            save(destination / 'policy_identity.json', identity)
            result = json.loads((destination / 'result.json').read_text())
            assert [result['instance'], result['seed']] == list(row['case']) and result['steps'] > 0
            assert json.loads((root / 'checks.json').read_text())['frozen_unchanged']
    save(folder / 'collected.json', {'at': time.time(), 'state': row['state'], 'bytes': len(body)})


def expected_policy_hash(arm):
    config = campaign()
    spec = config['eval_arms'][arm]
    if spec['checkpoint'] is None:
        return None  # checked on the worker against source_identity.cold_trainable_hash
    if spec['checkpoint'].startswith('training:'):
        ready = json.loads((HERE / 'checkpoints' / f"{spec['checkpoint'].split(':', 1)[1]}.json").read_text())
        return ready['trainable_sha256']
    return spec['trainable_sha256']


def _list_state(state):
    """One paginated listing of a state; returns (jobs, consistent)."""
    found, consistent = {}, True
    for page in range(1, 201):
        reply = subprocess.run([CLI, 'acp', 'jobs', 'list', '--workspace-name=share-space', '--user-name=260010165',
                                '--state=' + state, '--page-size=10', '--page-token=' + str(page), '-o', 'json'],
                               env=ENV, capture_output=True, text=True, timeout=40, check=True)
        batch = [] if reply.stdout.strip() == 'No jobs found' else json.loads(reply.stdout)
        for job in batch:
            assert job['state'] == state and job['ownership']['user_name'] == '260010165'
            if job['uid'] in found:
                consistent = False  # pages shifted while jobs changed state
            gpus = sum(int(spec.get('limits', {}).get('nvidia.com/gpu', 0)) * max(1, int(role.get('total_replicas', 1)), int(spec.get('replicas', 0)))
                       for role in job.get('roles', []) for spec in role.get('resource_spec', []))
            found[job['uid']] = {'state': state, 'gpus': max(gpus, found.get(job['uid'], {}).get('gpus', 0)),
                                 'quota': (job.get('scheduling') or {}).get('quota_type'),
                                 'name': job.get('display_name') or job.get('name')}
        if len(batch) < 10:
            return found, consistent
    raise RuntimeError('Incomplete pagination; submission closed')


def account_jobs():
    """Account-wide active jobs. A shifted page is re-listed up to 3 times; if it stays inconsistent
    the de-duplicated listing is used and marked approximate (callers then avoid Reserved)."""
    jobs, approximate = {}, False
    for state in ACTIVE_STATES:
        for _ in range(3):
            found, consistent = _list_state(state)
            if consistent:
                break
        approximate |= not consistent
        for uid, job in found.items():
            jobs[uid] = {**job, 'gpus': max(job['gpus'], jobs.get(uid, {}).get('gpus', 0))}
    account_jobs.approximate = approximate
    return jobs


def combo_pending():
    """True while the running water combination still has never-submitted evaluation cases."""
    combo = ANALYSIS / 'rl_combined_20261102'
    if (combo / 'outcome.json').exists():
        return False
    cases = {tuple(c) for c in json.loads((combo / 'plan.json').read_text())['eval_cases']}
    submitted = set()
    for path in (combo / 'jobs').glob('eval-*/submission.json'):
        case = json.loads(path.read_text()).get('case')
        if case:
            submitted.add(tuple(case))
    return bool(cases - submitted)


class Admission:
    """One account snapshot per scheduling pass, taken under the shared submit lock,
    then updated locally for each job this pass submits."""

    def __init__(self, rows):
        config = campaign()
        mine = [r for r in rows if r['state'] not in TERMINAL]
        jobs = account_jobs()
        self.cap = config['max_campaign_live_gpus']
        self.combo_cap = config['combo_guard_account_total_gpus'] if combo_pending() else None
        self.reserved_cap = config['reserved_quota_gpus']
        self.live = sum(r['gpu_count'] for r in mine)
        self.pending = sum(1 for r in mine if r['state'] != 'RUNNING')
        self.total = sum(j['gpus'] for j in jobs.values())
        self.reserved = sum(j['gpus'] for j in jobs.values() if j['quota'] == 'RESERVED')
        if getattr(account_jobs, 'approximate', False):
            self.reserved = self.reserved_cap  # uncertain listing: never risk over-asking the Reserved quota
        # A Reserved copy rejected in the last 30 min (e.g. quota not yet released): use Spot meanwhile.
        if any(r['state'] == NOT_CREATED and r['quota'] == 'reserved' and time.time() - r['submitted_at'] < 1800
               for r in rows):
            self.reserved = self.reserved_cap
        save(HERE / 'active_gpu_admission.json', {'at': time.time(), 'account_total_gpus': self.total,
                                                'account_reserved_gpus': self.reserved, 'campaign_live_gpus': self.live,
                                                'campaign_pending_jobs': self.pending, 'combo_guard': self.combo_cap})

    reserve = 0  # GPUs held back for training jobs that could not be admitted yet

    def __call__(self, role):
        required = 4 if role == 'train' else 1
        held = 0 if role == 'train' else self.reserve
        if self.live + required + held > self.cap:
            raise RuntimeError('Campaign GPU cap reached')
        if self.combo_cap is not None and self.total + required + held > self.combo_cap:
            raise RuntimeError('Leaving room for the water combination to submit its last cases')
        if self.pending >= MAX_PENDING_JOBS:
            raise RuntimeError('Too many queued jobs; wait for placement')
        quota = 'reserved' if role == 'eval' and self.reserved + 1 <= self.reserved_cap else 'spot'
        self.live += required
        self.total += required
        self.pending += 1
        self.reserved += required if quota == 'reserved' else 0
        return quota


def gpu_seconds_bound(rows):
    total = 0.
    for row in rows:
        if row['state'] in TERMINAL and row.get('start_time') and row.get('complete_time'):
            parse = lambda v: datetime.fromisoformat(v.replace('Z', '+00:00'))
            seconds = max(0., (parse(row['complete_time']) - parse(row['start_time'])).total_seconds())
        elif row['state'] in TERMINAL and not row.get('start_time'):
            seconds = 0.
        else:
            seconds = row['max_seconds']
        total += seconds * row['gpu_count']
    return total


def failure_kind(row):
    """'retry' for preemption/infrastructure/startup failures, 'fatal' otherwise."""
    if row['state'] == 'SUSPENDED':
        return 'retry'
    progress = row.get('progress') or {}
    if progress.get('state') == 'bootstrap_failed':
        return 'retry'
    if not progress:
        return 'retry'  # failed before the worker published anything
    folder = HERE / 'jobs' / row['id'] / 'retrieved'
    candidates = list((folder / 'failures').glob('*.json')) + [folder / 'failure.json']
    for path in candidates:
        if path.exists() and json.loads(path.read_text()).get('type') in RETRYABLE:
            return 'retry'
    return 'retry' if progress.get('error_type') in RETRYABLE else 'fatal'


def finished_at(row):
    """Epoch seconds: the API's ISO complete_time, else when the row was last checked (NOT_CREATED rows)."""
    if row.get('complete_time'):
        return datetime.fromisoformat(row['complete_time'].replace('Z', '+00:00')).timestamp()
    return row.get('checked_at') or 0.


def workers_exited(row):
    if row['state'] == NOT_CREATED:
        return True  # no job, so no workers
    job = describe(row['job'])
    assert job['uid'] == row['uid'] and job['state'] in TERMINAL
    reply = subprocess.run([CLI, 'acp', 'jobs', 'get-workers', row['job'], '--workspace-name=share-space'],
                           env=ENV, capture_output=True, text=True, timeout=40, check=True)
    return not any(word in reply.stdout for word in ['Running', 'Pending', 'Terminating'])


# ---------------------------------------------------------------- analysis

def mcnemar(gains, losses):
    n = gains + losses
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n)


def sign_flip(diffs, iterations=2000):
    """Paired sign-flip (randomization) test on score differences; Monte-Carlo, fixed seed."""
    values = [d for d in diffs if abs(d) > 1e-12]
    if not values:
        return 1.0
    observed = abs(sum(values))
    rng = random.Random(0)
    hits = sum(abs(sum(v if rng.random() < .5 else -v for v in values)) >= observed - 1e-12 for _ in range(iterations))
    return (hits + 1) / (iterations + 1)


RESULTS = {}  # evidence files are immutable once collected


def load_result(arm, case):
    if (arm, tuple(case)) in RESULTS:
        return RESULTS[(arm, tuple(case))]
    folder = HERE / 'evidence' / arm / key(case)
    if not (folder / 'result.json').exists():
        return None
    value = {'result': json.loads((folder / 'result.json').read_text()),
             'initial': json.loads((folder / 'initial.json').read_text())}
    RESULTS[(arm, tuple(case))] = value
    return value


def same_start(a, b):
    try:
        physical = max(abs(x - y) for k in a['physical_state'] for part in (0, 1)
                       for x, y in zip(a['physical_state'][k][part], b['physical_state'][k][part]))
        proprio = max(abs(x - y) for x, y in zip(a['proprio'], b['proprio']))
        return (a['physical_state'].keys() == b['physical_state'].keys() and physical <= 1e-4 and proprio <= 1e-4
                and all(a[k] == b[k] for k in ['physics_dt', 'rendering_dt', 'observation_dt']))
    except (KeyError, TypeError, ValueError):
        return False


def paired(arm, reference='orig', score_iterations=2000):
    pairs, invalid = [], 0
    for case in arm_cases(arm):
        a, b = load_result(reference, case), load_result(arm, case)
        if a is None or b is None:
            continue
        if not same_start(a['initial'], b['initial']):
            invalid += 1
            continue
        pairs.append({'instance': case[0], 'seed': case[1], 'baseline_success': bool(a['result']['success']),
                      'success': bool(b['result']['success']), 'baseline_score': a['result']['score'],
                      'score': b['result']['score'], 'baseline_steps': a['result']['steps'], 'steps': b['result']['steps']})
    n = len(pairs)
    gains = sum(p['success'] and not p['baseline_success'] for p in pairs)
    losses = sum(p['baseline_success'] and not p['success'] for p in pairs)
    report = {'reference': reference, 'expected_pairs': len(arm_cases(arm)), 'completed_pairs': n,
              'invalid_initial_state_pairs': invalid, 'gains': gains, 'losses': losses,
              'mcnemar_p': mcnemar(gains, losses), 'pairs': pairs}
    if n:
        report.update(baseline_success_rate=sum(p['baseline_success'] for p in pairs) / n,
                      success_rate=sum(p['success'] for p in pairs) / n,
                      success_rate_delta=(sum(p['success'] for p in pairs) - sum(p['baseline_success'] for p in pairs)) / n,
                      baseline_mean_score=sum(p['baseline_score'] for p in pairs) / n,
                      mean_score=sum(p['score'] for p in pairs) / n,
                      score_sign_flip_p=sign_flip([p['score'] - p['baseline_score'] for p in pairs], score_iterations),
                      discordant_fraction=(gains + losses) / n)
    return report


def training_summary():
    summary = {}
    for path in sorted((HERE / 'checkpoints').glob('*.json')):
        ready = json.loads(path.read_text())
        result = ready['training_result']
        steps = result['update_steps']
        summary[path.stem] = {
            'actor_lr': ready['actor_lr'], 'accepted_optimizer_steps': result['optimizer_steps'],
            'early_stopped': result['early_stopped'], 'final_batch_kl': result['final_batch_kl'],
            'per_step': [{k: s.get(k) for k in ['candidate', 'accepted', 'candidate_kl', 'ppo_clip_fraction',
                                                 'post_update_outside_clip_fraction', 'post_update_mean_abs_log_ratio',
                                                 'gradient_norm_before_clip']} for s in steps]}
    return summary


# ---------------------------------------------------------------- scheduling

def task_state(rows, role, arm, case):
    """('new'|'active'|'done'|'retry'|'abandoned', last_row)."""
    relevant = [r for r in rows if r['role'] == role and r['arm'] == arm and r.get('case') == case]
    if not relevant:
        return 'new', None
    last = max(relevant, key=lambda r: r['attempt'])
    if last['state'] == 'SUCCEEDED':
        return 'done', last
    if last['state'] not in TERMINAL:
        return 'active', last
    cap = campaign()['train_attempt_cap' if role == 'train' else 'eval_attempt_cap']
    if last['attempt'] >= cap or failure_kind(last) != 'retry':
        return 'abandoned', last
    return 'retry', last


def arm_available(arm, rows):
    """(available, skip_reason). A skip reason means the arm will never be evaluated."""
    spec = campaign()['eval_arms'][arm]
    if not (spec['checkpoint'] or '').startswith('training:'):
        return True, None
    training = spec['checkpoint'].split(':', 1)[1]
    ready_path = HERE / 'checkpoints' / f'{training}.json'
    if ready_path.exists():
        if json.loads(ready_path.read_text())['optimizer_steps'] == 0:
            return False, 'no update accepted under the KL guard; policy equals the original'
        return True, None
    if task_state(rows, 'train', training, None)[0] == 'abandoned':
        return False, 'training failed after the attempt cap'
    return False, None


def main():
    config = campaign()
    deadline = config['created_at'] + config['max_wall_hours'] * 3600
    (HERE / 'jobs').mkdir(exist_ok=True)
    with (HERE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(HERE / 'controller_process.json', {'pid': os.getpid(), 'started_at': time.time()})
        client = aoss_client()
        try:
            while time.time() < deadline:
                try:
                    rows = inspect(client)
                    for row in rows:
                        if row['state'] in TERMINAL:
                            collect(client, row)
                    availability = {arm: arm_available(arm, rows) for arm in config['eval_arms']}
                    full = {arm: paired(arm) for arm in config['eval_arms'] if arm != 'orig'}
                    save(HERE / 'comparisons.json', {'at': time.time(), 'arms': full})
                    save(HERE / 'training_summary.json', training_summary())
                    active = [r for r in rows if r['state'] not in TERMINAL]
                    fatal = [r['id'] for r in rows if r['state'] in TERMINAL and r['state'] != 'SUCCEEDED'
                             and r['role'] == 'eval' and failure_kind(r) == 'fatal']
                    abandoned = [[r, a, c] for r, a, c in
                                 [('train', arm, None) for arm in config['training_arms']] +
                                 [('eval', arm, case) for arm in config['eval_arms'] for case in arm_cases(arm)]
                                 if task_state(rows, r, a, c)[0] == 'abandoned']
                    save(HERE / 'status.json', {
                        'at': time.time(), 'active_jobs': len(active), 'active_gpus': sum(r['gpu_count'] for r in active),
                        'running_gpus': sum(r['gpu_count'] for r in active if r['state'] == 'RUNNING'),
                        'succeeded_evaluations': sum(1 for r in rows if r['role'] == 'eval' and r['state'] == 'SUCCEEDED'),
                        'availability': availability, 'abandoned': abandoned, 'fatal_eval_failures': fatal,
                        'gpu_hours_bound': gpu_seconds_bound(rows) / 3600,
                        'comparisons': {arm: {k: v for k, v in rep.items() if k != 'pairs'} for arm, rep in full.items()}})
                    finished = lambda role, arm, case: task_state(rows, role, arm, case)[0] in ('done', 'abandoned')
                    complete = all(finished('train', arm, None) for arm in config['training_arms'])
                    for arm in config['eval_arms']:
                        available, skip = availability[arm]
                        if skip:
                            continue
                        if not available or not all(finished('eval', arm, case) for case in arm_cases(arm)):
                            complete = False
                    if complete:
                        final = {arm: paired(arm, score_iterations=50000) for arm in config['eval_arms'] if arm != 'orig'}
                        save(HERE / 'outcome.json', {'at': time.time(), 'training': training_summary(),
                                                    'comparisons': final, 'availability': availability,
                                                    'abandoned': abandoned})
                        return
                    recent = sorted([r for r in rows if r['role'] == 'eval' and r['state'] in TERMINAL],
                                    key=finished_at)[-12:]
                    # Spot preemption (SUSPENDED / SIGTERM) is expected and never trips the breaker.
                    preempted = lambda r: r['state'] == 'SUSPENDED' or (r.get('progress') or {}).get('error_type') in (
                        'KeyboardInterrupt', 'ExecutionStopped')
                    recent_failures = sum(r['state'] != 'SUCCEEDED' and not preempted(r) for r in recent)
                    if len(fatal) >= FAILURE_BREAKER or (len(recent) >= 8 and recent_failures >= 8):
                        save(HERE / 'paused.json', {'at': time.time(), 'reason': 'evaluation failure breaker',
                                                    'fatal_jobs': fatal, 'recent_failures': recent_failures,
                                                    'recent_terminal': len(recent)})
                        time.sleep(60)
                        continue
                    if gpu_seconds_bound(rows) > config['max_total_gpu_hours'] * 3600:
                        save(HERE / 'budget_stop.json', {'at': time.time(), 'gpu_hours_bound': gpu_seconds_bound(rows) / 3600})
                        time.sleep(60)
                        continue
                    tasks = []
                    # Learners that gate an evaluation arm first, then diagnostic-only learners (plan order).
                    learners = sorted(config['training_arms'], key=lambda a: not config['training_arms'][a]['evaluate'])
                    for arm in learners:
                        state, last = task_state(rows, 'train', arm, None)
                        if state in ('new', 'retry'):
                            tasks.append(('train', arm, None, last))
                    arms = [arm for arm in config['eval_arms'] if availability[arm][0]]
                    canary_ok = {arm: load_result(arm, CANARY) is not None for arm in arms}
                    for arm in arms:
                        if not canary_ok[arm]:
                            state, last = task_state(rows, 'eval', arm, CANARY)
                            if state in ('new', 'retry'):
                                tasks.append(('eval', arm, CANARY, last))
                    for case in config['eval_cases']:
                        for arm in arms:
                            if canary_ok[arm] and case != CANARY and case in arm_cases(arm):
                                state, last = task_state(rows, 'eval', arm, case)
                                if state in ('new', 'retry'):
                                    tasks.append(('eval', arm, case, last))
                    launched = 0
                    mine = [r for r in rows if r['state'] not in TERMINAL]
                    local_room = (sum(1 for r in mine if r['state'] != 'RUNNING') < MAX_PENDING_JOBS
                                  and sum(r['gpu_count'] for r in mine) < config['max_campaign_live_gpus'])
                    if tasks and local_room:  # skip the account-wide snapshot when nothing could be admitted
                        # Same host lock file as the other BEHAVIOR controllers: snapshot + submissions are atomic.
                        with (ANALYSIS / 'rl_gpu_submit.lock').open('a') as submit_lock:
                            fcntl.flock(submit_lock, fcntl.LOCK_EX)
                            admission = Admission(rows)
                            for role, arm, case, last in tasks:
                                if launched >= MAX_PENDING_JOBS:
                                    break
                                if last is not None and not workers_exited(last):
                                    continue
                                try:
                                    quota = admission(role)
                                except RuntimeError as error:
                                    save(HERE / 'admission_wait.json', {'at': time.time(), 'reason': str(error),
                                                                        'next_task': [role, arm, case]})
                                    if role == 'train':
                                        admission.reserve += 4  # hold room so evaluations cannot starve training
                                        continue
                                    break
                                _submit_unlocked(role, arm, case, 1 if last is None else last['attempt'] + 1, quota)
                                launched += 1
                    time.sleep(5 if launched else 30)
                except Exception as error:
                    save(HERE / 'controller_error.json', {'at': time.time(), 'type': type(error).__name__,
                                                         'message': str(error).split('--command')[0][:300]})
                    print('controller_error', type(error).__name__, str(error)[:200], flush=True)
                    time.sleep(30)
            for row in inspect(client):
                if row['state'] not in TERMINAL:
                    job = describe(row['job'])
                    assert job['uid'] == row['uid']
                    subprocess.run([CLI, 'acp', 'jobs', 'stop', row['job'], '--workspace-name=share-space'],
                                   env=ENV, capture_output=True, timeout=45, check=True)
            save(HERE / 'deadline_reached.json', {'at': time.time(), 'only_this_campaign_stopped': True})
        finally:
            client.close()


if __name__ == '__main__':
    main()
