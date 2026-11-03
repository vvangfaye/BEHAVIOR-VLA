"""Round controller for rl_multiround_20261102.

Round k (1..R):  sample with policy k-1 on Spot 5090 -> learn on the learner pool -> publish policy k.
Evaluations of published policies run on Spot 5090 alongside the next round. Priority: learner >
samplers > evaluations. Nothing waits on a GPU: samplers exit after their episodes, the learner job is
submitted only when the round's receipts are complete and exits after publishing.
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
import subprocess
import tarfile
import time

import deployment as d
from deployment import CLI, ENV, HERE, TERMINAL, campaign, describe, save

ACTIVE_STATES = ['WAITING', 'INIT', 'QUEUEING', 'CREATING', 'STARTING', 'RESTARTING', 'RECOVERING', 'RUNNING',
                 'SUSPENDING', 'DELETING']
# DriverProcessExit: the simulator process died (e.g. 'No module named omni.kit.usd' during setup on a bad node,
# observed 2026-11-03 01:01-01:45). Environment failures, retried within the per-case attempt cap.
RETRYABLE = {'KeyboardInterrupt', 'ExecutionStopped', 'SimulatorInitializationError', 'TimeoutError', 'EOFError',
             'BrokenPipeError', 'ConnectionError', 'ConnectionResetError', 'DriverProcessExit'}
# A receipt whose `jobs copy` failed and whose job the API positively reports as nonexistent never ran.
# Observed in rl_eval_power_lr_20261102 on 2026-11-03 04:18: copy exit 70 (Reserved), then `describe` 404 on
# every pass for 3.5 h, aborting each pass. Such an attempt is terminal; it is retried like a startup failure.
NOT_CREATED = 'NOT_CREATED'
TERMINAL = TERMINAL | {NOT_CREATED}


def job_missing(name):
    """True only when the API says the job does not exist (any other describe error stays an error)."""
    reply = subprocess.run([CLI, 'acp', 'jobs', 'describe', name, '--workspace-name=share-space', '-o', 'json'],
                           env=ENV, capture_output=True, text=True, timeout=40)
    text = reply.stdout + reply.stderr
    return reply.returncode != 0 and ('trainingJobNotFound' in text or 'does not exist' in text)


# ---------------------------------------------------------------- shared helpers (same logic as rl_eval_power_lr_20261102)

def _list_state(state):
    found, consistent = {}, True
    for page in range(1, 201):
        reply = subprocess.run([CLI, 'acp', 'jobs', 'list', '--workspace-name=share-space', '--user-name=260010165',
                                '--state=' + state, '--page-size=10', '--page-token=' + str(page), '-o', 'json'],
                               env=ENV, capture_output=True, text=True, timeout=40, check=True)
        batch = [] if reply.stdout.strip() == 'No jobs found' else json.loads(reply.stdout)
        for job in batch:
            assert job['state'] == state and job['ownership']['user_name'] == '260010165'
            consistent &= job['uid'] not in found
            gpus = sum(int(spec.get('limits', {}).get('nvidia.com/gpu', 0)) * max(1, int(role.get('total_replicas', 1)), int(spec.get('replicas', 0)))
                       for role in job.get('roles', []) for spec in role.get('resource_spec', []))
            found[job['uid']] = {'state': state, 'gpus': max(gpus, found.get(job['uid'], {}).get('gpus', 0)),
                                 'quota': (job.get('scheduling') or {}).get('quota_type')}
        if len(batch) < 10:
            return found, consistent
    raise RuntimeError('Incomplete pagination; submission closed')


def account_jobs():
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


def workers_exited(row):
    if row['state'] == NOT_CREATED:
        return True  # no job, so no workers
    job = describe(row['job'])
    assert job['uid'] == row['uid'] and job['state'] in TERMINAL
    reply = subprocess.run([CLI, 'acp', 'jobs', 'get-workers', row['job'], '--workspace-name=share-space'],
                           env=ENV, capture_output=True, text=True, timeout=40, check=True)
    return not any(word in reply.stdout for word in ['Running', 'Pending', 'Terminating'])


def mcnemar(gains, losses):
    n = gains + losses
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n)


def same_start(a, b):
    try:
        physical = max(abs(x - y) for k in a['physical_state'] for part in (0, 1)
                       for x, y in zip(a['physical_state'][k][part], b['physical_state'][k][part]))
        proprio = max(abs(x - y) for x, y in zip(a['proprio'], b['proprio']))
        return (a['physical_state'].keys() == b['physical_state'].keys() and physical <= 1e-4 and proprio <= 1e-4
                and all(a[k] == b[k] for k in ['physics_dt', 'rendering_dt', 'observation_dt']))
    except (KeyError, TypeError, ValueError):
        return False


def key(case):
    return f'i{case[0]}_s{case[1]}'


# ---------------------------------------------------------------- job state and collection

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
        if 'uid' not in row:  # copy timed out locally but the job exists: adopt it
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


def extract(client, row):
    """Download and safely unpack the job's JSON result archive into jobs/<id>/retrieved."""
    folder = HERE / 'jobs' / row['id']
    try:
        response = client.get_object(Bucket='260010165', Key=row['aoss_prefix'] + 'result.tar.gz')
    except client.exceptions.ClientError as error:
        if error.response['Error']['Code'] in ['NoSuchKey', '404']:
            return None
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
    return root


def expected_sampling_hash(k):
    previous = d.policy_record(k - 1)
    return previous['trainable_sha256'] if previous else d.base_plan()['source_identity']['cold_trainable_hash']


def collect(client, row):
    folder = HERE / 'jobs' / row['id']
    if (folder / 'collected.json').exists():
        return
    root = extract(client, row)
    k = row['round']
    if row['role'] == 'sample' and root is not None:
        # Receipts are collected even from failed/preempted samplers: completed episodes stay valid.
        expected = expected_sampling_hash(k)
        contract = d.base_plan()['source_reward_contract_sha256']
        for path in sorted((root / 'training_receipts' / f'round_{k:02d}').glob('episode_*.json')):
            receipt = json.loads(path.read_text())
            assert receipt['round'] == k and receipt['policy_hash'] == expected and receipt['contract'] == contract
            target = HERE / 'receipts' / f'round_{k:02d}' / path.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert json.loads(target.read_text()) == receipt
            else:
                save(target, receipt)
    elif row['role'] == 'learn' and row['state'] == 'SUCCEEDED':
        ready = json.loads((root / 'checkpoints/round_01/ready.json').read_text())
        result = json.loads((root / 'training_updates/train1/result.json').read_text())
        assert ready['round_trip'] and ready['round'] == k and ready['collection_policy_hash'] == expected_sampling_hash(k)
        assert json.loads((root / 'parity_gate.json').read_text())['passed']
        extras = {name: json.loads((root / name).read_text()) for name in ['timing.json', 'replay_check.json']
                  if (root / name).exists()}
        save(HERE / 'policies' / f'round_{k:02d}.json', {**ready, 'training_result': result, **extras})
    elif row['role'] == 'eval' and row['state'] == 'SUCCEEDED':
        case = row['payload']['case']
        source = root / 'final' / key(case)
        destination = HERE / 'evidence' / f'policy_r{k:02d}' / key(case)
        for name in ['result.json', 'initial.json', 'spec.json', 'actions.json', 'goal_summary.json']:
            target = destination / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = (source / name).read_bytes()
            if target.exists():
                assert target.read_bytes() == data
            else:
                target.write_bytes(data)
        identity = json.loads((root / 'policy_identity.json').read_text())
        policy = d.policy_record(k)
        assert policy is None or identity['trainable_sha256'] == policy['trainable_sha256']
        assert json.loads((root / 'checks.json').read_text())['frozen_unchanged']
    save(folder / 'collected.json', {'at': time.time(), 'state': row['state'], 'result': root is not None})


def failure_kind(row):
    """'retry' for preemption, startup and known infrastructure failures; 'fatal' otherwise."""
    if row['role'] == 'sample':
        return 'retry'  # completed receipts are never lost; missing episodes are re-assigned (per-episode cap)
    if row['state'] == 'SUSPENDED':
        return 'retry'
    progress = row.get('progress') or {}
    if not progress or progress.get('state') == 'bootstrap_failed':
        return 'retry'
    folder = HERE / 'jobs' / row['id'] / 'retrieved'
    for path in list((folder / 'failures').glob('*.json')) + [folder / 'failure.json']:
        if path.exists() and json.loads(path.read_text()).get('type') in RETRYABLE:
            return 'retry'
    return 'retry' if progress.get('error_type') in RETRYABLE else 'fatal'


PREEMPTION_ERRORS = {'KeyboardInterrupt', 'ExecutionStopped'}


def preempted(row):
    """Spot reclaim, not a failure of the work: SUSPENDED, or SIGTERM (KeyboardInterrupt/ExecutionStopped)
    well before the job's own time limit (our SIGALRM timeout raises the same type near max_seconds)."""
    if row['state'] == 'SUSPENDED':
        return True
    if row['state'] not in TERMINAL or row['state'] == 'SUCCEEDED':
        return False
    if (row.get('progress') or {}).get('error_type') not in PREEMPTION_ERRORS:
        return False
    if row.get('start_time') and row.get('complete_time'):
        parse = lambda v: datetime.fromisoformat(v.replace('Z', '+00:00'))
        elapsed = (parse(row['complete_time']) - parse(row['start_time'])).total_seconds()
        return elapsed < row['max_seconds'] - 900
    return True


def retry_state(rows, role, k, case=None):
    """(allowed, next_attempt, last). Preemptions never consume the failure cap (attempt_cap); a separate
    hard cap (preemption_attempt_cap) bounds total attempts so a task can never retry forever."""
    config = campaign()
    relevant = [r for r in rows if r['role'] == role and r['round'] == k
                and (case is None or r['payload'].get('case') == case)]
    if not relevant:
        return True, 1, None
    last = max(relevant, key=lambda r: r['attempt'])
    if last['state'] == 'SUCCEEDED' or last['state'] not in TERMINAL:
        return False, None, last
    real = sum(1 for r in relevant if r['state'] in TERMINAL and r['state'] != 'SUCCEEDED' and not preempted(r))
    allowed = (real < config['attempt_cap'] and len(relevant) < config['preemption_attempt_cap']
               and (preempted(last) or failure_kind(last) == 'retry'))
    return allowed, last['attempt'] + 1, last


# ---------------------------------------------------------------- round bookkeeping

def receipts(k):
    folder = HERE / 'receipts' / f'round_{k:02d}'
    return sorted((json.loads(p.read_text()) for p in folder.glob('episode_*.json')), key=lambda r: r['episode'])


def episode_attempts(rows, k):
    """Episodes whose sampling should stop: >= attempt_cap real failures or >= preemption_attempt_cap total."""
    config = campaign()
    real, total = {}, {}
    for row in rows:
        if row['role'] == 'sample' and row['round'] == k and row['state'] in TERMINAL:
            for index in row['payload']['episodes']:
                total[index] = total.get(index, 0) + 1
                if not preempted(row):
                    real[index] = real.get(index, 0) + 1
    return {i for i in total if real.get(i, 0) >= config['attempt_cap'] or total[i] >= config['preemption_attempt_cap']}


def active_episodes(rows, k):
    """Episodes owned by a sampler that may still write: non-terminal jobs, and terminal jobs whose
    workers are not yet confirmed exited (a reclaimed worker can outlive the platform state)."""
    return {i for r in rows if r['role'] == 'sample' and r['round'] == k
            and (r['state'] not in TERMINAL or not r.get('workers_exited'))
            for i in r['payload']['episodes']}


def confirm_exited(rows):
    """Record (once) that terminal jobs' workers are gone, so their episodes can be re-assigned."""
    for row in rows:
        if row['state'] in TERMINAL and not row.get('workers_exited'):
            if workers_exited(row):
                row['workers_exited'] = True
                save(HERE / 'jobs' / row['id'] / 'latest.json', row)


def last_attempt(rows, role, k, case=None):
    relevant = [r for r in rows if r['role'] == role and r['round'] == k
                and (case is None or r['payload'].get('case') == case)]
    return max(relevant, key=lambda r: r['attempt']) if relevant else None


def original_result(case):
    folder = Path(campaign()['reuse_original_evidence']) / key(case)
    return ({'result': json.loads((folder / 'result.json').read_text()),
             'initial': json.loads((folder / 'initial.json').read_text())} if (folder / 'result.json').exists() else None)


def policy_result(k, case):
    folder = HERE / 'evidence' / f'policy_r{k:02d}' / key(case)
    return ({'result': json.loads((folder / 'result.json').read_text()),
             'initial': json.loads((folder / 'initial.json').read_text())} if (folder / 'result.json').exists() else None)


def learning_curve():
    curve = {}
    for k in campaign()['eval_rounds']:
        pairs = []
        for case in d.eval_cases_for(k):
            a, b = original_result(case), policy_result(k, case)
            if a and b and same_start(a['initial'], b['initial']):
                pairs.append((bool(a['result']['success']), bool(b['result']['success'])))
        gains = sum(b and not a for a, b in pairs)
        losses = sum(a and not b for a, b in pairs)
        curve[f'policy_r{k:02d}'] = {'pairs': len(pairs), 'original_success': sum(a for a, _ in pairs),
                                     'policy_success': sum(b for _, b in pairs), 'gains': gains, 'losses': losses,
                                     'mcnemar_p': mcnemar(gains, losses)}
    for k in range(1, campaign()['rounds'] + 1):
        sampled = receipts(k)
        if sampled:
            curve.setdefault('sampling_success', {})[f'round_{k:02d}'] = {
                'episodes': len(sampled), 'successes': sum(bool(r['status']['success']) for r in sampled)}
    return curve


def gpu_seconds_bound(rows):
    total = 0.
    parse = lambda v: datetime.fromisoformat(v.replace('Z', '+00:00'))
    for row in rows:
        if row['state'] in TERMINAL and row.get('start_time') and row.get('complete_time'):
            seconds = max(0., (parse(row['complete_time']) - parse(row['start_time'])).total_seconds())
        elif row['state'] in TERMINAL:
            seconds = 0.
        else:
            seconds = row['max_seconds']
        total += seconds * row['gpu_count']
    return total


# User 2026-11-03 10:4x: "还是把4张标准卡都给训练作业吧" -- the 4 Reserved standard GPUs are for the learner only.
# Samplers and evaluations use Spot. The learner takes Reserved when all 4 are free; if they are still in use
# (e.g. by another project) it waits up to this grace period before taking Spot, where an H100 node can queue long.
LEARNER_RESERVED_GRACE = 1800


def learner_ready_since(k):
    path = HERE / 'learner_ready.json'
    ready = json.loads(path.read_text()) if path.exists() else {}
    if str(k) not in ready:
        ready[str(k)] = time.time()
        save(path, ready)
    return ready[str(k)]


class Admission:
    def __init__(self, rows, current=0):
        config = campaign()
        self.final = current is None
        mine = [r for r in rows if r['state'] not in TERMINAL]
        jobs = account_jobs()
        self.cap, self.pending_cap = config['max_campaign_live_gpus'], config['max_pending_jobs']
        # User 2026-11-03 15:5x ("两个都做吧"): once every policy is published (final evaluation), up to
        # final_max_pending_jobs may wait in the congested Spot queue, and evaluations may also use the idle
        # Reserved quota (4, no learner needs it any more), which is not counted in the 32-GPU Spot cap.
        if self.final:
            self.pending_cap = config.get('final_max_pending_jobs', self.pending_cap)
        # User 2026-11-03 ("把H100训练卡单独计数吧"): the cap counts 5090 GPUs (samplers, evaluations); an H100
        # learner is counted on its own (one at a time, by construction of plan_tasks).
        self.learner_pool = config['learner_pool']
        self.live = sum(r['gpu_count'] for r in mine if r.get('pool', '5090') == '5090' and r.get('quota') != 'reserved')
        self.pending = sum(1 for r in mine if r['state'] != 'RUNNING')
        self.reserved = sum(j['gpus'] for j in jobs.values() if j['quota'] == 'RESERVED')
        if getattr(account_jobs, 'approximate', False):
            self.reserved = 4
        # A Reserved copy rejected in the last 30 min (e.g. quota not yet released): use Spot meanwhile.
        if any(r['state'] == NOT_CREATED and r['quota'] == 'reserved' and time.time() - r['submitted_at'] < 1800
               for r in rows):
            self.reserved = 4
        self.policy = config['learner_quota_policy']
        save(HERE / 'active_gpu_admission.json', {'at': time.time(), 'campaign_live_gpus': self.live,
                                                'campaign_pending_jobs': self.pending, 'account_reserved_gpus': self.reserved})

    def __call__(self, role, k=None):
        gpus = 4 if role == 'learn' else 1
        if self.pending >= self.pending_cap:
            raise RuntimeError('Too many queued jobs; wait for placement')
        quota = 'spot'
        if role == 'eval' and self.final and self.reserved + gpus <= 4:
            quota = 'reserved'
        counted = (role != 'learn' or self.learner_pool == '5090') and quota != 'reserved'
        if counted and self.live + gpus > self.cap:
            raise RuntimeError('Campaign GPU cap reached')
        if role == 'learn' and self.policy == 'reserved':
            quota = 'reserved'
        elif role == 'learn' and self.policy == 'reserved_if_free':
            if self.reserved + 4 <= 4:
                quota = 'reserved'
            elif k is not None and time.time() - learner_ready_since(k) < LEARNER_RESERVED_GRACE:
                raise RuntimeError('Learner waiting for the Reserved quota to be released')
        self.live += gpus if counted else 0
        self.pending += 1
        self.reserved += gpus if quota == 'reserved' else 0
        return quota


def plan_tasks(rows):
    """Ordered list of (role, round, payload, attempt, last_row) to submit now."""
    config = campaign()
    tasks = []
    k = next((r for r in range(1, config['rounds'] + 1) if d.policy_record(r) is None), None)
    if k is not None:
        collected = receipts(k)
        done = {r['episode'] - 1 for r in collected}
        abandoned = episode_attempts(rows, k) - done
        ready_to_learn = (len(done) == config['episodes_per_round'] or
                          (len(done) >= config['min_episodes_per_round'] and
                           len(done) + len(abandoned) == config['episodes_per_round']))
        if ready_to_learn:
            allowed, attempt, last = retry_state(rows, 'learn', k)
            if allowed:
                tasks.append(('learn', k, {'receipts': collected}, attempt, last))
        else:
            missing = [i for i in range(config['episodes_per_round'])
                       if i not in done and i not in abandoned and i not in active_episodes(rows, k)]
            existing = [r for r in rows if r['role'] == 'sample' and r['round'] == k]
            next_id = len(existing)
            for start in range(0, len(missing), config['episodes_per_sampler']):
                chunk = missing[start:start + config['episodes_per_sampler']]
                tasks.append(('sample', k, {'sampler_id': f's{next_id:03d}', 'episodes': chunk}, 1, None))
                next_id += 1
    for j in config['eval_rounds']:
        if d.policy_record(j) is None:
            continue
        for case in d.eval_cases_for(j):
            if policy_result(j, case) is not None:
                continue
            allowed, attempt, last = retry_state(rows, 'eval', j, case)
            if allowed:
                tasks.append(('eval', j, {'case': case}, attempt, last))
    return tasks, k


def main():
    config = campaign()
    deadline = config['created_at'] + config['max_wall_hours'] * 3600
    (HERE / 'jobs').mkdir(exist_ok=True)
    with (HERE / 'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save(HERE / 'controller_process.json', {'pid': os.getpid(), 'started_at': time.time()})
        client = d.aoss_client()
        try:
            while time.time() < deadline:
                try:
                    rows = inspect(client)
                    for row in rows:
                        if row['state'] in TERMINAL:
                            collect(client, row)
                    confirm_exited(rows)
                    tasks, current = plan_tasks(rows)
                    curve = learning_curve()
                    save(HERE / 'learning_curve.json', {'at': time.time(), **curve})
                    active = [r for r in rows if r['state'] not in TERMINAL]
                    save(HERE / 'status.json', {'at': time.time(), 'current_round': current,
                                              'published_policies': [k for k in range(1, config['rounds'] + 1) if d.policy_record(k)],
                                              'active_jobs': len(active), 'active_gpus': sum(r['gpu_count'] for r in active),
                                              'gpu_hours_bound': gpu_seconds_bound(rows) / 3600, 'pending_tasks': len(tasks)})
                    if current is None and not tasks and not active:
                        save(HERE / 'outcome.json', {'at': time.time(), 'learning_curve': curve,
                                                    'policies': {k: d.policy_record(k) for k in range(1, config['rounds'] + 1)}})
                        return
                    if gpu_seconds_bound(rows) > config['max_total_gpu_hours'] * 3600:
                        save(HERE / 'budget_stop.json', {'at': time.time(), 'gpu_hours_bound': gpu_seconds_bound(rows) / 3600})
                        time.sleep(60)
                        continue
                    launched = 0
                    if tasks:
                        with (d.ANALYSIS / 'rl_gpu_submit.lock').open('a') as submit_lock:
                            fcntl.flock(submit_lock, fcntl.LOCK_EX)
                            admission = Admission(rows, current)
                            for role, k, payload, attempt, last in tasks:
                                if launched >= admission.pending_cap:
                                    break
                                if last is not None and not (last.get('workers_exited') or workers_exited(last)):
                                    continue
                                try:
                                    quota = admission(role, k)
                                except RuntimeError as error:
                                    save(HERE / 'admission_wait.json', {'at': time.time(), 'reason': str(error), 'next': [role, k]})
                                    break
                                d.submit_unlocked(role, k, payload, attempt, quota)
                                launched += 1
                    time.sleep(5 if launched else 30)
                except Exception as error:
                    save(HERE / 'controller_error.json', {'at': time.time(), 'type': type(error).__name__,
                                                         'message': str(error).split('--command')[0][:300]})
                    print('controller_error', type(error).__name__, str(error)[:200], flush=True)
                    time.sleep(30)
            for row in inspect(client):
                if row['state'] not in TERMINAL:
                    assert describe(row['job'])['uid'] == row['uid']
                    subprocess.run([CLI, 'acp', 'jobs', 'stop', row['job'], '--workspace-name=share-space'],
                                   env=ENV, capture_output=True, timeout=45, check=True)
            save(HERE / 'deadline_reached.json', {'at': time.time(), 'only_this_campaign_stopped': True})
        finally:
            client.close()


if __name__ == '__main__':
    main()
