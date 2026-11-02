"""One bounded scheduler: one learner, then the same forty paired evaluations."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time

from deployment import HERE, SOURCE, CLI, ENV, TERMINAL, campaign, describe, save, submit, aoss_client


def inspect(client):
    paths = sorted((HERE / 'jobs').glob('*/submission.json'))
    def read(path):
        row = json.loads(path.read_text())
        cached = path.with_name('latest.json')
        previous = json.loads(cached.read_text()) if cached.exists() else None
        if previous and previous['state'] in TERMINAL:
            return previous
        job = describe(row['job'])
        if 'uid' not in row:
            row['uid'] = job['uid']
            save(path, row)
        assert job['uid'] == row['uid']
        row.update(state=job['state'], checked_at=time.time())
        for key in ['start_time', 'complete_time', 'update_time']:
            row[key] = job.get(key)
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
        if row['role'] == 'train':
            ready = json.loads((root / 'checkpoints/round_01/ready.json').read_text())
            update = json.loads((root / 'training_updates/train1/result.json').read_text())
            assert ready['variant'] == campaign()['variant'] and ready['round_trip']
            assert ready['optimizer_steps'] == update['optimizer_steps'] == 17
            assert not update['early_stopped'] and update['episodes'] == 32 and update['chunks'] == campaign()['source_chunks']
            assert json.loads((root / 'parity_gate.json').read_text())['passed']
            save(HERE / 'checkpoint.json', ready)
        else:
            instance, seed = row['case']
            source = root / 'final' / f'i{instance}_s{seed}'
            destination = HERE / 'evidence' / row['variant'] / f'i{instance}_s{seed}'
            for name in ['result.json', 'initial.json', 'spec.json', 'actions.json', 'goal_summary.json']:
                target = destination / name
                target.parent.mkdir(parents=True, exist_ok=True)
                data = (source / name).read_bytes()
                if target.exists():
                    assert target.read_bytes() == data
                else:
                    target.write_bytes(data)
            result = json.loads((destination / 'result.json').read_text())
            assert (result['instance'], result['seed']) == (instance, seed) and result['steps'] > 0
            assert json.loads((root / 'checks.json').read_text())['frozen_unchanged']
    save(folder / 'collected.json', {'at': time.time(), 'state': row['state'], 'bytes': len(body)})


def comparison(variant):
    config = campaign()
    pairs = []
    for instance, seed in config['eval_cases']:
        key = f'i{instance}_s{seed}'
        original = HERE / 'evidence/original' / key
        new = HERE / 'evidence' / variant / key
        assert hashlib.sha256((original / 'result.json').read_bytes()).hexdigest() == config['baseline_result_sha256'][key]
        if not (new / 'result.json').exists():
            continue
        before = json.loads((original / 'initial.json').read_text())
        after = json.loads((new / 'initial.json').read_text())
        assert before['physical_state'].keys() == after['physical_state'].keys()
        physical_error = max(abs(x - y) for k in before['physical_state'] for part in (0, 1)
                             for x, y in zip(before['physical_state'][k][part], after['physical_state'][k][part]))
        proprio_error = max(abs(x - y) for x, y in zip(before['proprio'], after['proprio']))
        assert physical_error <= 1e-4 and proprio_error <= 1e-4
        assert all(before[k] == after[k] for k in ['physics_dt', 'rendering_dt', 'observation_dt'])
        old = json.loads((original / 'result.json').read_text())
        result = json.loads((new / 'result.json').read_text())
        pairs.append({'instance': instance, 'seed': seed, 'valid_initial_state': True,
                      'baseline_score': old['score'], 'score': result['score'],
                      'baseline_success': old['success'], 'success': result['success']})
    n = len(pairs)
    return {'expected_pairs': len(config['eval_cases']), 'completed_pairs': n, 'complete': n == len(config['eval_cases']), 'pairs': pairs,
            'baseline_success_rate': sum(p['baseline_success'] for p in pairs) / n if n else None,
            'success_rate': sum(p['success'] for p in pairs) / n if n else None,
            'baseline_mean_score': sum(p['baseline_score'] for p in pairs) / n if n else None,
            'mean_score': sum(p['score'] for p in pairs) / n if n else None,
            'gains': sum(p['success'] and not p['baseline_success'] for p in pairs),
            'losses': sum(p['baseline_success'] and not p['success'] for p in pairs)}


def active_gpu_count(rows):
    # Count every active state for this owner, including pending/stopping jobs.
    states = ['WAITING', 'INIT', 'QUEUEING', 'CREATING', 'STARTING', 'RESTARTING',
              'RECOVERING', 'RUNNING', 'SUSPENDING', 'DELETING']
    jobs = {}
    for state in states:
        seen_in_this_state = set()
        for page in range(1, 201):
            reply = subprocess.run([CLI, 'acp', 'jobs', 'list', '--workspace-name=share-space',
                                    '--user-name=260010165', '--state=' + state,
                                    '--page-size=10', '--page-token=' + str(page), '-o', 'json'],
                                   env=ENV, capture_output=True, text=True, timeout=40, check=True)
            batch = [] if reply.stdout.strip() == 'No jobs found' else json.loads(reply.stdout)
            assert isinstance(batch, list)
            for job in batch:
                assert job['state'] == state and job['ownership']['user_name'] == '260010165'
                assert job['uid'] not in seen_in_this_state, 'Repeated page within state; no admission'
                seen_in_this_state.add(job['uid'])
                count = sum(int(spec.get('limits', {}).get('nvidia.com/gpu', 0)) * max(1, int(role.get('total_replicas', 1)), int(spec.get('replicas', 0)))
                            for role in job.get('roles', []) for spec in role.get('resource_spec', []))
                # A job may move from CREATING to RUNNING between state queries.
                # Count that UID once, conservatively using its largest observed request.
                count = max(count, jobs.get(job['uid'], {}).get('gpus', 0))
                jobs[job['uid']] = {'state': state, 'gpus': count}
            if len(batch) < 10:
                break
        else:
            raise RuntimeError('Incomplete pagination; submission closed')
    for row in rows:
        if row['state'] not in TERMINAL and row['uid'] not in jobs:
            jobs[row['uid']] = {'state': row['state'], 'gpus': row['gpu_count']}
    save(HERE / 'active_gpu_admission.json', {'at': time.time(), 'jobs': jobs,
                                            'total': sum(j['gpus'] for j in jobs.values())})
    return sum(j['gpus'] for j in jobs.values())


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


def may_retry(row):
    recovery_path = HERE / 'jobs' / row['id'] / 'input_recovery.json'
    if recovery_path.exists():
        recovery = json.loads(recovery_path.read_text())
        if recovery.get('old_uid') == row['uid'] and recovery.get('old_worker_exited') and recovery.get('accepted_optimizer_steps') == 0 and (HERE / 'input_restore_verified.json').exists():
            return True
    if row['state'] == 'SUSPENDED':
        return True
    progress = row.get('progress', {})
    error = progress.get('error_type')
    folder = HERE / 'jobs' / row['id'] / 'retrieved'
    candidates = list((folder / 'failures').glob('*.json')) if row['role'] == 'train' else [folder / 'failure.json']
    for path in candidates:
        if path.exists():
            failure = json.loads(path.read_text())
            if failure.get('type') in ['KeyboardInterrupt', 'ExecutionStopped', 'SimulatorInitializationError',
                                       'TimeoutError', 'EOFError', 'BrokenPipeError']:
                return True
    return error in ['KeyboardInterrupt', 'ExecutionStopped', 'SimulatorInitializationError',
                     'TimeoutError', 'EOFError', 'BrokenPipeError']


def workers_exited(row):
    job = describe(row['job'])
    assert job['uid'] == row['uid'] and job['state'] in TERMINAL
    reply = subprocess.run([CLI, 'acp', 'jobs', 'get-workers', row['job'], '--workspace-name=share-space'],
                           env=ENV, capture_output=True, text=True, timeout=40, check=True)
    return not any(word in reply.stdout for word in ['Running', 'Pending', 'Terminating'])


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
                    report = {v: comparison(v) for v in config['evaluation_variants']}
                    result = report[config['variant']]
                    save(HERE / 'comparisons.json', report)
                    active = [r for r in rows if r['state'] not in TERMINAL]
                    save(HERE / 'status.json', {'at': time.time(), 'workers': rows,
                                              'active_gpus': sum(r['gpu_count'] for r in active),
                                              'comparison': {k: v for k, v in result.items() if k != 'pairs'}})
                    if all(r['complete'] for r in report.values()):
                        save(HERE / 'outcome.json', {'at': time.time(), 'comparisons': report,
                                                    'scope': 'One fixed original-policy 32-trajectory batch; 17 optimizer updates; same task-specific fixed held-out cases'})
                        return
                    used = gpu_seconds_bound(rows)
                    tasks = []
                    if not (HERE / 'checkpoint.json').exists():
                        tasks.append(('train', None, config['variant']))
                    if config['control_needs_evaluation']:
                        tasks.extend(('eval', case, 'control') for case in config['eval_cases'])
                    if (HERE / 'checkpoint.json').exists():
                        tasks.extend(('eval', case, config['variant']) for case in config['eval_cases'])
                    global_active = None
                    launched = False
                    for role, case, variant in tasks:
                        relevant = [r for r in rows if r['role'] == role and r.get('case') == case and r['variant'] == variant]
                        last = max(relevant, key=lambda r: r['attempt']) if relevant else None
                        if last and (last['state'] == 'SUCCEEDED' or last['state'] not in TERMINAL):
                            continue
                        if last and (last['attempt'] >= config['attempt_cap'] or not may_retry(last) or not workers_exited(last)):
                            continue
                        if role == 'eval' and sum(r['gpu_count'] for r in active) >= config['eval_parallel_gpus']:
                            break
                        count = 4 if role == 'train' else 1
                        if sum(r['gpu_count'] for r in active) + count > config['eval_parallel_gpus']:
                            continue
                        if global_active is None:
                            global_active = active_gpu_count(rows)
                        if global_active + count > config['max_total_live_gpus']:
                            break
                        seconds = config['train_timeout_seconds' if role == 'train' else 'eval_timeout_seconds'] * count
                        if used + seconds > config['max_total_gpu_hours'] * 3600:
                            save(HERE / 'budget_stop.json', {'at': time.time(), 'gpu_hours_bound': used / 3600})
                            return
                        submit(role, case, 1 if last is None else last['attempt'] + 1, variant)
                        launched = True
                        break
                    time.sleep(2 if launched else 30)
                except Exception as error:
                    # Never log command strings containing signed object URLs.
                    save(HERE / 'controller_error.json', {'at': time.time(), 'type': type(error).__name__,
                                                         'message': str(error).split('--command')[0][:200]})
                    print('controller_error', type(error).__name__, flush=True)
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
