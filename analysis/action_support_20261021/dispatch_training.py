"""Finite local dispatcher for the three already frozen VLA pilot jobs.

It never retries a cloud job, changes a training plan, or submits evaluations.
An OS lock excludes another dispatcher. Every create still uses the existing
exclusive per-method submission intent and exact-name reconciliation.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
METHODS = ('standard', 'conditioned', 'projected')
PREFIX = 'behavior1k/action-support-20261021/vla-pilot-v1/'
TERMINAL = {'SUCCEEDED', 'FAILED', 'STOPPED', 'CANCELED', 'CANCELLED', 'DELETED'}


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def decision(probe, submitted, data_ready, results):
    """Gate actions on observations, never on queue elapsed time."""
    if probe is None:
        return 'wait_probe', None
    if not (probe.get('passed') and probe.get('worker_returncode') == 0
            and probe.get('completed_updates') == 2):
        return 'stop_failed_probe', None
    if any(not result.get('complete') or result.get('worker_returncode') != 0
           or result.get('completed_steps') != 1000 for result in results.values()):
        return 'stop_failed_training', None
    if len(results) == 3:
        return 'complete', None
    if 'standard' not in submitted:
        return 'submit', 'standard'
    for method in METHODS[1:]:
        if method not in submitted:
            return ('submit', method) if data_ready else ('wait_shared_data', None)
    return 'wait_training', None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=float, default=24.)
    args = parser.parse_args()
    if not 0 < args.hours <= 24:
        raise ValueError('Dispatcher lifetime must be between0 and24 hours')
    folder = HERE / 'training_dispatch'
    folder.mkdir(exist_ok=True)
    lock = (folder / 'dispatcher.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'launch.json').exists():
        raise RuntimeError('Dispatcher already has a launch receipt; audit it before any new run')
    pinned = {str(p.relative_to(HERE)): digest(p) for p in [
        HERE / 'submit_vla_training.py', HERE / 'vla_training_bootstrap.py',
        HERE / 'inspect_progress.py', HERE / 'vla_pilot/pilot_training_plan.json']}
    launch = {'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
              'max_hours': args.hours, 'methods': METHODS, 'new_gpu_hours_cap': 12,
              'source_sha256': pinned, 'no_retries': True, 'no_evaluation_submission': True}
    write(folder / 'launch.json', launch)
    sys.path.insert(0, str(PROJECT / 'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    from botocore.exceptions import ClientError
    client = aoss_client()
    deadline = time.monotonic() + args.hours * 3600
    last_event = None
    def remote_json(key):
        try:
            response = client.get_object(Bucket='260010165', Key=key)
        except ClientError as error:
            if error.response.get('Error', {}).get('Code') in ('NoSuchKey', 'NotFound', '404'):
                return None
            raise
        try:
            raw = response['Body'].read(256 * 1024 + 1)
        finally:
            response['Body'].close()
        if len(raw) > 256 * 1024:
            raise ValueError('Unexpected result object size')
        return json.loads(raw)
    try:
        while time.monotonic() < deadline:
            for relative, expected in pinned.items():
                if digest(HERE / relative) != expected:
                    raise ValueError('Pinned dispatcher dependency changed: ' + relative)
            try:
                observation = subprocess.run([sys.executable, str(HERE / 'inspect_progress.py')],
                    capture_output=True, text=True, timeout=90)
                if observation.returncode:
                    raise RuntimeError('Probe observation failed; do not infer terminal state')
                observed_probe = json.loads(observation.stdout)
                probe_path = HERE / 'gpu_probe_job/result.json'
                probe = load(probe_path) if probe_path.exists() else None
                if probe is None and observed_probe['gpu_job']['state'] in TERMINAL:
                    raise ValueError('GPU check ended without an accepted result')
                submitted, results, jobs, progress = set(), {}, {}, {}
                for method in METHODS:
                    work = HERE / 'vla_pilot' / ('submission_seed17_' + method)
                    intent_path = work / 'submission_intent.json'
                    if not intent_path.exists():
                        continue
                    submitted.add(method)
                    intent = load(intent_path)
                    if not intent.get('submission_attempted'):
                        raise ValueError('Incomplete submission preparation requires inspection: ' + method)
                    response = subprocess.run([str(Path.home() / '.local/bin/sco'), 'acp', 'jobs', 'describe',
                        intent['job_name'], '--workspace-name=share-space', '-o', 'json'],
                        capture_output=True, text=True, timeout=45)
                    if response.returncode:
                        raise RuntimeError('Exact-name job observation failed; preserve existing submission')
                    raw = json.loads(response.stdout)
                    if intent.get('uid') and raw['uid'] != intent['uid']:
                        raise ValueError('Job UID changed')
                    if raw['ownership']['user_name'] != '260010165':
                        raise ValueError('Job owner changed')
                    jobs[method] = {k: raw.get(k) for k in ('uid', 'name', 'state', 'start_time', 'complete_time')}
                    write(work / 'current_status.json', jobs[method])
                    result = remote_json(PREFIX + method + '-s17-result.json')
                    if result is not None:
                        results[method] = result
                        write(work / 'result.json', result)
                    elif raw['state'] in TERMINAL:
                        raise ValueError('Cloud job ended without a result; inspect before further submissions')
                    latest = remote_json(PREFIX + method + '-s17-progress.json')
                    if latest is not None:
                        progress[method] = latest
                        write(work / 'progress.json', latest)
                ready = remote_json(PREFIX + 'data_ready.json')
                data_ready = ready is not None and ready.get('ready') is True
                if data_ready:
                    plan = load(HERE / 'vla_pilot/pilot_training_plan.json')
                    if ready.get('data_plan_sha256') != plan['data_plan_sha256']:
                        raise ValueError('Shared data is from a different training plan')
                action, method = decision(probe, submitted, data_ready, results)
                state = {'observed_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                         'action': action, 'next_method': method, 'jobs': jobs,
                         'data_ready': data_ready, 'completed_results': list(results),
                         'training_progress': progress, 'observation_failed': False}
                write(folder / 'state.json', state)
                event = (action, method, tuple((k, v['state']) for k, v in jobs.items()))
                if event != last_event:
                    print(json.dumps({k: state[k] for k in ('observed_at', 'action', 'next_method', 'jobs')}), flush=True)
                    last_event = event
                if action == 'complete' or action.startswith('stop_'):
                    write(folder / 'terminal.json', state)
                    return 0 if action == 'complete' else 2
                if action == 'submit':
                    with (folder / (method + '-submission.log')).open('x') as output:
                        response = subprocess.run([sys.executable, str(HERE / 'submit_vla_training.py'),
                            '--method', method, '--submit'], stdout=output, stderr=subprocess.STDOUT, timeout=300)
                    if response.returncode:
                        raise ValueError('Submission returned nonzero; exact intent needs inspection')
                    # Reobserve immediately, but never replay the create call.
                    continue
            except (RuntimeError, subprocess.TimeoutExpired, ClientError) as error:
                write(folder / 'observation_error.json', {'observed_at': datetime.now(timezone.utc).isoformat(),
                      'error_type': type(error).__name__, 'terminal_state_unknown': True, 'no_resubmission': True})
                print(json.dumps({'event': 'observation_failed', 'error_type': type(error).__name__}), flush=True)
            threading.Event().wait(min(60., max(0., deadline - time.monotonic())))
        write(folder / 'terminal.json', {'reason': 'local_24h_dispatcher_deadline',
              'remote_jobs_modified': False, 'observed_at': datetime.now(timezone.utc).isoformat()})
        return 3
    finally:
        client.close()


if __name__ == '__main__':
    raise SystemExit(main())
