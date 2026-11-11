"""Execute the remaining frozen native panel after its first episode passes.

Every task-instance is evaluated by all four methods, on the same GPU and with
fresh processes. Existing accepted episodes are reused; existing failed or
unfinished episodes require inspection and are never retried here.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from advance_local_experiments import lease_idle_gpu, load, sha, training_ready, write

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'local_closed_loop_v1'


def schedule(plan):
    rows = [(method, task, instance) for task, instance in plan['pairs']
            for method in plan['protocol']['methods']]
    assert len(rows) == 48 and len(set(rows)) == 48
    assert list(rows[0]) == plan['resource_bound']['first_job']
    return rows


def accepted(plan, method, task, instance):
    folder = Path(plan['afs_root']) / method / f'{task}-{instance}'
    receipt = folder / 'receipt.json'
    if not folder.exists():
        return None
    if not receipt.exists():
        raise ValueError('Existing episode lacks a terminal receipt; inspect before proceeding: ' + str(folder))
    value = load(receipt)
    if not value['accepted']:
        raise ValueError('Existing episode failed; do not retry automatically: ' + str(folder))
    if (value['method'] != method or value['pair'] != [task, instance]
            or value['plan_sha256'] != sha(ROOT / 'plan.json')
            or value['protocol_sha256'] != plan['protocol_sha256']):
        raise ValueError('Episode provenance differs from frozen plan')
    result_path = folder / 'results/json' / f'{task}_{instance}_0.json'
    if sha(result_path) != value['result_sha256']:
        raise ValueError('Native result changed after acceptance')
    return value


def summarize(plan):
    rows, receipts = [], {}
    for method, task, instance in schedule(plan):
        value = accepted(plan, method, task, instance)
        if value is None:
            continue
        row = {'method': method, **value['record']}
        rows.append(row)
        receipts[(method, task, instance)] = value
    summaries = {}
    for method in plan['protocol']['methods']:
        group = [r for r in rows if r['method'] == method]
        summaries[method] = {'episodes': len(group),
            'mean_score': sum(r['q_score'] for r in group) / len(group) if group else None,
            'successes': sum(r['success'] for r in group)}
    by_instance, comparisons = [], {}
    for task, instance in plan['pairs']:
        group = {method: receipts.get((method, task, instance)) for method in plan['protocol']['methods']}
        existing = [r for r in group.values() if r]
        by_instance.append({'task': task, 'instance': instance,
            'results': {m: v['record'] if v else None for m, v in group.items()},
            'initial_physics_identical': len({v['initial_state']['physics_sha256'] for v in existing}) == 1 if len(existing) == 4 else None})
    if len(rows) == 48:
        for alternative, baseline in [('standard', 'original1688977'), ('conditioned', 'standard'),
                                      ('projected', 'standard'), ('projected', 'conditioned')]:
            pairs = [(r['results'][alternative], r['results'][baseline]) for r in by_instance]
            differences = [a['q_score'] - b['q_score'] for a, b in pairs]
            comparisons[alternative + '_minus_' + baseline] = {
                'mean_score_difference': sum(differences) / len(differences),
                'score_wins': sum(d > 0 for d in differences), 'score_ties': sum(d == 0 for d in differences),
                'score_losses': sum(d < 0 for d in differences),
                'success_gains': sum(a['success'] and not b['success'] for a, b in pairs),
                'success_losses': sum(b['success'] and not a['success'] for a, b in pairs)}
    return {'complete': len(rows) == 48, 'accepted_episodes': len(rows), 'expected_episodes': 48,
            'methods': summaries, 'by_instance': by_instance, 'paired_comparisons': comparisons,
            'scope': 'Single training seed and12 instances from3 development tasks; no broad generalization claim',
            'plan_sha256': sha(ROOT / 'plan.json')}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    p.add_argument('--max-hours', type=float, default=48)
    args = p.parse_args()
    if not 0 < args.max_hours <= 72:
        raise ValueError('Controller lifetime must not exceed72 hours')
    plan = load(ROOT / 'plan.json')
    rows = schedule(plan)
    first = rows[0]
    first_ready = accepted(plan, *first)
    if not args.execute:
        print(json.dumps({'episodes': len(rows), 'first_episode_accepted': first_ready is not None,
                          'remaining_cap_gpu_hours': 47 * 8, 'would_submit_cloud_jobs': False}))
        return
    if first_ready is None:
        raise ValueError('Wait for the existing first native episode; do not duplicate it')
    if not all(training_ready(m) for m in ('standard', 'conditioned', 'projected')):
        raise ValueError('All three training runs must be complete')
    control = ROOT / 'panel_controller'
    control.mkdir(exist_ok=True)
    lock = (control / 'controller.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    attempts = control / 'attempts'
    attempts.mkdir(exist_ok=True)
    run = attempts / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%f')
    run.mkdir()
    write(run / 'launch.json', {'pid': os.getpid(), 'max_hours': args.max_hours,
          'plan_sha256': sha(ROOT / 'plan.json'), 'first_episode_reused': True,
          'controller_sha256': sha(Path(__file__)), 'no_episode_retries': True, 'gpu': 0})
    plan_hash = sha(ROOT / 'plan.json')
    def interrupted(*_):
        raise InterruptedError('Panel controller terminated; clean up its own worker')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    deadline = time.monotonic() + args.max_hours * 3600
    process, lease = None, None
    try:
        for method, task, instance in rows:
            if sha(ROOT / 'plan.json') != plan_hash:
                raise ValueError('Frozen panel plan changed during execution')
            if accepted(plan, method, task, instance) is not None:
                continue
            lease = lease_idle_gpu(0)
            while lease is None and time.monotonic() < deadline:
                threading.Event().wait(30)
                lease = lease_idle_gpu(0)
            if lease is None or time.monotonic() >= deadline:
                return 2
            case = f'{method}-{task}-{instance}'
            launches = control / 'episode_intents'
            launches.mkdir(exist_ok=True)
            with (launches / (case + '.json')).open('x') as stream:
                json.dump({'method': method, 'task': task, 'instance': instance,
                    'plan_sha256': sha(ROOT / 'plan.json'), 'started_at': datetime.now(timezone.utc).isoformat()}, stream)
            with (run / (case + '.log')).open('x') as log:
                process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'frozen/closed_loop_worker.py'),
                    '--plan', str(ROOT / 'plan.json'), '--method', method, '--task', task,
                    '--instance', str(instance)], cwd=HERE, env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'),
                    stdout=log, stderr=subprocess.STDOUT)
                write(control / 'state.json', {'case': case, 'pid': process.pid,
                      'state': 'running', 'observed_at': datetime.now(timezone.utc).isoformat()})
                print(json.dumps({'started': case, 'pid': process.pid}), flush=True)
                try:
                    code = process.wait(timeout=min(28800, max(1, deadline - time.monotonic())))
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        process.wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return 2
            process = None
            lease.close()
            lease = None
            if code != 0 or accepted(plan, method, task, instance) is None:
                raise ValueError('Native episode did not pass; inspect its preserved logs: ' + case)
            summary = summarize(plan)
            write(ROOT / 'panel_summary.json', summary)
            write(control / 'state.json', {'case': case, 'state': 'accepted',
                  'accepted_total': summary['accepted_episodes'],
                  'observed_at': datetime.now(timezone.utc).isoformat()})
            print(json.dumps({'completed': case, 'accepted_total': summary['accepted_episodes']}), flush=True)
        summary = summarize(plan)
        write(ROOT / 'panel_summary.json', summary)
        write(control / 'state.json', {'state': 'complete' if summary['complete'] else 'incomplete',
              'accepted_total': summary['accepted_episodes'],
              'observed_at': datetime.now(timezone.utc).isoformat()})
        return 0 if summary['complete'] else 1
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=90)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        if lease is not None:
            lease.close()


if __name__ == '__main__':
    raise SystemExit(main())
