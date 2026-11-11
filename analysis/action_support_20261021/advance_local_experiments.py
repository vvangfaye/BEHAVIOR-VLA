"""Advance exactly two already planned checks when their own GPUs are free.

1. After conditioned SFT finishes: same-sample posthoc comparison on GPU1.
2. After all three SFT runs finish: one original-policy native episode on GPU0.
No training, episode or failed action is ever retried by this process.
"""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
TRAIN = HERE / 'local_vla_pilot_ada01'
NATIVE = HERE / 'local_closed_loop_v1'
METHODS = ('standard', 'conditioned', 'projected')


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def training_ready(method):
    folder = TRAIN / method
    completion = folder / 'completion.json'
    if not completion.exists():
        return False
    accepted = load(completion)
    run, result = load(folder / 'run.json'), load(folder / 'result.json')
    if not (accepted['accepted'] and accepted['worker_returncode'] == 0
            and result['complete'] and result['completed_steps'] == 1000
            and run['method'] == method and run['seed'] == 17
            and run['plan_sha256'] == sha(TRAIN / 'plan.json')
            and accepted['result_sha256'] == sha(folder / 'result.json')):
        raise ValueError('Training failed or does not match the frozen plan: ' + method)
    return True


def lease_idle_gpu(index):
    handle = (TRAIN / f'gpu{index}.lock').open('a')
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    try:
        info = subprocess.check_output(['nvidia-smi', '-i', str(index),
            '--query-gpu=name,memory.free,utilization.gpu', '--format=csv,noheader,nounits'],
            text=True, timeout=15).strip().split(',')
        apps = subprocess.check_output(['nvidia-smi', '-i', str(index),
            '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True, timeout=15).strip()
        if ('RTX 6000 Ada' not in info[0] or int(info[1]) < 32000
                or int(info[2]) != 0 or apps):
            handle.close()
            return None
        return handle
    except BaseException:
        handle.close()
        raise


def main():
    folder = HERE / 'local_advancement'
    folder.mkdir(exist_ok=True)
    own_lock = (folder / 'watcher.lock').open('a')
    fcntl.flock(own_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'launch.json').exists():
        raise ValueError('An advancement attempt already exists; inspect it before another run')
    pinned = {str(p): sha(p) for p in [TRAIN / 'plan.json', NATIVE / 'plan.json',
              HERE / 'evaluate_vla_posthoc.py', HERE / 'posthoc_vla_plan.json']}
    write(folder / 'launch.json', {'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
          'max_local_runtime_hours': 24, 'source_sha256': pinned,
          'max_extra_gpu_hours': 9, 'actions': ['posthoc', 'first_native'], 'no_retries': True})
    stop = [False]
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))
    processes, completed = {}, {}
    deadline = time.monotonic() + 24 * 3600
    last = None
    try:
        while time.monotonic() < deadline and not stop[0]:
            for path, expected in pinned.items():
                if sha(Path(path)) != expected:
                    raise ValueError('Planned source changed: ' + path)
            ready = {method: training_ready(method) for method in METHODS}
            for action in list(processes):
                item = processes[action]
                code = item['process'].poll()
                if code is None and time.monotonic() - item['start'] > item['limit']:
                    item['process'].terminate()
                    try:
                        code = item['process'].wait(timeout=90)
                    except subprocess.TimeoutExpired:
                        item['process'].kill()
                        code = item['process'].wait()
                if code is None:
                    continue
                result_path = item['result']
                result = load(result_path) if result_path.exists() else {}
                accepted = code == 0 and result.get('complete' if action == 'posthoc' else 'accepted') is True
                completed[action] = {'accepted': accepted, 'returncode': code, 'result': str(result_path)}
                item['log'].close()
                item['lease'].close()
                del processes[action]
                write(folder / (action + '-completion.json'), completed[action])
                print(json.dumps({'event': 'finished', 'action': action, **completed[action]}), flush=True)
            for action, gpu in [('posthoc', 1), ('first_native', 0)]:
                if action in completed or action in processes:
                    continue
                if not (ready['conditioned'] if action == 'posthoc' else all(ready.values())):
                    continue
                lease = lease_idle_gpu(gpu)
                if lease is None:
                    continue
                try:
                    if action == 'posthoc':
                        command = [sys.executable, '-u', str(HERE / 'evaluate_vla_posthoc.py'),
                            '--plan', str(TRAIN / 'plan.json'), '--trained-directory', str(TRAIN / 'conditioned'),
                            '--output', str(HERE / 'vla_posthoc_result')]
                        result_path = HERE / 'vla_posthoc_result/result.json'
                        limit = 3600
                    else:
                        plan = load(NATIVE / 'plan.json')
                        method, task, instance = plan['resource_bound']['first_job']
                        command = [sys.executable, '-u', str(NATIVE / 'frozen/closed_loop_worker.py'),
                            '--plan', str(NATIVE / 'plan.json'), '--method', method,
                            '--task', task, '--instance', str(instance)]
                        result_path = Path(plan['afs_root']) / method / f'{task}-{instance}/receipt.json'
                        limit = 28800
                    with (folder / (action + '-intent.json')).open('x') as stream:
                        json.dump({'gpu': gpu, 'command': command, 'max_seconds': limit,
                                   'created_at': datetime.now(timezone.utc).isoformat()}, stream, indent=2)
                    log = (folder / (action + '.log')).open('x')
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1')
                    process = subprocess.Popen(command, env=env, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
                    processes[action] = {'process': process, 'lease': lease, 'log': log,
                                         'start': time.monotonic(), 'limit': limit, 'result': result_path}
                    print(json.dumps({'event': 'started', 'action': action, 'gpu': gpu, 'pid': process.pid}), flush=True)
                except BaseException:
                    lease.close()
                    raise
            state = {'observed_at': datetime.now(timezone.utc).isoformat(), 'pid': os.getpid(),
                     'training_complete': ready, 'running': {k: v['process'].pid for k, v in processes.items()},
                     'completed': completed}
            write(folder / 'state.json', state)
            signature = (tuple(ready.values()), tuple(processes), tuple(completed))
            if signature != last:
                print(json.dumps(state), flush=True)
                last = signature
            if len(completed) == 2:
                return 0 if all(x['accepted'] for x in completed.values()) else 1
            threading.Event().wait(30)
        return 2
    finally:
        for item in processes.values():
            item['process'].terminate()
            try:
                item['process'].wait(timeout=90)
            except subprocess.TimeoutExpired:
                item['process'].kill()
                item['process'].wait()
            item['log'].close()
            item['lease'].close()


if __name__ == '__main__':
    raise SystemExit(main())
