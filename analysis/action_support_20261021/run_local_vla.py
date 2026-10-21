"""Run a bounded subset of the frozen pilot on one existing host GPU."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, choices=[0, 1], required=True)
    parser.add_argument('--methods', nargs='+', choices=['standard', 'conditioned', 'projected'], required=True)
    args = parser.parse_args()
    if len(args.methods) != len(set(args.methods)):
        raise ValueError('Do not repeat a training method')
    root = HERE / 'local_vla_pilot_ada01'
    lock = (root / f'gpu{args.gpu}.lock').open('a')
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path = root / 'plan.json'
    plan = load(plan_path)
    probe = load(HERE / 'local_gpu_probe_ada01/result.json')
    if not (probe['passed'] and probe['completed_updates'] == 2
            and probe['action_expert_attention_max_update'] > 0):
        raise ValueError('Local optimizer check did not pass')
    if not load(HERE / 'training_dispatch/stop_receipt.json')['stopped']:
        raise ValueError('Cloud dispatcher must stop before local training starts')
    if list((HERE / 'vla_pilot').glob('submission_*/submission_intent.json')):
        raise ValueError('Cloud training intent exists; reconcile before local training')
    for name, expected in plan['source_sha256'].items():
        if digest(HERE / name) != expected:
            raise ValueError('Frozen training source changed: ' + name)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    for method in args.methods:
        output = root / method
        output.mkdir(exist_ok=False)
        intent = {'method': method, 'gpu': args.gpu, 'runner_pid': os.getpid(),
                  'plan_sha256': digest(plan_path), 'max_seconds': 14400,
                  'started_at': datetime.now(timezone.utc).isoformat(), 'no_retry': True}
        (output / 'launch.json').write_text(json.dumps(intent, indent=2) + '\n')
        print(json.dumps({'event': 'launch', **intent}), flush=True)
        start = time.monotonic()
        with (output / 'worker.log').open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', str(HERE / 'train_vla_pilot.py'),
                '--plan', str(plan_path), '--method', method, '--seed', '17', '--output', str(output)],
                cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            intent['training_pid'] = child.pid
            (output / 'launch.json').write_text(json.dumps(intent, indent=2) + '\n')
            try:
                code = child.wait(timeout=14340)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    code = child.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
        result_path = output / 'result.json'
        result = load(result_path) if result_path.exists() else {}
        accepted = code == 0 and result.get('complete') is True and result.get('completed_steps') == 1000
        receipt = {**intent, 'worker_returncode': code, 'accepted': accepted,
                   'elapsed_seconds': time.monotonic() - start,
                   'completed_at': datetime.now(timezone.utc).isoformat(),
                   'result_sha256': digest(result_path) if result_path.exists() else None}
        (output / 'completion.json').write_text(json.dumps(receipt, indent=2) + '\n')
        print(json.dumps({'event': 'finished', **receipt}), flush=True)
        if not accepted:
            raise SystemExit(1)


if __name__ == '__main__':
    main()
