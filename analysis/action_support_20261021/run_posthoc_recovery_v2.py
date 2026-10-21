"""Reviewed retry of a pre-model import failure; no automatic retries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from advance_local_experiments import lease_idle_gpu, load, sha, training_ready, write

HERE = Path(__file__).resolve().parent


def main():
    plan = load(HERE / 'posthoc_recovery_v2_plan.json')
    script = HERE / 'evaluate_vla_posthoc_v2.py'
    assert sha(script) == plan['new_source_sha256']
    assert training_ready('conditioned')
    failed = load(HERE / 'local_advancement/posthoc-completion.json')
    assert not failed['accepted'] and failed['returncode'] == 1
    log_path = HERE / 'posthoc_recovery_v2.log'
    if log_path.exists() or (HERE / 'vla_posthoc_result_v2').exists():
        raise ValueError('Recovery already attempted; do not resubmit')
    lease = lease_idle_gpu(1)
    if lease is None:
        raise RuntimeError('GPU1 is not idle; no process started')
    start = time.monotonic()
    code = 1
    try:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1')
        with log_path.open('x') as log:
            child = subprocess.Popen([sys.executable, '-u', str(script), '--plan',
                str(HERE / 'local_vla_pilot_ada01/plan.json'), '--trained-directory',
                str(HERE / 'local_vla_pilot_ada01/conditioned'), '--output',
                str(HERE / 'vla_posthoc_result_v2')], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=HERE)
            write(HERE / 'posthoc_recovery_v2_launch.json', {'pid': child.pid, 'runner_pid': os.getpid(),
                  'source_sha256': sha(script), 'gpu': 1, 'max_seconds': 3600})
            try:
                code = child.wait(timeout=3540)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    code = child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    code = child.wait()
        path = HERE / 'vla_posthoc_result_v2/result.json'
        accepted = code == 0 and path.exists() and load(path).get('complete') is True
        write(HERE / 'posthoc_recovery_v2_completion.json', {'accepted': accepted, 'returncode': code,
              'elapsed_seconds': time.monotonic() - start, 'result': str(path),
              'first_failure_preserved': True, 'no_training_changes': True})
        if accepted:
            write(HERE / 'posthoc_result_source.json', {'directory': 'vla_posthoc_result_v2',
                  'result_sha256': sha(path), 'reason': 'Reviewed import-order recovery, same scientific protocol'})
        print(json.dumps({'accepted': accepted, 'returncode': code}), flush=True)
        return 0 if accepted else 1
    finally:
        lease.close()


if __name__ == '__main__':
    raise SystemExit(main())
