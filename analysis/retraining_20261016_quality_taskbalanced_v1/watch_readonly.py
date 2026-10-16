"""Finite local observation of existing239707; no submission or lifecycle actions."""
import argparse
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seconds', type=int, default=900)
    p.add_argument('--target-update', type=int, default=1)
    args = p.parse_args()
    if not 1 <= args.seconds <= 3600 or not 1 <= args.target_update <= 5000:
        raise ValueError('Observation window or target out of bounds')
    receipt = json.loads((ROOT / 'formal_stage/receipt.mirrored.json').read_text())
    if receipt['identifiers'] != ['239707']:
        raise ValueError('Read-only observer is restricted to existing239707')
    with (ROOT / 'read_only_watch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        until = time.monotonic() + args.seconds
        failures = 0
        while time.monotonic() < until:
            time.sleep(45)
            commands = [
                ['python3', str(ROOT / 'observe_training.py'), '--verified-ipv6', '--formal-only'],
                ['python3', str(ROOT / 'audit_training_progress.py')],
            ]
            try:
                for command in commands:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=150)
                    if result.returncode:
                        raise RuntimeError('Read-only observation/audit failed')
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                failures += 1
                print(json.dumps({'event': 'observation_requires_inspection', 'job': '239707',
                                  'error_type': type(error).__name__, 'consecutive': failures}), flush=True)
                if failures >= 3:
                    break
                continue
            failures = 0
            snapshot = json.loads((ROOT / 'latest_runtime_observation.json').read_text())
            audit = json.loads((ROOT / 'latest_progress_audit.json').read_text())
            formal = snapshot['stages']['formal']
            steps = audit['records'][-1]['completed_updates'] if audit['records'] else 0
            counts = {}
            for key, value in formal['metadata'].items():
                if key.endswith('_stdout_tail'):
                    matches = re.findall(r'Cached original files (\d+)/(\d+)', value)
                    if matches:
                        counts[key] = list(map(int, matches[-1]))
            errors = {k: v for k, v in formal['metadata'].items() if k.endswith('failure.json')}
            item = {'at': snapshot['observed_at'], 'event': 'read_only_live_observation', 'job': '239707',
                    'state': formal['status']['application_status'], 'cache_counts': counts,
                    'actual_updates': steps, 'errors': errors}
            with (ROOT / 'bounded_watch_history.jsonl').open('a') as stream:
                stream.write(json.dumps(item) + '\n')
            print(json.dumps(item), flush=True)
            if steps >= args.target_update or errors or formal['status']['application_status'] in (
                    'FINISHED', 'FAIL', 'FAILED', 'KILLED', 'STOPPED', 'SUCCESS', 'SUCCEEDED'):
                break
        print(json.dumps({'event': 'bounded_read_only_watch_ended', 'job': '239707',
                          'cloud_job_stopped_by_observer': False}), flush=True)


if __name__ == '__main__':
    main()
