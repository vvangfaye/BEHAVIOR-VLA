"""Bounded continuation of the authorized training/export/evaluation sequence."""
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent


def save(value):
    value['observed_at'] = time.time()
    temporary = ROOT / 'continuation_status.json.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(ROOT / 'continuation_status.json')
    print(json.dumps(value), flush=True)


def run(name, *args):
    # These entry points expose only public state, never signed URLs or credentials.
    result = subprocess.run([sys.executable, str(ROOT / name), *args], capture_output=True,
                            text=True, timeout=300)
    if result.returncode:
        save({'phase': 'requires_reconciliation', 'entry': name, 'exit_code': result.returncode,
              'automatic_resubmission': False})
        raise RuntimeError('Continuation entry failed: ' + name)
    if result.stdout.strip():
        print(result.stdout.strip(), flush=True)


def main():
    with (ROOT / 'continuation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = time.time() + 2 * 3600
        save({'phase': 'waiting_for_training_or_export', 'deadline': deadline})
        while time.time() < deadline:
            # Both scripts preserve exact existing intent/UID and reject ambiguity.
            run('export_checkpoint.py', '--submit')
            acceptance_path = ROOT / 'export_acceptance.json'
            acceptance = json.loads(acceptance_path.read_text()) if acceptance_path.exists() else {}
            if acceptance.get('accepted') and acceptance.get('export_native_terminal'):
                launch = ROOT / 'controller_launch.json'
                if launch.exists():
                    save({'phase': 'existing_evaluation_requires_observation',
                          'automatic_resubmission': False})
                    return
                run('admit_evaluation.py', '--start')
                if not launch.exists():
                    raise RuntimeError('Evaluation admission did not create launch evidence')
                save({'phase': 'evaluation_controller_started',
                      'controller_pid': json.loads(launch.read_text())['pid']})
                return
            save({'phase': 'waiting_for_training_or_export', 'deadline': deadline})
            time.sleep(60)
        save({'phase': 'wait_deadline_reached', 'automatic_resubmission': False})


if __name__ == '__main__':
    main()
