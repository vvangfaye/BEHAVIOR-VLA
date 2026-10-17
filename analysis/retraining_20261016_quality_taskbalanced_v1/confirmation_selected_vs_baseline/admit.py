"""Admit independent confirmation only after the complete tuning comparison."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'admission.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        selector = module('confirmation_model_selector', ROOT / 'select_candidate.py')
        with contextlib.redirect_stdout(io.StringIO()):
            selector.main()
        selection_path = ROOT / 'selection.json'
        if not selection_path.exists():
            print(json.dumps({'ready': False, 'reason': 'waiting_for_all_twenty_tuning_episodes', 'jobs_created': False}))
            return
        selection = load(selection_path)
        if not selection['confirmation_required']:
            print(json.dumps({'ready': False, 'reason': 'baseline_remains_best_continue_training_ablation', 'jobs_created': False}))
            return
        if not args.start:
            print(json.dumps({'ready': True, 'model': selection['selected_model'], 'jobs_created': False}))
            return
        control = module('confirmation_admitted_controller', ROOT / 'controller.py')
        plan = load(ROOT / 'plan.json')
        if not plan['execution_enabled']:
            if list((ROOT / 'reports').glob('candidate-*.json')) or list((ROOT / 'reports').glob('baseline-*.json')):
                raise ValueError('Confirmation outcomes already exist before selection admission')
            save(ROOT / 'admission.json', {'selection_sha256': digest(selection_path),
                 'selected_at': selection['selected_at'], 'admitted_at': time.time(),
                 'tuning_status_sha256': selection['tuning_status_sha256']})
            save(ROOT / 'prior_terminal.json', {'all_terminal': True,
                 'tuning_status_sha256': selection['tuning_status_sha256']})
            plan.update(candidate_checkpoint=selection['checkpoint'],
                        expected_checkpoint_manifest_sha256=selection['checkpoint_manifest_sha256'],
                        formal_contract=selection['formal_contract'], selected_model=selection['selected_model'],
                        execution_enabled=True, deadline_epoch=time.time() + 72 * 3600)
            control.verify_local(plan)
            save(ROOT / 'plan.json', plan)
        else:
            control.verify_local(plan)
        if (ROOT / 'controller_launch.json').exists():
            print(json.dumps({'ready': True, 'existing_controller_launch': True, 'jobs_resubmitted': False}))
            return
        python = '/home/faye/.local/share/behavior-tools/venv/bin/python'
        with (ROOT / 'controller.log').open('xb') as log:
            process = subprocess.Popen([python, str(ROOT / 'controller.py')], stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        save(ROOT / 'controller_launch.json', {'pid': process.pid, 'started_at': time.time(),
             'plan_sha256': digest(ROOT / 'plan.json'), 'entry': str(ROOT / 'controller.py')})
        print(json.dumps({'model': selection['selected_model'], 'controller_pid': process.pid,
                          'worker_execution_observed': False}))


if __name__ == '__main__':
    main()
