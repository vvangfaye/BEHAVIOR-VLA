"""Admit and optionally start evaluation only after training/export/reference proof."""
import argparse
import fcntl
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify():
    acceptance_path = ROOT / 'export_acceptance.json'
    if not acceptance_path.exists():
        return None
    acceptance = load(acceptance_path)
    if not acceptance.get('accepted') or not acceptance.get('export_native_terminal'):
        return None
    checks = [('reports/export.json', 'report_sha256'),
              ('completed_training_observation.json', 'training_observation_sha256'),
              ('private/export_source_snapshot.json', 'source_snapshot_sha256')]
    for name, key in checks:
        if digest(ROOT / name) != acceptance[key]:
            raise ValueError('Export acceptance evidence changed: ' + name)
    export_module = module('h50_admission_export', ROOT / 'export_checkpoint.py')
    if not export_module.training_ready(load(ROOT / 'completed_training_observation.json')):
        raise ValueError('Exact training completion is missing')
    plan, report = load(ROOT / 'plan.json'), load(ROOT / 'reports/export.json')
    export_module.verify_export(plan, report, load(ROOT / 'private/export_source_snapshot.json'),
                                digest(ROOT / 'frozen_sha256.json'))
    reference_root = PARENT / 'heldout20_baseline_vs_expert1k'
    reference = module('h50_admission_reference', reference_root / 'build_report.py')
    ref_plan, pairs, models, proofs, missing = reference.collect()
    if missing:
        return None
    native = reference.verify_native_completion(ref_plan)
    reference.verify_exclusion(pairs)
    if plan['pairs'] != [list(pair) for pair in pairs] or digest(reference_root / 'plan.json') != plan['reference_plan_sha256']:
        raise ValueError('Reference panel changed')
    if digest(PARENT / 'expert_huber50_candidate/selected_stage/bundle/inputs/formal_run_contract.json') != plan['training_contract_sha256']:
        raise ValueError('Selected training contract changed')
    return plan, report, acceptance, native


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admit', action='store_true')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'admission.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = verify()
        if result is None:
            print(json.dumps({'ready': False, 'reason': 'waiting_for_complete_training_export_and_reference',
                              'evaluation_started': False}))
            return
        plan, report, acceptance, native = result
        if not (args.admit or args.start):
            print(json.dumps({'ready': True, 'evaluation_started': False}))
            return
        control = module('h50_admitted_controller', ROOT / 'controller.py')
        intent = load(ROOT / 'intents/export.json')
        actual = control.base.observe(intent)
        if actual['uid'] != acceptance['native']['uid'] or actual['state'] != 'SUCCEEDED':
            raise ValueError('Exact export job is not successfully terminal')
        if not plan['execution_enabled']:
            save(ROOT / 'admission.json', {'training_iid': '241869', 'completed_updates': 1000,
                 'training_restore_verified': True, 'export_native_terminal': True,
                 'export_report_sha256': digest(ROOT / 'reports/export.json'),
                 'export_uid': actual['uid'], 'export_acceptance_sha256': digest(ROOT / 'export_acceptance.json')})
            save(ROOT / 'prior_terminal.json', {'all_terminal': True, 'reference': native})
            plan.update(expected_checkpoint_manifest_sha256=report['manifest_sha256'],
                        execution_enabled=True, deadline_epoch=time.time() + 72 * 3600)
            control.verify_local(plan)
            save(ROOT / 'plan.json', plan)
        else:
            control.verify_local(plan)
        if not args.start:
            print(json.dumps({'ready': True, 'admitted': True, 'evaluation_started': False}))
            return
        if (ROOT / 'controller_launch.json').exists():
            raise ValueError('Existing launch evidence; reconcile the original controller before restarting')
        python = Path('/home/faye/.local/share/behavior-tools/venv/bin/python')
        if not python.is_file():
            raise ValueError('Validated controller interpreter is unavailable')
        with (ROOT / 'controller.log').open('xb') as output:
            process = subprocess.Popen([str(python), str(ROOT / 'controller.py')],
                                       stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        save(ROOT / 'controller_launch.json', {'pid': process.pid, 'started_at': time.time(),
             'entry': str(ROOT / 'controller.py'), 'plan_sha256': digest(ROOT / 'plan.json')})
        print(json.dumps({'admitted': True, 'controller_pid': process.pid,
                          'worker_execution_observed': False}))


if __name__ == '__main__':
    main()
