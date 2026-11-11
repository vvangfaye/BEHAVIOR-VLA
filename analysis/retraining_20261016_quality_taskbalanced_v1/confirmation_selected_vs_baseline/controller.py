"""Finite independent confirmation after immutable full-panel model selection."""
import importlib.util
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REFERENCE = PARENT / 'heldout20_baseline_vs_expert1k'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


base = module('confirmation_reference_transport', REFERENCE / 'controller.py')
base.ROOT = ROOT
base.transport.ROOT = ROOT
load, save, sha = base.load, base.save, base.sha
_observe, _create, _finish = base.observe, base.native_create, base.finish


def observe(intent):
    for attempt in range(4):
        try:
            return _observe(intent)
        except RuntimeError as exc:
            if not str(exc).startswith('SCO command failed with exit '):
                raise
            save(ROOT / 'query_retry.json', {'at': time.time(), 'uid': intent.get('uid'), 'attempt': attempt + 1})
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def native_create(intent, command, cpu=False):
    if cpu:
        intent['name'] = 'b1k-confirm20-1017-' + intent['id']
    return _create(intent, command, cpu)


def verify_local(plan):
    if not plan.get('execution_enabled'):
        raise ValueError('Independent confirmation has not been admitted')
    if not time.time() < plan.get('deadline_epoch', 0) <= time.time() + 72 * 3600:
        raise ValueError('A finite future confirmation deadline is required')
    proof = load(ROOT / 'runtime_preparation.json')
    immutable = dict(plan)
    immutable.update(candidate_checkpoint=None, expected_checkpoint_manifest_sha256=None,
                     formal_contract=None, selected_model=None, execution_enabled=False, deadline_epoch=0)
    if sha(json.dumps(immutable, sort_keys=True, separators=(',', ':')).encode()) != proof['immutable_plan_sha256']:
        raise ValueError('Immutable confirmation plan fields changed')
    if sha((REFERENCE / 'controller.py').read_bytes()) != proof['reference_controller_sha256']:
        raise ValueError('Pinned shared transport changed')
    hashes = {p.name: sha(p.read_bytes()) for p in (ROOT / 'frozen').glob('*.py')}
    if hashes != proof['frozen_hashes'] or hashes != load(ROOT / 'frozen_sha256.json'):
        raise ValueError('Frozen evaluation runtime changed')
    if sha((ROOT / 'assemble.py').read_bytes()) != proof['assembler_sha256']:
        raise ValueError('Frozen assembler changed')
    assembly = load(ROOT / 'assembly_validation.json')
    if (not assembly['passed'] or assembly['groups_compiled'] != 10
            or assembly['episodes_covered_exactly_once'] != 40
            or assembly['assembler_sha256'] != proof['assembler_sha256']
            or assembly['frozen_manifest_sha256'] != sha((ROOT / 'frozen_sha256.json').read_bytes())):
        raise ValueError('Confirmation assembly validation differs')
    protocol = load(ROOT / 'protocol.json')
    if sha(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()) != plan['protocol_sha256']:
        raise ValueError('Confirmation protocol changed')
    if plan['pairs'] != protocol['pairs'] or plan['native_task_budgets'] != protocol['native_task_budgets']:
        raise ValueError('Confirmation instances or frame budgets changed')
    assignments = sorted((m['role'], m['index']) for g in plan['groups'] for m in g['members'])
    if assignments != sorted((role, i) for role in ('baseline', 'candidate') for i in range(20)):
        raise ValueError('Exactly forty paired confirmation episodes required')
    if len(plan['groups']) != 10 or any(len(g['members']) != 4 for g in plan['groups']):
        raise ValueError('Expected ten four-episode jobs')
    if not 4 * plan['member_walltime_seconds'] <= plan['job_walltime_seconds'] <= 128 * 3600:
        raise ValueError('Finite job runtime changed')
    selection = load(ROOT / 'selection.json')
    admission = load(ROOT / 'admission.json')
    if admission['selection_sha256'] != sha((ROOT / 'selection.json').read_bytes()):
        raise ValueError('Admitted selection changed')
    if selection['selected_model'] == 'baseline' or not selection['confirmation_required']:
        raise ValueError('Do not compare the baseline against itself')
    expected = {'selected_model': selection['selected_model'], 'candidate_checkpoint': selection['checkpoint'],
                'expected_checkpoint_manifest_sha256': selection['checkpoint_manifest_sha256'],
                'formal_contract': selection['formal_contract']}
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError('Selected model identity differs')
    tuning = PARENT / 'huber50_heldout20'
    if (sha((tuning / 'model_selection_policy.json').read_bytes()) != plan['model_selection_policy_sha256']
            or sha((tuning / 'comparison.json').read_bytes()) != selection['tuning_comparison_sha256']
            or sha((tuning / 'status.json').read_bytes()) != selection['tuning_status_sha256']):
        raise ValueError('Selected model evidence changed')
    tuning_report = module('confirmation_complete_tuning', tuning / 'build_report.py')
    if tuning_report.collect()['missing']:
        raise ValueError('Wait for all tuning episodes and native jobs')
    check = module('confirmation_exclusion_check', PARENT / 'prepare_confirmation_preflight.py')
    panel = load(PARENT / 'heldout20_confirmation_panel.json')
    check.verify_exclusion(panel)
    if panel['pairs'] != plan['pairs']:
        raise ValueError('Reserved confirmation panel changed')
    accepted = load(PARENT / 'heldout20_confirmation_preflight/acceptance.json')
    if (not accepted['accepted'] or accepted['native']['state'] != 'SUCCEEDED'
            or accepted['result_sha256'] != protocol['asset_proof_sha256']):
        raise ValueError('Confirmation asset proof is incomplete')


def finish(plan, reports):
    _finish(plan, reports)
    path = ROOT / 'comparison.json'
    result = load(path)
    result.update(scope='independent reserved confirmation panel', selected_model=plan['selected_model'],
                  selection_sha256=sha((ROOT / 'selection.json').read_bytes()),
                  selection_protocol_separate=True)
    save(path, result)
    module('confirmation_final_report', ROOT / 'build_report.py').main()


base.observe, base.native_create = observe, native_create
base.verify_local, base.finish = verify_local, finish


if __name__ == '__main__':
    while True:
        try:
            base.main()
            break
        except RuntimeError as exc:
            if not str(exc).startswith('SCO command failed with exit '):
                raise
            plan = load(ROOT / 'plan.json')
            if (ROOT / 'STOP').exists() or time.time() >= plan['deadline_epoch']:
                raise
            save(ROOT / 'observation_error.json', {'at': time.time(), 'category': 'native_read_query_failed',
                                                  'automatic_job_resubmit': False})
            time.sleep(60)
