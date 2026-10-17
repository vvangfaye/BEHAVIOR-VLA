"""Finite candidate-only evaluation, disabled until training/export admission."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REFERENCE = PARENT / 'heldout20_baseline_vs_expert1k'
sys.path.insert(0, str(ROOT))


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


base = module('h50_reference_controller', REFERENCE / 'controller.py')
base.ROOT = ROOT
base.transport.ROOT = ROOT
load, save, sha = base.load, base.save, base.sha
_native_create = base.native_create
_observe = base.observe


def native_create(intent, command, cpu=False):
    if cpu:
        intent['name'] = 'b1k-h50-heldout20-1017-' + intent['id']
    return _native_create(intent, command, cpu)


base.native_create = native_create


def observe(intent):
    for attempt in range(4):
        try:
            return _observe(intent)
        except RuntimeError as exc:
            if not str(exc).startswith('SCO command failed with exit '):
                raise
            with (ROOT / 'query_retries.jsonl').open('a') as stream:
                stream.write(json.dumps({'at': time.time(), 'name': intent['name'],
                                         'uid': intent.get('uid'), 'attempt': attempt + 1}) + '\n')
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


base.observe = observe


def verify_local(plan):
    if not plan.get('execution_enabled'):
        raise ValueError('Training and export admission not enabled')
    if not time.time() < plan.get('deadline_epoch', 0) <= time.time() + 72 * 3600:
        raise ValueError('A finite future campaign deadline is required')
    proof = load(ROOT / 'runtime_preparation.json')
    if sha((REFERENCE / 'controller.py').read_bytes()) != proof['source_controller_sha256']:
        raise ValueError('Pinned transport/controller source changed')
    hashes = {p.name: sha(p.read_bytes()) for p in (ROOT / 'frozen').glob('*.py')}
    if hashes != proof['frozen_hashes'] or hashes != load(ROOT / 'frozen_sha256.json'):
        raise ValueError('Frozen runtime changed')
    if sha((ROOT / 'assemble.py').read_bytes()) != proof['assembler_sha256']:
        raise ValueError('Assembler changed')
    protocol = load(ROOT / 'protocol.json')
    if sha(json.dumps(protocol, sort_keys=True, separators=(',', ':')).encode()) != plan['protocol_sha256']:
        raise ValueError('Protocol changed')
    if protocol['pairs'] != plan['pairs'] or protocol['native_task_budgets'] != plan['native_task_budgets']:
        raise ValueError('Instances or budgets changed')
    assignments = sorted((m['role'], m['index']) for g in plan['groups'] for m in g['members'])
    if assignments != [('candidate', i) for i in range(20)]:
        raise ValueError('Exactly twenty candidate episodes required')
    if len(plan['groups']) != 10 or any(len(g['members']) != 2 for g in plan['groups']):
        raise ValueError('Expected ten two-episode jobs')
    if not 2 * plan['member_walltime_seconds'] <= plan['job_walltime_seconds'] <= 128 * 3600:
        raise ValueError('Finite resource budget changed')
    admission = load(ROOT / 'admission.json')
    if (admission.get('training_iid') != '241869' or admission.get('completed_updates') != 1000
            or admission.get('training_restore_verified') is not True
            or admission.get('export_native_terminal') is not True):
        raise ValueError('Exact completed training and export proof required')
    export = load(ROOT / 'reports/export.json')
    if (export.get('state') != 'accepted' or export.get('completed_updates') != 1000
            or export.get('full_afs_sha256_readback') is not True
            or export.get('manifest_sha256') != plan['expected_checkpoint_manifest_sha256']
            or export.get('checkpoint') != plan['candidate_checkpoint']
            or sha((ROOT / 'reports/export.json').read_bytes()) != admission['export_report_sha256']):
        raise ValueError('Export receipt differs from the admitted checkpoint')
    assembly = load(ROOT / 'assembly_validation.json')
    if (not assembly.get('passed') or assembly.get('members') != 20
            or assembly.get('assembler_sha256') != proof['assembler_sha256']
            or assembly.get('frozen_manifest_sha256') != sha((ROOT / 'frozen_sha256.json').read_bytes())):
        raise ValueError('Candidate-only assembly has not been validated')
    reference = module('h50_reference_report', REFERENCE / 'build_report.py')
    old_plan, pairs, models, _, missing = reference.collect()
    if missing:
        raise ValueError('Reference panel remains incomplete')
    reference.verify_native_completion(old_plan)
    reference.verify_exclusion(pairs)


def finish(plan, reports, states):
    reference = module('h50_completed_reference', REFERENCE / 'build_report.py')
    old_plan, pairs, models, proofs, missing = reference.collect()
    if missing:
        raise ValueError('Reference panel incomplete')
    reference.verify_native_completion(old_plan)
    rows = [reports[f'candidate-{i:02d}']['record'] for i in range(20)]
    candidate = base.validate({'protocol_sha256': plan['protocol_sha256'], 'records': rows}, pairs, plan['protocol_sha256'])
    all_models = {'baseline': models['baseline'], 'expert_mse': models['candidate'], 'huber50': candidate}
    save(ROOT / 'comparison.json', {'complete': True, 'protocol_sha256': plan['protocol_sha256'],
         'summaries': {name: base.summary(rows, pairs) for name, rows in all_models.items()},
         'huber50_vs_expert_mse': base.paired(models['candidate'], candidate, pairs),
         'huber50_vs_baseline': base.paired(models['baseline'], candidate, pairs),
         'candidate_report_sha256': {identity: sha((ROOT / 'reports' / (identity + '.json')).read_bytes()) for identity in reports},
         'reference_proofs': proofs, 'native_jobs': states, 'baseline_pretraining_exclusion_proven': False})
    report_builder = module('h50_validated_report', ROOT / 'build_report.py')
    report_builder.main()


def main():
    with (ROOT / 'controller.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = (ROOT / 'plan.json').read_bytes()
        plan = json.loads(raw)
        verify_local(plan)
        save(ROOT / 'launch.json', {'pid': os.getpid(), 'started_at': time.time(), 'plan_sha256': sha(raw)})
        client = base.transport.aoss_client()
        base.prepare_stage(client, plan)
        identities = [f'candidate-{i:02d}' for i in range(20)]
        while time.time() < plan['deadline_epoch']:
            if (ROOT / 'plan.json').read_bytes() != raw:
                raise ValueError('Running plan changed')
            if not base.stage_receipt(client, plan):
                time.sleep(30)
                continue
            if (ROOT / 'STOP').exists():
                return
            for group in plan['groups']:
                base.submit_group(plan, group)
            intents = base.tracked_intents(plan)
            with ThreadPoolExecutor(max_workers=6) as pool:
                states = list(pool.map(base.observe, intents))
                values = list(pool.map(lambda identity: base.fetch(client, plan, identity), identities))
            reports = dict(zip(identities, values))
            accepted = sum(bool(r and r.get('state') == 'accepted') for r in values)
            complete = accepted == 20 and len(states) >= 10 and all(s['state'] in base.TERMINAL for s in states)
            save(ROOT / 'status.json', {'observed_at': time.time(), 'accepted': accepted,
                 'states': states, 'complete': complete, 'gpu_jobs_created': len(intents)})
            if complete:
                finish(plan, reports, states)
                return
            if len(states) >= 10 and all(s['state'] in base.TERMINAL for s in states):
                save(ROOT / 'terminal_incomplete.json', {'accepted': accepted, 'at': time.time()})
                return
            time.sleep(60)


if __name__ == '__main__':
    while True:
        try:
            main()
            break
        except RuntimeError as exc:
            if not str(exc).startswith('SCO command failed with exit '):
                raise
            plan = load(ROOT / 'plan.json')
            if (ROOT / 'STOP').exists() or time.time() >= plan['deadline_epoch']:
                raise
            save(ROOT / 'observation_error.json', {'at': time.time(),
                 'category': 'native_read_query_failed_after_retries', 'automatic_job_resubmit': False})
            time.sleep(60)
