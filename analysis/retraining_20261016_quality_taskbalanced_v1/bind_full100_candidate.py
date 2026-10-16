"""Bind only this trial's actual exported5000 checkpoint and accepted fixed20 results.

No cloud operations, no evaluation creation, no automatic enabling.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    first, dest = ROOT / 'paired_fixed20_step5000', ROOT / 'full100_followup'
    path = dest / 'plan.json'
    plan = load(path)
    if plan['execution_enabled'] or (dest / 'controller_launch.json').exists():
        raise ValueError('Never change an enabled or launched evaluation plan')
    needed = [first / 'reports/export.json', first / 'candidate_accepted.json', first / 'status.json']
    if not all(p.exists() for p in needed):
        print(json.dumps({'state': 'waiting_for_actual_new5000_and_fixed20', 'enabled': False}))
        return
    export, candidate, status = map(load, needed)
    first_plan = load(first / 'plan.json')
    if (status.get('state') != 'comparison_complete' or export['state'] != 'accepted'
            or export['completed_updates'] != 5000 or not export['full_afs_sha256_readback']
            or candidate.get('run_id') != plan['candidate_model_run'] or candidate.get('completed_updates') != 5000
            or export['manifest_sha256'] != candidate['checkpoint_manifest_sha256']):
        raise ValueError('New candidate has not completed its own export/fixed20 acceptance')
    if candidate['protocol_sha256'] != plan['parent_protocol_sha256']:
        raise ValueError('Fixed20 protocol differs')
    expected = set(map(tuple, first_plan['pairs']))
    if len(candidate['records']) != 20 or {(r['task'], r['instance_id']) for r in candidate['records']} != expected:
        raise ValueError('Incomplete fixed20 candidate')
    from compare_candidate import validate
    validate(candidate, first_plan['pairs'], plan['parent_protocol_sha256'])
    references = []
    for i, pair in enumerate(first_plan['pairs']):
        report_path = first / 'reports' / ('candidate-%02d.json' % i)
        report = load(report_path)
        if (report['state'] != 'accepted' or report['protocol_sha256'] != plan['parent_protocol_sha256']
                or report['checkpoint_manifest_sha256'] != export['manifest_sha256']
                or [report['record']['task'], report['record']['instance_id']] != pair):
            raise ValueError('Mixed/old candidate result in fixed20')
        for item in report['artifacts']:
            if not item['path'].startswith(plan['reused_candidate_afs_root'].rstrip('/') + '/'):
                raise ValueError('Artifact is outside this trial first20 source')
        references.append({'original_report_path': str(report_path),
                           'original_report_sha256': sha(report_path), 'report': report})
    for item in export['files']:
        rel = Path(item['relative_path'])
        if rel.is_absolute() or '..' in rel.parts or 'train_state' in rel.parts or item['bytes'] <= 0:
            raise ValueError('Invalid inference export file')
    plan.update(candidate_export_report_sha256=sha(needed[0]), candidate_reuse_source_sha256=sha(needed[1]),
                candidate_manifest_sha256=export['manifest_sha256'], candidate_files=export['files'],
                reuse_candidates=references, candidate_binding_complete=True)
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps(plan, indent=2) + '\n')
    tmp.replace(path)
    print(json.dumps({'state': 'actual_new_candidate_bound', 'reused_results': 20,
                      'enabled': False, 'jobs_created': 0}))


if __name__ == '__main__':
    main()
