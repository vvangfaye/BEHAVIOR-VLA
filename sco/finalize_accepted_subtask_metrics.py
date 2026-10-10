"""Finalize the frozen panel from independently accepted native/history audits.

Reads an immutable accepted-export snapshot, never changes live scoring files.
Without --output this audits the current accepted subset. Final reports require
every original task, H5, start and annotation to be accounted for.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from artifact_acceptance import accepted_evidence_kind

SCOPE = dict(tasks=100, h5=400, supported_starts=7223, unsupported_annotations=889)
PANEL = 'c7d3b48dd114b639ae49281e32c13ac48d10d0c59830e721dd92abdbf70ba04f'
ASSIGNMENTS = '4ee917d23af7420d28b3ee79df753d7fc67c8998f7d10c0d32ce337b96fe115c'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def contained(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), 'Evidence path escapes result root')
    return path


def normalization(root, evidence_hashes):
    """Link old report schemas to existing proofs and read-only cloud identities."""
    path = root / 'accepted_earlier_native_evidence_index.json'
    if not path.exists():
        return {}, {}
    data = path.read_bytes()
    index = json.loads(data)
    require(index['panel_fingerprint'] == PANEL and index['metric_values_changed'] is False,
            'Earlier evidence index changes the scoring contract')
    evidence_hashes[path.name] = hashlib.sha256(data).hexdigest()
    documents = {}
    for record in index['records'].values():
        for key in ['proof_file', 'identity_metadata_file', 'platform_file', 'role_recheck_file']:
            if key not in record:
                continue
            name = record[key]
            raw = contained(root, name).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            require(digest == record[key + '_sha256'], 'Earlier evidence hash changed: ' + name)
            evidence_hashes[name] = digest
            documents[name] = json.loads(raw)
    return {int(k): v for k, v in index['records'].items()}, documents


def normalize_native_identity(row, record, documents):
    require(fingerprint(row) == record['original_merged_row_sha256'],
            'Earlier accepted metrics changed after evidence indexing')
    identity = documents[record['identity_metadata_file']]
    require(identity['panel_fingerprint'] == PANEL and identity['read_only'] is True
            and identity['assignment_fingerprint'] == ASSIGNMENTS
            and identity['scoring_changed'] is False, 'Identity read changed scoring')
    matches = [x for x in identity['rows'] if x['index'] == row['index']]
    require(len(matches) == 1 and 'error_type' not in matches[0], 'Earlier identity evidence missing')
    item = matches[0]
    original, marker = item['original_assignment'], item['producer_marker']
    require(marker['panel_fingerprint'] == PANEL and marker['assignment_index'] == row['index']
            and marker['episode_jobs'] == 4 and marker['sample_ids'] == original['sample_ids'],
            'Original and producing H5/start identities differ')
    require(original['tasks'] == row['tasks'] and len(original['sample_ids']) == row['audited_samples'],
            'Earlier identity metadata changes task coverage')
    result = dict(row)
    require(row.get('original_scope_and_sample_ids_matched', True) is True,
            'Earlier report explicitly rejects original scope')
    for key in ['sample_ids', 'episode_keys']:
        require(key not in row or row[key] == original[key], 'Recorded native identity differs')
        result[key] = original[key]
    result['original_scope_and_sample_ids_matched'] = True
    return result


def audit(root, panel_path):
    panel_bytes = panel_path.read_bytes()
    panel = json.loads(panel_bytes)
    require(panel['fingerprint'] == PANEL == fingerprint({k: v for k, v in panel.items() if k != 'fingerprint'}), 'Frozen panel fingerprint changed')
    require(len(panel['samples']) == 7223 and len(panel['exclusions']) == 889 and len(panel['episodes']) == 400, 'Frozen panel scope changed')
    ids = [s['sample_id'] for s in panel['samples']]
    exclusions = [s['sample_id'] for s in panel['exclusions']]
    require(len(set(ids)) == 7223 and len(set(exclusions)) == 889 and not set(ids).intersection(exclusions), 'Duplicate panel annotation identities')
    require(all(x['reason'] == 'unsupported_predicate' for x in panel['exclusions']), 'Unsupported annotation disposition changed')
    by_task = defaultdict(set)
    episodes = defaultdict(set)
    expected_labels = defaultdict(Counter)
    for sample in panel['samples']:
        by_task[sample['task']].add(sample['sample_id'])
        expected_labels[(sample['task'], sample['subtask'])]['scheduled_segments'] += 1
    for sample in panel['exclusions']:
        expected_labels[(sample['task'], sample['subtask'])]['unsupported'] += 1
    for episode in panel['episodes']:
        episodes[episode['task']].add((episode['task'], episode['task_instance_id']))
    require(len(by_task) == 100 and all(len(x) == 4 for x in episodes.values()), 'Original four-H5 task coverage changed')

    exported = json.loads((root / 'accepted_success_rates_export_current.json').read_bytes())
    snapshot = contained(root, exported['input_and_output_snapshot'])
    require(json.loads((snapshot / 'export_report.json').read_bytes()) == exported, 'Export snapshot mismatch; retry a stable snapshot')
    require(exported['original_scope'] == SCOPE and exported['panel_fingerprint'] == PANEL, 'Accepted export scope changed')
    evidence_hashes = {}
    documents = {}
    for name, expected in {**exported['input_sha256'], **exported['outputs']}.items():
        data = contained(snapshot, name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == expected, 'Snapshot hash mismatch: ' + name)
        evidence_hashes[str(Path(exported['input_and_output_snapshot']) / name)] = expected
        if name.endswith('.json'):
            documents[name] = json.loads(data)
    scope_audit = documents['full100_completion_audit_current.json']
    metrics = documents['completed_subtask_metrics_partial.json']
    require(scope_audit['scope'] == SCOPE and metrics['panel_fingerprint'] == PANEL, 'Audit scope mismatch')
    tasks = {t['index']: t for t in scope_audit['tasks']}
    rows = {t['index']: t for t in metrics['assignments']}
    require(len(scope_audit['tasks']) == len(tasks) == 100 and set(tasks) == set(range(100)), 'Missing original task identities')
    require({t['task'] for t in tasks.values()} == set(by_task), 'Task names differ from frozen panel')
    accepted = {i for i, t in tasks.items() if t['state'] == 'full_task_independent_artifact_audit_accepted'}
    require(accepted == set(rows) == set(exported['accepted_task_indices']), 'Accepted assignment membership mismatch')
    require(exported['accepted_task_count'] == len(accepted)
            and exported['pending_task_count'] == 100 - len(accepted)
            and exported['accepted_original_h5'] == 4 * len(accepted), 'Accepted export counts differ')
    require(len(rows) == len(metrics['assignments']), 'Duplicate accepted assignments')
    selected = [s for s in metrics['subtasks'] if s['assignment_index'] in accepted]
    require(len(selected) == len(metrics['subtasks']), 'Unaccepted subtask rows included')
    require(len({(s['assignment_index'], s['task'], s['subtask']) for s in selected}) == len(selected), 'Duplicate subtask labels')
    normalized, normalized_documents = normalization(root, evidence_hashes)
    original_rows = rows
    rows = dict(rows)
    for i, record in normalized.items():
        require(i in accepted and accepted_evidence_kind(rows[i]) == 'recorded_native_fields',
                'Earlier schema index includes unaccepted/history results')
        rows[i] = normalize_native_identity(rows[i], record, normalized_documents)
    missing_metadata = [i for i in sorted(accepted)
                        if (not tasks[i].get('independent_proof_file') and i not in normalized)
                        or not rows[i].get('sample_ids')
                        or not rows[i].get('episode_keys')]
    require(not missing_metadata,
            'Earlier accepted audit metadata needs normalization for tasks: '
            + ','.join(map(str, missing_metadata)))
    protocols = Counter()
    cached_proofs = {}
    seen = set()
    for i in sorted(accepted):
        t, row = tasks[i], rows[i]
        task = t['task']
        require(t['planned_h5'] == 4 and row['tasks'] == [task], 'Task/H5 scope mismatch')
        require(row['original_scope_and_sample_ids_matched'] is True, 'Original scope audit absent')
        require(set(row['sample_ids']) == by_task[task] and len(row['sample_ids']) == len(by_task[task]), 'Original sample IDs changed')
        require(not seen.intersection(row['sample_ids']), 'Samples counted in multiple tasks')
        seen.update(row['sample_ids'])
        require({tuple(x) for x in row['episode_keys']} == episodes[task] and len(row['episode_keys']) == 4, 'Original H5 identities changed')
        require(row['audited_samples'] == row['expected_samples'] == len(by_task[task]) == t['independent_audited_starts'], 'Start accounting mismatch')
        require(row['eligible'] == t['eligible'] and row['successes'] == t['successes'], 'Accepted task counts differ')
        protocols[accepted_evidence_kind(row)] += 1
        require(not t.get('recorded_demonstration_identity_correction_pending') and not t.get('zero_eligible_annotation_identity_still_unverified'), 'Required correction remains pending')
        old = normalized.get(i)
        proof_file = t.get('independent_proof_file') or old['proof_file']
        path = contained(root, proof_file)
        if path not in cached_proofs:
            data = path.read_bytes()
            cached_proofs[path] = json.loads(data)
            evidence_hashes[str(path.relative_to(root.resolve()))] = hashlib.sha256(data).hexdigest()
        proof = cached_proofs[path]
        require(proof['panel_fingerprint'] == PANEL, 'Independent audit panel identity mismatch')
        if old and old['proof_kind'] == 'historical_existing_worker_native_audit':
            recheck = normalized_documents[old['role_recheck_file']]
            require(i == 24 and proof['kind'] == 'completed_task_subtask_metrics_partial'
                    and recheck['state'] == 'complete'
                    and recheck['native_raw_and_original_annotation_roles_verified'] is True
                    and i in recheck['completed_indices'], 'Historical worker native audit unverified')
        else:
            require(proof['stage'] == 'complete' and proof['gpu_count'] == 0, 'Independent CPU audit incomplete')
            platform = proof.get('accepted_platform_success')
            if platform is None and old:
                candidates = [x for x in normalized_documents[old['platform_file']]['rows']
                              if x['name'] == proof['name']]
                require(len(candidates) == 1, 'Earlier CPU platform identity missing')
                platform = {k: candidates[0][k] for k in ['name', 'uid', 'state']}
            require(platform is not None and platform['state'] == 'SUCCEEDED'
                    and platform['name'] == proof['name'] and len(platform['uid']) == 32,
                    'Independent CPU audit platform success unverified')
            expected_name = t.get('independent_audit_name') or row.get('corrective_audit_name')
            expected_uid = t.get('independent_audit_uid') or row.get('corrective_audit_uid')
            if expected_name:
                require(platform['name'] == expected_name, 'Independent CPU name differs')
            if expected_uid:
                require(platform['uid'] == expected_uid, 'Independent CPU UID differs')
        proof_rows = [x for x in proof['assignments'] if x['index'] == i]
        require(len(proof_rows) == 1, 'Independent audit assignment missing or duplicated')
        pr = proof_rows[0]
        if old:
            require(all(pr.get(k) == original_rows[i].get(k)
                        for k in ['tasks', 'expected_samples', 'audited_samples', 'eligible',
                                  'successes', 'setup_exclusions']), 'Earlier proof metrics differ')
            for key in ['sample_ids', 'episode_keys']:
                require(key not in pr or pr[key] == row[key], 'Earlier proof changes original identities')
            pr = dict(pr, sample_ids=row['sample_ids'], episode_keys=row['episode_keys'])
        require(accepted_evidence_kind(pr) == accepted_evidence_kind(row), 'Restoration evidence contract changed')
        for key in ['tasks', 'sample_ids', 'episode_keys', 'expected_samples', 'audited_samples', 'eligible', 'successes', 'setup_exclusions', 'producing_execution', 'producing_code_root', 'producing_source_manifest_sha256']:
            require(pr.get(key) == row.get(key), 'Merged metrics differ from independent proof: ' + key)
        subs = [s for s in selected if s['assignment_index'] == i]
        proven = [s for s in proof['subtasks'] if s['assignment_index'] == i]
        require(sorted(subs, key=lambda s: s['subtask']) == sorted(proven, key=lambda s: s['subtask']), 'Subtasks differ from independent proof')
        require({(s['task'], s['subtask']) for s in subs} == {k for k in expected_labels if k[0] == task}, 'Frozen subtask label coverage changed')
        for s in subs:
            expected = expected_labels[(s['task'], s['subtask'])]
            require(s['scheduled_segments'] == expected['scheduled_segments'] and s['unsupported'] == expected['unsupported'], 'Annotation dispositions changed')
            require(s['annotated_segments'] == s['scheduled_segments'] + s['unsupported'], 'Annotations unaccounted for')
            require(s['source_excluded'] == 0 and s['valid_results'] == s['scheduled_segments'] == s['metric_eligible'] + s['setup_excluded'], 'Scheduled starts unaccounted for')
            require(0 <= s['successes'] <= s['metric_eligible'], 'Invalid success denominator')
        require(sum(s['metric_eligible'] for s in subs) == row['eligible'] and sum(s['successes'] for s in subs) == row['successes'], 'Subtask/task totals differ')
        require(sum(row['setup_exclusions'].values()) + row['eligible'] == row['audited_samples'], 'Setup exclusions do not account for starts')

    successes = sum(row['successes'] for row in rows.values())
    eligible = sum(row['eligible'] for row in rows.values())
    require(exported['accepted_successes'] == successes and exported['accepted_eligible'] == eligible and exported['accepted_supported_starts'] == len(seen), 'Export aggregate mismatch')
    with (snapshot / 'task_success_rates_acceptance_current.csv').open(encoding='utf-8-sig', newline='') as file:
        task_csv = list(csv.DictReader(file))
    require(len(task_csv) == 100 and {int(t['assignment_index']) for t in task_csv} == set(tasks),
            'Task CSV omits original tasks')
    for t in task_csv:
        i = int(t['assignment_index'])
        require(t['task'] == tasks[i]['task'] and int(t['planned_h5']) == 4, 'Task CSV changes scope')
        require((t['independently_accepted'] == 'True') == (i in accepted), 'Task CSV changes acceptance')
        if i in accepted:
            row = rows[i]
            require(int(t['audited_h5']) == 4 and int(t['audited_starts']) == row['audited_samples']
                    and int(t['eligible']) == row['eligible'] and int(t['successes']) == row['successes'],
                    'Task CSV changes counts')
            require((float(t['success_rate']) == row['successes'] / row['eligible'])
                    if row['eligible'] else t['success_rate'] == '', 'Task CSV changes rate')
        else:
            require(t['success_rate'] == '' and t['eligible'] == '' and t['successes'] == '',
                    'Task CSV assigns a rate to unaccepted results')
    with (snapshot / 'subtask_success_rates_accepted_current.csv').open(encoding='utf-8-sig', newline='') as file:
        sub_csv = list(csv.DictReader(file))
    by_label = {(s['assignment_index'], s['task'], s['subtask']): s for s in selected}
    require(len(sub_csv) == len(by_label), 'Subtask CSV changes label count')
    csv_labels = set()
    for s in sub_csv:
        key = (int(s['assignment_index']), s['task'], s['subtask'])
        require(key in by_label and key not in csv_labels, 'Subtask CSV changes label identities')
        csv_labels.add(key)
        row = by_label[key]
        for field in ['annotated_segments', 'scheduled_segments', 'valid_results', 'metric_eligible',
                      'successes', 'setup_excluded', 'unsupported']:
            require(int(s[field]) == row[field], 'Subtask CSV changes counts: ' + field)
        require((float(s['success_rate']) == row['successes'] / row['metric_eligible'])
                if row['metric_eligible'] else s['success_rate'] == '', 'Subtask CSV changes rate')
    missing = sorted(set(range(100)) - accepted)
    complete = not missing
    if complete:
        require(seen == set(ids) and sum(s['unsupported'] for s in selected) == 889 and len(selected) == len(expected_labels), 'Full panel annotation coverage incomplete')
        require(protocols == Counter(recorded_native_fields=50, approved_physical_history=50), 'Expected two approved restoration cohorts not covered')
    summary = dict(kind='accepted_original_subtask_metrics_completion_audit', complete=complete,
                   scope=SCOPE, panel_fingerprint=PANEL, accepted_tasks=len(accepted),
                   audited_h5=4 * len(accepted), audited_starts=len(seen),
                   unsupported_annotations_accounted=sum(s['unsupported'] for s in selected),
                   missing_task_indices=missing, eligible=eligible, successes=successes,
                   success_rate=successes / eligible if eligible else None,
                   rate_definition='independent_subtask_pooled',
                   measured_continuous_whole_task_success_rate=None,
                   restoration_evidence_kinds=dict(protocols),
                   exact_legacy_grasp_anchor_bytes_verified=False,
                   input_snapshot=exported['input_and_output_snapshot'],
                   evidence_sha256=evidence_hashes,
                   frozen_panel_sha256=hashlib.sha256(panel_bytes).hexdigest())
    return summary, rows, selected


def finalize(output, summary, rows, subtasks):
    require(summary['complete'] is True, 'All100 completion unproven; no final reports written')
    require(summary['scope'] == SCOPE and summary['audited_h5'] == 400
            and summary['audited_starts'] == 7223
            and summary['unsupported_annotations_accounted'] == 889
            and summary['accepted_tasks'] == 100 and not summary['missing_task_indices'],
            'Full frozen scope not accounted for')
    require(not output.exists(), 'Final output directory already exists')
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    task_rows = [dict(assignment_index=i, task=row['tasks'][0], audited_h5=4,
                      audited_starts=row['audited_samples'], eligible=row['eligible'],
                      successes=row['successes'], success_rate=row['successes'] / row['eligible'] if row['eligible'] else '',
                      restoration_evidence_kind=accepted_evidence_kind(row),
                      metric_definition='independent_subtask_pooled', full100_complete=True)
                 for i, row in sorted(rows.items())]
    sub_rows = [dict(assignment_index=s['assignment_index'], task=s['task'], subtask=s['subtask'],
                     annotated_segments=s['annotated_segments'], scheduled_segments=s['scheduled_segments'],
                     metric_eligible=s['metric_eligible'], setup_excluded=s['setup_excluded'],
                     unsupported=s['unsupported'], successes=s['successes'],
                     success_rate=s['successes'] / s['metric_eligible'] if s['metric_eligible'] else '',
                     metric_definition='independent_subtask_pooled', full100_complete=True)
                for s in sorted(subtasks, key=lambda s: (s['assignment_index'], s['subtask']))]
    for name, values in [('task_success.csv', task_rows), ('subtask_success.csv', sub_rows)]:
        with (output / name).open('w', encoding='utf-8-sig', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir()}
    marker = dict(complete=True, scope=SCOPE, panel_fingerprint=PANEL,
                  evidence_contract='independently_accepted_native_and_approved_history',
                  report_sha256=hashes)
    (output / '_SUCCESS').write_text(json.dumps(marker, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--panel', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    summary, rows, subtasks = audit(args.root.resolve(), args.panel.resolve())
    if args.output:
        finalize(args.output, summary, rows, subtasks)
    print(json.dumps({k: v for k, v in summary.items() if k != 'evidence_sha256'}))


if __name__ == '__main__':
    main()
