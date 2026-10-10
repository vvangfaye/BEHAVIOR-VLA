"""Count frozen public-test originals after rehashing local bytes, without scoring partial coverage."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from public_test_artifacts import validate_report


def count_originals(root):
    plan = json.loads((root / 'production_plan.json').read_text())
    allowed = {(row['task'], i, 0) for row in plan['shards'] for i in row['instance_ids']}
    allowed.add(('turning_on_radio', 301, 0))
    assert len(allowed) == plan['total_rollouts'] == 2000
    assert plan['checkpoint_step'] == 1688977
    pilot = json.loads((root / 'pilot_accepted.json').read_text())
    assert pilot['state'] == 'accepted' and pilot['checkpoint_step'] == 1688977
    raw = (root / 'pilot_artifacts/turning_on_radio_301_0.json').read_bytes()
    expected = pilot['checked_artifacts']['json']
    assert len(raw) == expected['size'] and hashlib.sha256(raw).hexdigest() == expected['sha256']
    result = json.loads(raw)
    assert (result['task'], result['instance_id'], result['rollout_id']) == ('turning_on_radio', 301, 0)
    assert result['steps'] == pilot['model_steps'] and result['success'] == pilot['model_success']
    assert result['q_score']['final'] == pilot['q_score']
    seen = {('turning_on_radio', 301, 0)}
    accepted_labels = set()
    accepted_count = 1
    for path in sorted((root / 'production_audits').glob('a*_accepted.json')):
        accepted = json.loads(path.read_text())
        label = 'a%03d' % accepted['index']
        assert path.stem == label + '_accepted' and label not in accepted_labels
        assert accepted['state'] == 'accepted' and accepted['platform_state'] == 'SUCCEEDED'
        submission = json.loads((root / 'production_submissions' / (label + '.json')).read_text())
        assert accepted['production_uid'] == submission['uid']
        assert accepted['source_manifest_sha256'] == submission['source_manifest_sha256'] == pilot['source_manifest_sha256']
        assert accepted['driver_sha256'] == submission['driver_sha256'] == pilot['driver_sha256']
        report_bytes = (root / 'production_audits' / (label + '_report.json')).read_bytes()
        assert hashlib.sha256(report_bytes).hexdigest() == accepted['report_sha256']
        originals = validate_report(json.loads(report_bytes), submission)
        assert len(originals) == accepted['rollout_count']
        for name, data in originals.items():
            assert (root / 'production_artifacts' / label / name).read_bytes() == data
            item = json.loads(data)
            key = (item['task'], item['instance_id'], item['rollout_id'])
            assert key in allowed and key not in seen
            seen.add(key)
        accepted_count += len(originals)
        accepted_labels.add(label)
    partial_rows = []
    for path in sorted(root.glob('production_partial_results_a*_current.json')):
        label = path.name.split('_')[3]
        if label in accepted_labels:
            continue
        partial = json.loads(path.read_text())
        submission = json.loads((root / 'production_submissions' / (label + '.json')).read_text())
        provenance = submission
        recovery_proofs = []
        while partial['job_uid'] != provenance['uid']:
            assert provenance['generation'] > 0
            recovery = json.loads((root / provenance['recovery_receipt']).read_text())
            assert recovery['uid'] == provenance['uid'] and recovery['state'] == 'submitted'
            assert recovery['old_writer_terminal'] is True
            previous = recovery['old_submission']
            assert previous['generation'] + 1 == provenance['generation']
            for field in ('assignment_root', 'task', 'instance_ids', 'run_id',
                          'source_manifest_sha256', 'driver_sha256'):
                assert previous[field] == provenance[field]
            proof_bytes = (root / recovery['queue_reconciliation_proof']).read_bytes()
            if 'queue_reconciliation_proof_sha256' in recovery:
                assert hashlib.sha256(proof_bytes).hexdigest() == recovery['queue_reconciliation_proof_sha256']
            proof = json.loads(proof_bytes)
            assert proof['old_uid'] == previous['uid'] and proof['old_generation'] == previous['generation']
            assert proof['completed_artifacts_preserved'] is True and proof['old_writer_terminal'] is True
            recovery_proofs.append(proof)
            provenance = previous
        assert partial['source_manifest_sha256'] == submission['source_manifest_sha256'] == pilot['source_manifest_sha256']
        for row in partial['results']:
            name = '%s_%s_0.json' % (submission['task'], row['instance_id'])
            data = (root / 'production_artifacts' / label / name).read_bytes()
            assert len(data) == row['bytes'] and hashlib.sha256(data).hexdigest() == row['sha256']
            for proof in recovery_proofs:
                remote_name = 'results/pi05/' + submission['task'] + '/json/' + name
                assert proof['artifacts_after'][remote_name] == dict(bytes=row['bytes'], sha256=row['sha256'])
            item = json.loads(data)
            key = (item['task'], item['instance_id'], item['rollout_id'])
            assert key in allowed and key not in seen and item['instance_id'] == row['instance_id']
            assert item['steps'] == row['steps'] > 0 and item['success'] == row['success']
            assert item['q_score']['final'] == row['q_score']
            assert isinstance(item['success'], bool) and type(row['q_score']) in (int, float)
            assert math.isfinite(row['q_score']) and 0 <= row['q_score'] <= 1
            assert item['conditioning']['mode'] == 'whole_task'
            assert item['conditioning']['oracle'] is False and item['conditioning']['privileged'] is False
            events = item['subtask_events']
            assert any(event['source'] == 'generated' for event in events)
            assert all(event['source'] in ('generated', 'cached') for event in events)
            assert math.isfinite(item['time']['normalized_time']) and item['time']['normalized_time'] > 0
            assert all(math.isfinite(item['normalized_agent_distance'][side]) for side in ('base', 'left', 'right'))
            seen.add(key)
            partial_rows.append(dict(label=label, task=item['task'], instance_id=item['instance_id'],
                steps=item['steps'], success=item['success'], q_score=row['q_score'], sha256=row['sha256']))
    assert len(seen) == accepted_count + len(partial_rows)
    return dict(state='accepted_plus_byte_verified_partials', observed_at=datetime.now(timezone.utc).isoformat(),
        accepted_rollouts=accepted_count, partial_rollouts=len(partial_rows),
        actual_completed_rollouts=len(seen), partial_rows=partial_rows,
        pilot_and_accepted_production_original_bytes_rehashed=True, not_full_test_score=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    report = count_originals(args.evidence.resolve())
    (args.evidence / 'actual_original_results_count_current.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key != 'partial_rows'}))
