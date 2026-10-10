"""Read a live shard's original JSON without granting full-shard acceptance."""
import argparse
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

from submit_public_test_shard import build_payload


def collect(root, index, read_proxy_index=None):
    label = 'a' + str(index).zfill(3)
    submission = json.loads((root / 'production_submissions' / (label + '.json')).read_text())
    driver_sha = hashlib.sha256((root / 'pilot_driver_source.py').read_bytes()).hexdigest()
    expected = build_payload(json.loads((root / 'preparation_ready.json').read_text()),
        json.loads((root / 'production_plan.json').read_text()),
        json.loads((root / 'pilot_accepted.json').read_text()), index, submission['name'], driver_sha)
    generation = submission['generation']
    assert type(generation) is int and generation >= 0
    if generation:
        recovery = json.loads((root / submission['recovery_receipt']).read_text())
        assert recovery['state'] == 'submitted' and recovery['uid'] == submission['uid']
        assert recovery['old_writer_terminal'] is True
        assert recovery['old_submission']['generation'] + 1 == generation
        for key in ('assignment_root', 'run_id', 'source_manifest_sha256', 'driver_sha256',
                    'task', 'instance_ids'):
            assert recovery['old_submission'][key] == submission[key]
    expected['generation'] = generation
    assert all(submission[key] == value for key, value in expected.items())
    reader = submission
    if read_proxy_index is not None:
        env = os.environ.copy()
        env.update(SCO_CONFIG='/private/tmp/behavior-sco-config',
                   SCO_HOME='/private/tmp/behavior-sco-runtime',
                   SCO_DATA_HOME='/private/tmp/behavior-sco-data')
        def describe(row):
            job = json.loads(subprocess.run([
                '/private/tmp/behavior-sco-runtime/bin/sco', 'acp', 'jobs', 'describe',
                row['name'], '--workspace-name=share-space', '-o', 'json'],
                env=env, capture_output=True, text=True, check=True, timeout=30).stdout)
            assert job['uid'] == row['uid'] and job['ownership']['user_name'] == '260010165'
            return job
        assert describe(submission)['state'] == 'SUCCEEDED', 'Proxy collection requires exact completed writer'
        reader = json.loads((root / 'production_submissions' /
                             ('a' + str(read_proxy_index).zfill(3) + '.json')).read_text())
        assert reader['uid'] != submission['uid'] and describe(reader)['state'] == 'RUNNING'
    template = root / 'operational_helpers/behavior-public1688977-a012-worker-health.py'
    code = template.read_text()
    assert hashlib.sha256(template.read_bytes()).hexdigest() == '81ed25789f0da21f4f87941b10d255ee6ea983f25795dc7838593f4dae4a9165'
    code = code.replace('assignments/a012', 'assignments/' + label)
    code = code.replace('cleaning_up_plates_and_food', submission['task'])
    code = code.replace('generation-0', 'generation-' + str(generation))
    if read_proxy_index is not None:
        # Keep assignment-specific AFS observations, omit the proxy worker's hardware/processes.
        start = code.index("r['gpu']=")
        end = code.index("for name in ['status.json'", start)
        code = code[:start] + code[end:]
        start = code.index("r['processes']=[]")
        end = code.index("p=root/'queue.sqlite3'", start)
        code = code[:start] + code[end:]
    marker = "for path in sorted((root/'results/pi05/" + submission['task'] + "/json').glob('*.json')):\n"
    assert code.count(marker) == 1
    code = code.replace(marker, marker +
        " if path.stat().st_size > 256*1024:\n"
        "  r.setdefault('large_canonical_files',[]).append(dict(path=str(path.relative_to(root)),bytes=path.stat().st_size))\n"
        "  continue\n")
    compile(code, 'partial_read_only_observer', 'exec')
    source = root / 'operational_helpers' / ('partial_' + label + '_g' + str(generation) + '.py')
    source.write_text(code)
    health_path = root / ('production_' + label +
        ('_afs_originals_current.json' if read_proxy_index is not None else '_worker_health_current.json'))
    reply = subprocess.run(['/private/tmp/behavior-transfer-env/bin/python',
        '/private/tmp/behavior-read-worker-json.py', '--name', reader['name'],
        '--uid', reader['uid'], '--source', str(source), '--output', str(health_path),
        '--timeout', '45'], capture_output=True, text=True, timeout=65)
    assert reply.returncode == 0, 'Observer did not finish; do not restart or accept stale output'
    health = json.loads(health_path.read_text())
    assert health['assignment_root'] == submission['assignment_root']
    assert health['status.json']['source_manifest_sha256'] == submission['source_manifest_sha256']
    assert health['status.json']['checkpoint_step'] == 1688977
    output = root / 'production_artifacts' / label
    output.mkdir(exist_ok=True, parents=True)
    for large in health.get('large_canonical_files', []):
        relative = Path(large['path'])
        assert relative.parent.as_posix() == 'results/pi05/' + submission['task'] + '/json'
        path = output / relative.name
        proof_path = root / 'production_transfer_records' / (label + '_' + relative.name + '.transfer.json')
        transfer = subprocess.run(['/private/tmp/behavior-transfer-env/bin/python',
            str(Path(__file__).with_name('read_afs_file_stream.py')), '--name', reader['name'],
            '--uid', reader['uid'], '--remote-path', str(Path(submission['assignment_root']) / relative),
            '--output', str(path), '--state-path', str(proof_path), '--timeout', '3600'],
            capture_output=True, text=True, timeout=3650)
        assert transfer.returncode == 0, 'Retain verified transfer partial; never restart model after observer failure'
        proof = json.loads(proof_path.read_text())
        assert proof['state'] in ('original_bytes_verified', 'verified_existing_original')
        raw = path.read_bytes()
        assert len(raw) == proof['bytes'] == large['bytes'] and hashlib.sha256(raw).hexdigest() == proof['sha256']
        value = json.loads(raw)
        health['canonical_results'].append(dict(local_stream_path=str(path), bytes=len(raw),
            sha256=proof['sha256'], task=value['task'], instance_id=value['instance_id'],
            rollout_id=value['rollout_id'], steps=value['steps'], success=value['success'],
            q_score=value['q_score']['final']))
        del value, raw
    rows = []
    seen = set()
    for row in health['canonical_results']:
        raw = (Path(row['local_stream_path']).read_bytes() if 'local_stream_path' in row
               else base64.b64decode(row['result_json_base64'], validate=True))
        assert len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256']
        result = json.loads(raw)
        key = (result['task'], result['instance_id'], result['rollout_id'])
        assert key not in seen
        seen.add(key)
        assert key[0] == submission['task'] and key[1] in submission['instance_ids'] and key[2] == 0
        assert type(result['steps']) is int and result['steps'] == row['steps'] > 0
        assert isinstance(result['success'], bool) and result['success'] == row['success']
        score = result['q_score']['final']
        assert type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1
        assert score == row['q_score']
        conditioning = result['conditioning']
        assert conditioning['mode'] == 'whole_task'
        assert conditioning['oracle'] is False and conditioning['privileged'] is False
        events = result['subtask_events']
        assert any(event['source'] == 'generated' for event in events)
        assert all(event['source'] in ('generated', 'cached') for event in events)
        path = output / ('%s_%s_%s.json' % key)
        if path.exists():
            assert path.read_bytes() == raw, 'Never overwrite a previously collected original'
        else:
            with path.open('xb') as stream:
                stream.write(raw)
        rows.append({key: row[key] for key in ('instance_id', 'bytes', 'sha256', 'steps', 'success', 'q_score')})
    report = dict(state='original_bytes_verified_full_shard_audit_pending',
        observed_at=health['observed_at'], job_uid=submission['uid'],
        read_proxy_uid=reader['uid'] if read_proxy_index is not None else None,
        source_manifest_sha256=submission['source_manifest_sha256'], results=rows)
    (root / ('production_partial_results_' + label + '_current.json')).write_text(json.dumps(report, indent=2) + '\n')
    return dict(index=index, originals_verified=len(rows), results=rows,
        formally_accepted=False, native_messages=health.get('native_generation_log_messages', 0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--read-proxy-index', type=int,
                        help='Read a confirmed SUCCEEDED assignment through an existing live owned worker.')
    args = parser.parse_args()
    print(json.dumps(collect(args.evidence.resolve(), args.index, args.read_proxy_index)))
