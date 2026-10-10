"""Submit a CPU-only shard audit and collect verified original results after success."""
import argparse
import base64
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import zlib

from submit_public_test_shard import build_payload


def validate_report(report, submission):
    assert report['state'] == 'verified' and report['checkpoint_step'] == 1688977
    assert report['gpu_count'] == 0
    assert report['full_artifact_sha256_verified'] is True
    assert report['whole_task_native_generated_subtasks'] is True
    assert report['canonical_results_kept_regardless_of_score'] is True
    for key in ('source_manifest_sha256', 'driver_sha256', 'task', 'instance_ids', 'run_id'):
        assert report[key] == submission[key], 'Audit must match the exact submitted shard'
    assert report['job_uid'] == submission['uid']
    expected = {(submission['task'], i, 0) for i in submission['instance_ids']}
    keys = [(r['task'], r['instance_id'], r['rollout_id']) for r in report['results']]
    assert len(keys) == len(expected) == report['rollout_count'] and set(keys) == expected
    originals = {}
    for row, key in zip(report['results'], keys):
        raw = base64.b64decode(row['result_json_base64'], validate=True)
        assert len(raw) == row['json']['size'] and hashlib.sha256(raw).hexdigest() == row['json']['sha256']
        result = json.loads(raw)
        assert (result['task'], result['instance_id'], result['rollout_id']) == key
        assert result['steps'] == row['model_steps'] > 0
        assert result['success'] == row['model_success'] and isinstance(result['success'], bool)
        assert result['q_score']['final'] == row['q_score']
        originals[f'{key[0]}_{key[1]}_{key[2]}.json'] = raw
    return originals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--index', type=int, required=True)
    parser.add_argument('--action', choices=('submit', 'collect'), required=True)
    parser.add_argument('--transport', choices=('afs', 'aoss'), default='afs')
    parser.add_argument('--cli', default='/private/tmp/behavior-sco-runtime/bin/sco')
    args = parser.parse_args()
    root = args.evidence.resolve()
    label = 'a' + str(args.index).zfill(3)
    submission = json.loads((root / 'production_submissions' / (label + '.json')).read_text())
    ready = json.loads((root / 'preparation_ready.json').read_text())
    plan = json.loads((root / 'production_plan.json').read_text())
    pilot = json.loads((root / 'pilot_accepted.json').read_text())
    driver_sha = hashlib.sha256((root / 'pilot_driver_source.py').read_bytes()).hexdigest()
    expected = build_payload(ready, plan, pilot, args.index, submission['name'], driver_sha)
    generation = submission.get('generation', 0)
    assert type(generation) is int and generation >= 0
    if generation:
        recovery = json.loads((root / submission['recovery_receipt']).read_text())
        assert recovery['state'] == 'submitted' and recovery['uid'] == submission['uid']
        assert recovery['generation'] == generation
        assert recovery['old_writer_terminal'] is True
        assert recovery['old_submission']['generation'] + 1 == generation
        for key in ('assignment_root', 'run_id', 'source_manifest_sha256', 'driver_sha256',
                    'task', 'instance_ids'):
            assert recovery['old_submission'][key] == submission[key]
        expected['generation'] = generation
    for key, value in expected.items():
        assert submission[key] == value, 'Submission must match the frozen full-test plan'
    env = os.environ.copy()
    env.update(SCO_CONFIG='/private/tmp/behavior-sco-config', SCO_HOME='/private/tmp/behavior-sco-runtime',
               SCO_DATA_HOME='/private/tmp/behavior-sco-data')

    def describe(name, uid=None):
        job = json.loads(subprocess.run([args.cli, 'acp', 'jobs', 'describe', name,
            '--workspace-name=share-space', '-o', 'json'], env=env, capture_output=True,
            text=True, check=True, timeout=30).stdout)
        assert job['ownership']['user_name'] == '260010165'
        if uid is not None:
            assert job['uid'] == uid
        return job

    production = describe(submission['name'], submission['uid'])
    assert production['state'] == 'SUCCEEDED', 'Production must actually succeed before acceptance'
    audit_dir = root / 'production_audits'
    audit_dir.mkdir(exist_ok=True)
    receipt = audit_dir / (label + '_submission.json')
    from behavior_aoss_client import client
    storage = client()
    if args.action == 'submit':
        assert not receipt.exists(), 'Inspect existing audit; never duplicate or retry blindly'
        source = (root / 'finalization_sources' / 'audit_public_test_assignment.py').read_bytes()
        verified = json.loads((root / 'finalization_implementation_verified.json').read_text())
        source_sha = hashlib.sha256(source).hexdigest()
        assert source_sha == verified['source_sha256']['audit_public_test_assignment.py']
        compile(source, 'production_artifact_audit', 'exec')
        name = 'behavior-public1688977-audit-' + label + '-' + datetime.now(
            timezone(timedelta(hours=8))).strftime('%Y%m%d-%H%M%S')
        key = 'behavior1k/test2000-1688977-20261010/production/' + label + '_' + submission['uid'] + '_audit.json'
        payload = {k: submission[k] for k in ('assignment_root', 'source_manifest_sha256',
            'driver_sha256', 'uid', 'run_id', 'task', 'instance_ids')}
        report_path = str(Path(submission['assignment_root']).parent.parent / 'audits' /
                          label / (name + '.json'))
        if args.transport == 'aoss':
            payload['report_put'] = storage.generate_presigned_url('put_object',
                Params={'Bucket': '260010165', 'Key': key}, ExpiresIn=86400)
            tail = '\nrun(' + repr(payload) + ')\n'
        else:
            # Keep the pinned inspector intact; persist its report before leaving the CPU worker.
            tail = ('\nreport=inspect(' + repr(payload) + ')\n'
                    'report_raw=json.dumps(report).encode()\n'
                    'report_path=Path(' + repr(report_path) + ')\n'
                    'report_path.parent.mkdir(parents=True,exist_ok=True)\n'
                    'with report_path.open("xb") as stream: stream.write(report_raw)\n'
                    'print("STATUS_JSON="+json.dumps(dict(state="verified",'
                    'report_sha256=hashlib.sha256(report_raw).hexdigest())),flush=True)\n')
        wire = base64.b64encode(zlib.compress(source + tail.encode())).decode()
        startup = 'import base64,zlib;exec(compile(zlib.decompress(base64.b64decode(' + repr(wire) + ')),"production_artifact_audit","exec"))'
        record = dict(name=name, state='submitting', index=args.index, production_uid=submission['uid'],
            source_sha256=source_sha, report_key=key, gpu_count=0, read_only_afs=True)
        record.update(report_transport=args.transport, report_path=report_path,
                      read_only_afs=args.transport == 'aoss',
                      original_results_read_only=True)
        with receipt.open('x') as output:
            output.write(json.dumps(record, indent=2) + '\n')
        command = [args.cli, 'acp', 'jobs', 'create', '--workspace-name=share-space',
            '--aec2-name=share-cluster', '--name=' + name, '--job-name=' + name,
            '--training-framework=pytorch',
            '--container-image-url=registry.cn-sh-01g.sensecore.cn/lepton-trainingjob/ubuntu22.04-pytorch2.1.0-py310-cuda12.2:0630',
            '--worker-spec=n6ls.iu.i40.8c32g', '--worker-nodes=1', '--quota-type=spot',
            '--storage-mount=01a04263-91e5-7603-bc01-c67e503da6b5:/mnt/afs',
            '--command=timeout --signal=TERM --kill-after=15 600 python3 -c ' + shlex.quote(startup)]
        record['state'] = 'submission_unknown'
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        result = subprocess.run(command, env=env, capture_output=True, text=True, timeout=90)
        record.update(state='submitted' if result.returncode == 0 else 'submission_unknown', exit_code=result.returncode)
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        assert result.returncode == 0, 'Inspect exact saved audit name before retry'
        audit = describe(name)
        record.update(uid=audit['uid'], platform_state=audit['state'])
        receipt.write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps(record))
    else:
        record = json.loads(receipt.read_text())
        assert record['production_uid'] == submission['uid'] and record['gpu_count'] == 0
        verified = json.loads((root / 'finalization_implementation_verified.json').read_text())
        assert record['source_sha256'] == verified['source_sha256']['audit_public_test_assignment.py']
        audit = describe(record['name'], record['uid'])
        assert audit['state'] == 'SUCCEEDED', 'CPU audit must actually succeed before acceptance'
        assert audit['resource_pool']['name'] == 'share-cluster'
        for role in audit['roles']:
            specs = role.get('resource_spec', [])
            for spec in ([specs] if isinstance(specs, dict) else specs):
                assert int(spec.get('limits', {}).get('nvidia.com/gpu', 0)) == 0
        if record.get('report_transport', 'aoss') == 'aoss':
            raw = storage.get_object(Bucket='260010165', Key=record['report_key'])['Body'].read()
        else:
            expected_path = str(Path(submission['assignment_root']).parent.parent / 'audits' /
                                label / (record['name'] + '.json'))
            assert record['report_path'] == expected_path
            candidates = json.loads((root / 'production_all_platform_current.json').read_text())['jobs']
            observer = None
            for candidate in candidates:
                if candidate['state'] != 'RUNNING':
                    continue
                proxy = describe(candidate['name'], candidate['uid'])
                if proxy['state'] == 'RUNNING':
                    observer = candidate
                    break
            assert observer is not None, 'No confirmed live read proxy; retain completed audit and originals'
            transferred = audit_dir / (label + '_report_stream.json')
            reply = subprocess.run(['/private/tmp/behavior-transfer-env/bin/python',
                str(Path(__file__).with_name('read_afs_file_stream.py')), '--name', observer['name'],
                '--uid', observer['uid'], '--remote-path', expected_path, '--output', str(transferred),
                '--timeout', '3600'], capture_output=True, text=True, timeout=3650)
            assert reply.returncode == 0, 'Read transport failed; retain verified partial and same audit; do not restart model'
            transfer = json.loads(Path(str(transferred) + '.transfer.json').read_text())
            assert transfer['state'] in ('original_bytes_verified', 'verified_existing_original')
            raw = transferred.read_bytes()
            assert len(raw) == transfer['bytes'] and hashlib.sha256(raw).hexdigest() == transfer['sha256']
        report = json.loads(raw)
        originals = validate_report(report, submission)
        output = root / 'production_artifacts' / label
        output.mkdir(parents=True, exist_ok=True)
        for name, data in originals.items():
            path = output / name
            if path.exists():
                assert path.read_bytes() == data, 'Preserve previously collected canonical results'
            else:
                with path.open('xb') as stream:
                    stream.write(data)
        (audit_dir / (label + '_report.json')).write_bytes(raw)
        accepted = dict(state='accepted', checkpoint_step=1688977, index=args.index,
            production_uid=submission['uid'], audit_uid=record['uid'], platform_state='SUCCEEDED',
            source_manifest_sha256=submission['source_manifest_sha256'], driver_sha256=driver_sha,
            assignment_root=submission['assignment_root'], run_id=submission['run_id'],
            task=submission['task'], instance_ids=submission['instance_ids'], rollout_count=len(originals),
            full_artifact_sha256_verified=True, whole_task_native_generated_subtasks=True,
            canonical_results_kept_regardless_of_score=True, report_sha256=hashlib.sha256(raw).hexdigest())
        path = audit_dir / (label + '_accepted.json')
        if path.exists():
            assert json.loads(path.read_text()) == accepted
        else:
            with path.open('x') as stream:
                stream.write(json.dumps(accepted, indent=2) + '\n')
        print(json.dumps(accepted))


if __name__ == '__main__':
    main()
