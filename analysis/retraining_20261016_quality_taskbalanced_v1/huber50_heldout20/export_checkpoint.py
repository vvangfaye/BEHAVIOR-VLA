"""One bounded CPU export, after exact Huber 50 training completion."""
import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
TERMINAL_SUCCESS = {'FINISHED', 'SUCCEEDED', 'SUCCESS'}


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def training_ready(observed):
    native = observed['native']
    if native['iid'] != '241869' or native['application_status'].upper() not in TERMINAL_SUCCESS:
        return False
    workers = [row for row in native['containers'] if row['role'] == 'worker']
    if len(workers) != 2 or any(str(row['exitCode']) != '0' or row['status'].upper() not in TERMINAL_SUCCESS for row in workers):
        raise ValueError('Training worker success is unproven')
    done = observed['outputs'].get('_TRAINING_STAGE_SUCCESS.json', {})
    expected = {'run_id': 'b1k-expert-huber50-1688977-20261017', 'completed_updates': 1000,
                'gpus': 16, 'processes': 2, 'checkpoint_restore': True, 'all_state_leaves_exact': True}
    if any(done.get(key) != value for key, value in expected.items()):
        raise ValueError('Exact training completion and restore proof missing')
    if observed.get('bootstrap_failures') or any(key.endswith('failure.json') for key in observed['outputs']):
        raise ValueError('Training contains failure evidence')
    return True


def verify_export(plan, report, snapshot, bundle_hash):
    expected = {'state': 'accepted', 'id': 'export', 'completed_updates': 1000,
                'parent_checkpoint': 1688977, 'checkpoint': plan['candidate_checkpoint'],
                'protocol_sha256': plan['protocol_sha256'], 'worker_bundle_sha256': bundle_hash,
                'source_objects_etag_and_size_verified': True, 'full_afs_sha256_readback': True,
                'optimizer_exported': False, 'training_modified': False}
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError('Inference export identity or verification differs')
    if report['commit'] != snapshot['metadata'] or not report['commit'].get('commit_timestamp_nsecs'):
        raise ValueError('Committed checkpoint identity differs')
    files = report['files']
    source = {row['relative_path']: row for row in snapshot['files']}
    if len(files) != len(source) or {row['relative_path'] for row in files} != set(source):
        raise ValueError('Export file coverage differs')
    for row in files:
        name = row['relative_path']
        path = Path(name)
        if path.is_absolute() or '..' in path.parts or 'train_state' in path.parts:
            raise ValueError('Unsafe inference file path')
        original = source[name]
        if (row['bytes'] != original['bytes'] or row['source_etag'] != original['etag']
                or len(row['sha256']) != 64
                or (original.get('sha256') and row['sha256'] != original['sha256'])):
            raise ValueError('Source or destination file verification differs')
    by_name = {row['relative_path']: row for row in files}
    if by_name['assets/' + plan['repo_id'] + '/norm_stats.json']['sha256'] != plan['norm_sha256']:
        raise ValueError('Normalization differs')
    raw = ''.join(row['sha256'] + '  ' + row['relative_path'] + '\n' for row in files).encode()
    if hashlib.sha256(raw).hexdigest() != report['manifest_sha256'] or report['bytes'] != sum(row['bytes'] for row in files):
        raise ValueError('Export manifest or byte count differs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    with (ROOT / 'export.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observer = PARENT / 'expert_huber50_candidate/observe_training.py'
        result = subprocess.run([sys.executable, str(observer)], capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError('Training observation unavailable; no export submission')
        observed = load(PARENT / 'expert_huber50_candidate/current_training_observation.json')
        if not training_ready(observed):
            print(json.dumps({'state': 'waiting_for_completed_training', 'native': observed['native']['application_status'],
                              'export_submitted': False}))
            return
        completed_path = ROOT / 'completed_training_observation.json'
        if completed_path.exists():
            if not training_ready(load(completed_path)):
                raise ValueError('Frozen training completion proof is invalid')
        else:
            with completed_path.open('x') as stream:
                json.dump(observed, stream, indent=2)
        probe = module('h50_export_probe', ROOT / 'probe_checkpoint.py')
        with contextlib.redirect_stdout(io.StringIO()):
            probe.main()
        if load(ROOT / 'checkpoint_probe_status.json')['state'] != 'committed_checkpoint1000':
            raise ValueError('Completed training lacks committed checkpoint')
        snapshot = load(ROOT / 'private/checkpoint_snapshot.json')
        source_path = ROOT / 'private/export_source_snapshot.json'
        if source_path.exists():
            frozen = load(source_path)
            identities = lambda value: [(f['relative_path'], f['bytes'], f['etag'], f.get('sha256')) for f in value['files']]
            if (frozen['metadata'] != snapshot['metadata'] or frozen['formal_contract'] != snapshot['formal_contract']
                    or identities(frozen) != identities(snapshot)):
                raise ValueError('Committed export source changed')
            snapshot = frozen
        else:
            with source_path.open('x') as stream:
                json.dump(snapshot, stream)
            source_path.chmod(0o600)
        plan = load(ROOT / 'plan.json')
        evaluation = module('h50_export_transport', ROOT / 'controller.py')
        client = evaluation.base.transport.aoss_client()
        intent_path = ROOT / 'intents/export.json'
        if not intent_path.exists():
            if not args.submit:
                print(json.dumps({'state': 'ready_for_export', 'export_submitted': False}))
                return
            bundle_hash = digest(ROOT / 'frozen_sha256.json')
            report_url = evaluation.base.signed(client, plan['object_prefix'] + '/reports/export.json')
            payload = {'id': 'export', 'mode': 'export', 'job_name': 'b1k-h50-heldout20-1017-export',
                       'plan': plan, 'snapshot': snapshot, 'worker_bundle_sha256': bundle_hash}
            source = evaluation.base.transport.bootstrap(plan, payload, report_url)
            command = 'timeout --signal=TERM --kill-after=30 7200 ' + shlex.quote(plan['runtime'] + '/envs/b1k-eval/bin/python') + ' -c ' + shlex.quote(source)
            if len(command.encode()) >= 100000:
                raise ValueError('Export startup exceeds the bounded CLI argument size')
            private = ROOT / 'private/export.startup.py'
            private.write_text(source)
            private.chmod(0o600)
            intent = {'id': 'export', 'name': 'b1k-h50-heldout20-1017-export', 'uid': None,
                      'mode': 'export', 'gpu': 0, 'image': plan['image'], 'creation_attempted': False,
                      'startup_sha256': digest(private), 'worker_bundle_sha256': bundle_hash}
            evaluation.base.transport.new(intent_path, intent)
            evaluation.native_create(intent, command, cpu=True)
        intent = load(intent_path)
        state = evaluation.base.observe(intent)
        evaluation.base.save(ROOT / 'observations/export.json', state)
        key = plan['object_prefix'] + '/reports/export.json'
        try:
            response = client.get_object(Bucket='260010165', Key=key)
        except Exception as exc:
            if getattr(exc, 'response', {}).get('Error', {}).get('Code') not in ('NoSuchKey', '404'):
                raise
            if state['state'] in evaluation.base.TERMINAL:
                raise ValueError('Export terminal without a valid report; reconcile the existing job')
            print(json.dumps({'state': state['state'], 'uid': state['uid'], 'report_received': False}))
            return
        try:
            report = json.loads(response['Body'].read(4 * 1024**2))
        finally:
            response['Body'].close()
        verify_export(plan, report, snapshot, digest(ROOT / 'frozen_sha256.json'))
        evaluation.base.save(ROOT / 'reports/export.json', report)
        evaluation.base.save(ROOT / 'export_acceptance.json', {'accepted': True,
            'export_native_terminal': state['state'] == 'SUCCEEDED', 'native': state,
            'report_sha256': digest(ROOT / 'reports/export.json'),
            'training_observation_sha256': digest(completed_path),
            'source_snapshot_sha256': digest(source_path),
            'observed_at': time.time()})
        print(json.dumps({'export_accepted': True, 'native_state': state['state'], 'uid': state['uid']}))


if __name__ == '__main__':
    main()
