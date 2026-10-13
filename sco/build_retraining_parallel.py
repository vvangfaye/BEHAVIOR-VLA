"""Disjoint, finite CPU processes inside one reserved ACP data-preparation job."""
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

from build_retraining_cloud import SignedPreparation
from prepare_retraining_data import error_detail, save


def check_resume_assignment(path, inputs):
    if path.exists():
        previous = json.loads(path.read_text())
        fields = ('key', 'sha256', 'bytes', 'etag')
        identity = lambda data: [tuple(row[k] for k in fields) for row in data['source_archives']]
        if previous['fingerprint'] != inputs['fingerprint'] or identity(previous) != identity(inputs):
            raise ValueError('Resume assignment differs from preserved source bytes')


def record_failure(prep, error, **fields):
    detail = dict(error_detail(error), **fields)
    save(prep.public / 'failure.json', detail)
    try:
        prep.publish_status({'phase': 'needs_diagnostic', **detail})
    except Exception as publication_error:
        save(prep.public / 'failure_publication.json', error_detail(publication_error))


def balanced_parts(rows, count):
    if not 1 <= count <= 16 or len({r['key'] for r in rows}) != len(rows):
        raise ValueError('Invalid disjoint partition')
    parts, loads = [[] for _ in range(count)], [0.0] * count
    # Measured pilot: read/unpack ~8 MB/s, export/check ~23 frames/s.
    def cost(row):
        return row['bytes'] / 8e6 + (row['frames'] / 23 if row['kind'] == 'full_episode' else row['frames'] / 800)
    for row in sorted(rows, key=lambda r: (-cost(r), r['key'])):
        index = min(range(count), key=lambda i: (loads[i], i))
        parts[index].append(row)
        loads[index] += cost(row)
    return parts


def checked_receipts(rows, receipts, require_complete=True):
    expected = {r['key']: r for r in rows}
    if len(expected) != len(rows):
        raise ValueError('Duplicate frozen source')
    result = {}
    for receipt in receipts:
        key = receipt['source_key']
        if key not in expected or key in result:
            raise ValueError('Unexpected or duplicate source receipt')
        if not (receipt.get('accepted') and receipt['source_sha256'] == expected[key]['sha256']
                and receipt['target']['full_destination_readback']
                and receipt['index_target']['full_destination_readback']
                and receipt['acceptance']['real_lerobot_reader']
                and receipt['acceptance']['episodes'] == expected[key]['episodes']
                and receipt['acceptance']['frames'] == expected[key]['frames']):
            raise ValueError('Unverified source receipt')
        result[key] = receipt
    if require_complete and set(result) != set(expected):
        raise ValueError('Missing source receipts')
    return result


def child(cache, public):
    prep = SignedPreparation(cache, public)
    try:
        for row in prep.inputs['source_archives']:
            if (Path(public) / 'STOP').exists():
                raise RuntimeError('Child data preparation STOP')
            prep.current_row = row
            receipt = prep.process(row)
            receipt.update(execution_location='source_cloud_cpu', training_executed=False)
            identity = hashlib.sha256(row['key'].encode()).hexdigest()[:20]
            path = Path(public) / 'receipts' / (identity + '.json')
            save(path, receipt)
            prep.upload(path, prep.prefix + 'receipts/' + identity + '.json')
        prep.publish_status({'phase': 'assigned_sources_verified'})
    except BaseException as error:
        record_failure(prep, error,
                       source_id=prep.current_row['id'] if prep.current_row else None)
        raise


def run(cache, public):
    cache, public = Path(cache), Path(public)
    coordinator = SignedPreparation(cache, public)
    inputs = coordinator.inputs
    count = inputs['parallel_execution']['cpu_processes']
    assert 1 <= count <= 4
    assert inputs['authorization']['cpu_only'] and not inputs['authorization']['training_allowed']
    rows = inputs['source_archives']
    partitions = balanced_parts(rows, count)
    processes = []
    context = multiprocessing.get_context('spawn')
    try:
        for i, assigned in enumerate(partitions):
            if not assigned:
                continue
            child_cache = cache / 'processes' / str(i)
            child_public = child_cache / 'output'
            child_cache.mkdir(parents=True, exist_ok=True)
            # Only small private manifests are copied; code and dependencies are shared read-only.
            child_inputs = {**inputs, 'source_archives': assigned}
            check_resume_assignment(child_cache / 'private_sources.json', child_inputs)
            save(child_cache / 'private_sources.json', child_inputs)
            storage = dict(coordinator.storage, status_put=coordinator.storage['child_status_put'][i])
            save(child_cache / 'private_storage.json', storage)
            for name in ['private_sources.json', 'private_storage.json']:
                (child_cache / name).chmod(0o600)
            process = context.Process(target=child, args=(child_cache, child_public))
            process.start()
            processes.append((process, child_public))
        while True:
            receipts = [json.loads(path.read_text()) for _, directory in processes
                        for path in (directory / 'receipts').glob('*.json')]
            accepted = checked_receipts(rows, receipts, require_complete=False)
            coordinator.publish_status({'phase': 'cpu_multiprocess_build',
                'cpu_processes': count, 'assigned_sources': len(rows),
                'accepted_archives': len(accepted),
                'accepted_episodes': sum(r['acceptance']['episodes'] for r in accepted.values()),
                'gpu_computation_enabled': False})
            if (public / 'STOP').exists():
                raise RuntimeError('Parallel data preparation STOP')
            if any(p.exitcode not in (0, None) for p, _ in processes):
                raise RuntimeError('CPU child needs diagnosis; preserving accepted receipts')
            if all(not p.is_alive() for p, _ in processes):
                break
            for process, _ in processes:
                process.join(timeout=2)
        receipts = [json.loads(path.read_text()) for _, directory in processes
                    for path in (directory / 'receipts').glob('*.json')]
        accepted = checked_receipts(rows, receipts)
        manifest = {'kind': 'source_cloud_prepared_training_shard',
            'source_fingerprint': inputs['fingerprint'],
            'parallel_execution': inputs['parallel_execution'],
            'sources': list(accepted.values()), 'expected_sources': len(rows),
            'all_assigned_sources_accounted_for': True, 'data_ready': False,
            'training_allowed': False, 'training_executed': False}
        path = public / 'cloud_prepared_manifest.json'
        save(path, manifest)
        target = coordinator.upload(path, coordinator.prefix + 'cloud_prepared_manifest.json')
        coordinator.publish_status({'phase': 'all_cloud_exports_verified', 'manifest': target,
            'source_count': len(rows), 'episodes': sum(r['acceptance']['episodes'] for r in accepted.values()),
            'training_bytes': sum(r['target']['bytes'] for r in accepted.values()),
            'gpu_computation_enabled': False})
    except BaseException as error:
        for process, _ in processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
        record_failure(coordinator, error, gpu_computation_enabled=False)
        raise


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2])
