"""Bounded, read-only collection of existing terminal labels; no simulator jobs."""
from pathlib import Path
import collections
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PREFIX = 'behavior1k/full-episode-collection/native-full1000-1688977-20261014-r3/ddea3ef13b17/episodes/'
SOURCE = ROOT / 'analysis/retraining_20261015_round3_v2/inventory/all_retained_episode_index_current.json'
LIMIT = 20 * 1024**2


def sha(b):
    return hashlib.sha256(b).hexdigest()


def main():
    source_bytes = SOURCE.read_bytes()
    groups = collections.defaultdict(list)
    for row in json.loads(source_bytes)['episode_rows']:
        groups[row['task']].append(row)
    # Outcome-independent selection, fixed before any new receipt is opened.
    tasks = sorted((t for t, rows in groups.items() if len(rows) >= 10),
                   key=lambda t: sha(t.encode()))[:10]
    selected = [r for t in tasks for r in sorted(groups[t], key=lambda r: r['instance_id'])[:10]]
    plan = {'source': str(SOURCE.relative_to(ROOT)), 'source_sha256': sha(source_bytes),
            'bucket': '260010165', 'prefix': PREFIX, 'max_total_bytes': LIMIT,
            'max_object_bytes': 2 * 1024**2, 'max_objects': 100,
            'selection': '10 tasks sorted by task-name SHA256 among tasks with >=10 episodes; lowest 10 instance IDs; no outcome selection',
            'tasks': tasks, 'rows': selected, 'new_jobs': 0, 'uploads': 0}
    plan_path = OUT / 'read_plan.json'
    if plan_path.exists():
        assert json.loads(plan_path.read_text()) == plan, 'Existing plan differs'
    else:
        plan_path.write_text(json.dumps(plan, indent=2) + '\n')
    sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    client = aoss_client()
    total = 0
    records = []
    skipped = []
    directory = OUT / 'receipts'
    directory.mkdir(exist_ok=True)
    for row in selected:
        key = row['receipt_key']
        assert key.startswith(PREFIX) and key.rsplit('/', 1)[-1] in ('attempt-1.json', 'attempt-1-g1.json'), 'key_scope'
        target = directory / (row['task'] + '__' + str(row['instance_id']) + '.json')
        if target.exists():
            payload = target.read_bytes()
            origin = 'local'
        else:
            head = client.head_object(Bucket=plan['bucket'], Key=key)
            size = head['ContentLength']
            if size > plan['max_object_bytes'] or total + size > LIMIT:
                skipped.append({'key': key, 'bytes': size, 'reason': 'byte_limit'})
                continue
            response = client.get_object(Bucket=plan['bucket'], Key=key, IfMatch=head['ETag'])
            body = response['Body']
            try:
                payload = body.read(size + 1)
            finally:
                body.close()
            assert len(payload) == size, 'incomplete_read'
            origin = 'aoss_existing_object'
        assert len(payload) <= plan['max_object_bytes'], 'payload_size'
        total += len(payload)
        assert total <= LIMIT, 'total_limit'
        receipt = json.loads(payload)
        assert sha(json.dumps(receipt, sort_keys=True).encode()) == row['receipt_sha256'], 'canonical_hash_mismatch'
        assert receipt['task'] == row['task'] and receipt['instance_id'] == row['instance_id'], 'task_instance'
        assert receipt['success'] == row['success'] and receipt['trajectory_sha256'] == row['raw_sha256'], 'outcome_trajectory'
        if not target.exists():
            target.write_bytes(payload)
        records.append({'file': str(target.relative_to(OUT)), 'sha256': sha(payload),
                        'bytes': len(payload), 'origin': origin})
    result = {'read_plan_sha256': sha(plan_path.read_bytes()), 'files': records, 'skipped': skipped,
              'total_bytes': total, 'uploads': 0, 'cloud_jobs': 0}
    (OUT / 'read_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'verified': len(records), 'skipped': skipped, 'bytes': total, 'tasks': tasks}))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # Avoid exposing credentials/headers in errors from external SDKs.
        print(json.dumps({'error_type': type(exc).__name__, 'check': str(exc) if isinstance(exc, AssertionError) else 'external_sdk_failure'}), file=sys.stderr)
        sys.exit(1)
