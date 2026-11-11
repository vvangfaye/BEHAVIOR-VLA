"""Join verified per-file receipts into a durable, unambiguous restore index."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from control import aoss_client, save


def main():
    plan = json.loads((HERE / 'plan.json').read_text())
    client = aoss_client()
    bucket = plan['bucket']
    destination = 'behavior1k/storage-reclaim-20261101/water-unblock-index-a1/'

    def get(key):
        return client.get_object(Bucket=bucket, Key=key)['Body'].read()

    try:
        released = {}
        attempts = []
        for folder in [HERE] + sorted(HERE.glob('retry_a*')):
            attempt = json.loads((folder / 'plan.json').read_text())
            status = json.loads(get(attempt['prefix'] + 'status.json'))
            assert status['metadata']['complete']
            for item in status['released']:
                assert item['index'] not in released, 'Duplicate source release'
                released[item['index']] = (attempt['prefix'], item)
            attempts.append({'prefix': attempt['prefix'], 'metadata': status['metadata']})
        assert set(released) == {item['index'] for item in plan['files']}, 'Archive not complete'

        def verify(item):
            prefix, record = released[item['index']]
            base = prefix + f"receipts/{item['index']:04d}/"
            raw = get(base + 'manifest.json')
            manifest = json.loads(raw)
            complete = json.loads(get(base + 'complete.json'))
            assert complete['verified']
            assert complete['manifest_sha256'] == hashlib.sha256(raw).hexdigest()
            assert complete['source_sha256'] == manifest['sha256'] == record['sha256']
            assert manifest['source'] == item['path'] == record['path']
            assert manifest['identity']['size'] == item['size'] == record['size']
            assert manifest['key'] == record['key'] and manifest['full_origin_readback']
            result = {'index': item['index'], 'source': item['path'], 'key': manifest['key'],
                    'size': item['size'], 'sha256': manifest['sha256'],
                    'allocated_bytes_released': record['allocated_bytes'],
                    'manifest_key': base + 'manifest.json', 'complete_key': base + 'complete.json',
                    'metadata_key': manifest['metadata_key']}
            if manifest.get('storage_kind') == 'ordered_parts':
                result.update(storage_kind='ordered_parts',parts=manifest['parts'])
                offset = 0
                for number, part in enumerate(manifest['parts']):
                    assert part['number'] == number and part['offset'] == offset
                    assert part['size'] > 0 and len(part['sha256']) == 64
                    offset += part['size']
                assert offset == item['size']
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            files = list(pool.map(verify, plan['files']))
        index = {'bucket': bucket, 'created_at': time.time(), 'files': files,
                 'attempts': attempts, 'source_plan_sha256': hashlib.sha256((HERE/'plan.json').read_bytes()).hexdigest(),
                 'released_allocated_bytes': sum(x['allocated_bytes_released'] for x in files),
                 'restore': 'Restore into a new empty root; verify metadata archives, then every listed object size and SHA-256; restore symlinks from metadata manifests.'}
        keys = {x['Key'] for x in client.list_objects_v2(Bucket=bucket, Prefix=destination, MaxKeys=10).get('Contents', [])}
        assert keys <= {destination+'restore_index.json', destination+'COMPLETE.json'}
        if destination+'restore_index.json' in keys:
            raw = get(destination+'restore_index.json')
            previous = json.loads(raw)
            assert previous['files'] == index['files']
            assert previous['source_plan_sha256'] == index['source_plan_sha256']
            assert previous['attempts'] == index['attempts']
            index = previous
        else:
            raw = (json.dumps(index, ensure_ascii=False, indent=2) + '\n').encode()
            client.put_object(Bucket=bucket, Key=destination+'restore_index.json', Body=raw)
        assert get(destination+'restore_index.json') == raw
        marker = {'verified': True, 'index_key': destination+'restore_index.json',
                  'index_sha256': hashlib.sha256(raw).hexdigest(), 'files': len(files),
                  'released_allocated_bytes': index['released_allocated_bytes']}
        body = (json.dumps(marker, sort_keys=True) + '\n').encode()
        if destination+'COMPLETE.json' not in keys:
            client.put_object(Bucket=bucket, Key=destination+'COMPLETE.json', Body=body)
        assert get(destination+'COMPLETE.json') == body
        save(HERE/'restore_index.json', index)
        save(HERE/'archive_complete.json', marker)
        print(json.dumps(marker))
    finally:
        client.close()


if __name__ == '__main__':
    for key in list(os.environ):
        if key.lower() in {'http_proxy', 'https_proxy', 'all_proxy'}:
            os.environ.pop(key)
    main()
