"""Stage a completed immutable RGB cache; independent of GPU scheduling."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PREFIX = 'behavior1k/action-support-20261021/vla-pilot-v1/data/'
NAMES = ('data_plan.json', 'raw_targets.npz', 'rgb.npy', 'image_receipt.json')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--upload', action='store_true')
    args = parser.parse_args()
    data = HERE / 'vla_pilot'
    if not (data / 'image_receipt.json').exists():
        print(json.dumps({'ready': False, 'reason': 'image cache incomplete'}))
        return
    receipt = json.loads((data / 'image_receipt.json').read_text())
    assert receipt['complete'] and receipt['shape'] == [5448, 3, 224, 224, 3]
    alignment = json.loads((data / 'timestamp_alignment.json').read_text())
    assert alignment['passed'] and alignment['checked_anchors'] == 5448
    assert json.loads((data / 'full_data_verification.json').read_text())['passed']
    objects = {name: {'bytes': (data / name).stat().st_size, 'sha256': digest(data / name),
                      'key': PREFIX + name} for name in NAMES}
    assert objects['rgb.npy']['sha256'] == receipt['image_sha256']
    assert objects['data_plan.json']['sha256'] == receipt['data_plan_sha256']
    assert sum(x['bytes'] for x in objects.values()) <= 3 * 1024**3
    if not args.upload:
        print(json.dumps({'ready': True, 'objects': objects}))
        return
    with (data / 'data_upload.lock').open('a') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        intent_path = data / 'data_stage_intent.json'
        if intent_path.exists():
            intent = json.loads(intent_path.read_text())
            assert intent['objects'] == objects
        else:
            intent = {'created_at': datetime.now(timezone.utc).isoformat(), 'objects': objects,
                      'bucket': '260010165', 'max_bytes': 3 * 1024**3,
                      'max_seconds': 1800,
                      'authorization': 'research_autonomy_authorization_20261020.json'}
            intent_path.write_text(json.dumps(intent, indent=2) + '\n')
        sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
        from video_eval_adapter import aoss_client
        from botocore.exceptions import ClientError
        client = aoss_client()
        verified = {}
        try:
            for name, item in objects.items():
                try:
                    remote = client.head_object(Bucket='260010165', Key=item['key'])
                except ClientError as error:
                    if error.response.get('Error', {}).get('Code') not in ('NoSuchKey', 'NotFound', '404'):
                        raise
                    remote = None
                if remote is None:
                    print(json.dumps({'uploading': name, 'bytes': item['bytes']}), flush=True)
                    client.upload_file(str(data / name), '260010165', item['key'],
                                       ExtraArgs={'Metadata': {'sha256': item['sha256']}})
                    remote = client.head_object(Bucket='260010165', Key=item['key'])
                assert remote['ContentLength'] == item['bytes'] and remote['Metadata']['sha256'] == item['sha256']
                verified[name] = {**item, 'etag': remote['ETag']}
                (data / 'data_stage_progress.json').write_text(json.dumps(verified, indent=2) + '\n')
            result = {'complete': True, 'objects': verified, 'bucket': '260010165',
                      'source_image_sha256': receipt['image_sha256'],
                      'scope': 'Existing expert observations only; no new collection or GPU work'}
            (data / 'data_stage_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
            print(json.dumps({'complete': True, 'verified_objects': len(verified)}), flush=True)
        finally:
            client.close()


if __name__ == '__main__':
    main()
