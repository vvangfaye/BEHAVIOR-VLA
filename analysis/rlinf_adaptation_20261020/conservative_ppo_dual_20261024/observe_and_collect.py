"""Finite observer for this exact pair; never creates or retries an experiment."""
import fcntl
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from status import status

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / 'BEHAVIOR-VLAs/sco'))
from video_eval_adapter import aoss_client

lock = (HERE / 'observer.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
terminal = {'FAILED', 'TERMINATED', 'CANCELED', 'STOPPED', 'SUSPENDED', 'SUCCEEDED'}
previous = None
for iteration in range(1440):
    try:
        result = status()
        compact = [{'rank': r['rank'], 'state': r['state'], 'progress': r.get('progress')} for r in result['workers']]
        encoded = json.dumps(compact, ensure_ascii=False, sort_keys=True)
        if encoded != previous:
            print(encoded, flush=True)
            previous = encoded
        failed = [r for r in result['workers'] if r['state'] in terminal - {'SUCCEEDED'}]
        if failed:
            # A failed partner cannot complete the synchronous batch. Stop only this pair.
            for row in result['workers']:
                if row['state'] not in terminal:
                    reply = subprocess.run([str(Path.home() / '.local/bin/sco'), 'acp', 'jobs', 'stop', row['name'],
                                            '--workspace-name=share-space'], capture_output=True, text=True, timeout=45)
                    print(json.dumps({'stopped_dependent_peer': row['name'], 'uid': row['uid'],
                                      'returncode': reply.returncode, 'reason': 'other worker terminated'}), flush=True)
        if all(r['state'] in terminal for r in result['workers']):
            client = aoss_client()
            try:
                key = 'behavior1k/rlinf-conservative-20261024/dual01/result_rank0.tar.gz'
                head = client.head_object(Bucket='260010165', Key=key)
                assert head['ContentLength'] < 256 * 1024**2
                raw = client.get_object(Bucket='260010165', Key=key)['Body'].read()
                assert len(raw) == head['ContentLength']
                (HERE / 'result.tar.gz').write_bytes(raw)
                with tarfile.open(fileobj=io.BytesIO(raw), mode='r:gz') as archive:
                    for item in archive:
                        relative = Path(item.name)
                        assert not relative.is_absolute() and '..' not in relative.parts and item.isfile()
                        if relative.suffix != '.json':continue
                        target = HERE / 'retrieved' / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(archive.extractfile(item).read())
                path = HERE / 'retrieved/comparison.json'
                if path.is_file():print(json.dumps({'comparison': json.loads(path.read_text())}), flush=True)
                (HERE / 'collection_receipt.json').write_text(json.dumps({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(),
                    'comparison_present': path.is_file(), 'workers': compact}, ensure_ascii=False, indent=2) + '\n')
            except client.exceptions.ClientError as error:
                print(json.dumps({'terminal_artifact_unavailable': type(error).__name__}), flush=True)
            finally:client.close()
            break
    except Exception as error:
        print(json.dumps({'observation_error': type(error).__name__, 'message': str(error)[:600]}), flush=True)
    time.sleep(30)
