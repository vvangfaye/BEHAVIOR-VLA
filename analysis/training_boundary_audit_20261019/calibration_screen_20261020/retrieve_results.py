"""Retrieve the already submitted four-checkpoint diagnostic, without new jobs."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile

HERE = Path(__file__).resolve().parent
KEY = 'behavior1k/action-change-audit-20261020/b1k-action-change-20261020-screen01/results.tar.gz'
LIMIT = 64 * 1024**2
MODELS = ['baseline1688977', 'mixed1000', 'mixed5000', 'expert1000']


def main():
    sys.path.insert(0, str(HERE.parents[2] / 'BEHAVIOR-VLAs/sco'))
    from video_eval_adapter import aoss_client
    client = aoss_client()
    try:
        head = client.head_object(Bucket='260010165', Key=KEY)
        size = head['ContentLength']
        if not 0 < size <= LIMIT:
            raise ValueError('Unexpected result size')
        stream = client.get_object(Bucket='260010165', Key=KEY)['Body']
        try:
            body = stream.read(size+1)
        finally:
            stream.close()
    finally:
        client.close()
    if len(body) != size:
        raise ValueError('Incomplete transfer')
    allowed = {'progress.json', 'failure.json', 'device_preflight.log'}
    allowed.update(name+suffix for name in MODELS for suffix in ['.json', '.log', '.weights_verified.json'])
    files = {}
    with tarfile.open(fileobj=io.BytesIO(body), mode='r:gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (member.name not in allowed or len(path.parts) != 1 or not member.isfile()
                    or member.issparse() or member.name in files or not 0 <= member.size <= LIMIT):
                raise ValueError('Unexpected archive member')
            with archive.extractfile(member) as stream:
                data = stream.read(member.size+1)
            if len(data) != member.size:
                raise ValueError('Incomplete archive member')
            files[member.name] = data
        if sum(map(len, files.values())) > LIMIT:
            raise ValueError('Expanded result exceeds bound')
    archive_path = HERE / 'results.tar.gz'
    if archive_path.exists() and archive_path.read_bytes() != body:
        raise ValueError('Published result changed')
    if not archive_path.exists():
        archive_path.write_bytes(body)
    destination = HERE / 'retrieved'
    destination.mkdir(exist_ok=True)
    for name, data in files.items():
        path = destination/name
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Existing local result differs')
        if not path.exists():
            path.write_bytes(data)
    progress = json.loads(files['progress.json'])
    receipt = {'key': KEY, 'bytes': size, 'sha256': hashlib.sha256(body).hexdigest(),
               'worker_complete': progress.get('complete', False),
               'failure_file_present': 'failure.json' in files,
               'new_training': False, 'new_inference': False}
    if receipt['worker_complete']:
        if len(progress['results']) != 4:
            raise ValueError('Unexpected checkpoint result count')
        verified = []
        for item in progress['results']:
            name = item['name']
            if name not in MODELS or item['returncode'] != 0:
                raise ValueError('Invalid checkpoint completion')
            if hashlib.sha256(files[name+'.json']).hexdigest() != item['result_sha256']:
                raise ValueError('Result hash mismatch')
            result = json.loads(files[name+'.json'])
            if result['samples'] != 36 or len(result['sample_keys']) != 36:
                raise ValueError('Unexpected sample count')
            if not all('raw_actions' in sample for sample in result['sample_keys']):
                raise ValueError('Missing saved predictions')
            verified.append(name)
        if sorted(verified) != sorted(MODELS):
            raise ValueError('Missing or repeated checkpoint')
        receipt['verified_checkpoints'] = verified
        receipt['worker_elapsed_seconds'] = progress.get('elapsed_seconds')
    (HERE/'retrieval_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
